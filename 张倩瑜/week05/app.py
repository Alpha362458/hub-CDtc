"""
shallowSeek —— Week 05 基于预训练 BERT 改造的 Prefix-LM / Decoder-Only SFT 交互测试后端

支持:
  1. GET  /                  渲染 templates/index.html ("shallowSeek" 界面)
  2. GET  /api/info          返回已加载的 SFT 模型元信息（BERT 102M Prefix-LM 或 Scratch）与预设测试样例
  3. GET  /api/random_sample 从验证集/训练集中随机抽取一条 (instruction, output) 供对比测试
  4. POST /api/chat          非流式生成回答 + SFT Question Loss Mask 与 Prefix+Causal 注意力掩码透视
  5. POST /api/chat/stream   SSE 流式生成回答 + 实时候选概率分布与 Question Mask 详情

启动方式:
  python app.py
"""

import os
import json
import time
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from transformers import BertTokenizer

from train_sft import (
    PAD_TOKEN,
    UNK_TOKEN,
    USER_TOKEN,
    ASST_TOKEN,
    END_TOKEN,
    IGNORE_INDEX,
    DEFAULT_DATA_DIR,
    DEFAULT_BERT_DIR,
    BertPrefixDecoderLM,
    TransformerDecoderOnlyLM,
    resolve_data_paths,
    resolve_bert_dir,
    load_jsonl,
    clean_bert_subword,
    generate_reply,
    generate_reply_stream,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.environ.get("SHALLOWSEEK_CKPT", os.path.join(HERE, "best_sft_model.pt"))
TEMPLATE_PATH = os.path.join(HERE, "templates", "index.html")

# 全局模型与数据缓存
_STATE = {
    "loaded": False,
    "backbone": "bert",
    "model": None,
    "tokenizer": None,
    "token2idx": None,
    "idx2token": None,
    "device": None,
    "meta": {},
    "train_records": [],
    "val_records": [],
}


def ensure_loaded():
    if _STATE["loaded"] and _STATE["model"] is not None:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _STATE["device"] = device

    try:
        tr_path, va_path = resolve_data_paths(DEFAULT_DATA_DIR)
        _STATE["train_records"] = load_jsonl(tr_path)
        _STATE["val_records"]   = load_jsonl(va_path)
    except Exception:
        _STATE["train_records"] = []
        _STATE["val_records"]   = []

    if not os.path.isfile(DEFAULT_CKPT):
        raise FileNotFoundError(
            f"未找到模型权重文件 {DEFAULT_CKPT}，请先运行 `python train_sft.py` 完成训练。"
        )

    ckpt = torch.load(DEFAULT_CKPT, map_location=device, weights_only=False)
    backbone = ckpt.get("backbone", "scratch")
    token2idx = ckpt["token2idx"]
    idx2token = {int(k): v for k, v in ckpt["idx2token"].items()}
    args = ckpt.get("args", {})

    if backbone == "bert":
        bert_dir = ckpt.get("bert_dir") or resolve_bert_dir(DEFAULT_BERT_DIR)
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
        prefix_bi = ckpt.get("prefix_bidirectional", True)
        bert_cfg  = ckpt.get("bert_config")
        if bert_cfg is not None:
            model = BertPrefixDecoderLM(config_dict=bert_cfg, prefix_bidirectional=prefix_bi).to(device)
        else:
            model = BertPrefixDecoderLM(bert_dir=bert_dir, prefix_bidirectional=prefix_bi).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        num_layers = model.bert_mlm.config.num_hidden_layers
        nhead      = model.bert_mlm.config.num_attention_heads
        embed_dim  = model.bert_mlm.config.hidden_size
        hidden_dim = model.bert_mlm.config.intermediate_size
        arch_label = "BERT-base-chinese (Prefix-LM: 问题双向 + 回答下三角因果)"
    else:
        tokenizer = None
        prefix_bi = False
        num_layers = args.get("num_layers", 4)
        nhead      = args.get("nhead", 8)
        embed_dim  = args.get("embed_dim", 256)
        hidden_dim = args.get("hidden_dim", 512)
        model = TransformerDecoderOnlyLM(
            vocab_size=len(token2idx),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=0.0,
            max_len=max(512, args.get("max_len", 128) + 64),
            pad_id=token2idx[PAD_TOKEN],
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        arch_label = "Scratch Transformer Decoder-Only"

    total_params = sum(p.numel() for p in model.parameters())
    _STATE["backbone"]  = backbone
    _STATE["model"]     = model
    _STATE["tokenizer"] = tokenizer
    _STATE["token2idx"] = token2idx
    _STATE["idx2token"] = idx2token
    _STATE["loaded"]    = True
    _STATE["meta"] = {
        "model_name": arch_label,
        "backbone": backbone,
        "prefix_bidirectional": prefix_bi,
        "device": str(device),
        "vocab_size": len(token2idx),
        "total_params": total_params,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "nhead": nhead,
        "mask_question": args.get("mask_question", True),
        "best_epoch": ckpt.get("best_epoch", ckpt.get("epoch", "-")),
        "best_val_loss": round(float(ckpt.get("best_val_loss", 0.0)), 4),
        "best_val_ppl": round(float(ckpt.get("best_val_ppl", 0.0)), 2),
        "train_size": len(_STATE["train_records"]),
        "val_size": len(_STATE["val_records"]),
    }


def build_sft_inspection(question, answer, token2idx, tokenizer=None):
    """
    构建 SFT Question Mask (Loss Mask = -100) 与 Prefix-LM 注意力模式的可视化明细数据：
    展示输入 x 与标签 y 的逐位对齐，标注哪些位置属于双向问题段 (label=-100)，哪些位置属于下三角因果回答段。
    """
    if tokenizer is not None:
        q_ids = tokenizer.encode(question.strip(), add_special_tokens=False)
        a_ids = tokenizer.encode(answer, add_special_tokens=False) if answer else []
        q_tokens = [clean_bert_subword(t) for t in tokenizer.convert_ids_to_tokens(q_ids)]
        a_tokens = [clean_bert_subword(t) for t in tokenizer.convert_ids_to_tokens(a_ids)]
        start_tok = "[CLS]"
        bridge_tok = "[SEP]"
        end_tok = "[SEP]"
    else:
        q_tokens = list(question.strip())
        a_tokens = list(answer)
        start_tok = USER_TOKEN
        bridge_tok = ASST_TOKEN
        end_tok = END_TOKEN

    prompt_tokens = [start_tok] + q_tokens + [bridge_tok]
    answer_tokens = a_tokens + [end_tok]
    full_tokens   = prompt_tokens + answer_tokens

    x_tokens = full_tokens[:-1]
    y_tokens = full_tokens[1:]
    prompt_len = len(prompt_tokens)

    alignment = []
    for i, (xt, yt) in enumerate(zip(x_tokens, y_tokens)):
        is_masked_question = i < (prompt_len - 1)
        attn_mode = "双向互看 (Prefix)" if i < prompt_len else "下三角因果 (Causal)"
        alignment.append({
            "pos": i,
            "x_token": xt,
            "target_token": yt,
            "label": IGNORE_INDEX if is_masked_question else yt,
            "masked": is_masked_question,
            "attn_mode": attn_mode,
            "segment": "question" if i < (prompt_len - 1) else ("bridge" if i == (prompt_len - 1) else "answer"),
        })

    masked_count = max(0, prompt_len - 1)
    active_count = len(answer_tokens)
    total_steps  = masked_count + active_count

    return {
        "template_str": f"{start_tok} {question.strip()} {bridge_tok} {answer} {end_tok}",
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
        "prefix_len": prompt_len,
        "masked_count": masked_count,
        "active_count": active_count,
        "loss_density_pct": round(100.0 * active_count / max(1, total_steps), 1),
        "alignment": alignment,
    }


def find_reference_answer(question):
    q_norm = question.strip()
    for r in _STATE["val_records"]:
        if r["instruction"] == q_norm:
            return {"split": "验证集 (sft_val)", "output": r["output"]}
    for r in _STATE["train_records"]:
        if r["instruction"] == q_norm:
            return {"split": "训练集 (sft_train)", "output": r["output"]}
    return None


def _extract_params(data):
    question = (data.get("question") or "").strip()
    if not question and data.get("messages"):
        for msg in reversed(data["messages"]):
            if msg.get("role") == "user" and msg.get("content"):
                question = str(msg["content"]).strip()
                break

    greedy             = bool(data.get("greedy", False))
    temperature        = float(data.get("temperature", 0.3))
    top_k              = int(data.get("top_k", 10))
    top_p              = float(data.get("top_p", 0.9))
    repetition_penalty = float(data.get("repetition_penalty", 1.15))
    max_new_tokens     = min(200, max(1, int(data.get("max_new_tokens", 80))))
    step_delay_ms      = min(200, max(0, int(data.get("step_delay_ms", 12))))

    return {
        "question": question,
        "greedy": greedy,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "step_delay_ms": step_delay_ms,
    }


def handle_info():
    ensure_loaded()
    val_recs = _STATE["val_records"]
    train_recs = _STATE["train_records"]
    return {
        "meta": _STATE["meta"],
        "presets": {
            "val": val_recs[:8] if val_recs else [],
            "train": train_recs[:8] if train_recs else [],
        },
    }


def handle_random_sample(split="val"):
    ensure_loaded()
    pool = _STATE["val_records"] if split == "val" else _STATE["train_records"]
    if not pool:
        return {"error": "数据集为空"}, 404
    item = random.choice(pool)
    return {
        "split": "验证集 (sft_val)" if split == "val" else "训练集 (sft_train)",
        "instruction": item["instruction"],
        "output": item["output"],
    }, 200


def handle_chat(data):
    ensure_loaded()
    params = _extract_params(data)
    question = params["question"]
    if not question:
        return {"error": "请输入问题内容"}, 400

    t0 = time.perf_counter()
    reply, steps_info = generate_reply(
        model=_STATE["model"],
        question=question,
        token2idx=_STATE["token2idx"],
        idx2token=_STATE["idx2token"],
        device=_STATE["device"],
        tokenizer=_STATE["tokenizer"],
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        greedy=params["greedy"],
    )
    elapsed_s = round(time.perf_counter() - t0, 3)

    sft_inspect = build_sft_inspection(question, reply, _STATE["token2idx"], _STATE["tokenizer"])
    ref_info    = find_reference_answer(question)

    return {
        "content": reply,
        "elapsed_s": elapsed_s,
        "sft_inspect": sft_inspect,
        "reference": ref_info,
        "steps": steps_info,
    }, 200


def iter_chat_stream(data):
    ensure_loaded()
    params = _extract_params(data)
    question = params["question"]
    if not question:
        yield f"data: {json.dumps({'error': '请输入问题内容'}, ensure_ascii=False)}\n\n"
        return

    ref_info = find_reference_answer(question)
    pieces = []
    steps_info = []

    init_inspect = build_sft_inspection(question, "", _STATE["token2idx"], _STATE["tokenizer"])
    yield f"data: {json.dumps({'sft_init': init_inspect, 'reference': ref_info}, ensure_ascii=False)}\n\n"

    for item in generate_reply_stream(
        model=_STATE["model"],
        question=question,
        token2idx=_STATE["token2idx"],
        idx2token=_STATE["idx2token"],
        device=_STATE["device"],
        tokenizer=_STATE["tokenizer"],
        max_new_tokens=params["max_new_tokens"],
        temperature=params["temperature"],
        top_k=params["top_k"],
        top_p=params["top_p"],
        repetition_penalty=params["repetition_penalty"],
        greedy=params["greedy"],
    ):
        steps_info.append(item)
        if item["token"]:
            pieces.append(item["token"])
        payload = {
            "content": item["token"],
            "step": item,
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if params["step_delay_ms"] > 0:
            time.sleep(params["step_delay_ms"] / 1000.0)

    full_reply = "".join(pieces)
    final_inspect = build_sft_inspection(question, full_reply, _STATE["token2idx"], _STATE["tokenizer"])
    yield f"data: {json.dumps({'sft_inspect': final_inspect, 'reference': ref_info, 'steps': steps_info}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


class ShallowSeekHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[shallowSeek] {self.address_string()} - {fmt % args}")

    def _send_json(self, obj, status=200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                html = f.read().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if path == "/api/info":
            try:
                self._send_json(handle_info(), 200)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/random_sample":
            try:
                qs = parse_qs(parsed.query)
                split = (qs.get("split", ["val"])[0] or "val").lower()
                obj, code = handle_random_sample(split)
                self._send_json(obj, code)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"error": "Not Found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(raw_body) if raw_body else {}
        except Exception:
            data = {}

        if path == "/api/chat":
            try:
                obj, code = handle_chat(data)
                self._send_json(obj, code)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            return

        if path == "/api/chat/stream":
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                for chunk in iter_chat_stream(data):
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except Exception as e:
                try:
                    err = f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
                    self.wfile.write(err.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass
            return

        self._send_json({"error": "Not Found"}, 404)


def main():
    port = int(os.environ.get("PORT", 5000))
    ensure_loaded()
    meta = _STATE["meta"]
    print("=" * 72)
    print(f"shallowSeek SFT 服务已就绪: http://127.0.0.1:{port}")
    print(f"模型: {meta['model_name']} | 设备: {meta['device']} | 参数量: {meta['total_params']:,}")
    print(f"词表: {meta['vocab_size']:,} | Question Masking: {meta['mask_question']} | Best Val PPL: {meta['best_val_ppl']}")
    print("=" * 72, flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), ShallowSeekHTTPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
