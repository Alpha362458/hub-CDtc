"""
离线评估：加载 checkpoint，把各类分析结果写成 eval_report/ 下的 JSON，
供展示页面（server.py + web/index.html）读取。

产出:
  report.json    汇总指标、逐 epoch 曲线、逐位置 loss、生成样例、拼音转换对照
  attention.json 一条样例句逐层的注意力热力图
用法:
    python evaluate.py                    # 默认加载 best_transformer.pt
    python evaluate.py --max_batches 20   # 减少验证批次数，快速评估
"""

import argparse
import glob
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_corpus, build_vocab, split_train_val, CharDataset, build_unigram_logprob
from model import build_lm, generate
from model_handwritten import from_official
from pinyin_ime import load_pinyin_map, PinyinIME

REPORT_DIR = "eval_report"


def load_model(model_path, device):
    """从 checkpoint 恢复模型与词表（dropout 置 0，推理确定性），
    返回 (model, char2idx, idx2char, cfg, ckpt原始dict)。"""
    ckpt     = torch.load(model_path, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]

    model = build_lm(
        vocab_size=len(char2idx),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        ff_dim=cfg["ff_dim"],
        dropout=0.0,
        impl=cfg.get("impl", "official"),   # 老 checkpoint 未记录 impl 时按官方版处理
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, char2idx, idx2char, cfg, ckpt


@torch.no_grad()
def eval_loss_by_position(model, loader, criterion, device, max_batches):
    """验证集上按序列位置统计平均 loss，返回 (逐位置 loss 列表, 整体 loss, PPL)。

    位置 0 只有 1 个字的上文（几乎是瞎猜），位置越靠后上文越长、预测应越准
    —— 这条曲线是否下降，直观展示"上下文信息量"对语言模型的价值。
    """
    total_loss, total_tokens = 0.0, 0
    pos_loss_sum, pos_count = None, None

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # reduction='none' 保留每个 token 的 loss，才能按位置聚合
        loss_mat = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss_mat = loss_mat.view(x.size(0), x.size(1))

        total_loss += loss_mat.sum().item()
        total_tokens += y.numel()
        if pos_loss_sum is None:
            pos_loss_sum = torch.zeros(x.size(1), device=device)
            pos_count = torch.zeros(x.size(1), device=device)
        pos_loss_sum += loss_mat.sum(dim=0)
        pos_count += torch.ones_like(pos_loss_sum)

    avg = total_loss / total_tokens
    pos = (pos_loss_sum / pos_count).tolist()
    return pos, avg, float(torch.exp(torch.tensor(avg)))


@torch.no_grad()
def collect_attention(model, text, char2idx, device, max_len=48):
    """对一条样例句前向一次，收集每层的注意力权重。

    返回 {"chars": [...], "layers": {层号: {"avg": T×T, "head0": T×T}}}，
    avg 为所有头的平均（该层整体在关注谁），head0 保留单头视角
    （个别头常有更鲜明的模式，如只盯相邻字）。
    """
    for m in model.modules():
        # MultiHeadAttention 用 store_attn 标志控制是否保存权重
        if hasattr(m, "store_attn"):
            m.store_attn = True

    chars = [c for c in text if c in char2idx][:max_len]
    x = torch.tensor([[char2idx[c] for c in chars]], dtype=torch.long, device=device)
    model(x)

    layers = {}
    for li, block in enumerate(model.blocks):
        attn = block.attn.last_attn[0]   # (nhead, T, T) → 去掉 batch 维
        layers[str(li)] = {
            "avg": attn.mean(dim=0).tolist(),
            "head0": attn[0].tolist(),
        }

    for m in model.modules():
        if hasattr(m, "store_attn"):
            m.store_attn = False
            m.last_attn = None
    return {"chars": chars, "layers": layers}


def pick_demo_line(val_text, char2idx, max_len=48):
    """从验证集挑一条长度适中、字都在词表内的句子做注意力可视化样例。"""
    for line in val_text.splitlines():
        line = line.strip()
        if 15 <= len(line) <= max_len and all(c in char2idx for c in line):
            return line
    # 兜底：验证集没找到就退回词表头几个字
    return "".join(list(char2idx)[:10])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_transformer.pt")
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json")
    parser.add_argument("--max_batches", type=int, default=30, help="逐位置 loss 统计用的验证批数")
    parser.add_argument("--topk",        type=int, default=5, help="拼音候选展示数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(REPORT_DIR, exist_ok=True)

    model, char2idx, idx2char, cfg, ckpt = load_model(args.model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {args.model_path}  参数量 {total_params:,}  词表 {len(char2idx)}")

    # 复现训练时的验证集（同 seed、同 val_ratio、同按行 shuffle）
    text = load_corpus(args.corpus)
    train_text, val_text = split_train_val(text, cfg["val_ratio"], cfg["seed"])
    val_ds = CharDataset(val_text, char2idx, cfg["seq_len"], cfg["stride"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], drop_last=True)

    criterion = nn.CrossEntropyLoss(reduction="none")

    # 1. 逐位置 loss 与整体 PPL
    pos_loss, avg_loss, ppl = eval_loss_by_position(
        model, val_loader, criterion, device, args.max_batches)
    print(f"验证 loss: {avg_loss:.4f}  PPL: {ppl:.2f}")

    # 2. 多温度生成样例（观察模型学到的语言风格与温度的作用）
    prompts = ["为什么", "如何才能", "我最近"]
    samples = []
    for temp in (0.5, 0.8, 1.0):
        for p in prompts:
            samples.append({
                "prompt": p,
                "temperature": temp,
                "text": generate(model, char2idx, idx2char, p, n_chars=40,
                                 temperature=temp, top_k=8, device=device),
            })
    print(f"生成样例: {len(samples)} 条")

    # 3. 注意力热力图 —— 官方 API 不暴露注意力矩阵，转成等价的手写镜像来取权重
    demo_line = pick_demo_line(val_text, char2idx)
    attn_model = model if cfg.get("impl") == "handwritten" else from_official(model)
    attention = collect_attention(attn_model, demo_line, char2idx, device)
    print(f"注意力样例: 「{demo_line}」  ({len(attn_model.blocks)} 层)")

    # 4. 拼音转换对照（含切分歧义案例）
    ime = PinyinIME(
        model=model,
        char2idx=char2idx,
        idx2char=idx2char,
        pinyin_map=load_pinyin_map(args.pinyin_map),
        uni_logprob=build_unigram_logprob(args.corpus, char2idx, device),
        device=device,
        beam_size=10,
    )
    pinyin_cases = [
        ime.convert_detail(p) for p in
        ["wo ai ni", "fangan", "xian", "zhongguo", "qingting", "yinyue"]
    ]

    # 训练曲线优先从 checkpoint 内取，找不到再读训练时落盘的 history.json
    history = ckpt.get("history", [])
    if not history and os.path.exists("history.json"):
        with open("history.json", encoding="utf-8") as f:
            history = json.load(f)

    best_val_ppl = min((h["val_ppl"] for h in history), default=None)
    report = {
        "config": {**{k: v for k, v in cfg.items()},
                   "total_params": total_params, "vocab_size": len(char2idx),
                   "corpus_chars": len(text)},
        "metrics": {"val_loss": avg_loss, "val_ppl": ppl,
                    "best_val_ppl_in_history": best_val_ppl},
        "history": history,
        "position_loss": pos_loss,
        "samples": samples,
        "pinyin_cases": [{**c, "results": c["results"][:args.topk]} for c in pinyin_cases],
        "attention_demo_line": demo_line,
    }
    with open(os.path.join(REPORT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    with open(os.path.join(REPORT_DIR, "attention.json"), "w", encoding="utf-8") as f:
        json.dump(attention, f, ensure_ascii=False)

    print(f"已写入 {REPORT_DIR}/report.json 和 {REPORT_DIR}/attention.json")
    print(f"展示: python server.py 后浏览器打开 http://localhost:8765")


if __name__ == "__main__":
    main()
