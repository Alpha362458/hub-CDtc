"""
Week 05: 基于预训练 BERT 改造的 Prefix-LM / Decoder-Only 监督微调 (SFT)
同时保留从零训练的 Transformer Decoder-Only 模型支持。

核心设计 (Prefix-LM: 问题段双向互看 + 回答段下三角因果生成 + Question Loss Mask):
  1. 预训练底座: 加载 `bert-base-chinese` (12层, 12头, 768维, 21128词表, 含预训练 MLM 词表预测头)。
  2. Attention Mask 改造 (Prefix + Causal):
     - 问题段 (Prefix: `[CLS] + question_tokens + [SEP]`，共 P 个 token)：内部双向全可见 (`mask[:P, :P] = True`)，充分利用 BERT 双向语义理解能力；
     - 回答段 (Answer: `answer_tokens + [SEP]`)：下三角因果掩码 (`mask[i, j] = (j <= i)`)，第 i 步只能看见问题段和 <= i 的已生成回答；
  3. Question Loss Mask (`ignore_index = -100`):
     - 问题段所有预测位置 `0 .. P-2` 的标签全部设为 `-100`，不计算 Loss；
     - 从 `x[P-1] == [SEP]` 预测第一个回答 token 开始，到预测结束符 `[SEP]` 为止，计算 CrossEntropyLoss。

用法:
    # 默认使用 bert-base-chinese 进行 Prefix-LM SFT 训练
    python train_sft.py --backbone bert --epochs 10

    # 也可切换从零训练的 Transformer Decoder-Only
    python train_sft.py --backbone scratch --epochs 40 --save scratch_sft_model.pt
"""

import os
import json
import math
import zipfile
import random
import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


# ─────────────────────────── 常量与路径配置 ───────────────────────────

PAD_TOKEN  = "<pad>"
UNK_TOKEN  = "<unk>"
USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
END_TOKEN  = "<|end|>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, USER_TOKEN, ASST_TOKEN, END_TOKEN]
IGNORE_INDEX = -100

DEFAULT_DATA_DIR = r"C:\baidunetdiskdownload\sft_train"
DEFAULT_BERT_DIR = r"C:\baidunetdiskdownload\第四周 语言模型\bert-base-chinese"
DEFAULT_BERT_ZIP = r"C:\baidunetdiskdownload\第四周 语言模型\bert-base-chinese.zip"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", "masked", "mask"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0", "unmasked", "nomask"):
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {v}")


def resolve_data_paths(data_dir=DEFAULT_DATA_DIR):
    candidates = [
        data_dir,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_train"),
        os.path.dirname(os.path.abspath(__file__)),
        DEFAULT_DATA_DIR,
    ]
    for d in candidates:
        if not d:
            continue
        tr = os.path.join(d, "sft_train.jsonl")
        va = os.path.join(d, "sft_val.jsonl")
        if os.path.isfile(tr) and os.path.isfile(va):
            return tr, va
    raise FileNotFoundError(f"未找到 sft_train.jsonl / sft_val.jsonl，请检查路径: {data_dir}")


def resolve_bert_dir(bert_dir=DEFAULT_BERT_DIR):
    if os.path.isdir(bert_dir) and os.path.isfile(os.path.join(bert_dir, "vocab.txt")):
        return bert_dir
    if os.path.isfile(DEFAULT_BERT_ZIP):
        parent = os.path.dirname(DEFAULT_BERT_ZIP)
        with zipfile.ZipFile(DEFAULT_BERT_ZIP, "r") as z:
            for m in z.namelist():
                if "/.git/" not in m and not m.endswith("/.git"):
                    target = os.path.join(parent, m)
                    if not os.path.exists(target):
                        z.extract(m, parent)
        extracted = os.path.join(parent, "bert-base-chinese")
        if os.path.isdir(extracted):
            return extracted
    raise FileNotFoundError(f"未找到 bert-base-chinese 目录或压缩包: {bert_dir}")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                records.append({
                    "instruction": str(obj["instruction"]).strip(),
                    "output": str(obj["output"]).strip(),
                })
    return records


# ─────────────────────────── Prefix + Causal 掩码构造 ───────────────────────────

def build_prefix_causal_mask_2d(seq_len, prefix_len, valid_len=None, prefix_bidirectional=True, device=None):
    """
    构造单条样本的 L×L 注意力可见性布尔矩阵 (True = 可见, False = 屏蔽 -inf):
      - 基础为下三角因果矩阵: j <= i
      - 若 prefix_bidirectional=True: 前 prefix_len 个位置 (问题段 `[CLS] + Q + [SEP]`) 互相全可见:
        mask[:prefix_len, :prefix_len] = True
      - 若提供 valid_len: 超出有效长度的右侧 padding 列禁止被看见
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    if prefix_bidirectional and prefix_len > 1:
        p = min(prefix_len, seq_len)
        mask[:p, :p] = True
    if valid_len is not None and valid_len < seq_len:
        mask[:, valid_len:] = False
        mask[valid_len:, :] = False
        # 避免全 False 行导致 softmax NaN，让 pad 位置只看自己
        idx = torch.arange(valid_len, seq_len, device=device)
        mask[idx, idx] = True
    return mask


def build_batch_prefix_causal_mask_4d(seq_len, prefix_lens, valid_lens, prefix_bidirectional=True, device=None):
    """
    构造 Batch 级 4D 注意力掩码: shape = (B, 1, L, L), dtype = torch.bool
    可直接传入 HuggingFace BertForMaskedLM(attention_mask=...) 或 PyTorch SDPA。
    """
    bsz = len(prefix_lens)
    masks = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=device)
    base_tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))

    for b in range(bsz):
        m = base_tril.clone()
        p = min(int(prefix_lens[b]), seq_len)
        v = min(int(valid_lens[b]), seq_len)
        if prefix_bidirectional and p > 1:
            m[:p, :p] = True
        if v < seq_len:
            m[:, v:] = False
            m[v:, :] = False
            idx = torch.arange(v, seq_len, device=device)
            m[idx, idx] = True
        masks[b, 0] = m
    return masks


# ─────────────────────────── BERT Prefix-LM 模型与数据集 ───────────────────────────

class BertPrefixDecoderLM(nn.Module):
    """
    基于 `bert-base-chinese` 改造的 Prefix-LM / Decoder-Only 生成模型：
    - 复用 BERT 12层 Transformer 权重 + 预训练 MLM Prediction Head (21,128 词表)
    - token_type_ids 区分问题段 (segment 0) 与回答段 (segment 1)
    - 注意力掩码采用「问题段双向互看 + 回答段下三角因果生成」
    """

    def __init__(self, bert_dir=None, config_dict=None, prefix_bidirectional=True):
        super().__init__()
        self.prefix_bidirectional = prefix_bidirectional
        if bert_dir is not None:
            self.bert_mlm = BertForMaskedLM.from_pretrained(bert_dir, return_dict=True)
        elif config_dict is not None:
            cfg = BertConfig.from_dict(config_dict)
            cfg.return_dict = True
            self.bert_mlm = BertForMaskedLM(cfg)
        else:
            raise ValueError("必须提供 bert_dir 或 config_dict")

        self.max_len = self.bert_mlm.config.max_position_embeddings

    def forward(self, input_ids, prefix_lens, valid_lens=None):
        """
        input_ids: (B, L)
        prefix_lens: 长度为 B 的张量或列表，表示每个样本问题段 `[CLS] + Q + [SEP]` 的 token 数 P
        valid_lens: 长度为 B 的张量或列表，表示每个样本非 pad 的有效长度
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if valid_lens is None:
            valid_lens = [seq_len] * bsz

        # 构造 (B, 1, L, L) Prefix + Causal 注意力掩码
        attn_mask_4d = build_batch_prefix_causal_mask_4d(
            seq_len=seq_len,
            prefix_lens=prefix_lens,
            valid_lens=valid_lens,
            prefix_bidirectional=self.prefix_bidirectional,
            device=device,
        )

        # 构造 token_type_ids: 问题段 (0 .. P-1) 为 0，回答段 (P .. L-1) 为 1
        token_type_ids = torch.zeros_like(input_ids)
        for b in range(bsz):
            p = min(int(prefix_lens[b]), seq_len)
            if p < seq_len:
                token_type_ids[b, p:] = 1

        out = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attn_mask_4d,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return out.logits  # (B, L, vocab_size=21128)


def encode_bert_sft_sample(instruction, output, tokenizer, max_len=128, mask_question=True):
    """
    使用 BERT Tokenizer 编码单条 (instruction, output)：
      full_ids   = [CLS] + q_ids + [SEP] + a_ids + [SEP]
      prompt_ids = [CLS] + q_ids + [SEP]   (长度 P)
      answer_ids = a_ids + [SEP]           (长度 A)

    自回归错位 (shift-by-one):
      x = full_ids[:-1]
      y = full_ids[1:]

    Question Loss Mask:
      - x[0 .. P-2] 预测的是问题内部的 token，标签全部置为 -100 (屏蔽)
      - x[P-1] 是问题末尾的 `[SEP]` (充当 <|assistant|> 触发位)，它预测回答的第 1 个字 a_ids[0]，计入 Loss！
      - x[P .. P+A-2] 依次预测 a_ids[1:] 和结束符 `[SEP]`，全部计入 Loss！
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    q_ids = tokenizer.encode(instruction.strip(), add_special_tokens=False)
    a_ids = tokenizer.encode(output.strip(), add_special_tokens=False)

    # 预留至少 16 个 token 给回答
    max_q = max(16, max_len - 3 - min(len(a_ids), max_len // 2))
    if len(q_ids) > max_q:
        q_ids = q_ids[:max_q]

    prompt_ids = [cls_id] + q_ids + [sep_id]
    answer_ids = a_ids + [sep_id]
    full_ids   = prompt_ids + answer_ids

    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        if full_ids[-1] != sep_id:
            full_ids[-1] = sep_id

    x = full_ids[:-1]
    y = full_ids[1:].copy()
    prefix_len = len(prompt_ids)
    valid_len  = len(x)

    prompt_target_len = max(0, min(prefix_len - 1, len(y)))
    if mask_question:
        for i in range(prompt_target_len):
            y[i] = IGNORE_INDEX

    return x, y, prefix_len, valid_len


class BertSFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=128, mask_question=True):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seq_len = max_len - 1
        self.pad_id = tokenizer.pad_token_id

        self.samples = []
        for item in records:
            x, y, prefix_len, valid_len = encode_bert_sft_sample(
                item["instruction"],
                item["output"],
                tokenizer,
                max_len=max_len,
                mask_question=mask_question,
            )
            pad_len = self.seq_len - len(x)
            x_pad = x + [self.pad_id] * pad_len
            y_pad = y + [IGNORE_INDEX] * pad_len
            self.samples.append((
                torch.tensor(x_pad, dtype=torch.long),
                torch.tensor(y_pad, dtype=torch.long),
                int(prefix_len),
                int(valid_len),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────── 从零训练版 (Scratch) 模型与数据 ───────────────────────────

def build_vocab(records, max_vocab=None):
    counter = Counter()
    for item in records:
        counter.update(item["instruction"])
        counter.update(item["output"])

    limit = (max_vocab - len(SPECIAL_TOKENS)) if max_vocab else None
    chars = [c for c, _ in counter.most_common(limit)]
    vocab_list = list(SPECIAL_TOKENS) + [c for c in chars if c not in SPECIAL_TOKENS]

    token2idx = {t: i for i, t in enumerate(vocab_list)}
    idx2token = {i: t for i, t in enumerate(vocab_list)}
    return token2idx, idx2token


def encode_sft_sample(instruction, output, token2idx, max_len=128, mask_question=True):
    unk_id  = token2idx[UNK_TOKEN]
    user_id = token2idx[USER_TOKEN]
    asst_id = token2idx[ASST_TOKEN]
    end_id  = token2idx[END_TOKEN]

    q_ids = [token2idx.get(c, unk_id) for c in instruction]
    a_ids = [token2idx.get(c, unk_id) for c in output]

    prompt_ids = [user_id] + q_ids + [asst_id]
    answer_ids = a_ids + [end_id]
    full_ids   = prompt_ids + answer_ids

    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        if full_ids[-1] != end_id:
            full_ids[-1] = end_id

    x = full_ids[:-1]
    y = full_ids[1:].copy()

    prompt_target_len = max(0, min(len(prompt_ids) - 1, len(y)))
    if mask_question:
        for i in range(prompt_target_len):
            y[i] = IGNORE_INDEX

    return x, y, len(prompt_ids)


class SFTDataset(Dataset):
    def __init__(self, records, token2idx, max_len=128, mask_question=True):
        self.seq_len = max_len - 1
        self.pad_id = token2idx[PAD_TOKEN]
        self.samples = []
        for item in records:
            x, y, prompt_len = encode_sft_sample(
                item["instruction"], item["output"], token2idx, max_len=max_len, mask_question=mask_question
            )
            valid_len = len(x)
            pad_len = self.seq_len - valid_len
            x_pad = x + [self.pad_id] * pad_len
            y_pad = y + [IGNORE_INDEX] * pad_len
            self.samples.append((
                torch.tensor(x_pad, dtype=torch.long),
                torch.tensor(y_pad, dtype=torch.long),
                int(prompt_len),
                int(valid_len),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions) + self.pe[:, :seq_len, :] * 0.02
        return self.drop(x)


class TransformerDecoderOnlyLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        nhead=8,
        dropout=0.1,
        max_len=512,
        pad_id=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.max_len = max_len
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()

        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout, max_len=max_len)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc.weight = self.embed.weight

    def _get_causal_mask(self, seq_len, device):
        if (
            not hasattr(self, "_cached_mask")
            or self._cached_mask is None
            or self._cached_mask.size(0) != seq_len
            or self._cached_mask.device != device
        ):
            self._cached_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )
        return self._cached_mask

    def forward(self, x, prefix_lens=None, valid_lens=None):
        seq_len = x.size(1)
        h = self.embed(x)
        h = self.pos_encoder(h)
        causal_mask = self._get_causal_mask(seq_len, x.device)
        out = self.transformer(h, mask=causal_mask)
        return self.fc(out)


# ─────────────────────────── 解码生成 (含流式与采样策略) ───────────────────────────

def clean_bert_subword(tok_str):
    """将 BERT WordPiece 的 `##ing` 还原为 `ing`。"""
    if tok_str.startswith("##"):
        return tok_str[2:]
    return tok_str


@torch.no_grad()
def sample_next_token(
    logits,
    generated_ids=None,
    temperature=0.7,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.15,
    greedy=False,
    ban_ids=None,
):
    logits = logits.clone().float()

    if ban_ids:
        for bid in ban_ids:
            if 0 <= bid < logits.size(-1):
                logits[bid] = -float("inf")

    if generated_ids and len(generated_ids) >= 3:
        prefix3 = tuple(generated_ids[-3:])
        for i in range(len(generated_ids) - 3):
            if tuple(generated_ids[i : i + 3]) == prefix3:
                logits[generated_ids[i + 3]] -= 4.0
        if generated_ids[-1] == generated_ids[-2] == generated_ids[-3]:
            logits[generated_ids[-1]] -= 5.0

    scaled_for_view = logits / max(temperature, 1e-3) if not greedy else logits
    probs_view = F.softmax(scaled_for_view, dim=-1)
    topv, topi = torch.topk(probs_view, k=min(5, logits.size(-1)))
    top_candidates = [
        {"id": int(i.item()), "prob": round(float(p.item()), 4)}
        for p, i in zip(topv, topi)
    ]

    if greedy or temperature <= 1e-4:
        next_id = int(torch.argmax(logits).item())
        return next_id, top_candidates

    logits = logits / max(temperature, 1e-4)

    if top_k is not None and 0 < top_k < logits.size(-1):
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float("inf")

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    next_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_id, top_candidates


@torch.no_grad()
def generate_reply_stream(
    model,
    question,
    token2idx,
    idx2token,
    device,
    tokenizer=None,
    max_new_tokens=80,
    temperature=0.3,
    top_k=10,
    top_p=0.9,
    repetition_penalty=1.15,
    greedy=True,
):
    """
    统一流式生成器：
    - 若提供 `tokenizer` (BERT Prefix-LM 模式)：
      Prompt 为 `[CLS] + question_tokens + [SEP]`，问题段内部双向互看，回答段下三角因果生成，遇到 `[SEP]` 停止。
    - 若未提供 `tokenizer` (Scratch Decoder-Only 模式)：
      Prompt 为 `<|user|> + question + <|assistant|>`，遇到 `<|end|>` 停止。
    """
    model.eval()
    is_bert = tokenizer is not None

    if is_bert:
        cls_id  = tokenizer.cls_token_id
        sep_id  = tokenizer.sep_token_id
        pad_id  = tokenizer.pad_token_id
        unk_id  = tokenizer.unk_token_id
        mask_id = tokenizer.mask_token_id
        q_ids = tokenizer.encode(question.strip(), add_special_tokens=False)
        prompt_ids = [cls_id] + q_ids + [sep_id]
        ban_ids = [pad_id, unk_id, cls_id, mask_id]
        end_id = sep_id
        end_str = "[SEP]"
    else:
        unk_id  = token2idx[UNK_TOKEN]
        pad_id  = token2idx[PAD_TOKEN]
        user_id = token2idx[USER_TOKEN]
        asst_id = token2idx[ASST_TOKEN]
        end_id  = token2idx[END_TOKEN]
        q_ids = [token2idx.get(c, unk_id) for c in question.strip()]
        prompt_ids = [user_id] + q_ids + [asst_id]
        ban_ids = [pad_id, unk_id, user_id, asst_id]
        end_str = END_TOKEN

    prefix_len = len(prompt_ids)
    generated_ids = []
    max_ctx = getattr(model, "max_len", 512) - 2

    for step in range(1, max_new_tokens + 1):
        ctx_ids = (prompt_ids + generated_ids)[-max_ctx:]
        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)

        if is_bert:
            logits = model(x, prefix_lens=[prefix_len], valid_lens=[len(ctx_ids)])[0, -1, :]
        else:
            logits = model(x)[0, -1, :]

        step_ban = ban_ids + ([end_id] if step == 1 else [])
        next_id, top_candidates = sample_next_token(
            logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            greedy=greedy,
            ban_ids=step_ban,
        )

        for cand in top_candidates:
            raw_t = idx2token.get(cand["id"], "[UNK]")
            cand["token"] = end_str if cand["id"] == end_id else (clean_bert_subword(raw_t) if is_bert else raw_t)

        if next_id == end_id:
            yield {
                "token": "",
                "raw_token": end_str,
                "token_id": next_id,
                "step": step,
                "finished": True,
                "candidates": top_candidates,
            }
            break

        generated_ids.append(next_id)
        raw_tok = idx2token.get(next_id, "")
        tok_str = clean_bert_subword(raw_tok) if is_bert else raw_tok
        yield {
            "token": tok_str,
            "raw_token": raw_tok,
            "token_id": next_id,
            "step": step,
            "finished": False,
            "candidates": top_candidates,
        }


@torch.no_grad()
def generate_reply(
    model,
    question,
    token2idx,
    idx2token,
    device,
    tokenizer=None,
    max_new_tokens=80,
    temperature=0.3,
    top_k=10,
    top_p=0.9,
    repetition_penalty=1.15,
    greedy=True,
):
    pieces = []
    steps_info = []
    for item in generate_reply_stream(
        model=model,
        question=question,
        token2idx=token2idx,
        idx2token=idx2token,
        device=device,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        greedy=greedy,
    ):
        steps_info.append(item)
        if item["token"]:
            pieces.append(item["token"])
    return "".join(pieces), steps_info


# ─────────────────────────── 训练与验证循环 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, grad_clip=1.0):
    model.train(train)
    total_loss = 0.0
    total_active_tokens = 0
    use_amp = device.type == "cuda"

    for x, y, prefix_lens, valid_lens in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        active_tokens = int((y != IGNORE_INDEX).sum().item())
        if active_tokens == 0:
            continue

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x, prefix_lens=prefix_lens, valid_lens=valid_lens)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().item()) * active_tokens
        total_active_tokens += active_tokens

    avg_loss = total_loss / max(1, total_active_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Week05 Prefix-LM (BERT) / Decoder-Only SFT 训练脚本")
    parser.add_argument("--backbone",      default="bert", choices=["bert", "scratch"],
                        help="选择模型底座: bert (加载 bert-base-chinese 改造成 Prefix-LM) 或 scratch (从零训练)")
    parser.add_argument("--bert_dir",      default=DEFAULT_BERT_DIR, help="bert-base-chinese 路径")
    parser.add_argument("--data_dir",      default=DEFAULT_DATA_DIR, help="SFT 数据集目录")
    parser.add_argument("--mask_question", type=str2bool, default=True,
                        help="是否对 Question 部分做 Loss Mask (设为 -100)，默认 True")
    parser.add_argument("--prefix_bi",     type=str2bool, default=True,
                        help="问题段是否开启双向注意力 (True=问题段双向+回答段因果，False=全序列纯下三角因果)")
    parser.add_argument("--epochs",        type=int,   default=8)
    parser.add_argument("--max_len",       type=int,   default=128)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--weight_decay",  type=float, default=1e-2)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--save",          default="best_sft_model.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    train_path, val_path = resolve_data_paths(args.data_dir)
    train_records = load_jsonl(train_path)
    val_records   = load_jsonl(val_path)

    pin_mem = device.type == "cuda"

    if args.backbone == "bert":
        bert_dir = resolve_bert_dir(args.bert_dir)
        tokenizer = BertTokenizer.from_pretrained(bert_dir)
        token2idx = tokenizer.get_vocab()
        idx2token = {int(v): k for k, v in token2idx.items()}
        vocab_size = len(token2idx)

        attn_mode_str = "问题段双向互看 + 回答段下三角因果 (Prefix-LM)" if args.prefix_bi else "全序列下三角因果 (Pure Causal)"
        print(f"device: {device}  backbone: bert-base-chinese")
        print(f"注意力模式: {attn_mode_str}  |  Question Loss Mask: {args.mask_question}")
        print(f"训练集: {len(train_records):,} 条  |  验证集: {len(val_records):,} 条  |  BERT 词表大小: {vocab_size:,}")

        train_ds = BertSFTDataset(train_records, tokenizer, max_len=args.max_len, mask_question=args.mask_question)
        val_ds   = BertSFTDataset(val_records,   tokenizer, max_len=args.max_len, mask_question=args.mask_question)

        sample_x, sample_y, sample_p, sample_v = train_ds[0]
        active_cnt = int((sample_y != IGNORE_INDEX).sum().item())
        print(f"示例[0] 问题段双向长度 P={sample_p} (全部打 -100) | 有效回答监督 token 数: {active_cnt} / 有效总长 {sample_v}")

        model = BertPrefixDecoderLM(bert_dir=bert_dir, prefix_bidirectional=args.prefix_bi).to(device)
    else:
        tokenizer = None
        token2idx, idx2token = build_vocab(train_records + val_records)
        vocab_size = len(token2idx)
        print(f"device: {device}  backbone: scratch Decoder-Only  |  Question Loss Mask: {args.mask_question}")
        train_ds = SFTDataset(train_records, token2idx, max_len=args.max_len, mask_question=args.mask_question)
        val_ds   = SFTDataset(val_records,   token2idx, max_len=args.max_len, mask_question=args.mask_question)
        model = TransformerDecoderOnlyLM(vocab_size=vocab_size, pad_id=token2idx[PAD_TOKEN]).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=pin_mem)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_val_loss = float("inf")
    best_val_ppl  = float("inf")
    best_epoch    = 1

    save_path = args.save
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'LR':>10}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        marker = "  *" if va_loss < best_val_loss else ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_ppl  = va_ppl
            best_epoch    = epoch
            torch.save({
                "backbone": args.backbone,
                "bert_dir": resolve_bert_dir(args.bert_dir) if args.backbone == "bert" else None,
                "bert_config": model.bert_mlm.config.to_dict() if args.backbone == "bert" else None,
                "prefix_bidirectional": args.prefix_bi,
                "model_state": model.state_dict(),
                "token2idx": token2idx,
                "idx2token": idx2token,
                "args": vars(args),
                "best_val_loss": best_val_loss,
                "best_val_ppl": best_val_ppl,
                "best_epoch": best_epoch,
                "epoch": epoch,
            }, save_path)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}  {cur_lr:>10.6f}{marker}")

    print(f"\n训练完成。最佳验证 Loss: {best_val_loss:.4f} (Epoch {best_epoch}) | 最佳验证 PPL: {best_val_ppl:.2f}")
    print(f"最佳模型已保存至: {save_path}")

    # 加载最佳权重进行抽测对比
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\n" + "=" * 72)
    print("shallowSeek SFT 模型生成测试 (问题段双向互看 + 回答段下三角因果生成):")
    print("=" * 72)
    demo_samples = [
        train_records[0],
        train_records[4],
        train_records[7],
        val_records[0],
        val_records[1],
        val_records[11],
        val_records[12],
        val_records[20],
    ]
    for i, item in enumerate(demo_samples, 1):
        q = item["instruction"]
        ref = item["output"]
        pred, _ = generate_reply(
            model, q, token2idx, idx2token, device, tokenizer=tokenizer, greedy=True
        )
        split_tag = "Train" if i <= 3 else "Val"
        print(f"[{split_tag} #{i}] 问: {q}")
        print(f"       参考: {ref}")
        print(f"       模型: {pred}\n")


if __name__ == "__main__":
    main()
