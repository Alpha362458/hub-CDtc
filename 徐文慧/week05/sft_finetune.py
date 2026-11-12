# -*- coding: utf-8 -*-
"""
SFT（Supervised Fine-Tuning）—— 在预训练字符级 Transformer LM 上做指令微调

前置：
    先用 language_model_transformer.py 训好 best_transformer_v2.pt（或其它 Transformer 权重）

和预训练 LM 的差别：
    · 预训练：海量普通文本，学「下一个字像不像中文」
    · SFT：instruction → output 成对数据，学「看到问题后该怎么答」
    · 损失只算在「回答」部分（指令部分 ignore），避免模型只学会复读问题

数据：
    sft_train.jsonl / sft_val.jsonl
    每行: {"instruction": "...", "output": "..."}

用法（在 sft_train 目录下）:
    python sft_finetune.py
    python sft_finetune.py --ckpt ../循环神经网络语言模型/best_transformer_v2.pt --epochs 5

训完会保存 best_sft.pt，并打印几条问答演示。
"""

import os
import sys
import json
import math
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
LM_DIR = HERE.parent / "循环神经网络语言模型"
sys.path.insert(0, str(LM_DIR))

from language_model_transformer import TransformerLM  # noqa: E402

INSTR_PREFIX = "指令："
REPLY_PREFIX = "回答："
EOS = "\n"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "instruction" in obj and "output" in obj:
                rows.append(obj)
    return rows


def format_example(instruction, output):
    """拼成一条训练文本，并返回「回答正文」从第几个字符开始。"""
    head = INSTR_PREFIX + instruction + "\n" + REPLY_PREFIX
    text = head + output + EOS
    answer_start = len(head)  # 回答第一个字的下标（含在 text 里）
    return text, answer_start


def expand_vocab(char2idx, texts):
    """预训练词表上补 SFT 里新出现的字；旧编号保持不变，方便加载旧权重。"""
    char2idx = dict(char2idx)
    for t in texts:
        for ch in t:
            if ch not in char2idx:
                char2idx[ch] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def resize_model_vocab(model, old_vocab, new_vocab):
    """词表变大时，扩 Embedding / 输出层；旧行权重原样拷贝。"""
    if new_vocab == old_vocab:
        return model
    device = next(model.parameters()).device
    d_model = model.embed.embedding_dim

    new_embed = nn.Embedding(new_vocab, d_model).to(device)
    new_fc = nn.Linear(d_model, new_vocab).to(device)
    with torch.no_grad():
        new_embed.weight[:old_vocab] = model.embed.weight
        new_fc.weight[:old_vocab] = model.fc.weight
        new_fc.bias[:old_vocab] = model.fc.bias
    model.embed = new_embed
    model.fc = new_fc
    return model


class SFTDataset(Dataset):
    """
    每条样本：
        x = text[:-1] 的字编号
        y = text[1:]  的字编号
        指令部分对应的 y 位置设为 -100（CrossEntropy 的 ignore_index）
    """

    def __init__(self, rows, char2idx, max_len=128):
        self.samples = []
        unk_drop = 0
        for r in rows:
            text, answer_start = format_example(r["instruction"], r["output"])
            # 超长则截断，但尽量保住「回答：」之后的内容
            if len(text) > max_len:
                text = text[:max_len]
            ids = []
            ok = True
            for ch in text:
                if ch not in char2idx:
                    ok = False
                    break
                ids.append(char2idx[ch])
            if not ok or len(ids) < 2:
                unk_drop += 1
                continue

            x = ids[:-1]
            y = ids[1:]
            # text[i] 预测 text[i+1]；若 text[i+1] 仍属于指令区，则忽略
            # 回答正文从 answer_start 开始 → 预测目标下标 j=i+1，当 j < answer_start 时忽略
            labels = []
            for i in range(len(y)):
                target_pos = i + 1  # y[i] 对应 text 里的下标
                if target_pos < answer_start:
                    labels.append(-100)
                else:
                    labels.append(y[i])

            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            ))
        self.unk_drop = unk_drop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pad(batch, pad_id=0):
    """把不同长度的样本补齐到本 batch 最长；label 的 pad 用 -100。"""
    xs, ys = zip(*batch)
    max_t = max(x.size(0) for x in xs)
    x_out, y_out = [], []
    for x, y in zip(xs, ys):
        pad_n = max_t - x.size(0)
        if pad_n:
            x = torch.cat([x, torch.full((pad_n,), pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_n,), -100, dtype=torch.long)])
        x_out.append(x)
        y_out.append(y)
    return torch.stack(x_out), torch.stack(y_out)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        # 只统计非 ignore 的 token
        n = (y != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    avg = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


def sft_reply(model, char2idx, idx2char, instruction, max_new=80, device="cpu"):
    """推理：拼上「指令/回答：」前缀，只续写回答；遇到换行就停。"""
    prompt = INSTR_PREFIX + instruction + "\n" + REPLY_PREFIX
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        ids = [0]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-128:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            next_id = torch.multinomial(torch.softmax(logits / 0.7, dim=-1), 1).item()
            ch = idx2char[next_id]
            ids.append(next_id)
            if ch == "\n":
                break
    return "".join(idx2char[i] for i in ids)


def main():
    ap = argparse.ArgumentParser(description="字符级 Transformer 的 SFT 微调")
    ap.add_argument("--ckpt", default=str(LM_DIR / "best_transformer_v2.pt"),
                    help="预训练 Transformer LM 权重")
    ap.add_argument("--train", default=str(HERE / "sft_train.jsonl"))
    ap.add_argument("--val", default=str(HERE / "sft_val.jsonl"))
    ap.add_argument("--save", default=str(HERE / "best_sft.pt"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(
            f"找不到预训练权重: {args.ckpt}\n"
            f"请先运行 language_model_transformer.py 得到 best_transformer_v2.pt"
        )

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    char2idx = dict(ck["char2idx"])
    old_vocab = len(char2idx)

    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    print(f"SFT 数据: train {len(train_rows)} / val {len(val_rows)}")

    # 扩展词表（SFT 里可能有预训练没见过的符号）
    all_text = []
    for r in train_rows + val_rows:
        t, _ = format_example(r["instruction"], r["output"])
        all_text.append(t)
    char2idx, idx2char = expand_vocab(char2idx, all_text)
    new_vocab = len(char2idx)
    print(f"词表: 预训练 {old_vocab} → SFT 后 {new_vocab} "
          f"(+{new_vocab - old_vocab})")

    train_ds = SFTDataset(train_rows, char2idx, max_len=args.max_len)
    val_ds = SFTDataset(val_rows, char2idx, max_len=args.max_len)
    print(f"可用样本: train {len(train_ds)} / val {len(val_ds)} "
          f"(过滤 {train_ds.unk_drop + val_ds.unk_drop} 条)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pad, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pad, drop_last=False,
    )

    model = TransformerLM(
        vocab_size=old_vocab,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_ff=cfg["dim_ff"],
        dropout=cfg.get("dropout", 0.2),
        max_len=max(512, args.max_len),
    )
    model.load_state_dict(ck["model_state"], strict=True)
    model = resize_model_vocab(model, old_vocab, new_vocab)
    model = model.to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, False)
        mark = ""
        if va_ppl < best_val:
            best_val = va_ppl
            mark = "  *"
            torch.save({
                "arch": "transformer_sft",
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": {int(k): v for k, v in idx2char.items()},
                "args": vars(args),
                "base_ckpt": os.path.basename(args.ckpt),
                "lm_args": cfg,
            }, args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{mark}")

    print(f"\nSFT 完成。最佳验证 PPL: {best_val:.2f}  → {args.save}")

    # 演示：同一问题，看微调后怎么答
    demos = [
        "给出两个数字，求它们的平均值。 25, 41",
        "请解释以下缩写的含义：NLP。",
        "请问北京奥运会的举办时间和地点分别是什么？",
    ]
    print("\n--- 问答演示 ---")
    model.eval()
    for q in demos:
        ans = sft_reply(model, char2idx, idx2char, q, max_new=60, device=device)
        # 只展示「回答：」之后
        if REPLY_PREFIX in ans:
            ans = ans.split(REPLY_PREFIX, 1)[1].strip()
        print(f"Q: {q}")
        print(f"A: {ans}\n")


if __name__ == "__main__":
    main()
