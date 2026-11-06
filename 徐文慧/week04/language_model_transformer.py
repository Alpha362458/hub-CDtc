# -*- coding: utf-8 -*-
"""
字符级语言模型 —— Transformer 版（作业另存）

对应课堂脚本 language_model.py（RNN / LSTM）。
数据切分、损失、PPL 计算完全一致；只把骨干网络换成带因果掩码的 Transformer。

任务（和 RNN 版相同）：
    输入 x = 连续 seq_len 个字
    目标 y = 往后错一位（预测下一个字）
    指标 = 交叉熵 / 困惑度 PPL（越小越好）

和 RNN/LSTM 的关键差别：
    · RNN 天然只能看见过去；Transformer 默认能看见整句，所以必须加「因果掩码」
      （下三角可见、上三角 -inf），否则等于偷看了未来的字，不算合法语言模型。
    · Transformer 没有位置概念，要加 PositionalEncoding。

用法：
    python language_model_transformer.py --epochs 5
    python language_model_transformer.py --epochs 20 --stride 64 --save best_transformer.pt

提示：
    语料约 1700 万字。课堂默认 stride=1 会切出极多样本，Transformer 更慢。
    本脚本默认 stride=seq_len（不重叠），一轮能跑完；若要和课堂严格对齐，加 --stride 1。
"""

import math
import argparse
import glob
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据（与 language_model.py 一致） ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    x = 连续 seq_len 个字符，y = 往后错一位。
    每条长度固定，Transformer 这边不需要 padding mask，只要因果 mask。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 位置编码 ───────────────────────────

class PositionalEncoding(nn.Module):
    """固定 sin/cos 位置表，加到 embedding 上。不参与训练。"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────── Transformer 语言模型 ───────────────────────────

class TransformerLM(nn.Module):
    """
    Embedding → 位置编码 → TransformerEncoder（因果掩码）→ Linear → 词表分数

    因果掩码示意（T=4，True/-inf = 不许看）：
            位置0  1  2  3
        0    ✓   ✗  ✗  ✗     预测第1个字时只能看自己
        1    ✓   ✓  ✗  ✗
        2    ✓   ✓  ✓  ✗
        3    ✓   ✓  ✓  ✓
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_ff, dropout, max_len=512):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)
        self._d_model = d_model

    def _causal_mask(self, t, device):
        # 上三角（不含对角线）填 -inf：位置 i 看不见 j>i
        return torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)

    def forward(self, x):
        # x: (B, T)
        b, t = x.shape
        h = self.embed(x) * math.sqrt(self._d_model)   # 常见缩放，稳住点积注意力
        h = self.pos(h)
        mask = self._causal_mask(t, x.device)
        h = self.encoder(h, mask=mask)
        return self.fc(self.drop(h))                   # (B, T, V)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            # Transformer 偶发梯度爆炸，裁一下更稳
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20))   # 防止 overflow 打印成 inf
    return avg_loss, ppl


def generate(model, char2idx, idx2char, prompt, max_new=50, temperature=1.0, device="cpu"):
    """贪心/采样续写，用来肉眼看模型有没有学会「像中文」。"""
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        ids = [0]
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-64:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
    return "".join(idx2char[i] for i in ids)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="字符级 Transformer 语言模型")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=None,
                        help="切块间隔；默认=seq_len（不重叠）。要对齐课堂 LSTM 用 --stride 1")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--nhead",      type=int,   default=4)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dim_ff",     type=int,   default=512)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer.pt")
    parser.add_argument("--sample",     default="今天天气", help="训练结束后的续写提示")
    args = parser.parse_args()

    if args.stride is None:
        args.stride = args.seq_len

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请在含 corpus_general.txt 的目录运行。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)
    print(f"训练样本: {len(train_ds):,}  验证样本: {len(val_ds):,}  "
          f"(seq_len={args.seq_len}, stride={args.stride})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        max_len=max(512, args.seq_len),
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"结构: d_model={args.d_model}  nhead={args.nhead}  "
          f"layers={args.num_layers}  ff={args.dim_ff}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  "
          f"{'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "arch": "transformer",
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": {int(k): v for k, v in idx2char.items()},
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    sample = generate(model, char2idx, idx2char, args.sample, max_new=40, device=device)
    print(f"续写示例: {sample}")


if __name__ == "__main__":
    main()
