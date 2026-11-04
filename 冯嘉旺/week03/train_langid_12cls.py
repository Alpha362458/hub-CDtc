# -*- coding: utf-8 -*-
"""
train_langid_12cls.py —— 12 类语种识别作业脚本

在原课堂 8 类代码的基础上独立实现，不依赖 train_langid_cls.py。

任务：一句话 -> 它是哪种语言？
类别：ar en es hi ja ko pt ru th tr vi zh
模型：Embedding -> RNN -> 掩码 MaxPool -> LayerNorm -> Dropout -> Linear

用法：
    python train_langid_12cls.py
    python train_langid_12cls.py --epochs 10 --save langid_rnn_v12.pt

首次运行会根据 data/langid/v12/train.csv 生成
data/langid/vocab/vocab_v12.txt（字符 \t id \t 出现次数）。

训练完成后可直接用配套评估脚本：
    python eval_langid_12cls.py --model langid_rnn_v12.pt --data-dir data/langid/v12 --split test
    python eval_langid_12cls.py --model langid_rnn_v12.pt --data-dir data/langid/v12 --split test_hard
    python eval_langid_12cls.py --model langid_rnn_v12.pt --data-dir data/langid/v12 --openset
"""

import os
import sys
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Windows 默认 GBK 控制台无法打印印地语等字符，统一走 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


SEED = 42
MAX_LEN = 40  # 中文中位长度约 10 字符，统一截到 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 12 类按字母序排列。模型输出维度 = len(LABELS)。
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


def normalize(text):
    # NFKC 归一化 + 转小写 + 只保留字母(L)与组合符号(M)。
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def build_char_vocab(sentences, min_freq=1):
    """从训练集统计字符表，返回 (vocab, freq)。"""
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(
        ((c, n) for c, n in freq.items() if n >= min_freq),
        key=lambda kv: (-kv[1], kv[0]),
    )
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def load_vocab_file(path):
    # 读取「字符 \t id \t 出现次数」格式的字符表。
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


class LangIDClassifier(nn.Module):
    # Embedding -> RNN -> 掩码 MaxPool -> LayerNorm -> Dropout -> Linear

    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.embedding(x))
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)


def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=3):
    pairs = [
        (mat[g][p], labels[g], labels[p])
        for g in range(len(labels))
        for p in range(len(labels))
        if g != p
    ]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


def save_vocab(vocab, freq, path):
    """按字符表格式写出：字符 \\t id \\t 出现次数。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def ensure_vocab(train_rows, path, rebuild=False):
    """字符表不存在或指定重建时，从训练集生成。"""
    if os.path.exists(path) and not rebuild:
        vocab = load_vocab_file(path)
        print(f"字符表：复用 {path}（{len(vocab)} 个字符）")
        return vocab

    vocab, freq = build_char_vocab([t for t, _ in train_rows])
    save_vocab(vocab, freq, path)
    print(f"字符表：已从训练集生成 {path}（{len(vocab)} 个字符）")
    return vocab


def sample_per_class(rows, per_class, seed=SEED):
    """每类只取前 N 条，用于演示数据量不足时的过拟合。"""
    by_label = {}
    for text, label in rows:
        by_label.setdefault(label, []).append(text)
    rng = random.Random(seed)
    sampled = [
        (t, l)
        for l, texts in by_label.items()
        for t in rng.sample(texts, min(per_class, len(texts)))
    ]
    rng.shuffle(sampled)
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument(
        "--rebuild-vocab", action="store_true", help="强制从训练集重新生成字符表"
    )
    ap.add_argument("--save", default=os.path.join(HERE, "langid_rnn_v12.pt"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 数据里的标签必须与 12 类列表完全一致，否则输出格子对不上。
    seen = sorted({l for _, l in train_rows})
    missing = [l for l in LABELS if l not in seen]
    extra = [l for l in seen if l not in LABELS]
    if missing or extra:
        raise SystemExit(
            f"训练集标签与 LABELS 不一致。\n"
            f"  缺失：{' '.join(missing) or '无'}\n"
            f"  多出：{' '.join(extra) or '无'}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )

    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    # 字符表始终用完整训练集统计，避免 --per-class 时字符覆盖不足。
    vocab = ensure_vocab(train_rows, args.vocab, args.rebuild_vocab)

    if args.per_class:
        train_rows = sample_per_class(train_rows, args.per_class)

    print(f"数据目录：{args.data_dir}")
    print(
        f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条"
    )
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（{os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1 / num_classes:.1%}")

    train_loader = DataLoader(
        TextDataset(train_rows, vocab, label2id), batch_size=BATCH_SIZE, shuffle=True
    )
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), num_classes)
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        preds, golds = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(
            f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
            f"dev_acc={acc:.4f}"
        )
    elapsed = time.time() - t0
    print(
        f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）"
    )

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1, len(idx)):.3f}")

    print("\n--- 推理示例 ---")
    model.eval()
    with torch.no_grad():
        for text in [t for t, _ in test_rows[:6]]:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab": dict(vocab),
                "labels": LABELS,
                "maxlen": MAX_LEN,
                "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT,
                "data_dir": args.data_dir,
            },
            args.save,
        )
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
