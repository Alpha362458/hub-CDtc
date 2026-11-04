# -*- coding: utf-8 -*-
"""
12 类语种识别：输入一句话，判断它属于哪种语言(12 分类)

数据：../语种识别实战/data/langid/v12  (train 9600 / dev 1200 / test 1800)
模型：Embedding -> RNN -> 掩码 MaxPool -> BatchNorm -> Dropout -> Linear
"""

import os
import csv
import random
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

######################### 全局量

SEED        = 42
MAX_LEN     = 40
EMBED_DIM   = 64
HIDDEN_DIM  = 64
DROPOUT     = 0.3
BATCH_SIZE  = 2 ** 6
LR          = 1e-3
EPOCHS      = 8
NUM_THREADS = 1        # 小模型单线程比多线程快，别改成 16

torch.set_num_threads(NUM_THREADS)
random.seed(SEED)
torch.manual_seed(SEED)

HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.normpath(os.path.join(HERE, "..", "语种识别实战", "data", "langid", "v12"))
VOCAB_OUT = os.path.join(HERE, "data", "vocab_v12.txt")
MODEL_OUT = os.path.join(HERE, "langid_rnn_v12.pt")

LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


########################### 文本归一化

def normalize(text):
    """NFKC + 小写 + 只保留字母(L)与组合符号(M)，必须和建数据时一致。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


########################### 构建词表

def build_vocab(sentences):
    """从训练集统计字符表：0=<PAD>，1=<UNK>，其余按出现次数从高到低编号。"""
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    """写成和 vocab_v8.txt 一样的格式：字符 \\t id \\t 出现次数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def coverage(vocab, freq, ratio=0.95):
    total = sum(freq.values())
    acc = n = 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq[ch]
        if acc >= ratio * total:
            break
    return n


def encode(text, vocab, max_len=MAX_LEN):
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in normalize(text)][:max_len]
    return ids + [vocab["<PAD>"]] * (max_len - len(ids))


########################### 读数据

def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, data, vocab, label2id):
        self.X = [encode(text, vocab) for text, _ in data]
        self.y = [label2id[label] for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


########################## RNN模型

class RNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, drop_out=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                  # 补齐位不参与池化
        e, _ = self.rnn(self.embedding(x))
        e = e.masked_fill(~mask, -1e9)
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)


########################## 评测

def evaluate(model, loader):
    """模型测试，返回准确率。"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


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


def print_report(preds, golds, top=5):
    """逐类准确率 + 最常混淆的组合。"""
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit:3d}/{len(idx)} = {hit / max(1, len(idx)):.3f}")

    pairs = Counter((LABELS[g], LABELS[p]) for p, g in zip(preds, golds) if p != g)
    print("\n  最常混淆的组合（真实 -> 预测）：")
    for (g, p), n in pairs.most_common(top):
        print(f"    {g} -> {p}: {n} 条")


########################## 训练

def train(epochs=EPOCHS):
    train_rows = load_split("train")
    dev_rows = load_split("dev")
    test_rows = load_split("test")

    seen = sorted({label for _, label in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}")
    label2id = {l: i for i, l in enumerate(LABELS)}

    vocab, freq = build_vocab([text for text, _ in train_rows])
    save_vocab(vocab, freq, VOCAB_OUT)

    print(f"数据目录：{DATA_DIR}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {len(LABELS)} 个：{' '.join(LABELS)}")
    print(f"  词表 {len(vocab)} 个字符（从训练集生成），前 {coverage(vocab, freq)} 个覆盖 95%")
    print(f"  随机猜的准确率基准：{1 / len(LABELS):.1%}")

    training_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                                 batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = RNN(vocab_size=len(vocab), num_classes=len(LABELS))
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in training_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(training_loader)
        dev_acc = evaluate(model, dev_loader)
        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  dev_acc={dev_acc:.4f}")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {macro_f1(preds, golds, len(LABELS)):.4f}")
    print_report(preds, golds)

    print("\n--- 推理示例 ---")
    model.eval()
    with torch.no_grad():
        for text, gold in test_rows[:6]:
            prob = torch.softmax(model(torch.tensor([encode(text, vocab)])), dim=1)[0]
            pid = int(prob.argmax())
            mark = "OK" if LABELS[pid] == gold else f"X({gold})"
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}] {mark}  {text[:50]}")

    torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                "labels": LABELS, "maxlen": MAX_LEN, "embed_dim": EMBED_DIM,
                "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT}, MODEL_OUT)
    print(f"\n模型已保存：{MODEL_OUT}")


train()
