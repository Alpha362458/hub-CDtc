# -*- coding: utf-8 -*-
"""
run_experiments_v12.py —— 在同一份 v12 数据上对照替换组件

每次只改一件事，种子固定为 42。作业脚本 train_langid_cls_v12.py 不改动。

    python run_experiments_v12.py
"""

import time
import random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_langid_cls_v12 import (
    SEED, MAX_LEN, EMBED_DIM, HIDDEN_DIM, BATCH_SIZE, LR, EPOCHS, NUM_THREADS,
    DATA_DIR, LABELS, load_split, build_char_vocab, TextDataset,
    predict, macro_f1, confusion,
)

LATIN = {"en", "es", "pt", "vi", "tr"}
DISTANT = {"ar", "hi", "ja", "ru", "zh", "ko", "th"}


class AblationClassifier(nn.Module):
    """课堂模型的可替换版：encoder / pool / dropout / layernorm 都能关。"""

    def __init__(self, vocab_size, num_classes, encoder="rnn", pool="max",
                 dropout=0.3, layernorm=True, use_mask=True,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder_name = encoder
        self.pool = pool
        self.use_mask = use_mask
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if encoder == "rnn":
            self.seq = nn.RNN(embed_dim, hidden_dim, batch_first=True)
            feat_dim = hidden_dim
        elif encoder == "lstm":
            self.seq = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
            feat_dim = hidden_dim
        elif encoder == "gru":
            self.seq = nn.GRU(embed_dim, hidden_dim, batch_first=True)
            feat_dim = hidden_dim
        elif encoder == "mean":
            self.seq = None
            feat_dim = embed_dim
        else:
            raise ValueError(f"未知 encoder: {encoder}")

        self.norm = nn.LayerNorm(feat_dim) if layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

    def _pool(self, h, x):
        valid = (x != 0)
        if self.pool == "max":
            if self.use_mask:
                h = h.masked_fill(~valid.unsqueeze(-1), -1e9)
            return h.max(dim=1)[0]
        if self.pool == "mean":
            weight = valid.unsqueeze(-1).float() if self.use_mask else torch.ones_like(h[..., :1])
            return (h * weight).sum(1) / weight.sum(1).clamp(min=1)
        if self.pool == "last":
            lengths = valid.sum(1).clamp(min=1) - 1
            return h[torch.arange(h.size(0), device=h.device), lengths]
        raise ValueError(f"未知 pool: {self.pool}")

    def forward(self, x):
        emb = self.embedding(x)
        if self.encoder_name == "mean":
            weight = (x != 0).unsqueeze(-1).float()
            pooled = (emb * weight).sum(1) / weight.sum(1).clamp(min=1)
        else:
            h, _ = self.seq(emb)
            pooled = self._pool(h, x)
        return self.fc(self.dropout(self.norm(pooled)))


def group_acc(preds, golds, labels, group):
    idx = [i for i, g in enumerate(golds) if labels[g] in group]
    if not idx:
        return 0.0
    return sum(preds[i] == golds[i] for i in idx) / len(idx)


def per_class_acc(preds, golds, labels):
    out = OrderedDict()
    for c, lab in enumerate(labels):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        out[lab] = hit / max(1, len(idx))
    return out


def top_confusions(mat, labels, top=3):
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    return [(g, p, n) for n, g, p in pairs[:top] if n > 0]


def subsample_per_class(rows, n, seed=SEED):
    by_label = {}
    for t, l in rows:
        by_label.setdefault(l, []).append(t)
    out = [(t, l) for l, ts in by_label.items()
           for t in random.Random(seed).sample(ts, min(n, len(ts)))]
    random.Random(seed).shuffle(out)
    return out


def run_one(name, train_rows, dev_loader, test_loader, vocab, label2id,
            encoder="rnn", pool="max", dropout=0.3, layernorm=True,
            use_mask=True, epochs=EPOCHS):
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    model = AblationClassifier(
        len(vocab), len(LABELS), encoder=encoder, pool=pool,
        dropout=dropout, layernorm=layernorm, use_mask=use_mask,
    )
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    t0 = time.time()
    last_dev = 0.0
    last_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        last_loss = total / len(train_loader)
        preds, golds = predict(model, dev_loader)
        last_dev = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"    Epoch {epoch:2d}/{epochs}  loss={last_loss:.4f}  dev_acc={last_dev:.4f}")
    elapsed = time.time() - t0

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, len(LABELS))
    pca = per_class_acc(preds, golds, LABELS)
    conf = top_confusions(confusion(preds, golds, LABELS), LABELS)
    result = {
        "name": name,
        "params": n_params,
        "train_n": len(train_rows),
        "loss": last_loss,
        "dev_acc": last_dev,
        "acc": acc,
        "f1": f1,
        "latin": group_acc(preds, golds, LABELS, LATIN),
        "distant": group_acc(preds, golds, LABELS, DISTANT),
        "per_class": pca,
        "confusion": conf,
        "seconds": elapsed,
    }
    print(f"    test_acc={acc:.4f}  macro-F1={f1:.4f}  "
          f"拉丁语族={result['latin']:.3f}  远距语种={result['distant']:.3f}  "
          f"({elapsed:.0f}s, {n_params:,} params)")
    print("    混淆: " + ", ".join(f"{g}→{p} {n}" for g, p, n in conf))
    return result


def main():
    torch.set_num_threads(NUM_THREADS)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(DATA_DIR, "train")
    dev_rows = load_split(DATA_DIR, "dev")
    test_rows = load_split(DATA_DIR, "test")
    vocab, _ = build_char_vocab([t for t, _ in train_rows])
    label2id = {l: i for i, l in enumerate(LABELS)}

    # 架构对照共用同一张字符表、同一份编码，只有 per-class 实验会少用训练句
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    print(f"数据：train {len(train_rows)} / dev {len(dev_rows)} / test {len(test_rows)}")
    print(f"字符表 {len(vocab)}，类别 {len(LABELS)}：{' '.join(LABELS)}")
    print(f"拉丁语族 {sorted(LATIN)}  |  远距书写系统 {sorted(DISTANT)}\n")

    jobs = [
        dict(name="1 课堂原版 RNN+MaxPool", encoder="rnn", pool="max"),
        dict(name="2 RNN→LSTM", encoder="lstm", pool="max"),
        dict(name="3 RNN→GRU", encoder="gru", pool="max"),
        dict(name="4 去掉RNN，平均embedding", encoder="mean", pool="mean"),
        dict(name="5 Dropout=0", encoder="rnn", pool="max", dropout=0.0),
        dict(name="6 去掉LayerNorm", encoder="rnn", pool="max", layernorm=False),
        dict(name="7 每类只用50条", encoder="rnn", pool="max", per_class=50),
    ]

    results = []
    for i, job in enumerate(jobs, 1):
        per_class = job.pop("per_class", 0)
        name = job.pop("name")
        rows = subsample_per_class(train_rows, per_class) if per_class else train_rows
        print(f"[{i}/{len(jobs)}] {name}  (train={len(rows)})")
        results.append(run_one(name, rows, dev_loader, test_loader, vocab, label2id, **job))
        print()

    # ── 总表 ──────────────────────────────────────────────────────────
    print("=" * 78)
    print(f"{'实验':<28}{'test':>8}{'F1':>8}{'dev':>8}{'拉丁':>8}{'远距':>8}{'参数':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<28}{r['acc']:>8.3f}{r['f1']:>8.3f}{r['dev_acc']:>8.3f}"
              f"{r['latin']:>8.3f}{r['distant']:>8.3f}{r['params']:>10,}")
    print("=" * 78)

    print("\n各语种准确率：")
    header = f"{'实验':<28}" + "".join(f"{lab:>6}" for lab in LABELS)
    print(header)
    for r in results:
        print(f"{r['name']:<28}" + "".join(f"{r['per_class'][lab]:>6.2f}" for lab in LABELS))


if __name__ == "__main__":
    main()
