# -*- coding: utf-8 -*-
"""
train_langid_cls.py —— 多语种「语种识别」分类器（12 语种 GPU 最优版）

任务：一句话 → 它是什么语言（12 分类：ar en es hi ja ko pt ru th tr vi zh）
硬件适配：针对 NVIDIA GeForce RTX 5060 (8GB VRAM, Compute Capability 12.0) 深度调优
模型：Embedding(192) → EmbDropout(0.15)
      ├─ 多尺度字符 N-Gram 卷积分支 (k=1,2,3,5, 捕捉拉丁语系词缀/重音与局部字符组合)
      └─ 双向 GRU 时序分支 (BiGRU, 捕捉正反向全局上下文依赖)
      → 掩码双池化 (Masked MaxPool ⊕ Masked AvgPool)
      → LayerNorm → Dropout(0.35) → Linear + GELU + Linear → 12 类 Logits
损失：CrossEntropyLoss
优化器：AdamW (lr=2e-3, weight_decay=1e-2) + CosineAnnealingLR + 梯度裁剪 + 验证集最优权重保存
数据：data/langid/v12（12 语种：每类 800 训练 / 100 验证 / 150 测试）

运行示例：
    python train_langid_cls.py                           # 默认训练 12 语种最优模型并保存 langid_rnn_v12.pt
    python train_langid_cls.py --save langid_rnn_v12.pt  # 指定保存路径供 eval_langid.py / demo_mixed_text.py 使用
    python train_langid_cls.py --baseline                # 运行原始单向小 RNN 基线作对照
"""

import os
import csv
import copy
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── 超参数（针对 RTX 5060 8GB GPU 调优）────────────────────────────────
SEED = 42
MAX_LEN = 40          # 统一截断/补齐长度 40
EMBED_DIM = 192       # 原版 64 → 192（适配 3276 字符表与 12 语种细粒度表示）
HIDDEN_DIM = 192      # 原版 64 → 192（双向 + 多尺度卷积特征维度）
DROPOUT = 0.35        # 原版 0.3 → 0.35（配合更大模型容量防过拟合）
EMB_DROPOUT = 0.15    # 新增：Embedding 层后轻量 Dropout
BATCH_SIZE = 128      # 原版 64 → 128（充分利用 RTX 5060 Tensor Cores 并行吞吐）
LR = 2e-3             # 原版 1e-3 → 2e-3（配合大 Batch 与余弦退火调度）
WEIGHT_DECAY = 1e-2   # 新增：AdamW 解耦权重衰减
GRAD_CLIP = 1.0       # 新增：梯度范数裁剪阈值，稳定循环层训练
EPOCHS = 16           # 原版 8 → 16（配合 CosineAnnealingLR 充分收敛）
NUM_THREADS = 4       # CPU 辅助线程数

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")
DEFAULT_SAVE = os.path.join(HERE, "langid_rnn_v12.pt")

# 作业 12 个语种（课内 8 语种 + ko 韩语 / th 泰语 / tr 土耳其语 / vi 越南语）
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


# ─── 1. 文本归一化（复用已有实现，与建数据时完全一致）────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建与读写（复用 build_langid_dataset.py / train_langid_cls.py）
def build_char_vocab(sentences, min_freq=1):
    """
    统计训练集所有字符的出现次数 → 按频次从高到低编号。
    第 0 号 <PAD>，第 1 号 <UNK>。
    """
    freq = Counter()
    for s in sentences:
        freq.update(normalize(s))
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in chars:
        vocab[ch] = len(vocab)
    return vocab, freq


def save_vocab(vocab, freq, path):
    """写出字符表文件（格式：字符 \\t id \\t 出现次数）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def coverage(vocab, freq, ratio=0.95):
    """前多少个字符覆盖 ratio 的字符出现次数。"""
    total = sum(freq.values())
    acc, n = 0, 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if acc >= ratio * total:
            return n
    return n


def load_vocab_file(path):
    """读取已经生成好的字符表。"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def ensure_vocab(vocab_path, train_rows):
    """若 vocab_v12.txt 不存在，自动从 v12/train.csv 构建并保存。"""
    if not os.path.exists(vocab_path):
        vocab, freq = build_char_vocab([t for t, _ in train_rows])
        save_vocab(vocab, freq, vocab_path)
        cov95 = coverage(vocab, freq, 0.95)
        print(f"  [自动生成词表] {vocab_path}（共 {len(vocab)} 个字符，前 {cov95} 个覆盖 95%）")
    return load_vocab_file(vocab_path)


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["text"], r["label"]))
    return rows


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        # 一次性转为 Tensor，避免每个 epoch 在 __getitem__ 里重复构造小张量
        self.X = torch.tensor([encode(t, vocab) for t, _ in rows], dtype=torch.long)
        self.y = torch.tensor([label2id[l] for _, l in rows], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class FastCharConv1d(nn.Module):
    """
    基于滑动窗口拼接 + Linear 的一维字符 N-Gram 卷积层。
    数学上等价于 Conv1d(in_channels, out_channels, kernel_size, padding=k//2)，
    在 GPU 上直接调用 cuBLAS 矩阵乘法（Tensor Cores），捕获局部字符组合特征。
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.proj = nn.Linear(in_channels * kernel_size, out_channels)

    def forward(self, x):
        if self.kernel_size == 1:
            return F.gelu(self.proj(x))
        L = x.size(1)
        x_pad = F.pad(x, (0, 0, self.pad, self.pad))
        cat_x = torch.cat([x_pad[:, i:i + L, :] for i in range(self.kernel_size)], dim=-1)
        return F.gelu(self.proj(cat_x))


class LangIDClassifier(nn.Module):
    """
    12 语种最优分类器（兼容 eval_langid.py 的调用签名）：
    Embedding → EmbDropout → [多尺度字符 N-Gram CNN (k=1,2,3,5) ⊕ 双向 GRU]
              → 掩码双池化 (Masked MaxPool ⊕ Masked AvgPool)
              → LayerNorm → Dropout → MLP(Linear-GELU-Dropout-Linear)
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 baseline=False):
        super().__init__()
        self.baseline = baseline
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if baseline:
            # 原始 8 类课堂脚本里的单向简单 RNN 结构（供 --baseline 对比）
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            self.emb_drop = nn.Dropout(EMB_DROPOUT)
            c_dim = hidden_dim // 2
            # 多尺度字符 N-Gram 卷积：k=1(单字), k=2(二元), k=3(三元词缀), k=5(五元词根)
            self.convs = nn.ModuleList([
                FastCharConv1d(embed_dim, c_dim, kernel_size=k)
                for k in (1, 2, 3, 5)
            ])
            # 双向 GRU：同时建模从左到右、从右到左的字符序列转移规律
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, num_layers=1,
                batch_first=True, bidirectional=True
            )
            feat_dim = c_dim * 4 + hidden_dim * 2          # 4*c_dim + 2*H = 4*H
            pooled_dim = feat_dim * 2                      # MaxPool ⊕ AvgPool
            self.norm = nn.LayerNorm(pooled_dim)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(pooled_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim * 2, num_classes)
            )

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)                      # (B, L, 1) 补齐位为 False
        if self.baseline:
            h, _ = self.rnn(self.embedding(x))             # (B, L, H)
            h = h.masked_fill(~mask, -1e9)
            pooled = h.max(dim=1)[0]                       # (B, H)
            pooled = self.dropout(self.norm(pooled))
            return self.fc(pooled)

        emb = self.emb_drop(self.embedding(x)) * mask      # (B, L, E) 屏蔽补齐位
        cnn_feat = torch.cat([conv(emb) for conv in self.convs], dim=-1)
        rnn_feat, _ = self.rnn(emb)
        h = torch.cat([cnn_feat, rnn_feat], dim=-1)        # (B, L, 4*H)

        # 掩码最大池化（抓最具代表性的语种特征）+ 掩码均值池化（抓整句主体语种分布）
        h_max = h.masked_fill(~mask, -1e9).max(dim=1)[0]
        lengths = mask.sum(dim=1).clamp(min=1).float()
        h_avg = (h * mask).sum(dim=1) / lengths

        pooled = torch.cat([h_max, h_avg], dim=-1)         # (B, 8*H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                             # (B, C)


# ─── 5. 评测（复用已有指标函数）────────────────────────────────────────
def predict(model, loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds += model(X).argmax(dim=1).cpu().tolist()
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """每一类各自算 F1，再对类别取平均。"""
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


def print_confusion(mat, labels, top=5):
    """打印最常混淆的语种组合。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练主流程 ─────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH,
                    help="12 语种字符表文件（不存在时自动从 v12/train.csv 构建）")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    ap.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    ap.add_argument("--dropout", type=float, default=DROPOUT)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来演示「数据少了会过拟合」")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--baseline", action="store_true",
                    help="使用原版小 RNN(64d, 8 epochs) 配置在 12 类上训练作对照")
    ap.add_argument("--save", default=DEFAULT_SAVE,
                    help="保存模型到该路径，供 eval_langid.py 与 demo_mixed_text.py 使用")
    args = ap.parse_args()

    if args.baseline:
        args.embed_dim = 64
        args.hidden_dim = 64
        args.dropout = 0.3
        args.batch_size = 64
        args.lr = 1e-3
        args.epochs = 8

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(args.device)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    vocab = ensure_vocab(args.vocab, train_rows)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_info = f"{props.name} ({props.total_memory / 1024**3:.1f} GB VRAM, CC {props.major}.{props.minor})"
    else:
        gpu_info = "CPU"

    print(f"运行设备：{device} — {gpu_info}")
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（来自 {os.path.basename(args.vocab)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=256)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=256)

    model = LangIDClassifier(
        len(vocab), num_classes,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout, baseline=args.baseline
    ).to(device)
    print(f"  模型结构：{'Baseline 单向 RNN' if args.baseline else 'Multi-Scale CharCNN (k=1,2,3,5) + BiGRU + DualPool'}")
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    if args.baseline:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-5
        )

    best_dev_acc = 0.0
    best_epoch = 1
    best_state = None

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if not args.baseline:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total += loss.item()
        if scheduler is not None:
            scheduler.step()

        preds, golds = predict(model, dev_loader, device)
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        cur_lr = optimizer.param_groups[0]["lr"]
        mark = ""
        if acc >= best_dev_acc:
            best_dev_acc = acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " ★"
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}  lr={cur_lr:.2e}{mark}")

    if best_state is not None and not args.baseline:
        model.load_state_dict(best_state)
        print(f"\n已加载验证集最优权重（Epoch {best_epoch}, dev_acc={best_dev_acc:.4f}）")

    elapsed = time.time() - t0
    print(f"训练耗时：{elapsed:.2f} 秒（{args.epochs} 个 epoch，设备：{device}）")

    # 测试集
    preds, golds = predict(model, test_loader, device)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集（{len(test_rows)} 条）：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:6]]
    with torch.no_grad():
        for text in demos:
            inp = torch.tensor([encode(text, vocab)], dtype=torch.long, device=device)
            logits = model(inp)
            prob = torch.softmax(logits, dim=1)[0].cpu()
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        cpu_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": cpu_sd, "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": args.embed_dim, "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout, "baseline": args.baseline}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
