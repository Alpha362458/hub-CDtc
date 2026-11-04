# -*- coding: utf-8 -*-
"""
train_langid_cls_v12.py —— 多语种「语种识别」分类器（综合实战 · 12 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear → Softmax
损失：CrossEntropyLoss      优化器：Adam
运行：CPU 即可；解释器用 py312（torch 2.6.0+cpu），8 个 epoch 几秒到十几秒。

【数据在哪】按顺序自动找，不用改代码、也不用带参数：
  1. 课堂目录结构：<脚本目录>/data/langid/v12/{train,dev,test}.csv
     字符表：<脚本目录>/data/langid/vocab/vocab_v12.txt
  2. 交作业的扁平结构：train.csv / test.csv / vocab_v12.txt 直接放在脚本旁边
  · 没有 dev.csv 时，从训练集里按标签分层切 10% 当验证集（只用于打日志）
  · 也可以用 --data-dir / --vocab 显式指定
  · 字符表缺失时用 build_char_vocab() 从 train.csv 现场统计并写出来

【和课堂 8 类版 train_langid_cls.py 相比，只改了三处「跟语种数量绑定」的地方】
  1. LABELS：8 个语种 → 12 个语种，模型输出维度 = len(LABELS) = 12
     （旧模型只有 8 个输出格子，多出来的 ko/th/vi/tr 没有地方放，必须加宽）
  2. 数据：data/langid/v8 → data/langid/v12（课堂 8 语种 + ko / th / vi / tr）
  3. 字符表：改读 vocab_v12.txt
     —— 课堂那张 vocab_v8.txt 里没有韩文、泰文，也没有 vi / tr 的带附加符号拉丁字母，
        直接拿旧表训练，这些字会全部落到 1 号 <UNK>，模型等于睁眼瞎。

【字符表是怎么生成的】见 build_char_vocab()：
  把训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号；
  第 0 号固定 <PAD>、第 1 号固定 <UNK>，次数相同按字符编码排序（保证可复现）。
  生成的文件每行「字符 \\t id \\t 出现次数」，和 vocab_v8.txt 格式一致。

【运行（py312，CPU 即可）】
    conda activate py312
    python train_langid_cls_v12.py                      # 训练 + 评测（缺字符表会自动生成）
    python train_langid_cls_v12.py --save langid_rnn_v12.pt
    也可以直接双击「运行_py312.bat」，它写死了 py312 的解释器路径

【用改好的模型跑评测/试玩】
    python eval_langid.py --model langid_rnn_v12.pt --data-dir <你的数据目录>
"""

import os
import sys
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

# Windows 控制台默认是 GBK，打印韩文/泰文/越南语示例句会抛 UnicodeEncodeError，
# 这里把无法编码的字符降级成 ?，保证推理示例那一栏不会把整个训练脚本打断。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:        # 最常见的原因：用了 base 的 python，那个环境没装 torch
    raise SystemExit(
        f"\n导入 PyTorch 失败：{exc}\n"
        f"  当前解释器：{sys.executable}\n"
        f"  Python 版本：{sys.version.split()[0]}\n"
        "本脚本要跑在 py312 环境里，任选一种：\n"
        "  1) 双击 运行_py312.bat\n"
        "  2) conda activate py312  然后  python train_langid_cls_v12.py\n"
        "  3) 用 py312 的解释器直接跑：\n"
        "     D:\\anaconda3\\envs\\py312\\python.exe train_langid_cls_v12.py\n"
    )

# ─── 超参数（与课堂 8 类版保持一致）─────────────────────────────────────
SEED = 42
MAX_LEN = 40
EMBED_DIM = 64
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 8
NUM_THREADS = 1       # 小模型单线程比多线程快，别改成 16

HERE = os.path.dirname(os.path.abspath(__file__))

# 作业的 12 个语种 = 课堂 8 个 + ko / th / vi / tr
# 顺序固定写死：模型的第 i 个输出格子就是 LABELS[i]，存档时一起存进 checkpoint
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]


# ─── 0. 数据 / 字符表位置（课堂结构 和 交作业的扁平结构 都认）──────────
def default_vocab_path(data_dir):
    """字符表默认落点：扁平结构 → 脚本旁边；课堂结构 → data/langid/vocab/。"""
    if os.path.abspath(data_dir) == os.path.abspath(HERE):
        return os.path.join(HERE, "vocab_v12.txt")
    return os.path.join(os.path.dirname(os.path.abspath(data_dir)), "vocab", "vocab_v12.txt")


def find_data_dir(data_dir=None):
    """命令行 --data-dir > 课堂结构 data/langid/v12 > 脚本同目录（扁平结构）。"""
    if data_dir:
        return os.path.abspath(data_dir)
    candidates = [os.path.join(HERE, "data", "langid", "v12"), HERE]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "train.csv")):
            return cand
    raise SystemExit(
        "找不到 train.csv。\n"
        "  把 train.csv / test.csv 放在脚本同目录，或放到 data/langid/v12/ 下，\n"
        "  也可以用 --data-dir 指定。已尝试：\n    "
        + "\n    ".join(candidates)
    )


def find_vocab_path(vocab=None, data_dir=HERE):
    """命令行 --vocab > 数据目录旁边 > 脚本同目录 > 课堂结构 vocab/。"""
    if vocab:
        return os.path.abspath(vocab)
    near = default_vocab_path(data_dir)
    for cand in (near,
                 os.path.join(HERE, "vocab_v12.txt"),
                 os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")):
        if os.path.exists(cand):
            return cand
    return near


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि、ที่ 里的 ่），
    只留字母会把印地语砍成裸辅音、把泰语的声调符号丢光，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """
    字符表 = 训练集里所有出现过的字符，每个字符发一个整数编号。

    第 0 号 <PAD>：补齐用（Embedding 里 padding_idx=0）
    第 1 号 <UNK>：没见过的字符（字符级下几乎用不到）
    其余按出现次数从高到低编号，次数相同按字符编码排序（保证可复现）

    12 类版的关键点：这张表把 12 种语言的字符全排在同一张表里——
    汉字、假名、韩文、拉丁字母（含 vi/tr 的附加符号）、西里尔字母、
    阿拉伯字母、天城文、泰文。字符级天然多语种，不需要按语种分开建表。
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
    """写出字符表：每行「字符 \\t id \\t 出现次数」，格式与 vocab_v8.txt 完全一致。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def load_vocab_file(path):
    """读字符表文件；# 开头与空行跳过，每行三列「字符 id 次数」。"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)
    return vocab


def ensure_vocab(data_dir, vocab_path, rebuild=False):
    """
    12 类字符表的唯一入口：
      · 文件不存在（或 --rebuild-vocab）→ 从 train.csv 统计并写文件
      · 文件已存在 → 直接读
    这样第一次跑训练就顺手把 vocab_v12.txt 生成好了。
    """
    if rebuild or not os.path.exists(vocab_path):
        print(f"字符表{'重建' if rebuild else '缺失'}：从 {os.path.join(data_dir, 'train.csv')} 统计 ...")
        rows = load_split(data_dir, "train")
        vocab, freq = build_char_vocab([t for t, _ in rows])
        rel = os.path.relpath(vocab_path, HERE).replace("\\", "/")
        save_vocab(vocab, freq, vocab_path)
        print(f"  已写出 {rel}：{len(vocab)} 个字符")
        return vocab
    return load_vocab_file(vocab_path)


def coverage(vocab, freq=None, ratio=0.95):
    """前多少个字符覆盖 ratio 的非 PAD/UNK 字符频次（freq 为空时返回 0）。"""
    if not freq:
        return 0
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


def split_dev_from_train(rows, ratio=0.1, seed=SEED):
    """
    dev.csv 缺失时的兜底：按标签分层切 ratio 比例做验证集。

    验证集只用来打日志、不参与训练，所以直接从训练集里拿走这一小部分。
    """
    rng = random.Random(seed)
    by_label = {}
    for t, l in rows:
        by_label.setdefault(l, []).append((t, l))
    dev, keep = [], []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        k = max(1, int(round(len(items) * ratio)))
        dev += items[:k]
        keep += items[k:]
    rng.shuffle(dev)
    rng.shuffle(keep)
    return keep, dev


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


# ─── 4. 模型（结构与 8 类版完全相同，只是 num_classes=12）─────────────
class LangIDClassifier(nn.Module):
    """Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear。"""

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1)  补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H)
        h = h.masked_fill(~mask, -1e9)
        pooled = h.max(dim=1)[0]                   # (B, H)
        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                     # (B, C)  logits


# ─── 5. 评测 ───────────────────────────────────────────────────────────
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
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion(preds, golds, labels):
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=5):
    """只打印错的格子，按错误数排序 —— 12 类版看 es↔pt、zh↔ja 最直观。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


def accuracy_by_length(texts, preds, golds):
    buckets = [(2, 7), (8, 12), (12, 20), (20, 30), (30, 999)]
    print("\n  按长度分档的准确率：")
    for lo, hi in buckets:
        idx = [i for i, t in enumerate(texts) if lo <= len(normalize(t)) < hi]
        if not idx:
            continue
        hit = sum(preds[i] == golds[i] for i in idx)
        label = f"{lo}~{hi if hi < 999 else '+'}"
        print(f"    {label:>8} 字符：{hit:4d}/{len(idx):4d} = {hit/len(idx):.3f}")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="12 类语种识别训练（自动找数据）")
    ap.add_argument("--data-dir", default=None,
                    help="数据目录（默认自动：data/langid/v12/ 或脚本同目录）")
    ap.add_argument("--vocab", default=None,
                    help="字符表文件（默认自动找 vocab_v12.txt，缺了会生成）")
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="强制从 train.csv 重新统计字符表并覆盖")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练（演示数据少了会过拟合）")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径，供 eval_langid.py 使用")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data_dir)
    vocab_path = find_vocab_path(args.vocab, data_dir)

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(data_dir, "train")
    test_rows = load_split(data_dir, "test")

    # dev.csv 可选：交作业的扁平结构里常常只带了 train / test
    if os.path.exists(os.path.join(data_dir, "dev.csv")):
        dev_rows = load_split(data_dir, "dev")
    else:
        train_rows, dev_rows = split_dev_from_train(train_rows)
        print(f"[提示] {data_dir} 下没有 dev.csv，已从训练集按标签分层切出 "
              f"{len(dev_rows)} 条当验证集（训练集剩 {len(train_rows)} 条）")

    # 数据里的标签必须都在 LABELS 里；反过来也提醒哪些类训练集里一条都没有
    seen = sorted({l for _, l in train_rows})
    extra = [l for l in seen if l not in LABELS]
    if extra:
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )
    missing = [l for l in LABELS if l not in seen]
    if missing:
        print(f"[警告] 这些类在训练集里没有样本：{' '.join(missing)}")
    label2id = {l: i for i, l in enumerate(LABELS)}
    num_classes = len(LABELS)

    if args.per_class:
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)

    vocab = ensure_vocab(data_dir, vocab_path, rebuild=args.rebuild_vocab)

    print(f"解释器：{sys.executable}")
    print(f"  Python {sys.version.split()[0]} / torch {torch.__version__}")
    print(f"数据目录：{data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  字符表 {len(vocab)} 个字符（{os.path.relpath(vocab_path, HERE)}）")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    # 字符表覆盖率：v12 里有多少字符能覆盖 95% 的字符出现次数
    freq = Counter()
    for t, _ in train_rows:
        freq.update(normalize(t))
    print(f"  字符表覆盖：前 {coverage(vocab, freq)} 个字符覆盖 95% 的字符出现次数")

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
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    # 测试集
    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)
    accuracy_by_length([t for t, _ in test_rows], preds, golds)

    # 逐类准确率
    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        if not idx:
            continue
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1, len(idx)):.3f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:6]]
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)
        print(f"\n模型已保存：{args.save}")


if __name__ == "__main__":
    main()
