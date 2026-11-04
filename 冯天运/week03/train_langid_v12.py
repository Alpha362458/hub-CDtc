# -*- coding: utf-8 -*-
"""

任务：一句话 → 它属于 12 种语言中的哪一种
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss        优化器：Adam
数据：data/v12（在 v8 的 8 语种基础上增加 kor / tha / vie / tur 四个语种）

"""

import os
import csv
import time
import random
import argparse
import unicodedata
from collections import Counter, OrderedDict

import sys
# Windows 控制台默认 GBK，遇到土耳其语 ı 等字符会抛 UnicodeEncodeError。
# 强制 stdout/stderr 用 UTF-8，避免打印多语样本时崩溃。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────────────────
SEED = 42                              # 随机种子：固定后结果可复现
MAX_LEN = 40                          # 句子统一截/补到 40 字符
EMBED_DIM = 64                        # 字符嵌入向量维度
HIDDEN_DIM = 64                       # RNN 隐状态维度（也是句子向量维度）
DROPOUT = 0.3                         # Dropout 比例：训练时随机置零 30% 神经元
BATCH_SIZE = 64                       # 每个训练批次的样本数
LR = 1e-3                             # 学习率
EPOCHS = 8                            # 训练轮数
NUM_THREADS = 1                       # 单线程（小模型单线程比多线程快）

HERE = os.path.dirname(os.path.abspath(__file__))           # 本脚本所在目录
DATA_DIR = os.path.join(HERE, "data", "v12")                # 12 语种数据目录
VOCAB_PATH = os.path.join(HERE, "vocab_v12.txt")            # 12 类字符表路径

# 12 个语种：Tatoeba 代码 → 两字标签。这里的顺序就是模型输出的下标 0..11
LABELS = ["zh", "ja", "en", "es", "pt", "ru", "ar",
          "hi", "ko", "th", "vi", "tr"]


# ─── 1. 文本归一化（必须和建数据时一致）────────────────────────────────
def normalize(text):
    """NFKC 归一化 + 转小写 + 只保留字母(L)与组合符号(M)。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


# ─── 2. 字符表构建 ─────────────────────────────────────────────────────
def build_char_vocab(sentences, min_freq=1):
    """训练集所有出现过的字符，每个字符发一个整数编号。"""
    freq = Counter()                                   # 统计每个字符出现次数
    for s in sentences:
        freq.update(normalize(s))                      # 归一化后按字符累加
    # 过滤低频字符，按出现次数降序、次数相同则按字符编码升序，保证可复现
    chars = sorted(((c, n) for c, n in freq.items() if n >= min_freq),
                   key=lambda kv: (-kv[1], kv[0]))
    vocab = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])  # 0 号补齐，1 号未登录
    for ch, _ in chars:                                # 其余字符从 2 号起顺序编号
        vocab[ch] = len(vocab)
    return vocab


def load_vocab_file(path):
    """读已生成的字符表：每行「字符 \\t id \\t 出现次数」，# 开头是注释。"""
    vocab = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue                               # 跳过注释行和空行
            ch, idx, _freq = line.rstrip("\n").split("\t")
            vocab[ch] = int(idx)                       # 字符 -> 整数 id
    return vocab


def save_vocab_file(vocab, path):
    """把字符表写回文件，下次直接读，不必每次重新统计。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t0\n")


def encode(text, vocab, maxlen=MAX_LEN):
    """文本 → 定长字符 id 序列：归一化、查表、截断、补齐。"""
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]  # 未登录字符用 <UNK>=1
    ids += [0] * (maxlen - len(ids))                  # 不足 maxlen 的尾部用 <PAD>=0 补齐
    return ids


# ─── 3. 读数据 ─────────────────────────────────────────────────────────
def load_split(data_dir, name):
    path = os.path.join(data_dir, f"{name}.csv")      # 拼出 train/dev/test 的 csv 路径
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):                   # 按表头读成字典
            rows.append((r["text"], r["label"]))      # 取 text 与 label 两列
    return rows


class TextDataset(Dataset):
    """把 (文本, 标签) 列表封装成 PyTorch 的 Dataset。"""
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]  # 每条文本编码成定长 id 序列
        self.y = [label2id[l] for _, l in rows]       # 标签字符串转成类别 id

    def __len__(self):
        return len(self.y)                            # 样本总数

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]  # 第 i 条：(id序列, 类别id)


# ─── 4. 模型 ───────────────────────────────────────────────────────────
class LangIDClassifier(nn.Module):
    """
    Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
    结构与 v8 完全相同，仅输出维度随 LABELS 变成 12。
    """

    def __init__(self, vocab_size, num_classes,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, L) 每句话的字符 id 序列，补齐位 = 0
        mask = (x != 0).unsqueeze(-1)              # (B, L, 1) 真字符为 True，补齐位为 False
        h, _ = self.rnn(self.embedding(x))         # (B, L, H) 逐字符读，输出每个位置隐状态
        h = h.masked_fill(~mask, -1e9)             # 补齐位填 -1e9，最大池化时不参与
        pooled = h.max(dim=1)[0]                   # (B, H) 句子向量：每个特征通道取全程最大值
        pooled = self.dropout(self.norm(pooled))   # (B, H) 归一化 + Dropout
        return self.fc(pooled)                     # (B, C) 输出 12 类的 logits（未过 softmax）


# ─── 5. 评测 ───────────────────────────────────────────────────────────
def predict(model, loader):
    """在 loader 上推理，返回 (预测类别列表, 真实类别列表)。"""
    model.eval()                                    # 评估模式：关闭 Dropout
    preds, golds = [], []
    with torch.no_grad():                           # 不计算梯度，省内存、提速
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()  # 最大 logit 下标为预测类
            golds += y.tolist()
    return preds, golds


def macro_f1(preds, golds, num_classes):
    """逐类算 F1 再取平均，少数类不会被大类淹没。"""
    f1s = []
    for c in range(num_classes):                    # 逐类
        tp = sum(p == c and g == c for p, g in zip(preds, golds))   # 真正例
        fp = sum(p == c and g != c for p, g in zip(preds, golds))   # 假正例
        fn = sum(p != c and g == c for p, g in zip(preds, golds))   # 假负例
        if tp + fp + fn == 0:                       # 该类无样本则跳过
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0   # 精确率
        rec = tp / (tp + fn) if tp + fn else 0.0    # 召回率
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)  # F1
    return sum(f1s) / len(f1s)                      # 各类 F1 算术平均


def confusion(preds, golds, labels):
    """构造混淆矩阵：mat[真实][预测] 计数。"""
    n = len(labels)
    mat = [[0] * n for _ in range(n)]
    for p, g in zip(preds, golds):
        mat[g][p] += 1
    return mat


def print_confusion(mat, labels, top=5):
    """只打印错判中最多的 top 个组合。"""
    pairs = [(mat[g][p], labels[g], labels[p])
             for g in range(len(labels)) for p in range(len(labels)) if g != p]
    pairs.sort(reverse=True)
    print("  最常混淆的组合（真实 → 预测）：")
    for cnt, g, p in pairs[:top]:
        if cnt == 0:
            break
        print(f"    {g} → {p}: {cnt} 条")


# ─── 6. 训练 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab", default=VOCAB_PATH)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--per-class", type=int, default=0,
                    help="只用每类前 N 条训练，用来观察数据变少时的过拟合")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="保存模型到该路径")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)             # 设置 PyTorch 线程数
    random.seed(SEED)                               # 固定 Python 随机种子
    torch.manual_seed(SEED)                         # 固定 PyTorch 随机种子

    train_rows = load_split(args.data_dir, "train")  # 读训练集
    dev_rows = load_split(args.data_dir, "dev")      # 读验证集
    test_rows = load_split(args.data_dir, "test")    # 读测试集

    seen = sorted({l for _, l in train_rows})        # 训练集出现过的标签
    extra = [l for l in seen if l not in LABELS]     # 找出不在 LABELS 里的标签
    if extra:                                        # 存在未知标签则报错退出
        raise SystemExit(
            f"数据里出现了 LABELS 之外的标签：{' '.join(extra)}\n"
            f"  当前 LABELS（{len(LABELS)} 类）：{' '.join(LABELS)}"
        )
    label2id = {l: i for i, l in enumerate(LABELS)}  # 标签 -> 类别下标
    num_classes = len(LABELS)                        # 类别数 = 12

    # 字符表：优先读已生成的 vocab_v12.txt；没有则从训练集现算并落盘
    if os.path.exists(args.vocab):
        vocab = load_vocab_file(args.vocab)
        print(f"  字符表：读 {args.vocab}（{len(vocab)} 个字符）")
    else:
        train_texts = [t for t, _ in train_rows]
        vocab = build_char_vocab(train_texts)
        save_vocab_file(vocab, args.vocab)
        print(f"  字符表：从训练集现算并写入 {args.vocab}（{len(vocab)} 个字符）")

    if args.per_class:                               # 若指定每类只用前 N 条
        by_label = {}
        for t, l in train_rows:
            by_label.setdefault(l, []).append(t)     # 同标签归到一起
        train_rows = [(t, l) for l, ts in by_label.items()
                      for t in random.Random(SEED).sample(ts, min(args.per_class, len(ts)))]
        random.Random(SEED).shuffle(train_rows)      # 打乱（固定种子可复现）

    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {num_classes} 个：{' '.join(LABELS)}")
    print(f"  随机猜的准确率基准：{1/num_classes:.1%}")   # 随机基准 = 1/12

    train_loader = DataLoader(TextDataset(train_rows, vocab, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)    # 训练加载器：打乱
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)   # 验证加载器
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128) # 测试加载器

    model = LangIDClassifier(len(vocab), num_classes)  # 实例化模型
    print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()                # 交叉熵损失（内含 Softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam 优化器

    t0 = time.time()                                 # 记录开始时刻
    for epoch in range(1, args.epochs + 1):          # 逐轮训练
        model.train()                                # 训练模式：启用 Dropout
        total = 0.0                                  # 本轮 loss 累加器
        for X, y in train_loader:                    # 逐批次
            logits = model(X)                        # 前向传播
            loss = criterion(logits, y)              # 计算损失
            optimizer.zero_grad()                    # 清空梯度
            loss.backward()                          # 反向传播
            optimizer.step()                         # 更新参数
            total += loss.item()                     # 累加 loss
        preds, golds = predict(model, dev_loader)    # 验证集预测
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total/len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    preds, golds = predict(model, test_loader)       # 测试集预测
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, num_classes)
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print_confusion(confusion(preds, golds, LABELS), LABELS)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):                 # 逐类
        idx = [i for i, g in enumerate(golds) if g == c]   # 该类样本下标
        hit = sum(preds[i] == c for i in idx)        # 预测对的个数
        print(f"    {lab}: {hit}/{len(idx)} = {hit/max(1,len(idx)):.3f}")

    if args.save:                                    # 若指定 --save
        torch.save({"state_dict": model.state_dict(), "vocab": dict(vocab),
                    "labels": LABELS, "maxlen": MAX_LEN,
                    "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                    "dropout": DROPOUT}, args.save)  # 保存权重 + 字符表 + 标签 + 超参
        print(f"\n模型已保存：{args.save}")

    print("\n--- 推理示例 ---")
    model.eval()
    demos = [t for t, _ in test_rows[:6]]            # 取测试集前 6 句做演示
    with torch.no_grad():
        for text in demos:
            logits = model(torch.tensor([encode(text, vocab)]))  # 编码并前向
            prob = torch.softmax(logits, dim=1)[0]   # 转成概率分布
            pid = int(prob.argmax())                 # 最大概率对应类别
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")


if __name__ == "__main__":
    main()
