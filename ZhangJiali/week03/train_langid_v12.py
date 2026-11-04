# -*- coding: utf-8 -*-
"""
train_langid_v12.py —— 多语种「语种识别」分类器（12 语种版）

任务：一句话 → 它是什么语言（12 分类）
模型：Embedding → RNN → 掩码 MaxPool → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss      优化器：Adam
数据：data/langid/v12（ar en es hi ja ko pt ru th tr vi zh）

相对8类脚本的不同：

  1. LABELS 扩展为 12 个语种（新增 ko / th / vi / tr）。
     分类头输出维度 = len(LABELS)，训练与评测共用同一份列表。

  2. 字符表从 v12/train.csv 现算，写出 data/langid/vocab/vocab_v12.txt，
     不再读取 vocab_v8.txt。
     韩文、泰文等字符不在 8 类词表里；若沿用旧表，encode() 会把它们
     全部映射为 <UNK>，对应语种等于没有有效输入。

模型结构复用 train_langid_cls.py 中的 LangIDClassifier。

"""

import os
import time
import random
import argparse
import unicodedata
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_langid_cls import (
    LangIDClassifier,
    TextDataset,
    build_char_vocab,
    confusion,
    encode,
    load_split,
    load_vocab_file,
    macro_f1,
    normalize,
    predict,
    print_confusion,
    BATCH_SIZE,
    DROPOUT,
    EMBED_DIM,
    HIDDEN_DIM,
    LR,
    MAX_LEN,
    NUM_THREADS,
    SEED,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "langid", "v12")
VOCAB_V8_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v8.txt")
VOCAB_OUT = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")

# 12 个语种标签（8 类基础上增加 ko / th / vi / tr），按字母序排列。
# 下标即分类头对应输出维；保存 checkpoint 时一并写入，供评测脚本读取。
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]

# 数据卡给出的参考规模，用于核对字符表是否建对。
VOCAB_SIZE_HINT = 3276
COVERAGE95_HINT = 803


def char_freq(sentences):
    """统计归一化后每个字符的出现次数（归一化方式与 build_char_vocab 一致）。"""
    freq = Counter()
    for text in sentences:
        freq.update(normalize(text))
    return freq


def save_vocab(vocab, freq, path):
    """
    写出字符表，格式与 vocab_v8.txt 相同：
    每行「字符 \\t id \\t 出现次数」；<PAD> / <UNK> 的次数记为 0。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")


def coverage_at(vocab, freq, ratio=0.95):
    """按词表顺序（高频优先）累加，返回覆盖 ratio 出现次数所需的字符数。"""
    total = sum(freq.values())
    acc = 0
    n = 0
    for ch in vocab:
        if ch in ("<PAD>", "<UNK>"):
            continue
        n += 1
        acc += freq.get(ch, 0)
        if total and acc >= ratio * total:
            return n
    return n


def script_name(ch):
    """按 Unicode 区间粗分书写系统，便于观察 12 类词表相对 8 类的增量来源。"""
    o = ord(ch)
    # 含基本汉字区、扩展区，以及日文迭代记号「々」
    if (0x3400 <= o <= 0x9FFF or 0x20000 <= o <= 0x3FFFF or ch == "々"):
        return "汉字"
    if 0x3040 <= o <= 0x30FF:
        return "假名"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
        return "韩文"
    if 0x0E00 <= o <= 0x0E7F:
        return "泰文"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
        return "阿拉伯文"
    if 0x0400 <= o <= 0x04FF:
        return "西里尔字母"
    if 0x0900 <= o <= 0x097F:
        return "天城文"
    name = unicodedata.name(ch, "")
    if "LATIN" in name:
        return "拉丁字母"
    # 例如土耳其语 İ 小写后出现的组合点（Mn）
    if unicodedata.category(ch).startswith("M"):
        return "组合符号"
    return "其他"


def report_vocab(vocab, freq, train_rows):
    """打印字符表规模与对照信息，仅用于核对，不参与梯度更新。"""
    n95 = coverage_at(vocab, freq)
    print(f"  字符表 {len(vocab)} 个（含 <PAD>/<UNK>），前 {n95} 个覆盖 95% 的出现次数")
    print(f"  数据卡参考值：{VOCAB_SIZE_HINT} 个字符，95% 覆盖约 {COVERAGE95_HINT} 个")
    if len(vocab) != VOCAB_SIZE_HINT:
        print("  （与参考值不一致时，请确认是否基于训练集、且经过 normalize）")

    buckets = Counter(script_name(ch) for ch in vocab if ch not in ("<PAD>", "<UNK>"))
    print("  字符表按书写系统：")
    for name, cnt in buckets.most_common():
        print(f"    {name:<8} {cnt:5d}")

    if not os.path.exists(VOCAB_V8_PATH):
        return
    old = load_vocab_file(VOCAB_V8_PATH)
    new_chars = [ch for ch in vocab if ch not in old and ch not in ("<PAD>", "<UNK>")]
    print(f"  相对 vocab_v8.txt 新出现 {len(new_chars)} 个字 "
          f"（8 类表 {len(old)} → 12 类表 {len(vocab)}）")
    grown = Counter(script_name(ch) for ch in new_chars)
    if grown:
        detail = "，".join(f"{k} {v}" for k, v in grown.most_common())
        print(f"    增量主要来自：{detail}")
        print("    说明：韩文/泰文来自新增语种；汉字数量差异来自 v8/v12 采样句子不同")

    # 对照实验：若错误沿用 8 类词表，新语种会大量变成 <UNK>
    total_unk = Counter()
    total_chars = Counter()
    for text, label in train_rows:
        chars = normalize(text)
        total_chars[label] += len(chars)
        total_unk[label] += sum(ch not in old for ch in chars)
    print("  若误用 vocab_v8.txt，各语种字符落到 <UNK> 的比例：")
    for lab in LABELS:
        denom = total_chars[lab]
        ratio = total_unk[lab] / denom if denom else 0.0
        mark = "  ← 输入几乎失效" if ratio > 0.5 else ""
        print(f"    {lab}: {ratio:6.1%}{mark}")


def show_one_batch(model, loader):
    """
    训练前打印一个 batch 的张量形状，核对数据流是否正确。
    关注点：Linear 输出最后一维应等于类别数 12。
    """
    model.eval()
    X, y = next(iter(loader))
    with torch.no_grad():
        emb = model.embedding(X)
        h, _ = model.rnn(emb)
        logits = model(X)
    print("一次前向的张量形状：")
    print(f"  输入 X          {tuple(X.shape)}   batch × 截断长度 {MAX_LEN}（字符 id）")
    print(f"  Embedding       {tuple(emb.shape)}   每个 id → {EMBED_DIM} 维向量")
    print(f"  RNN 隐状态      {tuple(h.shape)}   每个位置 → {HIDDEN_DIM} 维")
    print(f"  掩码 MaxPool 后 ({X.size(0)}, {HIDDEN_DIM})   序列汇聚为句向量")
    print(f"  Linear 输出     {tuple(logits.shape)}   最后一维 = 类别数 {len(LABELS)}（logits）")
    print(f"  标签 y          {tuple(y.shape)}   类别下标，示例标签 = {LABELS[int(y[0])]}")
    print("  CrossEntropyLoss 直接接收 logits 与 y（内部含 log_softmax）。")
    print()


def main():
    ap = argparse.ArgumentParser(description="12 类语种识别：从 v12 训练集构建字符表并训练")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--vocab-out", default=VOCAB_OUT, help="12 类字符表输出路径")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=0,
                    help="每类仅使用前 N 条训练样本（用于观察数据量影响）")
    ap.add_argument("--threads", type=int, default=NUM_THREADS)
    ap.add_argument("--save", default="", help="模型保存路径，可供 eval_langid.py 加载")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_rows = load_split(args.data_dir, "train")
    dev_rows = load_split(args.data_dir, "dev")
    test_rows = load_split(args.data_dir, "test")

    # 保证 LABELS 与数据标签集合一致，避免输出维与真实类别错位
    seen = sorted({label for _, label in train_rows + dev_rows + test_rows})
    if seen != sorted(LABELS):
        raise SystemExit(
            "LABELS 与 csv 中的标签不一致。\n"
            f"  脚本 LABELS：{' '.join(LABELS)}\n"
            f"  数据标签：{' '.join(seen)}\n"
            "  二者必须完全相同（输出维 = len(LABELS)）。"
        )
    label2id = {label: i for i, label in enumerate(LABELS)}

    if args.per_class:
        by_label = {}
        for text, label in train_rows:
            by_label.setdefault(label, []).append(text)
        rng = random.Random(SEED)
        train_rows = [
            (text, label)
            for label, texts in by_label.items()
            for text in rng.sample(texts, min(args.per_class, len(texts)))
        ]
        rng.shuffle(train_rows)

    # 从当前训练集构建字符表：0=<PAD>，1=<UNK>，其余按频次降序编号
    freq = char_freq(text for text, _ in train_rows)
    vocab = build_char_vocab(text for text, _ in train_rows)
    save_vocab(vocab, freq, args.vocab_out)

    print("12 类语种识别")
    print(f"数据目录：{args.data_dir}")
    print(f"  训练 {len(train_rows)} 条 / 验证 {len(dev_rows)} 条 / 测试 {len(test_rows)} 条")
    print(f"  类别 {len(LABELS)} 个：{' '.join(LABELS)}")
    print(f"  随机猜测准确率基准：{1 / len(LABELS):.1%}")
    report_vocab(vocab, freq, train_rows)
    print(f"  字符表已写入：{args.vocab_out}")

    train_loader = DataLoader(
        TextDataset(train_rows, vocab, label2id),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    dev_loader = DataLoader(TextDataset(dev_rows, vocab, label2id), batch_size=128)
    test_loader = DataLoader(TextDataset(test_rows, vocab, label2id), batch_size=128)

    model = LangIDClassifier(len(vocab), len(LABELS), dropout=DROPOUT)
    n_embed = model.embedding.weight.numel()
    n_head = model.fc.weight.numel() + model.fc.bias.numel()
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量：{n_all:,}")
    print(f"  Embedding {n_embed:,}（词表扩大后参数主要增加在这里）")
    print(f"  分类头   {n_head:,}（8→12 类仅多 4 个输出）")
    print()
    show_one_batch(model, train_loader)

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
        print(f"Epoch {epoch:2d}/{args.epochs}  loss={total / len(train_loader):.4f}  "
              f"dev_acc={acc:.4f}")
    elapsed = time.time() - t0
    print(f"\n训练耗时：{elapsed:.0f} 秒（{args.epochs} 个 epoch，{args.threads} 线程）")

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    f1 = macro_f1(preds, golds, len(LABELS))
    print(f"\n测试集：accuracy = {acc:.4f}   macro-F1 = {f1:.4f}")
    print("  预期较易混淆的组合：")
    print("    es ↔ pt：同属拉丁字母体系，主要依赖字母组合差异")
    print("    en / vi / tr：同为拉丁字母，重音与特有字母是重要线索")
    print("    zh ↔ ja：汉字重叠较多，假名是区分日语的关键特征")
    print_confusion(confusion(preds, golds, LABELS), LABELS, top=5)

    print("\n  各语种准确率：")
    for c, lab in enumerate(LABELS):
        idx = [i for i, g in enumerate(golds) if g == c]
        hit = sum(preds[i] == c for i in idx)
        print(f"    {lab}: {hit}/{len(idx)} = {hit / max(1, len(idx)):.3f}")

    print("\n--- 推理示例（测试集前 4 条）---")
    model.eval()
    with torch.no_grad():
        for text, _gold in test_rows[:4]:
            logits = model(torch.tensor([encode(text, vocab)]))
            prob = torch.softmax(logits, dim=1)[0]
            pid = int(prob.argmax())
            print(f"  [{LABELS[pid]} {prob[pid]:.2f}]  {text[:50]}")

    if args.save:
        torch.save({
            "state_dict": model.state_dict(),
            "vocab": dict(vocab),
            "labels": LABELS,
            "maxlen": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
        }, args.save)
        print(f"\n模型已保存：{args.save}")
        print("评测命令：")
        print(f"  python eval_langid.py --model {args.save} --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
