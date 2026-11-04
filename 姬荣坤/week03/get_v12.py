import os
import csv
import unicodedata
from collections import Counter, OrderedDict

# 保持与原代码一致的路径配置
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "langid")

def normalize(text):
    """NFKC + 小写 + 只保留字母(L)与组合符号(M)。"""
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))

def build_char_vocab(sentences, min_freq=1):
    """统计词频并按规则生成词表"""
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
    """保存词表文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{freq.get(ch, 0)}\n")

def main():
    # 1. 读取已有的 v12 train.csv
    train_csv_path = os.path.join(OUT_ROOT, "v12", "train.csv")
    if not os.path.exists(train_csv_path):
        print(f"找不到文件: {train_csv_path}，请确认路径是否正确。")
        return

    print(f"正在读取: {train_csv_path} ...")
    sentences = []
    with open(train_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头 ["text", "label"]
        for row in reader:
            if row and len(row) >= 1:
                sentences.append(row[0])  # 第一列是 text

    print(f"共读取到 {len(sentences)} 条训练样本。")

    # 2. 构建词表
    vocab, freq = build_char_vocab(sentences)

    # 3. 保存词表
    vocab_save_path = os.path.join(OUT_ROOT, "vocab", "vocab_v12.txt")
    save_vocab(vocab, freq, vocab_save_path)

    print(f"✅ 成功生成词表！包含 {len(vocab)} 个字符（含 <PAD>/<UNK>）。")
    print(f"词表已保存至: {vocab_save_path}")

if __name__ == "__main__":
    main()