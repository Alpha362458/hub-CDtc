# -*- coding: utf-8 -*-
"""
build_vocab_v12.py —— 为 12 语种数据集(v12)生成字符表

为什么要它:
  train_langid_cls.py 是「读现成字符表」的(load_vocab_file)，课堂只配了 vocab_v8.txt。
  换成 12 语种后，新语种(ko 韩/th 泰/tr 土/vi 越)的字符在 v8 表里几乎全缺，
  不重建表的话它们会全部落到 <UNK>，模型对这些语种等于睁眼瞎。

它做什么:
  把 v12 训练集每句话按字拆开 → 统计每个字的出现次数 → 按次数从高到低编号，
  写成和 vocab_v8.txt 一模一样的格式(每行「字符 \t id \t 出现次数」)。

用法:
  python build_vocab_v12.py
  生成 data/langid/vocab/vocab_v12.txt
"""

import os
import csv
import unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "data", "langid", "v12", "train.csv")
OUT_PATH = os.path.join(HERE, "data", "langid", "vocab", "vocab_v12.txt")


# ⚠️ 必须和 train_langid_cls.py 里的 normalize 完全一致，否则建表/训练对不上
def normalize(text):
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def main():
    # 1. 读训练集，统计每个字符的出现次数
    freq = Counter()
    n_rows = 0
    with open(TRAIN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            freq.update(normalize(r["text"]))
            n_rows += 1

    # 2. 按「次数从高到低、次数相同按字符编码」排序(保证可复现，和 build_char_vocab 一致)
    chars = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))

    # 3. 写文件：第 0 号 <PAD>、第 1 号 <UNK>，其余按次序编号
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("# 字符\tid\t出现次数\n")
        f.write("<PAD>\t0\t0\n")
        f.write("<UNK>\t1\t0\n")
        idx = 2
        for ch, n in chars:
            f.write(f"{ch}\t{idx}\t{n}\n")
            idx += 1

    print(f"读入训练集 {n_rows} 条")
    print(f"字符表大小 {idx} 个（含 <PAD> <UNK>）")
    print(f"已生成：{OUT_PATH}")


if __name__ == "__main__":
    main()
