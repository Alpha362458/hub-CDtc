# -*- coding: utf-8 -*-
"""
demo_langid_v12.py —— 12 语种模型推理演示

加载 train_langid_v12.py 训练保存的权重，做两件事：
  1) 手写示例：12 个语种各一句，打印概率分布，观察相似语种（es/pt、zh/ja、ko/ja）的区分；
  2) 完整测试集评测：读取 data/v12/test.csv（1800 条，12 类各 150），
     打印每类条数、每类准确率、整体 accuracy 与 macro-F1，确认测试集覆盖全部 12 个语言。

用法：
  python demo_langid_v12.py --model langid_rnn_v12.pt
"""

import os
import argparse

import sys
# Windows 控制台默认 GBK，遇到土耳其语 ı 等字符会抛 UnicodeEncodeError。
# 强制 stdout/stderr 用 UTF-8，避免打印多语样本时崩溃。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from torch.utils.data import DataLoader
from train_langid_v12 import (LangIDClassifier, encode, normalize,
                              MAX_LEN, EMBED_DIM, HIDDEN_DIM, DROPOUT,
                              load_split, TextDataset, predict, macro_f1)

HERE = os.path.dirname(os.path.abspath(__file__))   # 本脚本所在目录


def load_model(path):
    """读 checkpoint，重建模型结构并载入权重。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]                             # 训练时保存的字符表
    labels = ckpt["labels"]                           # 12 个类别标签
    model = LangIDClassifier(len(vocab), len(labels),
                             embed_dim=ckpt.get("embed_dim", EMBED_DIM),
                             hidden_dim=ckpt.get("hidden_dim", HIDDEN_DIM),
                             dropout=ckpt.get("dropout", DROPOUT))
    model.load_state_dict(ckpt["state_dict"])         # 载入权重
    model.eval()                                      # 切到评估模式
    return model, vocab, labels


# 12 个语种的示例句子；刻意包含易混淆对：es/pt（同套拉丁字母）、zh/ja、ko/ja
SAMPLES = {
    "zh": "今天天气真好，我们去公园散步吧。",
    "ja": "今日は良い天気ですね、公園に散歩に行きましょう。",
    "en": "The weather is nice today, let's go for a walk.",
    "es": "Hoy hace buen tiempo, vamos a pasear por el parque.",
    "pt": "O tempo está bom hoje, vamos passear no parque.",
    "ru": "Сегодня хорошая погода, пойдём гулять в парк.",
    "ar": "الطقس جميل اليوم، لنخرج لنتمشى في الحديقة.",
    "hi": "आज मौसम अच्छा है, चलो पार्क में टहलने चलते हैं।",
    "ko": "오늘 날씨가 좋으니 공원에 산책하러 갑시다.",
    "th": "วันนี้อากาศดี ไปเดินเล่นในสวนกันเถอะ",
    "vi": "Hôm nay trời đẹp, chúng ta đi dạo công viên nhé.",
    "tr": "Bugün hava güzel, parkta yürüyüşe gidelim.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "langid_rnn_v12.pt"))
    ap.add_argument("--test-csv", default=os.path.join(HERE, "data", "v12", "test.csv"),
                    help="完整测试集路径（默认 data/v12/test.csv，1800 条，12 类各 150）")
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, vocab, labels = load_model(args.model)

    print(f"模型：{args.model}")
    print(f"支持 {len(labels)} 个语种：{' '.join(labels)}\n")

    for true_lab, text in SAMPLES.items():
        ids = torch.tensor([encode(text, vocab)])     # 编码成 (1, MAX_LEN) 的张量
        with torch.no_grad():
            prob = torch.softmax(model(ids), dim=1)[0]  # 该句在 12 类上的概率分布
        topk = prob.topk(3)                            # 取概率最高的 3 个语种
        preds = [(labels[i], float(topk.values[k])) for k, i in enumerate(topk.indices)]
        guess = preds[0][0]                            # 最可能的语种
        mark = "OK" if guess == true_lab else "错"     # 预测是否正确
        detail = "  ".join(f"{lab}:{p:.2f}" for lab, p in preds)
        print(f"[{mark} 真={true_lab}] {detail}")
        print(f"     原文: {text[:60]}")

    # --- 完整测试集评测：用 data/v12/test.csv 的 1800 条（12 类各 150）---
    print(f"\n=== 完整测试集评测（{args.test_csv}）===")
    rows = load_split(os.path.dirname(args.test_csv), "test")   # 读 test.csv
    loader = DataLoader(TextDataset(rows, vocab,
                                    {l: i for i, l in enumerate(labels)}),
                        batch_size=128)                         # 批加载器
    preds, golds = predict(model, loader)                      # 全量预测
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)   # 整体准确率
    f1 = macro_f1(preds, golds, len(labels))                   # macro-F1（各类平均）
    # 逐类统计：per[语种] = [命中数, 总数]
    per = {l: [0, 0] for l in labels}
    for p, g in zip(preds, golds):
        per[labels[g]][1] += 1
        if p == g:
            per[labels[g]][0] += 1
    print(f"测试集总条数：{len(golds)}（{len(labels)} 类 × {len(golds) // len(labels)}）")
    print(f"整体 accuracy = {acc:.4f}   macro-F1 = {f1:.4f}\n")
    for l in labels:
        hit, tot = per[l]
        print(f"  {l}: {hit}/{tot} = {hit / tot:.3f}")


if __name__ == "__main__":
    main()
