# -*- coding: utf-8 -*-
"""第三周作业：使用字符级 RNN 完成 12 类语种识别。

姓名：从国晗
依赖：PyTorch，其余模块均为 Python 标准库。
数据：课程提供的 v12/train.csv、dev.csv、test.csv，字段为 text、label。

训练：python 作业01.py --data-dir data/langid/v12
预测：python 作业01.py --model outputs/langid_rnn_v12.pt --text "这是一段中文文本"

实现要点：只用训练集建立字符表，PAD=0、UNK=1；输出扩展为 12 类。
模型：Embedding -> RNN -> 掩码 MaxPool -> LayerNorm -> Dropout -> Linear。
训练：Adam + CrossEntropyLoss，按验证损失保存最佳权重，再评测测试集。
输出：字符表、最佳权重和包含逐轮记录的 summary.json，默认保存在 outputs/。
局限：训练文本归一化后长度为 8～40，短词和未知字符容易误判。
softmax 分数未经校准，不能作为预测正确的保证。

在课堂代码基础上改编，代码整理与参考训练验收有 AI 助手协助。
"""

import argparse
import copy
import csv
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]
MAX_LEN, EMBED_DIM, HIDDEN_DIM, DROPOUT = 40, 64, 64, 0.3


def normalize(text):
    """保留字母与组合标记，避免丢掉印地语、泰语等语言的元音标记。"""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "M"))


def load_split(data_dir, name):
    path = data_dir / f"{name}.csv"
    if not path.is_file():
        raise ValueError(f"找不到 {path}，请用 --data-dir 指定课程 v12 数据目录。")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{name}.csv 必须包含 text、label 两列。")
        rows = [(r["text"], r["label"]) for r in reader]
    if not rows or any(not normalize(text) for text, _ in rows):
        raise ValueError(f"{name}.csv 为空或含有归一化后为空的文本。")
    if {label for _, label in rows} != set(LABELS):
        raise ValueError(f"{name}.csv 的类别必须为：{' '.join(LABELS)}")
    return rows


def build_char_vocab(train_rows):
    freq = Counter(c for text, _ in train_rows for c in normalize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for char, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0])):
        vocab[char] = len(vocab)
    return vocab, freq


def encode(text, vocab, maxlen=MAX_LEN):
    ids = [vocab.get(c, 1) for c in normalize(text)][:maxlen]
    return ids + [0] * (maxlen - len(ids))


def make_loader(rows, vocab, batch_size=128, shuffle=False):
    label2id = {label: i for i, label in enumerate(LABELS)}
    features = torch.tensor([encode(text, vocab) for text, _ in rows], dtype=torch.long)
    targets = torch.tensor([label2id[label] for _, label in rows], dtype=torch.long)
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0)


class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=12, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):
        mask = (tokens != 0).unsqueeze(-1)
        hidden, _ = self.rnn(self.embedding(tokens))
        pooled = hidden.masked_fill(~mask, -1e9).max(dim=1).values
        pooled = torch.where(mask.any(dim=1), pooled, torch.zeros_like(pooled))
        # CrossEntropyLoss 直接接收 logits，预测时才使用 softmax。
        return self.fc(self.dropout(self.norm(pooled)))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_sum, predictions, golds = 0.0, [], []
    for features, targets in loader:
        logits = model(features)
        loss_sum += nn.functional.cross_entropy(logits, targets).item() * len(targets)
        predictions.extend(logits.argmax(dim=1).tolist())
        golds.extend(targets.tolist())
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for pred, gold in zip(predictions, golds):
        matrix[gold][pred] += 1
    f1s = []
    for c in range(len(LABELS)):
        denominator = sum(matrix[c]) + sum(row[c] for row in matrix)
        f1s.append(2 * matrix[c][c] / denominator if denominator else 0.0)
    correct = sum(pred == gold for pred, gold in zip(predictions, golds))
    return {"loss": loss_sum / len(golds), "accuracy": correct / len(golds),
            "macro_f1": sum(f1s) / len(f1s), "correct": correct,
            "total": len(golds), "confusion_matrix": matrix}


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    splits = {name: load_split(args.data_dir, name) for name in ("train", "dev", "test")}
    # 按归一化文本检查重复，防止同一句出现在不同划分中。
    sets = {name: {normalize(text) for text, _ in rows} for name, rows in splits.items()}
    if any(len(sets[name]) != len(rows) for name, rows in splits.items()):
        raise ValueError("同一划分内存在归一化重复文本。")
    if any(sets[a] & sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("训练、验证和测试划分存在交叉文本。")
    vocab, freq = build_char_vocab(splits["train"])
    train_loader = make_loader(splits["train"], vocab, batch_size=64, shuffle=True)
    dev_loader = make_loader(splits["dev"], vocab)
    model = LangIDClassifier(len(vocab), len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    print(f"训练/验证/测试：{len(splits['train'])}/{len(splits['dev'])}/{len(splits['test'])}")
    print(f"类别 {len(LABELS)}，字符表 {len(vocab)}，按验证损失选择模型。")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("训练损失不是有限数，已停止。")
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        dev = evaluate(model, dev_loader)
        history.append({"epoch": epoch, "train_loss": train_loss / len(splits["train"]),
                        "dev_loss": dev["loss"], "dev_accuracy": dev["accuracy"]})
        if dev["loss"] < best_loss:
            best_loss, best_epoch, stale = dev["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(f"Epoch {epoch:02d}/{args.epochs}  train_loss={history[-1]['train_loss']:.4f}  "
              f"dev_loss={dev['loss']:.4f}  dev_acc={dev['accuracy']:.2%}", flush=True)
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    best_dev = evaluate(model, dev_loader)
    # 轮次确定后再评测测试集，测试结果不参与模型选择。
    test = evaluate(model, make_loader(splits["test"], vocab))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "langid_rnn_v12.pt"
    torch.save({"state_dict": best_state, "vocab": vocab, "labels": LABELS,
                "maxlen": MAX_LEN, "embed_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM,
                "dropout": DROPOUT, "seed": args.seed, "best_epoch": best_epoch}, model_path)
    with (args.output_dir / "vocab_v12.txt").open("w", encoding="utf-8") as stream:
        stream.write("# 字符\tid\t出现次数\n")
        for char, idx in vocab.items():
            stream.write(f"{char}\t{idx}\t{freq.get(char, 0)}\n")
    summary = {"labels": LABELS, "vocab_size": len(vocab), "best_epoch": best_epoch,
               "seed": args.seed, "torch_version": str(torch.__version__),
               "best_dev": best_dev, "test": test, "history": history}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n最佳第 {best_epoch} 轮，测试 {test['correct']}/{test['total']}，"
          f"accuracy={test['accuracy']:.2%}，macro-F1={test['macro_f1']:.4f}")
    for c, label in enumerate(LABELS):
        row = test["confusion_matrix"][c]
        print(f"  {label}: {row[c]}/{sum(row)} = {row[c]/sum(row):.2%}")
    print(f"模型已保存：{model_path}")


@torch.inference_mode()
def predict(model_path, text):
    normalized = normalize(text)
    if not normalized:
        raise ValueError("输入必须包含字母、汉字或其他有效文字。")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    vocab, labels = checkpoint["vocab"], checkpoint["labels"]
    model = LangIDClassifier(len(vocab), len(labels), checkpoint["embed_dim"],
                             checkpoint["hidden_dim"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokens = torch.tensor([encode(text, vocab, checkpoint["maxlen"])], dtype=torch.long)
    values, indices = model(tokens).softmax(dim=1)[0].topk(min(3, len(labels)))
    print(" / ".join(f"{labels[idx]} {float(value):.2%}"
                     for value, idx in zip(values, indices.tolist())))
    unknown = sum(c not in vocab for c in normalized[:checkpoint["maxlen"]])
    if len(normalized) < 8 or unknown:
        print(f"提示：有效字符 {len(normalized)} 个，输入窗口内未知字符 {unknown} 个；"
              "短词或未知字符可能影响判断，以上分数仅供参考。")
    if len(normalized) > checkpoint["maxlen"]:
        print(f"仅使用归一化后的前 {checkpoint['maxlen']} 个字符。")


def main():
    parser = argparse.ArgumentParser(description="12 类语种分类：训练或加载模型预测。")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data" / "langid" / "v12")
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--model", type=Path, default=HERE / "outputs" / "langid_rnn_v12.pt")
    parser.add_argument("--text", help="提供文本时只加载模型预测，不重新训练。")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs 和 patience 必须为正整数。")
    torch.set_num_threads(1)
    try:
        if args.text is not None:
            predict(args.model, args.text)
        else:
            train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
