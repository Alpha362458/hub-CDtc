import argparse
import random

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

MODEL_NAME = "FacebookAI/xlm-roberta-base"
NUM_CLASSES = 12
MAX_LENGTH = 128


# 1. 选择训练设备
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# 2. 自定义数据集
class LanguageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        encoded["labels"] = self.labels[index]
        return encoded


# 3. 验证模型
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.inference_mode():
        for batch in dataloader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)

            predictions.extend(preds.cpu().tolist())
            targets.extend(batch["labels"].cpu().tolist())

    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    return accuracy, macro_f1


# 4. 训练入口
def train(args):
    random.seed(42)
    torch.manual_seed(42)

    device = get_device()
    print(f"训练设备: {device}")

    # 读取训练数据
    df = pd.read_csv(args.data)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV 必须包含 text 和 label 两列")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    languages = sorted(df["label"].unique())

    if len(languages) != NUM_CLASSES:
        raise ValueError(
            f"需要 {NUM_CLASSES} 种语言，"
            f"实际发现 {len(languages)} 种: {languages}"
        )

    if df["label"].value_counts().min() < 2:
        raise ValueError("每种语言至少需要 2 条训练数据")

    label2id = {
        language: index
        for index, language in enumerate(languages)
    }
    id2label = {
        index: language
        for language, index in label2id.items()
    }

    df["label_id"] = df["label"].map(label2id)

    # 分层划分训练集与验证集
    val_size = max(NUM_CLASSES, int(len(df) * 0.2))

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=42,
        stratify=df["label_id"],
    )

    print(f"训练样本: {len(train_df)}")
    print(f"验证样本: {len(val_df)}")
    print(f"语言类别: {languages}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = LanguageDataset(
        train_df["text"],
        train_df["label_id"],
        tokenizer,
    )
    val_dataset = LanguageDataset(
        val_df["text"],
        val_df["label_id"],
        tokenizer,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # 加载预训练模型，设置为 12 分类
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    best_f1 = -1.0

    # 5. 正式训练
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            batch_size = batch["labels"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples

        accuracy, macro_f1 = evaluate(
            model, val_loader, device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f}"
        )

        # 保存验证集表现最好的模型
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print("已保存当前最佳模型")

    print(f"训练完成！最佳 Macro-F1: {best_f1:.4f}")
    print(f"模型路径: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data", default="languages.csv"
    )
    parser.add_argument(
        "--output", default="./language_model"
    )
    parser.add_argument(
        "--epochs", type=int, default=3
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5
    )

    args = parser.parse_args()
    train(args)
