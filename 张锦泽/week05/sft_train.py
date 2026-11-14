import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "uer/gpt2-chinese-cluecorpussmall"
MAX_LENGTH = 256


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=MAX_LENGTH):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                instruction = str(item["instruction"]).strip()
                context = str(item.get("input", "")).strip()
                answer = str(item["output"]).strip()
                if not instruction or not answer:
                    continue

                prompt = f"指令：{instruction}\n"
                if context:
                    prompt += f"输入：{context}\n"
                prompt += "回答："

                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                # 中文 GPT-2 的样本以 [CLS] 开始，以 [SEP] 结束
                if tokenizer.cls_token_id is not None:
                    prompt_ids = [tokenizer.cls_token_id] + prompt_ids
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                end_id = tokenizer.sep_token_id
                if end_id is None:
                    raise ValueError("Tokenizer 缺少 sep_token_id")

                answer_ids = answer_ids[:max_length - 2] + [end_id]
                prompt_ids = prompt_ids[:max_length - len(answer_ids)]

                self.samples.append({
                    "input_ids": prompt_ids + answer_ids,
                    "labels": [-100] * len(prompt_ids) + answer_ids,
                })

        if len(self.samples) < 2:
            raise ValueError("至少需要 2 条有效训练数据")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = torch.tensor(item["input_ids"])
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(item["labels"])

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(model(**batch).loss.item())
    return sum(losses) / max(len(losses), 1)


def train(args):
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.to(device)

    dataset = SFTDataset(args.data, tokenizer, args.max_length)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    make_collator = lambda rows: collate_fn(rows, tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=make_collator
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=make_collator
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"训练样本：{len(train_set)}，验证样本：{len(val_set)}")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(len(train_losses), 1)
        val_loss = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"已保存最佳模型：{output_dir}")

    print(f"训练完成，最佳验证 Loss：{best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基础中文 GPT-2 SFT 训练")
    parser.add_argument("--data", default="sft_data.jsonl")
    parser.add_argument("--output", default="./sft_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    train(parser.parse_args())
