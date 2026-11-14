import json

import torch
from torch.utils.data import Dataset

SYSTEM_PROMPT = "你是一个乐于助人的中文助手。"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.samples = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["instruction"]},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        answer = sample["output"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + answer_ids)[: self.max_length]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate_fn(batch, pad_token_id):
    """同 batch 内做右侧 padding，padding 位置不参与 loss。"""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat(
                [torch.ones(len(item["input_ids"]), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]
            )
        )
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }
