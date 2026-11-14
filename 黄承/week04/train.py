"""MLM 掩码语言模型预训练：手写 BERT(bert.py) + HuggingFace datasets/transformers

依赖安装：pip install datasets
"""
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_dataset

from bert import BertConfig, BertForMaskedLM

TOKENIZER_NAME = "bert-base-uncased"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_dataset(tokenizer):
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    tokenized = raw.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LEN),
        batched=True,
        remove_columns=raw.column_names,
    )
    return tokenized.filter(lambda x: len(x["input_ids"]) > 0)

def main():
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_NAME)
    dataset = build_dataset(tokenizer)
    print(f"样本数：{len(dataset)}")

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=512,
        num_hidden_layers=4,
        pad_token=tokenizer.pad_token_id,
        position_size=MAX_LEN,
    )
    model = BertForMaskedLM(config).to(DEVICE)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    optim = AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            if step % 200 == 0:
                print(f"Epoch {epoch} step {step}/{len(loader)} loss={loss.item():.4f}")
        print(f"Epoch {epoch} avg_loss={total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "mlm_bert.pt")
    print("已保存 mlm_bert.pt")

if __name__ == "__main__":
    main()
