import argparse
import functools
import json
import math
import os
import random
import time
import pathlib

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import SYSTEM_PROMPT, SFTDataset, collate_fn

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "sft_train"
MODEL_PATH = ROOT_DIR.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
CKPT_PATH = ROOT_DIR / "checkpoints"
ADAPTER_PATH = CKPT_PATH / "sft_lora"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loader(path, tokenizer, args, shuffle):
    dataset = SFTDataset(path, tokenizer, args.max_length)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total_loss += model(**batch).loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.to(device)

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_path = args.data /"sft_train.jsonl"
    val_path = args.data /"sft_val.jsonl"
    train_loader = make_loader(train_path, tokenizer, args, shuffle=True)
    val_loader = make_loader(val_path, tokenizer, args, shuffle=False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        loss_sum, loss_count = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            loss_sum += loss.item()
            loss_count += 1

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0 or global_step == total_steps:
                    print(
                        f"epoch {epoch}/{args.epochs} step {global_step}/{total_steps} "
                        f"loss {loss_sum / loss_count:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                        f"耗时 {time.time() - start:.0f}s"
                    )
                    loss_sum, loss_count = 0.0, 0

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch} 结束，验证集 loss: {val_loss:.4f}")

        model.save_pretrained(str(ADAPTER_PATH))
        tokenizer.save_pretrained(str(ADAPTER_PATH))

    with open(os.path.join(ADAPTER_PATH, "sft_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.model,
                "system_prompt": SYSTEM_PROMPT,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"训练完成，LoRA adapter 已保存到 {ADAPTER_PATH}")


if __name__ == "__main__":
    main()
