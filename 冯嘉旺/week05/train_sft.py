"""从 week04 的预训练 Transformer 权重进行回答区域监督微调。"""

import argparse
import json
import math
import random
import time
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sft_model import (
    EOS, IGNORE_INDEX, PAD, ROOT, UNK, checked_output, encode, format_prompt,
    get_device, init_from_pretrained, load_records,
)


class SFTDataset(Dataset):
    def __init__(self, records, vocab, seq_len):
        self.samples = []
        self.unknown_chars = 0
        self.max_length = 0
        for number, row in enumerate(records, 1):
            prompt = encode(format_prompt(row["instruction"]), vocab)
            answer = encode(row["output"], vocab) + [vocab[EOS]]
            ids = prompt + answer
            self.max_length = max(self.max_length, len(ids) - 1)
            self.unknown_chars += ids.count(vocab[UNK])
            if len(ids) - 1 > seq_len:
                raise ValueError(
                    f"第 {number} 条样本需 {len(ids) - 1} 个位置，超过 seq_len={seq_len}；"
                    "请增大 --seq_len，避免截掉指令或答案"
                )
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            # y[t] 预测 ids[t+1]；最后一个提示字符预测第一个回答字符。
            y[:len(prompt) - 1] = IGNORE_INDEX
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_batch(samples, pad_id):
    length = max(len(x) for x, _ in samples)
    x_batch = torch.full((len(samples), length), pad_id, dtype=torch.long)
    y_batch = torch.full((len(samples), length), IGNORE_INDEX, dtype=torch.long)
    for index, (x, y) in enumerate(samples):
        x_batch[index, :len(x)] = x
        y_batch[index, :len(y)] = y
    return x_batch, y_batch


def run_epoch(model, loader, device, optimizer=None, log_every=20):
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    total_loss, total_tokens = 0.0, 0
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("损失不是有限数，请检查学习率与数据")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = (y != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * count
            total_tokens += count
            if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
                print(
                    f"  {'Train' if training else 'Val'} {step}/{len(loader)}"
                    f" loss={total_loss / total_tokens:.4f}"
                    f" elapsed={time.perf_counter() - started:.1f}s", flush=True,
                )
    if not total_tokens:
        raise ValueError("没有可用于监督的回答字符")
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", default=str(ROOT.parent / "week04/best_model_transformer.pt"))
    parser.add_argument("--train", default=str(ROOT / "sft_train.jsonl"))
    parser.add_argument("--val", default=str(ROOT / "sft_val.jsonl"))
    parser.add_argument("--save", default=str(ROOT / "best_model_sft.pt"))
    parser.add_argument("--history", default=str(ROOT / "sft_history.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--limit_train", type=int, default=0, help="仅用于冒烟测试；0=全部")
    parser.add_argument("--limit_val", type=int, default=0, help="仅用于冒烟测试；0=全部")
    args = parser.parse_args()
    for key in ("epochs", "seq_len", "batch_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} 必须大于 0")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须是正的有限数")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout 必须在 [0, 1) 内")
    if min(args.log_every, args.limit_train, args.limit_val) < 0:
        parser.error("日志间隔和样本限制不能为负数")
    protected = (args.pretrained, args.train, args.val)
    save_path = checked_output(args.save, protected)
    history_path = checked_output(args.history, protected)
    if save_path == history_path:
        parser.error("--save 和 --history 不能相同")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    train_records, val_records = load_records(args.train), load_records(args.val)
    if args.limit_train:
        train_records = train_records[:args.limit_train]
    if args.limit_val:
        val_records = val_records[:args.limit_val]
    # 仅加载自己信任的本地检查点，不执行 pickle 中的任意对象。
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    model, vocab, config = init_from_pretrained(
        checkpoint, train_records, args.seq_len, args.dropout,
    )
    train_ds = SFTDataset(train_records, vocab, args.seq_len)
    val_ds = SFTDataset(val_records, vocab, args.seq_len)
    collate = partial(collate_batch, pad_id=vocab[PAD])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"device: {device} | 从预训练权重做全参数 SFT", flush=True)
    print(
        f"词表: {len(checkpoint['char2idx'])} -> {len(vocab)} | "
        f"上下文: {checkpoint['args']['seq_len']} -> {args.seq_len}\n"
        f"训练/验证样本: {len(train_ds)}/{len(val_ds)} | "
        f"最大输入长度: {train_ds.max_length}/{val_ds.max_length}\n"
        f"训练/验证未知字符: {train_ds.unknown_chars}/{val_ds.unknown_chars} | "
        f"参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True,
    )
    baseline_loss, baseline_ppl = run_epoch(model, val_loader, device, log_every=args.log_every)
    print(f"微调前 Val Loss={baseline_loss:.4f}，回答区域 PPL={baseline_ppl:.2f}", flush=True)
    history = {
        "baseline": {"val_loss": baseline_loss, "val_ppl": baseline_ppl},
        "training_args": vars(args), "model_config": config, "epochs": [],
    }
    best_loss = float("inf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        tr_loss, tr_ppl = run_epoch(model, train_loader, device, optimizer, args.log_every)
        va_loss, va_ppl = run_epoch(model, val_loader, device, log_every=args.log_every)
        improved = va_loss < best_loss
        if improved:
            best_loss = va_loss
            torch.save({
                "format": "char_transformer_sft_v1",
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "char2idx": vocab, "idx2char": {i: c for c, i in vocab.items()},
                "args": config, "training_args": vars(args),
                "epoch": epoch, "val_loss": va_loss, "val_ppl": va_ppl,
            }, save_path)
        history["epochs"].append({
            "epoch": epoch, "train_loss": tr_loss, "train_ppl": tr_ppl,
            "val_loss": va_loss, "val_ppl": va_ppl, "saved_best": improved,
        })
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"Train Loss={tr_loss:.4f} PPL={tr_ppl:.2f} | "
            f"Val Loss={va_loss:.4f} PPL={va_ppl:.2f}"
            f"{' * 已保存' if improved else ''}", flush=True,
        )
    print(f"\n最佳 SFT 权重：{save_path}\n训练记录：{history_path}")


if __name__ == "__main__":
    main()
