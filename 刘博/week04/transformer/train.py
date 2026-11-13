"""
Transformer 语言模型训练入口。
用法:
    python train.py --epochs 20
    python train.py --epochs 3 --stride 64 --seq_len 64   # 快速冒烟
产出: checkpoint（--save）+ 训练历史（--history_json，展示页面的曲线数据源）。
"""

import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import load_corpus, build_vocab, split_train_val, CharDataset
from model import build_lm


def build_warmup_scheduler(optimizer, warmup_steps):
    """线性 warmup：前 warmup_steps 个优化步内学习率从 0 线性升到设定值，之后保持不变。
    Transformer 对初始大学习率敏感，不 warmup 容易开局发散。"""
    def lr_lambda(step):
        return min(1.0, (step + 1) / max(1, warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, optimizer, device, train=True, scheduler=None):
    """跑一轮训练或评估，损失按 token 数加权平均，返回 (平均 loss, PPL)。
    scheduler 按优化步推进（warmup 以步计，不以轮计）。"""
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl",         choices=["official", "handwritten"], default="official",
                        help="模型实现：official=nn.TransformerEncoder 官方 API（默认），handwritten=逐组件手写版（两版数值等价）")
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--seq_len",      type=int,   default=64)
    parser.add_argument("--stride",       type=int,   default=64,
                        help="相邻样本起点间隔：1=逐字符滑动(重叠、慢)，seq_len=不重叠(默认)")
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--d_model",      type=int,   default=256, help="模型隐层维度（词嵌入维度同此值）")
    parser.add_argument("--nhead",        type=int,   default=8, help="注意力头数，需整除 d_model")
    parser.add_argument("--ff_dim",       type=int,   default=1024, help="前馈层维度")
    parser.add_argument("--num_layers",   type=int,   default=4, help="Transformer 层数")
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int,   default=1000, help="学习率线性 warmup 的优化步数")
    parser.add_argument("--val_ratio",    type=float, default=0.05)
    parser.add_argument("--max_vocab",    type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--corpus",       default="*.txt")
    parser.add_argument("--save",         default="best_transformer.pt")
    parser.add_argument("--history_json", default="history.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    print(f"词表大小: {len(char2idx)}")

    # 固定 seed 切分，evaluate.py 以相同参数复现同一份验证集
    train_text, val_text = split_train_val(text, args.val_ratio, args.seed)

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = build_lm(
        vocab_size=len(char2idx),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        impl=args.impl,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"每轮步数: train {len(train_loader)}, val {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = build_warmup_scheduler(optimizer, args.warmup_steps)

    history = []
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'Sec':>6}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, scheduler=scheduler)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        sec = time.time() - t0

        marker = "  *" if va_ppl < best_val_ppl else ""
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_ppl": tr_ppl,
            "val_loss": va_loss, "val_ppl": va_ppl,
            "lr": optimizer.param_groups[0]["lr"], "sec": round(sec, 1),
        })
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            # checkpoint 带上含本轮在内的完整 history（先 append 再保存）
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
                "history": history,
            }, args.save)
        # 每轮落盘一次，中断也不丢曲线数据
        with open(args.history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  "
              f"{va_ppl:>10.2f}  {sec:>6.1f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
