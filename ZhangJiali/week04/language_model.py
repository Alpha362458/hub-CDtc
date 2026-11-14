"""
字符级 Transformer 语言模型训练脚本（Decoder-only / 因果注意力）。
用法:
    python language_model.py --epochs 5 --device cuda
    python language_model.py --epochs 3 --stride 64 --batch_size 128 --amp
"""

import math
import argparse
import glob
import json
import os
import random
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def resolve_device(device_str: str) -> torch.device:
    """解析 --device：auto / cuda / cuda:0 / cpu。"""
    device_str = (device_str or "auto").lower()
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境检测不到可用 GPU（请确认已安装 GPU 版 PyTorch）。")
    return device


def setup_cuda(seed: int, device: torch.device):
    """GPU 侧确定性与性能相关设置。"""
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type != "cuda":
        return
    torch.cuda.manual_seed_all(seed)
    # 固定输入长度时可加速；作业场景优先吞吐
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """输入 x = 连续 seq_len 个字符，目标 y = 往后错一位（下一字符预测）。"""

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    """可学习位置编码（与课件 BERT 一致的思路，更利于作业演示）。"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.drop(x + self.pe(positions))


class TransformerLM(nn.Module):
    """
    Decoder-only Transformer 语言模型：
    Embedding + 位置编码 → TransformerEncoder(带因果 mask) → Linear → 词表 logits
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, seq_len, device):
        # True 表示该位置被屏蔽；上三角（不含对角线）禁止看未来
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x):
        # x: (B, T)
        if x.size(1) > self.max_len:
            raise ValueError(f"序列长度 {x.size(1)} 超过 max_len={self.max_len}")
        h = self.embed(x) * math.sqrt(self.d_model)
        h = self.pos(h)
        mask = self._causal_mask(x.size(1), x.device)
        h = self.encoder(h, mask=mask)
        return self.fc(h)  # (B, T, V)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    use_amp = scaler is not None and device.type == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))  # 防止 overflow
    return avg_loss, ppl


def build_loader(dataset, batch_size, shuffle, drop_last, device, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
    }
    if device.type == "cuda":
        kwargs["pin_memory"] = True
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="训练字符级 Transformer 语言模型")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64, help="切块间隔：seq_len=不重叠（推荐）")
    parser.add_argument("--batch_size", type=int, default=128, help="GPU 默认可开更大；显存不够再调小")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--max_vocab", type=int, default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--max_train_chars", type=int, default=0, help="限制训练语料字符数，0=不限")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cuda / cuda:0 / cpu")
    parser.add_argument("--amp", action="store_true", help="启用 CUDA 混合精度（fp16）")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers；默认 GPU=2 / CPU=0")
    parser.add_argument(
        "--corpus",
        default=os.path.join("..", "循环神经网络语言模型", "corpus_general.txt"),
        help="语料路径或 glob",
    )
    parser.add_argument("--save", default="best_model.pt")
    parser.add_argument("--log", default="train_log.txt")
    parser.add_argument("--report", default="train_report.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    setup_cuda(args.seed, device)
    if args.num_workers is None:
        args.num_workers = 2 if device.type == "cuda" else 0
    use_amp = bool(args.amp and device.type == "cuda")

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"device: {device}  model: TransformerLM  amp={use_amp}")
    if device.type == "cuda":
        log(f"gpu: {torch.cuda.get_device_name(device)}  capability={torch.cuda.get_device_capability(device)}")
    log(f"args: {vars(args)}")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到语料: {args.corpus}")
    if args.max_train_chars and len(text) > args.max_train_chars:
        text = text[: args.max_train_chars]
        log(f"语料已截断至 {args.max_train_chars:,} 字符")
    log(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    log(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)
    log(f"训练样本数: {len(train_ds):,}  验证样本数: {len(val_ds):,}")

    train_loader = build_loader(
        train_ds, args.batch_size, shuffle=True, drop_last=True, device=device, num_workers=args.num_workers
    )
    val_loader = build_loader(
        val_ds, args.batch_size, shuffle=False, drop_last=False, device=device, num_workers=args.num_workers
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=max(args.seq_len, 512),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    best_val_ppl = float("inf")
    history = []
    t0 = time.time()

    header = f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}"
    log(header)
    log("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler if use_amp else None, train=True
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, scaler=None, train=False
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            # 权重落到 CPU 再存，方便在无 GPU 机器上加载提交
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": state,
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                    "best_val_ppl": best_val_ppl,
                    "vocab_size": vocab_size,
                    "device_trained": str(device),
                },
                args.save,
            )

        row = f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}"
        log(row)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_ppl": tr_ppl,
                "val_loss": va_loss,
                "val_ppl": va_ppl,
            }
        )

    elapsed = time.time() - t0
    log(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")
    log(f"总耗时: {elapsed:.1f}s")

    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    report = {
        "model": "TransformerLM",
        "device": str(device),
        "amp": use_amp,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "params": total_params,
        "vocab_size": vocab_size,
        "corpus_chars": len(text),
        "best_val_ppl": best_val_ppl,
        "elapsed_sec": elapsed,
        "args": vars(args),
        "history": history,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"报告已写入 {args.report}")


if __name__ == "__main__":
    main()
