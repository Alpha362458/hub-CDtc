"""
字符级自回归 Transformer 语言模型训练脚本，含 PPL 计算。
用法:
    python language_model.py --epochs 20
    python language_model.py --epochs 2 --embed_dim 64 --num_heads 4
"""

import math
import argparse
import glob
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型的训练样本：输入 x = 连续 seq_len 个字符，目标 y = 往后错一位。

    stride 控制切块间隔：
        stride=1  —— 每个字符位置都当一个样本（样本大量重叠，一期语料能切出上百万个样本）
        stride=seq_len —— 不重叠切块（标准做法，样本数 = 语料长度 / seq_len）
    语料一大，stride=1 会让一轮训练变得极慢，这不是模型慢，而是样本数被放大了 seq_len 倍。
    """

    def __init__(self, text, char2idx, seq_len, stride=1):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        start = idx * self.stride
        # 每个位置预测下一字，例如 x="今天天"，对应 y="天天气"。
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    """字符嵌入 + 可学习位置嵌入 + 因果 Transformer，输出各位置的下一字分数。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, seq_len, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 自注意力本身不表示顺序，位置嵌入用于区分字符所在位置。
        self.position_embed = nn.Embedding(seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # 虽使用 Encoder 层 API，加上因果掩码后只关注上文，构成无交叉注意力的自回归模型。
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        # Encoder 会克隆同一层，各层权重单独初始化，避免初始参数完全相同。
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.xavier_uniform_(block.linear2.weight)
        # 上三角的 True 禁止关注未来位置；当前字符和之前的字符仍可见。
        # buffer 随模型迁移设备，但不参与训练；掩码可重建，无需写入模型文件。
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        # 对每个位置的特征维度做归一化，再映射到整个字表。
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (B, T) 字符编号；B 为批次大小，T 为序列长度，V 为字表大小。
        length = x.size(1)
        if length == 0 or length > self.position_embed.num_embeddings:
            raise ValueError("输入序列长度必须在 1 到 seq_len 之间。")
        positions = torch.arange(length, device=x.device)
        e = self.drop(self.embed(x) + self.position_embed(positions))
        out = self.transformer(e, mask=self.causal_mask[:length, :length])
        # 返回原始 logits；CrossEntropyLoss 内部会处理 log-softmax，无需先做 softmax。
        return self.fc(self.norm(out))   # (B, T, V)


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, log_every=50):
    """遍历一轮数据；验证时由调用方使用 no_grad 禁用梯度计算。"""
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    started = time.perf_counter()
    phase = "Train" if train else "Val"

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        # 将 (B, T, V) 展平成 (B*T, V)，把每个字符位置视为一次分类预测。
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            loss.backward()
            # 限制梯度范数，减轻梯度过大导致训练不稳定的问题。
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 按目标字符数加权，避免最后一个较小批次对平均损失产生过大影响。
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        if log_every and (step == 1 or step % log_every == 0 or step == len(loader)):
            elapsed = time.perf_counter() - started
            print(
                f"  {phase} {step}/{len(loader)}  "
                f"loss={total_loss / total_tokens:.4f}  elapsed={elapsed:.1f}s",
                flush=True,
            )

    if not total_tokens:
        raise ValueError("数据集没有可用批次，请增加语料或减小 seq_len。")
    avg_loss = total_loss / total_tokens
    # 困惑度 PPL = exp(平均交叉熵)，衡量下一字预测的不确定性，越低越好。
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=0, help="切块间隔：0=使用 seq_len，1=密集重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=512, help="Transformer 前馈层维度")
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--num_heads",  type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_transformer.pt")
    parser.add_argument("--log_every",  type=int, default=50, help="批次日志间隔，0=只显示每轮结果")
    args = parser.parse_args()
    for name in ("epochs", "seq_len", "batch_size", "embed_dim", "hidden_dim", "num_layers", "num_heads"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} 必须大于 0")
    # 多头注意力将嵌入维度等分给各个头，因此必须整除。
    if args.embed_dim % args.num_heads:
        parser.error("--embed_dim 必须能被 --num_heads 整除")
    if not 0 < args.val_ratio < 1:
        parser.error("--val_ratio 必须在 0 和 1 之间")
    if args.stride < 0 or args.log_every < 0 or args.max_vocab < 0:
        parser.error("--stride、--log_every 和 --max_vocab 不能为负数")
    if not 0 <= args.dropout < 1 or args.lr <= 0:
        parser.error("--dropout 必须在 [0, 1) 内，--lr 必须大于 0")
    args.stride = args.stride or args.seq_len

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 优先使用 NVIDIA CUDA，其次使用 Mac MPS；不支持 GPU 时回退 CPU。
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    # 按行随机划分，默认 95% 训练、5% 验证；这是行数比例而非字符数比例。
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    # 仅用训练集构建字表，避免使用验证集字频；字表外的字符会被数据集过滤。
    char2idx, idx2char = build_vocab(train_text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    if not len(train_ds) or not len(val_ds):
        parser.error("训练集或验证集不足以切出样本，请增加语料或减小 --seq_len")
    # 每轮打乱训练样本；验证顺序固定，两者都保留最后不足 batch_size 的批次。
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    print(f"训练样本: {len(train_ds):,}  验证样本: {len(val_ds):,}  stride: {args.stride}")

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        started = time.perf_counter()
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True, log_every=args.log_every,
        )
        # 验证仅衡量泛化效果，不计算梯度，也不更新参数；eval 模式会关闭 dropout。
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False, log_every=args.log_every,
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            # 只保存验证 PPL 最低的模型；同时保存字表和结构参数，以便重建模型。
            # 未保存优化器状态，当前文件用于恢复模型推理，不是完整的续训检查点。
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
                "epoch": epoch,
                "val_ppl": va_ppl,
            }, args.save)

        elapsed = time.perf_counter() - started
        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
            f"{va_loss:>10.4f}  {va_ppl:>10.2f}{marker}  ({elapsed:.1f}s)",
            flush=True,
        )

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
