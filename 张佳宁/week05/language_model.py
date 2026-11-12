import math
import sys
import argparse
import glob
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────── 数据 ───────────────────────────


DEFAULT_CORPUS = str(Path(__file__).resolve().parent / "*.txt")
DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parent / "transformer_encoder_model.pt"
)


def load_corpus(pattern=DEFAULT_CORPUS):
    texts = []
    for path in sorted(glob.glob(pattern)):
        if not Path(path).is_file():
            continue
        print(path)
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
    chars = (
        [c for c, _ in counter.most_common(max_vocab)]
        if max_vocab
        else sorted(set(text))
    )
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
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────


class LM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        nhead,
        num_encoder_layers,
        dropout,
        max_seq_len=64,
    ):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError("embed_dim 必须能被 nhead 整除。")
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.tf_ers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        # Encoder 会复制模板层；给各层权重分别初始化，避免初始值完全相同。保证批量传入的transformer层初始化参数完全随机，而不是复制上一层。
        for layer in self.tf_ers.layers:
            for parameter in layer.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)

        if seq_len > self.position_embed.num_embeddings:
            raise ValueError("输入长度超过模型的位置编码长度。")
        positions = torch.arange(seq_len, device=x.device)
        e = self.drop(self.embed(x) + self.position_embed(positions))
        # True 表示禁止关注未来位置；只有自注意力，因此只需要一个因果掩码。
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        out = self.tf_ers(e, mask=causal_mask)
        logits = self.fc(self.drop(out))  # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────


def run_epoch(model, loader, criterion, optimizer, device, train=True):
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

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        raise ValueError("没有可用批次，请增加语料，或减小 seq_len、batch_size。")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="transformer", choices=["transformer"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument(
        "--stride", type=int, default=8, help="切块间隔：1=密集重叠，seq_len=不重叠"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument(
        "--max_vocab", type=int, default=4000, help="字表最大字数，0=不限"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help="默认读取脚本所在目录的所有 .txt 文件（不包含子目录）；也可指定路径或通配符",
    )
    parser.add_argument("--save", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            f"CUDA 不可用：Python={sys.executable}，PyTorch={torch.__version__}。"
            "请使用安装了 CUDA 版 PyTorch 的 py312 环境。"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("当前环境不支持 MPS。")
    if args.device != "auto":
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}  CUDA build: {torch.version.cuda}")
    print(f"device: {device}  model: {args.model.upper()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    elif args.device == "auto":
        print("未检测到可用 CUDA，自动使用其他设备；可用 --device cuda 强制检查 GPU。")

    if args.epochs < 1:
        parser.error("epochs 必须至少为 1。")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到 .txt 语料或内容为空，查找路径：{args.corpus}")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds = CharDataset(val_text, char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(
        f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}"
    )
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save(
                {
                    "architecture": "causal_transformer_encoder",
                    "model_state": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                args.save,
            )

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}"
        )

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
