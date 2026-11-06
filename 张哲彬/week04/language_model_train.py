import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import json
import random
import math
from tqdm.auto import tqdm

def load_corpus(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        files.extend(
            os.path.join(root, filename)
            for filename in filenames
            if not filename.startswith(".")
        )
    files.sort()  # 保证每次运行顺序一致，便于调试

    texts = []
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSON 解析失败: {path}:{line_number}: {error.msg}"
                    ) from error

                if isinstance(record, dict):
                    text = record.get("text")
                    if isinstance(text, str):
                        texts.append(text)
    return texts

def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {"<unk>": 0, "<eos>": 1}
    char2idx.update({c: i for i, c in enumerate(chars, start=len(char2idx))})
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def encode_documents(texts, char2idx):
    unk_id = char2idx["<unk>"]
    eos_id = char2idx["<eos>"]

    encoded = []
    for document in texts:
        encoded.extend([char2idx.get(ch, unk_id) for ch in document])
        encoded.append(eos_id)  # 每篇文章末尾加入 EOS
    return encoded

class CharDataset(Dataset):
    def __init__(self, ids, seq_len, stride):
        self.seq_len = seq_len
        self.stride = stride
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len + self.stride - 1) // self.stride)

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        x = self.data[start_idx: start_idx + self.seq_len]
        y = self.data[start_idx + 1: start_idx + self.seq_len + 1]
        return x, y    

# 包含8层transformer的网络
class DIY_Transformer_Network(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        # TransformerEncoder clones the supplied layer, so initialize each clone independently.
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len), containing token IDs.
        if input_ids.ndim != 2:
            raise ValueError("input_ids 的形状应为 (batch_size, seq_len)")
        batch_size, seq_len = input_ids.shape

        hidden = self.embedding(input_ids)  #(batch_size, seq_len, embed_dim)
        hidden = self.dropout(hidden)

        # True 表示屏蔽该位置，确保当前位置无法关注未来 token。
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.output_projection(hidden)  # (batch_size, seq_len, vocab_size)

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    phase = "训练" if train else "验证"
    progress = tqdm(loader, desc=phase, leave=False)

    for x, y in progress:
        x, y = x.to(device), y.to(device)
        logits = model(x)   # (B, T, V)
        # 损失函数输入维度 (B*T, V)，标签维度 (B*T)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if total_tokens:
            progress.set_postfix(loss=f"{total_loss / total_tokens:.4f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len",    type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--corpus",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata"),
    )
    parser.add_argument(
            "--save",
            type=str,
            default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt"),
        )
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # 数据准备
    texts = load_corpus(args.corpus)
    if not texts:
        raise FileNotFoundError("语料目录中没有可读取的文件内容，请确认目录路径和文件内容。")
    print(f"语料字符数: {len(''.join(texts)):,}")

    random.shuffle(texts)
    split = int(len(texts) * (1 - args.val_ratio))
    train_texts = texts[:split]
    val_texts   = texts[split:]

    char2idx, idx2char = build_vocab("".join(train_texts))
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    train_ids = encode_documents(train_texts, char2idx)
    val_ids = encode_documents(val_texts, char2idx)

    train_ds = CharDataset(train_ids, args.seq_len, args.stride)
    val_ds   = CharDataset(val_ids,   args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = DIY_Transformer_Network(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")



if __name__ == "__main__":
    main()