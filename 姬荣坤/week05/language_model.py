import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：预留特殊的 <pad>, <bos>, <eos>，然后统计字符。
    """
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    
    # 特殊 token 放在最前面 (0, 1, 2)
    special_tokens = ["<pad>", "<bos>", "<eos>"]
    char2idx = {c: i for i, c in enumerate(special_tokens)}
    
    # 把语料里的字符加进去
    for c in chars:
        if c not in char2idx:
            char2idx[c] = len(char2idx)
            
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """
    字符级语言模型数据集。
    """
    def __init__(self, text, char2idx, seq_len, stride):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.pad_id = char2idx["<pad>"]
        
        # 过滤掉词表里没有的字（保险起见）
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

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, nhead=8):
        super().__init__()
        self.model_type = model_type
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        if self.model_type == "transformer":
            assert embed_dim % nhead == 0, "embed_dim 必须能被 nhead 整除"
            # 修正 FFN 维度比例，保证参数量合理
            hidden_dim = hidden_dim * 4 
            
            self.pos_embedding = nn.Embedding(512, embed_dim)
            
            # 🌟 核心改进：加入 norm_first=True (Pre-LN)，保证深层 Transformer 不崩
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=nhead, 
                dim_feedforward=hidden_dim, 
                dropout=dropout,
                batch_first=True,
                norm_first=True 
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(embed_dim, vocab_size)
            
        else:
            rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
            self.rnn = rnn_cls(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        e = self.drop(self.embed(x))
        if self.model_type == "transformer":
            B, T, E = e.size()
            positions = torch.arange(0, T, device=x.device).unsqueeze(0).expand(B, T)
            pe = self.pos_embedding(positions)
            e = e + pe
            attn_mask = self.generate_square_subsequent_mask(T, x.device)
            out = self.transformer(e, mask=attn_mask)
        else:
            out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))
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
            # 🌟 核心改进：梯度裁剪，防止 Transformer 梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm"])
    parser.add_argument("--epochs",     type=int,   default=10) # 数据大了，10轮足够了
    parser.add_argument("--seq_len",    type=int,   default=128) # 上下文长度增加到 128
    parser.add_argument("--stride",     type=int,   default=128, help="切块间隔：128=不重叠切块，加速训练")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=384)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=4) # 数据大了，网络加深到4层
    parser.add_argument("--dropout",    type=float, default=0.1) # 降低 dropout，加快拟合
    parser.add_argument("--lr",         type=float, default=5e-4) # 学习率调低
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=0, help="0=不限字数，收录所有汉字")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="corpus_general.txt") # 🌟 默认使用你的大语料文件
    parser.add_argument("--save",       default="best_model_transformer.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 1. 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError(f"未找到 {args.corpus} 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    # 2. 构建词表
    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size} (覆盖了绝大多数常用汉字)")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 3. 模型初始化
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    # 4. 开始训练
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
