"""
字符级语言模型训练脚本，只用 BERT，含 PPL 计算。

用法:
    python bert_language_model.py                       # 默认参数训练
    python bert_language_model.py --epochs 20 --seq_len 128
    python bert_language_model.py --bidirectional       # 双向 BERT（仅作对比，见下）

关于"用 BERT 做语言模型"的一个坑:
    BERT 原生是双向编码器。如果直接让它预测"下一个字"，它同时能看到右边的字——
    而右边那个字正是要预测的标签，等于开卷考试，loss 会低得不真实，PPL 没有意义。
    所以下面用 BertConfig(is_decoder=True) 把它配成因果（单向）形态，
    transformers 会自动生成因果注意力掩码，每个位置只能看到自己和左边的字。
    --bidirectional 可以关掉这个约束，用来做对比实验，但那条曲线不能当 PPL 解读。
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel

# BERT 位置编码上限，seq_len 不能超过它
POS_MAX = 512


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
        x = self.data[start: start + self.seq_len]
        y = self.data[start + 1: start + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

def pick_heads(hidden_dim):
    """注意力头数必须整除 hidden_size，从"每头 64 维"往下找一个能整除的头数。"""
    heads = max(1, hidden_dim // 64)
    while heads > 1 and hidden_dim % heads:
        heads -= 1
    return heads


class BertLM(nn.Module):
    """
    BERT 编码器 + 线性输出头，每个位置输出"下一个字"的词表 logits。

    直接复用 BertModel 自带的词嵌入、位置编码和 Transformer 层，
    不再自己写 nn.Embedding —— 那层维度和 BERT 的 hidden_size 对不上时就会报错。
    """

    def __init__(self, vocab_size, hidden_dim, num_layers, dropout, causal=True, tie_weights=True):
        super().__init__()
        self.config_bert = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=pick_heads(hidden_dim),
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=POS_MAX,
            type_vocab_size=1,
            # is_decoder=True：传入 2D attention_mask 时自动加因果掩码（只看左边）
            is_decoder=causal,
        )
        self.bert = BertModel(self.config_bert)
        self.drop = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        if tie_weights:
            # 输入词嵌入与输出投影共享权重：小语料上能省参数、也常能缓解过拟合
            self.lm_head.weight = self.bert.embeddings.word_embeddings.weight

    def forward(self, x):
        # 每条样本都是等长的完整序列，没有 padding，掩码全 1 即可
        attention_mask = torch.ones_like(x)
        hidden = self.bert(input_ids=x, attention_mask=attention_mask).last_hidden_state
        return self.lm_head(self.drop(hidden))


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, clip=0.0):
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
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=1, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip",       type=float, default=1.0, help="梯度裁剪阈值，0=关闭")
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model_bert.pt",
                        help="默认不覆盖 LSTM 的 best_model.pt")
    parser.add_argument("--bidirectional", action="store_true",
                        help="关掉因果掩码（双向 BERT），仅用于对比，PPL 不可信")
    parser.add_argument("--tie_weights",   action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not 1 <= args.seq_len <= POS_MAX:
        raise ValueError(f"seq_len 必须在 1~{POS_MAX} 之间（受 BERT 位置编码上限限制）")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    causal = not args.bidirectional
    print(f"device: {device}  model: BERT  causal: {causal}")
    if not causal:
        print("警告: --bidirectional 下每个位置能看到右边的字（也就是标签），PPL 会虚低，只能作对比。")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text, args.max_vocab or None)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 模型
    model = BertLM(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        causal=causal,
        tie_weights=args.tie_weights,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}  (注意力头数: {pick_heads(args.hidden_dim)})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device,
                                    train=True, clip=args.clip)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "causal": causal,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
