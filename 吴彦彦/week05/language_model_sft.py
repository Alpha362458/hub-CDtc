"""
字符级语言模型训练脚本，支持 RNN / LSTM 切换，含 PPL 计算。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    BASE_DIR = Path(__file__).resolve().parent
    path = BASE_DIR / pattern
    with open(path, encoding="utf-8", errors="ignore") as f:
        texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按出现频率取前 max_vocab 个字（None = 全部字符）不包括特殊token。
    语料里的生僻字超出字表后会被 CharDataset 过滤掉 —— 字表越小模型越小，
    但覆盖不足会丢信息，这是词表大小的取舍。
    """
    from collections import Counter
    special_tokens = ["<|user|>", "<|asst|>", "<|end|>", "<|pad|>", "<|unk|>"]
    # # 先把特殊串从原始文本剔除，防止特殊串的碎片被统计成普通字符
    # clean_text = text
    # for sp in special_tokens:
    #     text = clean_text.replace(sp, "")
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    for sp in special_tokens:
        if sp not in char2idx:
            char2idx[sp] = len(char2idx)
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, special_tokens


"""优先匹配特殊token，否则按字符分词"""
class CharTokenizerWithSpecial:
    def __init__(self, char2idx, idx2char, special_tokens):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.special_tokens = special_tokens
        self.special_set = set(special_tokens)
        if self.special_tokens:
            self.max_sp_len = max(len(t) for t in self.special_tokens)
        else:
            self.max_sp_len = 0

    def encode(self, text: str):
        """
        编码：贪心最长优先匹配特殊 token；失败则取单个字符
        """
        ids = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            # 从最长候选开始试，保证长特殊 token 优先
            for L in range(min(self.max_sp_len, n - i), 0, -1):
                seg = text[i:i+L]
                if seg in self.special_set:
                    matched = seg
                    break
            if matched is not None:
                ids.append(self.char2idx[matched])
                i += len(matched)
            else:
                c = text[i]
                # ========== 词表外字符处理 ==========
                if c not in self.char2idx:
                    # 方案A：抛异常（严格模式）
                    # raise KeyError(f"字符不在词表：{repr(c)}")
                    # 方案B：如果你加了 <unk>，就启用下面这句，注释上面
                    ids.append(self.char2idx["<|unk|>"])
                else:
                    ids.append(self.char2idx[c])
                i += 1
        return ids

    def decode(self, ids):
        """id 还原成字符串"""
        parts = [self.idx2char[i] for i in ids]
        return "".join(parts)

    def tokenize(self, text: str):
        """方便调试：返回 token 字符串列表，和 hf tokenizer.tokenize 对齐"""
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for L in range(min(self.max_sp_len, n - i), 0, -1):
                seg = text[i:i+L]
                if seg in self.special_set:
                    matched = seg
                    break
            if matched is not None:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    @property
    def vocab_size(self):
        return len(self.char2idx)


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

Test_lable = True
class SFTEncoderDataset(Dataset):
    """基于语言模型继续SFT训练, 教会模型问答"""
    def __init__(self, path, char2idx, tokenizer, seq_len, stride=1):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.tok = tokenizer
        self.asst_token_id = char2idx["<|asst|>"]
        self.pad_id = char2idx["<|pad|>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        ins, out = d["instruction"], d["output"]
        prompt = "<|user|>" + ins + "<|asst|>"
        full = prompt + out + "<|end|>"
        input_ids = self.tok.encode(full)

        # 截断
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[:self.seq_len]

        labels = input_ids.copy()
        try:
            # 找到<|asst|>位置，它之前全部标记为-100
            asst_pos = input_ids.index(self.asst_token_id)
            for i in range(asst_pos + 1):
                labels[i] = -100
        except ValueError:
            labels = [-100] * len(labels)

        # ========== 补齐到固定 seq_len ==========
        pad_len = self.seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [self.pad_id] * pad_len
            labels += [-100] * pad_len

        # 测试代码
        global Test_lable
        if Test_lable:
            print(labels)
            Test_lable = False

        # 因果mask 在训练循环里按当前序列长度生成，不要放这里（batch内长度会变）
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout, seq_len=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if model_type == "lstm" or model_type == "rnn":
            model_cls = nn.LSTM if model_type == "lstm" else nn.RNN
            self.model = model_cls(
                embed_dim, hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.pos_embed = nn.Embedding(seq_len, embed_dim)
            model_cls = nn.TransformerEncoder
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=16,
                dim_feedforward=hidden_dim,
                batch_first=True,
                activation="gelu"
            )
            self.model = model_cls(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, pad_mask):
        e = self.drop(self.embed(x))
        if isinstance(self.model, (nn.RNN, nn.LSTM)):
            out, _ = self.model(e)
            logits = self.fc(self.drop(out))   # (B, L, V)
        else:
            B, L = x.shape
            pos_idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            pos_emb = self.pos_embed(pos_idx)
            e = e + pos_emb
            # causal_mask = nn.Transformer.generate_square_subsequent_mask(L).to(e.device)
            causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            # 合并pad_mask到casual mask上
            # 扩展 pad mask 到 [B, 1, L]，作用在 j（key）维度
            # pad_mask_j = pad_mask.unsqueeze(1)   # [B, 1, L]
            # 扩展 pad mask 到 [B, L, 1]，作用在 i（query）维度
            # pad_mask_i = pad_mask.unsqueeze(2)   # [B, L, 1]
            # 合并：因果掩码 OR query是pad OR key是pad
            # full_mask = causal_mask.unsqueeze(0) | pad_mask_j | pad_mask_i
            out = self.model(e, mask=causal_mask, src_key_padding_mask = pad_mask)
            logits = self.fc(out)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, pad_id, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pad_mask = (x == pad_id)   # shape [B, L]
        logits = model(x, pad_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
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
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--stride",     type=int,   default=64, help="切块间隔：1=密集重叠，seq_len=不重叠")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--max_vocab",  type=int,   default=4000, help="字表最大字数，0=不限")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--corpus",     default="corpus_general.txt")
    parser.add_argument("--save",       default="best_model_transformer_sft.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char, special_tokens = build_vocab(text, args.max_vocab or None)
    tokenizer = CharTokenizerWithSpecial(char2idx, idx2char, special_tokens)
    new_vocab_size = tokenizer.vocab_size
    old_vocab_size = new_vocab_size - len(special_tokens)
    print(f"词表大小: {new_vocab_size}")
    for sp in special_tokens:
        print(f"{sp} -> id: {char2idx[sp]}")

    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent

    # 测试CharTokenizerWithSpecial分词
    # sample = "<|user|>今天吃什么<|end|><|asst|>吃面条<|end|>"
    # print("\ntokenize：", tokenizer.tokenize(sample))
    # ids = tokenizer.encode(sample)
    # print("encode ids：", ids)
    # restore = tokenizer.decode(ids)
    # print("decode：", repr(restore))
    # print("是否一致：", restore == sample)
    # return

    # lines = text.splitlines()
    # random.shuffle(lines)
    # split = int(len(lines) * (1 - args.val_ratio))
    # train_text = "\n".join(lines[:split])
    # val_text   = "\n".join(lines[split:])

    # train_ds = CharDataset(train_text, char2idx, args.seq_len, args.stride)
    # val_ds   = CharDataset(val_text,   char2idx, args.seq_len, args.stride)

    train_path = BASE_DIR/"sft_train.jsonl"
    val_path = BASE_DIR/"sft_val.jsonl"
    print(train_path)
    train_ds = SFTEncoderDataset(train_path, char2idx, tokenizer, args.seq_len, args.stride)
    val_ds   = SFTEncoderDataset(val_path, char2idx, tokenizer, args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=old_vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        dropout=args.dropout,
        seq_len=args.seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 加载训练好的参数
    model_dict = torch.load(BASE_DIR/"best_model_transformer.pt", map_location=device)
    model.load_state_dict(model_dict['model_state'])  # 加载训练好的权重

    # ========= 扩容 embedding =========
    old_emb = model.embed
    new_emb = nn.Embedding(new_vocab_size, old_emb.embedding_dim)
    # 复制原来的权重
    print(f"new_vocab_size:{new_vocab_size}, old_vocab_size:{old_vocab_size}")
    new_emb.weight.data[:old_vocab_size] = old_emb.weight.data
    # 新增行初始化：常用两种：均值 / 高斯
    new_emb.weight.data[old_vocab_size:] = old_emb.weight.data.mean(dim=0, keepdim=True)
    model.embed = new_emb.to(device)

    # ========= 扩容 lm_head（非常容易漏掉！！） =========
    old_head = model.fc
    new_head = nn.Linear(old_head.in_features, new_vocab_size, bias=old_head.bias is not None)
    new_head.weight.data[:old_vocab_size] = old_head.weight.data
    if old_head.bias is not None:
        new_head.bias.data[:old_vocab_size] = old_head.bias.data
    # 新增部分初始化
    new_head.weight.data[old_vocab_size:] = old_head.weight.data.mean(dim=0, keepdim=True)
    if old_head.bias is not None:
        new_head.bias.data[old_vocab_size:] = old_head.bias.data.mean()
    model.fc = new_head.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = -100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    pad_id = char2idx["<|pad|>"]
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, pad_id, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, pad_id, train=False)

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
