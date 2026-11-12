import argparse
import json
import math
import os
import random
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------------------
# 0. 特殊符号与模板
# --------------------------------------------------------------------------------------
# ★ 顺序就是 id：只在末尾追加！改动前面的顺序会让旧 checkpoint 的 id 整体错位。
#   原来的 5 个保持 4000..4004 不动，<unk> 追加到 4005。
SPECIALS = ["<pad>", "<bos>", "<eos>", "<|user|>", "<|assistant|>", "<unk>"]
PAD, BOS, EOS, USER, ASSISTANT, UNK = SPECIALS

NHEAD = 8          # args 里没存 nhead，用预训练语料 loss 反推：8 头 + gelu + pre-LN 最优
NORM_FIRST = True  # 你的 TransformerEncoderLayer 是 pre-LN
ACT = "gelu"       # 默认是 relu，你训练时显式传了 gelu
IGNORE_INDEX = -100


def build_prompt(instruction):
    """模型看到的问题部分：<bos><|user|>问题<eos><|assistant|>"""
    return f"{BOS}{USER}{instruction}{EOS}{ASSISTANT}"


# --------------------------------------------------------------------------------------
# 1. 分词器：char-level + 6 个特殊符号（特殊符号要整体切成 1 个 id）
# --------------------------------------------------------------------------------------
class Tokenizer:
    def __init__(self, char2idx, idx2char):
        self.char2idx = char2idx
        self.idx2char = {int(k): v for k, v in idx2char.items()}
        self.vocab_size = len(char2idx)
        self.pad_id = char2idx[PAD]
        self.bos_id = char2idx[BOS]
        self.eos_id = char2idx[EOS]
        self.user_id = char2idx[USER]
        self.assistant_id = char2idx[ASSISTANT]
        self.unk_id = char2idx[UNK]
        # 先匹配特殊符号再按字切分，否则 "<|user|>" 会被拆成 8 个字符
        self._special_re = re.compile("|".join(re.escape(s) for s in SPECIALS))

    def _char_ids(self, text):
        """普通文本查表，词表外的字 → <unk>（不再丢弃，长度和原文一一对应）"""
        return [self.char2idx.get(ch, self.unk_id) for ch in text]

    def encode_with_oov(self, text):
        """返回 (ids, oov)，oov[i]=True 表示 ids[i] 是 <unk>（原文里是个词表外的字）"""
        ids, oov, pos = [], [], 0
        for m in self._special_re.finditer(text):
            part = self._char_ids(text[pos:m.start()])
            ids.extend(part)
            oov.extend([i == self.unk_id for i in part])
            ids.append(self.char2idx[m.group(0)])
            oov.append(False)                       # 特殊符号永远不是 <unk>
            pos = m.end()
        part = self._char_ids(text[pos:])
        ids.extend(part)
        oov.extend([i == self.unk_id for i in part])
        return ids, oov

    def encode(self, text):
        return self.encode_with_oov(text)[0]

    def decode(self, ids, skip_special=False):
        out = []
        for i in ids:
            s = self.idx2char.get(int(i), "")
            if skip_special and s in SPECIALS:
                continue
            out.append(s)
        return "".join(out)


# --------------------------------------------------------------------------------------
# 2. 模型：结构 = 你的预训练模型，只是词表变成 4006
# --------------------------------------------------------------------------------------
class MiniLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, d_model, nhead, num_layers,
                 dim_ff, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation=ACT, batch_first=True, norm_first=NORM_FIRST,
        )
        # enable_nested_tensor=False：pre-LN 下本来也用不了，关掉省得警告
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                 enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.register_buffer("_causal", torch.empty(0), persistent=False)

    def _causal_mask(self, L, device):
        """上三角为 True：位置 t 看不到 t+1 及以后"""
        if self._causal.numel() != L * L or self._causal.device != device:
            self._causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool),
                                      diagonal=1)
        return self._causal

    def forward(self, idx, padding_mask=None):
        """idx: (B,T)  padding_mask: (B,T) True=pad  →  (B,T,V) logits"""
        B, T = idx.shape
        x = self.proj(self.embed(idx))
        x = x + self.pos_emb(torch.arange(T, device=idx.device))[None]
        x = self.transformer(x, mask=self._causal_mask(T, idx.device),
                             src_key_padding_mask=padding_mask)
        return self.fc(self.norm(x))

    @torch.no_grad()
    def generate(self, tokenizer, instruction, max_new_tokens=150, temperature=0.8,
                 top_k=20, greedy=False, repetition_penalty=1.15, recent_window=48):
        """输入问题，遇到 <eos> 停"""
        self.eval()
        device = next(self.parameters()).device
        ids = tokenizer.encode(build_prompt(instruction))
        x = torch.tensor([ids], device=device)

        out = []
        # 这些 token 不该出现在"回答"里：<pad> 无意义，<bos>/<|user|> 是模板，
        # <unk> 更是没人想让模型开口就说"我不认识这个字"。直接禁掉。
        banned = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.user_id, tokenizer.unk_id]
        for _ in range(max_new_tokens):
            x_in = x[:, -self.block_size:]
            logits = self(x_in)[0, -1].float()
            logits[banned] = -float("inf")

            # 重复惩罚：小模型容易"人工智能和工智能和工智能"死循环
            if repetition_penalty and repetition_penalty > 1.0:
                recent = torch.tensor(sorted(set(x[0, -recent_window:].tolist())), device=device)
                vals = logits[recent]
                logits[recent] = torch.where(vals > 0, vals / repetition_penalty,
                                             vals * repetition_penalty)
                logits[banned] = -float("inf")   # 惩罚可能把 -inf 变成 nan，重新压一遍

            if greedy:
                nxt = int(logits.argmax())
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k and top_k < logits.numel():
                    kth = torch.topk(logits, top_k).values[-1]
                    logits[logits < kth] = -float("inf")
                nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

            if nxt == tokenizer.eos_id:      # 特殊符号的好处：停机就是"一个 token"的事
                break
            out.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

        return tokenizer.decode(out, skip_special=True).strip()


# --------------------------------------------------------------------------------------
# 3. 数据：只对"回答 + <eos>"算 loss
# --------------------------------------------------------------------------------------
class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                instruction = str(d.get("instruction", "")).strip()
                extra = str(d.get("input", "")).strip()
                if extra:                                    # 兼容 alpaca 风格
                    instruction = instruction + "\n" + extra
                output = str(d.get("output", "")).strip()
                if instruction and output:
                    self.items.append((instruction, output))

        self.examples = [e for e in (self._build(i, o) for i, o in self.items) if e]

    def _build(self, instruction, output):
        tok = self.tokenizer
        # 分两段，才能精确控制从哪个 token 开始算 loss
        head, _ = tok.encode_with_oov(f"{BOS}{USER}{instruction}{EOS}{ASSISTANT}")  # 不算 loss
        body, body_oov = tok.encode_with_oov(f"{output}{EOS}")                      # 算 loss（含 eos）
        if not body:
            return None

        reserve = min(len(body), 32)                 # 至少给回答留 32 个位置
        if len(head) > self.block_size - reserve:
            head = head[-(self.block_size - reserve):]
        keep = self.block_size - len(head)
        body, body_oov = body[:keep], body_oov[:keep]   # oov 标记要跟着一起截断
        if not body:
            return None

        input_ids = head + body
        # ★ SFT 的全部秘密：模板/问题一律 -100，回答（含 eos）才算 loss。
        #   回答里词表外的字也是 -100 —— 模型学不会"凭空生成一个不存在的字"，
        #   让它去猜只会在生成时到处吐 <unk>，不如直接不计算这些位置的 loss。
        #   ⚠ body 本身（input_ids 用的）必须是真 id，-100 只能出现在 labels 里，
        #     否则 embedding 会拿到 -100 这个非法下标，CUDA 直接 assert 崩溃。
        body_labels = [IGNORE_INDEX if is_oov else i for i, is_oov in zip(body, body_oov)]
        labels = [IGNORE_INDEX] * len(head) + body_labels
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate(pad_id):
    """右侧补 <pad>；padding_mask 里 True 表示该位置忽略"""
    def collate(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, mask = [], [], []
        for b in batch:
            n_pad = max_len - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n_pad)
            labels.append(b["labels"] + [IGNORE_INDEX] * n_pad)
            mask.append([False] * len(b["input_ids"]) + [True] * n_pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "padding_mask": torch.tensor(mask, dtype=torch.bool),
        }
    return collate


def lm_loss(logits, labels):
    """错位：位置 t 的 logits 预测 labels[t+1]；-100 不计入 loss"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1), ignore_index=IGNORE_INDEX, reduction="sum")
    n_tokens = (shift_labels != IGNORE_INDEX).sum().clamp(min=1)
    with torch.no_grad():
        correct = ((shift_logits.argmax(-1) == shift_labels)
                   & (shift_labels != IGNORE_INDEX)).sum()
    return loss / n_tokens, n_tokens.item(), correct.item()


# --------------------------------------------------------------------------------------
# 4. 训练 / 评估
# --------------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, log_every=20):
    model.train()
    tot_loss, tot_tok, tot_ok = 0.0, 0, 0
    for step, batch in enumerate(loader, 1):
        logits = model(batch["input_ids"].to(device),
                       padding_mask=batch["padding_mask"].to(device))
        loss, n_tok, n_ok = lm_loss(logits, batch["labels"].to(device))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tot_loss += loss.item() * n_tok
        tot_tok += n_tok
        tot_ok += n_ok
        if step % log_every == 0 or step == len(loader):
            print(f"    step {step:4d}/{len(loader)}  loss {tot_loss / tot_tok:.4f}  "
                  f"answer-acc {tot_ok / tot_tok:.3f}  lr {scheduler.get_last_lr()[0]:.2e}")
    return tot_loss / tot_tok, tot_ok / tot_tok


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_loss, tot_tok, tot_ok = 0.0, 0, 0
    for batch in loader:
        logits = model(batch["input_ids"].to(device),
                       padding_mask=batch["padding_mask"].to(device))
        loss, n_tok, n_ok = lm_loss(logits, batch["labels"].to(device))
        tot_loss += loss.item() * n_tok
        tot_tok += n_tok
        tot_ok += n_ok
    return tot_loss / max(tot_tok, 1), tot_ok / max(tot_tok, 1)


# --------------------------------------------------------------------------------------
# 5. 加载 checkpoint：老权重 + 扩词表
# --------------------------------------------------------------------------------------
def build_model(char2idx, args):
    d_model = int(args.get("hidden_dim", 256))
    return MiniLM(
        vocab_size=len(char2idx),
        embed_dim=int(args.get("embed_dim", 128)),
        d_model=d_model,
        nhead=NHEAD,
        num_layers=int(args.get("num_layers", 2)),
        dim_ff=4 * d_model,
        block_size=int(args.get("seq_len", 512)),
        dropout=float(args.get("dropout", 0.1)),
    )


@torch.no_grad()
def expand_vocab(state, old_vocab, new_vocab):
    """embed / fc 从 4000 行扩到新词表大小：老行照搬，新行 = 均值 + 小噪声"""
    out = dict(state)
    for key in ("embed.weight", "fc.weight", "fc.bias"):
        if key not in out:
            continue
        w = out[key]
        if w.size(0) >= new_vocab:
            continue
        if w.dim() == 1:                                    # bias：新行置 0
            new = torch.zeros(new_vocab, dtype=w.dtype)
            new[:w.size(0)] = w
        else:                                               # 权重：均值 + 噪声
            new = w[:old_vocab].mean(0, keepdim=True).repeat(new_vocab, 1)
            new[:old_vocab] = w
            new[old_vocab:] += 0.02 * torch.randn_like(new[old_vocab:])
        out[key] = new
    return out


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get("model_state") or ck.get("model_state_dict") or ck
    char2idx = dict(ck["char2idx"])
    idx2char = {int(k): v for k, v in ck["idx2char"].items()}
    args = ck.get("args", {})

    # 老 checkpoint 没有特殊符号 → 追加到词表末尾（id 4000..4004）
    old_vocab = len(char2idx)
    added = []
    for s in SPECIALS:
        if s not in char2idx:
            i = len(char2idx)
            char2idx[s] = i          # 字符串 → 新 id
            idx2char[i] = s          # 新 id → 字符串
            added.append(s)
    new_vocab = len(char2idx)

    model = build_model(char2idx, args)
    if added:
        print(f"词表扩容 {old_vocab} → {new_vocab}：新增 {added}")
        state = expand_vocab(state, old_vocab, new_vocab)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={missing}\n[warn] unexpected={unexpected}")
    model.to(device)
    return model, Tokenizer(char2idx, idx2char), args


def save_checkpoint(path, model, tokenizer, args, extra=None):
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": tokenizer.char2idx,
        "idx2char": tokenizer.idx2char,
        "specials": SPECIALS,
        "args": {**args, **(extra or {})},
    }, path)


# --------------------------------------------------------------------------------------
# 6. 主流程
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="best_model_transformer.pt")
    ap.add_argument("--train-file", default="sft_train.jsonl")
    ap.add_argument("--val-file", default="sft_val.jsonl")
    ap.add_argument("--save", default="sft_model.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--portrait", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device = {device}")

    # ---------- 聊天 ----------
    if args.chat:
        if not os.path.exists(args.save):
            raise SystemExit(f"[!] 找不到 {args.save}，先跑 python sft.py")
        model, tok, _ = load_checkpoint(args.save, device)
        print("输入问题，回车发送；:q 退出")
        while True:
            try:
                q = input("\n你  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q in (":q", "quit", "exit"):
                break
            if q:
                print(f"模型 > {model.generate(tok, q)}")
        return

    # ---------- 训练 ----------
    model, tokenizer, base_args = load_checkpoint(args.base, device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"{n_param / 1e6:.2f}M 参数 | 词表 {tokenizer.vocab_size} "
          f"(pad={tokenizer.pad_id} bos={tokenizer.bos_id} eos={tokenizer.eos_id} "
          f"user={tokenizer.user_id} assistant={tokenizer.assistant_id}) | "
          f"block {model.block_size}")

    train_set = SFTDataset(args.train_file, tokenizer, model.block_size)
    val_set = SFTDataset(args.val_file, tokenizer, model.block_size)
    print(f"训练集 {len(train_set)} 条 / 验证集 {len(val_set)} 条")

    # 看一眼模板和 mask 的位置
    ex = train_set[0]
    print("\n[样例检查] 完整序列（特殊符号按文字显示）:")
    print("  " + tokenizer.decode(ex["input_ids"]).replace("\n", "\\n"))
    print("  " + "".join("_" if y == IGNORE_INDEX else "A" for y in ex["labels"]))
    print("  ↑ _ = 模板/问题（只做输入），A = 回答 + <eos>（算 loss）")
    print("  注意：空格等词表外的字现在显示成 <unk>，位置和原文一一对应\n")

    # 统计一下 <unk> 占多少：输入里可以放 <unk>，但要清楚它丢掉了多少信息
    n_unk = sum(1 for ex in train_set.examples for i in ex["input_ids"] if i == tokenizer.unk_id)
    n_all = sum(len(ex["input_ids"]) for ex in train_set.examples)
    n_unk_label = sum(1 for ex in train_set.examples for y in ex["labels"] if y == tokenizer.unk_id)
    n_lab = sum(1 for ex in train_set.examples for y in ex["labels"] if y != IGNORE_INDEX)
    print(f"[<unk>] 输入 token 里占 {n_unk}/{n_all} = {100 * n_unk / max(n_all, 1):.2f}%；"
          f"回答 token 里占 {n_unk_label}/{n_lab} = {100 * n_unk_label / max(n_lab, 1):.2f}%"
          f"（这些位置的 loss 已被 -100 跳过）\n")

    collate = make_collate(tokenizer.pad_id)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)

    base_loss, base_acc = evaluate(model, val_loader, device)
    print(f"[基线] 预训练模型 + 全新随机初始化的特殊符号: "
          f"val_loss {base_loss:.4f} (ppl {math.exp(base_loss):.1f}) / acc {base_acc:.3f}")
    print("      （很高是正常的：<|assistant|> 这类 token 此刻对它毫无意义）")

    demo_qs = [val_set.items[i][0] for i in range(min(args.portrait, len(val_set)))]
    print("=" * 70)
    print("SFT 之前:")
    for q in demo_qs:
        print(f"  问：{q[:40]}")
        print(f"  答：{model.generate(tokenizer, q, max_new_tokens=60, greedy=True)}")
    print("=" * 70)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n--- epoch {epoch}/{args.epochs} ---")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})
        print(f"  epoch {epoch}  train_loss {tr_loss:.4f} (ppl {math.exp(tr_loss):.1f}) | "
              f"val_loss {va_loss:.4f} (ppl {math.exp(va_loss):.1f}) | "
              f"val answer-acc {va_acc:.3f} | {time.time() - t0:.1f}s")

        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(args.save, model, tokenizer, base_args,
                            {"sft": True, "sft_val_loss": va_loss, "sft_history": history})
            print(f"  ✓ 保存 → {args.save}")

    model, tokenizer, _ = load_checkpoint(args.save, device)
    print("\n" + "=" * 70)
    print(f"SFT 之后（best val_loss = {best_val:.4f}，基线 {base_loss:.4f}）:")
    for q in demo_qs:
        print(f"  问：{q[:40]}")
        print(f"  答：{model.generate(tokenizer, q, max_new_tokens=150, greedy=True)}")
    print("=" * 70)
    print("\n完成。聊天：python sft.py --chat")


if __name__ == "__main__":
    main()
