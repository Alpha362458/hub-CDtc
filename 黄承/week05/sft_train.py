# -*- coding: utf-8 -*-
"""
基于 Transformer 的单向（因果）语言模型，用于文本生成。

流程：
    1. 读取语料，构建字符级词表。
    2. 构造 Decoder-only（causal self-attention）Transformer。
    3. 以“预测下一个字符”为目标训练（交叉熵损失）。
    4. 训练结束后自回归采样生成新文本。
"""

import json
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 数据加载与词表
# ---------------------------------------------------------------------------
def load_corpus(file: str) -> str:
    try:
        with open(file, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"加载训练文本失败：{e}")
        raise


def build_vocabs(text: str) -> tuple[dict, dict]:
    """构建字符 <-> 编号 映射，并加入 <pad>、<unk>。"""
    chars = sorted(set(text) | {"<pad>", "<unk>"})
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


def encode(text: str, char2idx: dict) -> list[int]:
    """字符串 -> id 列表，未知字符回退到 <unk>。"""
    unk = char2idx["<unk>"]
    return [char2idx.get(ch, unk) for ch in text]


def decode(ids: list[int], idx2char: dict) -> str:
    """id 列表 -> 字符串。"""
    return "".join(idx2char.get(i, "") for i in ids)


class CharDataset(Dataset):
    def __init__(self, text: str, char2idx: dict, seq_len: int, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride
        ids = [char2idx[ch] for ch in text if ch in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len) // self.stride)

    def __getitem__(self, idx):
        real_idx = idx * self.stride
        x = self.data[real_idx:real_idx + self.seq_len]
        y = self.data[real_idx + 1:real_idx + self.seq_len + 1]
        return x, y


def build_dataset(text: str, char2idx: dict, text_train_ratio: float,
                  seq_len: int, stride: int = 1) -> tuple[CharDataset, CharDataset]:
    train_size = int(len(text) * text_train_ratio)
    train_dataset = CharDataset(text[:train_size], char2idx, seq_len, stride)
    val_dataset = CharDataset(text[train_size:], char2idx, seq_len, stride)
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# SFT 数据（指令微调）
# ---------------------------------------------------------------------------
SFT_PROMPT = "指令：{instruction}\n回答："
SFT_EOS = "<eos>"
IGNORE_INDEX = -100


def load_sft_data(file: str) -> list[dict]:
    """读取 jsonl 格式的 SFT 数据，每行形如 {"instruction": ..., "output": ...}。"""
    items = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_sft_vocabs(text: str, items: list[dict]) -> tuple[dict, dict]:
    """在预训练词表基础上追加 <eos> 与 SFT 中新出现的字符。

    原有字符的编号保持不变（预训练权重可直接复用），新增字符追加在词表末尾。
    """
    chars = sorted(set(text) | {"<pad>", "<unk>"})
    seen = set(chars)

    extra = set()
    for item in items:
        extra.update(SFT_PROMPT.format(instruction=item["instruction"]))
        extra.update(item["output"])

    chars += [SFT_EOS] + sorted(extra - seen)
    char2idx = {ch: i for i, ch in enumerate(chars)}
    idx2char = {i: ch for ch, i in char2idx.items()}
    return char2idx, idx2char


class SFTDataset(Dataset):
    """把 (instruction, output) 转成 (输入序列, 标签序列)。

    只对“回答”部分计算损失：指令与模板部分的标签置为 IGNORE_INDEX。
    回答末尾追加 <eos>，让模型学会在回答结束后停止。
    """

    def __init__(self, items: list[dict], char2idx: dict, max_len: int):
        eos = char2idx[SFT_EOS]
        self.samples = []
        for item in items:
            prompt_ids = encode(SFT_PROMPT.format(instruction=item["instruction"]), char2idx)
            answer_ids = encode(item["output"], char2idx) + [eos]

            # logits[i] 预测的是 input[i+1]，因此标签要整体左移一位对齐：
            # 位置 len(prompt_ids)-1（"回答：" 的最后一个字符）的标签是回答的第 1 个字符。
            input_ids = (prompt_ids + answer_ids)[:max_len]
            labels = ([IGNORE_INDEX] * (len(prompt_ids) - 1) + answer_ids + [IGNORE_INDEX])[:max_len]
            if len(input_ids) > len(prompt_ids):  # 至少保留一个回答字符
                self.samples.append((input_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sft_collate_fn(batch, pad_id: int):
    """按 batch 内最长序列右侧补齐：输入补 pad，标签补 IGNORE_INDEX，mask 补 0。"""
    max_len = max(len(x) for x, _ in batch)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (x, y) in enumerate(batch):
        n = len(x)
        inputs[i, :n] = torch.tensor(x, dtype=torch.long)
        labels[i, :n] = torch.tensor(y, dtype=torch.long)
        mask[i, :n] = 1
    return inputs, labels, mask


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
class CharsConfig:
    def __init__(self, **kwargs):
        self.vocab_size = 100            # 训练时按实际词表覆盖
        self.hidden_size = 256
        self.layer_norm_eps = 1e-12
        self.hidden_dropout_prob = 0.1
        self.position_size = 256         # 位置编码最大长度
        self.segment_size = 2
        self.pad_token = 0
        self.num_attention_heads = 8
        self.attention_probs_dropout_prob = 0.1
        self.intermediate_size = 1024
        self.num_hidden_layers = 6
        self.initializer_range = 0.02

        for key, value in kwargs.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# 模型组件
# ---------------------------------------------------------------------------
class CharsEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.position_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.segment_size, config.hidden_size)

        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs, segments=None):
        seq_len = inputs.size(1)
        positions = torch.arange(seq_len, dtype=torch.long, device=inputs.device)
        positions = positions.unsqueeze(0).expand_as(inputs)

        if segments is None:
            segments = torch.zeros_like(inputs)

        words_emb = self.word_embeddings(inputs)
        position_emb = self.position_embeddings(positions)
        segment_emb = self.token_type_embeddings(segments)

        embedding = words_emb + position_emb + segment_emb
        embedding = self.layerNorm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class BertSelfAttention(nn.Module):
    """带因果掩码的多头自注意力：每个位置只能看到自身及之前的位置。"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.head_dim * self.num_heads
        assert self.all_head_size == config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch, seq_len, _ = hidden_states.size()

        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # 因果掩码：屏蔽未来位置
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=hidden_states.device)
        ).view(1, 1, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # padding 掩码（可选）
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(pad_mask == 0, float("-inf"))

        attn_prob = F.softmax(attn_scores, dim=-1)
        attn_prob = self.dropout(attn_prob)

        out = torch.matmul(attn_prob, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch, seq_len, self.all_head_size)
        return out


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        out = self.dropout(self.dense(hidden_states))
        out = self.layerNorm(out + input_tensor)
        return out


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, inputs, attention_mask=None):
        attention_output = self.attention(inputs, attention_mask)
        return self.output(attention_output, inputs)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        out = self.dropout(self.dense(hidden_states))
        out = self.layerNorm(out + input_tensor)
        return out


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attn_out = self.attention(hidden_states, attention_mask)
        inter_out = self.intermediate(attn_out)
        return self.output(inter_out, attn_out)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = CharsEmbedding(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoder_out = self.encoder(embedding_output, attention_mask)
        return encoder_out


# ---------------------------------------------------------------------------
# 单向语言模型（Decoder-only）
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BertModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 输入嵌入与输出头共享权重（权重绑定）
        self.lm_head.weight = self.model.embeddings.word_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden_states = self.model(input_ids, attention_mask, token_type_ids)
        return self.lm_head(hidden_states)  # (batch, seq_len, vocab_size)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=150, temperature=1.0, top_k=None):
        """自回归采样生成文本。

        参数：
            input_ids: 起始上下文，形状 (1, T)。
            max_new_tokens: 生成的新字符数量。
            temperature: 采样温度，越低越保守。
            top_k: 仅从概率最高的 k 个字符中采样（可选）。
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 超出位置编码范围时，只保留最后 position_size 个字符
            if input_ids.size(1) > self.config.position_size:
                input_ids = input_ids[:, -self.config.position_size:]

            logits = self(input_ids)[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# 训练与主程序
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_gpt.pth")
SFT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_train")
SFT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "char_gpt_sft.pth")


def save_checkpoint(model, epoch, path):
    """保存模型权重、配置与已训练轮数。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "epoch": epoch,
        },
        path,
    )


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # 兼容没有 weights_only 参数的旧版 torch
        return torch.load(path, map_location=device)


def load_checkpoint(path, device):
    """加载已保存的模型，返回 (模型, 已训练轮数)。"""
    ckpt = _torch_load(path, device)

    config = CharsConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("epoch", 0)


def load_pretrained_weights(model, path, device):
    """把预训练权重灌入（可能更大的）SFT 模型。

    词表变大导致 word_embeddings / lm_head 尺寸变化时，只复制重叠部分，
    新增字符对应的行保持模型初始化值。
    """
    state = _torch_load(path, device)["model_state_dict"]
    new_state = model.state_dict()

    for name, old in state.items():
        if name not in new_state:
            continue
        new = new_state[name]
        if old.shape == new.shape:
            new.copy_(old)
        else:
            # 仅第 0 维（词表 / 位置编码）可能变化
            n = min(old.shape[0], new.shape[0])
            new[:n].copy_(old[:n])


def save_sft_checkpoint(model, epoch, char2idx, idx2char, path):
    """SFT 检查点额外保存词表，推理时无需再从语料重建。"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.__dict__,
            "epoch": epoch,
            "char2idx": char2idx,
            "idx2char": idx2char,
        },
        path,
    )


def train_model(model, train_loader, val_loader, vocab_size, epochs, lr, device, print_every=50, start_epoch=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    num_batches = len(train_loader)
    end_epoch = start_epoch + epochs
    print(f"开始训练: 第 {start_epoch + 1}~{end_epoch} 轮, lr={lr}, 每轮 batch 数={num_batches}\n")

    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        total_loss = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 每隔一定 batch 打印一次进度
            if step % print_every == 0 or step == num_batches:
                avg_loss = total_loss / step
                print(f"  epoch {epoch:3d} | step {step:5d}/{num_batches:5d} "
                      f"| loss {loss.item():.4f} | avg loss {avg_loss:.4f}")

        train_loss = total_loss / max(1, num_batches)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))

        print(f"==> epoch {epoch:3d}/{end_epoch} 完成 | train loss {train_loss:.4f} | val loss {val_loss:.4f}\n")

    return end_epoch


def train_sft(model, train_loader, val_loader, vocab_size, epochs, lr, device, print_every=20):
    """指令微调：只在回答部分的标签上计算交叉熵。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_batches = len(train_loader)
    print(f"开始 SFT 训练: {epochs} 轮, lr={lr}, 每轮 batch 数={num_batches}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (x, y, mask) in enumerate(train_loader, start=1):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, attention_mask=mask)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                   ignore_index=IGNORE_INDEX)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % print_every == 0 or step == num_batches:
                print(f"  sft epoch {epoch:2d} | step {step:4d}/{num_batches:4d} "
                      f"| loss {loss.item():.4f} | avg loss {total_loss / step:.4f}")

        train_loss = total_loss / max(1, num_batches)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                logits = model(x, attention_mask=mask)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1),
                                       ignore_index=IGNORE_INDEX)
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))

        print(f"==> sft epoch {epoch:2d}/{epochs} 完成 | train loss {train_loss:.4f} | val loss {val_loss:.4f}\n")

    return epochs


def build_sft_prompt(instruction: str) -> str:
    return SFT_PROMPT.format(instruction=instruction)


def sft_generate(model, instruction, char2idx, idx2char, device, max_new_tokens=100,
                 temperature=0.8, top_k=40):
    """按 SFT 模板拼接指令并生成回答，遇到 <eos> 即截断。"""
    model.eval()
    prompt = build_sft_prompt(instruction)
    prompt_ids = encode(prompt, char2idx)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    answer_ids = out[0].tolist()[len(prompt_ids):]
    return decode(answer_ids, idx2char).split(SFT_EOS)[0]


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据
    text = load_corpus("corpus.txt")
    char2idx, idx2char = build_vocabs(text)
    vocab_size = len(char2idx)

    seq_len = 128
    batch_size = 64
    train_dataset, val_dataset = build_dataset(text, char2idx, 0.9, seq_len, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 2. 模型（已存在则加载上次权重，继续训练）
    if os.path.exists(MODEL_PATH):
        model, start_epoch = load_checkpoint(MODEL_PATH, device)
        print(f"已加载模型 {MODEL_PATH}，上次训练到第 {start_epoch} 轮，继续训练")
    else:
        config = CharsConfig(vocab_size=vocab_size, position_size=seq_len)
        model = GPT(config).to(device)
        start_epoch = 0
        print("未找到已有模型，从头开始训练")
    print(f"词表大小: {vocab_size} | 参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)} | 设备: {device}\n")

    # 3. 训练
    # end_epoch = train_model(model, train_loader, val_loader, vocab_size,
    #                         epochs=2, lr=3e-4, device=device, start_epoch=start_epoch)

    # # 训练完成后保存模型
    # save_checkpoint(model, end_epoch, MODEL_PATH)
    # print(f"模型已保存到 {MODEL_PATH}（累计训练 {end_epoch} 轮）\n")

    # 4. 生成
    prompt = text[:20]
    start_ids = torch.tensor([encode(prompt, char2idx)], dtype=torch.long, device=device)

    print("\n========== 生成结果 ==========")
    print(f"起始上下文: {prompt!r}\n")
    for temperature, top_k in [(0.8, 40), (1.0, None), (0.6, 10)]:
        out = model.generate(start_ids, max_new_tokens=150, temperature=temperature, top_k=top_k)
        print(f"--- temperature={temperature}, top_k={top_k} ---")
        print(decode(out[0].tolist(), idx2char))
        print()

    # 5. SFT 指令微调（在预训练权重基础上）
    sft_train_file = os.path.join(SFT_DIR, "sft_train.jsonl")
    sft_val_file = os.path.join(SFT_DIR, "sft_val.jsonl")
    if not (os.path.exists(sft_train_file) and os.path.exists(sft_val_file)):
        print("未找到 SFT 数据，跳过指令微调")
        return

    sft_items = load_sft_data(sft_train_file)
    sft_val_items = load_sft_data(sft_val_file)
    sft_char2idx, sft_idx2char = build_sft_vocabs(text, sft_items + sft_val_items)

    sft_config = CharsConfig(**model.config.__dict__)
    sft_config.vocab_size = len(sft_char2idx)
    sft_model = GPT(sft_config).to(device)
    load_pretrained_weights(sft_model, MODEL_PATH, device)

    print("========== SFT 指令微调 ==========")
    print(f"SFT 词表大小: {sft_config.vocab_size}"
          f"（较预训练新增 {sft_config.vocab_size - vocab_size} 个字符，含 {SFT_EOS}）")

    pad_id = sft_char2idx["<pad>"]
    sft_collate = partial(sft_collate_fn, pad_id=pad_id)
    sft_train_set = SFTDataset(sft_items, sft_char2idx, sft_config.position_size)
    sft_val_set = SFTDataset(sft_val_items, sft_char2idx, sft_config.position_size)
    sft_train_loader = DataLoader(sft_train_set, batch_size=32, shuffle=True, collate_fn=sft_collate)
    sft_val_loader = DataLoader(sft_val_set, batch_size=32, shuffle=False, collate_fn=sft_collate)
    print(f"SFT 训练样本: {len(sft_train_set)} | 验证样本: {len(sft_val_set)}\n")

    sft_epochs = train_sft(sft_model, sft_train_loader, sft_val_loader,
                           sft_config.vocab_size, epochs=3, lr=1e-4, device=device)
    save_sft_checkpoint(sft_model, sft_epochs, sft_char2idx, sft_idx2char, SFT_MODEL_PATH)
    print(f"SFT 模型已保存到 {SFT_MODEL_PATH}（累计微调 {sft_epochs} 轮）\n")

    # 6. 指令推理示例
    print("========== SFT 指令推理示例 ==========")
    demo_instructions = [
        "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 12 数字2: 87",
        "为以下句子生成一个标题。 2024年公司利润同比增长20%",
    ]
    for instruction in demo_instructions:
        answer = sft_generate(sft_model, instruction, sft_char2idx, sft_idx2char, device)
        print(f"指令: {instruction}")
        print(f"回答: {answer}\n")


if __name__ == "__main__":
    main()
