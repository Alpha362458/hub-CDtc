"""参照 week04 的字符级因果 Transformer；不导入或修改 week04 文件。"""

import json
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
IGNORE_INDEX = -100
MODEL_KEYS = ("embed_dim", "hidden_dim", "num_layers", "num_heads", "seq_len", "dropout")


def format_prompt(instruction):
    return f"用户：{instruction}\n助手："


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是有效 JSON") from exc
            if not isinstance(row, dict) or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("instruction", "output")
            ):
                raise ValueError(f"{path}:{line_number} 必须有非空 instruction 和 output")
            records.append(row)
    if not records:
        raise ValueError(f"{path} 没有有效样本")
    return records


def encode(text, vocab):
    return [vocab.get(char, vocab[UNK]) for char in text]


def extend_vocab(original, records):
    vocab = dict(original)
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("检查点中的字符编号必须从 0 开始且连续")
    # 原有字符编号不变；只从训练集补充字符，验证集未知字符映射到 UNK。
    for token in (PAD, UNK, EOS):
        if token not in vocab:
            vocab[token] = len(vocab)
    chars = set("".join(format_prompt(r["instruction"]) + r["output"] for r in records))
    for char in sorted(chars):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


class LM(nn.Module):
    """与 week04 权重键名兼容的自回归 Transformer。"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_heads, seq_len, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.position_embed = nn.Embedding(seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight)
            nn.init.xavier_uniform_(block.linear2.weight)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        length = x.size(1)
        if not 1 <= length <= self.position_embed.num_embeddings:
            raise ValueError("输入长度必须在 1 到 seq_len 之间")
        positions = torch.arange(length, device=x.device)
        e = self.drop(self.embed(x) + self.position_embed(positions))
        # 批次右侧补齐，因果掩码保证有效位置看不到后面的 PAD。
        out = self.transformer(e, mask=self.causal_mask[:length, :length])
        return self.fc(self.norm(out))


def init_from_pretrained(checkpoint, records, seq_len, dropout):
    vocab = extend_vocab(checkpoint["char2idx"], records)
    config = {key: checkpoint["args"][key] for key in MODEL_KEYS}
    if seq_len < config["seq_len"]:
        raise ValueError("SFT 上下文不能小于预训练上下文长度")
    config.update(seq_len=seq_len, dropout=dropout)
    model = LM(vocab_size=len(vocab), **config)
    state = model.state_dict()
    expandable = {"embed.weight", "fc.weight", "fc.bias", "position_embed.weight"}
    # 新字符和新位置保持随机初始化，旧权重按原编号完整复制。
    with torch.no_grad():
        for name, old in checkpoint["model_state"].items():
            if name in expandable:
                target = state[name]
                if old.shape[1:] != target.shape[1:] or old.shape[0] > target.shape[0]:
                    raise ValueError(f"检查点形状不兼容：{name}")
                target[:old.shape[0]].copy_(old)
            else:
                state[name].copy_(old)
    model.load_state_dict(state, strict=True)
    return model, vocab, config


def get_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checked_output(path, protected=()):
    path = Path(path).expanduser().resolve()
    week04 = ROOT.parent / "week04"
    if path == week04 or week04 in path.parents:
        raise ValueError("不能向 week04 写入文件")
    sources = [ROOT / "sft_train.jsonl", ROOT / "sft_val.jsonl"]
    sources.extend(Path(p).resolve() for p in protected)
    if path in sources:
        raise ValueError("输出文件不能覆盖数据或输入检查点")
    return path
