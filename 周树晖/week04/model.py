"""仅含因果自注意力的 Transformer，逐位置输出下一字符 logits。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.head_dim, self.dropout = n_heads, d_model // n_heads, dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)]
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.drop(self.proj(h.transpose(1, 2).contiguous().reshape(b, t, d)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config["d_model"]
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalAttention(d, config["n_heads"], config["dropout"])
        self.ffn = nn.Sequential(nn.Linear(d, config["d_ff"]), nn.GELU(),
                                 nn.Linear(config["d_ff"], d), nn.Dropout(config["dropout"]))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_embedding = nn.Embedding(config["block_size"], config["d_model"])
        self.drop = nn.Dropout(config["dropout"])
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config["n_layers"])))
        self.norm = nn.LayerNorm(config["d_model"])
        self.head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.head.weight = self.token_embedding.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "ffn.2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config["n_layers"]))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if not 0 < ids.shape[1] <= self.config["block_size"]:
            raise ValueError("输入长度必须处于 1 到 block_size 之间")
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = self.drop(self.token_embedding(ids) + self.position_embedding(pos))
        logits = self.head(self.norm(self.blocks(x)))
        # PAD 仅供末块补齐，不是正常字符，也不参与输出分布。
        logits[..., 0] = torch.finfo(logits.dtype).min
        return logits


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TransformerLM(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint
