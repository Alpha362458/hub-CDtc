"""
手写版 Transformer（逐组件实现），是 model.py 官方 API 的"拆开看"对照：
    MultiHeadAttention  ↔  nn.MultiheadAttention
    Block               ↔  nn.TransformerEncoderLayer（一层子层结构）
    TransformerLM       ↔  nn.TransformerEncoder 组装的整体
两版前向数值等价（同权重最大误差 ~1e-6）。train.py --impl handwritten
用本版训练，可与官方版对比 PPL 曲线；evaluate.py 取注意力权重也用本版
（官方 API 不暴露注意力矩阵，见 from_official）。

按本文件自上而下读注释，即是 Transformer 的学习路径。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PositionalEncoding


class MultiHeadAttention(nn.Module):
    """
    多头自注意力：让每个位置从所有（因果允许的）位置收集信息。

    核心公式 Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V：
      Q·Kᵀ   用"查询"和"键"算出任意两个位置的相关度
      /√d_k  点积的方差随维度 d_k 线性增长，不缩放的话大值进 softmax
             会接近 one-hot，梯度消失 —— √d_k 恰好把方差归一到 1
      softmax 把相关度归一成权重，再对"值"加权求和 ——
             本质是可微分的"按相关性查表"

    多头 = 把 d_model 维拆成 nhead 个 d_model/nhead 维子空间各自做注意力，
    不同头可以学到不同粒度的关系（相邻字搭配 / 长距离搭配），结果拼接后
    再过一个线性层融合。
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被头数整除"
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.store_attn = False   # 置 True 时保存最近一次注意力权重，供可视化
        self.last_attn = None     # (B, nhead, T, T)，已 detach
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        B, T, _ = x.shape
        # 三份线性投影后拆头: (B,T,d_model) → (B, nhead, T, d_k)
        q = self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)

        # 相关度打分 (B, nhead, T, T)；因果 mask 处填 -inf，
        # softmax 后权重恰为 0 —— 位置 t 看不到 t 之后的信息，自回归的关键
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)

        # 按权重聚合各位置的 v，拼回头维度，过输出投影融合各头信息
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """逐位置前馈层：升维 4 倍过 GELU 再降回。注意力负责跨位置混合信息，
    前馈层在每个位置内部独立做非线性变换，两者分工互补 —— 只有注意力的话
    模型本质上只是在做加权平均，表达能力不够。"""

    def __init__(self, d_model, ff_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """
    一层 Transformer 块 = 注意力子层 + 前馈子层，各配 pre-norm 残差连接。

    残差 x + f(norm(x))：子层学的是"对输入的修正量"而非全新表示，梯度可以
    经加法直通底层 —— 深层网络可训练的关键。pre-norm（先进 LayerNorm 再进
    子层）比原始论文的 post-norm 从头训练更稳，GPT 系列均采用。
    """

    def __init__(self, d_model, nhead, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, causal_mask):
        x = x + self.attn(self.ln1(x), causal_mask)   # 残差：子层输出加回输入
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """GPT 式 decoder-only 语言模型（手写组装），结构参数与官方版一致。"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.pos = PositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, nhead, ff_dim, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)   # 输出前的最终归一化
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, T) 字符 id；因果 mask 与官方版同一构造
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.pos(self.embed(x) * math.sqrt(self.d_model))
        for block in self.blocks:
            h = block(h, causal_mask)
        return self.fc(self.ln_f(h))   # (B, T, V)


def from_official(official):
    """把官方版 TransformerLM 的权重复制到等价的手写版并返回。

    官方 API（nn.TransformerEncoder）不暴露注意力矩阵，注意力可视化需要
    能保存权重的手写镜像。映射关系与等价性已核验：官方把 QKV 三个投影
    拼在 in_proj 里（顺序 Q,K,V），其余参数与手写版一一对应。
    """
    layer0 = official.encoder.layers[0]
    hw = TransformerLM(
        vocab_size=official.fc.out_features,   # 词表大小从输出层反推
        d_model=official.d_model,
        nhead=layer0.self_attn.num_heads,
        num_layers=len(official.encoder.layers),
        ff_dim=layer0.linear1.out_features,
    )
    with torch.no_grad():
        # embedding / 输出层逐项复制（位置编码无参数，同 d_model 下值相同）
        hw.embed.weight.copy_(official.embed.weight)
        hw.fc.weight.copy_(official.fc.weight)
        hw.fc.bias.copy_(official.fc.bias)
        for tl, blk in zip(official.encoder.layers, hw.blocks):
            q, k, v = tl.self_attn.in_proj_weight.chunk(3, dim=0)
            blk.attn.q_proj.weight.copy_(q); blk.attn.k_proj.weight.copy_(k); blk.attn.v_proj.weight.copy_(v)
            qb, kb, vb = tl.self_attn.in_proj_bias.chunk(3, dim=0)
            blk.attn.q_proj.bias.copy_(qb); blk.attn.k_proj.bias.copy_(kb); blk.attn.v_proj.bias.copy_(vb)
            blk.attn.out_proj.weight.copy_(tl.self_attn.out_proj.weight)
            blk.attn.out_proj.bias.copy_(tl.self_attn.out_proj.bias)
            blk.ffn.fc1.weight.copy_(tl.linear1.weight); blk.ffn.fc1.bias.copy_(tl.linear1.bias)
            blk.ffn.fc2.weight.copy_(tl.linear2.weight); blk.ffn.fc2.bias.copy_(tl.linear2.bias)
            blk.ln1.weight.copy_(tl.norm1.weight); blk.ln1.bias.copy_(tl.norm1.bias)
            blk.ln2.weight.copy_(tl.norm2.weight); blk.ln2.bias.copy_(tl.norm2.bias)
        hw.ln_f.weight.copy_(official.encoder.norm.weight)
        hw.ln_f.bias.copy_(official.encoder.norm.bias)
    # copy_ 跨设备只搬值不搬设备，镜像需要显式移到与原模型相同的设备上
    return hw.to(next(official.parameters()).device)
