"""
Transformer 语言模型（GPT 式 decoder-only），默认直接使用 PyTorch 官方 API:
    nn.TransformerEncoderLayer / nn.TransformerEncoder

官方类内部做的事（注释里逐处指出，逐行拆开的版本见 model_handwritten.py）:
    TransformerEncoderLayer = 注意力子层 + 前馈子层 + pre-norm 残差
    nn.TransformerEncoder   = 把 N 层摞起来 + 最终 LayerNorm
两版前向数值等价（最大误差 ~1e-6，浮点噪声级），train.py --impl 可切换。

整体数据流（B=批大小, T=序列长度, V=词表大小）:
    (B,T) 字符id → Embedding ×√d_model → +正弦位置编码 → N×EncoderLayer
    → LayerNorm → Linear → (B,T,V)，每个位置 = 看过左侧全部上文后对下一字的预测。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    正弦位置编码：用不同频率的 sin/cos 给每个位置生成固定向量。

    自注意力本身对顺序不敏感（打乱输入顺序、输出跟着打乱），没有位置信息
    就无法区分"我爱你"和"你爱我"。选正弦而非可学习的位置表，是因为它按需
    计算、对任意长度可用 —— 推理（拼音输入法）时上文长度无上界。
    偶数维 sin、奇数维 cos，频率从 1 到 1/10000 几何递减：低维变化快
    （区分相邻位置），高维变化慢（区分远距离），类比钟表秒针/分针/时针。
    """

    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.drop = nn.Dropout(dropout)
        # 每一对 (sin, cos) 共用的频率 10000^(-2i/d_model)，构造时固定下来
        self.register_buffer("div_term", torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)))

    def forward(self, x):
        # x: (B, T, d_model)，位置从 0 开始编号
        T = x.size(1)
        pos = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(1)  # (T,1)
        pe = torch.zeros(T, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(pos * self.div_term)
        pe[:, 1::2] = torch.cos(pos * self.div_term)
        return self.drop(x + pe.unsqueeze(0))


class TransformerLM(nn.Module):
    """
    官方 API 组装的 GPT 式 decoder-only 语言模型。

    词嵌入表示"这个字是什么"，位置编码表示"它在第几位"，相加后进入编码器。
    embedding 乘 √d_model 是让缩放后与位置编码量级匹配的经典做法（配 0.02
    小方差初始化，初始 logits 接近 0，开局 loss 稳定在 ln(词表) 附近）。
    """

    def __init__(self, vocab_size, d_model, nhead, num_layers, ff_dim, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.pos = PositionalEncoding(d_model, dropout)
        # 一层 = 自注意力子层 + 前馈子层，各配 pre-norm 残差
        # （norm_first=True 即 pre-norm，GPT-2 风格，从头训练比 post-norm 稳）
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        # norm 参数 = 全部层之后的最终 LayerNorm（手写版里的 ln_f）
        # enable_nested_tensor 只在 post-norm 下生效，pre-norm 显式关掉避免告警
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, T) 字符 id。因果 mask 为上三角布尔矩阵（True = 禁止注意），
        # 位置 t 看不到 t 之后的信息 —— 与 RNN/LSTM 的自回归口径一致，PPL 可
        # 直接对比。训练样本全部等长（drop_last=True），无 padding mask。
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.pos(self.embed(x) * math.sqrt(self.d_model))
        out = self.encoder(h, mask=causal_mask, is_causal=True)
        return self.fc(out)   # (B, T, V)


def build_lm(vocab_size, d_model, nhead, num_layers, ff_dim, dropout=0.3, impl="official"):
    """按 impl 构建模型：official = nn.TransformerEncoder（默认），
    handwritten = model_handwritten 里的逐组件手写版。两版前向数值等价，
    但 state_dict 键名不同，checkpoint 会记录 impl 以便恢复对应结构。"""
    if impl == "handwritten":
        from model_handwritten import TransformerLM as HandwrittenLM
        return HandwrittenLM(vocab_size, d_model, nhead, num_layers, ff_dim, dropout)
    return TransformerLM(vocab_size, d_model, nhead, num_layers, ff_dim, dropout)


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, n_chars=50,
             temperature=1.0, top_k=0, device=None):
    """
    自回归生成：从 prompt 出发逐字采样，每步把"prompt + 已生成"整体作为
    上文重新前向，取最后位置的分布采样一个字，循环往复。

    temperature 对 logits 做除法：< 1 放大好坏差距 → 保守；> 1 压平 → 随机。
    top_k > 0 时只在概率前 k 大的候选里采样，砍掉长尾低质候选。
    返回完整文本（词表外的 prompt 字符原样保留但不参与预测）。
    """
    device = device or next(model.parameters()).device
    ids = [char2idx[c] for c in prompt if c in char2idx]
    out = "".join(c for c in prompt if c in char2idx)
    for _ in range(n_chars):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            kth = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        ids.append(nxt)
        out += idx2char[nxt]
    return out
