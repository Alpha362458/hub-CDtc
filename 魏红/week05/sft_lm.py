# coding: utf-8
"""
Week05 作业：小型 SFT（有监督微调）—— 在 Week04 的 train_bert_lm.py 上改出来

Week04：字符级语言模型，给前面的字，预测下一个字。
Week05：同一个模型，换一份「问题 → 回答」的数据，教它「被问就答、答完就停」。

字表 CharVocabulary、因果性验证 CausalityChecker 原样沿用。
模型 TransformerLm 改成「先手写、旁边注释封装好的写法」：Week04 作业直接用 nn.TransformerEncoderLayer，
里面的注意力、残差、LayerNorm 都看不到；这里把它们一步一步写出来，并用 --compare 证明和封装版输出一致。
SFT 新增的是下面三处：

    ① 数据：不再是一大段连续文字切块，而是一条一条的 (问题, 回答)，拼成
            <user> 问题的字…… <assistant> 回答的字…… <eos>
       <user> / <assistant> / <eos> / <pad> 是四个新增的特殊符号，各占字表的一个编号。
    ② 损失：问题那一段的 labels 设为 -100（不算损失），只对回答和 <eos> 算损失。
            手动错位：位置 i 的输出，预测 labels 的第 i+1 个。
    ③ 续写：不再接着 prompt 往下写，而是先放 <user> 问题 <assistant>，再逐字生成，遇到 <eos> 就停。

代码结构（与 Week04 相同的职责拆分）：
    Config            设置                                   （改：去掉语料切块相关）
    CharVocabulary    字表                                   （原样）
    SpecialTokens     四个特殊符号的名字                     （新）
    SftRepository     读 jsonl                               （新，替代 CorpusRepository）
    SftDataset        一条 = (input_ids, labels)             （新，替代 CharLmDataset）
    BatchCollator     把一批不等长的样本补齐                 （新）
    MultiHeadSelfAttention / TransformerBlock / TransformerLm
                      模型，手写（旁边注释对应的封装写法）   （改）
    PackagedComparer  手写版 vs nn.TransformerEncoderLayer 对拍 （新）
    Trainer           训练一轮 / 验证一轮                    （改：损失换成错位 + 忽略 -100）
    CausalityChecker  因果性验证                             （原样）
    LabelInspector    打印 labels 对齐表                     （新）
    TextPredictor     回答问题                               （改）
    Program           入口                                   （改）

运行（在本文件所在目录）：
    python sft_lm.py --compare             # 手写的一层 vs 封装的 nn.TransformerEncoderLayer，输出是否一致
    python sft_lm.py --check               # 看 labels 对齐表、验证因果掩码
    python sft_lm.py --overfit             # 先在 32 条上过拟合，确认流程能拟合
    python sft_lm.py                       # 正式训练 + 独立验证集评估
    python sft_lm.py --generate "问题"     # 用训练好的模型回答
"""

import argparse                     # API 注释：Python 自带，解析命令行参数（--epochs 3 这种）
import math                         # API 注释：Python 自带，数学函数，这里用 math.exp 算困惑度
import os                           # API 注释：Python 自带，路径、文件夹相关操作
import random                       # API 注释：Python 自带的随机数模块（不是 PyTorch 的），这里用来打乱语料的行
import time                         # API 注释：Python 自带，time.time() 返回当前时间（秒），两次相减得到耗时
from collections import Counter     # API 注释：Counter 是「数数」专用的字典，Counter("abca") → {a: 2, b: 1, c: 1}
from dataclasses import dataclass, replace   # API 注释：@dataclass 装饰器，根据类里写的字段自动生成 __init__ 等方法；replace(对象, 字段=新值) 复制一份 frozen 对象并只改指定字段
from typing import Dict, List, Tuple    # API 注释：只用来写「类型标注」（如 List[int]），运行时不起作用，给人和编辑器看

import torch                        # API 注释：PyTorch 主库：张量（tensor）、自动求导、GPU 计算
import torch.nn as nn               # API 注释：神经网络模块：各种层（Linear 等）和损失函数；as nn 是起个短别名
from torch.utils.data import DataLoader, Dataset   # API 注释：Dataset = 「第 i 条数据是什么」的接口；DataLoader = 按批取出数据

import json                         # API 注释：Python 自带，json.loads(一行文字) 把 JSON 文本变成字典


# ═════════════════════════════ 设置 ═════════════════════════════
@dataclass(frozen=True)
class Config:
    """<summary>所有可调的设置。和 Week04 相比：去掉了语料切块、验证比例，加了训练/验证文件路径。</summary>"""
    train_path: str                     # 训练集 jsonl
    val_path: str                       # 验证集 jsonl
    save_path: str                      # 保存最好模型的路径
    epochs: int = 20                    # 数据只有 3000 条，多过几遍
    seq_len: int = 128                  # 一条序列最长 128 个字（数据里最长 114），也是位置编码的最大长度
    batch_size: int = 32                # 每批 32 条
    learning_rate: float = 1e-3         # 与 Week04 相同
    grad_clip: float = 1.0
    d_model: int = 192
    heads: int = 4
    layers: int = 4
    dropout: float = 0.1
    norm_first: bool = False            # False = 先残差再 LayerNorm（课上讲的写法，Post-LN）；True = 先 LayerNorm 再进子层（Pre-LN，Week04 作业用的）
    max_vocab: int = 4000               # 数据里一共约 2300 个不同的字，不会触到上限
    seed: int = 42
    use_position: bool = True           # False = --no-pos：去掉位置编码
    use_causal_mask: bool = True        # False = --no-mask：去掉因果掩码（泄漏实验）


IGNORE_INDEX = -100                     # labels 里填这个数 = 这个位置不算损失（nn.CrossEntropyLoss 默认就忽略 -100）


class SpecialTokens:
    """<summary>四个新增的特殊符号。它们在字表里和普通的字一样，各占一个编号。</summary>"""
    PAD = "<pad>"                       # 补齐用
    USER = "<user>"                     # 问题开始的标记
    ASSISTANT = "<assistant>"           # 回答开始的标记
    EOS = "<eos>"                       # 回答结束的标记（End Of Sequence）
    ALL = [PAD, USER, ASSISTANT, EOS]


# ═════════════════════════════ 字表 ═════════════════════════════
class CharVocabulary:
    """<summary>字表：字 ↔ 编号。和 language_model.py 的 build_vocab 一样，按出现次数取前 max_vocab 个字。</summary>"""

    def __init__(self, chars: List[str]) -> None:
        self._id_to_char: List[str] = chars
        # API 注释：enumerate(列表) 每一轮给出 (下标, 元素)；字典推导式 {键: 值 for ...} 据此造出 {字: 编号}
        self._char_to_id: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}   # 字典推导式：字 → 编号

    # API 注释：@staticmethod：静态方法，不需要先创建对象，直接 CharVocabulary.build(...) 调用（相当于 C# 的 static 方法）
    @staticmethod
    def build(text: str, max_vocab: int) -> "CharVocabulary":
        # Counter(text)：数每个字出现几次；most_common(n)：按次数从多到少取前 n 个，返回 [(字, 次数), ...]
        chars = [ch for ch, _ in Counter(text).most_common(max_vocab)]
        return CharVocabulary(chars)

    # API 注释：@property：让 vocab.size 像属性一样不加括号就能取值（相当于 C# 的只读属性）
    @property
    def size(self) -> int:
        return len(self._id_to_char)

    def contains(self, ch: str) -> bool:
        return ch in self._char_to_id   # API 注释：键 in 字典：判断这个键是否存在，返回 True / False

    def to_id(self, ch: str) -> int:
        return self._char_to_id[ch]

    def to_char(self, index: int) -> str:
        return self._id_to_char[index]

    def encode(self, text: str) -> List[int]:
        """<summary>文本 → 编号列表；字表外的字直接丢掉（与课件一致，课件的字表里没有 &lt;UNK&gt;）。</summary>"""
        # API 注释：列表推导式，等价于：ids = []; for ch in text: if ch in 字典: ids.append(字典[ch])；字表外的字被丢掉
        return [self._char_to_id[ch] for ch in text if ch in self._char_to_id]

    def to_list(self) -> List[str]:
        return list(self._id_to_char)   # API 注释：list(列表) 复制一份新列表，改它不会影响内部的字表


# ═════════════════════════════ 数据 ═════════════════════════════
class SftRepository:
    """<summary>读 jsonl 文件：每一行是一个 JSON，形如 {"instruction": 问题, "output": 回答}。</summary>"""

    @staticmethod
    def load(path: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(path, encoding="utf-8") as reader:
            for line in reader:                      # API 注释：文件对象可以直接 for 遍历，每次给一行
                if line.strip():                     # 跳过空行
                    rows.append(json.loads(line))    # 一行 JSON 文本 → 字典
        return rows


class SftDataset(Dataset):
    """
    <summary>一条样本 = (input_ids, labels)，两个等长的编号序列。</summary>
    例：问题「你好」，回答「嗨」：
        input_ids: <user> 你 好 <assistant> 嗨 <eos>
        labels   : -100   -100 -100 -100       嗨 <eos>        ← 问题段（含两个标记）全是 -100
    """

    def __init__(self, rows: List[Dict[str, str]], vocab: CharVocabulary, seq_len: int) -> None:
        self._items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for row in rows:
            prompt = SftDataset.prompt_ids(vocab, row["instruction"])
            answer = vocab.encode(row["output"]) + [vocab.to_id(SpecialTokens.EOS)]
            input_ids = (prompt + answer)[:seq_len]                          # 太长就截断（本数据不会触发）
            labels = ([IGNORE_INDEX] * len(prompt) + answer)[:seq_len]       # 问题段不算损失
            self._items.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))

    @staticmethod
    def prompt_ids(vocab: CharVocabulary, instruction: str) -> List[int]:
        """<summary>前缀：<user> 问题 <assistant>。训练拼序列和生成时起头，都用它，保证两边一致。</summary>"""
        return [vocab.to_id(SpecialTokens.USER)] + vocab.encode(instruction) + [vocab.to_id(SpecialTokens.ASSISTANT)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._items[index]


class BatchCollator:
    """
    <summary>一批样本长短不一，要补齐成矩形才能叠成张量。</summary>
    input_ids 右侧补 <pad>；labels 右侧补 -100（补齐的位置不算损失）。
    """

    def __init__(self, pad_id: int) -> None:
        self._pad_id = pad_id

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        longest = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), longest), self._pad_id, dtype=torch.long)       # API 注释：torch.full(形状, 值)：造一个全是同一个数的张量
        y = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
        for row, (ids, labels) in enumerate(batch):
            x[row, :len(ids)] = ids          # 前面放真实内容，后面留着补齐的值
            y[row, :len(labels)] = labels
        return x, y



# ═════════════════════════════ 模型 ═════════════════════════════
class MultiHeadSelfAttention(nn.Module):
    """
    <summary>多头自注意力：softmax(Q·Kᵀ/√d + 因果Mask) · V，切成多个头并行做，最后拼回来。</summary>
    【封装版】nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)，
              它把 Q、K、V 三个 Linear 合并成一个 in_proj_weight (3D, D)，输出的 Linear 叫 out_proj。
    """

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self._heads = heads
        self._head_dim = d_model // heads                  # 192 / 4 = 每个头 48 维
        self.query = nn.Linear(d_model, d_model)           # x → Q   (封装版：in_proj_weight 的第 1 段)
        self.key = nn.Linear(d_model, d_model)             # x → K   (第 2 段)
        self.value = nn.Linear(d_model, d_model)           # x → V   (第 3 段)
        self.output = nn.Linear(d_model, d_model)          # 多头拼回来之后再过一个 Linear   (封装版：out_proj)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape                   # (B, L, D)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (B, L, D) → (B, L, 头数, 每头维度) → (B, 头数, L, 每头维度)：每个头各自一份 (L, 每头维度)
            return t.view(batch, length, self._heads, self._head_dim).transpose(1, 2)

        q = split_heads(self.query(x))
        k = split_heads(self.key(x))
        v = split_heads(self.value(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self._head_dim)    # ① Q·Kᵀ ② ÷√d   → (B, 头数, L, L)
        if mask is not None:
            scores = scores + mask                                       # 因果 Mask：未来位置加 -inf
        weights = self._dropout(torch.softmax(scores, dim=-1))           # ③ softmax：每行和为 1（-inf 的位置权重 = 0）
        context = weights @ v                                            # ④ · V → (B, 头数, L, 每头维度)
        context = context.transpose(1, 2).reshape(batch, length, d_model)   # 把各个头拼回 (B, L, D)
        return self.output(context)


class TransformerBlock(nn.Module):
    """
    <summary>一个 Transformer 层 = 多头注意力 + 前馈网络，各自「残差 + LayerNorm」。</summary>
    课上讲的写法（norm_first=False，Post-LN）：
        Z = LayerNorm( X + 注意力(X) )
        Y = LayerNorm( Z + FFN(Z) )
    【封装版】nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*D, dropout, activation="gelu",
                                          batch_first=True, norm_first=False)
              对应关系：self_attn ↔ attention；linear1 ↔ expand；linear2 ↔ shrink；norm1/norm2 ↔ norm1/norm2
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        d = config.d_model
        self._norm_first = config.norm_first
        self.attention = MultiHeadSelfAttention(d, config.heads, config.dropout)
        self.expand = nn.Linear(d, 4 * d)                  # FFN 第一层：192 → 768（放宽 4 倍）
        self.shrink = nn.Linear(4 * d, d)                  # FFN 第二层：768 → 192（压回来，形状才能接残差）
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self._dropout = nn.Dropout(config.dropout)

    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个字单独加工：Linear → GELU → Dropout → Linear     （封装版：linear2(dropout(gelu(linear1(x)))) ）
        return self.shrink(self._dropout(nn.functional.gelu(self.expand(x))))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self._norm_first:
            # Post-LN（课上讲的）：先残差相加，再 LayerNorm
            x = self.norm1(x + self._dropout(self.attention(x, mask)))
            return self.norm2(x + self._dropout(self._feed_forward(x)))
        # Pre-LN（Week04 作业用的 norm_first=True）：先 LayerNorm 再进子层，残差主干不经过 LayerNorm
        x = x + self._dropout(self.attention(self.norm1(x), mask))
        return x + self._dropout(self._feed_forward(self.norm2(x)))


class TransformerLm(nn.Module):
    """
    <summary>字 embedding + 位置 embedding + N 个 TransformerBlock（带因果掩码）+ 输出层 = 字符级语言模型。</summary>
    形状（B=批大小，L=序列长度，D=d_model，V=字表大小）：
        x (B, L) → embedding (B, L, D) → N 个 Block (B, L, D) → 输出层 (B, L, V)
    【封装版】Week04 作业把中间的 N 个 Block 写成：
        layer = nn.TransformerEncoderLayer(...)
        self._encoder = nn.TransformerEncoder(layer, num_layers=N)     # 把 layer 复制 N 份串起来
    """

    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self._use_position = config.use_position
        self._use_causal_mask = config.use_causal_mask
        self._token_embedding = nn.Embedding(vocab_size, config.d_model)                 # 字表：编号 → 向量
        self._position_embedding = nn.Embedding(config.seq_len, config.d_model) if config.use_position else None
        self._dropout = nn.Dropout(config.dropout)
        # nn.ModuleList：装一组子模块，里面每个都是独立的一层（各有自己的参数）；封装版的 nn.TransformerEncoder 就是这么装的
        self._blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.layers)])
        # Pre-LN 的最后要补一个 LayerNorm（封装版：nn.TransformerEncoder(..., norm=...)）；Post-LN 每层末尾已经有了
        self._final_norm = nn.LayerNorm(config.d_model) if config.norm_first else nn.Identity()
        self._output_layer = nn.Linear(config.d_model, vocab_size)                       # 每个位置：D → V 个分数
        nn.init.normal_(self._token_embedding.weight, std=0.02)                          # 小方差初始化，初始 loss 才接近 ln(V)
        if self._position_embedding is not None:
            nn.init.normal_(self._position_embedding.weight, std=0.02)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        <summary>因果掩码 (L, L)：对角线及以下为 0（能看），右上角为 -inf（看不到）。L=3 时：
            [[0, -inf, -inf],
             [0,    0, -inf],
             [0,    0,    0]]
        </summary>
        【封装版】nn.Transformer.generate_square_subsequent_mask(seq_len)，结果完全一样。
        """
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """<summary>model(x) 会自动调用本方法。</summary><returns>logits (B, L, V)，没做 softmax（CrossEntropyLoss 里已包含）</returns>"""
        hidden = self._token_embedding(x)                                  # (B, L) → (B, L, D)
        if self._use_position:
            # 位置是「相加」不是「拼接」：形状不变，残差才接得上
            hidden = hidden + self._position_embedding(torch.arange(x.size(1), device=x.device))
        hidden = self._dropout(hidden)
        mask = TransformerLm.causal_mask(x.size(1), x.device) if self._use_causal_mask else None
        for block in self._blocks:
            hidden = block(hidden, mask)                                   # 前一层的输出交给后一层：(B, L, D) → (B, L, D)
        return self._output_layer(self._final_norm(hidden))                # (B, L, D) → (B, L, V)


class PackagedComparer:
    """
    <summary>把手写的一层，和 nn.TransformerEncoderLayer（封装版）放在一起比：权重拷成一样，输入一样，输出应当一样。</summary>
    这一步证明：封装版里面做的事，就是上面手写的那几步，没有别的魔法。
    """

    @staticmethod
    @torch.no_grad()
    def run(config: Config, device: torch.device) -> None:
        d = config.d_model
        mine = TransformerBlock(replace(config, norm_first=False, dropout=0.0)).to(device).eval()
        packaged = nn.TransformerEncoderLayer(d_model=d, nhead=config.heads, dim_feedforward=4 * d, dropout=0.0,
                                              activation="gelu", batch_first=True, norm_first=False).to(device).eval()
        # 把手写版的每个权重，拷到封装版对应的位置
        attention = mine.attention
        packaged.self_attn.in_proj_weight.copy_(torch.cat([attention.query.weight, attention.key.weight, attention.value.weight]))
        packaged.self_attn.in_proj_bias.copy_(torch.cat([attention.query.bias, attention.key.bias, attention.value.bias]))
        packaged.self_attn.out_proj.weight.copy_(attention.output.weight)
        packaged.self_attn.out_proj.bias.copy_(attention.output.bias)
        packaged.linear1.weight.copy_(mine.expand.weight); packaged.linear1.bias.copy_(mine.expand.bias)
        packaged.linear2.weight.copy_(mine.shrink.weight); packaged.linear2.bias.copy_(mine.shrink.bias)
        packaged.norm1.load_state_dict(mine.norm1.state_dict())
        packaged.norm2.load_state_dict(mine.norm2.state_dict())

        x = torch.randn(2, 10, d, device=device)
        mask = TransformerLm.causal_mask(10, device)
        # dropout=0，所以 train 模式的计算也是确定的；这条是 PyTorch 的普通计算路径，应当和手写逐步一致
        mine.train(); packaged.train()
        normal = (mine(x, mask) - packaged(x, src_mask=mask)).abs().max().item()
        print(f"普通路径（train 模式，dropout=0）：手写 vs nn.TransformerEncoderLayer 输出最大差 = {normal:.2e}（接近 0 = 一致）")
        # eval + 不记梯度时，封装版会自动换成融合加速的内核（fast path）；GPU 上数值会差到 1e-4 量级，不是手写算错
        mine.eval(); packaged.eval()
        fast = (mine(x, mask) - packaged(x, src_mask=mask)).abs().max().item()
        print(f"加速路径（eval + no_grad）：最大差 = {fast:.2e}（融合内核的数值差异）")
        mine_params = sum(p.numel() for p in mine.parameters())
        packaged_params = sum(p.numel() for p in packaged.parameters())
        print(f"参数量：手写 {mine_params:,} / 封装 {packaged_params:,}")



# ═════════════════════════════ 训练 ═════════════════════════════
class Trainer:
    """<summary>训练一轮 / 验证一轮。和 Week04 相比，只有损失那几行变了。</summary>"""

    def __init__(self, model: nn.Module, config: Config, device: torch.device) -> None:
        self._model = model
        self._config = config
        self._device = device
        # CrossEntropyLoss 默认 ignore_index=-100：labels 是 -100 的位置不进损失，也不进平均的分母
        self._criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """<returns>(平均损失, 困惑度 PPL)；只统计回答和 <eos> 这些「有标签」的位置</returns>"""
        self._model.train(train)
        total_loss = 0.0
        total_tokens = 0
        with torch.set_grad_enabled(train):
            for x, labels in loader:
                x, labels = x.to(self._device), labels.to(self._device)
                logits = self._model(x)                               # (B, L, V)
                # ★ 手动错位：位置 i 的输出，要预测的是 labels 的第 i+1 个
                shift_logits = logits[:, :-1, :]                      # 去掉最后一个位置的输出  (B, L-1, V)
                shift_labels = labels[:, 1:]                          # 去掉第一个位置的标签    (B, L-1)
                loss = self._criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                if train:
                    self._optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.grad_clip)
                    self._optimizer.step()
                count = int((shift_labels != IGNORE_INDEX).sum())     # 这一批有几个「算损失」的位置
                total_loss += loss.item() * count                     # 按有效位置数加权
                total_tokens += count
        average_loss = total_loss / total_tokens
        return average_loss, math.exp(average_loss)


class CausalityChecker:
    """<summary>因果性验证：扰动最后一个字，比较前面位置输出的变化。有掩码必须是 0，无掩码前面位置会跟着变。</summary>"""

    @staticmethod
    @torch.no_grad()
    def max_change_before_last(model: nn.Module, vocab_size: int, seq_len: int, device: torch.device) -> float:
        model.eval()                                           # 关掉 Dropout，否则两次前向本来就不一样
        # API 注释：torch.randint(低, 高, 形状)：在 [低, 高) 里随机取整数；这里造一段随机的字，形状 (1, L)；
        #   torch.Generator().manual_seed(0)：固定随机数起点，每次造出同一段字
        generator = torch.Generator().manual_seed(0)
        original = torch.randint(0, vocab_size, (1, seq_len), generator=generator).to(device)
        changed = original.clone()                             # API 注释：.clone()：复制一份，改它不影响原来的
        changed[0, -1] = (original[0, -1] + 1) % vocab_size    # 只改最后一个字
        difference = (model(original) - model(changed)).abs()  # (1, L, V)：两次输出逐元素相差多少
        return float(difference[0, :-1].max())                 # 只看最后一个位置「之前」的所有位置，取最大变化

    @staticmethod
    def run(vocab_size: int, config: Config, device: torch.device) -> None:
        print("generate_square_subsequent_mask(4) =")
        print(TransformerLm.causal_mask(4, torch.device("cpu")))
        for use_mask in (True, False):
            torch.manual_seed(0)
            # API 注释：replace(config, use_causal_mask=...)：复制一份 frozen 的 Config，只改指定的字段
            model = TransformerLm(vocab_size, replace(config, use_causal_mask=use_mask)).to(device)
            change = CausalityChecker.max_change_before_last(model, vocab_size, config.seq_len, device)
            print(f"{'有' if use_mask else '无'}因果掩码：改最后一个字后，前 {config.seq_len - 1} 个位置输出的最大变化 = {change:.3e}")


class LabelInspector:
    """<summary>打印一条样本的「输入字 / labels / 由哪个位置的输出来预测它」，肉眼确认对齐。</summary>"""

    @staticmethod
    def show(vocab: CharVocabulary, input_ids: torch.Tensor, labels: torch.Tensor) -> None:
        print(f"{'位置':>4}  {'输入':<12}{'labels':<14}由哪个位置的输出来预测它")
        for i, (token_id, label) in enumerate(zip(input_ids.tolist(), labels.tolist())):
            label_text = "-100(忽略)" if label == IGNORE_INDEX else vocab.to_char(label)
            who = "—" if label == IGNORE_INDEX else f"位置 {i - 1} 的输出"
            print(f"{i:>4}  {vocab.to_char(token_id):<12}{label_text:<14}{who}")


# ═════════════════════════════ 预测 ═════════════════════════════
class TextPredictor:
    """<summary>训练完之后用：给一个问题，逐字写出回答。</summary>"""

    def __init__(self, model: nn.Module, vocab: CharVocabulary, seq_len: int, device: torch.device) -> None:
        self._model = model
        self._vocab = vocab
        self._seq_len = seq_len
        self._device = device

    def reply(self, instruction: str) -> Tuple[str, bool]:
        """
        <summary>贪心生成：<user> 问题 <assistant> 之后，每次取概率最大的字接上去，遇到 <eos> 就停。</summary>
        这就是课上的「生成循环」：前向 → 选字 → 拼回 → 再前向，停止条件是 <eos> 或到最长长度。
        <returns>(回答文字, 是不是遇到 <eos> 正常停止的)</returns>
        """
        self._model.eval()
        ids = SftDataset.prompt_ids(self._vocab, instruction)
        start = len(ids)
        eos_id = self._vocab.to_id(SpecialTokens.EOS)
        stopped_by_eos = False
        with torch.no_grad():
            while len(ids) < self._seq_len:                       # 位置编码只有 seq_len 行，不能更长
                x = torch.tensor([ids], dtype=torch.long, device=self._device)    # (1, T)
                next_id = int(self._model(x)[0, -1].argmax())     # 最后一个位置的输出 → 概率最大的字
                if next_id == eos_id:
                    stopped_by_eos = True
                    break
                ids.append(next_id)
        return "".join(self._vocab.to_char(i) for i in ids[start:]), stopped_by_eos



# ═════════════════════════════ 入口 ═════════════════════════════
class Program:
    """<summary>程序入口，相当于 C# 的 class Program。</summary>"""

    script_dir = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def parse_args() -> Tuple[Config, argparse.Namespace]:
        parser = argparse.ArgumentParser(description="小型 SFT：在字符级 Transformer 上学「被问就答」")
        parser.add_argument("--train", default=os.path.join(Program.script_dir, "data", "sft_train.jsonl"))
        parser.add_argument("--val", default=os.path.join(Program.script_dir, "data", "sft_val.jsonl"))
        parser.add_argument("--save", default=os.path.join(Program.script_dir, "checkpoints", "sft_lm.pt"))
        parser.add_argument("--epochs", type=int, default=Config.epochs)
        parser.add_argument("--batch-size", type=int, default=Config.batch_size)
        parser.add_argument("--lr", type=float, default=Config.learning_rate)
        parser.add_argument("--no-mask", action="store_true", help="去掉因果掩码（泄漏实验）")
        parser.add_argument("--no-pos", action="store_true", help="去掉位置编码")
        parser.add_argument("--pre-norm", action="store_true", help="用 Pre-LN（先 LayerNorm 再进子层，Week04 作业的写法）")
        parser.add_argument("--compare", action="store_true", help="手写的一层 vs 封装的 nn.TransformerEncoderLayer 对拍")
        parser.add_argument("--check", action="store_true", help="打印 labels 对齐表并验证因果掩码")
        parser.add_argument("--overfit", action="store_true", help="只用 32 条样本训练，看流程能不能拟合")
        parser.add_argument("--generate", default=None, help="用 --save 指向的模型回答一个问题")
        args = parser.parse_args()
        config = Config(train_path=args.train, val_path=args.val, save_path=args.save,
                        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                        use_position=not args.no_pos, use_causal_mask=not args.no_mask, norm_first=args.pre_norm)
        return config, args

    @staticmethod
    def show_replies(predictor: TextPredictor, rows: List[Dict[str, str]], count: int) -> None:
        for row in rows[:count]:
            answer, eos = predictor.reply(row["instruction"])
            print(f"  问题：{row['instruction']}")
            print(f"  标准：{row['output']}")
            print(f"  模型：{answer}   [{'遇到 <eos> 停止' if eos else '到最长长度才停'}]\n")

    @staticmethod
    def main() -> None:
        config, args = Program.parse_args()
        torch.manual_seed(config.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.compare:
            PackagedComparer.run(config, device)
            return

        if args.generate:
            checkpoint = torch.load(config.save_path, map_location=device)
            saved = Config(**checkpoint["config"])
            vocab = CharVocabulary(checkpoint["chars"])
            model = TransformerLm(vocab.size, saved).to(device)
            model.load_state_dict(checkpoint["model_state"])
            print(TextPredictor(model, vocab, saved.seq_len, device).reply(args.generate)[0])
            return

        # ── 步骤 1：读数据、建字表 ──
        train_rows = SftRepository.load(config.train_path)
        val_rows = SftRepository.load(config.val_path)
        if args.overfit:                                       # 过拟合实验：训练集和验证集都用同样的 32 条
            train_rows = train_rows[:32]
            val_rows = train_rows
            config = replace(config, epochs=100, batch_size=8)
        # 字表只用训练集的字建，再把 4 个特殊符号放在最前面
        text = "".join(r["instruction"] + r["output"] for r in train_rows)
        vocab = CharVocabulary(SpecialTokens.ALL + CharVocabulary.build(text, config.max_vocab).to_list())
        print(f"device: {device}  训练 {len(train_rows)} 条 / 验证 {len(val_rows)} 条  字表 {vocab.size}（含 4 个特殊符号）")

        # ── 步骤 2：数据集 ──
        train_set = SftDataset(train_rows, vocab, config.seq_len)
        val_set = SftDataset(val_rows, vocab, config.seq_len)
        collator = BatchCollator(vocab.to_id(SpecialTokens.PAD))
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, collate_fn=collator)

        # ── 步骤 3：模型（和 Week04 一样）──
        model = TransformerLm(vocab.size, config).to(device)
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        print("\n── labels 对齐表（第 1 条训练样本）──")
        LabelInspector.show(vocab, *train_set[0])
        print("\n── 因果性验证 ──")
        CausalityChecker.run(vocab.size, config, device)
        if args.check:
            return

        # ── 步骤 4：训练 ──
        trainer = Trainer(model, config, device)
        predictor = TextPredictor(model, vocab, config.seq_len, device)
        start_loss, start_ppl = trainer.run_epoch(val_loader, train=False)
        print(f"\n训练前验证集  Loss {start_loss:.4f}  PPL {start_ppl:.2f}")
        Program.show_replies(predictor, val_rows, 2)

        os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)
        best_val_loss = float("inf")
        print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val PPL':>9}  {'用时':>5}")
        for epoch in range(1, config.epochs + 1):
            start_time = time.time()
            train_loss, _ = trainer.run_epoch(train_loader, train=True)
            val_loss, val_ppl = trainer.run_epoch(val_loader, train=False)
            marker = ""
            if val_loss < best_val_loss:                          # 只保存验证损失最低的那一版
                best_val_loss = val_loss
                marker = "  *"
                torch.save({"model_state": model.state_dict(), "chars": vocab.to_list(),
                            "config": config.__dict__}, config.save_path)
            if not args.overfit or epoch % 20 == 0:
                print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_ppl:>9.2f}  {time.time() - start_time:>4.0f}s{marker}")

        # ── 步骤 5：看看它学到了什么 ──
        checkpoint = torch.load(config.save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print("\n── 验证集里的问题（模型没见过）──" if not args.overfit else "\n── 这 32 条（训练时见过）──")
        Program.show_replies(predictor, val_rows, 5)
        if not args.overfit:
            print("── 训练集里的问题（开卷）──")
            Program.show_replies(predictor, train_rows, 3)
        exact = sum(predictor.reply(r["instruction"])[0] == "".join(vocab.to_char(i) for i in vocab.encode(r["output"]))
                    for r in val_rows)                            # 标准答案里字表外的字会被丢掉，所以也要先 encode 再转回字
        print(f"回答与标准答案逐字一致：{exact}/{len(val_rows)}")


if __name__ == "__main__":
    Program.main()
