# coding: utf-8
"""
Week04 作业：用 BERT 完成「循环神经网络语言模型 / language_model.py」的功能

功能（和 language_model.py 一样）：
    字符级语言模型：给前面的字，预测下一个字；训练、验证，每轮打印 Loss 和困惑度 PPL，保存验证 PPL 最好的模型。

和 language_model.py 的区别：把 Embedding + RNN/LSTM 换成预训练好的 BERT（bert-base-chinese）。
    BERT 本来是双向的：每个字都能看到后面的字，直接拿来「预测下一个字」等于偷看答案。
    所以这里给自注意力加上「因果掩码」（下三角矩阵）：第 t 个位置只能看到第 0..t 个位置，
    BERT 就变成了单向的语言模型（课件第 28 页：自回归生成需要因果掩码，把右侧的注意力屏蔽掉）。

为了让 PPL 能和课件里的 LSTM（38.32）、RNN（40.83）直接比较，这几处和课件保持一致：
    · 语料：data/corpus_general.txt（从课件复制过来）
    · 字表：按频率取前 4000 个字（含换行符），字表外的字丢掉
    · 划分：按行打乱（random.seed(42)），最后 5% 做验证
    · 验证：seq_len=64、不重叠切块、批大小 128、丢掉最后不满的一批（与 tools/eval_ngram.py 的口径一致）

代码结构（延续 C# 风格：一个类管一件事、方法大写开头、局部变量小驼峰、私有字段加下划线）：
    Config            设置
    CharVocabulary    字表：字 ↔ 编号
    BertIdMapper      把我们的字编号转成 BERT 字表里的编号
    CorpusRepository  读语料、按行划分训练/验证
    CharLmDataset     切块：x = 连续 seq_len 个字，y = 往后错一位
    BertCausalLm      模型：BERT（加因果掩码）+ 输出层
    Trainer           训练 / 验证一轮
    TextPredictor     训练完之后：预测下一个字、续写
    Program           入口

运行（在本文件所在目录）：
    python train_bert_lm.py                         # 默认：3 轮，训练切块步长 16
    python train_bert_lm.py --epochs 1 --train-stride 64     # 快速试跑（每轮约 1 分钟，RTX 4090）
默认使用已下载的本地 BERT；也可以通过 --pretrained 指定其他模型目录。
"""

# ═══ API 注释说明（后加的）═════════════════════════════════════════════════
# 以「API 注释：」开头的注释是后来补的，只解释库函数 / 方法的作用、参数和形状，没有改任何代码。
# 速查（形状里的符号）：B = 一批有几个样本，L = 每个样本几个字（seqLen），H = 768（BERT 隐藏维度），V = 4000（我们的字表大小）。
# ══════════════════════════════════════════════════════════════════════════

import argparse                     # API 注释：Python 自带，解析命令行参数（--epochs 3 这种）
import math                         # API 注释：Python 自带，数学函数，这里用 math.exp 算困惑度
import os                           # API 注释：Python 自带，路径、文件夹相关操作
import random                       # API 注释：Python 自带的随机数模块（不是 PyTorch 的），这里用来打乱语料的行
import time                         # API 注释：Python 自带，time.time() 返回当前时间（秒），两次相减得到耗时
from collections import Counter     # API 注释：Counter 是「数数」专用的字典，Counter("abca") → {a: 2, b: 1, c: 1}
from dataclasses import dataclass   # API 注释：@dataclass 装饰器，根据类里写的字段自动生成 __init__ 等方法
from typing import Dict, List, Tuple    # API 注释：只用来写「类型标注」（如 List[int]），运行时不起作用，给人和编辑器看

import torch                        # API 注释：PyTorch 主库：张量（tensor）、自动求导、GPU 计算
import torch.nn as nn               # API 注释：神经网络模块：各种层（Linear 等）和损失函数；as nn 是起个短别名
from torch.utils.data import DataLoader, Dataset   # API 注释：Dataset = 「第 i 条数据是什么」的接口；DataLoader = 按批取出数据
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
# API 注释：transformers 是 Hugging Face 的库。BertModel：BERT 网络本体（不带任务输出头）；
#   BertTokenizer：分词器（字 ↔ BERT 的编号）；get_linear_schedule_with_warmup：学习率「先升后降」的调度器
from transformers import __version__ as transformersVersion    # API 注释：transformers 库的版本号字符串，如 "4.55.0"；forward 里要根据大版本号选掩码格式
from transformers import logging as hfLogging                  # API 注释：transformers 自带的日志模块，用来控制它打印多少提示

# 加载 bert-base-chinese 时会提示「有些权重没用上」（MLM/NSP 的输出头，我们用不到），属于正常情况，这里只显示错误
# API 注释：set_verbosity_error()：只显示错误级别的日志，警告和普通提示不再打印
hfLogging.set_verbosity_error()


# ═════════════════════════════ 设置 ═════════════════════════════
# API 注释：@dataclass(frozen=True)：自动生成构造函数；frozen=True 让对象创建后不能再改字段（改了会报 FrozenInstanceError）
@dataclass(frozen=True)
class Config:
    """<summary>所有可调的设置。frozen=True：创建后不能再改（相当于 C# 的 record / readonly）。</summary>"""
    CorpusPath: str                    # 语料文件
    SavePath: str                      # 保存最好模型的路径
    PretrainedName: str = r"D:\BaiduNetdiskDownload\第四周 语言模型\bert-base-chinese\bert-base-chinese"   # 包含 config.json、权重和 vocab.txt 的本地目录
    Epochs: int = 3                    # 把训练集完整过几遍（微调一般 2~4 轮就够）
    SeqLen: int = 64                   # 每个样本 64 个字（BERT 最多 512 个位置）
    TrainStride: int = 64              # 训练集切块步长：1 = 每个字都当起点（样本最多、最慢），64 = 不重叠
    BatchSize: int = 64                # 训练时每批 64 个样本
    EvalBatchSize: int = 128           # 验证时每批 128 个样本（与课件评估口径一致）
    LearningRate: float = 5e-5         # 微调用很小的学习率：BERT 已经训练好了，步子大了会把学到的东西冲掉
    WeightDecay: float = 0.01          # AdamW 的权重衰减
    WarmupRatio: float = 0.1           # 前 10% 的步数学习率从 0 慢慢升上去，之后线性降到 0
    GradClip: float = 1.0              # 梯度裁剪：梯度的总长度超过 1 就按比例缩小，防止偶尔一步走太远
    MaxVocab: int = 4000               # 字表大小（与课件 LSTM 相同，PPL 才能比较）
    ValRatio: float = 0.05             # 最后 5% 的行做验证集
    Seed: int = 42                     # 随机种子
    UseBf16: bool = True               # 在显卡上用 bfloat16 混合精度：更快、更省显存


# ═════════════════════════════ 字表 ═════════════════════════════
class CharVocabulary:
    """<summary>字表：字 ↔ 编号。和 language_model.py 的 build_vocab 一样，按出现次数取前 maxVocab 个字。</summary>"""

    def __init__(self, chars: List[str]) -> None:
        self._idToChar: List[str] = chars
        # API 注释：enumerate(列表) 每一轮给出 (下标, 元素)；字典推导式 {键: 值 for ...} 据此造出 {字: 编号}
        self._charToId: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}   # 字典推导式：字 → 编号

    # API 注释：@staticmethod：静态方法，不需要先创建对象，直接 CharVocabulary.Build(...) 调用（相当于 C# 的 static 方法）
    @staticmethod
    def Build(text: str, maxVocab: int) -> "CharVocabulary":
        # Counter(text)：数每个字出现几次；most_common(n)：按次数从多到少取前 n 个，返回 [(字, 次数), ...]
        chars = [ch for ch, _ in Counter(text).most_common(maxVocab)]
        return CharVocabulary(chars)

    # API 注释：@property：让 vocab.Size 像属性一样不加括号就能取值（相当于 C# 的只读属性）
    @property
    def Size(self) -> int:
        return len(self._idToChar)

    def Contains(self, ch: str) -> bool:
        return ch in self._charToId   # API 注释：键 in 字典：判断这个键是否存在，返回 True / False

    def ToId(self, ch: str) -> int:
        return self._charToId[ch]

    def ToChar(self, index: int) -> str:
        return self._idToChar[index]

    def Encode(self, text: str) -> List[int]:
        """<summary>文本 → 编号列表；字表外的字直接丢掉（与课件一致，课件的字表里没有 &lt;UNK&gt;）。</summary>"""
        # API 注释：列表推导式，等价于：ids = []; for ch in text: if ch in 字典: ids.append(字典[ch])；字表外的字被丢掉
        return [self._charToId[ch] for ch in text if ch in self._charToId]

    def ToList(self) -> List[str]:
        return list(self._idToChar)   # API 注释：list(列表) 复制一份新列表，改它不会影响内部的字表


class BertIdMapper:
    """
    <summary>把我们字表里的每个字，映射成 BERT 字表（21128 个）里的编号。</summary>
    为什么需要：输入给 BERT 时必须用它自己的编号，它预训练时学到的字向量才对得上；
    输出时仍然在我们自己的 4000 个字上打分，所以 PPL 能和课件的 LSTM 直接比较。
    """

    @staticmethod
    def Build(vocab: CharVocabulary, tokenizer: BertTokenizer) -> Tuple[torch.Tensor, int]:
        """<returns>(映射表：长度为 vocab.Size 的整数张量, 映射成 [UNK] 的字数)</returns>"""
        bertIds: List[int] = []
        unknownCount = 0
        for index in range(vocab.Size):
            bertId = BertIdMapper.ToBertId(vocab.ToChar(index), tokenizer)
            if bertId == tokenizer.unk_token_id:
                unknownCount += 1
            bertIds.append(bertId)
        # API 注释：torch.tensor(列表, dtype=torch.long)：Python 列表 → PyTorch 张量；long 即 int64 整数，用作下标/编号必须是整数
        return torch.tensor(bertIds, dtype=torch.long), unknownCount

    @staticmethod
    def ToBertId(ch: str, tokenizer: BertTokenizer) -> int:
        # API 注释（分词器 tokenizer 的三个常用接口）：
        #   tokenizer.sep_token_id / unk_token_id：特殊符号 [SEP]（句子分隔）、[UNK]（未知字）在 BERT 字表里的编号
        #   tokenizer.convert_tokens_to_ids(字)：查 BERT 字表，字 → 编号；查不到返回 unk_token_id，不会报错
        #   tokenizer.tokenize(文本)：按 BERT 的规则切分并规范化（全角转半角、去重音等），返回片段列表；这里 len(pieces)==1 表示规范化后正好是一个字
        if ch == "\n":
            return tokenizer.sep_token_id                      # 换行 = 一个问句结束，用 BERT 的句子分隔符 [SEP] 表示
        bertId = tokenizer.convert_tokens_to_ids(ch)           # 直接查 BERT 字表
        if bertId != tokenizer.unk_token_id:
            return bertId
        bertId = tokenizer.convert_tokens_to_ids(ch.lower())   # bert-base-chinese 的字表里英文字母只有小写
        if bertId != tokenizer.unk_token_id:
            return bertId
        pieces = tokenizer.tokenize(ch)                        # 再让分词器自己规范化一次（全角、重音等）
        if len(pieces) == 1:
            return tokenizer.convert_tokens_to_ids(pieces[0])
        return tokenizer.unk_token_id                          # 实在没有，就当作 [UNK]（只影响输入，不影响要预测的答案）


# ═════════════════════════════ 数据 ═════════════════════════════
class CorpusRepository:
    """<summary>读语料、划分训练集和验证集。</summary>"""

    @staticmethod
    def LoadText(path: str) -> str:
        # API 注释：open(路径, encoding=...)：打开文本文件；with 用完自动关闭；encoding="utf-8" 指定编码（不写时 Windows 可能按 GBK 读，中文会乱码）
        with open(path, encoding="utf-8", errors="ignore") as reader:   # errors="ignore"：遇到坏字节直接跳过
            return reader.read()   # API 注释：read()：一次读出整个文件，得到一个很长的字符串

    @staticmethod
    def SplitByLines(text: str, valRatio: float, seed: int) -> Tuple[str, str]:
        """
        <summary>按行打乱，最后 valRatio 的行做验证集。</summary>
        和 language_model.py / eval_ngram.py 完全相同：先 random.seed(seed)，再打乱，所以划分结果一样。
        按「行」划分：训练和验证是不同的问句，不会把同一句话拆成两半。
        """
        random.seed(seed)                                      # API 注释：固定 Python 随机数的起点，同一个种子每次打乱的顺序一样
        lines = text.splitlines()                              # API 注释：字符串.splitlines()：按换行符切成「行」列表（不含换行符）
        random.shuffle(lines)                                  # 原地打乱顺序
        split = int(len(lines) * (1 - valRatio))               # API 注释：int() 取整数部分，得到「前 95% 有多少行」
        # API 注释：lines[:split] 是切片，取前 split 行；lines[split:] 取剩下的；"\n".join(列表)：用换行符把一列字符串连成一个大字符串
        return "\n".join(lines[:split]), "\n".join(lines[split:])


class CharLmDataset(Dataset):
    """
    <summary>语言模型的训练样本：x = 连续 seqLen 个字，y = 同样的位置往后错一位。</summary>
    例：seqLen=3，文本「今天天气」：x = 今天天，y = 天天气，相当于三道题：今→天、今天→天、今天天→气。
    答案就是文本本身，不需要人工标注（自监督学习）。
    stride：相邻两个样本起点相差几个字；1 = 每个字都当一次起点，seqLen = 不重叠。
    """

    def __init__(self, text: str, vocab: CharVocabulary, seqLen: int, stride: int) -> None:
        # API 注释：整篇语料一次性转成编号，存成一个一维长张量 (N,)；之后取样本只是切片，很快
        self._data = torch.tensor(vocab.Encode(text), dtype=torch.long)
        self._seqLen = seqLen
        self._stride = max(1, stride)   # API 注释：max(1, stride)：取较大者，保证步长至少是 1（0 或负数会让 __len__ 出错）

    def __len__(self) -> int:
        # 样本数 = (总字数 − seqLen − 1) ÷ stride + 1；max(0, …) 防止语料太短时变成负数
        # API 注释：len(张量) 是第 0 维的长度（总字数）；// 是整除（向下取整）
        return max(0, (len(self._data) - self._seqLen - 1) // self._stride + 1)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self._stride
        # API 注释：张量切片 t[a:b] 取第 a 到 b−1 个元素，得到形状 (seqLen,) 的张量（只是原数据的视图，不复制）
        x = self._data[start: start + self._seqLen]
        y = self._data[start + 1: start + self._seqLen + 1]   # 往后错一位：y[t] 就是 x[t] 的下一个字
        return x, y


# ═════════════════════════════ 模型 ═════════════════════════════
class BertCausalLm(nn.Module):
    """
    <summary>BERT + 因果掩码 + 输出层 = 单向的字符级语言模型。</summary>
    形状（B=批大小，L=seqLen，H=768，V=我们的字表大小 4000）：
        x (B, L) 我们的编号 → BERT 编号 (B, L) → BERT (B, L, H) → 输出层 (B, L, V)
    每个位置都输出一份「下一个字是 V 个字中每一个的分数」，和 language_model.py 的 LM 输出形状相同。
    """

    def __init__(self, pretrainedName: str, toBertIds: torch.Tensor, unkTokenId: int) -> None:
        super().__init__()
        # BertModel.from_pretrained：加载预训练好的 BERT（12 层 Transformer，约 1 亿个参数），没有 MLM/NSP 的输出头
        # eager：使用标准注意力计算；forward 中按 Transformers 版本准备对应的掩码格式。
        # API 注释：BertModel.from_pretrained(路径或名字, ...)：加载预训练好的结构和权重；路径可以是本地文件夹（里面要有 config.json、权重文件、vocab.txt）
        #   attn_implementation="eager"：注意力用最朴素的写法（Q·Kᵀ → 加掩码 → softmax → ·V）逐步算，便于直接传入自定义的因果掩码
        self._bert = BertModel.from_pretrained(pretrainedName, attn_implementation="eager")
        # register_buffer：把映射表登记成模型的一部分。它不是要训练的参数，但会跟着 .to(device) 搬到显卡，也会存进 state_dict
        self.register_buffer("_toBertIds", toBertIds)
        # API 注释：.config 是 BERT 的配置（来自 config.json），.hidden_size 是隐藏维度 768；
        #   张量.numel()：张量里一共有几个元素，toBertIds 是长度 4000 的一维张量，所以得到 4000
        hiddenSize = self._bert.config.hidden_size              # 768
        vocabSize = toBertIds.numel()                           # 4000
        # API 注释：nn.Linear(入, 出)：全连接层 y = x·Wᵀ + b；权重形状 (出, 入) = (4000, 768)，把每个位置的 768 个数变成 4000 个字各自的分数
        self._outputLayer = nn.Linear(hiddenSize, vocabSize)   # y = h · Wᵀ + b，W 的形状 (4000, 768)
        self.InitOutputLayerFromBert(toBertIds, unkTokenId)

    def InitOutputLayerFromBert(self, toBertIds: torch.Tensor, unkTokenId: int) -> None:
        """
        <summary>用 BERT 自己的字向量初始化输出层的权重（BERT 预训练时的 MLM 输出层也是这样和字向量共用的）。</summary>
        这样一开始「隐藏向量和哪个字的字向量更像，就更可能是哪个字」，比随机初始化起步快。
        映射成 [UNK] 的字共用同一行字向量，所以这些字保持随机初始化。
        """
        wordEmbeddings = self._bert.embeddings.word_embeddings.weight   # (21128, 768)
        with torch.no_grad():                                           # 只是赋初值，不需要记录梯度
            known = toBertIds != unkTokenId                            # 布尔张量：哪些字有自己的 BERT 编号
            # API 注释：这一行有两层「用下标取元素」：
            #   toBertIds[known]：用布尔张量当下标（布尔索引），只保留 known 为 True 的那些 BERT 编号；
            #   wordEmbeddings[编号列表]：用整数列表当下标，取出这些编号对应的行，得到 (n, 768)；
            #   self._outputLayer.weight[known] = ...：把这些行复制到输出层权重里对应的行（赋值，不是新建）
            self._outputLayer.weight[known] = wordEmbeddings[toBertIds[known]]
            self._outputLayer.bias.zero_()   # API 注释：方法名末尾的下划线 _ 表示「原地修改」：把偏置全部改成 0

    @staticmethod
    def CausalMask(batchSize: int, seqLen: int, device: torch.device) -> torch.Tensor:
        """
        <summary>因果掩码：(B, L, L) 的下三角矩阵，mask[b, i, j] = 1 表示第 i 个位置可以看第 j 个位置。</summary>
        torch.tril：只保留对角线及以下（j ≤ i），上三角全变成 0，所以每个字只能看到自己和前面的字。例如 L=3：
            [[1, 0, 0],
             [1, 1, 0],
             [1, 1, 1]]
        Transformers 4.x 的 BERT 会把此三维掩码中的 0 转为很大的负数；5.x 需要在 forward 中先转换为四维加法掩码。
        """
        # API 注释：torch.ones(L, L, ...)：全 1 的 L×L 矩阵；dtype 指定整数类型，device 指定放在 CPU 还是显卡；
        #   torch.tril(矩阵)：只保留下三角（含对角线），上三角全变 0
        mask = torch.tril(torch.ones(seqLen, seqLen, dtype=torch.long, device=device))
        # API 注释：.unsqueeze(0)：在最前面插入一个长度为 1 的维度；.expand(B, -1, -1)：把这个维度「看成」B 份，-1 表示该维度保持不变；
        #   expand 不会真的复制数据（所有批共用同一块内存），所以几乎不占内存
        return mask.unsqueeze(0).expand(batchSize, -1, -1)   # (L, L) → (1, L, L) → (B, L, L)；expand 不复制内存

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """<summary>model(x) 会自动调用本方法。</summary><returns>logits (B, L, V)，未经 softmax。</returns>"""
        batchSize, seqLen = x.shape   # API 注释：张量.shape 是各维大小，一次拆包给两个变量：B 和 L
        # API 注释：用整数张量当下标取元素（查表）：_toBertIds 是长度 4000 的映射表，x 形状 (B, L)，结果形状也是 (B, L)
        bertInputIds = self._toBertIds[x]                     # 用编号当下标查表：(B, L) 我们的编号 → (B, L) BERT 的编号
        attentionMask = BertCausalLm.CausalMask(batchSize, seqLen, x.device)
        # API 注释：transformersVersion 形如 "4.55.0"；.split(".") 按点切开得到 ["4", "55", "0"]，[0] 取第一段，int() 转成数字 4
        if int(transformersVersion.split(".")[0]) >= 5:
            # Transformers 5.x：自定义掩码需为 (B, 1, L, L)，中间的 1 广播到所有注意力头。
            # eager 直接把它加到注意力分数上：允许的位置为 0，禁止的位置为极大负数。
            # 不能直接传四维 0/1，否则未来位置并没有真正被屏蔽。
            # API 注释：.unsqueeze(1)：在第 1 维插入长度 1 的维度，(B, L, L) → (B, 1, L, L)；== 0 逐元素比较，得到布尔张量：True 表示「禁止看」
            blocked = attentionMask.unsqueeze(1) == 0
            # API 注释：torch.zeros(形状, dtype, device)：全 0 的张量；self._bert.dtype 是 BERT 权重的数据类型（float32）
            attentionMask = torch.zeros(blocked.shape, dtype=self._bert.dtype, device=x.device)
            # API 注释：masked_fill_(布尔条件, 值)：条件为 True 的位置原地填成该值（末尾下划线 = 原地修改）；
            #   torch.finfo(类型).min：这个浮点类型能表示的最小（最负）的数，加到注意力分数上，softmax 之后权重就是 0
            attentionMask.masked_fill_(blocked, torch.finfo(attentionMask.dtype).min)
        # last_hidden_state：最后一层每个位置的向量 (B, L, 768)；位置编号默认 0..L−1，句子编号默认全是 0
        # 明确返回带字段的结果，兼容本地 config.json 中 return_dict=false 的设置
        # API 注释：self._bert(...) 即调用模型的 forward；input_ids 是编号，attention_mask 是「谁能看谁」；
        #   return_dict=True：返回带名字的结果对象，用 .last_hidden_state 取最后一层的输出
        hidden = self._bert(input_ids=bertInputIds, attention_mask=attentionMask, return_dict=True).last_hidden_state
        return self._outputLayer(hidden)                      # (B, L, 768) → (B, L, 4000)


# ═════════════════════════════ 训练 ═════════════════════════════
class Trainer:
    """<summary>训练一轮 / 验证一轮。对应 language_model.py 的 run_epoch。</summary>"""

    def __init__(self, model: nn.Module, config: Config, totalSteps: int, device: torch.device) -> None:
        self._model = model
        self._config = config
        self._device = device
        # API 注释：nn.CrossEntropyLoss()：交叉熵损失，输入 logits (N, V) 和整数标签 (N,)，内部已含 softmax，所以模型不能先做 softmax
        self._criterion = nn.CrossEntropyLoss()                 # 每个位置：−log(softmax(logits)[正确的下一个字])，再取平均
        # AdamW：Adam + 权重衰减（把衰减和梯度更新分开算），微调 BERT 的常用优化器
        # API 注释：torch.optim.AdamW(参数, lr, weight_decay)：优化器；model.parameters() 依次给出模型里所有可训练的张量；
        #   lr 学习率；weight_decay 每步把权重按比例缩小一点，防止权重变得过大
        self._optimizer = torch.optim.AdamW(model.parameters(), lr=config.LearningRate,
                                            weight_decay=config.WeightDecay)
        # 学习率先线性升高（warmup）、再线性降到 0；scheduler.step() 每训练一步调用一次
        # API 注释：get_linear_schedule_with_warmup(优化器, 预热步数, 总步数)：返回学习率调度器；
        #   前 num_warmup_steps 步学习率从 0 线性升到设定值，之后线性降到 0；总步数 = 轮数 × 每轮批数
        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer, num_warmup_steps=int(config.WarmupRatio * totalSteps), num_training_steps=totalSteps)
        # API 注释：device.type 是设备类型字符串（"cuda" 或 "cpu"）；bfloat16 混合精度只在显卡上启用
        self._useBf16 = config.UseBf16 and device.type == "cuda"

    def RunEpoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """<returns>(这一轮的平均损失, 困惑度 PPL = exp(平均损失))</returns>"""
        # API 注释：model.train(True) = 训练模式（Dropout 生效）；model.train(False) 等价于 model.eval()（Dropout 关闭）
        self._model.train(train)                               # True = 训练模式（Dropout 生效），False = 预测模式
        totalLoss = 0.0
        totalTokens = 0
        # torch.set_grad_enabled(False)：验证时不记录梯度，省显存、更快（等价于 with torch.no_grad()）
        # API 注释：torch.set_grad_enabled(开关)：作为 with 使用时，里面是否记录梯度由开关决定；train=False 时不建计算图，省显存
        with torch.set_grad_enabled(train):
            for x, y in loader:
                # API 注释：张量.to(device)：把张量搬到显卡（或 CPU）；数据和模型必须在同一个设备上才能计算
                x, y = x.to(self._device), y.to(self._device)
                # autocast：矩阵乘法等用 bfloat16 计算，其余保持 float32；enabled=False 时什么也不做
                # （bfloat16 的数值范围和 float32 一样大，只是精度低一些，所以不需要像 float16 那样做损失缩放）
                with torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=self._useBf16):
                    logits = self._model(x)                    # (B, L, V)
                # CrossEntropyLoss 要 (N, V) 和 (N,)：把「批 × 位置」合并成一维，一批里的 B×L 道题一起批改
                # API 注释：.float() 转成 float32，让损失用高精度计算；.reshape(-1, V)：-1 表示这一维自动算出（= B×L），(B, L, V) → (B×L, V)；
                #   logits.size(-1) 取最后一维的大小（V）；y.reshape(-1) 把 (B, L) 拉平成 (B×L,)
                loss = self._criterion(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
                if train:
                    self._optimizer.zero_grad()                # 清空上一批的梯度
                    loss.backward()                            # 反向传播，算出每个参数的梯度
                    # API 注释：clip_grad_norm_(参数, 最大范数)：先算所有梯度合在一起的长度（L2 范数），超过最大值就把全部梯度按同一比例缩小；原地修改
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.GradClip)
                    self._optimizer.step()                     # 按 AdamW 更新参数
                    self._scheduler.step()                     # 更新学习率
                # API 注释：loss 是只有一个数的张量，.item() 取出成 Python 小数才能累加；y.numel() = 这一批一共有几道题（B×L）
                totalLoss += loss.item() * y.numel()           # 按字数加权，最后一批大小不同也公平
                totalTokens += y.numel()
        averageLoss = totalLoss / totalTokens
        return averageLoss, math.exp(averageLoss)   # API 注释：math.exp(x) = e 的 x 次方；PPL = exp(平均损失)


# ═════════════════════════════ 预测 ═════════════════════════════
class TextPredictor:
    """<summary>训练完之后用：给一段上文，预测下一个字、或者一直往下续写。</summary>"""

    def __init__(self, model: nn.Module, vocab: CharVocabulary, device: torch.device) -> None:
        self._model = model
        self._vocab = vocab
        self._device = device

    def NextCharProbabilities(self, context: str) -> torch.Tensor:
        """<returns>下一个字是字表中每个字的概率，形状 (V,)，加起来等于 1。</returns>"""
        self._model.eval()                                     # API 注释：切到预测模式（关闭 Dropout）
        # API 注释：外面再套一层 [ ]：一条文本也要变成「只有一个样本的一批」，形状 (T,) → (1, T)；device= 直接把张量建在显卡上
        ids = torch.tensor([self._vocab.Encode(context)], dtype=torch.long, device=self._device)
        with torch.no_grad():                                  # API 注释：这段代码里不记录梯度，预测时不需要反向传播，省显存
            logits = self._model(ids)                          # (1, T, V)
        # API 注释：logits[0, -1]：第 0 个样本、最后一个位置，得到 (V,)；torch.softmax(x, dim=-1)：沿最后一维把分数变成和为 1 的概率
        return torch.softmax(logits[0, -1].float(), dim=-1)   # 只要最后一个位置：它看过整段上文，代表「下一个字」

    def TopNextChars(self, context: str, k: int = 5) -> List[Tuple[str, float]]:
        probabilities = self.NextCharProbabilities(context)
        top = probabilities.topk(k)                            # 取概率最大的 k 个：.values 是概率，.indices 是编号
        # API 注释：zip(甲, 乙) 把两个序列按位置配成一对对 (概率, 编号)；int()/float() 把单个数的张量转成 Python 数；
        #   round(x, 3) 保留 3 位小数；.replace("\n", "⏎") 把换行符换成可见的符号
        return [(self._vocab.ToChar(int(i)).replace("\n", "⏎"), round(float(p), 3))
                for p, i in zip(top.values, top.indices)]

    def Generate(self, prompt: str, maxChars: int = 20) -> str:
        """<summary>贪心续写：每次取概率最大的字接到后面，遇到换行（问句结束）就停。</summary>"""
        text = prompt
        for _ in range(maxChars):
            # API 注释：张量.argmax()：最大值所在的位置（编号）；这里取概率最大的那个字
            nextChar = self._vocab.ToChar(int(self.NextCharProbabilities(text).argmax()))
            if nextChar == "\n":
                break
            text += nextChar
        return text


# ═════════════════════════════ 入口 ═════════════════════════════
class Program:
    """<summary>程序入口，相当于 C# 的 class Program。</summary>"""

    # API 注释：__file__ 是本脚本的路径；os.path.abspath 转成绝对路径；os.path.dirname 取它所在的文件夹；这样在哪个目录运行都能找到旁边的 data
    ScriptDir: str = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def ParseArgs() -> Config:
        # API 注释：argparse.ArgumentParser(...)：造命令行解析器；add_argument("--名字", type, default, help) 声明一个开关；
        #   type=int 把命令行里的字符串转成整数；os.path.join 按系统规则拼路径
        parser = argparse.ArgumentParser(description="用 BERT（加因果掩码）训练字符级语言模型")
        parser.add_argument("--corpus", default=os.path.join(Program.ScriptDir, "data", "corpus_general.txt"))
        parser.add_argument("--save", default=os.path.join(Program.ScriptDir, "checkpoints", "bert_lm_best.pt"))
        parser.add_argument("--pretrained", default=Config.PretrainedName, help="Hugging Face 模型名或本地文件夹")
        parser.add_argument("--epochs", type=int, default=Config.Epochs)
        parser.add_argument("--train-stride", type=int, default=Config.TrainStride,
                            help="训练集切块步长：1=密集重叠（最慢），64=不重叠")
        parser.add_argument("--batch-size", type=int, default=Config.BatchSize)
        parser.add_argument("--lr", type=float, default=Config.LearningRate)
        # API 注释：parse_args()：真正去读命令行，返回一个对象，用 args.epochs 取值
        args = parser.parse_args()                             # --train-stride → args.train_stride（短横线变下划线）
        return Config(CorpusPath=args.corpus, SavePath=args.save, PretrainedName=args.pretrained,
                      Epochs=args.epochs, TrainStride=args.train_stride, BatchSize=args.batch_size,
                      LearningRate=args.lr)

    @staticmethod
    def Main() -> None:
        config = Program.ParseArgs()
        torch.manual_seed(config.Seed)      # API 注释：固定 PyTorch 的随机数起点（初始化、DataLoader 打乱顺序、Dropout 都用它），结果可复现
        # API 注释：torch.cuda.is_available()：有可用的 NVIDIA 显卡返回 True；torch.device("cuda") 表示显卡，"cpu" 表示处理器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}  model: BERT ({config.PretrainedName}) + 因果掩码")

        # ── 步骤 1：读语料、建字表、划分 ──
        text = CorpusRepository.LoadText(config.CorpusPath)
        vocab = CharVocabulary.Build(text, config.MaxVocab)
        trainText, valText = CorpusRepository.SplitByLines(text, config.ValRatio, config.Seed)
        print(f"语料字符数: {len(text):,}  词表大小: {vocab.Size}")

        # ── 步骤 2：字表映射到 BERT 编号 ──
        # API 注释：BertTokenizer.from_pretrained(文件夹)：加载分词器，会读文件夹里的 vocab.txt（BERT 的 21128 个字）
        tokenizer = BertTokenizer.from_pretrained(config.PretrainedName)
        toBertIds, unknownCount = BertIdMapper.Build(vocab, tokenizer)
        print(f"映射到 BERT 字表：{vocab.Size - unknownCount} 个字有对应，{unknownCount} 个字只能当 [UNK] 输入")

        # ── 步骤 3：数据集 ──
        trainSet = CharLmDataset(trainText, vocab, config.SeqLen, config.TrainStride)
        valSet = CharLmDataset(valText, vocab, config.SeqLen, config.SeqLen)          # 验证集固定不重叠切块
        # API 注释：DataLoader(数据集, batch_size, shuffle, drop_last)：按批取数据；shuffle=True 每个 epoch 重新打乱；
        #   drop_last=True 丢掉最后凑不满一批的样本；遍历它每次得到 (x, y)，形状 (B, L)；len(loader) 是一共几批
        trainLoader = DataLoader(trainSet, batch_size=config.BatchSize, shuffle=True, drop_last=True)
        valLoader = DataLoader(valSet, batch_size=config.EvalBatchSize, shuffle=False, drop_last=True)
        print(f"训练样本 {len(trainSet):,}（步长 {config.TrainStride}，每轮 {len(trainLoader):,} 批）"
              f" / 验证样本 {len(valSet):,}（不重叠，{len(valLoader)} 批）")

        # ── 步骤 4：模型 ──
        # API 注释：模型.to(device)：把模型所有参数搬到显卡；返回模型本身，所以可以接在构造函数后面一句写完
        model = BertCausalLm(config.PretrainedName, toBertIds, tokenizer.unk_token_id).to(device)
        # API 注释：model.parameters() 依次给出所有参数张量，p.numel() 是每个张量的元素个数，sum 全部加起来 = 总参数量
        parameterCount = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {parameterCount:,}")

        # ── 步骤 5：训练 ──
        trainer = Trainer(model, config, totalSteps=config.Epochs * len(trainLoader), device=device)
        startLoss, startPpl = trainer.RunEpoch(valLoader, train=False)
        print(f"训练前验证集  Loss {startLoss:.4f}  PPL {startPpl:.2f}")

        # API 注释：os.makedirs(文件夹, exist_ok=True)：创建文件夹（含上级），已存在也不报错；
        #   `A or "."`：A 为空字符串时用 "."（当前目录）；float("inf") 是正无穷，作为「还没有最好成绩」的初始值，任何 PPL 都比它小
        os.makedirs(os.path.dirname(config.SavePath) or ".", exist_ok=True)
        bestValPpl = float("inf")
        # API 注释：f-string 里的格式：{x:>6} 右对齐占 6 格；{x:>10.4f} 右对齐占 10 格、保留 4 位小数；{x:,} 数字每三位加逗号；{x:>5.0f} 保留 0 位小数
        print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'用时':>6}")
        print("-" * 66)
        for epoch in range(1, config.Epochs + 1):
            startTime = time.time()                            # API 注释：记下开始时间（秒）；结束时用 time.time() - startTime 得到这一轮用了几秒
            trainLoss, trainPpl = trainer.RunEpoch(trainLoader, train=True)
            valLoss, valPpl = trainer.RunEpoch(valLoader, train=False)
            marker = ""
            if valPpl < bestValPpl:                            # 只保存验证 PPL 最好的那一版
                bestValPpl = valPpl
                marker = "  *"
                # API 注释：torch.save(对象, 路径)：把一个字典存成文件；model.state_dict() 是「参数名 → 张量」的有序字典，即全部学到的权重；
                #   和字表、预训练名字一起存，以后加载模型时才不会对不上
                torch.save({"model_state": model.state_dict(),
                            "chars": vocab.ToList(),
                            "pretrained": config.PretrainedName,
                            "seq_len": config.SeqLen}, config.SavePath)
            print(f"{epoch:>6}  {trainLoss:>10.4f}  {trainPpl:>10.2f}  {valLoss:>10.4f}  {valPpl:>10.2f}"
                  f"  {time.time() - startTime:>5.0f}s{marker}")
        print(f"\n训练完成。最佳验证 PPL: {bestValPpl:.2f}  已保存至 {config.SavePath}")
        print("参考（同一份语料、同一个划分、同样的评估方式）：LSTM 38.32 / RNN 40.83 / 2-gram 88.7 / 3-gram 247.1")

        # ── 步骤 6：看看它学到了什么 ──
        predictor = TextPredictor(model, vocab, device)
        print("\n预测下一个字 / 贪心续写：")
        for prompt in ("今天天气", "我想", "请问", "怎么才能", "银行卡"):
            print(f"  {prompt}: {predictor.TopNextChars(prompt)}  →  {predictor.Generate(prompt)}")


if __name__ == "__main__":
    Program.Main()
