# coding: utf-8
"""
Week03 作业：12 种语言的「语种识别」

任务：给一句话，判断它是 12 种语言中的哪一种
      ar 阿拉伯 / en 英 / es 西 / hi 印地 / ja 日 / ko 韩 / pt 葡 / ru 俄 / th 泰 / tr 土 / vi 越 / zh 中
模型：Embedding → RNN → 掩码最大池化 → LayerNorm → Dropout → Linear
损失：CrossEntropyLoss        优化器：Adam

和课堂 8 类版相比，只改了三处：
  1. 语种列表 Languages.Labels：8 个 → 12 个（新增 ko th vi tr）
  2. 数据：读 data/langid/v12
  3. 字符表：课件没有提供 12 类的表，这里用 12 类训练集自己生成（Vocabulary.Build）

代码结构（写法尽量贴近 C#：一个类管一件事、方法大写开头、局部变量小驼峰、私有字段加下划线）：
  Config           设置（相当于 C# 的配置类）
  Languages        语种列表（相当于 static class + 常量）
  TextNormalizer   文本清洗
  Vocabulary       字符表：生成 / 编码 / 保存
  DataRepository   读 CSV
  LangIdDataset    数据集（实现 __len__、__getitem__，相当于实现接口）
  LangIdClassifier 模型（继承 nn.Module，相当于 class X : Module）
  Evaluator        评测：准确率 / 混淆矩阵 / P-R-F1 / 难度切片
  Trainer          训练循环
  Program          入口，Main() 把上面的类串起来

运行（在本文件所在目录）：
    python train_langid_12.py
    python train_langid_12.py --data-dir <langid 数据目录> --epochs 10
    python train_langid_12.py --save langid12.pt --save-vocab vocab_v12.txt

数据：已复制到本目录的 data/langid/（v12/train.csv、v12/dev.csv、v12/test.csv、test_hard.csv，
      来源 Tatoeba，CC-BY 许可，说明见 data/langid/DATA_CARD.md），默认就读这里。
--data-dir 可以改成别处的数据目录，里面同样要有  v12/train.csv、v12/dev.csv、v12/test.csv 和 test_hard.csv。
"""

import argparse
import csv
import os
import random
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 一行数据：(句子, 语种)
Row = Tuple[str, str]


# ═════════════════════════════ 设置 ═════════════════════════════
@dataclass(frozen=True)
class Config:
    """
    <summary>所有可调的设置集中放在这里。</summary>
    frozen=True 表示创建后不能再改（相当于 C# 的 readonly / record）。
    """
    DataDir: str                   # 数据目录（里面有 v12 文件夹和 test_hard.csv）
    SavePath: str = ""             # 保存模型的路径，空字符串 = 不保存
    SaveVocabPath: str = ""        # 保存字符表的路径，空字符串 = 不保存
    Epochs: int = 8                # 把训练集完整过几遍
    NumThreads: int = 1            # CPU 线程数（小模型单线程反而更快）
    Seed: int = 42                 # 随机种子：固定后初始权重、打乱顺序都可复现
    MaxLen: int = 40               # 每句统一截断/补齐到 40 个字符
    EmbedDim: int = 64             # d：每个字符用 64 个数表示
    HiddenDim: int = 64            # H：RNN 的"便签"有 64 个数
    Dropout: float = 0.3           # 训练时每个数有 30% 的概率被置 0
    BatchSize: int = 64            # 训练时每批 64 句
    EvalBatchSize: int = 128       # 验证/测试时每批 128 句（不算梯度，可以大一点）
    LearningRate: float = 1e-3     # Adam 的学习率


class Languages:
    """<summary>语种列表。列表下标就是类别编号：ar→0, en→1, ...</summary>"""
    Labels: List[str] = ["ar", "en", "es", "hi", "ja", "ko",
                         "pt", "ru", "th", "tr", "vi", "zh"]

    @staticmethod
    def LabelToId() -> Dict[str, int]:
        """<returns>{"ar": 0, "en": 1, ...}</returns>"""
        return {label: i for i, label in enumerate(Languages.Labels)}


# ═════════════════════════════ 文本与字符表 ═════════════════════════════
class TextNormalizer:
    """<summary>文本清洗。必须和建数据集时用的清洗方式一致。</summary>"""

    @staticmethod
    def Normalize(text: str) -> str:
        """
        <summary>NFKC 归一化 + 转小写 + 只保留「字母(L)」和「组合符号(M)」。</summary>
        标点、数字、空格、emoji 都会被丢掉。保留 M 是因为印地语、泰语的元音是组合符号。
        """
        # NFKC：把"长得不同、含义相同"的字符统一，例如全角 Ａ → A
        lowered = unicodedata.normalize("NFKC", text).lower()
        # unicodedata.category(c) 返回类别码（两个字母），[0] 取大类：L 字母、M 组合符号、N 数字、P 标点、S 符号
        return "".join(c for c in lowered if unicodedata.category(c)[0] in ("L", "M"))


class Vocabulary:
    """
    <summary>字符表：字符 ↔ 整数编号。0 号 &lt;PAD&gt; 补齐，1 号 &lt;UNK&gt; 没见过的字符。</summary>
    其余字符按出现次数从高到低编号；次数相同按字符编码排序，保证每次结果一样。
    """
    PadToken: str = "<PAD>"
    UnkToken: str = "<UNK>"
    PadId: int = 0
    UnkId: int = 1

    def __init__(self, charToId: Dict[str, int], counts: Dict[str, int]) -> None:
        self._charToId: Dict[str, int] = charToId
        self._counts: Dict[str, int] = counts

    @property
    def Size(self) -> int:
        """<summary>字符表里一共有几个字符（含 PAD 和 UNK）。</summary>"""
        return len(self._charToId)

    @staticmethod
    def Build(sentences: Sequence[str]) -> "Vocabulary":
        """<summary>从一批句子统计出字符表。<br/>课件没有提供 12 类的字符表，所以只能自己生成。</summary>"""
        counter: Counter = Counter()
        for sentence in sentences:
            counter.update(TextNormalizer.Normalize(sentence))   # 字符串会被拆成一个个字符并累加次数

        # 先按出现次数从高到低，次数相同再按字符本身排序
        ordered = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))

        charToId: Dict[str, int] = {Vocabulary.PadToken: Vocabulary.PadId,
                                    Vocabulary.UnkToken: Vocabulary.UnkId}
        counts: Dict[str, int] = {Vocabulary.PadToken: 0, Vocabulary.UnkToken: 0}
        for ch, n in ordered:
            charToId[ch] = len(charToId)     # len() 就是"目前已有几个"，正好是下一个空号
            counts[ch] = n
        return Vocabulary(charToId, counts)

    def Encode(self, text: str, maxLen: int) -> List[int]:
        """
        <summary>一句话 → 长度恒为 maxLen 的编号列表。</summary>
        三步：换编号（查不到给 UNK）→ 截断到 maxLen → 不够就在末尾补 PAD。
        """
        cleaned = TextNormalizer.Normalize(text)
        ids = [self._charToId.get(ch, Vocabulary.UnkId) for ch in cleaned][:maxLen]
        ids += [Vocabulary.PadId] * (maxLen - len(ids))
        return ids

    def ToDict(self) -> Dict[str, int]:
        """<summary>返回 {字符: 编号} 的拷贝，用来存进模型文件。</summary>"""
        return dict(self._charToId)

    def Save(self, path: str) -> None:
        """<summary>保存成文本文件，格式和课堂的 vocab_v8.txt 一样：每行「字符 TAB 编号 TAB 出现次数」。</summary>"""
        with open(path, "w", encoding="utf-8", newline="\n") as writer:
            writer.write("# 字符\tid\t出现次数\n")
            for ch, idx in self._charToId.items():
                writer.write(f"{ch}\t{idx}\t{self._counts[ch]}\n")


# ═════════════════════════════ 数据 ═════════════════════════════
class DataRepository:
    """<summary>负责读 CSV 文件。</summary>"""

    @staticmethod
    def LoadSplit(dataDir: str, name: str) -> List[Row]:
        """<summary>读 v12/{name}.csv，得到 [(句子, 语种), ...]。</summary>"""
        path = os.path.join(dataDir, "v12", f"{name}.csv")
        rows: List[Row] = []
        with open(path, encoding="utf-8") as reader:
            # DictReader 用第一行表头 text,label 当键，每一行读成字典
            for record in csv.DictReader(reader):
                rows.append((record["text"], record["label"]))
        return rows

    @staticmethod
    def LoadHardRows(dataDir: str) -> List[Dict[str, str]]:
        """<summary>读难度诊断集 test_hard.csv（比普通测试集多一列 slice）。</summary>"""
        path = os.path.join(dataDir, "test_hard.csv")
        with open(path, encoding="utf-8") as reader:
            return list(csv.DictReader(reader))


class LangIdDataset(Dataset):
    """
    <summary>数据集：第 i 条 = (长度 MaxLen 的编号张量, 标签整数)。</summary>
    DataLoader 只要求实现两个方法：__len__（一共几条）和 __getitem__（取第 i 条），
    相当于 C# 里实现一个接口。
    """

    def __init__(self, rows: Sequence[Row], vocab: Vocabulary,
                 labelToId: Dict[str, int], maxLen: int) -> None:
        # 在构造时一次性把所有句子编码好，之后取数据就很快
        self._inputs: List[List[int]] = [vocab.Encode(text, maxLen) for text, _ in rows]
        self._targets: List[int] = [labelToId[label] for _, label in rows]

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # dtype=torch.long：Embedding 要求输入是整数编号
        return torch.tensor(self._inputs[index], dtype=torch.long), self._targets[index]


# ═════════════════════════════ 模型 ═════════════════════════════
class LangIdClassifier(nn.Module):
    """
    <summary>Embedding → RNN → 掩码最大池化 → LayerNorm → Dropout → Linear。</summary>
    形状（B=批大小，L=句长，d=EmbedDim，H=HiddenDim，C=类别数）：
        输入 (B, L) → (B, L, d) → (B, L, H) → (B, H) → (B, H) → (B, H) → (B, C)
    """

    def __init__(self, vocabSize: int, numClasses: int,
                 embedDim: int, hiddenDim: int, dropout: float) -> None:
        super().__init__()   # 必须先初始化父类，后面的层才会被登记、参数才找得到
        # Embedding(V, d)：一张 (V, d) 的可训练查找表，输入编号、取出对应的行
        # padding_idx：PAD 那一行恒为 0，并且训练时不被更新
        self._embedding = nn.Embedding(vocabSize, embedDim, padding_idx=Vocabulary.PadId)
        # RNN：h_t = tanh(W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh)，一个字一个字往后读
        # batch_first=True：输入输出按 (B, L, ·) 排，和 Embedding 的输出刚好接上
        self._rnn = nn.RNN(embedDim, hiddenDim, batch_first=True)
        # LayerNorm：对每个样本自己的 H 个数做 (x − 均值) / √(方差 + ε) · γ + β
        self._norm = nn.LayerNorm(hiddenDim)
        # Dropout：训练时随机置 0 并放大 1/(1−p)；model.eval() 之后什么也不做
        self._dropout = nn.Dropout(dropout)
        # Linear(H, C)：y = x · Wᵀ + b，输出每个类别的分数 logits
        self._classifier = nn.Linear(hiddenDim, numClasses)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """<summary>model(x) 会自动调用本方法。</summary><returns>未经 softmax 的 logits，形状 (B, C)。</returns>"""
        # (B, L, 1)：编号不是 PAD 的位置为 True
        notPad = (x != Vocabulary.PadId).unsqueeze(-1)
        hidden, _ = self._rnn(self._embedding(x))           # (B, L, H)，每一步一张"便签"
        # PAD 位置的 RNN 输出不是 0，所以要填成极小的数，取最大值时才不会选到它们（不能填 0）
        hidden = hidden.masked_fill(~notPad, -1e9)
        pooled = hidden.max(dim=1)[0]                       # 沿句长取最大 → (B, H)
        pooled = self._dropout(self._norm(pooled))
        # 这里不做 softmax：CrossEntropyLoss 内部已经包含 softmax
        return self._classifier(pooled)


# ═════════════════════════════ 评测 ═════════════════════════════
class Evaluator:
    """<summary>评测相关的静态方法。</summary>"""

    @staticmethod
    def Predict(model: nn.Module, loader: DataLoader) -> Tuple[List[int], List[int]]:
        """<returns>(预测的类别编号列表, 真实的类别编号列表)</returns>"""
        model.eval()                  # 切到预测模式（Dropout 不再随机丢弃）
        preds: List[int] = []
        golds: List[int] = []
        with torch.no_grad():         # 评估不需要梯度，省内存
            for inputs, targets in loader:
                preds += model(inputs).argmax(dim=1).tolist()   # 每行取分数最高的类别
                golds += targets.tolist()
        return preds, golds

    @staticmethod
    def PredictTexts(model: nn.Module, vocab: Vocabulary, texts: Sequence[str], maxLen: int) -> List[int]:
        """<summary>直接预测一批句子（用于难度切片）。</summary>"""
        model.eval()
        inputs = torch.tensor([vocab.Encode(t, maxLen) for t in texts], dtype=torch.long)
        with torch.no_grad():
            return model(inputs).argmax(dim=1).tolist()

    @staticmethod
    def Accuracy(preds: Sequence[int], golds: Sequence[int]) -> float:
        """<summary>准确率 = 预测对的条数 ÷ 总条数。</summary>"""
        correct = sum(1 for p, g in zip(preds, golds) if p == g)
        return correct / len(golds)

    @staticmethod
    def ConfusionMatrix(preds: Sequence[int], golds: Sequence[int], numClasses: int) -> List[List[int]]:
        """<summary>混淆矩阵：行 = 真实类别，列 = 预测类别；对角线是答对的。</summary>"""
        matrix = [[0] * numClasses for _ in range(numClasses)]
        for p, g in zip(preds, golds):
            matrix[g][p] += 1
        return matrix

    @staticmethod
    def PerClassMetrics(matrix: List[List[int]]) -> List[Tuple[int, float, float, float]]:
        """
        <summary>逐类算 (样本数, precision, recall, F1)。</summary>
            precision = TP / (TP + FP)    预测为 c 的里面有多少是对的
            recall    = TP / (TP + FN)    真的是 c 的里面找出了多少
            F1        = 2 · P · R / (P + R)
        """
        n = len(matrix)
        result: List[Tuple[int, float, float, float]] = []
        for c in range(n):
            tp = matrix[c][c]
            fn = sum(matrix[c]) - tp                          # 这一行其余的：真是 c 却没预测成 c
            fp = sum(matrix[g][c] for g in range(n)) - tp     # 这一列其余的：预测成 c 但其实不是
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            result.append((tp + fn, precision, recall, f1))
        return result

    @staticmethod
    def PrintReport(preds: Sequence[int], golds: Sequence[int], labels: Sequence[str], top: int = 6) -> None:
        """<summary>打印总体指标、逐类指标和最常混淆的组合。</summary>"""
        matrix = Evaluator.ConfusionMatrix(preds, golds, len(labels))
        metrics = Evaluator.PerClassMetrics(matrix)
        macroF1 = sum(m[3] for m in metrics) / len(metrics)   # macro-F1：各类 F1 的平均，每类权重相同
        print(f"  accuracy = {Evaluator.Accuracy(preds, golds):.4f}   macro-F1 = {macroF1:.4f}")

        print(f"\n  {'语种':<6}{'样本':>6}{'precision':>12}{'recall':>10}{'f1':>10}")
        for label, (count, precision, recall, f1) in zip(labels, metrics):
            print(f"  {label:<6}{count:>6}{precision:>12.3f}{recall:>10.3f}{f1:>10.3f}")

        # 只看错的格子（g != p），按错误条数从多到少排序
        mistakes = [(matrix[g][p], labels[g], labels[p])
                    for g in range(len(labels)) for p in range(len(labels))
                    if g != p and matrix[g][p] > 0]
        mistakes.sort(reverse=True)
        print("\n  最常混淆的组合（真实 → 预测）：")
        for count, gold, pred in mistakes[:top]:
            print(f"    {gold} → {pred}: {count} 条")

    @staticmethod
    def EvaluateSlices(model: nn.Module, vocab: Vocabulary, hardRows: Sequence[Dict[str, str]],
                       labelToId: Dict[str, int], maxLen: int) -> None:
        """
        <summary>难度诊断：按 slice 分组算准确率。</summary>
            short      只有 2~7 个字符的短句
            nodeaccent 去掉重音符号
            nokana     日语去掉全部假名，只剩汉字
        """
        for sliceName in ("short", "nodeaccent", "nokana"):
            subset = [r for r in hardRows if r["slice"] == sliceName and r["label"] in labelToId]
            preds = Evaluator.PredictTexts(model, vocab, [r["text"] for r in subset], maxLen)
            golds = [labelToId[r["label"]] for r in subset]
            print(f"  [{sliceName:<10}] {len(subset):4d} 条  accuracy = {Evaluator.Accuracy(preds, golds):.3f}")


# ═════════════════════════════ 训练 ═════════════════════════════
class Trainer:
    """<summary>训练循环。</summary>"""

    def __init__(self, config: Config) -> None:
        self._config: Config = config

    def Train(self, model: nn.Module, trainLoader: DataLoader, devLoader: DataLoader) -> None:
        criterion = nn.CrossEntropyLoss()
        # Adam：m = β1·m + (1−β1)·g；v = β2·v + (1−β2)·g²；w ← w − lr · m̂ / (√v̂ + ε)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._config.LearningRate)

        startTime = time.time()
        for epoch in range(1, self._config.Epochs + 1):    # epoch 从 1 开始数
            model.train()            # 训练模式：Dropout 生效（Predict 会把模型切成 eval，所以每轮都要重新设）
            totalLoss = 0.0
            for inputs, targets in trainLoader:            # 每一轮取一批 (B, MaxLen) 和 (B,)
                logits = model(inputs)
                loss = criterion(logits, targets)          # −log(softmax(logits)[正确类别])，对这一批取平均
                optimizer.zero_grad()    # 清空上一批的梯度（梯度会累加）
                loss.backward()          # 反向传播：算出每个参数的梯度（只算，不改参数）
                optimizer.step()         # 按 Adam 公式更新参数
                totalLoss += loss.item()

            preds, golds = Evaluator.Predict(model, devLoader)
            accuracy = Evaluator.Accuracy(preds, golds)
            averageLoss = totalLoss / len(trainLoader)     # len(loader) 是批数
            print(f"Epoch {epoch:2d}/{self._config.Epochs}  loss={averageLoss:.4f}  dev_acc={accuracy:.4f}")

        print(f"\n训练耗时：{time.time() - startTime:.0f} 秒")


# ═════════════════════════════ 入口 ═════════════════════════════
class Program:

    # 默认读本文件旁边的 data/langid（数据已复制到作业目录里，在哪个目录运行都能找到）；
    # 想用别处的数据，用 --data-dir 指定。
    # __file__ 是本脚本的路径；abspath 转成绝对路径；dirname 取它所在的文件夹；os.path.join 按系统拼路径
    DefaultDataDir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "langid")

    @staticmethod
    def ParseArgs() -> Config:
        """<summary>读取命令行参数，返回 Config。</summary>"""
        parser = argparse.ArgumentParser(description="12 种语言的语种识别")
        parser.add_argument("--data-dir", default=Program.DefaultDataDir,
                            help="数据目录（里面有 v12 文件夹和 test_hard.csv）")
        parser.add_argument("--epochs", type=int, default=Config.Epochs)     # type=int：把命令行的字符串转成整数
        parser.add_argument("--save", default="", help="保存模型的路径（含字符表和语种列表）")
        parser.add_argument("--save-vocab", default="", help="把字符表另存成文本文件")
        args = parser.parse_args()           # 参数名里的短横线会变成下划线：--data-dir → args.data_dir
        return Config(DataDir=args.data_dir, Epochs=args.epochs,
                      SavePath=args.save, SaveVocabPath=args.save_vocab)

    @staticmethod
    def CheckLabels(rows: Sequence[Row], labels: Sequence[str]) -> None:
        """<summary>数据里的语种必须都在 Languages.Labels 里，否则模型的输出格子不够用。</summary>"""
        seen = sorted({label for _, label in rows})                 # 集合推导式：自动去重
        unknown = [label for label in seen if label not in labels]
        if unknown:
            raise SystemExit(f"数据里出现了 Labels 之外的语种：{' '.join(unknown)}")

    @staticmethod
    def Main() -> None:
        config = Program.ParseArgs()
        labels = Languages.Labels
        labelToId = Languages.LabelToId()
        numClasses = len(labels)

        # ── 步骤 1：环境与随机种子 ──
        torch.set_num_threads(config.NumThreads)
        random.seed(config.Seed)             # 固定 Python 的随机数
        torch.manual_seed(config.Seed)       # 固定 PyTorch 的随机数（初始权重、打乱顺序、Dropout）

        # ── 步骤 2：读数据 ──
        if not os.path.isdir(os.path.join(config.DataDir, "v12")):
            raise SystemExit(f"找不到数据目录：{config.DataDir}\\v12，请用 --data-dir 指定")
        trainRows = DataRepository.LoadSplit(config.DataDir, "train")
        devRows = DataRepository.LoadSplit(config.DataDir, "dev")
        testRows = DataRepository.LoadSplit(config.DataDir, "test")
        Program.CheckLabels(trainRows, labels)

        # ── 步骤 3：用训练集生成字符表（只用训练集，验证/测试里没见过的字符才会走 UNK）──
        vocab = Vocabulary.Build([text for text, _ in trainRows])
        if config.SaveVocabPath:
            vocab.Save(config.SaveVocabPath)

        print(f"数据目录：{config.DataDir}")
        print(f"  训练 {len(trainRows)} 条 / 验证 {len(devRows)} 条 / 测试 {len(testRows)} 条")
        print(f"  类别 {numClasses} 个：{' '.join(labels)}")
        print(f"  字符表 {vocab.Size} 个字符")
        print(f"  随机猜的准确率基准：{1 / numClasses:.1%}")

        # ── 步骤 4：数据集与加载器 ──
        trainLoader = DataLoader(LangIdDataset(trainRows, vocab, labelToId, config.MaxLen),
                                 batch_size=config.BatchSize, shuffle=True)    # 训练集每个 epoch 重新打乱
        devLoader = DataLoader(LangIdDataset(devRows, vocab, labelToId, config.MaxLen),
                               batch_size=config.EvalBatchSize)
        testLoader = DataLoader(LangIdDataset(testRows, vocab, labelToId, config.MaxLen),
                                batch_size=config.EvalBatchSize)

        # ── 步骤 5：模型 ──
        model = LangIdClassifier(vocab.Size, numClasses, config.EmbedDim, config.HiddenDim, config.Dropout)
        parameterCount = sum(p.numel() for p in model.parameters())   # numel()：张量里数的个数
        print(f"  模型参数量：{parameterCount:,}\n")

        # ── 步骤 6：训练 ──
        Trainer(config).Train(model, trainLoader, devLoader)

        # ── 步骤 7：测试集评测 ──
        print("\n=== 测试集 ===")
        preds, golds = Evaluator.Predict(model, testLoader)
        Evaluator.PrintReport(preds, golds, labels)

        # ── 步骤 8：难度诊断 ──
        print("\n=== 难度诊断（test_hard）===")
        hardRows = DataRepository.LoadHardRows(config.DataDir)
        Evaluator.EvaluateSlices(model, vocab, hardRows, labelToId, config.MaxLen)

        # ── 步骤 9：保存（可选）──
        if config.SavePath:
            torch.save({"state_dict": model.state_dict(),    # 所有学到的权重：参数名 → 张量
                        "vocab": vocab.ToDict(),
                        "labels": labels,
                        "maxLen": config.MaxLen,
                        "embedDim": config.EmbedDim,
                        "hiddenDim": config.HiddenDim,
                        "dropout": config.Dropout}, config.SavePath)
            print(f"\n模型已保存：{config.SavePath}")


if __name__ == "__main__":
    Program.Main()
