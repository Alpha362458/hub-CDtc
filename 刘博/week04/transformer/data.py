"""数据侧：语料加载、字表构建、训练/验证切分、字符级样本切块、字频先验。"""

import glob
import random
from collections import Counter

import torch
from torch.utils.data import Dataset


def load_corpus(pattern="*.txt"):
    """读取所有匹配 pattern 的 txt 文件并拼接为单个字符串。"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text, max_vocab=None):
    """
    构建字表：按字频降序取前 max_vocab 个字（None = 全部保留）。
    落在字表外的生僻字会被 CharDataset 过滤 —— 字表是覆盖度与模型大小的取舍。
    """
    counter = Counter(text)
    chars = [c for c, _ in counter.most_common(max_vocab)] if max_vocab else sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def split_train_val(text, val_ratio=0.05, seed=42):
    """
    按行打乱后切分训练/验证文本。以"行"（问句）为单位切而非按字符位置切，
    避免同一句话被腰斩到训练和验证两侧造成泄漏。固定 seed，
    train.py 与 evaluate.py 各自调用会得到同一份验证集。
    """
    lines = text.splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    split = int(len(lines) * (1 - val_ratio))
    return "\n".join(lines[:split]), "\n".join(lines[split:])


class CharDataset(Dataset):
    """
    字符级训练样本：x = 连续 seq_len 个字符，y = 同一起点后移一位（next-char 预测）。

    stride 为相邻样本起始位置的间隔：
        stride=1       逐字符滑动，样本大量重叠，数量 ≈ 语料长度，一轮训练极慢
        stride=seq_len 不重叠切块，数量 ≈ 语料长度 / seq_len（标准做法）
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


def build_unigram_logprob(pattern, char2idx, device):
    """统计语料字频，返回词表上的 log 概率（拉普拉斯平滑，避免未出现字 log(0)）。
    供拼音输入法给"没有上文的第一个字"打分用。"""
    counter = Counter()
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                counter.update(c for c in line if c in char2idx)

    probs = torch.ones(len(char2idx), dtype=torch.float)
    for c, n in counter.items():
        probs[char2idx[c]] += n
    probs = probs / probs.sum()
    return torch.log(probs).to(device)
