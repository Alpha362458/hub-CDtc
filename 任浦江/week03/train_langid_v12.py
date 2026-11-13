import argparse  # 读取 --epochs 等命令行选项
import csv  # 按列名读取 CSV
import unicodedata  # 统一字符写法并查询 Unicode 类别
from collections import Counter  # 统计每个字符出现了多少次
from copy import deepcopy  # 独立复制最佳轮次的权重
from pathlib import Path  # 处理文件路径

import torch
from torch import nn  # Embedding、RNN、损失函数等组件
from torch.utils.data import DataLoader, Dataset  # 取单条样本、组装批次

# __file__ 是这个 .py 文件的路径。resolve() 得到绝对路径，parent 取所在目录。
HERE = Path(__file__).resolve().parent
LABELS = ["ar", "en", "es", "hi", "ja", "ko", "pt", "ru", "th", "tr", "vi", "zh"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}  # 例如 zh -> 11

PAD_ID, UNK_ID = 0, 1  # 0 为空位置，1 为字符表里查不到的字符
MAX_LEN = 40          # 每句话保留或补齐到 40 个位置
EMBED_DIM = 64        # 一个字符查出来的向量长度
HIDDEN_DIM = 64       # RNN 每读一步得到的状态长度
DROPOUT = 0.3         # 训练时每格特征被置零的概率
BATCH_SIZE = 64       # 一次用多少条训练样本更新模型
LR = 0.001           # Adam 的学习率


def read_split(data_dir, name):
    # name="train" 时打开 train.csv。用 UTF-8 读取多种文字，
    # newline="" 让 csv 模块按 CSV 的规则处理换行。
    with (data_dir / f"{name}.csv").open(encoding="utf-8", newline="") as f:
        # DictReader 用表头作为键。返回 [(句子, 标签), ...]。
        return [(row["text"], row["label"]) for row in csv.DictReader(f)]


def normalize(text):
    # NFKC 统一全角与半角等写法，lower() 转小写。
    text = unicodedata.normalize("NFKC", text).lower()
    # category(ch)[0] 取 Unicode 类别的首字母。L 为字母，M 为标记。
    # join 把留下的字符重新拼起来，字符之间不加分隔符。
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in {"L", "M"})


def build_vocab(train_rows):
    counts = Counter()
    for text, _ in train_rows:
        # 只数训练句子中的字符。这里暂时用不到标签，所以用 _ 接住。
        counts.update(normalize(text))

    vocab = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}
    # 次数多的先编号。次数相同时按字符排序，让编号顺序可复现。
    for ch in sorted(counts, key=lambda ch: (-counts[ch], ch)):
        vocab[ch] = len(vocab)  # 先有两个特殊 ID，所以第一个字符的 ID 是 2
    return vocab


def encode(text, vocab, maxlen=MAX_LEN):
    # 先清洗和截断，再逐字符查 ID。查不到就用 UNK_ID。
    ids = [vocab.get(ch, UNK_ID) for ch in normalize(text)[:maxlen]]
    if not ids:
        ids = [UNK_ID]  # 留一个有效位置，避免后面池化时整句全被遮掉
    # 右侧补上恰好缺少的个数，例如缺 2 位时加 [0, 0]。
    return ids + [PAD_ID] * (maxlen - len(ids))


class LangDataset(Dataset):
    def __init__(self, rows, vocab):
        # 创建数据集时预先把句子编码好。每行 40 个字符 ID。
        self.inputs = torch.tensor(
            [encode(text, vocab) for text, _ in rows], dtype=torch.long
        )
        # 用标签表查每句话的正确答案。dtype=torch.long 保存整数 ID。
        self.targets = torch.tensor(
            [LABEL_TO_ID[label] for _, label in rows], dtype=torch.long
        )

    def __len__(self):
        # 一句话对应一个标签，所以标签总数就是样本总数。
        return len(self.targets)

    def __getitem__(self, index):
        # DataLoader 给出样本下标，同时取出这句的输入和正确答案。
        return self.inputs[index], self.targets[index]


class LangIDClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        # 字符表每个 ID 对应一行，padding_idx 指定补齐行的 ID。
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        # 每步输入一个字符向量，输出一个新状态。批次轴放最前面。
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # LayerNorm 在每句话自己的 hidden_dim 个特征上计算。
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)  # dropout 参数就是置零概率 p
        # 每句的 hidden_dim 个特征 -> num_classes 个分数。
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x 是 [B, L] 的字符 ID。标记所有非 PAD 的位置，包括 UNK。
        valid = (x != PAD_ID).unsqueeze(-1)      # [B, L, 1]
        vectors = self.embedding(x)             # [B, L, E]，每格 ID 换成一条向量
        # RNN 返回每步的状态和最终状态。后面的池化使用每步的状态。
        states, _ = self.rnn(vectors)            # [B, L, H]
        # ~valid 选中补齐位置。把这些位置的所有特征填成负无穷。
        states = states.masked_fill(~valid, float("-inf"))
        # dim=1 是字符位置轴。每句话的每个特征各取一个最大值。
        pooled = states.max(dim=1).values       # [B, H]
        # 从括号最里面往外读。先归一化，再 Dropout，最后为每个语种打分。
        return self.fc(self.dropout(self.norm(pooled)))  # [B, C]，原始 logits


@torch.no_grad()  # 评估只需要前向计算，整个函数内不记录梯度
def evaluate(model, loader):
    model.eval()  # 让 Dropout 原样传递特征
    num_classes = len(LABELS)
    # 建一个全零的 12×12 表。行是真实类别，列是预测类别。
    matrix = [[0] * num_classes for _ in LABELS]

    for x, y in loader:
        # model(x) 是 [B, C]。沿 dim=1（类别轴）选最高分的下标，得到 [B]。
        predictions = model(x).argmax(dim=1)
        # tolist() 把张量转成列表，zip() 把同一句的正确编号和预测编号配成一对。
        for gold, pred in zip(y.tolist(), predictions.tolist()):
            matrix[gold][pred] += 1

    correct = sum(matrix[i][i] for i in range(num_classes))  # 对角线，答对数
    total = sum(sum(row) for row in matrix)                 # 所有格，样本总数
    f1s = []
    for i in range(num_classes):
        tp = matrix[i][i]                 # 真实与预测都是这一类
        actual = sum(matrix[i])           # 这一行加总，实际有多少条这一类
        predicted = sum(row[i] for row in matrix)  # 这一列加总，预测成这一类有多少条
        # 分母为 0 时，把这一类的 F1 记为 0。
        f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
    return correct / total, sum(f1s) / num_classes, matrix


def train_epoch(model, loader, loss_fn, optimizer):
    model.train()  # 上次验证可能切到了 eval，每轮开始重新开启训练模式
    loss_sum, total = 0.0, 0

    for x, y in loader:
        optimizer.zero_grad()      # 清除上一批的梯度
        logits = model(x)          # 前向计算，得到 [B, 12] 的原始分数
        loss = loss_fn(logits, y)  # 交叉熵在内部处理 logits，y 是 [B] 的类别编号
        loss.backward()            # 算每个参数的梯度
        optimizer.step()           # Adam 根据梯度更新参数

        # loss 是批内平均值，先乘回本批样本数，再加到总损失里。
        loss_sum += loss.item() * y.numel()
        total += y.numel()

    return loss_sum / total  # 按样本数算一整轮的平均损失


def main():
    # 命令行里没传某个选项时，用 default 给出的值。
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data" / "langid" / "v12")
    parser.add_argument("--save", type=Path, default=HERE / "langid_rnn_v12.pt")
    parser.add_argument("--dev-only", action="store_true")  # 出现此选项就跳过测试评分
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 至少为 1")

    torch.manual_seed(42)      # 设定 PyTorch 的随机种子，便于复现
    torch.set_num_threads(1)   # 这个小模型在 CPU 上用一个计算线程

    # 先读三份数据。下面检查标签集合，及时发现误传 v8 目录等情况。
    train_rows = read_split(args.data_dir, "train")
    dev_rows = read_split(args.data_dir, "dev")
    test_rows = read_split(args.data_dir, "test")
    for name, rows in [("train", train_rows), ("dev", dev_rows), ("test", test_rows)]:
        found = {label for _, label in rows}
        if found != set(LABELS):
            # 集合相减，找出缺的标签和多出来的标签。
            raise ValueError(
                f"{name} 标签不匹配，缺少 {sorted(set(LABELS) - found)}，"
                f"多出 {sorted(found - set(LABELS))}"
            )

    # 字符表只由训练集决定。验证、测试也用同一张表查 ID。
    vocab = build_vocab(train_rows)
    train_loader = DataLoader(LangDataset(train_rows, vocab), batch_size=BATCH_SIZE, shuffle=True)
    # 评估只做前向和统计，所以这里可以一次取 128 条，按原顺序读取。
    dev_loader = DataLoader(LangDataset(dev_rows, vocab), batch_size=128)
    test_loader = DataLoader(LangDataset(test_rows, vocab), batch_size=128)

    # len(vocab) 决定 Embedding 有多少行，len(LABELS) 决定每句有几个分数。
    model = LangIDClassifier(len(vocab), len(LABELS))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")
    print(f"classes={len(LABELS)} vocab={len(vocab)}")
    # 准确率至少为 0，所以第一轮一定会比初始值 -1 高并保存快照。
    best_acc, best_state, best_epoch = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        # evaluate 返回三个结果。这里用 _ 接住暂时用不到的混淆矩阵。
        dev_acc, dev_f1, _ = evaluate(model, dev_loader)
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} "
              f"dev_acc={dev_acc:.4f} dev_macro_f1={dev_f1:.4f}")
        if dev_acc > best_acc:
            best_acc = dev_acc
            best_state = deepcopy(model.state_dict())  # 留一份独立的参数快照
            best_epoch = epoch

    # 用验证成绩选定的快照，作为之后测试和保存的模型。
    model.load_state_dict(best_state)
    print(f"best_epoch={best_epoch} best_dev_acc={best_acc:.4f}")
    if not args.dev_only:
        test_acc, test_f1, _ = evaluate(model, test_loader)
        print(f"test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),  # 已训练的层参数
            "vocab": vocab,                    # 预测时把字符查回相同的 ID
            "labels": LABELS,                  # 预测时把最高分的编号换回语种
            "maxlen": MAX_LEN,                 # 预测时沿用训练的截断和补齐长度
            "embed_dim": EMBED_DIM,            # 重建 Embedding 所需的向量长度
            "hidden_dim": HIDDEN_DIM,          # 重建 RNN 所需的状态长度
            "dropout": DROPOUT,                # 重建模型时沿用的置零概率
        },
        args.save,  # 保存到这个路径
    )
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()  # 从终端直接运行此文件时，从这里进入训练流程
