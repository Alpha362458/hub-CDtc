# 作业：用 Transformer 替换字符级语言模型里的 LSTM
# 原版 language_model.py 用 nn.LSTM / nn.RNN 做循环层，本文件改成因果自注意力 Transformer。
#
# 三个关键点(和 LSTM 版的区别，务必理解):
#   1. 因果掩码 causal mask
#      语言模型不能偷看未来。自注意力默认是每个位置看所有位置，
#      必须把上三角(未来位置)屏蔽掉。不屏蔽的话模型会直接看到答案，
#      PPL 会假得很低，但生成的时候完全不可用。
#   2. 位置编码
#      自注意力本身对顺序不敏感(把一句话打乱，输出的集合不变)，
#      所以必须显式加上位置向量，模型才知道谁在前谁在后。
#      LSTM 靠逐字滚动天然带顺序，Transformer 没有这个性质。
#   3. 没有隐藏状态、整句并行
#      LSTM 要沿着时间步一格一格算；Transformer 一次算完整个序列，
#      所以在 GPU 上快很多。但本机没有 GPU(只有 CPU 10 线程)，
#      Transformer 反而比 LSTM 慢，这里主要做原理对照实验。
#
# 配置(已写成默认值 = 推荐档: 200 万字 / 15 轮):
#   实测: 单轮 161 秒(200万字)，15 轮约 40 分钟；第 1 轮 Val PPL 已达 130.86
#   基准: 50万字 5轮 实测 121；预计跑满 15 轮可降到 55 到 70
#   stride = seq_len   不重叠切块:样本数从约600万降到约9万，理论快约64倍
#   limit  = 2000000   只取语料前200万字训练(字表仍按全语料建，保持4000)
#   epochs = 15
#   模型: embed=128 nhead=4 layers=2 ff=256 seq_len=64 batch=128
#
# 想跑完整语料做对比(慢):
#   python transformer_lm.py --limit 0 --epochs 20
# 想用最慢最准的密集切块:
#   python transformer_lm.py --stride 1 --limit 0 --epochs 20

import math
import time
import os
import glob
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = HERE  # 语料已拷入作业目录，与脚本同级，自包含
DEFAULT_SAVE = os.path.join(HERE, 'transformer_lm.pt')


def load_corpus(pattern):
    text = ''
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8', errors='ignore') as f:
            text += f.read()
    return text


def build_vocab(text, max_vocab):
    from collections import Counter
    chars = [c for c, _ in Counter(text).most_common(max_vocab)]
    char2idx = dict((c, i) for i, c in enumerate(chars))
    idx2char = dict((i, c) for c, i in char2idx.items())
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len, stride):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride + 1)

    def __getitem__(self, idx):
        s = idx * self.stride
        x = self.data[s:s + self.seq_len]
        y = self.data[s + 1:s + self.seq_len + 1]
        return x, y


class TransformerLM(nn.Module):
    # 字符级 Transformer 语言模型结构:
    # 词向量 + 位置向量 经过 N 层因果自注意力(每层含 FFN)，最后输出层预测下一个字
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, ff_dim, max_len, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.use_causal = True

    def forward(self, x, use_causal=None):
        if use_causal is None:
            use_causal = self.use_causal
        # 关键1: 位置编码。自注意力本身对顺序不敏感，必须显式把位置向量加进去
        T = x.size(1)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.embed(x) + self.pos(pos))
        if use_causal:
            # 关键2: 因果掩码。把上三角(未来位置)屏蔽掉，保证预测第 t 个字时只看得到前 t 个字
            causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            h = self.encoder(h, mask=causal)
        else:
            # 对照实验: 关掉因果掩码，模型能看到未来，训练 PPL 会假得很低但生成会崩坏
            h = self.encoder(h)
        return self.fc(self.drop(h))


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    if total_tokens == 0:
        return float('inf'), float('inf')
    avg = total_loss / total_tokens
    return avg, math.exp(avg)


def generate(model, char2idx, idx2char, prompt, n_tokens, temperature, device):
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if len(ids) == 0:
        ids = [0]
    model.eval()
    for _ in range(n_tokens):
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits = model(x)
        last = logits[0, -1] / max(temperature, 1e-06)
        probs = torch.softmax(last, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        ids.append(nxt)
    text = ''.join(idx2char.get(i, '?') for i in ids)
    return text.replace(chr(10), ' ')

def main():
    ap = argparse.ArgumentParser(description='字符级 Transformer 语言模型（把 LSTM 换成自注意力的作业）')

    # ── 训练规模：决定要跑多久 ──────────────────────────────
    # epochs 训练轮数：语料完整过一遍算一轮。调大 → PPL 继续降但收益递减；调小 → 快但欠拟合。
    #   实测 200 万字单轮 161 秒，默认 15 轮约 40 分钟。
    ap.add_argument('--epochs', type=int, default=15)
    # limit 只取语料前 N 个字符训练（0 = 不限，用完整 602 万字）。这是最有效的提速开关：
    #   默认 200 万字约为全量的 1/3。字表仍按全语料建立，所以词表还是 4000，PPL 可与全量实验对比。
    ap.add_argument('--limit', type=int, default=2000000, help='只取前N字符训练,0=不限')
    # stride 滑窗切块步长：0 表示用 seq_len（不重叠，默认最快）；1 表示每个字符都切一刀（密集重叠，
    #   样本数放大 64 倍，最慢但样本最多，最接近原 LSTM 实验）。调小 → 样本变多、更慢、通常 PPL 更低。
    ap.add_argument('--stride', type=int, default=0, help='0=用 seq_len(不重叠,默认快); 1=密集重叠(最慢最准)')

    # ── 序列与批次：决定上下文多长、一次算多少 ─────────────
    # seq_len 每条样本的字符长度，即模型最多能看多远的上下文。调大 → 长依赖更强，
    #   但注意力计算是 O(T 的平方)，CPU 上明显变慢；调小 → 快但上下文变短。
    ap.add_argument('--seq_len', type=int, default=64)
    # batch_size 一次前向加反传处理多少条样本。调大 → 梯度更稳、CPU 利用率更高，但内存占用上升；
    #   太大会内存溢出。CPU 上 128 到 256 通常比较划算。
    ap.add_argument('--batch_size', type=int, default=128)

    # ── 模型结构：决定参数量与表达能力（改这几个最影响 PPL 和速度）──
    # embed_dim 字向量维度，也是整个 Transformer 的主干宽度（d_model）。
    #   调大 → 表达力强、参数多、更慢；调小 → 快但容易欠拟合。常用 128 或 256。
    ap.add_argument('--embed_dim', type=int, default=128)
    # nhead 多头注意力的头数，必须能整除 embed_dim（128 除以 4 等于 32，即每头 32 维）。
    #   调大 → 可同时关注更多种关系；但头太多会让每头维度太小，反而不划算。
    ap.add_argument('--nhead', type=int, default=4)
    # num_layers 编码器层数（堆几层 自注意力 加 前馈）。调大 → 更深更强，但更慢；
    #   小数据上过深容易过拟合，2 层足以做本作业的对照。
    ap.add_argument('--num_layers', type=int, default=2)
    # ff_dim 每层前馈网络（FFN）的隐层宽度，通常取 embed_dim 的 2 到 4 倍。
    #   调大 → 每层非线性表达更强，参数与计算量同时上升。
    ap.add_argument('--ff_dim', type=int, default=256)
    # max_len 位置编码表的最大长度，必须不小于实际用到的序列长度（含生成时 prompt 加生成长度）。
    #   超了会索引越界报错，所以留 512 的余量。
    ap.add_argument('--max_len', type=int, default=512)
    # dropout 随机丢弃比例，用于抑制过拟合。调大 → 正则更强、训练 loss 变高但泛化可能更好；
    #   数据少或训练久时可试 0.2 到 0.3，短训练时 0.1 足够。
    ap.add_argument('--dropout', type=float, default=0.1)

    # ── 优化与数据划分 ─────────────────────────────────────
    # lr 学习率（Adam）。调大 → 收敛快但可能震荡甚至发散；调小 → 稳但慢。
    #   Transformer 对学习率比 LSTM 敏感，1e-3 是常见起点，不收敛就降到 3e-4。
    ap.add_argument('--lr', type=float, default=1e-3)
    # val_ratio 验证集比例。调大 → 评估更可信，但可用于训练的数据变少；0.05 是常见折中。
    ap.add_argument('--val_ratio', type=float, default=0.05)
    # max_vocab 词表最多保留多少字（按词频取前 N）。调大 → 覆盖更多生僻字，但输出层 softmax 更大更慢；
    #   4000 已覆盖绝大多数常用字，与 LSTM 版保持一致便于横向比较。
    ap.add_argument('--max_vocab', type=int, default=4000)

    # ── 复现与产物 ────────────────────────────────────────
    # seed 随机种子：固定后数据划分与权重初始化可复现。换一个值可判断结果是真实提升还是随机波动。
    ap.add_argument('--seed', type=int, default=42)
    # save 模型保存路径。每次验证 PPL 刷新最好成绩就覆盖保存；换路径可并存多个实验。
    ap.add_argument('--save', default=DEFAULT_SAVE)

    # ── 对照实验与生成模式 ─────────────────────────────────
    # no_causal 关掉因果掩码（仅用于对照实验）。打开后模型能看到未来，训练 PPL 会假低，但生成会崩坏。
    ap.add_argument('--no_causal', action='store_true', help='对照实验: 关掉因果掩码(偷看未来,PPL假低)')
    # generate 不进训练，直接加载 --save 里的模型做文本续写。
    ap.add_argument('--generate', action='store_true', help='生成模式: 加载 --save 里的模型续写')
    # prompt 生成模式的起始文字（会被编码成 id 序列再喂给模型）。
    ap.add_argument('--prompt', default='今天天气', help='生成起点文字')
    # n_tokens 一共生成多少个字（不含 prompt 本身）。
    ap.add_argument('--n_tokens', type=int, default=30)
    # temperature 采样温度：小于 1 更保守（更贴近高频字），大于 1 更发散；接近 0 就退化成贪心。
    ap.add_argument('--temperature', type=float, default=1.0)

    # 真正解析命令行，得到所有参数的最终取值（默认值或用户传入）
    args = ap.parse_args()

    # stride 传 0 是「不重叠」的约定写法：运行时替换成 seq_len，样本数最少、训练最快
    if args.stride == 0:
        args.stride = args.seq_len

    # 记录起始时间，最后用来打印总耗时
    t_start = time.time()

    # 固定 torch 的随机种子：保证权重初始化可复现
    torch.manual_seed(args.seed)

    # 固定 python random 的种子：保证训练/验证集划分可复现
    random.seed(args.seed)

    # 有 GPU 就用 GPU，否则用 CPU（本机无 CUDA，一律走 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 打印设备与模型类型，方便确认这次跑的是哪一个
    print(f'device: {device}   model: TransformerLM')

    # ── 生成模式：不训练，直接加载 --save 的模型续写 ────────────
    if args.generate:
        # 推理不需要梯度；map_location='cpu' 让权重先落到 CPU，再按需搬运
        ck = torch.load(args.save, map_location='cpu')

        # 取出保存时的词表：id 与字的对应关系必须和训练时完全一致
        c2i = ck['char2idx']
        i2c = ck['idx2char']

        # 取出训练时的超参，用来重建同样结构的模型
        ca = ck['args']

        # 按保存的超参重建模型结构（128 维 / 4 头 / 2 层 / FFN 256 / 位置表 512）
        gmodel = TransformerLM(
            vocab_size=len(c2i),
            embed_dim=ca['embed_dim'],
            nhead=ca['nhead'],
            num_layers=ca['num_layers'],
            ff_dim=ca['ff_dim'],
            max_len=ca['max_len'],
            dropout=ca['dropout'],
        ).to(device)

        # 把训练好的权重灌进结构
        gmodel.load_state_dict(ck['model_state'])

        # 生成时是否加因果掩码，要和训练保持一致（想复现「崩坏」就传 --no_causal）
        gmodel.use_causal = not args.no_causal

        # 推理模式：关闭 dropout，保证输出确定
        gmodel.eval()

        # 真正生成：从 prompt 起步，逐字采样 n_tokens 个字
        out = generate(gmodel, c2i, i2c, args.prompt, args.n_tokens, args.temperature, device)

        # 打印结果
        print('生成结果:', out)

        # return 结束 main()，避免继续往下走训练流程
        return

    # ── 数据准备 ───────────────────────────────────────────
    # 读取作业目录下所有 *.txt 当作语料（离线、自包含，不依赖上级目录）
    text = load_corpus(os.path.join(CORPUS_DIR, '*.txt'))

    # 一个字都没读到，说明语料缺失，直接报错，避免后面建出空词表
    if not text:
        raise FileNotFoundError('没找到语料，请确认目录: ' + CORPUS_DIR)

    # 打印语料规模，确认读的是哪份、有多大
    print(f'语料字符数: {len(text):,}')

    # 按字频取前 max_vocab 个字建词表（注意用全语料建表，不是截断后的）
    char2idx, idx2char = build_vocab(text, args.max_vocab)

    # 词表大小就是输出层类别数（softmax 的维度）
    vocab_size = len(char2idx)

    # 打印词表大小，应与原 LSTM 版同为 4000，便于横向比较
    print(f'词表大小: {vocab_size}')

    # limit 是提速开关：只截断训练文本，词表仍是全语料的，所以 PPL 仍可与全量实验比较
    if args.limit:
        text = text[:args.limit]
        print(f'已截断为前 {len(text):,} 字符(快速实验用)')

    # 按行切开再打乱：以「行」为单位划分训练/验证，避免同一行被切成两半造成信息泄漏
    lines = text.splitlines()
    random.shuffle(lines)

    # 按 val_ratio 切分（默认 95 百分比训练、5 百分比验证）
    split = int(len(lines) * (1 - args.val_ratio))

    # 训练集：前 split 行拼回整段文本，再按 seq_len/stride 切样本
    train_ds = CharDataset('\n'.join(lines[:split]), char2idx, args.seq_len, args.stride)

    # 验证集：剩下的行，同样切样本
    val_ds = CharDataset('\n'.join(lines[split:]), char2idx, args.seq_len, args.stride)

    # 训练 DataLoader：shuffle 打散批次顺序；drop_last 丢掉不足一个 batch 的尾巴
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 验证 DataLoader：不打乱、保留尾批（drop_last=False），否则小验证集会一个 batch 不剩 → 除零
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 打印样本数，用来估算一轮要跑多少步
    print(f'训练样本: {len(train_ds):,}   验证样本: {len(val_ds):,}')

    # ── 建模型 ─────────────────────────────────────────────
    # 按命令行参数实例化 Transformer（默认 128 维 / 4 头 / 2 层 / FFN 256）
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)

    # 是否使用因果掩码，由 --no_causal 控制（做对照实验时设为 False）
    model.use_causal = not args.no_causal

    # 统计可训练参数量，可与 LSTM 版对比（本模型约 1.36M，LSTM 约 2.46M）
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {n_params:,}')

    # 损失函数：交叉熵。每个位置算一个分类样本，所以后面要把 (B,T,V) 展平成 (B 乘 T, V)
    criterion = nn.CrossEntropyLoss()

    # 优化器：Adam，学习率取 --lr（默认 1e-3，即 0.001）
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 记录历史最佳验证 PPL，初始为无穷大，用于「只在变好时保存模型」
    best_val_ppl = float('inf')

    # 打印表头（空行 + 列名 + 分隔线）
    print('')
    print(' Epoch  TrainLoss  TrainPPL   ValLoss    ValPPL')
    print('----------------------------------------------')

    # 主训练循环：共 args.epochs 轮
    for epoch in range(1, args.epochs + 1):
        # 训练一轮：返回平均交叉熵与 PPL（PPL 就是 exp(平均交叉熵)）
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, True)

        # 验证一轮：不传优化器，内部只前向不反传
        va_loss, va_ppl = run_epoch(model, val_loader, criterion, None, device, False)

        # 标记位：本轮是否刷新最佳成绩，刷新了就在行尾打星号
        flag = ''

        # 只有验证 PPL 创新低才保存，避免把过拟合的最后一轮存下来
        if va_ppl == min(va_ppl, best_val_ppl):
            # 更新历史最佳
            best_val_ppl = va_ppl

            # 行尾标记
            flag = ' *'

            # 保存三件套：权重 + 词表 + 超参，齐了才能独立重建模型做推理
            torch.save({
                'model_state': model.state_dict(),
                'char2idx': char2idx,
                'idx2char': idx2char,
                'args': vars(args),
            }, args.save)

        # 打印本轮四个指标（:6d 右对齐 6 位，:9.4f 保留 4 位小数）
        print(f'{epoch:6d}  {tr_loss:9.4f}  {tr_ppl:8.2f}  {va_loss:8.4f}  {va_ppl:8.2f}{flag}')

    # 空行，和表格拉开
    print('')

    # 汇总：最佳验证 PPL 与模型落盘路径
    print(f'训练完成。最佳验证 PPL: {best_val_ppl:.2f}  模型保存路径: {args.save}')

    # 总耗时（秒），用来判断要不要再调 limit 或 epochs
    print(f'总耗时: {time.time() - t_start:.1f} 秒')

if __name__ == '__main__':
    main()
