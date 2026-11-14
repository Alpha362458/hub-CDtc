import argparse
import math
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


# 张量形状里的 B 是批大小，T 是输入长度，D 是每个向量的宽度，V 是字表大小。
class Block(nn.Module):
    """模型里的一层。先在各位置间取信息，再单独处理每个位置。"""

    def __init__(self, d_model, heads):
        super().__init__()
        # LayerNorm 在每一格自己的 D 个数上调整尺度。
        self.ln1 = nn.LayerNorm(d_model)
        # batch_first=True 表示传入 [B, T, D]，第一维放不同文本。
        self.attn = nn.MultiheadAttention(
            d_model, heads, dropout=0.1, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        # FFN 独立处理每一格。先把 96 个数扩到 384 个，经过 GELU，
        # 再变回 96 个，才能与这一步的原输入相加。
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, h, mask):
        # h 包含整批文本每个位置的表示。先归一化，再做自注意力。
        z = self.ln1(h)
        # Q、K、V 的输入都用 z，注意力层会用不同的系数算出三者。
        # 这里只要更新后的向量，第二个返回值用下划线接住。
        attended, _ = self.attn(z, z, z, attn_mask=mask, need_weights=False)
        # 两次相加都是残差连接。第二次用的是已经过注意力更新的 h。
        h = h + attended
        return h + self.ffn(self.ln2(h))


class CharTransformer(nn.Module):
    """接收一批字符 ID，给每个输入位置预测下一字。"""

    def __init__(self, vocab_size, context, d_model=96, heads=4, layers=2):
        super().__init__()
        self.context = context
        # 字符表按字的 ID 查，位置表按该字在窗口里的位置查。
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        # 创建 layers 个参数各自独立的 Block。
        # ModuleList 让 PyTorch 知道这些 Block 的参数也需要训练和保存。
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        # 对每个位置，把 D 个特征变成 V 个候选分数。
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x 是 [B, T]。size(1) 取第二维，也就是每段的输入长度 T。
        t = x.size(1)
        positions = torch.arange(t, device=x.device)
        # 字符向量是 [B, T, D]，位置向量是 [T, D]。
        # 相加时，每段文字都会加上同一套位置向量。
        h = self.token_emb(x) + self.pos_emb(positions)  # [B, T, D]
        # 右边的列号大于当前行号时为 True，表示当前格读不到那个位置。
        mask = positions[None, :] > positions[:, None]  # [T, T]
        # 同一张因果掩码传给每层，避免中途混入未来位置的信息。
        for block in self.blocks:
            h = block(h, mask)
        # 返回原始分数，交叉熵会自己处理分数到概率的转换。
        return self.lm_head(self.norm(h))  # [B, T, V]


def encode(text, char_to_id):
    """逐字查编号。字表里没找到时，使用编号 0（<UNK>）。"""
    return [char_to_id.get(char, 0) for char in text]


def training_batch(ids, context, batch_size):
    """随机选一批起点，返回错开一格的 x 和 y，形状都是 [B, T]。"""
    # randint 的上界取不到，最大起点是 len(ids) - context - 1，
    # 所以从该处起往后取 context + 1 个字刚好不会越界。
    starts = torch.randint(len(ids) - context, (batch_size,))
    # 每个起点都加上 0 到 context，得到 [B, T+1] 个原文下标。
    # 用这些下标到 ids 里取值，才得到实际的字符 ID。
    windows = ids[starts[:, None] + torch.arange(context + 1)]
    # 左边是每行去掉最后一个字，右边是每行去掉第一个字。
    return windows[:, :-1], windows[:, 1:]


def token_loss(logits, y, reduction="mean"):
    """把每道猜字题摆在单独一行，再用对应的答案 ID 算交叉熵。"""
    # [B, T, V] 变成 [B*T, V]，[B, T] 变成 [B*T]。
    # -1 表示这一维让 PyTorch 按元素总数推算。
    # 训练用 mean 求平均，验证用 sum 汇总各题的损失。
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction=reduction
    )


@torch.no_grad()
def validate(model, val_ids, batch_size, device):
    """用留出的文本做题，返回每道题的平均损失。"""
    # eval() 关闭 Dropout。上面的装饰器让这整个函数不记录反向所需的信息。
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # unfold 从第 0 维切窗口。假设 context=4，先取原文下标 0 到 4，
    # 再取 4 到 8。两个窗口共用下标 4，答案分别是下标 1 到 4、5 到 8。
    windows = val_ids.unfold(0, model.context + 1, model.context)
    for batch in windows.split(batch_size):
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        # .item() 取出普通数字，numel() 数一数 y 里共有多少个答案。
        total_loss += token_loss(model(x), y, reduction="sum").item()
        total_tokens += y.numel()
    return total_loss / total_tokens


@torch.no_grad()
def check_causality(model, x):
    """改动右半段，检查左半段各位置的预测分数是否保持一致。"""
    # 关闭 Dropout，免得两次前向的随机丢弃影响比较。
    model.eval()
    cut = x.size(1) // 2
    changed = x.clone()
    # 只改复制出来的张量。ID 加一后取余，保证新 ID 仍在字表范围内。
    changed[:, cut:] = (changed[:, cut:] + 1) % model.lm_head.out_features
    before = model(x)[:, :cut]
    after = model(changed)[:, :cut]
    # 允许浮点数计算中极小的误差。差异太大时会报 AssertionError。
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    print("causal check passed")


@torch.no_grad()
def generate(model, prompt, vocab, device, steps=40, temperature=0.8):
    """从提示词出发，按模型给出的概率一个字一个字地续写。"""
    model.eval()
    char_to_id = {char: i for i, char in enumerate(vocab)}
    ids = encode(prompt, char_to_id)
    continuation = []
    for _ in range(steps):
        # 只送入最近 context 个 ID。外面加一对方括号组成一行，形状是 [1, T]。
        x = torch.tensor([ids[-model.context:]], device=device)
        # 取第 0 段的最后一个位置，保留这格对字表所有项的分数。
        next_logits = model(x)[0, -1]
        # 温度小于 1 时，高分候选的概率会更集中。
        probs = (next_logits / temperature).softmax(dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        # ID 接回模型输入。文字放进输出列表，最后一起拼出来。
        ids.append(next_id)
        continuation.append(vocab[next_id])
    return prompt + "".join(continuation)


def main():
    # 解析终端给的选项，例如 --steps 1。没传的选项使用下面的默认值。
    parser = argparse.ArgumentParser()
    # 默认从脚本所在目录找语料，启动脚本时可以在其他目录。
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(__file__).resolve().with_name("corpus_general.txt"),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--save", type=Path, default=Path("char_transformer.pt"))
    parser.add_argument("--load", type=Path)
    parser.add_argument("--prompt", default="今天天气")
    args = parser.parse_args()
    if not args.prompt:
        parser.error("--prompt 需要至少一个字符")
    if args.load is None and (args.steps < 1 or args.context < 2):
        parser.error("--steps 至少为 1，--context 至少为 2")

    # 固定随机种子，让同一环境下的试跑更容易复现。
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)

    if args.load is not None:
        # 有 --load 就走续写流程。** 把设置字典拆成具名参数传给模型。
        checkpoint = torch.load(args.load, map_location="cpu", weights_only=True)
        model = CharTransformer(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        print(generate(model, args.prompt, checkpoint["vocab"], device))
        # 到这里退出 main()，后面的训练流程从无 --load 的命令进入。
        return

    # 练习只取前 12 万个字符。找 95% 位置之前最后一个换行处作为切分边界。
    text = args.corpus.read_text(encoding="utf-8")[:120_000]
    split = text.rfind("\n", 0, int(len(text) * 0.95)) + 1
    train_text, val_text = text[:split], text[split:]
    if min(len(train_text), len(val_text)) <= args.context:
        parser.error("训练和验证文本都需要至少 context + 1 个字符，请检查语料或缩短窗口")

    # Counter 只统计训练段的字频。选最常见的 1499 个字，另留 0 号给未知字。
    vocab = ["<UNK>"] + [c for c, _ in Counter(train_text).most_common(1499)]
    char_to_id = {char: i for i, char in enumerate(vocab)}
    train_ids = torch.tensor(encode(train_text, char_to_id), dtype=torch.long)
    val_ids = torch.tensor(encode(val_text, char_to_id), dtype=torch.long)

    # 用配置建模型，把模型移到运行设备。让 AdamW 管理模型的可训练参数。
    config = dict(vocab_size=len(vocab), context=args.context, d_model=96, heads=4, layers=2)
    model = CharTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch_size = 16

    print(f"device={device} train_chars={len(train_ids)} val_chars={len(val_ids)} V={len(vocab)}")
    # unsqueeze(0) 在开头加一维，把一段文本变成一批里只有一段的形状。
    check_causality(model, train_ids[:args.context].unsqueeze(0).to(device))
    # 更新参数前先测一次，作为后续对照用的 step=0。
    initial_val = validate(model, val_ids, batch_size, device)
    print(f"step=0 val_loss={initial_val:.3f} val_ppl={math.exp(initial_val):.2f}")

    best_val = float("inf")
    model.train()
    for step in range(1, args.steps + 1):
        # 取一批数据，把输入和答案都移到模型所在的设备。
        x, y = training_batch(train_ids, args.context, batch_size)
        x, y = x.to(device), y.to(device)
        # 清旧梯度，再跑前向计算。只在第一步打印大小，方便核对。
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if step == 1:
            print(f"x={tuple(x.shape)} y={tuple(y.shape)} logits={tuple(logits.shape)}")
        # 先对答案算损失，接着算梯度，最后由 optimizer.step() 改参数。
        loss = token_loss(logits, y)
        loss.backward()
        optimizer.step()

        # 第 1 步、每 100 步和最后一步做验证。
        if step == 1 or step % 100 == 0 or step == args.steps:
            val_loss = validate(model, val_ids, batch_size, device)
            print(f"step={step} train_batch_loss={loss.item():.3f} "
                  f"val_loss={val_loss:.3f} val_ppl={math.exp(val_loss):.2f}")
            if val_loss < best_val:
                best_val = val_loss
                # 保存验证损失更低的模型参数，以及重建它需要的字表和配置。
                torch.save({"model": model.state_dict(), "vocab": vocab, "config": config}, args.save)
            # 验证会调用 eval()，这里切回训练模式，让下一批继续使用 Dropout。
            model.train()

    print(f"saved={args.save} best_val_ppl={math.exp(best_val):.2f}")


# 在终端直接运行这个 .py 文件时，从 main() 开始。
if __name__ == "__main__":
    main()