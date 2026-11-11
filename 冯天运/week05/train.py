import argparse  # 提供可直接实操的命令行参数。
from functools import partial  # 给 DataLoader 的 collate 函数绑定固定配置。
import hashlib  # 记录数据和基座文件哈希，便于核对实验来源。
import json  # 保存可阅读的实验配置和指标。
import math  # 将平均负对数似然转换成困惑度。
from pathlib import Path  # 所有默认路径以脚本目录为准。
import random  # 固定 Python 随机种子。
import time  # 记录训练耗时。
import torch  # 提供自动求导与优化器。
from torch.nn import functional as F  # 使用带 ignore_index 的交叉熵。
from torch.utils.data import DataLoader  # 按批次加载样本。
from torch import nn  # nn 提供 Embedding、Transformer 和 Linear 等网络层。
import tempfile  # 使用系统临时目录，不把测试中间产物留在项目中。

# 第一部分：读取课程数据，构造对话模板与回答区域标签。
IGNORE = -100  # PyTorch 交叉熵默认使用的忽略标签值。

def read_rows(path):  # 每行读取一个 instruction/output 样本。
    rows = []  # 保留样本原始顺序。
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):  # 兼容 UTF-8 BOM 并记录行号。
        if not line.strip():  # JSONL 中的空行不代表有效样本。
            continue  # 跳过空行。
        row = json.loads(line)  # 非法 JSON 直接报错，不隐藏坏数据。
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("instruction", "output")):  # 检查必需字段。
            raise ValueError(f"{path} 第 {number} 行必须包含非空字符串 instruction 和 output。")  # 给出可定位的数据错误。
        rows.append(row)  # 保留原文，不擅自修正课程标注。
    if not rows:  # 防止空数据集产生虚假的训练结果。
        raise ValueError(f"数据集为空：{path}")  # 提前停止。
    return rows  # 返回可被 DataLoader 直接使用的 Python 列表。

def check_split(train_rows, val_rows):  # 检查训练与验证之间的指令泄漏。
    train_questions = {row["instruction"].strip() for row in train_rows}  # 归一化首尾空白后统计训练指令。
    val_questions = {row["instruction"].strip() for row in val_rows}  # 同样处理验证指令。
    overlap = train_questions & val_questions  # 同一指令即使答案不同也算泄漏。
    if overlap:  # 验证集应完全独立于训练指令。
        raise ValueError(f"训练与验证集重叠指令数：{len(overlap)}")  # 不通过静默过滤掩盖数据问题。

def encode(row, vocab, max_length):  # 将一条指令和答案变成对齐的输入与监督标签。
    prompt = [vocab["<user>"]] + [vocab.get(char, vocab["<unk>"]) for char in row["instruction"]] + [vocab["<assistant>"]]  # 角色标记各占一个位置。
    answer = [vocab.get(char, vocab["<unk>"]) for char in row["output"]] + [vocab["<eos>"]]  # 回答末尾追加一个可学习的 EOS。
    ids = prompt + answer  # 训练时一次喂入完整序列，即 teacher forcing。
    if len(ids) > max_length:  # 本课程样本较短，默认 128 足够；自定义数据超长则显式报错。
        raise ValueError(f"样本长度 {len(ids)} 超过 max_length={max_length}，请提高上限或整理数据。")  # 避免截断造成答案全被遮住。
    labels = [IGNORE] * len(prompt) + [token if token != vocab["<unk>"] else IGNORE for token in answer]  # 问题不监督，验证集未知答案字符也不作为 UNK 学习。
    return ids, labels  # 此处还未错位，交给 collate 统一处理。

def collate(rows, vocab, max_length):  # 按当前批次最长样本动态右侧补齐，减少无效计算。
    encoded = [encode(row, vocab, max_length) for row in rows]  # 每条样本独立拼接，不跨样本串联。
    width = max(len(ids) - 1 for ids, _ in encoded)  # 输入去掉最后 token，目标去掉第一个 token。
    inputs = torch.full((len(rows), width), vocab["<pad>"], dtype=torch.long)  # 默认用 PAD 填充输入。
    targets = torch.full((len(rows), width), IGNORE, dtype=torch.long)  # 所有补齐位置的 loss 都被忽略。
    for index, (ids, labels) in enumerate(encoded):  # 逐样本写入当前批次。
        size = len(ids) - 1  # 有效预测位置数量。
        inputs[index, :size] = torch.tensor(ids[:-1], dtype=torch.long)  # 输入位置 t 提供已知历史。
        targets[index, :size] = torch.tensor(labels[1:], dtype=torch.long)  # 位置 t 预测原序列的 t+1；只错位一次。
    return inputs, targets  # 不需要额外 padding attention mask：右侧 PAD 不能被左侧有效位置看到。

# 第二部分：沿用 week4 因果 Transformer，迁移权重并自回归生成。
SPECIALS = ("<pad>", "<unk>", "<user>", "<assistant>", "<eos>")  # 这五个字符串各自对应一个独立 token。

class TransformerLM(nn.Module):  # 保持 week4 的模型层名，以便加载已有预训练参数。
    def __init__(self, vocab_size, config):  # 词表大小允许增加，主干结构沿用基座配置。
        super().__init__()  # 初始化 Module，保证子模块和参数能被优化器发现。
        self.embed = nn.Embedding(vocab_size, config["embed_dim"])  # 将字符 ID 转换成向量。
        self.pos = nn.Embedding(config["max_len"], config["embed_dim"])  # 沿用 week4 可学习位置编码。
        layer = nn.TransformerEncoderLayer(  # EncoderLayer 配上因果掩码即可构造本课的自回归主干。
            d_model=config["embed_dim"],  # 每个位置的隐藏向量维度。
            nhead=config["nhead"],  # 多头注意力的头数。
            dim_feedforward=config["ff_dim"],  # 前馈网络内部的隐藏维度。
            dropout=config["dropout"],  # 训练时随机丢弃部分激活，评估时关闭。
            activation="gelu",  # 与 week4 保持相同激活函数。
            batch_first=True,  # 统一使用 batch、序列、特征的维度顺序。
        )  # 完成单个 Transformer 层的定义。
        self.encoder = nn.TransformerEncoder(layer, config["num_layers"], enable_nested_tensor=False)  # 关闭无关的嵌套张量优化。
        self.drop = nn.Dropout(config["dropout"])  # 保留 week4 的嵌入与输出 dropout。
        self.fc = nn.Linear(config["embed_dim"], vocab_size)  # 每个位置输出整个词表的 logits。

    def forward(self, ids):  # 输入形状为 B×T，输出形状为 B×T×V。
        length = ids.size(1)  # 读取当前批次的序列长度 T。
        positions = torch.arange(length, device=ids.device).unsqueeze(0)  # 得到可广播的 1×T 位置编号。
        hidden = self.drop(self.embed(ids) + self.pos(positions))  # 将字符向量与位置向量相加。
        causal = torch.triu(torch.ones(length, length, device=ids.device, dtype=torch.bool), diagonal=1)  # True 表示遮住未来。
        hidden = self.encoder(hidden, mask=causal)  # 每个位置只能关注自己以及左侧上下文。
        return self.fc(self.drop(hidden))  # 训练时直接交给交叉熵，勿提前做 softmax。

def load_base(path, rows):  # 基于训练集扩充词表，并迁移原基座参数。
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)  # 只加载本课程可信的张量和基础类型。
    config = checkpoint["args"]  # 读取真实训练配置，避免猜测层数或向量维度。
    if config.get("no_causal", False):  # 禁止拿 week4 的偷看未来对照模型做基座。
        raise ValueError("基座必须使用因果注意力训练。")  # 明确提示不兼容的基座。
    vocab = dict(checkpoint["char2idx"])  # 复制原字表，绝不改变已有字符 ID。
    old_size = len(vocab)  # 记录要原样迁移的旧词表行数。
    for token in SPECIALS:  # 在尾部追加角色、结束和补齐标记。
        if token in vocab:  # 原课程字表是字符表，不应已经占用保留标记。
            raise ValueError(f"基座存在保留标记：{token}")  # 避免不同协议混用。
        vocab[token] = len(vocab)  # 新 token 的 ID 紧接原字表。
    characters = sorted(set("".join(row["instruction"] + row["output"] for row in rows)))  # 仅从训练集获取缺失字符。
    for character in characters:  # 按固定顺序扩充，保证相同数据产生相同 ID。
        if character not in vocab:  # 不覆盖任何旧字符的编号。
            vocab[character] = len(vocab)  # 训练集生僻字也保留，不静默丢字。
    model = TransformerLM(len(vocab), config)  # 只扩充输入输出词表，Transformer 主干保持不变。
    state = model.state_dict()  # 取得新模型的参数字典，保留新增行的随机初始化。
    for name, value in checkpoint["model_state"].items():  # 遍历预训练的每一个参数。
        if name in ("embed.weight", "fc.weight", "fc.bias"):  # 这三个参数的第一维随词表扩展。
            state[name][:old_size].copy_(value)  # 原词表的每一行都原样迁移。
        else:  # 其余层的形状应与 week4 完全一致。
            state[name].copy_(value)  # 复制全部主干及位置编码参数。
    model.load_state_dict(state, strict=True)  # 严格检查键名与形状，防止漏加载。
    return model, vocab, config, old_size  # 返回训练、推理与统计所需的信息。

def load_trained(path, device):  # 加载本作业保存的完整 SFT 模型。
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)  # 先落在 CPU，便于跨设备加载。
    model = TransformerLM(len(checkpoint["vocab"]), checkpoint["config"])  # 从保存配置重建结构。
    model.load_state_dict(checkpoint["model_state"], strict=True)  # 加载包含新增 token 行的完整权重。
    return model.to(device).eval(), checkpoint  # 移到目标设备并关闭 dropout。

@torch.no_grad()  # 自回归生成不需要梯度，减少内存开销。
def generate(model, vocab, instruction, device, max_new_tokens=64):  # 用贪心解码便于训练前后公平对照。
    prompt = [vocab["<user>"]] + [vocab.get(char, vocab["<unk>"]) for char in instruction] + [vocab["<assistant>"]]  # 推理模板与训练相同。
    if len(prompt) >= model.pos.num_embeddings:  # 留出至少一个预测位置。
        raise ValueError("指令超出模型上下文，请缩短输入。")  # 不静默裁掉用户指令。
    ids = list(prompt)  # 保存不断增长的完整历史。
    tokens = []  # 仅收集模型新生成的 token，避免把问题回显当作答案。
    inverse = {index: token for token, index in vocab.items()}  # 将模型输出 ID 还原成字符。
    reason = "max_new_tokens"  # 未遇到 EOS 时如实记录截断原因。
    model.eval()  # 关闭 dropout，确保固定输入得到稳定结果。
    for _ in range(max_new_tokens):  # 每一步只预测一个新 token。
        if len(ids) >= model.pos.num_embeddings:  # 禁止位置编码索引越界。
            reason = "context_limit"  # 如实区分上下文耗尽和正常结束。
            break  # 停止生成。
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1].clone()  # 只使用最后位置预测下一 token。
        for marker in ("<pad>", "<unk>", "<user>", "<assistant>"):  # 本作业只生成单轮回答，不允许中途切换角色。
            logits[vocab[marker]] = -torch.inf  # 这是固定解码约束，训练前后使用同一规则。
        next_id = int(logits.argmax().item())  # 贪心选择最高概率 token，不引入采样随机性。
        if next_id == vocab["<eos>"]:  # EOS 是需要通过 SFT 学会的停止标记。
            reason = "eos"  # 记录模型主动结束回答。
            break  # EOS 不显示在答案正文里。
        ids.append(next_id)  # 将预测结果回填输入，构成自回归生成。
        tokens.append(inverse[next_id])  # 保存实际生成字符。
    return {"text": "".join(tokens), "stop_reason": reason}  # 同时返回正文与停止原因，避免误判截断。

# 第三部分：关键逻辑自检，不改变正式实验文件。
def verify():  # 用真实课程基座做关键行为检查，无需额外测试框架。
    torch.set_num_threads(2)  # 小张量测试使用少量线程即可。
    torch.manual_seed(42)  # 保持扩充词表的随机初始化可复现。
    here = Path(__file__).resolve().parent  # 所有输入以脚本所在目录定位。
    rows, val = read_rows(here / "data/sft_train.jsonl"), read_rows(here / "data/sft_val.jsonl")  # 读取真实课程数据。
    check_split(rows, val)  # 训练与验证集不能出现相同指令。
    model, vocab, config, old_size = load_base(here / "base/transformer_lm.pt", rows)  # 迁移 week4 参数。
    source = torch.load(here / "base/transformer_lm.pt", map_location="cpu", weights_only=True)  # 仅用于核对迁移精度。
    assert all(vocab[char] == index for char, index in source["char2idx"].items())  # 检查所有旧字符 ID 未改变。
    for name, value in source["model_state"].items():  # 检查每个预训练参数都没有丢失。
        actual = model.state_dict()[name]  # 取迁移后的同名参数。
        actual = actual[:old_size] if name in ("embed.weight", "fc.weight", "fc.bias") else actual  # 扩展矩阵只比较旧行。
        assert torch.equal(actual, value), name  # 使用逐元素完全相等，避免近似误差掩盖漏加载。
    example = {"instruction": "你好", "output": "好"}  # 构造可直接手算监督位置的短样本。
    ids, _ = encode(example, vocab, 128)  # 得到 user、你、好、assistant、好、eos。
    x, y = collate([example, rows[0]], vocab, 128)  # 第二条更长，用于同时检查右侧 padding。
    boundary = 1 + len(example["instruction"])  # assistant 在原输入中的位置，也是第一个答案的预测位置。
    assert y[0, :boundary].eq(IGNORE).all()  # prompt 的预测不参与回答 loss。
    assert y[0, boundary].item() == vocab["好"]  # assistant 位置应预测第一个答案字符，避免 off-by-one。
    assert y[0, boundary + 1].item() == vocab["<eos>"]  # 最后一个答案字符应预测 EOS。
    assert y[0, len(ids) - 1:].eq(IGNORE).all()  # 右侧补齐位置全部忽略。
    assert x[0, len(ids) - 1:].eq(vocab["<pad>"]).all()  # 输入按约定右侧补 PAD。
    model.eval()  # 关闭 dropout，确保前后数值可比较。
    with torch.no_grad():  # 检查数值时无需保留计算图。
        single = model(torch.tensor([ids[:-1]]))  # 单独计算不带 padding 的短样本。
        batched = model(x)  # 计算同一样本在补齐批次中的结果。
        assert torch.allclose(single, batched[:1, :len(ids) - 1], atol=1e-5)  # PAD 不得改变有效位置预测。
        changed = x.clone()  # 保留原输入，构造只改变未来字符的样本。
        changed[:, -1] = vocab["你"]  # 最后一个位置属于前面位置的未来。
        assert torch.allclose(model(x)[:, :-1], model(changed)[:, :-1], atol=1e-5)  # 未来改变不得影响过去 logits。
    logits = model(x)  # 用真实模型输出核对 ignore_index 的含义。
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE)  # 使用训练时相同的标签。
    valid = y.ne(IGNORE)  # 筛选出真正参与监督的回答位置。
    manual = F.cross_entropy(logits[valid], y[valid])  # 手工去掉 prompt 与 padding 后计算 loss。
    assert torch.allclose(loss, manual)  # 两种算法必须相等。
    before = model.fc.weight.detach().clone()  # 保存一步更新前的权重。
    loss.backward()  # 真实执行反向传播。
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # 使用和训练一致的优化器类型。
    optimizer.step()  # 执行一次参数更新。
    assert not torch.equal(before, model.fc.weight)  # 确认 loss 可以驱动真实权重更新。
    with tempfile.TemporaryDirectory(prefix="sft-coursework-verify-") as temporary:  # 自动清理仅由本任务创建的临时目录。
        path = Path(temporary) / "roundtrip.pt"  # 临时保存权重，不污染正式实验。
        torch.save({"model_state": model.state_dict(), "vocab": vocab, "config": config}, path)  # 保存完整推理状态。
        restored, _ = load_trained(path, torch.device("cpu"))  # 使用正式推理加载流程。
        with torch.no_grad():  # 加载一致性比较无需梯度。
            assert torch.equal(model(x), restored(x))  # 保存加载前后 logits 应完全一致。
    try:  # 确认超长样本不会静默损失答案监督。
        encode(example, vocab, 4)  # 人为设置过小的长度上限。
    except ValueError:  # 正确行为是明确报错。
        pass  # 预期异常表示通过。
    else:  # 没有报错意味着发生了静默裁剪或漏校验。
        raise AssertionError("超长样本必须显式报错")  # 阻止交付错误数据逻辑。
    print("PASS：数据无交叉、旧 ID/权重完整迁移、回答标签错位、EOS/padding、因果性、loss mask、参数更新、保存加载、超长校验。")  # 汇总已验证范围。

# 第四部分：训练、验证、保存模型与统一命令行入口。
HERE = Path(__file__).resolve().parent  # 切换终端目录也不会改变默认数据路径。

def run_epoch(model, loader, device, optimizer=None):  # 同一套计算用于训练与验证。
    training = optimizer is not None  # 传优化器才允许更新参数。
    model.train(training)  # 训练打开 dropout，验证关闭 dropout。
    total_loss, total_tokens = 0.0, 0  # 用回答 token 数加权，而非直接平均各 batch 的 loss。
    with torch.set_grad_enabled(training):  # 验证不建立计算图。
        for inputs, targets in loader:  # DataLoader 保留不足一批的尾部样本。
            inputs, targets = inputs.to(device), targets.to(device)  # 数据与模型必须位于相同设备。
            logits = model(inputs)  # 因果前向，形状 B×T×V。
            token_count = int(targets.ne(IGNORE).sum().item())  # 只统计回答字符和 EOS。
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=IGNORE, reduction="sum")  # 累加有效位置损失。
            loss = loss_sum / token_count  # 每次梯度更新使用本批有效回答 token 的平均损失。
            if not torch.isfinite(loss):  # 发现数值异常时不保存无效结果。
                raise FloatingPointError("出现非有限 loss，请检查数据或降低学习率。")  # 明确暴露失败。
            if training:  # 仅在训练阶段执行反向传播。
                optimizer.zero_grad(set_to_none=True)  # 清除上一步梯度，防止意外累加。
                loss.backward()  # 求出全部模型参数的梯度。
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，减轻不稳定更新。
                optimizer.step()  # AdamW 更新所有参数，这里没有使用 LoRA。
            total_loss += float(loss_sum.detach().item())  # 用脱离计算图的数值累计指标。
            total_tokens += token_count  # 为最后一小批提供正确权重。
    mean = total_loss / total_tokens  # 整个数据集的回答 token 平均交叉熵。
    return {"loss": mean, "ppl": math.exp(mean) if mean < 700 else float("inf"), "tokens": total_tokens}  # PPL 只描述本任务已知字符与 EOS。

def main():  # 组织数据校验、基座评估、SFT、最佳模型选择和生成对照。
    parser = argparse.ArgumentParser(description="week4 因果 Transformer → week5 回答区域 SFT")  # --help 展示脚本用途。
    parser.add_argument("--train", type=Path, default=HERE / "data/sft_train.jsonl")  # 默认使用课程训练数据副本。
    parser.add_argument("--val", type=Path, default=HERE / "data/sft_val.jsonl")  # 验证集不参与参数更新。
    parser.add_argument("--base", type=Path, default=HERE / "base/transformer_lm.pt")  # 默认加载 week4 已训练模型副本。
    parser.add_argument("--output", type=Path, default=HERE / "outputs/course_run")  # 每次实验要用新的输出目录。
    parser.add_argument("--epochs", type=int, default=10)  # 先做三轮小规模微调，观察泛化再决定是否增加。
    parser.add_argument("--batch-size", type=int, default=64)  # CPU 默认每批 32 条。
    parser.add_argument("--lr", type=float, default=0.0001)  # 微调学习率低于 week4 的 0.001。
    parser.add_argument("--max-length", type=int, default=128)  # 包括两个角色标记、完整问题、答案和 EOS。
    parser.add_argument("--seed", type=int, default=42)  # 固定初始化与打乱顺序。
    parser.add_argument("--threads", type=int, default=4)  # 避免 CPU 线程过多导致争用。
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")  # 当前机器为 CPU，也允许后续迁移 GPU。
    parser.add_argument("--limit-train", type=int, default=0)  # 正数只取前 N 条，0 使用完整训练集。
    parser.add_argument("--limit-val", type=int, default=0)  # 快速试跑可缩小验证集，报告会记录实际数量。
    modes = parser.add_mutually_exclusive_group()  # 自检与推理互斥；不传两者时执行训练。
    modes.add_argument("--verify", action="store_true")  # 将原 verify.py 合并为一个自检开关。
    modes.add_argument("--instruction")  # 将原 infer.py 合并为一条指令参数。
    parser.add_argument("--checkpoint", type=Path, default=HERE / "outputs/course_run/best_sft.pt")  # 用此参数明确选择已有实验权重。
    parser.add_argument("--max-new-tokens", type=int, default=64)  # 推理最多生成的字符数。
    args = parser.parse_args()  # 读取用户参数。
    if args.verify:  # 自检不启动训练，不修改已有权重。
        verify()  # 复用原有全部关键检查。
        return  # 检查结束后退出。
    if args.instruction is not None:  # 有指令时进入推理模式。
        if args.max_new_tokens <= 0 or args.threads <= 0:  # 避免无效生成长度或线程数。
            parser.error("生成长度和线程数必须为正整数。")  # 用中文提示参数错误。
        torch.set_num_threads(args.threads)  # CPU 推理沿用可配置线程限制。
        model, checkpoint = load_trained(args.checkpoint, torch.device("cpu"))  # 兼容精简前所有已训练权重。
        result = generate(model, checkpoint["vocab"], args.instruction, torch.device("cpu"), args.max_new_tokens)  # 模板与解码逻辑保持不变。
        print(result["text"])  # 输出模型生成的回答。
        print(f"停止原因：{result['stop_reason']}")  # 区分主动 EOS 与长度截断。
        return  # 推理完成后不进入训练。

    if min(args.epochs, args.batch_size, args.threads) <= 0 or args.lr <= 0 or args.max_length < 4 or min(args.limit_train, args.limit_val) < 0:  # 校验会导致无意义实验的参数。
        parser.error("轮数、批量、线程、学习率须为正；max_length 至少为 4；样本上限不可为负。")  # 使用标准命令行错误退出。
    if args.output.exists():  # 不自动覆盖已经完成的实验。
        parser.error(f"输出目录已存在，请用 --output 指定新目录：{args.output}")  # 保留既有模型与记录。
    torch.set_num_threads(args.threads)  # 明确限制 CPU 并行度。
    torch.manual_seed(args.seed)  # 固定模型新增行初始化与 DataLoader 顺序。
    random.seed(args.seed)  # 固定 Python 随机数。
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)  # 自动模式检测 GPU。
    if device.type == "cuda" and not torch.cuda.is_available():  # 显式请求 GPU 时不静默回退。
        parser.error("当前 PyTorch 不支持可用 CUDA，请使用 --device cpu。")  # 提供直接可执行的替代参数。
    train_rows, val_rows = read_rows(args.train), read_rows(args.val)  # 读取原始标注，不合并训练验证。
    check_split(train_rows, val_rows)  # 在缩小数据之前检查完整数据是否存在指令重叠。
    train_rows = train_rows[:args.limit_train] if args.limit_train else train_rows  # 试跑只缩小训练样本。
    val_rows = val_rows[:args.limit_val] if args.limit_val else val_rows  # 同样显式缩小验证样本。
    model, vocab, config, old_size = load_base(args.base, train_rows)  # 只有训练集用于扩充词表。
    if args.max_length > config["max_len"]:  # 本基座位置表长度是硬上限。
        parser.error(f"max_length 不得超过位置表长度 {config['max_len']}")  # 不修改基座位置结构绕过限制。
    batcher = partial(collate, vocab=vocab, max_length=args.max_length)  # 统一训练与验证的模板及长度规则。
    for row in train_rows + val_rows:  # 在创建输出目录之前验证每条数据能编码。
        batcher([row])  # 超长数据立即暴露，不训练到中途才报错。
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=batcher, num_workers=0)  # 单进程兼容 Windows。
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=batcher, num_workers=0)  # 验证顺序固定且保留尾批。
    model.to(device)  # 将基座及新增 token 行移到计算设备。
    args.output.mkdir(parents=True, exist_ok=False)  # 独立目录保证不会覆盖旧实验。
    started = time.perf_counter()  # 开始计时，包括基线评估与生成对照。
    baseline = run_epoch(model, val_loader, device)  # 在同一模板、词表和验证集上测量训练前指标。
    examples = [{"instruction": row["instruction"], "reference": row["output"], "before": generate(model, vocab, row["instruction"], device)} for row in val_rows[:3]]  # 固定未训练的验证指令做前后对照。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)  # 全参数 SFT，沿用自回归目标。
    history, best_loss, best_epoch = [], float("inf"), 0  # 最佳轮次只根据验证 loss 选择。
    print(f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, train={len(train_rows)}, val={len(val_rows)}, vocab={old_size}->{len(vocab)}", flush=True)  # 打印运行规模。
    print(f"before: {baseline}", flush=True)  # 公开训练前基线。
    for epoch in range(1, args.epochs + 1):  # 遍历指定的微调轮数。
        training = run_epoch(model, train_loader, device, optimizer)  # 训练集反向传播。
        validation = run_epoch(model, val_loader, device)  # 验证集仅前向计算。
        history.append({"epoch": epoch, "train": training, "val": validation})  # 保存逐轮指标，方便发现过拟合。
        if validation["loss"] < best_loss:  # 保存验证表现最好的微调模型。
            best_loss, best_epoch = validation["loss"], epoch  # 更新当前最佳成绩和对应轮次。
            torch.save({"model_state": model.state_dict(), "vocab": vocab, "config": config, "epoch": epoch, "val": validation}, args.output / "best_sft.pt")  # 保存完整推理所需状态。
        print(f"epoch={epoch} train_loss={training['loss']:.4f} val_loss={validation['loss']:.4f} val_ppl={validation['ppl']:.2f}", flush=True)  # 持续显示训练进度。
    best_model, checkpoint = load_trained(args.output / "best_sft.pt", device)  # 重新从磁盘加载，确认保存产物可独立推理。
    for example in examples:  # 对相同验证指令生成训练后答案。
        example["after"] = generate(best_model, vocab, example["instruction"], device)  # 使用最佳轮次，而不是默认最后一轮。
    unknowns = {key: sum(char not in vocab for row in val_rows for char in row[key]) for key in ("instruction", "output")}  # 如实报告验证集未知字符数量。
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (("train", args.train), ("val", args.val), ("base", args.base))}  # 记录输入文件身份。
    report = {"torch": str(torch.__version__), "device": str(device), "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "hashes": hashes, "train_samples": len(train_rows), "val_samples": len(val_rows), "vocab_before": old_size, "vocab_after": len(vocab), "val_unknown_characters": unknowns, "baseline": baseline, "history": history, "best_epoch": best_epoch, "best_val": checkpoint["val"], "examples": examples, "seconds": time.perf_counter() - started}  # 汇总复现实验所需证据。
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")  # 使用 JSON 保留精确数值和中文生成样例。
    print(f"完成，最佳轮次={best_epoch}，记录：{args.output / 'metrics.json'}", flush=True)  # 输出交付位置。

if __name__ == "__main__":  # 直接执行本文件才进入命令行流程。
    main()  # 默认训练，也支持 --verify 自检或 --instruction 推理。
