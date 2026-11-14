"""从国晗 · Week 5 · 中文字符语言模型 SFT

依赖 Python 和 PyTorch。微调从第四周的 best_model.pt 加载网络、词表与参数，
保留因果注意力，继续做下一字预测。训练只对答案、结束换行计算交叉熵。

检查一条样本的标签对齐（可不传权重和数据）：
    python 作业01.py --check-sample
训练与评测：
    python 作业01.py --base-model <第四周权重> --data <样本目录> --output <新输出目录>
比较一条投诉：
    python 作业01.py --base-model <第四周权重> --output <训练输出目录> --predict "酒店临时取消预订。"

样本目录包含 train.jsonl、dev.jsonl、test.jsonl，每行字段为 complaint 和 answer。
例如 {"complaint": "酒店临时取消预订。", "answer": "住宿"}。
类别为住宿、交通、门票。实验使用助手编写的模拟数据，由助手在本机运行。
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
PAD_ID = 0


CATEGORIES = ("住宿", "交通", "门票")
IGNORE = -100
SEED = 42
MAX_NEW_CHARACTERS = 12


# 底模结构与第四周保持兼容，以便直接加载其权重。
@dataclass
class ModelConfig:
    vocabulary_size: int
    context_length: int = 128
    model_dimension: int = 192
    number_of_heads: int = 6
    number_of_layers: int = 3
    dropout: float = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension = config.model_dimension
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, config.number_of_heads, dropout=config.dropout, batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(dimension)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, 4 * dimension), nn.GELU(),
            nn.Linear(4 * dimension, dimension), nn.Dropout(config.dropout),
        )
        self.output_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, causal_mask):
        # Pre-LN：先归一化，再计算子层，最后残差相加。
        normalized = self.attention_norm(hidden)
        attention_output, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False,
        )
        hidden = hidden + self.output_dropout(attention_output)
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class TransformerLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_dimension % config.number_of_heads:
            raise ValueError("model_dimension 必须能被 number_of_heads 整除。")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.model_dimension)
        self.position_embedding = nn.Embedding(config.context_length, config.model_dimension)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.number_of_layers))
        self.final_norm = nn.LayerNorm(config.model_dimension)
        self.language_head = nn.Linear(config.model_dimension, config.vocabulary_size, bias=False)
        # 输入字向量与输出分类器共享权重，减少小语料训练的参数量。
        self.language_head.weight = self.token_embedding.weight
        mask = torch.triu(torch.ones(config.context_length, config.context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        length = input_ids.size(1)
        if not 0 < length <= self.config.context_length:
            raise ValueError(f"输入长度需在 1 到 {self.config.context_length} 之间。")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.embedding_dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            hidden = block(hidden, self.causal_mask[:length, :length])
        return self.language_head(self.final_norm(hidden))


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 Python 的 PyTorch 无法使用 CUDA；请使用课程 GPU 环境或 --device cpu。")
    return torch.device(name)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TransformerLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def generate(model, vocabulary, prompt, device, maximum_new_characters=60, temperature=0.8, top_k=40, seed=42):
    if not prompt.strip():
        raise ValueError("请输入至少一个非空白字符作为开头。")
    if temperature <= 0 or maximum_new_characters < 1 or top_k < 1:
        raise ValueError("temperature、maximum_new_characters、top_k 必须大于零。")
    torch.manual_seed(seed)
    character_to_id = {char: index for index, char in enumerate(vocabulary)}
    unknown = sorted(set(prompt) - set(character_to_id))
    if unknown:
        raise ValueError(f"开头包含训练词表以外的字符：{unknown}，请换一个开头。")
    ids = torch.tensor([[character_to_id[char] for char in prompt]], device=device)
    model.eval()
    result = prompt
    for _ in range(maximum_new_characters):
        logits = model(ids[:, -model.config.context_length:])[:, -1, :].float() / temperature
        logits[:, :2] = -torch.inf  # 不将 PAD/UNK 当成实际文字输出。
        values, indices = torch.topk(logits, min(top_k, len(vocabulary) - 2), dim=-1)
        next_id = indices.gather(-1, torch.multinomial(F.softmax(values, dim=-1), 1))
        char = vocabulary[next_id.item()]
        if char == "\n":
            break
        result += char
        ids = torch.cat((ids, next_id), dim=1)
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_prompt(complaint):
    complaint = complaint.strip()
    if not complaint or len(complaint.splitlines()) != 1:
        raise ValueError("请输入一条非空、单行的投诉。")
    return f"\n问：{complaint}答："


def make_example(complaint, answer, char_to_id, context_length):
    prompt = make_prompt(complaint)
    sequence = prompt + answer + "\n"
    if len(sequence) - 1 > context_length:
        raise ValueError(f"样本超过 {context_length} 字的上下文限制。")
    ids = [char_to_id[character] for character in sequence]

    # logits[i] 预测 sequence[i + 1]。最后一个提示字要预测第一个答案字。
    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
    labels = torch.tensor(ids[1:], dtype=torch.long)
    labels[:len(prompt) - 1] = IGNORE
    return input_ids, labels


def check_example(vocabulary=None, context_length=128):
    complaint, answer = "酒店临时取消预订。", "住宿"
    if vocabulary is None:
        vocabulary = list(dict.fromkeys(make_prompt(complaint) + answer + "\n"))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    input_ids, labels = make_example(complaint, answer, char_to_id, context_length)
    first_answer = len(make_prompt(complaint)) - 1
    assert torch.all(labels[:first_answer] == IGNORE)
    assert vocabulary[input_ids[first_answer]] == "："
    assert [vocabulary[label] for label in labels[first_answer:]] == list(answer + "\n")
    print("位置  当前输入字  下一个字的标签")
    for index, (character_id, label) in enumerate(zip(input_ids, labels)):
        target = "-100" if label == IGNORE else repr(vocabulary[label])
        print(f"{index:>4}  {vocabulary[character_id]!r:>10}  {target}")
    print("通过：提示词不计损失，答案首字、末字和结束符均计入损失。")


def load_splits(data_directory, vocabulary, context_length):
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    seen = set()
    splits = {}
    for name in ("train", "dev", "test"):
        path = data_directory / f"{name}.jsonl"
        rows = []
        examples = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            complaint, answer = row["complaint"].strip(), row["answer"]
            if answer not in CATEGORIES or complaint in seen:
                raise ValueError(f"{path.name}:{line_number} 类别非法或投诉重复。")
            try:
                examples.append(make_example(complaint, answer, char_to_id, context_length))
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path.name}:{line_number} {error}") from error
            rows.append({"complaint": complaint, "answer": answer})
            seen.add(complaint)
        if {row["answer"] for row in rows} != set(CATEGORIES):
            raise ValueError(f"{path.name} 必须覆盖三个类别。")
        splits[name] = rows, examples
    return splits


def collate_examples(examples):
    input_ids, labels = zip(*examples)
    return (
        pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE),
    )


def epoch_loss(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = 0.0
    token_count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for input_ids, labels in batches:
            target_count = labels.ne(IGNORE).sum().item()
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
            if not math.isfinite(loss.item()):
                raise RuntimeError("损失出现非有限值，训练已停止。")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            loss_sum += loss.item() * target_count
            token_count += target_count
    return loss_sum / token_count


def predict(model, vocabulary, complaint, device):
    prompt = make_prompt(complaint)
    if len(prompt) > model.config.context_length:
        raise ValueError("投诉超过模型的上下文长度。")
    sequence = generate(model, vocabulary, prompt, device, maximum_new_characters=MAX_NEW_CHARACTERS, top_k=1, seed=SEED)
    answer = sequence[len(prompt):]
    stopped = len(answer) < MAX_NEW_CHARACTERS
    return {"answer": answer, "stopped": stopped, "valid_format": stopped and answer in CATEGORIES}


def train(model, checkpoint, args, device):
    vocabulary = checkpoint["vocabulary"]
    splits = load_splits(args.data, vocabulary, model.config.context_length)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    counts = {name: dict(Counter(row["answer"] for row in rows)) for name, (rows, _) in splits.items()}
    write_json(output / "run_config.json", {
        "base_checkpoint_sha256": sha256(args.base_model),
        "data_sha256": {name: sha256(args.data / f"{name}.jsonl") for name in splits},
        "counts": counts, "seed": SEED, "epochs": args.epochs, "batch_size": 8,
        "learning_rate": args.learning_rate, "device": str(device), "torch": torch.__version__,
    })
    train_batches = DataLoader(splits["train"][1], batch_size=8, shuffle=True, collate_fn=collate_examples,
                               generator=torch.Generator().manual_seed(SEED))
    dev_batches = DataLoader(splits["dev"][1], batch_size=8, collate_fn=collate_examples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    base_dev_loss = epoch_loss(model, dev_batches, device)
    print(f"{device} | {counts} | 微调前验证损失 {base_dev_loss:.4f}", flush=True)

    best_loss = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = epoch_loss(model, train_batches, device, optimizer)
        dev_loss = epoch_loss(model, dev_batches, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss})
        print(f"第 {epoch:02d} 轮 | 训练损失 {train_loss:.4f} | 验证损失 {dev_loss:.4f}", flush=True)
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            torch.save({"model_config": checkpoint["model_config"], "vocabulary": vocabulary,
                        "model_state": model.state_dict()}, output / "best_model.pt")
    write_json(output / "history.json", history)

    # 只在权重选定后运行最终测试。两边使用相同提示和贪心解码。
    base_model, _ = load_checkpoint(args.base_model, device)
    tuned_model, _ = load_checkpoint(output / "best_model.pt", device)
    test_rows = splits["test"][0]
    comparisons = []
    for row in test_rows:
        complaint = row["complaint"]
        comparisons.append({**row, "base": predict(base_model, vocabulary, complaint, device),
                            "sft": predict(tuned_model, vocabulary, complaint, device)})
    write_json(output / "test_predictions.json", comparisons)
    summary = {"best_epoch": best_epoch, "base_dev_loss": base_dev_loss, "best_dev_loss": best_loss,
               "test_samples": len(comparisons)}
    for version in ("base", "sft"):
        summary[version] = {
            "correct": sum(row[version]["valid_format"] and row[version]["answer"] == row["answer"]
                           for row in comparisons),
            "valid_format": sum(row[version]["valid_format"] for row in comparisons),
        }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-sample", action="store_true", help="打印一条样本的逐字标签。")
    mode.add_argument("--predict", help="对一条投诉比较微调前后的输出。")
    parser.add_argument("--base-model", type=Path, help="第四周的 best_model.pt。")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="含三个 JSONL 划分的目录。")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.learning_rate < math.inf:
        parser.error("训练轮数与学习率必须大于零。")
    if not (args.check_sample or args.predict is not None) and args.output.exists():
        parser.error("输出目录已存在，请用 --output 指定新的目录。")

    if args.check_sample and args.base_model is None:
        check_example()
        return
    if args.base_model is None:
        parser.error("训练和预测需要用 --base-model 指定第四周权重。")

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.base_model, device)
    vocabulary = checkpoint["vocabulary"]
    if args.check_sample:
        check_example(vocabulary, model.config.context_length)
    elif args.predict is not None:
        tuned_model, _ = load_checkpoint(args.output / "best_model.pt", device)
        for name, version in (("微调前", model), ("微调后", tuned_model)):
            print(f"{name} {json.dumps(predict(version, vocabulary, args.predict, device), ensure_ascii=False)}")
    else:
        train(model, checkpoint, args, device)


if __name__ == "__main__":
    main()
