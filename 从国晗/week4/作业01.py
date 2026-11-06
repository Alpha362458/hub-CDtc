"""第四周作业：基于 Transformer 的字符级中文语言模型。
用法：python 作业01.py --corpus corpus_general.txt
仅使用课程语料；默认从随机权重训练。依赖：torch>=2.6、numpy>=1.26。
结构参考：https://github.com/karpathy/nanoGPT （MIT）。
"""


import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parent
PAD_ID, UNKNOWN_ID = 0, 1


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(corpus_path, data_directory, seed=42):
    """先对整行去重并划分，再只用 train 创建词表；不删除未知字符。"""
    data_directory = Path(data_directory)
    data_directory.mkdir(parents=True, exist_ok=True)
    source_text = Path(corpus_path).read_text(encoding="utf-8-sig")
    source_lines = source_text.splitlines()
    unique_lines = list(dict.fromkeys(line.strip() for line in source_lines if line.strip()))
    random.Random(seed).shuffle(unique_lines)
    train_end, dev_end = int(len(unique_lines) * 0.90), int(len(unique_lines) * 0.95)
    split_lines = {
        "train": unique_lines[:train_end],
        "dev": unique_lines[train_end:dev_end],
        "test": unique_lines[dev_end:],
    }
    if any(not lines for lines in split_lines.values()):
        raise ValueError("去重后的语料太少，无法划分三个集合。")
    texts = {name: "\n".join(lines) + "\n" for name, lines in split_lines.items()}
    counts = Counter(texts["train"])
    # 频率相同的字按 Unicode 排序，保证词表顺序稳定。
    vocabulary = ["<PAD>", "<UNK>"] + sorted(counts, key=lambda char: (-counts[char], char))
    character_to_id = {char: index for index, char in enumerate(vocabulary)}
    audit = {
        "source_sha256": sha256(corpus_path), "source_characters": len(source_text),
        "source_lines": len(source_lines), "unique_lines": len(unique_lines),
        "removed_empty_or_duplicate_lines": len(source_lines) - len(unique_lines),
        "seed": seed, "split_ratio": [0.90, 0.05, 0.05],
        "vocabulary_source": "train only", "vocabulary_size": len(vocabulary),
        "normalization": "splitlines, strip surrounding whitespace, exact line deduplication",
        "splits": {},
    }
    for name, text in texts.items():
        path = data_directory / f"{name}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")
        unknown = sum(char not in character_to_id for char in text)
        audit["splits"][name] = {
            "lines": len(split_lines[name]), "characters": len(text),
            "sha256": sha256(path), "unknown_characters": unknown,
            "unknown_ratio": unknown / len(text),
        }
    write_json(data_directory / "vocab.json", vocabulary)
    audit["vocabulary_sha256"] = sha256(data_directory / "vocab.json")
    write_json(data_directory / "data_audit.json", audit)
    return audit


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
        next_id = indices.gather(-1, torch.multinomial(functional.softmax(values, dim=-1), 1))
        char = vocabulary[next_id.item()]
        if char == "\n":
            break
        result += char
        ids = torch.cat((ids, next_id), dim=1)
    return result


def is_prose(line):
    return len(line) >= 30 and any(mark in line for mark in "，。！？；") and "?" not in line and "？" not in line


class SentenceDataset(Dataset):
    def __init__(self, text, vocabulary, context_length):
        character_to_id = {char: index for index, char in enumerate(vocabulary)}
        self.boundary_id = character_to_id["\n"]
        self.samples = []
        self.prose_indices = []
        for line in text.splitlines():
            if not line:
                continue
            if len(line) + 1 > context_length:
                raise ValueError("存在超出上下文的句子；请增加 context_length，不能悄悄截断。")
            ids = [character_to_id.get(char, UNKNOWN_ID) for char in line]
            self.samples.append(torch.tensor([self.boundary_id] + ids + [self.boundary_id], dtype=torch.long))
            if is_prose(line):
                self.prose_indices.append(len(self.samples) - 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample[:-1], sample[1:]


class LengthBatchSampler(Sampler):
    """长度相近的句子一起训练，避免短句填充到 128 字造成浪费。"""
    def __init__(self, dataset, batch_size, shuffle=False, seed=42, prose_repeat=1):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.generator = random.Random(seed)
        indices = list(range(len(dataset))) + dataset.prose_indices * (prose_repeat - 1)
        self.indices = sorted(indices, key=lambda index: len(dataset.samples[index]))

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __iter__(self):
        batches = [self.indices[start:start + self.batch_size] for start in range(0, len(self.indices), self.batch_size)]
        if self.shuffle:
            self.generator.shuffle(batches)
        yield from batches


def collate_sentences(samples):
    length = math.ceil(max(len(inputs) for inputs, _ in samples) / 8) * 8
    inputs = torch.full((len(samples), length), PAD_ID, dtype=torch.long)
    targets = torch.full((len(samples), length), -100, dtype=torch.long)
    for row, (source, target) in enumerate(samples):
        inputs[row, :len(source)] = source
        targets[row, :len(target)] = target
    return inputs, targets


def sentence_loaders(data_directory, vocabulary, context_length, batch_size=128, seed=42, prose_repeat=1):
    loaders = {}
    for name in ["train", "dev", "test"]:
        text = (Path(data_directory) / f"{name}.txt").read_text(encoding="utf-8")
        dataset = SentenceDataset(text, vocabulary, context_length)
        sampler = LengthBatchSampler(dataset, batch_size, name == "train", seed, prose_repeat if name == "train" else 1)
        loaders[name] = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_sentences, num_workers=0)
    return loaders


@torch.inference_mode()
def evaluate_sentences(model, loader, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs).float()
        valid = targets.ne(-100)
        total_loss += functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum").item()
        total_correct += (logits.argmax(-1).eq(targets) & valid).sum().item()
        total_tokens += valid.sum().item()
    loss = total_loss / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss), "next_character_accuracy": total_correct / total_tokens,
            "evaluated_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(ROOT / "corpus_general.txt"))
    parser.add_argument("--initialize-from", default=None, help="只允许本项目自己从零训练的权重；省略时随机初始化。")
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--prose-repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.prose_repeat) < 1 or args.learning_rate <= 0:
        parser.error("训练参数必须大于零。")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device(args.device)
    output, data = Path(args.output), Path(args.data)
    output.mkdir(parents=True, exist_ok=True)
    if not (data / "data_audit.json").exists():
        prepare_data(args.corpus, data, args.seed)
    vocabulary = json.loads((data / "vocab.json").read_text(encoding="utf-8"))
    audit = json.loads((data / "data_audit.json").read_text(encoding="utf-8"))
    if args.initialize_from:
        model, initial_checkpoint = load_checkpoint(args.initialize_from, device)
        if initial_checkpoint["vocabulary"] != vocabulary or initial_checkpoint["data_audit"] != audit:
            raise ValueError("初始权重的词表或数据划分与本次训练不一致。")
    else:
        model = TransformerLanguageModel(ModelConfig(len(vocabulary), 128, 256, 4, 6, 0.1)).to(device)
    loaders = sentence_loaders(data, vocabulary, model.config.context_length, args.batch_size, args.seed, args.prose_repeat)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    use_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = 100
    before_training = evaluate_sentences(model, loaders["dev"], device)
    run_info = {
        "started_at": datetime.now().astimezone().isoformat(), "arguments": vars(args),
        "model_config": asdict(model.config), "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "mixed_precision": "bfloat16" if use_bfloat16 else "float32", "evaluation_protocol": "sentence_bos",
        "training_code_sha256": sha256(__file__), "sentence_training_code_sha256": sha256(__file__),
        "data_audit": audit, "initialization": "own scratch-trained checkpoint" if args.initialize_from else "random weights",
        "initial_checkpoint_sha256": sha256(args.initialize_from) if args.initialize_from else None,
        "initial_dev": before_training,
    }
    write_json(output / "run_config.json", run_info)
    print(f"device={device}, dtype={run_info['mixed_precision']}, batches={len(loaders['train'])}, initial dev PPL={before_training['perplexity']:.2f}", flush=True)
    started, history, best_loss, bad_epochs, step = time.perf_counter(), [], math.inf, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started, loss_sum, tokens = time.perf_counter(), 0.0, 0
        for batch, (inputs, targets) in enumerate(loaders["train"], 1):
            step += 1
            factor = step / warmup_steps if step <= warmup_steps else 0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))) / 2
            optimizer.param_groups[0]["lr"] = args.learning_rate * factor
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bfloat16):
                logits = model(inputs)
                loss = functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("非有限 loss，停止更新。")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = targets.ne(-100).sum().item()
            loss_sum += loss.item() * count
            tokens += count
            if batch % 500 == 0:
                print(f"epoch {epoch:02d}, batch {batch}/{len(loaders['train'])}, train loss={loss_sum/tokens:.4f}", flush=True)
        validation = evaluate_sentences(model, loaders["dev"], device)
        improved = validation["loss"] < best_loss
        history.append({"epoch": epoch, "train_loss": loss_sum / tokens, "train_perplexity": math.exp(loss_sum / tokens),
                        "dev_loss": validation["loss"], "dev_perplexity": validation["perplexity"],
                        "dev_accuracy": validation["next_character_accuracy"], "seconds": time.perf_counter() - epoch_started,
                        "learning_rate": optimizer.param_groups[0]["lr"], "best": improved})
        write_json(output / "history.json", history)
        if improved:
            best_loss, bad_epochs = validation["loss"], 0
            checkpoint = {"model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                          "model_config": asdict(model.config), "vocabulary": vocabulary, "epoch": epoch,
                          "dev_metrics": validation, "data_audit": audit, "seed": args.seed,
                          "training_code_sha256": run_info["training_code_sha256"],
                          "sentence_training_code_sha256": run_info["sentence_training_code_sha256"],
                          "evaluation_protocol": "sentence_bos", "initial_checkpoint_sha256": run_info["initial_checkpoint_sha256"]}
            temporary = output / "best_model.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best_model.pt")
        else:
            bad_epochs += 1
        print(f"epoch {epoch:02d}: train PPL={math.exp(loss_sum/tokens):.2f}, dev PPL={validation['perplexity']:.2f}, dev accuracy={validation['next_character_accuracy']:.2%}, {history[-1]['seconds']:.1f}s {'BEST' if improved else ''}", flush=True)
        if bad_epochs >= args.patience:
            break
    elapsed = time.perf_counter() - started
    model, checkpoint = load_checkpoint(output / "best_model.pt", device)
    test_metrics = evaluate_sentences(model, loaders["test"], device)
    examples = [{"prompt": prompt, "generated": generate(model, vocabulary, "\n" + prompt, device, temperature=0.6, top_k=30).removeprefix("\n")}
                for prompt in ["想吃麦当劳", "今天天气", "今天下班后，我打算", "为了身体健康，我们应该"]
                if set(prompt).issubset(set(vocabulary))]
    summary = {"best_epoch": checkpoint["epoch"], "epochs_completed": len(history), "parameters": run_info["parameters"],
               "initial_dev": before_training, "best_dev": checkpoint["dev_metrics"], "test": test_metrics,
               "training_seconds": elapsed, "examples": examples, "checkpoint_sha256": sha256(output / "best_model.pt"),
               "evaluation_protocol": "sentence_bos", "initialization": run_info["initialization"],
               "note": "句子独立评测，每句以换行开始并预测到换行结束；与连续切块 PPL 不是同一口径。"}
    write_json(output / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
