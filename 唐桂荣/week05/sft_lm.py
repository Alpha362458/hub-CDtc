"""第 5 周：加载第 4 周的字符级 Transformer，进行全参数 SFT。

先运行 python sft_lm.py inspect，再运行 python sft_lm.py train。
用法及原理见同目录 README_SFT.md。训练依赖 PyTorch，自动绘图使用 Matplotlib。
"""

import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 第 4 周的代码与权重已经复制到本项目内，不再依赖外部课程目录。
from pretrained.transformer_lm import TransformerLM


HERE = Path(__file__).resolve().parent
BASE = HERE / "pretrained" / "tf_mask.pt"
TRAIN = HERE / "sft_train" / "sft_train.jsonl"
VAL = HERE / "sft_train" / "sft_val.jsonl"
OUTPUT = HERE / "sft_output"
PAD, EOS, UNK = "<pad>", "<eos>", "<unk>"
IGNORE = -100
DEFAULT_TEMPLATE = {"prompt": "问题：{instruction}\n回答：", "answer_prefix": "结论："}


def read_rows(path, limit=0):
    """严格检查数据；不修改原始 JSONL，也不默默丢掉错误行。"""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                for key in ("instruction", "output"):
                    if not isinstance(row[key], str) or not row[key].strip():
                        raise ValueError(f"{key} 必须是非空字符串")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            rows.append({key: row[key].strip() for key in ("instruction", "output")})
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"数据文件为空：{path}")
    return rows


def load_checkpoint(path):
    # 仅加载权重及基础类型；不使用 weights_only=False 执行 pickle 中的任意对象。
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    for key in ("model_state", "char2idx", "args"):
        if key not in ck:
            raise ValueError(f"检查点缺少 {key}：{path}")
    vocab = ck["char2idx"]
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("词表编号必须从 0 开始连续排列")
    if ck["args"].get("no_mask", False):
        raise ValueError("SFT 必须使用有因果掩码的模型，请选择 tf_mask.pt")
    return ck


def make_model(vocab, config):
    return TransformerLM(
        len(vocab), config["d_model"], config["heads"], config["layers"],
        config["seq_len"], config.get("dropout", 0.1),
        use_pos=not config.get("no_pos", False), causal=True,
    )


def prompt_text(instruction, template):
    return template["prompt"].format(instruction=instruction.strip())


def completion_text(row, template):
    # 前缀属于需要模型生成的答案，不把它预先塞到推理输入里。
    return template["answer_prefix"] + row["output"]


def extend_vocab(original, rows, template):
    """只用训练集扩词表；旧字符的编号保持不变，验证集不参与建表。"""
    vocab = dict(original)
    for special in (PAD, EOS, UNK):
        if special not in vocab:
            vocab[special] = len(vocab)
    text = "".join(prompt_text(r["instruction"], template) + completion_text(r, template)
                   for r in rows)
    for char in sorted(set(text)):
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab


def prepare_model(base, rows, template, max_len):
    """扩展嵌入与位置编码，只初始化新增行，逐项复制其余预训练权重。"""
    if base.get("stage"):
        raise ValueError("--base-ckpt 应指向第 4 周的预训练模型，而不是 SFT 检查点")
    vocab = extend_vocab(base["char2idx"], rows, template)
    config = dict(base["args"])
    # 不缩短原模型的位置表；max_len 同时限制数据长度和生成上下文。
    config["seq_len"] = max(max_len, config["seq_len"])
    model = make_model(vocab, config)
    state = model.state_dict()
    with torch.no_grad():
        for key, old in base["model_state"].items():
            new = state[key]
            if old.shape == new.shape:
                new.copy_(old)
            elif key in ("tok.weight", "head.weight", "pos.weight"):
                if old.shape[1:] != new.shape[1:] or old.shape[0] > new.shape[0]:
                    raise ValueError(f"不能扩展权重：{key} {old.shape} -> {new.shape}")
                new[:old.shape[0]].copy_(old)
            else:
                raise ValueError(f"意外的权重形状变化：{key}")
    model.load_state_dict(state)
    # TransformerLM 的输入嵌入和输出头共享同一个参数，扩展后仍然保持共享。
    assert model.tok.weight is model.head.weight
    return model, vocab, config


def encode(text, vocab):
    # 不再像第 4 周那样删掉词表外字符；验证/推理的新字符用 UNK 表示。
    return [vocab.get(char, vocab[UNK]) for char in text]


class SFTDataset(Dataset):
    def __init__(self, rows, vocab, template, max_len):
        self.examples = []
        self.skipped = 0
        self.unknown_chars = 0
        for row in rows:
            prompt = prompt_text(row["instruction"], template)
            answer = completion_text(row, template)
            prompt_ids, answer_ids = encode(prompt, vocab), encode(answer, vocab)
            # EOS 是一个独立 id，不是把字符串 '<eos>' 拆成 5 个字符。
            full = prompt_ids + answer_ids + [vocab[EOS]]
            # x/y 错位后长度为 len(full)-1；超长样本整条跳过，避免截断答案。
            if len(full) - 1 > max_len:
                self.skipped += 1
                continue
            x, y = full[:-1], full[1:]
            # y[t] 对应 full[t+1]，因此只遮掉前 len(prompt_ids)-1 个标签。
            # 最后一个提示字符的位置，恰好负责预测答案的第一个字符！
            labels = [IGNORE] * (len(prompt_ids) - 1) + y[len(prompt_ids) - 1:]
            assert len(x) == len(labels)
            self.examples.append((x, labels))
            self.unknown_chars += sum(c not in vocab for c in prompt + answer)
        if not self.examples:
            raise ValueError(f"没有可用样本：{len(rows)} 条均超过 max_len={max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PadBatch:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        length = max(len(x) for x, _ in examples)
        x_batch = torch.full((len(examples), length), self.pad_id, dtype=torch.long)
        labels = torch.full((len(examples), length), IGNORE, dtype=torch.long)
        for index, (x, y) in enumerate(examples):
            x_batch[index, :len(x)] = torch.tensor(x)
            labels[index, :len(y)] = torch.tensor(y)
        # 右侧补齐：因果掩码使真实字符看不到后面的 PAD；PAD 标签不参与 loss。
        return x_batch, labels


def select_device(name):
    if name == "auto":
        name = ("cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, labels in loader:
        x, labels = x.to(device), labels.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=IGNORE, reduction="sum",
        ).item()
        total_tokens += labels.ne(IGNORE).sum().item()
    loss = total_loss / total_tokens
    return {"answer_loss": loss, "answer_ppl": math.exp(loss) if loss < 700 else None,
            "supervised_tokens": total_tokens}


@torch.no_grad()
def generate_text(model, vocab, config, template, instruction, device,
                  max_new_tokens=80, temperature=0.0, top_k=40, seed=42):
    model.eval()
    prompt = prompt_text(instruction, template)
    ids = encode(prompt, vocab)
    if len(ids) > config["seq_len"]:
        raise ValueError(f"提示有 {len(ids)} 字符，超过上下文 {config['seq_len']}；请缩短问题")
    unknown = sum(c not in vocab for c in prompt)
    inverse = {i: char for char, i in vocab.items()}
    generated = []
    # CPU 上采样可兼容 CPU/CUDA/MPS，局部生成器不会改变训练的随机状态。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-config["seq_len"]:]], device=device)
        logits = model(context)[0, -1].float().cpu().clone()
        logits[vocab[PAD]] = -float("inf")
        logits[vocab[UNK]] = -float("inf")
        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            logits /= temperature
            if top_k:
                threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits[logits < threshold] = -float("inf")
            next_id = torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
        if next_id == vocab[EOS]:
            return {"text": "".join(generated), "ended_with_eos": True,
                    "unknown_prompt_chars": unknown}
        ids.append(next_id)
        generated.append(inverse[next_id])
    return {"text": "".join(generated), "ended_with_eos": False,
            "unknown_prompt_chars": unknown}


def prefix_rate(model, vocab, config, template, rows, device, count):
    """只测能否自主生成固定前缀，不把这个分数当作内容正确率。"""
    eligible = [r for r in rows if len(prompt_text(r["instruction"], template))
                <= config["seq_len"]][:count]
    if not eligible:
        return None
    prefix = template["answer_prefix"]
    hits = sum(generate_text(model, vocab, config, template, r["instruction"], device,
                             max_new_tokens=len(prefix))["text"].startswith(prefix)
               for r in eligible)
    return {"correct": hits, "total": len(eligible), "rate": hits / len(eligible)}


def save_model(path, model, vocab, config, template, stage, metadata):
    # clone 生成独立的 CPU 快照，保存文件可在其他设备重新加载。
    torch.save({
        "format_version": 1, "stage": stage,
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "char2idx": vocab, "idx2char": {i: c for c, i in vocab.items()},
        "args": config, "template": template, "metadata": metadata,
    }, path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def inspect_data(args):
    base = load_checkpoint(args.base_ckpt)
    train, val = read_rows(args.train_data), read_rows(args.val_data)
    template = {**DEFAULT_TEMPLATE, "answer_prefix": args.answer_prefix}
    vocab = extend_vocab(base["char2idx"], train, template)
    print(f"预训练模型：{args.base_ckpt}\n原词表：{len(base['char2idx'])}，扩展后：{len(vocab)}")
    print(f"原上下文：{base['args']['seq_len']}，计划使用：{max(args.max_len, base['args']['seq_len'])}")
    for name, rows in (("训练集", train), ("验证集", val)):
        lengths = [len(prompt_text(r["instruction"], template) + completion_text(r, template))
                   for r in rows]  # 即 full 加 EOS 再错位后的输入长度
        dataset = SFTDataset(rows, vocab, template, max(args.max_len, base["args"]["seq_len"]))
        print(f"{name}：{len(rows)} 条，输入长度 {min(lengths)}~{max(lengths)}；"
              f"使用 {len(dataset)}，超长跳过 {dataset.skipped}，UNK 字符 {dataset.unknown_chars}")
    overlap = len({(r['instruction'], r['output']) for r in train} &
                  {(r['instruction'], r['output']) for r in val})
    print(f"训练/验证完全相同的问答对：{overlap}。原始答案需人工检查内容正确性。")


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = select_device(args.device)
    base = load_checkpoint(args.base_ckpt)
    train_rows = read_rows(args.train_data, args.train_limit)
    val_rows = read_rows(args.val_data, args.val_limit)
    template = {**DEFAULT_TEMPLATE, "answer_prefix": args.answer_prefix}
    model, vocab, config = prepare_model(base, train_rows, template, args.max_len)
    model.to(device)
    train_data = SFTDataset(train_rows, vocab, template, config["seq_len"])
    val_data = SFTDataset(val_rows, vocab, template, config["seq_len"])
    collate = PadBatch(vocab[PAD])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate)
    output = Path(args.output_dir).resolve()
    # 避免重复运行时覆盖上一次实验；换 output-dir 即可创建新的实验。
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录非空：{output}；请用 --output-dir 指定新的目录")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"base_checkpoint": str(Path(args.base_ckpt).resolve()),
                "seed": args.seed, "train_samples": len(train_data), "val_samples": len(val_data)}
    run_config = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
    write_json(output / "run_config.json", run_config)
    save_model(output / "pre_sft.pt", model, vocab, config, template, "pre_sft", metadata)
    print(f"device={device}，参数量={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"原词表 {len(base['char2idx'])} -> {len(vocab)}，"
          f"上下文 {base['args']['seq_len']} -> {config['seq_len']}", flush=True)
    for name, dataset in (("训练", train_data), ("验证", val_data)):
        print(f"{name}样本 {len(dataset)}，超长跳过 {dataset.skipped}，"
              f"UNK 字符 {dataset.unknown_chars}", flush=True)
    # baseline 必须在优化器更新之前测量，且和训练后的验证指标采用相同模板。
    baseline = evaluate(model, val_loader, device)
    baseline["prefix_accuracy"] = prefix_rate(
        model, vocab, config, template, val_rows, device, args.format_eval_samples)
    history = {"baseline": baseline, "epochs": []}
    write_json(output / "history.json", history)
    print(f"微调前：{json.dumps(baseline, ensure_ascii=False)}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_loss, stale, global_step = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        total_loss, total_tokens = 0.0, 0
        for batch_index, (x, labels) in enumerate(train_loader, 1):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, len(vocab)), labels.reshape(-1),
                                   ignore_index=IGNORE)
            if not torch.isfinite(loss):
                raise RuntimeError("训练出现非有限 loss，请降低学习率并检查数据")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = labels.ne(IGNORE).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            global_step += 1
            if batch_index == 1 or batch_index % args.log_every == 0:
                print(f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)} "
                      f"step={global_step} loss={loss.item():.4f}", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch, "global_step": global_step,
                        "train_answer_loss": total_loss / total_tokens,
                        "elapsed_seconds": round(time.time() - started, 2),
                        "prefix_accuracy": prefix_rate(model, vocab, config, template, val_rows,
                                                       device, args.format_eval_samples)})
        history["epochs"].append(metrics)
        write_json(output / "history.json", history)
        improved = metrics["answer_loss"] < best_loss
        if improved:
            best_loss, stale = metrics["answer_loss"], 0
            save_model(output / "best.pt", model, vocab, config, template, "sft",
                       {**metadata, "metrics": metrics})
        else:
            stale += 1
        print(f"验证：{json.dumps(metrics, ensure_ascii=False)}"
              f"{' [已保存最佳模型]' if improved else ''}", flush=True)
        if args.max_steps and global_step >= args.max_steps:
            print("达到 --max-steps，结束本次训练。", flush=True)
            break
        if args.patience and stale >= args.patience:
            print("验证 loss 连续多轮未改善，提前停止。", flush=True)
            break
    save_model(output / "last.pt", model, vocab, config, template, "sft",
               {**metadata, "metrics": history["epochs"][-1]})
    print(f"训练完成。最佳模型：{output / 'best.pt'}\n训练记录：{output / 'history.json'}", flush=True)
    try:
        from plot_sft import render_history
        for path in render_history(output / "history.json"):
            print(f"已生成：{path}", flush=True)
    except (ImportError, OSError, ValueError, KeyError) as error:
        print(f"模型已保存，绘图未完成：{error}\n"
              "安装 matplotlib 后可单独运行 plot_sft.py 绘图。", flush=True)


def load_runtime(path, device):
    ck = load_checkpoint(path)
    if "template" not in ck or not all(s in ck["char2idx"] for s in (PAD, EOS, UNK)):
        raise ValueError("请加载本脚本生成的 pre_sft.pt / best.pt / last.pt")
    model = make_model(ck["char2idx"], ck["args"])
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval(), ck


def infer(args):
    device = select_device(args.device)
    model, ck = load_runtime(args.checkpoint, device)
    models = [("SFT 后", model, ck)]
    if args.command == "compare":
        baseline_path = args.baseline or Path(args.checkpoint).parent / "pre_sft.pt"
        before, original = load_runtime(baseline_path, device)
        if original.get("stage") != "pre_sft":
            raise ValueError("对比基线应为训练前保存的 pre_sft.pt")
        if (original["char2idx"] != ck["char2idx"] or original["template"] != ck["template"]
                or original["args"] != ck["args"]):
            raise ValueError("基线与 SFT 模型的词表、结构或模板不同，请选同一次实验的模型")
        models.insert(0, ("SFT 前（扩展后、更新前的权重）", before, original))
    records = []
    for instruction in args.instruction:
        print(f"\n问题：{instruction}", flush=True)
        record = {"instruction": instruction, "outputs": {}}
        for name, current, checkpoint in models:
            result = generate_text(current, checkpoint["char2idx"], checkpoint["args"],
                                   checkpoint["template"], instruction, device,
                                   args.max_new_tokens, args.temperature, args.top_k, args.seed)
            print(f"{name}：{result['text']}", flush=True)
            print(f"  生成EOS={result['ended_with_eos']}，"
                  f"输入UNK字符={result['unknown_prompt_chars']}", flush=True)
            record["outputs"][name] = result
        records.append(record)
    if args.save_json:
        target = Path(args.save_json)
        if target.exists():
            raise ValueError(f"结果文件已存在：{target}；请指定新的文件名")
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, records)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "train", "generate", "compare"):
        child = sub.add_parser(command)
        child.add_argument("--threads", type=positive_int, default=4,
                           help="CPU 计算线程数；可根据机器调整")
        if command in ("inspect", "train"):
            child.add_argument("--base-ckpt", type=Path, default=BASE)
            child.add_argument("--train-data", type=Path, default=TRAIN)
            child.add_argument("--val-data", type=Path, default=VAL)
            child.add_argument("--max-len", type=positive_int, default=128)
            child.add_argument("--answer-prefix", default=DEFAULT_TEMPLATE["answer_prefix"])
        if command == "train":
            child.add_argument("--output-dir", type=Path, default=OUTPUT)
            child.add_argument("--epochs", type=positive_int, default=10)
            child.add_argument("--batch-size", type=positive_int, default=16)
            child.add_argument("--lr", type=float, default=3e-4)
            child.add_argument("--patience", type=nonnegative_int, default=3,
                               help="验证 loss 早停轮数，0 表示关闭")
            child.add_argument("--train-limit", type=nonnegative_int, default=0)
            child.add_argument("--val-limit", type=nonnegative_int, default=0)
            child.add_argument("--max-steps", type=nonnegative_int, default=0,
                               help="总更新步数上限；0 表示不限制，用于快速验证")
            child.add_argument("--format-eval-samples", type=nonnegative_int, default=20,
                               help="用多少条验证问题测前缀符合率，0 表示关闭")
            child.add_argument("--log-every", type=positive_int, default=20)
        if command in ("generate", "compare"):
            child.add_argument("--checkpoint", type=Path, default=OUTPUT / "best.pt")
            child.add_argument("--instruction", nargs="+", required=True,
                               help="一个或多个问题，每个问题用引号括起来")
            child.add_argument("--max-new-tokens", type=positive_int, default=80)
            child.add_argument("--temperature", type=float, default=0.0,
                               help="0 为贪心解码，便于复现实验")
            child.add_argument("--top-k", type=nonnegative_int, default=40)
            child.add_argument("--save-json", type=Path)
        if command == "compare":
            child.add_argument("--baseline", type=Path,
                               help="默认使用 checkpoint 同目录下的 pre_sft.pt")
        if command != "inspect":
            child.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
            child.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if hasattr(args, "lr") and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr 必须是有限正数")
    if hasattr(args, "temperature") and (not math.isfinite(args.temperature) or args.temperature < 0):
        parser.error("--temperature 必须是有限非负数")
    if hasattr(args, "answer_prefix") and not args.answer_prefix.strip():
        parser.error("--answer-prefix 不能为空")
    torch.set_num_threads(args.threads)
    if args.command == "inspect":
        inspect_data(args)
    elif args.command == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"错误：{error}") from error
