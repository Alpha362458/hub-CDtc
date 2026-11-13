"""从 week04 检查点开始 SFT：准备问答、训练、验证、保存最佳模型。

默认使用脚本旁边的检查点与 sft_train 目录，运行本文件即可开始训练。
所有执行操作放在 main() 中，导入本模块时不会加载模型或启动训练。
"""

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SFT_CHECKPOINT,
    PROMPT_TEMPLATE,
    format_prompt,
    load_checkpoint,
    select_device,
)

BASE_DIR = Path(__file__).resolve().parent
IGNORE_INDEX = -100


def load_jsonl(path):
    """每行读取一条问答；数据异常时给出文件名和行号。"""
    samples = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} JSON 格式错误：{error.msg}"
                ) from error
            if not isinstance(sample, dict) or any(
                not isinstance(sample.get(key), str) or not sample[key].strip()
                for key in ("instruction", "output")
            ):
                raise ValueError(
                    f"{path}:{line_number} instruction 和 output 必须为非空字符串。"
                )
            samples.append(sample)
    if not samples:
        raise ValueError(f"数据集为空：{path}")
    return samples


def ensure_unk(model, char2idx, idx2char):
    """追加专用 UNK，保留旧编号和旧权重；已有 UNK 时直接复用。"""
    if "<UNK>" not in char2idx:
        unk_id = len(char2idx)
        char2idx["<UNK>"] = unk_id
        idx2char[unk_id] = "<UNK>"

        # 只补充最后一行，新参数从零开始，原参数数值不变。
        # 必须在创建优化器之前完成，优化器才会管理这些新参数。
        with torch.no_grad():
            model.embed.weight = torch.nn.Parameter(
                F.pad(model.embed.weight, (0, 0, 0, 1))
            )
            model.fc.weight = torch.nn.Parameter(F.pad(model.fc.weight, (0, 0, 0, 1)))
            model.fc.bias = torch.nn.Parameter(F.pad(model.fc.bias, (0, 1)))
        model.embed.num_embeddings += 1
        model.fc.out_features += 1

    if (
        len(char2idx) != model.embed.num_embeddings
        or len(char2idx) != model.fc.out_features
    ):
        raise ValueError("字表大小与模型嵌入层、输出层不一致。")
    return char2idx["<UNK>"]


def prepare_sample(sample, char2idx, unk_id, max_seq_len):
    """统一完成问答格式整理、UNK 编码、回答截断和监督标签构造。"""
    prompt = format_prompt(sample["instruction"])
    response = sample["output"] + "\n"
    prompt_ids = [char2idx.get(char, unk_id) for char in prompt]
    response_ids = [char2idx.get(char, unk_id) for char in response]

    # 输入和目标要错开一位，所以完整序列可以比输入上限多一个编号。
    # 保留完整问题，剩余位置用于回答；当前语料的提示最长为 53。
    answer_budget = max_seq_len + 1 - len(prompt_ids)
    if answer_budget < 1:
        raise ValueError("问题过长，请缩短 instruction，为回答至少预留一个目标位置。")
    truncated = len(response_ids) > answer_budget
    response_ids = response_ids[:answer_budget]

    # 截断后不强行补结束换行，避免教模型在不完整的答案处结束。
    full_ids = prompt_ids + response_ids
    x = torch.tensor(full_ids[:-1], dtype=torch.long)
    y = torch.tensor(full_ids[1:], dtype=torch.long)

    # 标签已经错位：首个回答目标位于 len(prompt_ids) - 1。
    # 只屏蔽问题和格式文字，保留全部回答目标及真实结束换行。
    y[: len(prompt_ids) - 1] = IGNORE_INDEX
    return {"x": x, "y": y, "truncated": truncated}


def collate_batch(batch, pad_id):
    """右侧补齐；输入使用已有编号，新增补齐标签用 -100 忽略。"""
    # 因果掩码保证有效位置看不到右侧填充；真实换行标签仍参与损失。
    x = pad_sequence(
        [sample["x"] for sample in batch], batch_first=True, padding_value=pad_id
    )
    y = pad_sequence(
        [sample["y"] for sample in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    return x, y


def run_epoch(
    model, loader, criterion, device, optimizer=None, log_every=0, grad_clip=1.0
):
    """传入优化器时训练，不传时验证；按有效回答字符统计平均损失。"""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    # eval() 控制 Dropout；这里另行控制验证期间不建立梯度计算图。
    with torch.set_grad_enabled(training):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            valid_tokens = (y != IGNORE_INDEX).sum().item()
            if not valid_tokens:
                raise ValueError("当前批次没有有效回答标签。")
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            # 数据准备时已完成错位，这里直接对齐，不再次移动标签。
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if not torch.isfinite(loss).item():
                raise ValueError("损失出现 NaN 或 Inf，请检查数据和学习率。")

            if training:
                loss.backward()
                if grad_clip > 0:
                    # 限制过大的梯度，降低单次更新不稳定的可能性。
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip, error_if_nonfinite=True
                    )
                optimizer.step()

            # loss 已是有效标签的平均值，整轮统计也按有效字符数加权。
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            if (
                training
                and log_every
                and (step % log_every == 0 or step == len(loader))
            ):
                print(
                    f"  批次 {step}/{len(loader)}，回答 loss={total_loss / total_tokens:.4f}",
                    flush=True,
                )

    if not total_tokens:
        raise ValueError("数据集没有可计算损失的回答字符。")
    return total_loss / total_tokens


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT), help="微调起点"
    )
    parser.add_argument(
        "--train", type=Path, default=BASE_DIR / "sft_train/sft_train.jsonl"
    )
    parser.add_argument(
        "--val", type=Path, default=BASE_DIR / "sft_train/sft_val.jsonl"
    )
    parser.add_argument(
        "--save", type=Path, default=DEFAULT_SFT_CHECKPOINT, help="最佳模型保存路径"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="每隔多少训练批次打印进度；0 表示关闭"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="梯度裁剪上限；0 表示关闭"
    )
    parser.add_argument(
        "--limit_train", type=int, default=0, help="仅使用前 N 条训练数据；0 表示全部"
    )
    parser.add_argument(
        "--limit_val", type=int, default=0, help="仅使用前 N 条验证数据；0 表示全部"
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs 和 batch_size 必须大于 0。")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须为有限正数。")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        parser.error("grad_clip 必须为有限非负数。")
    if min(args.limit_train, args.limit_val, args.log_every) < 0:
        parser.error("limit_train、limit_val 和 log_every 不能为负数。")
    for name in ("checkpoint", "train", "val", "save"):
        setattr(args, name, getattr(args, name).resolve())
    if args.save == args.checkpoint:
        parser.error("save 必须与 checkpoint 不同，避免覆盖微调起点。")
    return parser, args


def main():
    parser, args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        device = select_device(args.device)
        model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)
        # 原结构配置必须保留，推理加载器依赖其中的层数、维度和位置长度。
        model_args = torch.load(args.checkpoint, map_location="cpu", weights_only=True)[
            "args"
        ]
        unk_id = ensure_unk(model, char2idx, idx2char)
        max_seq_len = model.position_embed.num_embeddings

        loaders = {}
        for name, path, limit in (
            ("训练集", args.train, args.limit_train),
            ("验证集", args.val, args.limit_val),
        ):
            raw_samples = load_jsonl(path)
            if limit:
                raw_samples = raw_samples[:limit]
            samples = [
                prepare_sample(sample, char2idx, unk_id, max_seq_len)
                for sample in raw_samples
            ]
            loaders[name] = DataLoader(
                samples,
                batch_size=args.batch_size,
                shuffle=name == "训练集",
                collate_fn=partial(collate_batch, pad_id=char2idx["\n"]),
                drop_last=False,
                num_workers=0,
            )
            truncated_count = sum(sample["truncated"] for sample in samples)
            print(
                f"{name}：{len(samples)} 条，回答截断：{truncated_count} 条，批次：{len(loaders[name])}"
            )

        # UNK 扩展完成后只创建一次优化器，每个批次复用它。
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        training_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        best_val_loss = float("inf")
        best_epoch = 0
        print(
            f"设备：{device}，字表：{len(char2idx)}，输入上限：{max_seq_len}，学习率：{args.lr}"
        )

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
            train_loss = run_epoch(
                model,
                loaders["训练集"],
                criterion,
                device,
                optimizer=optimizer,
                log_every=args.log_every,
                grad_clip=args.grad_clip,
            )
            val_loss = run_epoch(model, loaders["验证集"], criterion, device)
            elapsed = time.perf_counter() - started
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss, best_epoch = val_loss, epoch
                # 权重、两张字表与结构配置一起保存，确保 UNK 编号可恢复。
                # 微调参数独立保存，不能替代原来的模型结构配置 args。
                torch.save(
                    {
                        "architecture": "causal_transformer_encoder",
                        "model_state": model.state_dict(),
                        "char2idx": char2idx,
                        "idx2char": idx2char,
                        "args": model_args,
                        "training_args": training_args,
                        "sft_format": {
                            "prompt_template": PROMPT_TEMPLATE,
                            "stop_char": "\n",
                            "unk_token": "<UNK>",
                        },
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                    args.save,
                )
            marker = "，已保存最佳模型" if improved else ""
            print(
                f"训练回答 loss={train_loss:.4f}，验证回答 loss={val_loss:.4f}，耗时 {elapsed:.1f}s{marker}",
                flush=True,
            )

        print(f"\n训练完成：最佳轮次 {best_epoch}，验证回答 loss={best_val_loss:.4f}")
        print(f"模型：{args.save}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
