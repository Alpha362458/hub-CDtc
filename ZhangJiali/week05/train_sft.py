"""基于语言模型的 SFT（监督微调）训练脚本。

本脚本使用 Qwen2.5-0.5B-Instruct 作为基座模型，采用 LoRA 参数高效微调，
在 `sft_train.jsonl` / `sft_val.jsonl` 上完成指令微调，并将产物
（LoRA adapter、合并后的全权重模型、训练日志）输出到 `outputs/` 目录。

数据样例：
    {"instruction": "...", "output": "..."}

运行：
    python train_sft.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 国内访问 HuggingFace 较慢，默认走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关掉 Windows 下 symlink 警告
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = HERE.parent / "sft_train" / "sft_train.jsonl"
DEFAULT_VAL_FILE = HERE.parent / "sft_train" / "sft_val.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "outputs"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_dataset(train_file: Path, val_file: Path, tokenizer) -> dict:
    """将 instruction/output 原始数据转换为 chat 格式 (prompt/completion)。

    SFTTrainer 在 `completion_only_loss=True` 时只对 completion 部分计算损失，
    从而避免模型在指令部分上浪费学习容量。
    """
    def to_chat(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            instruction = (r.get("instruction") or "").strip()
            output = (r.get("output") or "").strip()
            if not instruction or not output:
                continue
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output},
            ]
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            completion = output + tokenizer.eos_token
            out.append({"prompt": prompt, "completion": completion})
        return out

    train_rows = to_chat(load_jsonl(train_file))
    val_rows = to_chat(load_jsonl(val_file)) if val_file and val_file.exists() else []
    print(f"[data] train={len(train_rows)} val={len(val_rows)}")
    return {
        "train": Dataset.from_list(train_rows),
        "val": Dataset.from_list(val_rows) if val_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT on Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="基座模型名或本地路径")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merge_and_save", action="store_true",
                        help="训练结束后合并 LoRA 权重并保存完整模型")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] device={torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data = build_dataset(args.train_file, args.val_file, tokenizer)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir / "lora"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="steps" if data["val"] else "no",
        eval_steps=100,
        bf16=torch.cuda.is_available(),
        max_length=args.max_seq_length,
        completion_only_loss=True,
        dataset_num_proc=4,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
    )

    print("[train] start ...")
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir / "lora"))
    tokenizer.save_pretrained(str(args.output_dir / "lora"))

    metrics = {f"train_{k}": v for k, v in train_result.metrics.items() if v is not None}
    if data["val"] is not None:
        eval_metrics = trainer.evaluate()
        metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
    metrics_file = args.output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] done. metrics -> {metrics_file}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.merge_and_save:
        print("[merge] merging LoRA weights into base model ...")
        merged_dir = args.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"[merge] saved merged model -> {merged_dir}")


if __name__ == "__main__":
    main()
