"""加载微调后的模型进行推理演示。

支持两种加载方式：
  1. 加载 LoRA adapter（默认，体积小）；
  2. 加载已合并的全权重模型（--merged）。

用法：
    python inference.py
    python inference.py --adapter outputs/lora
    python inference.py --merged outputs/merged
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = Path(__file__).resolve().parent
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = HERE / "outputs" / "lora"


def load_model(base: str, adapter: str | None, merged: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        merged or base, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if merged:
        model = AutoModelForCausalLM.from_pretrained(
            merged, dtype=torch.bfloat16, trust_remote_code=True
        )
    else:
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=torch.bfloat16, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, instruction: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--merged", default=None,
                        help="若指定则直接加载合并后的全权重模型")
    args = parser.parse_args()

    tokenizer, model = load_model(args.base, args.adapter, args.merged)

    samples = [
        "给定两个数字，请选择其中的较大数并将其打印出来。 数字1: 128 数字2: 96",
        "计算一组数字的平均值和标准差。 数字：10, 20, 30, 40, 50",
        "请把下面这句话翻译成英文：今天天气真好。",
    ]
    print("=" * 60)
    for s in samples:
        print(f"[user] {s}")
        print(f"[assistant] {generate(tokenizer, model, s)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
