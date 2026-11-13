import argparse
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import SYSTEM_PROMPT

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "sft_train"
MODEL_PATH = ROOT_DIR.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
CKPT_PATH = ROOT_DIR / "checkpoints"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_dir, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.to(get_device()).eval()
    return model, tokenizer


@torch.no_grad()
def generate(model, tokenizer, instruction, max_new_tokens=256, temperature=0.7):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    output = model.generate(**inputs, **gen_kwargs)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=CKPT_PATH/"qwen2-0.5b-sft", help="合并后的模型目录，不传则按是否加载 adapter 自动选择")
    parser.add_argument("--adapter", default=None, help="直接加载 LoRA adapter（此时 model_dir 为基座模型目录）")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model_dir = args.model_dir or (
        MODEL_PATH if args.adapter else CKPT_PATH/"qwen2-0.5b-sft"
    )
    model, tokenizer = load_model(model_dir, args.adapter)

    if args.instruction:
        print(generate(model, tokenizer, args.instruction, args.max_new_tokens, args.temperature))
        return

    print("输入问题回车即可生成回答，输入 exit 退出。")
    while True:
        instruction = input("\n你: ").strip()
        if instruction.lower() in {"exit", "quit", "q"} or not instruction:
            break
        answer = generate(model, tokenizer, instruction, args.max_new_tokens, args.temperature)
        print(f"模型: {answer}")


if __name__ == "__main__":
    main()
