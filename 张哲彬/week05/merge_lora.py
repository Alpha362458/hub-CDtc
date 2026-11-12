import argparse
import pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT_DIR / "sft_train"
MODEL_PATH = ROOT_DIR.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
CKPT_PATH = ROOT_DIR / "checkpoints"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=MODEL_PATH)
    parser.add_argument("--adapter", default=CKPT_PATH/"sft_lora")
    parser.add_argument("--output_dir", default=CKPT_PATH/"qwen2-0.5b-sft")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"合并完成，模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
