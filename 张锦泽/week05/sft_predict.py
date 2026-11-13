import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="使用 SFT 模型生成回答")
    parser.add_argument("--model", default="./sft_model")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    prompt = f"指令：{args.instruction}\n"
    if args.input:
        prompt += f"输入：{args.input}\n"
    prompt += "回答："

    # 手动构造输入，避免 tokenizer 自动在末尾添加 [SEP]
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if tokenizer.cls_token_id is not None:
        input_ids = [tokenizer.cls_token_id] + input_ids

    inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long).to(device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long).to(device),
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print("模型回答：")
    print(answer)


if __name__ == "__main__":
    main()
