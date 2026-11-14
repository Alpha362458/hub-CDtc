"""从已保存的模型继续生成字符，支持贪心或温度 + Top-k 采样。"""
import argparse
import unicodedata

import torch

from common import ROOT
from model import load_model
from train import amp_context


@torch.inference_mode()
def generate(model, vocab, prompt, max_new_tokens=60, temperature=0.8, top_k=40, seed=42):
    if max_new_tokens < 1 or temperature < 0 or top_k < 0:
        raise ValueError("生成长度必须为正，temperature 和 top_k 必须非负")
    model.eval()
    device = next(model.parameters()).device
    stoi, itos = vocab["stoi"], vocab["itos"]
    normalized_prompt = unicodedata.normalize("NFKC", prompt)
    text = normalized_prompt or "\n"
    ids = [stoi.get(c, 1) for c in text]
    if all(i == 1 for i in ids):
        raise ValueError("输入字符均不在训练字表中，请换用中文提示")
    generator = torch.Generator(device=device).manual_seed(seed)
    new_chars, stop_reason = [], "max_new_tokens"
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.config["block_size"]:]], dtype=torch.long, device=device)
        with amp_context(device):
            logits = model(x)[0, -1].float()
        logits[:2] = -float("inf")  # 不生成 PAD/UNK；不改变正式评估分布。
        if temperature == 0:
            idx = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                values, indices = torch.topk(logits, min(top_k, len(itos) - 2))
                choice = torch.multinomial(torch.softmax(values, -1), 1, generator=generator)
                idx = int(indices[choice].item())
            else:
                idx = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).item())
        ids.append(idx)
        ch = itos[idx]
        if ch == "\n":
            stop_reason = "newline"
            break
        new_chars.append(ch)
    continuation = "".join(new_chars)
    return {"prompt": prompt, "normalized_prompt": normalized_prompt, "continuation": continuation,
            "full_text": normalized_prompt + continuation, "generated_characters": len(new_chars),
            "stop_reason": stop_reason, "temperature": temperature, "top_k": top_k, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description="字符级 Transformer 续写")
    parser.add_argument("--prompt", default="今天天气")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 表示贪心解码")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(2)
    model, checkpoint = load_model(ROOT / "models" / "best_model.pt", device)
    result = generate(model, checkpoint["vocab"], args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed)
    print(result["full_text"])


if __name__ == "__main__":
    main()
