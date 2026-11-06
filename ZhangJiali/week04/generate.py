"""
用训练好的 Transformer 语言模型做下一字符生成演示。
用法:
    python generate.py --prompt "今天天气" --max_new 50 --device cuda
"""

import argparse
import torch
import torch.nn.functional as F

from language_model import TransformerLM, resolve_device


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = TransformerLM(
        vocab_size=ckpt["vocab_size"],
        d_model=args.get("d_model", 128),
        nhead=args.get("nhead", 4),
        num_layers=args.get("num_layers", 2),
        dim_feedforward=args.get("dim_feedforward", 256),
        dropout=0.0,
        max_len=max(args.get("seq_len", 64), 512),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["char2idx"], ckpt["idx2char"], ckpt


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new=50, temperature=1.0, top_k=20, device="cpu"):
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        # 提示词全不在词表时，从高频字附近随机起头
        ids = [0]

    for _ in range(max_new):
        x = torch.tensor([ids[-model.max_len :]], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(temperature, 1e-6)
        if top_k and top_k < logits.size(0):
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_id = indices[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)

    return "".join(idx2char[i] for i in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_model.pt")
    parser.add_argument("--prompt", default="今天天气")
    parser.add_argument("--max_new", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto / cuda / cuda:0 / cpu")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, char2idx, idx2char, ckpt = load_model(args.model_path, device)
    print(f"device: {device}")
    print(f"loaded: {args.model_path}  best_val_ppl={ckpt.get('best_val_ppl', 'n/a')}")
    text = generate(
        model,
        char2idx,
        idx2char,
        args.prompt,
        max_new=args.max_new,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
