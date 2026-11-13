# -*- coding: utf-8 -*-
"""
SFT 交互问答 —— 加载 best_sft.pt，命令行里一问一答

前置：
    先跑通 sft_finetune.py，得到 best_sft.pt

用法（在 sft_train 目录下）:
    python sft_chat.py
    python sft_chat.py --ckpt best_sft.pt --temperature 0.7 --max_new 100

命令：
    直接输入问题回车 → 模型回答
    q / quit / 退出 → 结束
    clear / 清空     → 清屏提示（本脚本不做多轮上下文，每次独立问答）
"""

import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LM_DIR = HERE.parent / "循环神经网络语言模型"
sys.path.insert(0, str(LM_DIR))

from language_model_transformer import TransformerLM  # noqa: E402

INSTR_PREFIX = "指令："
REPLY_PREFIX = "回答："


def load_sft_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到权重: {ckpt_path}\n请先运行: python sft_finetune.py"
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    char2idx = ck["char2idx"]
    idx2char = {int(k): v for k, v in ck["idx2char"].items()}
    cfg = ck.get("lm_args") or ck.get("args")
    # lm_args 里是预训练结构；若没有则从顶层 args 猜常见默认
    d_model = cfg.get("d_model", 256)
    nhead = cfg.get("nhead", 8)
    num_layers = cfg.get("num_layers", 4)
    dim_ff = cfg.get("dim_ff", 1024)
    dropout = cfg.get("dropout", 0.2)

    model = TransformerLM(
        vocab_size=len(char2idx),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        max_len=512,
    )
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()
    return model, char2idx, idx2char


def reply(model, char2idx, idx2char, instruction, max_new=100, temperature=0.7, device="cpu"):
    """拼「指令/回答：」前缀，逐字生成，遇到换行结束。"""
    prompt = INSTR_PREFIX + instruction.strip() + "\n" + REPLY_PREFIX
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return "（问题里的字几乎都不在词表里，换个中文问题试试）"

    temperature = max(float(temperature), 1e-6)
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-128:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1] / temperature
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            ch = idx2char[next_id]
            ids.append(next_id)
            if ch == "\n":
                break

    text = "".join(idx2char[i] for i in ids)
    if REPLY_PREFIX in text:
        text = text.split(REPLY_PREFIX, 1)[1]
    return text.strip() or "（模型没有生成有效回答）"


def main():
    ap = argparse.ArgumentParser(description="SFT 模型交互问答")
    ap.add_argument("--ckpt", default=str(HERE / "best_sft.pt"))
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="越小越稳、越像背答案；越大越随机")
    ap.add_argument("--max_new", type=int, default=100, help="回答最多生成多少字")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, char2idx, idx2char = load_sft_model(args.ckpt, device)

    print("=" * 52)
    print("  SFT 交互问答（字符级 Transformer）")
    print(f"  权重: {os.path.basename(args.ckpt)}")
    print(f"  device={device}  temperature={args.temperature}  max_new={args.max_new}")
    print("  输入问题后回车；q 退出")
    print("=" * 52)

    while True:
        try:
            raw = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit", "退出"):
            print("退出。")
            break
        if raw.lower() in ("clear", "cls", "清空"):
            print("\n" * 3)
            continue

        ans = reply(
            model, char2idx, idx2char, raw,
            max_new=args.max_new,
            temperature=args.temperature,
            device=device,
        )
        print(f"模型> {ans}")


if __name__ == "__main__":
    main()
