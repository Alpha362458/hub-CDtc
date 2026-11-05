"""加载 TransformerEncoder 检查点，进行单次或交互式文本续写。

在 week04 目录运行：
    python test_model.py
    python test_model.py --prompt "今天" --length 100
"""

import argparse
import math

import torch

from inference import DEFAULT_CHECKPOINT, generate_text, load_checkpoint, select_device


def preview_model(model, char2idx, idx2char, args, interactive=False):
    def show(prompt):
        try:
            result = generate_text(
                model, char2idx, idx2char, prompt,
                args.length, args.temperature, args.top_k,
            )
            print(f"续写结果：\n{result}\n")
        except ValueError as error:
            if not interactive:
                raise
            print(error)

    if args.prompt is not None:
        show(args.prompt)
    if interactive:
        print("输入文字查看续写；输入 /exit 退出。这是文本续写，不是问答。")
        while True:
            try:
                prompt = input("请输入文字：")
            except (EOFError, KeyboardInterrupt):
                print("\n已退出测试。")
                break
            if prompt.strip() == "/exit":
                break
            show(prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="默认加载脚本同目录的 transformer_encoder_model.pt")
    parser.add_argument("--prompt", default=None, help="指定文字进行单次续写；不传则交互输入")
    parser.add_argument("--interactive", action="store_true", help="输出 prompt 的续写后继续交互输入")
    parser.add_argument("--length", type=int, default=100, help="续写的新字符数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度；0 表示每次选最高分字符")
    parser.add_argument("--top_k", type=int, default=20, help="从分数最高的 k 个字符中采样；0 表示完整词表")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    if args.length < 0 or args.top_k < 0 or args.temperature < 0 or not math.isfinite(args.temperature):
        parser.error("length、temperature、top_k 必须非负，temperature 必须为有限数值。")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}")
    try:
        model, char2idx, idx2char = load_checkpoint(args.checkpoint, device)
        preview_model(model, char2idx, idx2char, args, args.interactive or args.prompt is None)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
