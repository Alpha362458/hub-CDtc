"""加载 train.py 生成的模型，识别输入文本的语言。"""
from pathlib import Path
import sys

import torch
from train import LanguageTransformer, encode

MODEL_PATH = Path(__file__).resolve().parent / "language_model.pth"


def main():
    if not MODEL_PATH.exists():
        print("找不到模型，请先运行 python train.py")
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    vocab = checkpoint["vocab"]
    languages = checkpoint["languages"]

    model = LanguageTransformer(len(vocab) + 2, len(languages))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 命令行输入文本；没有参数时运行 3 个演示例子。
    texts = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "Hello world", "你好世界", "こんにちは世界"
    ]
    for text in texts:
        inputs = torch.tensor([encode(text, vocab)])
        with torch.no_grad():
            probabilities = torch.softmax(model(inputs), dim=1)[0]
        index = probabilities.argmax().item()
        print(f"文本: {text} | 预测语言: {languages[index]} | 分数: {probabilities[index]:.1%}")


if __name__ == "__main__":
    main()
