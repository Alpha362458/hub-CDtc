"""python predict.py --text "这是中文。" "This is English." [--reject-unknown]"""
import argparse
import json
from pathlib import Path

import torch

from src.common import ROOT, load_checkpoint, read_json, sha256
from src.data import normalize
from src.evaluate import predict_probabilities
from src.model import from_checkpoint


def main():
    parser = argparse.ArgumentParser(description="十二类语种分类；输出前三类 softmax 分数")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.pt")
    parser.add_argument("--text", nargs="+", default=[])
    parser.add_argument("--input-file", type=Path, help="UTF-8 文本，每行一句")
    parser.add_argument("--reject-unknown", action="store_true", help="使用验证集标定的阈值；能力有限，详见 README")
    args = parser.parse_args()
    texts = args.text + (args.input_file.read_text(encoding="utf-8-sig").splitlines() if args.input_file else [])
    if not texts:
        parser.error("请通过 --text 或 --input-file 提供文本")
    if any(not normalize(t) for t in texts):
        parser.error("输入包含空白或只有数字、标点的文本，归一化后为空")
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.model)
    probs = predict_probabilities(from_checkpoint(checkpoint), [{"text": t} for t in texts],
                                 checkpoint["vocab"], checkpoint["max_len"])
    labels, threshold = checkpoint["labels"], None
    if args.reject_unknown:
        calibration = read_json(ROOT / "outputs" / "evaluation" / "openset_metrics.json")
        if calibration["model_sha256"] != sha256(args.model):
            parser.error("当前模型不匹配已有阈值，请先对该模型重新标定")
        threshold = calibration["threshold"]
    for text, p in zip(texts, probs):
        top = p.argsort()[::-1][:3]
        pred = labels[int(top[0])]
        if threshold is not None and float(p[top[0]]) < threshold:
            pred = "unknown"
        print(json.dumps({"text": text, "predicted_label": pred, "normalized_length": len(normalize(text)),
                          "truncated": len(normalize(text)) > checkpoint["max_len"],
                          "top3": [{"label": labels[int(i)], "softmax": float(p[i])} for i in top]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
