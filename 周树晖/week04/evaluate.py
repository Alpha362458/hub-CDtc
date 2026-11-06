"""验证集锁定权重后评测；统计基线使用相同字表、划分及目标字符。"""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, read_json, write_json, sha256
from generate import generate
from model import load_model
from prepare_data import TokenBlocks
from train import score


def ngram_baselines(arrays, vocab_size, alpha=0.1):
    train = arrays["train"].astype(np.int64)
    uni_counts = np.bincount(train[1:], minlength=vocab_size)
    bigram_counts = np.bincount(train[:-1] * vocab_size + train[1:], minlength=vocab_size*vocab_size).reshape(vocab_size, vocab_size)
    row_counts = bigram_counts.sum(1)
    uni_denominator = len(train) - 1 + alpha * (vocab_size - 1)
    uni_top = int(uni_counts.argmax())
    bi_top = bigram_counts.argmax(1)
    results = {}
    for name in ("val", "test"):
        x, y = arrays[name][:-1].astype(np.int64), arrays[name][1:].astype(np.int64)
        uni_nll = float(-np.log((uni_counts[y] + alpha) / uni_denominator).mean())
        bi_nll = float(-np.log((bigram_counts[x, y] + alpha) / (row_counts[x] + alpha * (vocab_size - 1))).mean())
        results[name] = {
            "uniform": {"nll": math.log(vocab_size - 1), "perplexity": vocab_size - 1},
            "unigram": {"nll": uni_nll, "perplexity": math.exp(uni_nll), "accuracy": float((y == uni_top).mean())},
            "bigram": {"nll": bi_nll, "perplexity": math.exp(bi_nll), "accuracy": float((y == bi_top[x]).mean())},
        }
    return {"smoothing_alpha": alpha, "vocab_size": vocab_size, "pad_excluded": True, "splits": results}


def evaluate():
    summary = read_json(ROOT / "results" / "training.json")
    path = ROOT / "models" / "best_model.pt"
    assert sha256(path) == summary["checkpoint_sha256"]
    config = summary["config"]
    device = torch.device(summary["environment"]["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4 if device.type == "cuda" else 2)
    model, checkpoint = load_model(path, device)
    assert checkpoint["tokens_sha256"] == sha256(ROOT / "data" / "tokens.npz")
    arrays = np.load(ROOT / "data" / "tokens.npz")
    metrics = {}
    for split in ("train", "val", "test"):
        loader = DataLoader(TokenBlocks(arrays[split], config["block_size"]), batch_size=config["batch_size"])
        metrics[split] = score(model, loader, device)
        assert metrics[split]["tokens"] == len(arrays[split]) - 1
        print(f"{split}: PPL={metrics[split]['perplexity']:.4f} NLL={metrics[split]['nll']:.4f} Accuracy={metrics[split]['accuracy']:.2%}", flush=True)
    baselines = ngram_baselines(arrays, config["vocab_size"])
    write_json(ROOT / "results" / "metrics.json", {
        "checkpoint_sha256": sha256(path), "best_epoch": checkpoint["epoch"], "evaluation_device": str(device),
        "target_policy": "all characters including newline and UNK; exclude padding; each target counted once",
        "transformer": metrics, "baselines": baselines,
    })
    prompts = ["今天天气", "我想学习", "人工智能", "请问", "我们可以", "中国的"]
    samples = [generate(model, checkpoint["vocab"], p, max_new_tokens=60, temperature=0.8, top_k=40, seed=42) for p in prompts]
    samples.append(generate(model, checkpoint["vocab"], "我想学习", max_new_tokens=60, temperature=0, top_k=0, seed=42))
    write_json(ROOT / "results" / "generations.json", {"checkpoint_sha256": sha256(path), "samples": samples})
    print(f"Bigram test PPL={baselines['splits']['test']['bigram']['perplexity']:.4f}", flush=True)
    for s in samples:
        print(f"[{s['temperature']}] {s['full_text']}", flush=True)


if __name__ == "__main__":
    evaluate()
