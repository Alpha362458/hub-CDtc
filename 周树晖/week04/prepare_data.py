"""先以完整句子去重和划分，再在各划分内拼接、编码和切块。"""
import random
import re
import unicodedata
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from common import ROOT, read_json, write_json, sha256


def clean_line(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def split_lines(raw_text, config):
    raw_lines = raw_text.splitlines()
    cleaned = [clean_line(line) for line in raw_lines]
    nonempty = [line for line in cleaned if line]
    unique = list(dict.fromkeys(nonempty))
    random.Random(config["seed"]).shuffle(unique)
    a = int(len(unique) * config["train_ratio"])
    b = int(len(unique) * (config["train_ratio"] + config["val_ratio"]))
    return {"train": unique[:a], "val": unique[a:b], "test": unique[b:]}, {
        "raw_lines": len(raw_lines), "empty_lines_removed": len(cleaned) - len(nonempty),
        "duplicates_removed": len(nonempty) - len(unique), "unique_lines": len(unique),
    }


def prepare(config=None):
    config = config or read_json(ROOT / "config.json")
    source = ROOT / "data" / "corpus.txt"
    text = source.read_text(encoding="utf-8-sig")
    lines, stats = split_lines(text, config)
    assert not (set(lines["train"]) & set(lines["val"]) or set(lines["train"]) & set(lines["test"]) or set(lines["val"]) & set(lines["test"]))
    streams = {name: "\n" + "\n".join(rows) + "\n" for name, rows in lines.items()}
    freq = Counter(streams["train"])
    characters = [c for c, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:config["vocab_size"] - 2]]
    itos = ["<PAD>", "<UNK>"] + characters
    stoi = {c: i for i, c in enumerate(itos)}
    arrays = {name: np.fromiter((stoi.get(c, 1) for c in stream), dtype=np.uint16, count=len(stream)) for name, stream in streams.items()}
    np.savez_compressed(ROOT / "data" / "tokens.npz", **arrays)
    write_json(ROOT / "data" / "vocab.json", {"itos": itos, "stoi": stoi})
    stats.update({"source": "第四周课件/循环神经网络语言模型/corpus_general.txt", "source_sha256": sha256(source),
                  "raw_characters": len(text), "normalization": "NFKC; collapse within-line whitespace; strip; drop empty; exact dedup",
                  "seed": config["seed"], "vocab_size": len(itos), "observed_train_characters": len(freq),
                  "split_overlap": 0, "splits": {}, "vocab_sha256": sha256(ROOT / "data" / "vocab.json"),
                  "tokens_sha256": sha256(ROOT / "data" / "tokens.npz")})
    for name in arrays:
        stats["splits"][name] = {"sentences": len(lines[name]), "characters": len(streams[name]),
                                  "predicted_tokens": len(arrays[name]) - 1,
                                  "unknown_tokens": int((arrays[name] == 1).sum()),
                                  "unknown_rate": float((arrays[name] == 1).mean()),
                                  "blocks": (len(arrays[name]) - 2) // config["block_size"] + 1}
    write_json(ROOT / "results" / "data_stats.json", stats)
    print(f"Data: {len(lines['train']):,}/{len(lines['val']):,}/{len(lines['test']):,} sentences, vocab={len(itos)}", flush=True)
    return arrays, {"itos": itos, "stoi": stoi}, stats


class TokenBlocks(Dataset):
    """每个目标字符恰好计分一次；末块补齐，使用 -100 忽略无效目标。"""
    def __init__(self, tokens, block_size=128):
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 2) // self.block_size + 1

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.block_size
        n = min(self.block_size, len(self.tokens) - start - 1)
        x = torch.zeros(self.block_size, dtype=torch.long)
        y = torch.full((self.block_size,), -100, dtype=torch.long)
        x[:n] = self.tokens[start:start + n]
        y[:n] = self.tokens[start + 1:start + n + 1]
        return x, y


if __name__ == "__main__":
    prepare()
