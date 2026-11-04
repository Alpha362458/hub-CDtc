"""沿用课程划分；仅用 train 构建字符表，保留组合字符 M。"""
import unicodedata
from collections import Counter

import torch
from torch.utils.data import TensorDataset

from .common import DATA, ROOT, LANG_NAMES, read_csv, write_json, sha256


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in text if unicodedata.category(c)[0] in ("L", "M"))


def build_char_vocab(sentences):
    counts = Counter(c for s in sentences for c in normalize(s))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        vocab[ch] = len(vocab)
    return vocab, counts


def encode(text, vocab, max_len=40):
    ids = [vocab.get(c, 1) for c in normalize(text)[:max_len]]
    return ids + [0] * (max_len - len(ids))


def make_dataset(rows, vocab, labels, max_len=40):
    label2id = {label: i for i, label in enumerate(labels)}
    if any(not normalize(r["text"]) for r in rows):
        raise ValueError("数据中包含归一化后为空的文本")
    return TensorDataset(
        torch.tensor([encode(r["text"], vocab, max_len) for r in rows], dtype=torch.long),
        torch.tensor([label2id[r["label"]] for r in rows], dtype=torch.long),
    )


def prepare_data():
    splits = {s: read_csv(DATA / "v12" / f"{s}.csv") for s in ("train", "dev", "test")}
    labels = sorted({r["label"] for r in splits["train"]})
    assert len(labels) == 12 and set(labels) == set(LANG_NAMES), "必须包含十二类语种"
    texts, audit = {}, {"policy": "保留课程原始划分；字符表仅由 train 生成；不使用测试集选择模型", "splits": {}}
    for split, rows in splits.items():
        normalized = [normalize(r["text"]) for r in rows]
        texts[split] = set(normalized)
        counts = Counter(r["label"] for r in rows)
        assert set(counts) == set(labels)
        assert all(8 <= len(t) <= 40 for t in normalized)
        assert len(texts[split]) == len(rows), f"{split} 包含归一化重复文本"
        audit["splits"][split] = {
            "count": len(rows), "class_counts": dict(sorted(counts.items())),
            "duplicates_after_normalization": len(rows) - len(texts[split]),
            "length_min": min(map(len, normalized)), "length_max": max(map(len, normalized)),
            "length_mean": sum(map(len, normalized)) / len(rows),
            "sha256": sha256(DATA / "v12" / f"{split}.csv"),
        }
    audit["cross_split_overlap"] = {}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        n = len(texts[a] & texts[b])
        audit["cross_split_overlap"][f"{a}__{b}"] = n
        assert n == 0, f"{a}/{b} 存在数据泄漏"
    vocab, counts = build_char_vocab(r["text"] for r in splits["train"])
    path = DATA / "vocab" / "vocab_v12.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Built only from v12/train.csv; char\tid\tfrequency\n")
        for ch, idx in vocab.items():
            f.write(f"{ch}\t{idx}\t{counts.get(ch, 0)}\n")
    coverage = {}
    for split, rows in splits.items():
        cs = Counter(c for r in rows for c in normalize(r["text"]))
        unknown = sum(n for c, n in cs.items() if c not in vocab)
        coverage[split] = {"characters": sum(cs.values()), "unknown_characters": unknown,
                           "unknown_rate": unknown / sum(cs.values())}
    audit["vocabulary"] = {"size_including_specials": len(vocab), "observed_characters": len(counts),
                            "pad_id": 0, "unk_id": 1, "coverage": coverage,
                            "sha256": sha256(path)}
    audit["diagnostic_sets"] = {}
    for name in ("test_hard", "test_openset"):
        rows = read_csv(DATA / f"{name}.csv")
        audit["diagnostic_sets"][name] = {
            "count": len(rows), "sha256": sha256(DATA / f"{name}.csv"),
            "class_counts": dict(sorted(Counter(r["label"] for r in rows).items())),
            "normalized_overlap": {s: sum(normalize(r["text"]) in ts for r in rows) for s, ts in texts.items()},
            "duplicates_after_normalization": len(rows) - len({normalize(r["text"]) for r in rows}),
        }
    write_json(ROOT / "models" / "vocab.json", vocab)
    write_json(ROOT / "models" / "labels.json", labels)
    write_json(ROOT / "outputs" / "data_audit.json", audit)
    print(f"Data ready: train={len(splits['train'])}, dev={len(splits['dev'])}, test={len(splits['test'])}; vocab={len(vocab)}", flush=True)
    return splits, vocab, labels


if __name__ == "__main__":
    prepare_data()
