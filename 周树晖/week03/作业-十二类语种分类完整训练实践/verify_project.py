"""独立检查可交付权重、数据边界以及结果是否能从 checkpoint 重现。"""
import ast
import math
import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.common import ROOT, DATA, now, read_csv, read_json, write_json, sha256, load_checkpoint
from src.data import build_char_vocab, encode, normalize
from src.evaluate import predict_probabilities
from src.model import from_checkpoint


def verify():
    torch.set_num_threads(1)
    checks = []

    def check(name, passed):
        if not passed:
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})
        print(f"PASS {name}")

    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    check("all Python source files parse", True)
    selection = read_json(ROOT / "outputs" / "selection.json")
    summaries = read_json(ROOT / "outputs" / "experiment_summary.json")
    check("selection uses dev only", selection["test_used_for_selection"] is False)
    winner = max(summaries, key=lambda s: (s["dev_macro_f1"], -s["dev_loss"]))
    check("selected model matches best dev result", winner["name"] == selection["name"])
    for summary in summaries:
        ck = load_checkpoint(ROOT / summary["checkpoint"])
        from_checkpoint(ck)
        check(f"{summary['name']} checkpoint hash and epoch", sha256(ROOT / summary["checkpoint"]) == summary["checkpoint_sha256"]
              and ck["best_epoch"] == summary["best_epoch"])
        hist = read_json(ROOT / "outputs" / "experiments" / summary["name"] / "history.json")
        best = max(hist, key=lambda r: (r["dev_macro_f1"], -r["dev_loss"]))
        check(f"{summary['name']} checkpoint is best historical epoch", best["epoch"] == ck["best_epoch"])
    model_path = ROOT / "models" / "best_model.pt"
    checkpoint = load_checkpoint(model_path)
    model = from_checkpoint(checkpoint)
    labels, vocab = checkpoint["labels"], checkpoint["vocab"]
    check("final checkpoint matches locked selection", sha256(model_path) == selection["final_checkpoint_sha256"])
    train = read_csv(DATA / "v12" / "train.csv")
    rebuilt, _ = build_char_vocab(r["text"] for r in train)
    check("vocab rebuilt only from train matches checkpoint", rebuilt == vocab == read_json(ROOT / "models" / "vocab.json"))
    check("12 outputs and correct special tokens", len(labels) == 12 and model.fc.out_features == 12 and vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1)
    check("NFKC and combining marks preserved", normalize("ＡＢＣ 123! कि") == "abcकि")
    check("unknown characters use UNK", encode("\U00010400", vocab, 40)[0] == 1)
    check("padding embedding stays zero", bool(model.embedding.weight[0].eq(0).all()))
    with torch.inference_mode():
        short_ids = torch.tensor([[vocab.get(c, 1) for c in normalize("This is a model.")]])
        padded_ids = torch.cat([short_ids, torch.zeros((1, 40 - short_ids.shape[1]), dtype=torch.long)], dim=1)
        check("right padding cannot change valid pooled logits", torch.allclose(model(short_ids), model(padded_ids), atol=1e-6))
    empty_rejected = False
    try:
        predict_probabilities(model, [{"text": "123 !?"}], vocab)
    except ValueError:
        empty_rejected = True
    check("empty normalized input explicitly rejected", empty_rejected)
    for lab in ("ko", "th", "vi", "tr"):
        text = next(r["text"] for r in train if r["label"] == lab)
        ids = encode(text, vocab)
        check(f"added language {lab} has known characters", all(i != 1 for i in ids) and any(i > 1 for i in ids))
    audit = read_json(ROOT / "outputs" / "data_audit.json")
    normalized_sets = {}
    for split in ("train", "dev", "test"):
        rows = read_csv(DATA / "v12" / f"{split}.csv")
        normalized_sets[split] = {normalize(r["text"]) for r in rows}
        check(f"{split} data hash and no duplicates", sha256(DATA / "v12" / f"{split}.csv") == audit["splits"][split]["sha256"]
              and len(rows) == len(normalized_sets[split]))
    check("all formal splits are disjoint", not (normalized_sets["train"] & normalized_sets["dev"] or normalized_sets["train"] & normalized_sets["test"] or normalized_sets["dev"] & normalized_sets["test"]))
    test = read_csv(DATA / "v12" / "test.csv")
    probabilities = predict_probabilities(model, test, vocab, checkpoint["max_len"])
    out = ROOT / "outputs" / "evaluation"
    stored = np.load(out / "test_probabilities.npy")
    check("all 1800 probability vectors reproduce", probabilities.shape == (len(test), 12) and np.allclose(probabilities, stored, atol=1e-7))
    check("probabilities sum to one and are finite", bool(np.isfinite(probabilities).all() and np.allclose(probabilities.sum(1), 1, atol=1e-6)))
    gold = np.array([labels.index(r["label"]) for r in test])
    pred = probabilities.argmax(1)
    final = read_json(out / "final_metrics.json")
    check("accuracy recomputed from model", math.isclose(accuracy_score(gold, pred), final["accuracy"], abs_tol=1e-12))
    check("macro-F1 recomputed from model", math.isclose(f1_score(gold, pred, average="macro"), final["macro_f1"], abs_tol=1e-12))
    cm = read_json(out / "confusion_matrix.json")
    check("confusion counts match all predictions", np.array_equal(confusion_matrix(gold, pred, labels=range(12)), cm["matrix"]))
    check("error export matches prediction errors", len(read_csv(out / "errors.csv")) == final["errors"] == int((gold != pred).sum()))
    csv_predictions = read_csv(out / "test_predictions.csv")
    check("CSV export matches every model prediction", len(csv_predictions) == len(test) and
          all(r["predicted_label"] == labels[int(p)] for r, p in zip(csv_predictions, pred)))
    hard = read_json(out / "hard_metrics.json")
    check("all hard samples accounted for", sum(r["count"] for r in hard.values()) == len(read_csv(DATA / "test_hard.csv")))
    openset = read_json(out / "openset_metrics.json")
    dev_probs = predict_probabilities(model, read_csv(DATA / "v12" / "dev.csv"), vocab, checkpoint["max_len"])
    check("open-set threshold derived only from dev", math.isclose(float(np.quantile(dev_probs.max(1), .05)), openset["threshold"], abs_tol=1e-7))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme)
    local = [t for t in targets if not t.startswith(("https://", "http://", "#"))]
    check("all README local links exist", all((ROOT / t).exists() for t in local))
    check("README matches final metrics", f"{final['accuracy']:.2%}" in readme and f"{final['macro_f1']:.4f}" in readme)
    check("eight nonempty chart files delivered", len(list((ROOT / "images").glob("*.png"))) == 8 and
          all(p.stat().st_size > 10_000 for p in (ROOT / "images").glob("*.png")))
    write_json(ROOT / "outputs" / "verification.json", {"verified_at": now(), "checks": checks, "all_passed": True,
                                                        "model_sha256": sha256(model_path), "sample_count": len(test)})
    manifest = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.name != "artifact_manifest.json":
            manifest.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_json(ROOT / "outputs" / "artifact_manifest.json", {"created_at": now(), "files": manifest})
    print(f"Verified {len(checks)} checks; {len(manifest)} files recorded in manifest.")


if __name__ == "__main__":
    verify()
