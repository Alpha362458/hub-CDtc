"""锁定验证集选择结果后，只对最终模型做正式测试和额外难度诊断。"""
from collections import Counter
from math import sqrt

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import ROOT, DATA, now, read_csv, read_json, write_json, write_csv, sha256, load_checkpoint
from .data import encode, normalize
from .model import from_checkpoint


@torch.inference_mode()
def predict_probabilities(model, rows, vocab, max_len=40, batch_size=256):
    model.eval()
    if any(not normalize(r["text"]) for r in rows):
        raise ValueError("归一化后为空的文本没有可识别的语种")
    result = []
    for start in range(0, len(rows), batch_size):
        x = torch.tensor([encode(r["text"], vocab, max_len) for r in rows[start:start + batch_size]], dtype=torch.long)
        result.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(result)


def metrics_from_probs(rows, probabilities, labels, active_only=False):
    label2id = {label: i for i, label in enumerate(labels)}
    gold = np.array([label2id[r["label"]] for r in rows])
    pred = probabilities.argmax(1)
    indices = sorted(set(gold)) if active_only else list(range(len(labels)))
    report = classification_report(gold, pred, labels=list(range(len(labels))), target_names=labels,
                                   output_dict=True, zero_division=0)
    n, hit = len(rows), int((gold == pred).sum())
    p, z = hit / n, 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    metrics = {"count": n, "correct": hit, "errors": n - hit, "accuracy": accuracy_score(gold, pred),
               "macro_f1": f1_score(gold, pred, labels=indices, average="macro", zero_division=0),
               "macro_f1_labels": [labels[i] for i in indices],
               "weighted_f1": f1_score(gold, pred, average="weighted", zero_division=0),
               "cross_entropy": float(-np.log(np.maximum(probabilities[np.arange(n), gold], 1e-30)).mean()),
               "accuracy_wilson_95": [center - half, center + half]}
    return metrics, report, confusion_matrix(gold, pred, labels=range(len(labels))).tolist()


def prediction_rows(rows, probabilities, labels):
    out = []
    for r, prob in zip(rows, probabilities):
        i = int(prob.argmax())
        out.append({"text": r["text"], "true_label": r["label"], "predicted_label": labels[i],
                    "confidence": float(prob[i]), "normalized_length": len(normalize(r["text"])),
                    "correct": labels[i] == r["label"], **({"slice": r["slice"]} if "slice" in r else {})})
    return out


def script_rule(text):
    # 与课堂 demo 同思路：Unicode 区段计票，五个拉丁字母语种只能统一猜 en。
    ranges = [(0x4e00, 0x9fff, "zh"), (0x3040, 0x30ff, "ja"), (0xac00, 0xd7af, "ko"),
              (0x0400, 0x04ff, "ru"), (0x0600, 0x06ff, "ar"), (0x0900, 0x097f, "hi"),
              (0x0e00, 0x0e7f, "th"), (ord("a"), ord("z"), "en")]
    votes = Counter()
    for c in normalize(text):
        for lo, hi, label in ranges:
            if lo <= ord(c) <= hi:
                votes[label] += 1
                break
    return votes.most_common(1)[0][0] if votes else "unknown"


def evaluate_final():
    torch.set_num_threads(1)
    selection = read_json(ROOT / "outputs" / "selection.json")
    path = ROOT / "models" / "best_model.pt"
    assert sha256(path) == selection["final_checkpoint_sha256"], "模型与锁定的选择记录不一致"
    checkpoint = load_checkpoint(path)
    model = from_checkpoint(checkpoint)
    vocab, labels, max_len = checkpoint["vocab"], checkpoint["labels"], checkpoint["max_len"]
    outdir = ROOT / "outputs" / "evaluation"
    outdir.mkdir(parents=True, exist_ok=True)
    test = read_csv(DATA / "v12" / "test.csv")
    probs = predict_probabilities(model, test, vocab, max_len)
    final, report, cm = metrics_from_probs(test, probs, labels)
    final.update({"evaluated_at": now(), "experiment": selection["name"], "best_epoch": checkpoint["best_epoch"],
                  "model_sha256": sha256(path), "test_sha256": sha256(DATA / "v12" / "test.csv"),
                  "closed_set": True, "threshold_applied": False})
    write_json(outdir / "final_metrics.json", final)
    write_json(outdir / "classification_report.json", report)
    write_json(outdir / "confusion_matrix.json", {"labels": labels, "rows": "true", "columns": "predicted", "matrix": cm})
    write_csv(outdir / "classification_report.csv", [{"label": lab, **report[lab]} for lab in labels])
    predictions = prediction_rows(test, probs, labels)
    write_csv(outdir / "test_predictions.csv", predictions)
    errors = sorted((r for r in predictions if not r["correct"]), key=lambda r: -r["confidence"])
    write_csv(outdir / "errors.csv", errors, list(predictions[0]))
    np.save(outdir / "test_probabilities.npy", probs)
    pairs = sorted(({'true': labels[g], 'predicted': labels[p], 'count': cm[g][p]}
                    for g in range(len(labels)) for p in range(len(labels)) if g != p and cm[g][p]),
                   key=lambda r: -r["count"])
    length_metrics = []
    # 明确包含 7；不沿用课堂评测中 [2,7) 遗漏七字符样本的边界。
    for lo, hi in ((8, 11), (12, 19), (20, 29), (30, 40)):
        sub = [r for r in predictions if lo <= r["normalized_length"] <= hi]
        length_metrics.append({"length_range": f"{lo}-{hi}", "count": len(sub),
                               "accuracy": sum(r["correct"] for r in sub) / len(sub) if sub else None})
    write_json(outdir / "error_analysis.json", {"top_confusions": pairs, "by_length": length_metrics,
                                               "highest_confidence_errors": errors[:12]})
    rule_preds = [script_rule(r["text"]) for r in test]
    write_json(outdir / "rule_baseline.json", {
        "description": "Unicode区段多数票；拉丁字母统一判en", "count": len(test),
        "accuracy": accuracy_score([r["label"] for r in test], rule_preds),
        "macro_f1": f1_score([r["label"] for r in test], rule_preds, labels=labels, average="macro", zero_division=0)})
    hard = read_csv(DATA / "test_hard.csv")
    hard_probs = predict_probabilities(model, hard, vocab, max_len)
    hard_summary = {}
    for name in ("short", "nodeaccent", "nokana"):
        idx = [i for i, r in enumerate(hard) if r["slice"] == name]
        rows = [hard[i] for i in idx]
        metrics, per_class, matrix = metrics_from_probs(rows, hard_probs[idx], labels, active_only=True)
        hard_summary[name] = {**metrics, "per_class": per_class, "confusion_matrix": matrix}
    write_json(outdir / "hard_metrics.json", hard_summary)
    write_csv(outdir / "hard_predictions.csv", prediction_rows(hard, hard_probs, labels))
    # 阈值只从已知语种验证集获取，未用开放集或常规测试集调阈值。
    dev = read_csv(DATA / "v12" / "dev.csv")
    dev_probs = predict_probabilities(model, dev, vocab, max_len)
    dev_conf = dev_probs.max(1)
    threshold = float(np.quantile(dev_conf, 0.05))
    openset = read_csv(DATA / "test_openset.csv")
    open_probs = predict_probabilities(model, openset, vocab, max_len)
    open_conf = open_probs.max(1)
    open_details = {}
    for lab in sorted({r["label"] for r in openset}):
        idx = [i for i, r in enumerate(openset) if r["label"] == lab]
        accepted = [labels[int(open_probs[i].argmax())] for i in idx if open_conf[i] >= threshold]
        open_details[lab] = {"count": len(idx), "rejection_rate": float((open_conf[idx] < threshold).mean()),
                             "mean_confidence": float(open_conf[idx].mean()), "accepted_predictions": dict(Counter(accepted))}
    openset_metrics = {"threshold": threshold, "threshold_source": "v12/dev max-softmax 5th percentile",
                       "model_sha256": sha256(path), "count": len(openset),
                       "dev_known_retention": float((dev_conf >= threshold).mean()),
                       "test_known_retention": float((probs.max(1) >= threshold).mean()),
                       "unknown_rejection_rate": float((open_conf < threshold).mean()), "by_language": open_details}
    write_json(outdir / "openset_metrics.json", openset_metrics)
    write_csv(outdir / "openset_predictions.csv", [
        {"text": r["text"], "true_label": r["label"], "closed_set_prediction": labels[int(p.argmax())],
         "confidence": float(p.max()), "rejected": float(p.max()) < threshold} for r, p in zip(openset, open_probs)])
    np.savez(outdir / "confidence_distributions.npz", dev=dev_conf, known_test=probs.max(1), unknown=open_conf)
    print(f"Final test: {final['correct']}/{final['count']} accuracy={final['accuracy']:.6f} macro-F1={final['macro_f1']:.6f}", flush=True)
    print(f"Hard accuracy: { {k: round(v['accuracy'], 4) for k,v in hard_summary.items()} }", flush=True)
    print(f"Open-set rejection={openset_metrics['unknown_rejection_rate']:.4f} at dev threshold={threshold:.4f}", flush=True)
    return final


if __name__ == "__main__":
    evaluate_final()
