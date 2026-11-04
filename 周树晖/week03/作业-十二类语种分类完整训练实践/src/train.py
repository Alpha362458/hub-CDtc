import argparse
import copy
import platform
import sys
import time

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .common import ROOT, now, read_json, write_json, sha256, set_seed
from .data import prepare_data, make_dataset
from .model import LangIDClassifier


@torch.inference_mode()
def evaluate_dev(model, loader, device):
    model.eval()
    total_loss, gold, pred = 0.0, [], []
    for x, y in loader:
        logits = model(x.to(device))
        total_loss += torch.nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        gold.extend(y.tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    return {"loss": total_loss / len(gold), "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro", zero_division=0)}


def train_one(config, plan, train_ds, dev_ds, vocab, labels):
    set_seed(plan["seed"], plan["threads"])
    device = torch.device(plan["device"])
    model_config = {k: config[k] for k in ("embed_dim", "hidden_dim", "dropout", "encoder", "layer_norm")}
    model = LangIDClassifier(len(vocab), len(labels), **model_config).to(device)
    loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(plan["seed"]), num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=256, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    history, best, stale = [], None, 0
    started_at, t0 = now(), time.perf_counter()
    exp_dir = ROOT / "outputs" / "experiments" / config["name"]
    checkpoint_path = ROOT / "models" / f"{config['name']}.pt"
    print(f"\n[{config['name']}] {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        used_lr = optimizer.param_groups[0]["lr"]
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += logits.detach().argmax(1).eq(y).sum().item()
            seen += len(y)
        dev = evaluate_dev(model, dev_loader, device)
        entry = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
                 "dev_loss": dev["loss"], "dev_accuracy": dev["accuracy"], "dev_macro_f1": dev["macro_f1"],
                 "lr": used_lr, "seconds": time.perf_counter() - epoch_start}
        history.append(entry)
        # 主键 Macro-F1，平分时以更低的验证损失决胜；绝不读取测试指标。
        improved = (best is None or dev["macro_f1"] > best["dev_macro_f1"] + 1e-12 or
                    (abs(dev["macro_f1"] - best["dev_macro_f1"]) <= 1e-12 and dev["loss"] < best["dev_loss"]))
        if improved:
            best, stale = copy.deepcopy(entry), 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "vocab": vocab, "labels": labels, "max_len": plan["max_len"],
                        "model_config": model_config, "experiment": config, "seed": plan["seed"],
                        "best_epoch": epoch, "dev_metrics": dev}, checkpoint_path)
        else:
            stale += 1
        write_json(exp_dir / "history.json", history)
        print(f"  epoch {epoch:02d} train_loss={entry['train_loss']:.4f} dev_loss={dev['loss']:.4f} dev_acc={dev['accuracy']:.4f} dev_f1={dev['macro_f1']:.4f} lr={used_lr:.6f}{' *' if improved else ''}", flush=True)
        if config["scheduler"]:
            scheduler.step(dev["macro_f1"])
        if config["patience"] and stale >= config["patience"]:
            break
    result = {"name": config["name"], "config": config, "best_epoch": best["epoch"],
              "epochs_run": len(history), "dev_accuracy": best["dev_accuracy"], "dev_macro_f1": best["dev_macro_f1"],
              "dev_loss": best["dev_loss"], "parameters": sum(p.numel() for p in model.parameters()),
              "training_seconds": time.perf_counter() - t0, "started_at": started_at, "finished_at": now(),
              "checkpoint": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
              "checkpoint_sha256": sha256(checkpoint_path)}
    write_json(exp_dir / "summary.json", result)
    return result


def train_all(config_path=None):
    plan = read_json(config_path or ROOT / "configs" / "experiments.json")
    write_json(ROOT / "outputs" / "experiment_plan.json", {"created_at": now(), **plan,
               "selection": "maximum dev macro-F1, then minimum dev cross entropy", "test_used_for_selection": False})
    write_json(ROOT / "logs" / "environment.json", {
        "recorded_at": now(), "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
        "training_device": plan["device"], "threads": plan["threads"], "seed": plan["seed"],
        "deterministic_algorithms": True,
    })
    splits, vocab, labels = prepare_data()
    train_ds = make_dataset(splits["train"], vocab, labels, plan["max_len"])
    dev_ds = make_dataset(splits["dev"], vocab, labels, plan["max_len"])
    summaries = [train_one(config, plan, train_ds, dev_ds, vocab, labels) for config in plan["experiments"]]
    winner = max(summaries, key=lambda s: (s["dev_macro_f1"], -s["dev_loss"]))
    import shutil
    shutil.copy2(ROOT / winner["checkpoint"], ROOT / "models" / "best_model.pt")
    write_json(ROOT / "outputs" / "experiment_summary.json", summaries)
    selection = {"selected_at": now(), "criterion": "dev_macro_f1 descending, dev_loss ascending",
                 "test_used_for_selection": False, **winner, "final_checkpoint": "models/best_model.pt",
                 "final_checkpoint_sha256": sha256(ROOT / "models" / "best_model.pt")}
    write_json(ROOT / "outputs" / "selection.json", selection)
    print(f"\nSelected {winner['name']} epoch {winner['best_epoch']} by dev macro-F1={winner['dev_macro_f1']:.6f}", flush=True)
    return selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    train_all(parser.parse_args().config)
