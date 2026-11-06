import argparse
from contextlib import nullcontext
import math
import platform
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from common import ROOT, read_json, write_json, sha256
from model import TransformerLM
from prepare_data import prepare, TokenBlocks


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and torch.cuda.is_bf16_supported() else nullcontext()


@torch.inference_mode()
def score(model, loader, device):
    model.eval()
    loss_sum, tokens, hits = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum")
        loss_sum += float(loss)
        tokens += int(y.ne(-100).sum())
        hits += int((logits.argmax(-1).eq(y) & y.ne(-100)).sum())
    nll = loss_sum / tokens
    return {"nll": nll, "perplexity": math.exp(nll), "accuracy": hits / tokens, "tokens": tokens, "correct": hits}


def train(config_path=None):
    config = read_json(config_path or ROOT / "config.json")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else config["device"])
    torch.set_num_threads(4 if device.type == "cuda" else 2)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    arrays, vocab, stats = prepare(config)
    config["vocab_size"] = len(vocab["itos"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(TokenBlocks(arrays["train"], config["block_size"]), batch_size=config["batch_size"],
                              shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(TokenBlocks(arrays["val"], config["block_size"]), batch_size=config["batch_size"], num_workers=0)
    model = TransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=config["weight_decay"])
    total_steps = len(train_loader) * config["epochs"]
    step, stale, history, best_nll = 0, 0, [], float("inf")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "precision": "bfloat16 autocast" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "float32",
                   "seed": config["seed"], "platform": platform.system()}
    initial_val = score(model, val_loader, device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"{device}: parameters={parameter_count:,}, initial val PPL={initial_val['perplexity']:.2f}, steps/epoch={len(train_loader)}", flush=True)
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total_loss, total_tokens, epoch_start = 0.0, 0, time.perf_counter()
        for batch_index, (x, y) in enumerate(train_loader, 1):
            step += 1
            if step <= config["warmup_steps"]:
                lr = config["learning_rate"] * step / config["warmup_steps"]
            else:
                progress = (step - config["warmup_steps"]) / max(1, total_steps - config["warmup_steps"])
                lr = config["min_learning_rate"] + 0.5 * (config["learning_rate"] - config["min_learning_rate"]) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError("训练损失出现 NaN/Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            n = int(y.ne(-100).sum())
            total_tokens += n
            total_loss += float(loss.detach()) * n
            if batch_index % 150 == 0:
                print(f"epoch {epoch:02d} step {batch_index}/{len(train_loader)} train_nll={total_loss/total_tokens:.4f}", flush=True)
        val = score(model, val_loader, device)
        improved = val["nll"] < best_nll
        if improved:
            best_nll, stale = val["nll"], 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "vocab": vocab, "epoch": epoch, "val_metrics": val,
                        "data_source_sha256": stats["source_sha256"], "tokens_sha256": stats["tokens_sha256"]}, ROOT / "models" / "best_model.pt")
        else:
            stale += 1
        entry = {"epoch": epoch, "train_nll": total_loss / total_tokens, "train_perplexity": math.exp(total_loss / total_tokens),
                 "val_nll": val["nll"], "val_perplexity": val["perplexity"], "val_accuracy": val["accuracy"],
                 "lr": lr, "seconds": time.perf_counter() - epoch_start, "train_tokens": total_tokens,
                 "best": improved}
        history.append(entry)
        write_json(ROOT / "results" / "history.json", history)
        print(f"Epoch {epoch:02d}: train PPL={entry['train_perplexity']:.2f}, val PPL={val['perplexity']:.2f}, val accuracy={val['accuracy']:.2%}, {entry['seconds']:.1f}s{' *' if improved else ''}", flush=True)
        if stale >= config["patience"]:
            break
    best_epoch = min(history, key=lambda h: h["val_nll"])["epoch"]
    summary = {"config": config, "environment": environment, "parameters": parameter_count,
               "initial_val": initial_val, "best_epoch": best_epoch, "epochs_run": len(history),
               "training_seconds": time.perf_counter() - start, "selection": "minimum validation token NLL",
               "test_used_for_selection": False, "checkpoint_sha256": sha256(ROOT / "models" / "best_model.pt")}
    write_json(ROOT / "results" / "training.json", summary)
    print(f"Training complete; best epoch={best_epoch}, val PPL={math.exp(best_nll):.3f}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    train(parser.parse_args().config)
