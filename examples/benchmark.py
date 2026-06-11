"""
Benchmark: LLM (Claude) vs LLM (Qwen) vs Optuna TPE vs Random Search.

Same evaluation budget, same search space, same train function.
The search space has 14 parameters with complex interactions — this is
where LLM-guided search should shine over black-box methods.

Usage:
    python examples/benchmark.py --n-evals 30
    python examples/benchmark.py --n-evals 30 --skip-qwen  # faster

Results are saved to benchmark_results.json.
"""

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from benchmark_utils import (
    header,
    print_convergence,
    print_summary,
    run_neuropt,
    run_optuna,
    run_random,
    save_results,
)
from torch.utils.data import DataLoader, Subset

# ── Shared setup ─────────────────────────────────────────────────────────

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 5


def get_dataloaders():
    transform = T.Compose([T.Grayscale(3), T.Resize(32), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    rng = np.random.default_rng(0)
    train_loader = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 5000, replace=False)),
                              batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_ds, rng.choice(len(val_ds), 1250, replace=False)),
                            batch_size=128, shuffle=False, num_workers=0)
    return train_loader, val_loader


class ConfigurableCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = ACTIVATIONS[cfg["activation"]]
        self.blocks, self.projs, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.use_res = cfg["use_residual"]
        prev, spatial = 3, 32
        for i in range(cfg["n_blocks"]):
            out = min(512, max(8, int(cfg["base_channels"] * cfg["channel_growth"] ** i)))
            layers = [nn.Conv2d(prev, out, cfg["kernel_size"], padding=cfg["kernel_size"]//2,
                                bias=not cfg["use_batchnorm"])]
            if cfg["use_batchnorm"]: layers.append(nn.BatchNorm2d(out))
            layers.append(act())
            if cfg["dropout"] > 0: layers.append(nn.Dropout2d(cfg["dropout"]))
            self.blocks.append(nn.Sequential(*layers))
            self.projs.append(nn.Conv2d(prev, out, 1, bias=False) if self.use_res and prev != out
                              else (nn.Identity() if self.use_res else None))
            if (i+1) % cfg["pool_every"] == 0 and spatial > 2:
                self.pools.append(nn.MaxPool2d(2) if cfg["pool_type"] == "max" else nn.AvgPool2d(2))
                spatial //= 2
            else:
                self.pools.append(None)
            prev = out
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = (nn.Sequential(nn.Linear(prev, cfg["fc_hidden"]), act(),
                                   nn.Dropout(cfg["dropout"]), nn.Linear(cfg["fc_hidden"], 10))
                     if cfg["fc_hidden"] > 0 else nn.Linear(prev, 10))

    def forward(self, x):
        for blk, proj, pool in zip(self.blocks, self.projs, self.pools):
            h = blk(x)
            if self.use_res and proj is not None: h = h + proj(x)
            x = pool(h) if pool else h
        return self.head(self.gap(x).flatten(1))


def evaluate(cfg, train_loader, val_loader):
    """Train and evaluate a single config. Returns (val_loss, accuracy, elapsed, *curves)."""
    t0 = time.time()
    try:
        model = ConfigurableCNN(cfg).to(DEVICE)
    except Exception:
        return float("inf"), 0.0, time.time() - t0

    opt_name = cfg.get("optimizer", "adamw")
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accs = [], [], []

    for _ in range(EPOCHS):
        model.train()
        tl, tn = 0.0, 0
        for imgs, tgts in train_loader:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(imgs), tgts)
            if math.isnan(loss.item()):
                return float("inf"), 0.0, time.time() - t0
            loss.backward(); opt.step()
            tl += loss.item() * imgs.size(0); tn += imgs.size(0)
        sched.step(); train_losses.append(tl / tn)

        model.eval()
        vl, vn, vc = 0.0, 0, 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                out = model(imgs)
                vl += crit(out, tgts).item() * imgs.size(0)
                vc += (out.argmax(1) == tgts).sum().item(); vn += imgs.size(0)
        val_losses.append(vl / vn); val_accs.append(vc / vn)

    return val_losses[-1], val_accs[-1], time.time() - t0, train_losses, val_losses, val_accs


# ── Search space (shared) ────────────────────────────────────────────────

SEARCH_SPACE = {
    "n_blocks": (2, 8), "base_channels": (16, 128), "channel_growth": (1.0, 2.5),
    "kernel_size": [3, 5], "activation": ["relu", "gelu", "leaky_relu", "silu"],
    "use_residual": [True, False], "use_batchnorm": [True, False],
    "dropout": (0.0, 0.5), "pool_every": (1, 4), "pool_type": ["max", "avg"],
    "fc_hidden": (0, 512), "lr": (1e-4, 0.1), "wd": (1e-6, 0.01),
    "optimizer": ["sgd", "adam", "adamw"],
}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark: LLM vs Optuna vs Random")
    parser.add_argument("--n-evals", type=int, default=15)
    parser.add_argument("--skip-qwen", action="store_true", help="Skip local Qwen")
    parser.add_argument("--skip-claude", action="store_true", help="Skip Claude API")
    parser.add_argument("--skip-optuna", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Budget: {args.n_evals} evaluations per method")
    print("Search space: 14 parameters (architecture + training)")
    print(f"Epochs per eval: {EPOCHS}")
    print()

    train_loader, val_loader = get_dataloaders()

    def eval_cfg(cfg):
        return evaluate(cfg, train_loader, val_loader)

    def train_fn(config):
        result = eval_cfg(config)
        if len(result) == 3:
            return {"score": result[0], "accuracy": result[1]}
        return {"score": result[0], "accuracy": result[1],
                "train_losses": result[3], "val_losses": result[4], "val_accuracies": result[5]}

    # Warmup
    print("Warmup eval...")
    from neuropt.arch_search import _normalize_search_space, _random_config
    eval_cfg(_random_config(_normalize_search_space(SEARCH_SPACE), random.Random(0)))

    all_results = {}

    for backend, label, skip in [("claude", "LLM (Claude)", args.skip_claude),
                                 ("qwen", "LLM (Qwen)", args.skip_qwen)]:
        if skip:
            continue
        header(f"LLM Search ({backend})")
        try:
            all_results[label] = run_neuropt(
                train_fn, SEARCH_SPACE, backend, args.n_evals,
                f"/tmp/bench_{backend}.jsonl", timeout=60)
        except Exception as e:
            print(f"  Skipped: {e}")

    if not args.skip_optuna:
        header("Optuna TPE")
        all_results["Optuna TPE"] = run_optuna(eval_cfg, SEARCH_SPACE, args.n_evals)

    header("Random Search")
    all_results["Random"] = run_random(eval_cfg, SEARCH_SPACE, args.n_evals)

    print_summary(all_results, args.n_evals, show_time=True)
    print_convergence(all_results, args.n_evals, col_width=16)

    # Save per-method results separately so they don't overwrite each other
    for name, r in all_results.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = f"/tmp/bench_200_{slug}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  Saved {name} → {path}")

    save_results(all_results, "benchmark_results.json")


if __name__ == "__main__":
    main()
