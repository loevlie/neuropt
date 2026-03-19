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
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
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
    """Train and evaluate a single config. Returns (val_loss, accuracy, elapsed)."""
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


def random_config(rng):
    return {
        "n_blocks": rng.randint(2, 8), "base_channels": rng.randint(16, 128),
        "channel_growth": rng.uniform(1.0, 2.5), "kernel_size": rng.choice([3, 5]),
        "activation": rng.choice(["relu", "gelu", "leaky_relu", "silu"]),
        "use_residual": rng.choice([True, False]), "use_batchnorm": rng.choice([True, False]),
        "dropout": rng.uniform(0.0, 0.5),
        "pool_every": rng.randint(1, 4), "pool_type": rng.choice(["max", "avg"]),
        "fc_hidden": rng.randint(0, 512),
        "lr": 10 ** rng.uniform(-4, -1), "wd": 10 ** rng.uniform(-6, -2),
        "optimizer": rng.choice(["sgd", "adam", "adamw"]),
    }


# ── Methods ──────────────────────────────────────────────────────────────

def run_llm_search(backend_name, n_evals, train_loader, val_loader):
    """Run neuropt ArchSearch with a given backend."""
    from neuropt import ArchSearch

    log_path = f"/tmp/bench_{backend_name}.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    def train_fn(config):
        result = evaluate(config, train_loader, val_loader)
        if len(result) == 3:
            return {"score": result[0], "accuracy": result[1]}
        return {"score": result[0], "accuracy": result[1],
                "train_losses": result[3], "val_losses": result[4], "val_accuracies": result[5]}

    search = ArchSearch(
        train_fn=train_fn,
        search_space=SEARCH_SPACE,
        backend=backend_name,
        log_path=log_path,
        batch_size=3,
        timeout=60,
    )
    t0 = time.time()
    search.run(max_evals=n_evals)
    wall_time = time.time() - t0

    with open(log_path) as f:
        results = [json.loads(line) for line in f]

    scores = [r["val_loss"] for r in results if r.get("status") == "ok"]
    accs = [r["val_accuracy"] for r in results if r.get("status") == "ok" and r.get("val_accuracy")]
    return {
        "scores": scores,
        "best_loss": min(scores) if scores else float("inf"),
        "best_acc": max(accs) if accs else 0,
        "wall_time": wall_time,
        "n_evals": len(results),
        "llm_success": search.llm_success,
        "llm_fallback": search.llm_fallback,
    }


def run_optuna(n_evals, train_loader, val_loader):
    """Run Optuna TPE over the same search space."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scores = []

    def objective(trial):
        cfg = {
            "n_blocks": trial.suggest_int("n_blocks", 2, 8),
            "base_channels": trial.suggest_int("base_channels", 16, 128),
            "channel_growth": trial.suggest_float("channel_growth", 1.0, 2.5),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "leaky_relu", "silu"]),
            "use_residual": trial.suggest_categorical("use_residual", [True, False]),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "pool_every": trial.suggest_int("pool_every", 1, 4),
            "pool_type": trial.suggest_categorical("pool_type", ["max", "avg"]),
            "fc_hidden": trial.suggest_int("fc_hidden", 0, 512),
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 0.01, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"]),
        }
        result = evaluate(cfg, train_loader, val_loader)
        scores.append({"val_loss": result[0], "accuracy": result[1]})
        print(f"  Optuna [{len(scores)}/{n_evals}] loss={result[0]:.4f} acc={result[1]:.4f} ({result[2]:.1f}s)")
        return result[0]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
    t0 = time.time()
    study.optimize(objective, n_trials=n_evals)
    wall_time = time.time() - t0

    losses = [s["val_loss"] for s in scores]
    accs = [s["accuracy"] for s in scores]
    return {
        "scores": losses,
        "best_loss": min(losses),
        "best_acc": max(accs),
        "wall_time": wall_time,
        "n_evals": len(scores),
    }


def run_random(n_evals, train_loader, val_loader):
    """Random search baseline."""
    rng = random.Random(42)
    scores = []
    t0 = time.time()

    for i in range(n_evals):
        cfg = random_config(rng)
        result = evaluate(cfg, train_loader, val_loader)
        scores.append({"val_loss": result[0], "accuracy": result[1]})
        print(f"  Random [{i+1}/{n_evals}] loss={result[0]:.4f} acc={result[1]:.4f} ({result[2]:.1f}s)")

    wall_time = time.time() - t0
    losses = [s["val_loss"] for s in scores]
    accs = [s["accuracy"] for s in scores]
    return {
        "scores": losses,
        "best_loss": min(losses),
        "best_acc": max(accs),
        "wall_time": wall_time,
        "n_evals": len(scores),
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
    print(f"Search space: 14 parameters (architecture + training)")
    print(f"Epochs per eval: {EPOCHS}")
    print()

    train_loader, val_loader = get_dataloaders()

    # Warmup
    print("Warmup eval...")
    cfg = random_config(random.Random(0))
    evaluate(cfg, train_loader, val_loader)
    print()

    all_results = {}

    # 1. LLM (Claude)
    if not args.skip_claude:
        print("=" * 60)
        print("LLM Search (Claude)")
        print("=" * 60)
        try:
            all_results["LLM (Claude)"] = run_llm_search("claude", args.n_evals, train_loader, val_loader)
        except Exception as e:
            print(f"  Skipped: {e}")

    # 2. LLM (Qwen local)
    if not args.skip_qwen:
        print("\n" + "=" * 60)
        print("LLM Search (Qwen local)")
        print("=" * 60)
        try:
            all_results["LLM (Qwen)"] = run_llm_search("qwen", args.n_evals, train_loader, val_loader)
        except Exception as e:
            print(f"  Skipped: {e}")

    # 3. Optuna TPE
    if not args.skip_optuna:
        print("\n" + "=" * 60)
        print("Optuna TPE")
        print("=" * 60)
        all_results["Optuna TPE"] = run_optuna(args.n_evals, train_loader, val_loader)

    # 4. Random search
    print("\n" + "=" * 60)
    print("Random Search")
    print("=" * 60)
    all_results["Random"] = run_random(args.n_evals, train_loader, val_loader)

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS ({args.n_evals} evals each)")
    print("=" * 60)
    print(f"{'Method':<20} {'Best Loss':>10} {'Best Acc':>10} {'Wall Time':>10}")
    print("-" * 55)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["best_loss"]):
        print(f"{name:<20} {r['best_loss']:>10.4f} {r['best_acc']:>10.4f} {r['wall_time']:>9.1f}s")

    # Convergence
    print(f"\nConvergence (best-so-far at each eval):")
    milestones = [5, 10, 15, 20, 25, 30, 40, 50]
    milestones = [m for m in milestones if m <= args.n_evals]
    header = f"{'Eval':>6}" + "".join(f"{name:>16}" for name in all_results.keys())
    print(header)
    for m in milestones:
        line = f"{m:>6}"
        for name, r in all_results.items():
            s = r["scores"][:m]
            best = min(s) if s else float("inf")
            line += f"{best:>16.4f}"
        print(line)

    # Save
    # Save per-method results separately so they don't overwrite each other
    for name, r in all_results.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = f"/tmp/bench_200_{slug}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"  Saved {name} → {path}")

    out_path = "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
