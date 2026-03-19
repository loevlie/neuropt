"""
Benchmark with a realistic search space — what you'd actually tune in practice.

5 params: lr, weight decay, dropout, activation, optimizer.
No architecture search, just training hyperparameters on a fixed ResNet-18.

Usage:
    python examples/benchmark_simple.py
    python examples/benchmark_simple.py --n-evals 30
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


DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 5

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "leaky_relu": nn.LeakyReLU}

SPACE = {
    "lr": (1e-4, 0.1),
    "wd": (1e-6, 0.01),
    "dropout": (0.0, 0.5),
    "activation": ["relu", "gelu", "silu", "leaky_relu"],
    "optimizer": ["sgd", "adam", "adamw"],
}


def get_dataloaders():
    transform = T.Compose([T.Grayscale(3), T.Resize(32), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    rng = np.random.default_rng(0)
    return (DataLoader(Subset(train_ds, rng.choice(len(train_ds), 5000, replace=False)),
                       batch_size=128, shuffle=True, num_workers=0),
            DataLoader(Subset(val_ds, rng.choice(len(val_ds), 1250, replace=False)),
                       batch_size=128, shuffle=False, num_workers=0))


train_loader, val_loader = get_dataloaders()


def evaluate(cfg):
    t0 = time.time()
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Swap activations
    act_cls = ACTIVATIONS[cfg["activation"]]
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ReLU):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            setattr(parent, parts[-1], act_cls(inplace=True) if cfg["activation"] != "gelu" else act_cls())

    # Add dropout before final FC
    model.fc = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(512, 10))
    model = model.to(DEVICE)

    opt_name = cfg["optimizer"]
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
                return float("inf"), 0.0, time.time() - t0, [], [], []
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


def random_config(rng):
    return {
        "lr": 10 ** rng.uniform(-4, -1),
        "wd": 10 ** rng.uniform(-6, -2),
        "dropout": rng.uniform(0.0, 0.5),
        "activation": rng.choice(["relu", "gelu", "silu", "leaky_relu"]),
        "optimizer": rng.choice(["sgd", "adam", "adamw"]),
    }


def run_neuropt(backend_name, n_evals):
    from neuropt import ArchSearch

    log_path = f"/tmp/bench_simple_{backend_name}.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    def train_fn(config):
        loss, acc, _, tl, vl, va = evaluate(config)
        return {"score": loss, "accuracy": acc, "train_losses": tl, "val_losses": vl, "val_accuracies": va}

    search = ArchSearch(train_fn=train_fn, search_space=SPACE, backend=backend_name,
                        log_path=log_path, batch_size=3)
    t0 = time.time()
    search.run(max_evals=n_evals)

    with open(log_path) as f:
        results = [json.loads(line) for line in f]
    scores = [r["val_loss"] for r in results if r.get("status") == "ok"]
    accs = [r.get("val_accuracy", 0) for r in results if r.get("status") == "ok"]
    return {"scores": scores, "best_loss": min(scores), "best_acc": max(accs),
            "wall_time": time.time() - t0}


def run_optuna(n_evals):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scores = []

    def objective(trial):
        cfg = {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 0.01, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu", "leaky_relu"]),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"]),
        }
        loss, acc, elapsed, _, _, _ = evaluate(cfg)
        scores.append({"val_loss": loss, "accuracy": acc})
        print(f"  Optuna [{len(scores)}/{n_evals}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)")
        return loss

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
    t0 = time.time()
    study.optimize(objective, n_trials=n_evals)
    losses = [s["val_loss"] for s in scores]
    accs = [s["accuracy"] for s in scores]
    return {"scores": losses, "best_loss": min(losses), "best_acc": max(accs),
            "wall_time": time.time() - t0}


def run_random(n_evals):
    rng = random.Random(42)
    scores = []
    t0 = time.time()
    for i in range(n_evals):
        cfg = random_config(rng)
        loss, acc, elapsed, _, _, _ = evaluate(cfg)
        scores.append({"val_loss": loss, "accuracy": acc})
        print(f"  Random [{i+1}/{n_evals}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)")
    losses = [s["val_loss"] for s in scores]
    accs = [s["accuracy"] for s in scores]
    return {"scores": losses, "best_loss": min(losses), "best_acc": max(accs),
            "wall_time": time.time() - t0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-evals", type=int, default=15)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Budget: {args.n_evals} evaluations per method")
    print(f"Search space: 5 params (lr, wd, dropout, activation, optimizer)")
    print(f"Model: ResNet-18 (fixed architecture)")
    print()

    # Warmup
    print("Warmup...")
    evaluate({"lr": 0.01, "wd": 1e-4, "dropout": 0.1, "activation": "relu", "optimizer": "adamw"})
    print()

    all_results = {}

    print("=" * 60)
    print("neuropt (Claude)")
    print("=" * 60)
    try:
        all_results["neuropt (Claude)"] = run_neuropt("claude", args.n_evals)
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n" + "=" * 60)
    print("Optuna TPE (n_startup_trials=3)")
    print("=" * 60)
    all_results["Optuna TPE"] = run_optuna(args.n_evals)

    print("\n" + "=" * 60)
    print("Random Search")
    print("=" * 60)
    all_results["Random"] = run_random(args.n_evals)

    print("\n" + "=" * 60)
    print(f"RESULTS ({args.n_evals} evals each)")
    print("=" * 60)
    print(f"{'Method':<22} {'Best Loss':>10} {'Best Acc':>10}")
    print("-" * 45)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["best_loss"]):
        print(f"{name:<22} {r['best_loss']:>10.4f} {r['best_acc']:>10.4f}")

    print(f"\nConvergence (best-so-far):")
    milestones = [m for m in [5, 10, 15, 20, 25, 30] if m <= args.n_evals]
    header = f"{'Eval':>6}" + "".join(f"{name:>22}" for name in all_results.keys())
    print(header)
    for m in milestones:
        line = f"{m:>6}"
        for name, r in all_results.items():
            s = r["scores"][:m]
            best = min(s) if s else float("inf")
            line += f"{best:>22.4f}"
        print(line)

    out_path = "benchmark_simple_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
