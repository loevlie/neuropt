"""
Benchmark with a realistic search space — what you'd actually tune in practice.

5 params: lr, weight decay, dropout, activation, optimizer.
No architecture search, just training hyperparameters on a fixed ResNet-18.

Usage:
    python examples/benchmark_simple.py
    python examples/benchmark_simple.py --n-evals 30
"""

import argparse
import math
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


def train_fn(config):
    loss, acc, _, tl, vl, va = evaluate(config)
    return {"score": loss, "accuracy": acc, "train_losses": tl, "val_losses": vl, "val_accuracies": va}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-evals", type=int, default=15)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Budget: {args.n_evals} evaluations per method")
    print("Search space: 5 params (lr, wd, dropout, activation, optimizer)")
    print("Model: ResNet-18 (fixed architecture)")
    print()

    # Warmup
    print("Warmup...")
    evaluate({"lr": 0.01, "wd": 1e-4, "dropout": 0.1, "activation": "relu", "optimizer": "adamw"})

    all_results = {}

    header("neuropt (Claude)")
    try:
        all_results["neuropt (Claude)"] = run_neuropt(
            train_fn, SPACE, "claude", args.n_evals, "/tmp/bench_simple_claude.jsonl")
    except Exception as e:
        print(f"  Skipped: {e}")

    header("Optuna TPE (n_startup_trials=3)")
    all_results["Optuna TPE"] = run_optuna(evaluate, SPACE, args.n_evals)

    header("Random Search")
    all_results["Random"] = run_random(evaluate, SPACE, args.n_evals)

    print_summary(all_results, args.n_evals)
    print_convergence(all_results, args.n_evals)
    save_results(all_results, "benchmark_simple_results.json")


if __name__ == "__main__":
    main()
