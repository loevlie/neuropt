"""
Ablation: LLM memory (insights) vs no memory.

Runs the CNN architecture search 3 times with and 3 times without
the insight extraction feature, then reports mean ± std.

Usage:
    python examples/benchmark_memory.py
    python examples/benchmark_memory.py --n-evals 30 --n-runs 3
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# ── Shared setup ─────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 5
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU}

SPACE = {
    "n_blocks": (2, 8), "base_channels": (16, 128), "channel_growth": (1.0, 2.5),
    "kernel_size": [3, 5], "activation": ["relu", "gelu", "leaky_relu", "silu"],
    "use_residual": [True, False], "use_batchnorm": [True, False],
    "dropout": (0.0, 0.5), "pool_every": (1, 4), "pool_type": ["max", "avg"],
    "fc_hidden": (0, 512), "lr": (1e-4, 0.1), "wd": (1e-6, 0.01),
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


def make_train_fn(train_loader, val_loader):
    def train_fn(config):
        try:
            model = ConfigurableCNN(config).to(DEVICE)
        except Exception as e:
            return {"score": float("inf"), "status": "build_error", "error": str(e)}

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt_name = config.get("optimizer", "adamw")
        if opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["wd"])
        elif opt_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        crit = nn.CrossEntropyLoss()
        tl_list, vl_list, va_list = [], [], []

        for _ in range(EPOCHS):
            model.train()
            tl, tn = 0.0, 0
            for imgs, tgts in train_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(imgs), tgts)
                if math.isnan(loss.item()):
                    return {"score": float("inf"), "status": "nan", "n_params": n_params,
                            "train_losses": tl_list, "val_losses": vl_list, "val_accuracies": va_list}
                loss.backward(); opt.step()
                tl += loss.item() * imgs.size(0); tn += imgs.size(0)
            sched.step(); tl_list.append(tl / tn)

            model.eval()
            vl, vn, vc = 0.0, 0, 0
            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                    out = model(imgs)
                    vl += crit(out, tgts).item() * imgs.size(0)
                    vc += (out.argmax(1) == tgts).sum().item(); vn += imgs.size(0)
            vl_list.append(vl / vn); va_list.append(vc / vn)

        return {"score": vl_list[-1], "accuracy": va_list[-1], "n_params": n_params,
                "train_losses": tl_list, "val_losses": vl_list, "val_accuracies": va_list}
    return train_fn


def run_one(train_fn, n_evals, use_memory, run_id):
    """Run one search and return best-so-far at each eval."""
    from neuropt import ArchSearch

    label = "memory" if use_memory else "no_memory"
    log_path = f"/tmp/bench_mem_{label}_{run_id}.jsonl"
    insights_path = log_path.rsplit(".", 1)[0] + "_insights.json"
    for f in [log_path, insights_path]:
        if os.path.exists(f):
            os.remove(f)

    search = ArchSearch(
        train_fn=train_fn,
        search_space=SPACE,
        backend="claude",
        log_path=log_path,
        batch_size=3,
        timeout=60,
    )

    # Disable memory if needed
    if not use_memory:
        search._extract_insights = lambda history: None

    t0 = time.time()
    search.run(max_evals=n_evals)
    elapsed = time.time() - t0

    with open(log_path) as f:
        rows = [json.loads(line) for line in f]

    scores = [r["val_loss"] for r in rows if r.get("status") == "ok"]
    bsf = list(np.minimum.accumulate(scores)) if scores else []

    return {
        "best_so_far": bsf,
        "best_loss": min(scores) if scores else float("inf"),
        "n_insights": len(search._insights),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-evals", type=int, default=15)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Ablation: memory vs no memory")
    print(f"  {args.n_runs} runs × {args.n_evals} evals each")
    print()

    train_loader, val_loader = get_dataloaders()
    train_fn = make_train_fn(train_loader, val_loader)

    # Warmup
    print("Warmup...")
    train_fn({"n_blocks": 3, "base_channels": 32, "channel_growth": 1.5, "kernel_size": 3,
              "activation": "relu", "use_residual": True, "use_batchnorm": True, "dropout": 0.1,
              "pool_every": 2, "pool_type": "max", "fc_hidden": 64, "lr": 0.01, "wd": 1e-4,
              "optimizer": "adamw"})
    print()

    results = {"memory": [], "no_memory": []}

    for run_id in range(args.n_runs):
        for use_memory in [True, False]:
            label = "memory" if use_memory else "no_memory"
            print(f"\n{'='*60}")
            print(f"Run {run_id+1}/{args.n_runs} — {label}")
            print(f"{'='*60}")
            r = run_one(train_fn, args.n_evals, use_memory, run_id)
            results[label].append(r)
            print(f"  → best: {r['best_loss']:.4f}, insights: {r['n_insights']}, time: {r['elapsed']:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for label in ["memory", "no_memory"]:
        bests = [r["best_loss"] for r in results[label]]
        print(f"  {label:>12}: {np.mean(bests):.4f} ± {np.std(bests):.4f}  (runs: {[f'{b:.4f}' for b in bests]})")
        if label == "memory":
            insights = [r["n_insights"] for r in results[label]]
            print(f"               insights per run: {insights}")

    out_path = "benchmark_memory_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
