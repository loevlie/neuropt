"""
Example training script for neuropt CLI.

    neuropt run examples/train_fashion.py --backend claude

Defines search_space + train_fn. That's it.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ── Search space ─────────────────────────────────────────────────────────
# Plain tuples and lists — neuropt infers LogUniform, IntUniform, etc.

search_space = {
    "n_blocks":       (2, 8),
    "base_channels":  (16, 128),
    "channel_growth": (1.0, 2.5),
    "kernel_size":    [3, 5],
    "activation":     ["relu", "gelu", "leaky_relu", "silu"],
    "use_residual":   [True, False],
    "use_batchnorm":  [True, False],
    "dropout":        (0.0, 0.5),
    "pool_every":     (1, 4),
    "pool_type":      ["max", "avg"],
    "fc_hidden":      (0, 512),
    "lr":             (1e-4, 0.1),
    "wd":             (1e-6, 0.01),
    "optimizer":      ["sgd", "adam", "adamw"],
}


# ── Optional: domain knowledge for the LLM ──────────────────────────────

ml_context = """\
You are designing CNN architectures for FashionMNIST (28x28 grayscale, 10 classes, resized to 32x32).

Key principles:
- Small images → 3-5 blocks is the sweet spot, >6 needs residual connections
- 100K-2M params is ideal, >5M is overkill
- GELU/SiLU generally outperform ReLU
- BatchNorm almost always helps
- Dropout >0.3 hurts small models. If underfitting, try dropout=0
- AdamW lr=1e-3 to 3e-3 is the safest default
- SGD needs higher lr (0.01-0.1) and more tuning
- Kernel 3 > kernel 5 for small images
- Pool every 2 blocks is standard

Read the training curves:
- Train drops + val rises = OVERFITTING → less capacity or more regularization
- Both stuck high = UNDERFITTING → more capacity, less regularization
- Both dropping smoothly = GOOD FIT → fine-tune nearby
- Loss exploding = lr too high → reduce 2-5x
"""


# ── Data ─────────────────────────────────────────────────────────────────

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU}
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

transform = T.Compose([T.Grayscale(3), T.Resize(32), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
val_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
rng = np.random.default_rng(0)
train_loader = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 5000, replace=False)),
                          batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(Subset(val_ds, rng.choice(len(val_ds), 1250, replace=False)),
                        batch_size=128, shuffle=False, num_workers=0)
EPOCHS = 10


# ── Model builder ────────────────────────────────────────────────────────

class ConfigurableCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        act = ACTIVATIONS[cfg["activation"]]
        self.blocks, self.projs, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.use_res = cfg["use_residual"]
        prev, spatial = 3, 32

        for i in range(cfg["n_blocks"]):
            out = min(512, max(8, int(cfg["base_channels"] * cfg["channel_growth"] ** i)))
            layers = [nn.Conv2d(prev, out, cfg["kernel_size"], padding=cfg["kernel_size"]//2, bias=not cfg["use_batchnorm"])]
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
        self.head = (nn.Sequential(nn.Linear(prev, cfg["fc_hidden"]), act(), nn.Dropout(cfg["dropout"]),
                                   nn.Linear(cfg["fc_hidden"], 10))
                     if cfg["fc_hidden"] > 0 else nn.Linear(prev, 10))

    def forward(self, x):
        for blk, proj, pool in zip(self.blocks, self.projs, self.pools):
            h = blk(x)
            if self.use_res and proj is not None: h = h + proj(x)
            x = pool(h) if pool else h
        return self.head(self.gap(x).flatten(1))


# ── Training function ────────────────────────────────────────────────────

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
    train_losses, val_losses, val_accuracies = [], [], []

    for _ in range(EPOCHS):
        model.train()
        tl, tn = 0.0, 0
        for imgs, tgts in train_loader:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(imgs), tgts)
            if math.isnan(loss.item()):
                return {"score": float("inf"), "status": "nan", "n_params": n_params,
                        "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accuracies}
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
        val_losses.append(vl / vn); val_accuracies.append(vc / vn)

    return {"score": val_losses[-1], "accuracy": val_accuracies[-1], "n_params": n_params,
            "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accuracies}
