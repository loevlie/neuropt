"""
LLM-guided CNN architecture search on FashionMNIST.

Usage:
    python examples/llm_arch_search.py
    python examples/llm_arch_search.py --backend claude
    python examples/llm_arch_search.py --backend none   # random baseline

Ctrl+C to stop. Results in arch_search.jsonl (resumable).
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from swarmopt import ArchSearch, LogUniform, Uniform, IntUniform, Categorical


# ── Configurable CNN ─────────────────────────────────────────────────────

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU}


class ConfigurableCNN(nn.Module):
    """CNN built from a config dict."""

    def __init__(self, cfg, in_channels=3, num_classes=10, input_size=32):
        super().__init__()
        act_cls = ACTIVATIONS[cfg["activation"]]
        use_res = cfg["use_residual"]
        use_bn = cfg["use_batchnorm"]
        drop = cfg["dropout"]

        self.blocks = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.use_residual = use_res

        spatial = input_size
        prev_ch = in_channels

        for i in range(cfg["n_blocks"]):
            out_ch = max(8, int(cfg["base_channels"] * (cfg["channel_growth"] ** i)))

            layers = [nn.Conv2d(prev_ch, out_ch, cfg["kernel_size"],
                                padding=cfg["kernel_size"] // 2, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(act_cls())
            if drop > 0:
                layers.append(nn.Dropout2d(drop))
            self.blocks.append(nn.Sequential(*layers))

            if use_res:
                self.projs.append(nn.Conv2d(prev_ch, out_ch, 1, bias=False)
                                  if prev_ch != out_ch else nn.Identity())
            else:
                self.projs.append(None)

            if (i + 1) % cfg["pool_every"] == 0 and spatial > 2:
                pool = nn.MaxPool2d(2) if cfg["pool_type"] == "max" else nn.AvgPool2d(2)
                self.pools.append(pool)
                spatial //= 2
            else:
                self.pools.append(None)

            prev_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if cfg["fc_hidden"] > 0:
            self.head = nn.Sequential(
                nn.Linear(prev_ch, cfg["fc_hidden"]), act_cls(),
                nn.Dropout(drop), nn.Linear(cfg["fc_hidden"], num_classes))
        else:
            self.head = nn.Linear(prev_ch, num_classes)

    def forward(self, x):
        for block, proj, pool in zip(self.blocks, self.projs, self.pools):
            identity = x
            x = block(x)
            if self.use_residual and proj is not None:
                x = x + proj(identity)
            if pool is not None:
                x = pool(x)
        return self.head(self.global_pool(x).flatten(1))


# ── Data & training ──────────────────────────────────────────────────────

def get_dataloaders(data_dir, subset_size, batch_size):
    transform = T.Compose([
        T.Grayscale(num_output_channels=3), T.Resize(32), T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    train_ds = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(0)
    train_idx = rng.choice(len(train_ds), min(subset_size, len(train_ds)), replace=False)
    val_idx = rng.choice(len(val_ds), min(subset_size // 4, len(val_ds)), replace=False)
    return (DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(Subset(val_ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0))


def make_train_fn(train_loader, val_loader, epochs, device):
    def train_fn(cfg):
        try:
            model = ConfigurableCNN(cfg).to(device)
        except Exception as e:
            return {"score": float("inf"), "status": "build_error", "error": str(e)}

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt_name = cfg.get("optimizer", "adamw")
        if opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
        elif opt_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        train_losses, val_losses, val_accuracies = [], [], []

        for ep in range(epochs):
            model.train()
            ep_loss, ep_n = 0.0, 0
            for imgs, tgts in train_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                opt.zero_grad()
                loss = crit(model(imgs), tgts)
                if math.isnan(loss.item()):
                    return {"score": float("inf"), "status": "nan", "n_params": n_params,
                            "train_losses": train_losses, "val_losses": val_losses,
                            "val_accuracies": val_accuracies}
                loss.backward()
                opt.step()
                ep_loss += loss.item() * imgs.size(0)
                ep_n += imgs.size(0)
            sched.step()
            train_losses.append(ep_loss / ep_n)

            model.eval()
            vl, vn, vc = 0.0, 0, 0
            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    out = model(imgs)
                    vl += crit(out, tgts).item() * imgs.size(0)
                    vc += (out.argmax(1) == tgts).sum().item()
                    vn += imgs.size(0)
            val_losses.append(vl / vn)
            val_accuracies.append(vc / vn)

        return {"score": val_losses[-1], "accuracy": val_accuracies[-1], "n_params": n_params,
                "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accuracies}

    return train_fn


# ── Search space & ML context ────────────────────────────────────────────

SEARCH_SPACE = {
    "n_blocks":       IntUniform(2, 8),
    "base_channels":  IntUniform(16, 128),
    "channel_growth": Uniform(1.0, 2.5),
    "kernel_size":    Categorical([3, 5]),
    "activation":     Categorical(["relu", "gelu", "leaky_relu", "silu"]),
    "use_residual":   Categorical([True, False]),
    "use_batchnorm":  Categorical([True, False]),
    "dropout":        Uniform(0.0, 0.5),
    "pool_every":     IntUniform(1, 4),
    "pool_type":      Categorical(["max", "avg"]),
    "fc_hidden":      IntUniform(0, 512),
    "lr":             LogUniform(1e-4, 0.1),
    "wd":             LogUniform(1e-6, 0.01),
    "optimizer":      Categorical(["sgd", "adam", "adamw"]),
}

ML_CONTEXT = """\
You are an expert ML researcher designing CNN architectures for FashionMNIST
(28x28 grayscale, 10 classes, resized to 32x32).

Key principles for this task:
- Small images (32x32) — 3-5 blocks is the sweet spot, >6 needs residual connections
- 100K-2M params is ideal, >5M is overkill for this dataset
- GELU/SiLU generally outperform ReLU — smoother gradients
- BatchNorm almost always helps, stabilizes training
- Dropout >0.3 hurts small models. If underfitting, dropout is actively harmful
- AdamW with lr 1e-3 to 3e-3 is the safest default
- SGD needs higher lr (0.01-0.1) and more tuning to match Adam
- Kernel 3 > kernel 5 for small images (same receptive field with depth)
- Pool every 2 blocks is standard. Every 1 is aggressive for 32x32

Read the training curves carefully:
- Train loss drops + val loss rises = OVERFITTING → less capacity, more regularization
- Both losses stuck high = UNDERFITTING → more capacity, less regularization, higher lr
- Both dropping smoothly = GOOD FIT → fine-tune nearby
- Loss exploding = lr too high → reduce 2-5x
"""


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM-guided CNN architecture search")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-per-iter", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log", default="arch_search.jsonl")
    parser.add_argument("--backend", default=None, choices=["claude", "openai", "qwen", "none"])
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    train_loader, val_loader = get_dataloaders(args.data_dir, args.subset_size, args.batch_size)

    search = ArchSearch(
        train_fn=make_train_fn(train_loader, val_loader, args.epochs, device),
        search_space=SEARCH_SPACE,
        backend=args.backend or "auto",
        log_path=args.log,
        batch_size=args.batch_per_iter,
        ml_context=ML_CONTEXT,
    )
    search.run()


if __name__ == "__main__":
    main()
