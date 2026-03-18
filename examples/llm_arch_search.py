"""
LLM-guided autonomous CNN architecture search on FashionMNIST.

The LLM is the search algorithm — no PSO, no Bayesian surrogate.
It reads the full experiment history (including per-epoch train/val curves)
and proposes what to try next based on ML domain knowledge.

Run overnight, wake up to results:
    python examples/llm_arch_search.py
    python examples/llm_arch_search.py --backend claude
    python examples/llm_arch_search.py --backend qwen --batch-per-iter 5

Ctrl+C to stop gracefully.
"""

import argparse
import json
import math
import os
import random
import re
import signal
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from llm_pso.backends import get_default_backend, get_backend_by_name

# ─────────────────────────────────────────────────────────────────────────────
# Config space definition
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_SPACE = {
    # Architecture
    "n_blocks":       {"type": "int",   "min": 2,    "max": 8,
                       "desc": "Number of conv blocks (each = conv + optional BN + activation)"},
    "base_channels":  {"type": "int",   "min": 16,   "max": 128,
                       "desc": "Channels in first block"},
    "channel_growth": {"type": "float", "min": 1.0,  "max": 2.5,
                       "desc": "Channel multiplier per block (channels_i = base * growth^i)"},
    "kernel_size":    {"type": "choice", "choices": [3, 5],
                       "desc": "Conv kernel size (same for all layers)"},
    "activation":     {"type": "choice", "choices": ["relu", "gelu", "leaky_relu", "silu"],
                       "desc": "Activation function"},
    "use_residual":   {"type": "bool",
                       "desc": "Add skip connections (1x1 proj when channels change)"},
    "use_batchnorm":  {"type": "bool",
                       "desc": "BatchNorm after each conv"},
    "dropout":        {"type": "float", "min": 0.0,  "max": 0.5,
                       "desc": "Dropout2d rate after each block (0 = none)"},
    "pool_every":     {"type": "int",   "min": 1,    "max": 4,
                       "desc": "Spatial 2x2 pooling every N blocks"},
    "pool_type":      {"type": "choice", "choices": ["max", "avg"],
                       "desc": "Pooling type for downsampling"},
    "fc_hidden":      {"type": "int",   "min": 0,    "max": 512,
                       "desc": "Hidden FC layer size before output (0 = direct linear)"},
    # Training
    "lr":             {"type": "float", "min": 1e-4,  "max": 0.1, "log": True,
                       "desc": "Learning rate"},
    "wd":             {"type": "float", "min": 1e-6,  "max": 0.01, "log": True,
                       "desc": "Weight decay"},
    "optimizer":      {"type": "choice", "choices": ["sgd", "adam", "adamw"],
                       "desc": "Optimizer"},
}

ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}

# ─────────────────────────────────────────────────────────────────────────────
# Configurable CNN
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Single conv block: Conv -> (BN) -> Activation -> (Dropout)."""

    def __init__(self, in_ch, out_ch, kernel_size, activation, use_bn, dropout):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConfigurableCNN(nn.Module):
    """CNN built from a config dict. Supports variable depth, width, residuals."""

    def __init__(self, cfg, in_channels=3, num_classes=10, input_size=32):
        super().__init__()
        n_blocks = cfg["n_blocks"]
        base_ch = cfg["base_channels"]
        growth = cfg["channel_growth"]
        ks = cfg["kernel_size"]
        act_cls = ACTIVATIONS[cfg["activation"]]
        use_res = cfg["use_residual"]
        use_bn = cfg["use_batchnorm"]
        drop = cfg["dropout"]
        pool_every = cfg["pool_every"]
        pool_type = cfg["pool_type"]
        fc_hidden = cfg["fc_hidden"]

        self.blocks = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.use_residual = use_res

        spatial = input_size
        prev_ch = in_channels
        self._pool_indices = set()

        for i in range(n_blocks):
            out_ch = int(base_ch * (growth ** i))
            out_ch = max(out_ch, 8)  # minimum 8 channels

            self.blocks.append(ConvBlock(prev_ch, out_ch, ks, act_cls, use_bn, drop))

            # Residual projection (1x1 conv if channels change, identity if same)
            if use_res:
                if prev_ch != out_ch:
                    self.residual_projs.append(
                        nn.Conv2d(prev_ch, out_ch, 1, bias=False))
                else:
                    self.residual_projs.append(nn.Identity())
            else:
                self.residual_projs.append(None)

            prev_ch = out_ch

            # Pool to downsample (but don't pool below 2x2)
            if (i + 1) % pool_every == 0 and spatial > 2:
                if pool_type == "max":
                    self.pools.append(nn.MaxPool2d(2))
                else:
                    self.pools.append(nn.AvgPool2d(2))
                self._pool_indices.add(i)
                spatial = spatial // 2
            else:
                self.pools.append(None)

        # Global average pool -> head
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if fc_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(prev_ch, fc_hidden),
                act_cls(),
                nn.Dropout(drop),
                nn.Linear(fc_hidden, num_classes),
            )
        else:
            self.head = nn.Linear(prev_ch, num_classes)

    def forward(self, x):
        for i, (block, proj, pool) in enumerate(
                zip(self.blocks, self.residual_projs, self.pools)):
            identity = x
            x = block(x)
            if self.use_residual and proj is not None:
                x = x + proj(identity)
            if pool is not None:
                x = pool(x)

        x = self.global_pool(x).flatten(1)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir, subset_size, batch_size):
    transform = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform)
    val_ds = torchvision.datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=transform)

    rng = np.random.default_rng(0)
    train_idx = rng.choice(len(train_ds), min(subset_size, len(train_ds)), replace=False)
    val_idx = rng.choice(len(val_ds), min(subset_size // 4, len(val_ds)), replace=False)

    return (
        DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                   shuffle=True, num_workers=0),
        DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                   shuffle=False, num_workers=0),
    )


def train_and_evaluate(cfg, train_loader, val_loader, epochs, device):
    """Train a ConfigurableCNN and return rich metrics."""
    try:
        model = ConfigurableCNN(cfg, in_channels=3, num_classes=10, input_size=32)
        model = model.to(device)
    except Exception as e:
        return {"score": float("inf"), "status": "build_error",
                "error": f"Failed to build model: {e}",
                "n_params": 0, "train_losses": [], "val_losses": [],
                "val_accuracies": []}

    n_params = count_params(model)

    # Optimizer
    opt_name = cfg.get("optimizer", "adamw")
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                    momentum=0.9, weight_decay=cfg["wd"])
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["wd"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["wd"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            if math.isnan(loss.item()):
                return {"score": float("inf"), "status": "nan",
                        "error": f"NaN loss at epoch {epoch+1}",
                        "n_params": n_params,
                        "train_losses": train_losses, "val_losses": val_losses,
                        "val_accuracies": val_accuracies}
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            epoch_n += images.size(0)
        scheduler.step()
        train_losses.append(epoch_loss / epoch_n)

        # Validate
        model.eval()
        v_loss, v_n, v_correct = 0.0, 0, 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                out = model(images)
                v_loss += criterion(out, targets).item() * images.size(0)
                v_correct += (out.argmax(1) == targets).sum().item()
                v_n += images.size(0)
        val_losses.append(v_loss / v_n)
        val_accuracies.append(v_correct / v_n)

    return {
        "score": val_losses[-1],
        "accuracy": val_accuracies[-1],
        "n_params": n_params,
        "train_loss_final": train_losses[-1],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "status": "ok",
        "error": "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# JSONL Logger
# ─────────────────────────────────────────────────────────────────────────────

class ArchSearchLogger:
    """Append-only JSONL logger for architecture search."""

    def __init__(self, path):
        self.path = path
        self._counter = 0
        if os.path.exists(path):
            with open(path) as f:
                self._counter = sum(1 for _ in f)

    def log(self, iteration, cfg, result, source="llm"):
        self._counter += 1
        entry = {
            "id": self._counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            "source": source,
            "config": cfg,
            "val_loss": result.get("score"),
            "val_accuracy": result.get("accuracy"),
            "n_params": result.get("n_params"),
            "train_losses": result.get("train_losses", []),
            "val_losses": result.get("val_losses", []),
            "val_accuracies": result.get("val_accuracies", []),
            "elapsed": result.get("elapsed"),
            "status": result.get("status"),
            "error": result.get("error", ""),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()

    def load_history(self):
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


# ─────────────────────────────────────────────────────────────────────────────
# LLM Architect — the search algorithm
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert deep learning researcher specializing in CNN architecture design.
You are conducting a neural architecture search for image classification on FashionMNIST
(28x28 grayscale images, 10 classes, resized to 32x32 with 3-channel grayscale→RGB conversion).

## Your ML Knowledge (use this to guide your decisions)

**Architecture principles:**
- FashionMNIST is small (32x32) — huge models waste parameters and overfit easily.
  A good model here is 100K-2M params. Over 5M is almost certainly overkill.
- Deeper networks (>6 blocks) need residual connections or they suffer from
  vanishing gradients. 3-5 blocks is the sweet spot for 32x32 images.
- Channel growth of ~1.5-2x per block is standard. Starting with 32-64 base channels works well.
- Kernel size 3 is almost always better than 5 for small images (more efficient, same receptive field with depth).
- Pool every 2 blocks is standard. Pool every 1 is aggressive (loses spatial info fast).
  For 32x32, you get at most 4 pooling ops before you're at 2x2.
- Global average pooling before the head is better than flattening (fewer params, more robust).

**Activation functions:**
- GELU and SiLU (Swish) generally outperform ReLU on modern tasks — smoother gradients.
- LeakyReLU helps with dying neuron problem if you see underfitting.
- ReLU is fastest but can cause dead neurons, especially with high learning rates.

**Regularization:**
- BatchNorm: almost always helps. Stabilizes training, acts as mild regularization.
  Only skip it if you want to test its effect or have very few channels.
- Dropout 0.1-0.2 is usually enough. >0.3 hurts small models.
  If the model is underfitting, dropout is ACTIVELY HARMFUL — reduce it.
- Weight decay 1e-4 to 1e-3 for AdamW, 1e-4 to 1e-2 for SGD.

**Optimizer selection:**
- AdamW: best default. Handles LR sensitivity well, good with cosine schedule.
  LR 1e-3 to 3e-3 is the sweet spot.
- Adam: similar to AdamW but weight decay is less principled.
- SGD+momentum: can match Adam with careful LR tuning (typically needs higher LR, 0.01-0.1).
  More sensitive to LR choice.

**Reading the training curves (CRITICAL — this is your main signal):**
- Train loss dropping + val loss rising = OVERFITTING → increase dropout, increase wd, reduce model size, or add data augmentation
- Train loss stuck high = UNDERFITTING → increase model capacity (more channels/blocks), reduce dropout/wd, increase LR
- Train AND val both dropping smoothly = GOOD FIT → try training longer or fine-tuning nearby
- Loss oscillating or exploding = LR too high → reduce LR by 2-5x
- Very slow convergence = LR too low → increase LR
- Train-val gap small but both plateaued = capacity ceiling → try different architecture

**Strategy for this search:**
- Iteration 0 (no history): propose diverse configs — vary architecture AND training params.
  Include one simple baseline (3 blocks, 32 channels, ReLU) and one ambitious one.
- Early iterations: explore broadly. Try different activation/residual/BN combos.
- After seeing results: exploit what works. If GELU+residual+BN consistently wins,
  focus on tuning depth/width/LR around that combo.
- If the best model is overfitting: try dropout, more wd, fewer channels.
- If the best model is underfitting: try more blocks, wider channels, lower wd.
- Always propose at least 1 "exploration" config that tries something new.

## Config Space

Each config is a JSON object with these keys:
"""

def build_config_space_description():
    lines = []
    for name, spec in CONFIG_SPACE.items():
        if spec["type"] == "choice":
            lines.append(f'- "{name}": one of {spec["choices"]} — {spec["desc"]}')
        elif spec["type"] == "bool":
            lines.append(f'- "{name}": true or false — {spec["desc"]}')
        elif spec["type"] == "int":
            lines.append(f'- "{name}": integer [{spec["min"]}, {spec["max"]}] — {spec["desc"]}')
        elif spec["type"] == "float":
            scale = " (log scale)" if spec.get("log") else ""
            lines.append(f'- "{name}": float [{spec["min"]}, {spec["max"]}]{scale} — {spec["desc"]}')
    return "\n".join(lines)


def build_prompt(history, best_result, n_configs):
    parts = [SYSTEM_PROMPT, build_config_space_description(), ""]

    # Best result
    if best_result:
        parts.append("## Best Result So Far")
        parts.append(f"Val loss: {best_result['val_loss']:.4f}, "
                     f"Val accuracy: {best_result['val_accuracy']:.4f}, "
                     f"Params: {best_result['n_params']:,}")
        parts.append(f"Config: {json.dumps(best_result['config'], default=str)}")
        parts.append("")

    # Recent history
    recent = history[-20:]
    if recent:
        parts.append("## Recent Experiments (last 20)\n")
        parts.append("| # | n_blocks | base_ch | growth | act | resid | BN | drop | optim | lr | val_loss | val_acc | params | status |")
        parts.append("|---|---------|---------|--------|-----|-------|----|------|-------|----|----------|---------|--------|--------|")
        for row in recent:
            cfg = row.get("config", {})
            vl = row.get("val_loss")
            va = row.get("val_accuracy")
            vl_s = f"{vl:.4f}" if vl and vl != float("inf") else "inf"
            va_s = f"{va:.4f}" if va else "-"
            np_s = f"{row.get('n_params', 0):,}" if row.get("n_params") else "-"
            cg = cfg.get("channel_growth", "")
            cg_s = f"{cg:.1f}" if isinstance(cg, (int, float)) else str(cg)
            parts.append(
                f"| {row.get('id', '')} "
                f"| {cfg.get('n_blocks', '')} "
                f"| {cfg.get('base_channels', '')} "
                f"| {cg_s} "
                f"| {cfg.get('activation', '')} "
                f"| {'Y' if cfg.get('use_residual') else 'N'} "
                f"| {'Y' if cfg.get('use_batchnorm') else 'N'} "
                f"| {cfg.get('dropout', '')} "
                f"| {cfg.get('optimizer', '')} "
                f"| {cfg.get('lr', '')} "
                f"| {vl_s} | {va_s} | {np_s} | {row.get('status', '')} |"
            )
        parts.append("")

        # Per-epoch curves for recent OK experiments
        ok_with_curves = [r for r in recent
                          if r.get("status") == "ok" and r.get("train_losses")]
        if ok_with_curves:
            parts.append("## Per-Epoch Learning Curves (last 8 successful)\n")
            parts.append("Look for overfitting (train drops, val rises), underfitting "
                         "(both stuck high), divergence (loss explodes).\n")
            for row in ok_with_curves[-8:]:
                cfg = row["config"]
                parts.append(
                    f'{cfg["n_blocks"]}blk/{cfg["base_channels"]}ch/'
                    f'{cfg["activation"]}/{"res" if cfg.get("use_residual") else "nores"}/'
                    f'{cfg["optimizer"]} lr={cfg["lr"]}:')
                tl = row.get("train_losses", [])
                vl = row.get("val_losses", [])
                va = row.get("val_accuracies", [])
                for e in range(max(len(tl), len(vl), len(va))):
                    line = f"  ep{e+1}:"
                    if e < len(tl): line += f" train={tl[e]:.4f}"
                    if e < len(vl): line += f" val={vl[e]:.4f}"
                    if e < len(va): line += f" acc={va[e]:.4f}"
                    parts.append(line)
            parts.append("")

        # Trend analysis
        ok_rows = [r for r in recent if r.get("status") == "ok" and r.get("val_loss")]
        if len(ok_rows) >= 3:
            parts.append("## Pre-Computed Trends\n")
            losses = [r["val_loss"] for r in ok_rows if r["val_loss"] != float("inf")]
            if losses:
                parts.append(f"- Val loss range: [{min(losses):.4f}, {max(losses):.4f}]")
                best_r = min(ok_rows, key=lambda r: r["val_loss"])
                parts.append(f"- Best config: {best_r['config'].get('activation')}, "
                             f"{best_r['config'].get('n_blocks')} blocks, "
                             f"lr={best_r['config'].get('lr')}, "
                             f"optimizer={best_r['config'].get('optimizer')}")

            # Overfitting analysis
            for row in ok_rows[-8:]:
                tl = row.get("train_losses", [])
                vl = row.get("val_losses", [])
                if len(tl) >= 2 and len(vl) >= 2:
                    gap = vl[-1] - tl[-1]
                    if tl[-1] < tl[0] and vl[-1] > min(vl) and gap > 0.3:
                        cfg = row["config"]
                        parts.append(
                            f"- OVERFITTING: {cfg['n_blocks']}blk/{cfg['base_channels']}ch "
                            f"(train {tl[0]:.3f}->{tl[-1]:.3f}, val {min(vl):.3f}->{vl[-1]:.3f})")
                    elif tl[-1] > 1.5:
                        cfg = row["config"]
                        parts.append(
                            f"- UNDERFITTING: {cfg['n_blocks']}blk/{cfg['base_channels']}ch "
                            f"(train still {tl[-1]:.3f} after {len(tl)} epochs)")
            parts.append("")

    # Instructions
    parts.append(f"## Task\n")
    parts.append(
        f"Propose exactly {n_configs} CNN configs to try next. "
        "Use the experiment history and your ML knowledge to make smart choices. "
        "Balance exploration (try new things) with exploitation (refine what works).\n"
        "Respond with ONLY a JSON array of config objects. No explanation, no markdown fences, no comments."
    )

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Config validation and random generation
# ─────────────────────────────────────────────────────────────────────────────

def validate_config(cfg):
    """Validate and clamp a config dict. Returns cleaned config or None."""
    if not isinstance(cfg, dict):
        return None

    cleaned = {}
    for name, spec in CONFIG_SPACE.items():
        if name not in cfg:
            return None
        val = cfg[name]

        if spec["type"] == "int":
            try:
                val = int(round(float(val)))
            except (TypeError, ValueError):
                return None
            val = max(spec["min"], min(val, spec["max"]))
            cleaned[name] = val

        elif spec["type"] == "float":
            try:
                val = float(val)
            except (TypeError, ValueError):
                return None
            val = max(spec["min"], min(val, spec["max"]))
            cleaned[name] = val

        elif spec["type"] == "choice":
            if val not in spec["choices"]:
                return None
            cleaned[name] = val

        elif spec["type"] == "bool":
            if not isinstance(val, bool):
                if isinstance(val, str):
                    val = val.lower() in ("true", "1", "yes")
                else:
                    val = bool(val)
            cleaned[name] = val

    return cleaned


def random_config(rng=None):
    """Generate a random valid config."""
    if rng is None:
        rng = random
    cfg = {}
    for name, spec in CONFIG_SPACE.items():
        if spec["type"] == "int":
            cfg[name] = rng.randint(spec["min"], spec["max"])
        elif spec["type"] == "float":
            if spec.get("log"):
                log_lo = math.log10(spec["min"])
                log_hi = math.log10(spec["max"])
                cfg[name] = 10 ** rng.uniform(log_lo, log_hi)
            else:
                cfg[name] = rng.uniform(spec["min"], spec["max"])
        elif spec["type"] == "choice":
            cfg[name] = rng.choice(spec["choices"])
        elif spec["type"] == "bool":
            cfg[name] = rng.choice([True, False])
    return cfg


def parse_llm_response(response, expected_count):
    """Parse LLM response into validated config list. Returns None on failure."""
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if not match:
        return None
    try:
        configs = json.loads(match.group())
    except json.JSONDecodeError:
        return None

    if not isinstance(configs, list) or len(configs) != expected_count:
        return None

    validated = []
    for cfg in configs:
        cleaned = validate_config(cfg)
        if cleaned is None:
            return None
        validated.append(cleaned)

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-guided autonomous CNN architecture search")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-per-iter", type=int, default=3,
                        help="Configs to try per LLM consultation")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log", default="arch_search.jsonl")
    parser.add_argument("--backend", default=None,
                        choices=["claude", "openai", "qwen", "none"])
    args = parser.parse_args()

    # Device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Backend
    if args.backend is not None:
        backend = get_backend_by_name(args.backend)
    else:
        backend = get_default_backend()

    backend_name = backend.name if backend else "None (random search)"

    # Data (load once, reuse across experiments)
    train_loader, val_loader = get_dataloaders(
        args.data_dir, args.subset_size, args.batch_size)

    logger = ArchSearchLogger(args.log)
    history = logger.load_history()

    # Shutdown handling
    shutdown = [False]
    def handler(signum, frame):
        if shutdown[0]:
            print("\nForce exit.")
            sys.exit(1)
        print("\nShutdown requested. Finishing current experiment...")
        shutdown[0] = True
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    print("=" * 70)
    print("LLM-Guided CNN Architecture Search")
    print(f"  Device: {device}")
    print(f"  Backend: {backend_name}")
    print(f"  Epochs/eval: {args.epochs}")
    print(f"  Configs/iter: {args.batch_per_iter}")
    print(f"  Log: {args.log}")
    print(f"  Existing experiments: {len(history)}")
    print("  Press Ctrl+C to stop gracefully")
    print("=" * 70)
    print()

    best_score = float("inf")
    best_config = None
    best_accuracy = 0.0
    total = len(history)
    iteration = 0

    # Restore best from history
    for row in history:
        if row.get("status") == "ok" and row.get("val_loss", float("inf")) < best_score:
            best_score = row["val_loss"]
            best_config = row.get("config")
            best_accuracy = row.get("val_accuracy", 0)

    if best_config:
        print(f"Resuming — best so far: loss={best_score:.4f}, acc={best_accuracy:.4f}")
        print()

    rng = random.Random(42 + len(history))
    llm_success = 0
    llm_fallback = 0

    while not shutdown[0]:
        iter_start = time.time()

        # Ask LLM for configs
        source = "random"
        configs = None

        if backend is not None:
            try:
                best_info = None
                if best_config is not None:
                    best_info = {"val_loss": best_score, "val_accuracy": best_accuracy,
                                 "config": best_config,
                                 "n_params": next(
                                     (r.get("n_params", 0) for r in reversed(history)
                                      if r.get("config") == best_config), 0)}
                prompt = build_prompt(history, best_info, args.batch_per_iter)
                response = backend.generate(prompt, max_tokens=2048)
                configs = parse_llm_response(response, args.batch_per_iter)
                if configs is not None:
                    source = "llm"
                    llm_success += 1
                else:
                    llm_fallback += 1
                    print(f"  [LLM parse failed, falling back to random]")
            except Exception as e:
                llm_fallback += 1
                print(f"  [LLM error: {e}, falling back to random]")

        if configs is None:
            configs = [random_config(rng) for _ in range(args.batch_per_iter)]
            source = "random"

        # Evaluate each config
        for idx, cfg in enumerate(configs):
            if shutdown[0]:
                break

            t_start = time.time()
            try:
                result = train_and_evaluate(cfg, train_loader, val_loader,
                                            args.epochs, device)
            except Exception as e:
                result = {"score": float("inf"), "status": "error",
                          "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                          "n_params": 0, "train_losses": [], "val_losses": [],
                          "val_accuracies": []}
            result["elapsed"] = time.time() - t_start

            # Update best
            improved = ""
            if result.get("score", float("inf")) < best_score:
                best_score = result["score"]
                best_config = cfg
                best_accuracy = result.get("accuracy", 0)
                improved = " *** NEW BEST ***"

            # Log
            logger.log(iteration, cfg, result, source)
            history.append({
                "id": total + 1,
                "config": cfg,
                "val_loss": result.get("score"),
                "val_accuracy": result.get("accuracy"),
                "n_params": result.get("n_params"),
                "train_losses": result.get("train_losses", []),
                "val_losses": result.get("val_losses", []),
                "val_accuracies": result.get("val_accuracies", []),
                "status": result.get("status"),
            })
            total += 1

            # Print
            acc_s = f"acc={result['accuracy']:.4f}" if result.get("accuracy") else "acc=-"
            n_params_s = f"{result.get('n_params', 0):,}p"
            status_s = f" [{result['status']}]" if result["status"] != "ok" else ""
            arch_s = (f"{cfg['n_blocks']}blk/{cfg['base_channels']}ch/"
                      f"{cfg['activation']}/{'res' if cfg['use_residual'] else 'nores'}")
            print(f"  [{iteration}.{idx}] {arch_s} {cfg['optimizer']} lr={cfg['lr']:.4e} "
                  f"→ loss={result['score']:.4f} {acc_s} ({n_params_s}, "
                  f"{result['elapsed']:.1f}s) [{source}]{status_s}{improved}")

        if shutdown[0]:
            break

        iter_elapsed = time.time() - iter_start
        print(f"\n  Iter {iteration} done in {iter_elapsed:.1f}s | "
              f"Best: loss={best_score:.4f} acc={best_accuracy:.4f} | "
              f"Total: {total} experiments")
        if backend:
            print(f"  LLM: {llm_success} advised, {llm_fallback} fallbacks")
        print()
        iteration += 1

    # Final summary
    print("\n" + "=" * 70)
    print("SHUTDOWN SUMMARY")
    print(f"  Total iterations: {iteration}")
    print(f"  Total experiments: {total}")
    print(f"  Best val loss: {best_score:.4f}")
    print(f"  Best accuracy: {best_accuracy:.4f}")
    if best_config:
        print(f"  Best config:")
        for k, v in best_config.items():
            print(f"    {k}: {v}")
    if backend:
        print(f"  LLM: {llm_success} advised, {llm_fallback} fallbacks")
    print("=" * 70)


if __name__ == "__main__":
    main()
