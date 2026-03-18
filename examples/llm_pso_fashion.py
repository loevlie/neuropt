"""
Autonomous LLM-guided PSO hyperparameter search for FashionMNIST.

Runs indefinitely until Ctrl+C. Logs everything to TSV.

Usage:
    python examples/llm_pso_fashion.py                    # auto-detect backend
    python examples/llm_pso_fashion.py --backend none     # pure PSO
    python examples/llm_pso_fashion.py --backend claude   # Claude API
    python examples/llm_pso_fashion.py --backend qwen     # local Qwen on CPU
    python examples/llm_pso_fashion.py --backend openai   # OpenAI API
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from swarmopt import LogUniform
from llm_pso import AutonomousRunner
from llm_pso.backends import get_default_backend, get_backend_by_name


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
    train_idx = rng.choice(len(train_ds), min(subset_size, len(train_ds)),
                           replace=False)
    val_idx = rng.choice(len(val_ds), min(subset_size // 4, len(val_ds)),
                         replace=False)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader


def make_train_fn(data_dir, subset_size, batch_size, epochs):
    """Create a training function with per-epoch loss tracking."""

    def train_fn(params):
        lr = params["lr"]
        wd = params["wd"]
        device = params.get("device", "cpu")

        train_loader, val_loader = get_dataloaders(
            data_dir, subset_size, batch_size)

        model = torchvision.models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                                bias=False)
        model.maxpool = nn.Identity()
        model = model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # Training with per-epoch train AND val tracking
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            # Train
            model.train()
            epoch_loss = 0.0
            epoch_n = 0
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                if math.isnan(loss.item()):
                    return {"score": float("inf"), "status": "nan"}
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)
                epoch_n += images.size(0)
            scheduler.step()
            train_losses.append(epoch_loss / epoch_n)

            # Validate after every epoch
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
            "train_loss_final": train_losses[-1],
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
        }

    return train_fn


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous LLM+PSO hyperparameter search (FashionMNIST)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-particles", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", type=str, default="experiments.tsv")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["claude", "openai", "qwen", "none"],
                        help="LLM backend (default: auto-detect)")
    args = parser.parse_args()

    # Device auto-detection
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Backend selection
    if args.backend is not None:
        backend = get_backend_by_name(args.backend)
    else:
        backend = get_default_backend()

    backend_name = backend.name if backend else "None (pure PSO)"
    print(f"Device: {device}")
    print(f"Backend: {backend_name}")
    print(f"Subset: {args.subset_size} train, {args.epochs} epochs/eval")
    print()

    train_fn = make_train_fn(args.data_dir, args.subset_size,
                             args.batch_size, args.epochs)

    search_space = {
        "lr": LogUniform(1e-4, 1e-1),
        "wd": LogUniform(1e-6, 1e-2),
    }

    runner = AutonomousRunner(
        train_fn=train_fn,
        search_space=search_space,
        log_path=args.log,
        backend=backend,
        n_particles=args.n_particles,
        timeout=args.timeout,
        device=device,
        seed=args.seed,
    )

    runner.run()


if __name__ == "__main__":
    main()
