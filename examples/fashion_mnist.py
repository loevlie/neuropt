"""
Full working example: PSO hyperparameter search for ResNet-18 on FashionMNIST.

Usage:
    python examples/fashion_mnist.py
    python examples/fashion_mnist.py --n-particles 2 --n-iters 2   # smoke test
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from swarmopt import SwarmTuner, LogUniform


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
    """Create a training function that captures the data config."""

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

        model.train()
        for epoch in range(epochs):
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        val_loss, n = 0.0, 0
        correct = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                out = model(images)
                val_loss += criterion(out, targets).item() * images.size(0)
                correct += (out.argmax(1) == targets).sum().item()
                n += images.size(0)

        return {
            "score": val_loss / n,
            "accuracy": correct / n,
            "model": model.state_dict(),
        }

    return train_fn


def main():
    parser = argparse.ArgumentParser(
        description="PSO search for lr & wd using swarmopt")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--subset-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-particles", type=int, default=5)
    parser.add_argument("--n-iters", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Subset: {args.subset_size} train, {args.epochs} epochs/eval\n")

    train_fn = make_train_fn(args.data_dir, args.subset_size,
                             args.batch_size, args.epochs)

    tuner = SwarmTuner(
        train_fn=train_fn,
        search_space={
            "lr": LogUniform(1e-4, 1e-1),
            "wd": LogUniform(1e-6, 1e-2),
        },
        n_particles=args.n_particles,
        n_iterations=args.n_iters,
        device=device,
        seed=args.seed,
    )

    tuner.fit()

    print(f"\nbest_params: {tuner.best_params}")
    print(f"best_score:  {tuner.best_score:.4f}")
    print(f"results shape: {tuner.results.shape}")

    tuner.plot()
    tuner.animate()


if __name__ == "__main__":
    main()
