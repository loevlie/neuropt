"""
Simplest possible swarmopt usage: give it a model, write a train loop.

    swarmopt run examples/train_resnet.py --backend claude

swarmopt introspects the model, figures out what to search
(activations, batch norm, dropout, lr, wd, optimizer), and runs.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


# ── Your model ───────────────────────────────────────────────────────────

model = torchvision.models.resnet18(weights=None, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()


# ── Your training loop ───────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 5

transform = T.Compose([T.Grayscale(3), T.Resize(32), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
val_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
rng = np.random.default_rng(0)
train_loader = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 5000, replace=False)),
                          batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(Subset(val_ds, rng.choice(len(val_ds), 1250, replace=False)),
                        batch_size=128, shuffle=False, num_workers=0)


def train_fn(config):
    m = config["model"].to(DEVICE)

    opt_name = config.get("optimizer", "adamw")
    if opt_name == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["wd"])
    elif opt_name == "adam":
        opt = torch.optim.Adam(m.parameters(), lr=config["lr"], weight_decay=config["wd"])
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=config["lr"], weight_decay=config["wd"])

    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    train_losses, val_losses, val_accuracies = [], [], []

    for _ in range(EPOCHS):
        m.train()
        tl, tn = 0.0, 0
        for imgs, tgts in train_loader:
            imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
            opt.zero_grad()
            loss = crit(m(imgs), tgts)
            loss.backward(); opt.step()
            tl += loss.item() * imgs.size(0); tn += imgs.size(0)
        sched.step(); train_losses.append(tl / tn)

        m.eval()
        vl, vn, vc = 0.0, 0, 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(DEVICE), tgts.to(DEVICE)
                out = m(imgs)
                vl += crit(out, tgts).item() * imgs.size(0)
                vc += (out.argmax(1) == tgts).sum().item(); vn += imgs.size(0)
        val_losses.append(vl / vn); val_accuracies.append(vc / vn)

    return {"score": val_losses[-1], "accuracy": val_accuracies[-1],
            "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accuracies}
