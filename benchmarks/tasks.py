"""Benchmark tasks with fixed train / val / test splits.

``task.build(device)`` loads the data once and returns ``evaluate(config, seed)``, which trains one
config and returns two dicts:

- ``public``: what an optimizer may see. Validation loss as ``score``, validation ``accuracy``, and
  per-epoch ``train_losses`` / ``val_losses`` curves.
- ``private``: ``test_score`` and ``test_accuracy``. The harness logs these and never passes them on.

Splits depend only on fixed data seeds, never on the run seed.
"""

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

DATA_DIR = os.environ.get("NEUROPT_BENCH_DATA", os.path.join(os.path.dirname(__file__), "data"))
N_THREADS = int(os.environ.get("NEUROPT_BENCH_THREADS", "8"))


@dataclass
class Task:
    name: str
    title: str
    space: dict
    default: dict
    n_classes: int
    build: Callable
    uses_torch: bool = True

    @property
    def fail_score(self):
        """Score of a failed eval: the loss of a uniform guess."""
        return math.log(self.n_classes)


# ── Shared PyTorch training loop ─────────────────────────────────────────


def _torch_evaluate(make_model, splits, epochs, default_batch_size=128):
    import torch
    import torch.nn as nn

    (x_train, y_train), val, test = splits
    crit = nn.CrossEntropyLoss()

    def score(model, x, y):
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(x), 250):
                out = model(x[i:i + 250])
                loss += crit(out, y[i:i + 250]).item() * len(out)
                correct += (out.argmax(1) == y[i:i + 250]).sum().item()
        return loss / len(x), correct / len(x)

    def evaluate(cfg, seed):
        torch.manual_seed(seed)
        model = make_model(cfg).to(x_train.device)
        params, lr, wd = model.parameters(), cfg["lr"], cfg["wd"]
        if cfg["optimizer"] == "sgd":
            opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        elif cfg["optimizer"] == "adam":
            opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        else:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        batch = int(cfg.get("batch_size", default_batch_size))
        order = torch.Generator().manual_seed(seed)
        n = len(x_train)

        train_losses, val_losses, val_accs = [], [], []
        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n, generator=order).to(x_train.device)
            total = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                if len(idx) < 2:  # batch norm can't train on a single sample
                    continue
                opt.zero_grad()
                loss = crit(model(x_train[idx]), y_train[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            train_losses.append(total / n)
            vl, va = score(model, *val)
            val_losses.append(vl)
            val_accs.append(va)

        test_loss, test_acc = score(model, *test)
        public = {"score": val_losses[-1], "accuracy": val_accs[-1], "train_losses": train_losses,
                  "val_losses": val_losses, "val_accuracies": val_accs}
        return public, {"test_score": test_loss, "test_accuracy": test_acc}

    return evaluate


def _activation(name):
    import torch.nn as nn
    return {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU, "silu": nn.SiLU,
            "tanh": nn.Tanh}[name]


# ── CNN architecture search on FashionMNIST ──────────────────────────────

CNN_SPACE = {
    "n_blocks": (2, 8), "base_channels": (16, 128), "channel_growth": (1.0, 2.5),
    "kernel_size": [3, 5], "activation": ["relu", "gelu", "leaky_relu", "silu"],
    "use_residual": [True, False], "use_batchnorm": [True, False],
    "dropout": (0.0, 0.5), "pool_every": (1, 4), "pool_type": ["max", "avg"],
    "fc_hidden": (0, 512), "lr": (1e-4, 0.1), "wd": (1e-6, 0.01),
    "optimizer": ["sgd", "adam", "adamw"],
}
CNN_DEFAULT = {
    "n_blocks": 4, "base_channels": 32, "channel_growth": 2.0, "kernel_size": 3,
    "activation": "relu", "use_residual": False, "use_batchnorm": True, "dropout": 0.1,
    "pool_every": 1, "pool_type": "max", "fc_hidden": 128, "lr": 1e-3, "wd": 1e-4,
    "optimizer": "adamw",
}


def _build_cnn(device):
    import torch
    import torch.nn as nn
    import torchvision

    def load(train):
        ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=train, download=True)
        x = ds.data.float().div(255).unsqueeze(1)
        x = nn.functional.interpolate(x, size=32, mode="bilinear", align_corners=False)
        return ((x - 0.5) / 0.5).repeat(1, 3, 1, 1), ds.targets.clone()

    x, y = load(train=True)
    idx = torch.from_numpy(np.random.default_rng(0).permutation(len(x))[:6250])
    tr, va = idx[:5000], idx[5000:]  # validation comes from the training split
    x_test, y_test = load(train=False)  # the official test split is the held-out test set
    splits = [(a.to(device), b.to(device))
              for a, b in ((x[tr], y[tr]), (x[va], y[va]), (x_test, y_test))]

    class CNN(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            act = _activation(cfg["activation"])
            self.blocks, self.projs, self.pools = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            self.use_res = cfg["use_residual"]
            prev, spatial = 3, 32
            for i in range(cfg["n_blocks"]):
                out = min(512, max(8, int(cfg["base_channels"] * cfg["channel_growth"] ** i)))
                k = cfg["kernel_size"]
                layers = [nn.Conv2d(prev, out, k, padding=k // 2, bias=not cfg["use_batchnorm"])]
                if cfg["use_batchnorm"]:
                    layers.append(nn.BatchNorm2d(out))
                layers.append(act())
                if cfg["dropout"] > 0:
                    layers.append(nn.Dropout2d(cfg["dropout"]))
                self.blocks.append(nn.Sequential(*layers))
                if not self.use_res:
                    self.projs.append(None)
                elif prev != out:
                    self.projs.append(nn.Conv2d(prev, out, 1, bias=False))
                else:
                    self.projs.append(nn.Identity())
                if (i + 1) % cfg["pool_every"] == 0 and spatial > 2:
                    self.pools.append(nn.MaxPool2d(2) if cfg["pool_type"] == "max" else nn.AvgPool2d(2))
                    spatial //= 2
                else:
                    self.pools.append(None)
                prev = out
            self.gap = nn.AdaptiveAvgPool2d(1)
            if cfg["fc_hidden"] > 0:
                self.head = nn.Sequential(nn.Linear(prev, cfg["fc_hidden"]), act(),
                                          nn.Dropout(cfg["dropout"]), nn.Linear(cfg["fc_hidden"], 10))
            else:
                self.head = nn.Linear(prev, 10)

        def forward(self, x):
            for blk, proj, pool in zip(self.blocks, self.projs, self.pools):
                h = blk(x)
                if proj is not None:
                    h = h + proj(x)
                x = pool(h) if pool is not None else h
            return self.head(self.gap(x).flatten(1))

    return _torch_evaluate(CNN, splits, epochs=5)


# ── XGBoost on Covertype ─────────────────────────────────────────────────

XGB_SPACE = {
    "max_depth": (3, 12), "learning_rate": (1e-3, 0.3), "n_estimators": (50, 500),
    "min_child_weight": (1, 10), "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (1e-5, 10.0), "reg_lambda": (1e-5, 10.0), "gamma": (1e-5, 10.0),
}
# XGBoost's library defaults; reg_alpha and gamma default to 0, below their log range.
XGB_DEFAULT = {
    "max_depth": 6, "learning_rate": 0.3, "n_estimators": 100, "min_child_weight": 1,
    "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 1e-5, "reg_lambda": 1.0, "gamma": 1e-5,
}


def _build_xgb(device):
    from sklearn.datasets import fetch_covtype
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier

    X, y = fetch_covtype(return_X_y=True, data_home=DATA_DIR)
    y = LabelEncoder().fit_transform(y)
    idx = np.random.default_rng(42).permutation(len(X))[:30000]
    tr, va, te = idx[:16000], idx[16000:20000], idx[20000:]
    labels = list(range(7))

    def evaluate(cfg, seed):
        model = XGBClassifier(
            max_depth=int(cfg["max_depth"]), learning_rate=cfg["learning_rate"],
            n_estimators=int(cfg["n_estimators"]), min_child_weight=int(cfg["min_child_weight"]),
            subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=cfg["reg_alpha"], reg_lambda=cfg["reg_lambda"], gamma=cfg["gamma"],
            eval_metric="mlogloss", verbosity=0, random_state=seed, n_jobs=N_THREADS,
        )
        model.fit(X[tr], y[tr], eval_set=[(X[tr], y[tr]), (X[va], y[va])], verbose=False)
        curves = model.evals_result()
        val_losses = curves["validation_1"]["mlogloss"]
        public = {"score": val_losses[-1], "accuracy": float(model.score(X[va], y[va])),
                  "train_losses": curves["validation_0"]["mlogloss"], "val_losses": val_losses}
        proba = model.predict_proba(X[te])
        private = {"test_score": float(log_loss(y[te], proba, labels=labels)),
                   "test_accuracy": float((proba.argmax(1) == y[te]).mean())}
        return public, private

    return evaluate


# ── Tabular MLP on Higgs-small ───────────────────────────────────────────

MLP_SPACE = {
    "n_layers": (1, 6), "hidden_dim": (32, 1024), "activation": ["relu", "gelu", "silu", "tanh"],
    "dropout": (0.0, 0.5), "use_batchnorm": [True, False], "lr": (1e-4, 0.1), "wd": (1e-6, 0.01),
    "optimizer": ["sgd", "adam", "adamw"], "batch_size": [64, 128, 256, 512, 1024],
}
MLP_DEFAULT = {
    "n_layers": 3, "hidden_dim": 256, "activation": "relu", "dropout": 0.1, "use_batchnorm": True,
    "lr": 1e-3, "wd": 1e-5, "optimizer": "adamw", "batch_size": 256,
}


def _build_mlp(device):
    import torch
    import torch.nn as nn
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=23512, as_frame=False, data_home=DATA_DIR)  # Higgs-small
    X = data.data.astype(np.float32)
    y = (data.target.astype(float) > 0.5).astype(np.int64)
    keep = ~np.isnan(X).any(axis=1)
    X, y = X[keep], y[keep]
    idx = np.random.default_rng(0).permutation(len(X))[:45000]
    tr, va, te = idx[:20000], idx[20000:25000], idx[25000:]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-8  # scaling uses training rows only
    splits = [(torch.from_numpy((X[i] - mean) / std).to(device), torch.from_numpy(y[i]).to(device))
              for i in (tr, va, te)]
    n_features = X.shape[1]

    def make_model(cfg):
        act = _activation(cfg["activation"])
        layers, prev = [], n_features
        for _ in range(cfg["n_layers"]):
            layers.append(nn.Linear(prev, cfg["hidden_dim"]))
            if cfg["use_batchnorm"]:
                layers.append(nn.BatchNorm1d(cfg["hidden_dim"]))
            layers.append(act())
            if cfg["dropout"] > 0:
                layers.append(nn.Dropout(cfg["dropout"]))
            prev = cfg["hidden_dim"]
        layers.append(nn.Linear(prev, 2))
        return nn.Sequential(*layers)

    return _torch_evaluate(make_model, splits, epochs=10)


TASKS = {
    "cnn": Task("cnn", "CNN architecture search (FashionMNIST, 14 params)",
                CNN_SPACE, CNN_DEFAULT, 10, _build_cnn),
    "xgb": Task("xgb", "XGBoost (Covertype, 9 params)", XGB_SPACE, XGB_DEFAULT, 7, _build_xgb,
                uses_torch=False),
    "mlp": Task("mlp", "Tabular MLP (Higgs-small, 9 params)", MLP_SPACE, MLP_DEFAULT, 2, _build_mlp),
}
