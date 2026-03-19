"""
Comprehensive benchmark: neuropt vs Optuna across vision, language, and tabular tasks.

Each task defines its own model, data, search space, and metrics. Runs both
neuropt (any backend) and Optuna TPE with the same budget, then compares.

Usage:
    # Run all tasks with Claude backend (15 evals each)
    python examples/benchmark_suite.py --backend claude --n-evals 15

    # Run specific tasks
    python examples/benchmark_suite.py --tasks cnn_fashion transformer_imdb xgb_covertype

    # Dry run (random backend, no LLM needed — good for testing)
    python examples/benchmark_suite.py --backend none --n-evals 5

    # Multiple runs for statistical significance
    python examples/benchmark_suite.py --backend claude --n-evals 15 --n-runs 3

Results are saved to benchmark_suite_results.json.
"""

import argparse
import json
import math
import os
import random
import time
import warnings

import numpy as np


# ── Device detection ─────────────────────────────────────────────────────

def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ══════════════════════════════════════════════════════════════════════════
#  TASK DEFINITIONS
#  Each task is a dict with: name, search_space, train_fn_factory,
#  optuna_space_fn, minimize, description
# ══════════════════════════════════════════════════════════════════════════

def make_tasks(device, n_evals):
    """Build all benchmark tasks. Data is loaded lazily inside factories."""
    tasks = {}

    # ── Vision: CNN on FashionMNIST ──────────────────────────────────────

    def cnn_fashion_factory():
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as T
        from torch.utils.data import DataLoader, Subset

        transform = T.Compose([T.Grayscale(3), T.Resize(32), T.ToTensor(),
                                T.Normalize([0.5]*3, [0.5]*3)])
        train_ds = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        rng = np.random.default_rng(0)
        train_loader = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 5000, replace=False)),
                                  batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(val_ds, rng.choice(len(val_ds), 1250, replace=False)),
                                batch_size=128, shuffle=False, num_workers=0)

        ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "leaky_relu": nn.LeakyReLU}

        def train_fn(config):
            act_cls = ACTS.get(config.get("activation", "relu"), nn.ReLU)
            layers = []
            ch = 3
            for i in range(config["n_blocks"]):
                out = min(256, max(8, int(config["base_channels"] * config["channel_growth"] ** i)))
                layers.extend([nn.Conv2d(ch, out, 3, padding=1), nn.BatchNorm2d(out), act_cls()])
                if config["dropout"] > 0:
                    layers.append(nn.Dropout2d(config["dropout"]))
                if (i + 1) % 2 == 0:
                    layers.append(nn.MaxPool2d(2))
                ch = out
            layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(ch, 10)])
            model = nn.Sequential(*layers).to(device)

            opt_map = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
            opt_cls = opt_map.get(config.get("optimizer", "adamw"), torch.optim.AdamW)
            opt_kw = {"lr": config["lr"], "weight_decay": config["wd"]}
            if config.get("optimizer") == "sgd":
                opt_kw["momentum"] = 0.9
            opt = opt_cls(model.parameters(), **opt_kw)
            crit = nn.CrossEntropyLoss()

            train_losses, val_losses, val_accs = [], [], []
            for _ in range(5):
                model.train()
                tl, tn = 0.0, 0
                for imgs, tgts in train_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    opt.zero_grad()
                    loss = crit(model(imgs), tgts)
                    if math.isnan(loss.item()):
                        return {"score": float("inf")}
                    loss.backward(); opt.step()
                    tl += loss.item() * imgs.size(0); tn += imgs.size(0)
                train_losses.append(tl / tn)

                model.eval()
                vl, vc, vn = 0.0, 0, 0
                with torch.no_grad():
                    for imgs, tgts in val_loader:
                        imgs, tgts = imgs.to(device), tgts.to(device)
                        out = model(imgs)
                        vl += crit(out, tgts).item() * imgs.size(0)
                        vc += (out.argmax(1) == tgts).sum().item(); vn += imgs.size(0)
                val_losses.append(vl / vn); val_accs.append(vc / vn)

            return {
                "score": val_losses[-1], "accuracy": val_accs[-1],
                "n_params": sum(p.numel() for p in model.parameters()),
                "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accs,
            }
        return train_fn

    tasks["cnn_fashion"] = {
        "name": "CNN / FashionMNIST",
        "description": "CNN architecture search (8 params) on FashionMNIST subset",
        "minimize": True,
        "search_space": {
            "n_blocks": (2, 7), "base_channels": (16, 64), "channel_growth": (1.0, 2.0),
            "activation": ["relu", "gelu", "silu", "leaky_relu"],
            "dropout": (0.0, 0.4), "lr": (1e-4, 0.05), "wd": (1e-6, 1e-2),
            "optimizer": ["sgd", "adam", "adamw"],
        },
        "train_fn_factory": cnn_fashion_factory,
        "optuna_space": lambda trial: {
            "n_blocks": trial.suggest_int("n_blocks", 2, 7),
            "base_channels": trial.suggest_int("base_channels", 16, 64),
            "channel_growth": trial.suggest_float("channel_growth", 1.0, 2.0),
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu", "leaky_relu"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 0.05, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adam", "adamw"]),
        },
    }

    # ── Vision: ResNet fine-tuning (pretrained, minimize=True) ───────────

    def resnet_finetune_factory():
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as T
        from torch.utils.data import DataLoader, Subset

        transform = T.Compose([T.Resize(64), T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        rng = np.random.default_rng(1)
        train_loader = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 3000, replace=False)),
                                  batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(val_ds, rng.choice(len(val_ds), 1000, replace=False)),
                                batch_size=64, shuffle=False, num_workers=0)

        def train_fn(config):
            import torchvision.models as models
            model = models.resnet18(weights="DEFAULT")
            model.fc = nn.Linear(model.fc.in_features, 10)

            # Freeze strategy
            strategy = config.get("freeze_strategy", "full")
            if strategy == "head_only":
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.fc.parameters():
                    p.requires_grad = True
            elif strategy == "last_two":
                for p in model.parameters():
                    p.requires_grad = False
                for p in list(model.layer4.parameters()) + list(model.fc.parameters()):
                    p.requires_grad = True

            model = model.to(device)
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config["lr"], weight_decay=config["wd"],
            )
            crit = nn.CrossEntropyLoss()

            train_losses, val_losses, val_accs = [], [], []
            for _ in range(3):
                model.train()
                tl, tn = 0.0, 0
                for imgs, tgts in train_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    opt.zero_grad()
                    loss = crit(model(imgs), tgts)
                    if math.isnan(loss.item()):
                        return {"score": float("inf")}
                    loss.backward(); opt.step()
                    tl += loss.item() * imgs.size(0); tn += imgs.size(0)
                train_losses.append(tl / tn)

                model.eval()
                vl, vc, vn = 0.0, 0, 0
                with torch.no_grad():
                    for imgs, tgts in val_loader:
                        imgs, tgts = imgs.to(device), tgts.to(device)
                        out = model(imgs)
                        vl += crit(out, tgts).item() * imgs.size(0)
                        vc += (out.argmax(1) == tgts).sum().item(); vn += imgs.size(0)
                val_losses.append(vl / vn); val_accs.append(vc / vn)

            return {
                "score": val_losses[-1], "accuracy": val_accs[-1],
                "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accs,
            }
        return train_fn

    tasks["resnet_cifar"] = {
        "name": "ResNet18 / CIFAR-10 (fine-tune)",
        "description": "Fine-tuning pretrained ResNet18 on CIFAR-10 subset (5 params)",
        "minimize": True,
        "search_space": {
            "freeze_strategy": ["full", "head_only", "last_two"],
            "lr": (1e-5, 1e-2),
            "wd": (1e-6, 1e-2),
            "optimizer": ["adam", "adamw"],
        },
        "train_fn_factory": resnet_finetune_factory,
        "optuna_space": lambda trial: {
            "freeze_strategy": trial.suggest_categorical("freeze_strategy", ["full", "head_only", "last_two"]),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        },
    }

    # ── Vision: ViT fine-tuning ────────────────────────────────────────

    def vit_finetune_factory():
        import torch
        import torch.nn as nn
        import torchvision
        import torchvision.transforms as T
        from torch.utils.data import DataLoader, Subset

        # ViT-B/16 expects 224x224 but we use a small subset to keep evals fast
        transform = T.Compose([T.Resize(224), T.ToTensor(),
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        val_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        rng = np.random.default_rng(2)
        train_loader = DataLoader(Subset(train_ds, rng.choice(len(train_ds), 1000, replace=False)),
                                  batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(Subset(val_ds, rng.choice(len(val_ds), 300, replace=False)),
                                batch_size=16, shuffle=False, num_workers=0)

        def train_fn(config):
            from torchvision.models import vit_b_16, ViT_B_16_Weights

            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            model.heads.head = nn.Linear(model.heads.head.in_features, 10)

            strategy = config.get("freeze_strategy", "full")
            if strategy == "head_only":
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.heads.parameters():
                    p.requires_grad = True
            elif strategy == "last_blocks":
                for p in model.parameters():
                    p.requires_grad = False
                # Unfreeze last 2 encoder blocks + head
                for p in model.encoder.layers[-2:].parameters():
                    p.requires_grad = True
                for p in model.heads.parameters():
                    p.requires_grad = True

            model = model.to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=config["wd"])
            crit = nn.CrossEntropyLoss()

            train_losses, val_losses, val_accs = [], [], []
            for _ in range(config.get("epochs", 3)):
                model.train()
                tl, tn = 0.0, 0
                for imgs, tgts in train_loader:
                    imgs, tgts = imgs.to(device), tgts.to(device)
                    opt.zero_grad()
                    loss = crit(model(imgs), tgts)
                    if math.isnan(loss.item()):
                        return {"score": float("inf")}
                    loss.backward(); opt.step()
                    tl += loss.item() * imgs.size(0); tn += imgs.size(0)
                train_losses.append(tl / tn)

                model.eval()
                vl, vc, vn = 0.0, 0, 0
                with torch.no_grad():
                    for imgs, tgts in val_loader:
                        imgs, tgts = imgs.to(device), tgts.to(device)
                        out = model(imgs)
                        vl += crit(out, tgts).item() * imgs.size(0)
                        vc += (out.argmax(1) == tgts).sum().item(); vn += imgs.size(0)
                val_losses.append(vl / vn); val_accs.append(vc / vn)

            return {
                "score": val_losses[-1], "accuracy": val_accs[-1],
                "n_params": sum(p.numel() for p in trainable),
                "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accs,
            }
        return train_fn

    tasks["vit_cifar"] = {
        "name": "ViT-B/16 / CIFAR-10 (fine-tune)",
        "description": "Fine-tuning pretrained ViT-B/16 on CIFAR-10 subset (5 params)",
        "minimize": True,
        "search_space": {
            "freeze_strategy": ["full", "head_only", "last_blocks"],
            "lr": (1e-6, 1e-3),
            "wd": (1e-6, 1e-2),
            "optimizer": ["adam", "adamw"],
            "epochs": (2, 5),
        },
        "train_fn_factory": vit_finetune_factory,
        "optuna_space": lambda trial: {
            "freeze_strategy": trial.suggest_categorical("freeze_strategy", ["full", "head_only", "last_blocks"]),
            "lr": trial.suggest_float("lr", 1e-6, 1e-3, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            "epochs": trial.suggest_int("epochs", 2, 5),
        },
    }

    # ── Language: LLM fine-tuning (DistilBERT on sentiment) ─────────────

    def llm_sentiment_factory():
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "LLM benchmark requires: pip install transformers datasets"
            )

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Load SST-2 sentiment dataset (binary)
        ds = load_dataset("glue", "sst2", trust_remote_code=True)
        rng_np = np.random.default_rng(42)

        # Subsample for speed
        train_idx = rng_np.choice(len(ds["train"]), 2000, replace=False)
        val_idx = rng_np.choice(len(ds["validation"]), 500, replace=False)

        def tokenize_subset(split, indices):
            texts = [split[int(i)]["sentence"] for i in indices]
            labels = [split[int(i)]["label"] for i in indices]
            enc = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt")
            return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

        train_ids, train_mask, train_labels = tokenize_subset(ds["train"], train_idx)
        val_ids, val_mask, val_labels = tokenize_subset(ds["validation"], val_idx)

        train_loader = DataLoader(
            TensorDataset(train_ids, train_mask, train_labels),
            batch_size=32, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(val_ids, val_mask, val_labels),
            batch_size=32, shuffle=False)

        def train_fn(config):
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2)

            strategy = config.get("freeze_strategy", "full")
            if strategy == "embeddings_only":
                # Freeze everything, unfreeze only classifier head
                for p in model.parameters():
                    p.requires_grad = False
                for p in model.classifier.parameters():
                    p.requires_grad = True
                for p in model.pre_classifier.parameters():
                    p.requires_grad = True
            elif strategy == "last_layers":
                for p in model.parameters():
                    p.requires_grad = False
                # Unfreeze last 2 transformer layers + classifier
                for p in model.distilbert.transformer.layer[-2:].parameters():
                    p.requires_grad = True
                for p in model.classifier.parameters():
                    p.requires_grad = True
                for p in model.pre_classifier.parameters():
                    p.requires_grad = True

            model = model.to(device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=config["wd"])

            train_losses, val_losses, val_accs = [], [], []
            for _ in range(config.get("epochs", 3)):
                model.train()
                tl, tn = 0.0, 0
                for ids, mask, labels in train_loader:
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    opt.zero_grad()
                    out = model(input_ids=ids, attention_mask=mask, labels=labels)
                    out.loss.backward(); opt.step()
                    tl += out.loss.item() * ids.size(0); tn += ids.size(0)
                train_losses.append(tl / tn)

                model.eval()
                vl, vc, vn = 0.0, 0, 0
                with torch.no_grad():
                    for ids, mask, labels in val_loader:
                        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                        out = model(input_ids=ids, attention_mask=mask, labels=labels)
                        vl += out.loss.item() * ids.size(0)
                        preds = out.logits.argmax(dim=-1)
                        vc += (preds == labels).sum().item(); vn += ids.size(0)
                val_losses.append(vl / vn); val_accs.append(vc / vn)

            return {
                "score": val_accs[-1],  # maximize accuracy
                "val_loss": val_losses[-1],
                "n_params": sum(p.numel() for p in trainable),
                "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accs,
            }
        return train_fn

    tasks["llm_sentiment"] = {
        "name": "DistilBERT / SST-2 (fine-tune)",
        "description": "Fine-tuning DistilBERT on SST-2 sentiment, maximize accuracy (5 params). "
                       "Requires: pip install transformers datasets",
        "minimize": False,  # maximize accuracy
        "search_space": {
            "freeze_strategy": ["full", "embeddings_only", "last_layers"],
            "lr": (1e-6, 5e-4),
            "wd": (1e-6, 1e-2),
            "optimizer": ["adam", "adamw"],
            "epochs": (1, 4),
        },
        "train_fn_factory": llm_sentiment_factory,
        "optuna_space": lambda trial: {
            "freeze_strategy": trial.suggest_categorical("freeze_strategy", ["full", "embeddings_only", "last_layers"]),
            "lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            "epochs": trial.suggest_int("epochs", 1, 4),
        },
    }

    # ── Language: Text classification with embeddings ────────────────────

    def text_cls_factory():
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Synthetic text classification: bag-of-embeddings → linear
        # Simulates IMDB-like sentiment but fast (no tokenizer dependency)
        rng_np = np.random.default_rng(42)
        vocab_size, seq_len, n_train, n_val = 5000, 64, 4000, 1000

        # Generate synthetic sequences with class-correlated patterns
        def make_data(n):
            labels = rng_np.integers(0, 2, n)
            seqs = rng_np.integers(1, vocab_size, (n, seq_len))
            # Class 1 has higher-index tokens on average (a learnable signal)
            for i in range(n):
                if labels[i] == 1:
                    seqs[i, :seq_len//4] = rng_np.integers(vocab_size//2, vocab_size, seq_len//4)
            return seqs, labels

        train_x, train_y = make_data(n_train)
        val_x, val_y = make_data(n_val)
        import torch
        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_x, dtype=torch.long), torch.tensor(train_y, dtype=torch.long)),
            batch_size=128, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(torch.tensor(val_x, dtype=torch.long), torch.tensor(val_y, dtype=torch.long)),
            batch_size=128, shuffle=False)

        def train_fn(config):
            embed_dim = config["embed_dim"]
            hidden = config["hidden_dim"]
            n_layers = config["n_layers"]

            layers = [nn.Embedding(vocab_size, embed_dim)]
            in_dim = embed_dim
            for _ in range(n_layers):
                layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
                if config["dropout"] > 0:
                    layers.append(nn.Dropout(config["dropout"]))
                in_dim = hidden
            layers.append(nn.Linear(in_dim, 2))

            class TextModel(nn.Module):
                def __init__(self, layers):
                    super().__init__()
                    self.embed = layers[0]
                    self.classifier = nn.Sequential(*layers[1:])
                def forward(self, x):
                    return self.classifier(self.embed(x).mean(dim=1))

            model = TextModel(layers).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
            crit = nn.CrossEntropyLoss()

            train_losses, val_losses, val_accs = [], [], []
            for _ in range(config.get("epochs", 8)):
                model.train()
                tl, tn = 0.0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward(); opt.step()
                    tl += loss.item() * x.size(0); tn += x.size(0)
                train_losses.append(tl / tn)

                model.eval()
                vl, vc, vn = 0.0, 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        vl += crit(out, y).item() * x.size(0)
                        vc += (out.argmax(1) == y).sum().item(); vn += x.size(0)
                val_losses.append(vl / vn); val_accs.append(vc / vn)

            return {
                "score": val_accs[-1],  # maximize accuracy
                "val_loss": val_losses[-1],
                "train_losses": train_losses, "val_losses": val_losses, "val_accuracies": val_accs,
            }
        return train_fn

    tasks["text_cls"] = {
        "name": "Text classification (synthetic)",
        "description": "Bag-of-embeddings text classifier, maximize accuracy (7 params)",
        "minimize": False,  # maximize accuracy
        "search_space": {
            "embed_dim": (16, 128), "hidden_dim": (32, 256), "n_layers": (1, 4),
            "dropout": (0.0, 0.5), "lr": (1e-4, 1e-2), "wd": (1e-6, 1e-2),
            "epochs": (3, 12),
        },
        "train_fn_factory": text_cls_factory,
        "optuna_space": lambda trial: {
            "embed_dim": trial.suggest_int("embed_dim", 16, 128),
            "hidden_dim": trial.suggest_int("hidden_dim", 32, 256),
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 3, 12),
        },
    }

    # ── Tabular: XGBoost on Covertype (multiclass) ──────────────────────

    def xgb_covertype_factory():
        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import train_test_split

        X, y = fetch_covtype(return_X_y=True)
        # Subsample for speed
        rng_np = np.random.default_rng(42)
        idx = rng_np.choice(len(X), 20000, replace=False)
        X, y = X[idx], y[idx]
        y = y - 1  # 0-indexed
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        def train_fn(config):
            from xgboost import XGBClassifier
            from sklearn.metrics import log_loss, accuracy_score

            model = XGBClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"],
                min_child_weight=config["min_child_weight"],
                verbosity=0,
                eval_metric="mlogloss",
                random_state=42,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            r = model.evals_result()
            val_losses = r["validation_0"]["mlogloss"]
            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)

            return {
                "score": val_losses[-1],
                "accuracy": accuracy_score(y_val, preds),
                "val_losses": val_losses,
                "n_classes": 7,
                "n_train": len(X_train),
            }
        return train_fn

    tasks["xgb_covertype"] = {
        "name": "XGBoost / Covertype (7-class)",
        "description": "XGBoost multiclass on Covertype subset, 8 hyperparams",
        "minimize": True,
        "search_space": {
            "n_estimators": (50, 500), "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3), "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0), "reg_alpha": (1e-5, 10.0),
            "reg_lambda": (1e-5, 10.0), "min_child_weight": (1, 10),
        },
        "train_fn_factory": xgb_covertype_factory,
        "optuna_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        },
    }

    # ── Tabular: Random Forest on Breast Cancer (binary, maximize AUROC) ─

    def rf_breast_cancer_factory():
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score

        X, y = load_breast_cancer(return_X_y=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        def train_fn(config):
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_split=config["min_samples_split"],
                min_samples_leaf=config["min_samples_leaf"],
                max_features=config["max_features"],
                random_state=42,
            )
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_val)[:, 1]
            preds = model.predict(X_val)

            return {
                "score": roc_auc_score(y_val, proba),  # maximize
                "accuracy": accuracy_score(y_val, preds),
                "n_train": len(X_train),
            }
        return train_fn

    tasks["rf_breast_cancer"] = {
        "name": "RandomForest / Breast Cancer (AUROC)",
        "description": "Random Forest binary classification, maximize AUROC (5 params)",
        "minimize": False,  # maximize AUROC
        "search_space": {
            "n_estimators": (50, 500), "max_depth": (2, 20),
            "min_samples_split": (2, 20), "min_samples_leaf": (1, 10),
            "max_features": (0.3, 1.0),
        },
        "train_fn_factory": rf_breast_cancer_factory,
        "optuna_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        },
    }

    # ── Tabular: LightGBM on California Housing (regression, minimize MSE)

    def lgbm_housing_factory():
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        def train_fn(config):
            from lightgbm import LGBMRegressor

            model = LGBMRegressor(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                num_leaves=config["num_leaves"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                reg_alpha=config["reg_alpha"],
                reg_lambda=config["reg_lambda"],
                verbose=-1,
                random_state=42,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)
            r2 = r2_score(y_val, preds)

            return {
                "score": mse,  # minimize MSE
                "r2": r2,
                "n_train": len(X_train),
            }
        return train_fn

    tasks["lgbm_housing"] = {
        "name": "LightGBM / California Housing (MSE)",
        "description": "LightGBM regression, minimize MSE (8 params)",
        "minimize": True,
        "search_space": {
            "n_estimators": (50, 500), "max_depth": (3, 12),
            "learning_rate": (0.01, 0.3), "num_leaves": (15, 127),
            "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0),
            "reg_alpha": (1e-5, 10.0), "reg_lambda": (1e-5, 10.0),
        },
        "train_fn_factory": lgbm_housing_factory,
        "optuna_space": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
        },
    }

    return tasks


# ══════════════════════════════════════════════════════════════════════════
#  RUNNERS
# ══════════════════════════════════════════════════════════════════════════

def run_neuropt(task, backend, n_evals):
    from neuropt import ArchSearch

    log_path = f"/tmp/bench_suite_{task['name'].replace(' ', '_').replace('/', '_')}.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    train_fn = task["train_fn_factory"]()

    search = ArchSearch(
        train_fn=train_fn,
        search_space=task["search_space"],
        backend=backend,
        log_path=log_path,
        batch_size=3,
        timeout=300,
        minimize=task["minimize"],
    )
    t0 = time.time()
    search.run(max_evals=n_evals)
    wall_time = time.time() - t0

    # Read back log
    with open(log_path) as f:
        results = [json.loads(line) for line in f]

    scores = []
    for r in results:
        s = r.get("score", r.get("val_loss"))
        if s is not None and r.get("status") == "ok":
            scores.append(s)

    if task["minimize"]:
        best_so_far = [min(scores[:i+1]) for i in range(len(scores))]
    else:
        best_so_far = [max(scores[:i+1]) for i in range(len(scores))]

    return {
        "scores": scores,
        "best_so_far": best_so_far,
        "best_score": best_so_far[-1] if best_so_far else None,
        "wall_time": wall_time,
        "n_evals": len(scores),
    }


def run_optuna(task, n_evals):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_fn = task["train_fn_factory"]()
    scores = []

    direction = "minimize" if task["minimize"] else "maximize"

    def objective(trial):
        cfg = task["optuna_space"](trial)
        result = train_fn(cfg)
        score = result["score"]
        scores.append(score)
        print(f"  Optuna [{len(scores)}/{n_evals}] score={score:.4f}")
        return score

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(3, n_evals)),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_evals)
    wall_time = time.time() - t0

    if task["minimize"]:
        best_so_far = [min(scores[:i+1]) for i in range(len(scores))]
    else:
        best_so_far = [max(scores[:i+1]) for i in range(len(scores))]

    return {
        "scores": scores,
        "best_so_far": best_so_far,
        "best_score": best_so_far[-1] if best_so_far else None,
        "wall_time": wall_time,
        "n_evals": len(scores),
    }


def run_random(task, n_evals):
    from neuropt import ArchSearch

    log_path = f"/tmp/bench_suite_random_{task['name'].replace(' ', '_').replace('/', '_')}.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    train_fn = task["train_fn_factory"]()

    search = ArchSearch(
        train_fn=train_fn,
        search_space=task["search_space"],
        backend="none",
        log_path=log_path,
        batch_size=3,
        timeout=300,
        minimize=task["minimize"],
    )
    t0 = time.time()
    search.run(max_evals=n_evals)
    wall_time = time.time() - t0

    with open(log_path) as f:
        results = [json.loads(line) for line in f]

    scores = []
    for r in results:
        s = r.get("score", r.get("val_loss"))
        if s is not None and r.get("status") == "ok":
            scores.append(s)

    if task["minimize"]:
        best_so_far = [min(scores[:i+1]) for i in range(len(scores))]
    else:
        best_so_far = [max(scores[:i+1]) for i in range(len(scores))]

    return {
        "scores": scores,
        "best_so_far": best_so_far,
        "best_score": best_so_far[-1] if best_so_far else None,
        "wall_time": wall_time,
        "n_evals": len(scores),
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark suite: neuropt vs Optuna vs Random")
    parser.add_argument("--backend", default="claude", help="LLM backend (claude, openai, qwen, none)")
    parser.add_argument("--n-evals", type=int, default=15, help="Evaluations per method per task")
    parser.add_argument("--n-runs", type=int, default=1, help="Runs per method for statistics")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Task names to run (default: all). Options: "
                             "cnn_fashion, resnet_cifar, vit_cifar, llm_sentiment, "
                             "text_cls, xgb_covertype, rf_breast_cancer, lgbm_housing")
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--output", default="benchmark_suite_results.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Backend: {args.backend}")
    print(f"Budget: {args.n_evals} evals × {args.n_runs} runs per method")
    print()

    all_tasks = make_tasks(device, args.n_evals)

    # Filter tasks
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = list(all_tasks.keys())

    all_results = {}

    for task_name in task_names:
        if task_name not in all_tasks:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task = all_tasks[task_name]
        direction = "minimize" if task["minimize"] else "maximize"
        print("=" * 70)
        print(f"  {task['name']} ({direction})")
        print(f"  {task['description']}")
        print("=" * 70)
        print()

        task_results = {"task": task["name"], "minimize": task["minimize"], "methods": {}}

        for run_idx in range(args.n_runs):
            run_label = f" (run {run_idx+1}/{args.n_runs})" if args.n_runs > 1 else ""

            # neuropt
            print(f"--- neuropt ({args.backend}){run_label} ---")
            try:
                nr = run_neuropt(task, args.backend, args.n_evals)
                task_results["methods"].setdefault("neuropt", []).append(nr)
                print(f"  Best: {nr['best_score']:.4f} in {nr['wall_time']:.1f}s")
            except Exception as e:
                print(f"  Failed: {e}")

            # Optuna
            if not args.skip_optuna:
                print(f"--- Optuna TPE{run_label} ---")
                try:
                    opr = run_optuna(task, args.n_evals)
                    task_results["methods"].setdefault("optuna", []).append(opr)
                    print(f"  Best: {opr['best_score']:.4f} in {opr['wall_time']:.1f}s")
                except Exception as e:
                    print(f"  Failed: {e}")

            # Random
            if not args.skip_random:
                print(f"--- Random{run_label} ---")
                rr = run_random(task, args.n_evals)
                task_results["methods"].setdefault("random", []).append(rr)
                print(f"  Best: {rr['best_score']:.4f} in {rr['wall_time']:.1f}s")

            print()

        all_results[task_name] = task_results

    # ── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for task_name, tr in all_results.items():
        direction = "↓" if tr["minimize"] else "↑"
        print(f"\n{tr['task']} ({direction})")
        print(f"  {'Method':<15} {'Best':>10} {'Mean±Std':>18} {'Time':>10}")
        print(f"  {'-'*55}")

        for method, runs in tr["methods"].items():
            bests = [r["best_score"] for r in runs if r["best_score"] is not None]
            times = [r["wall_time"] for r in runs]
            if bests:
                mean_b = np.mean(bests)
                std_b = np.std(bests) if len(bests) > 1 else 0
                mean_t = np.mean(times)
                best_single = min(bests) if tr["minimize"] else max(bests)
                print(f"  {method:<15} {best_single:>10.4f} {mean_b:>10.4f}±{std_b:.4f} {mean_t:>9.1f}s")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
