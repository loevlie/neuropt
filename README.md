# swarmopt

<p align="center">
  <img src="assets/banner.png" alt="Three robot researchers designing neural network architectures" width="700">
</p>

<p align="center">
  <em>An LLM reads your training curves and designs your next experiment.</em>
</p>

---

Point it at a training script, let it run overnight. The LLM sees full per-epoch train/val curves, spots overfitting, and proposes what to try next — like having a research assistant who never sleeps and actually reads the loss plots.

We let it run for 8 hours on a MacBook. Starting from random configs, it independently converged on GELU + residual connections + BatchNorm + AdamW + zero dropout — the same recipe a senior ML engineer would pick. It noticed dropout was hurting performance from the training curves and stopped using it. 1,239 experiments, zero human intervention.

## Quick start

```bash
pip install swarmopt[llm]
export ANTHROPIC_API_KEY="sk-ant-..."
```

Write a training script. You define two things: a `search_space` dict (what to search over) and a `train_fn` function (how to train). The config dict your function receives has the same keys as your search space — the LLM picks the values.

```python
# train.py
import torch.nn as nn

# Plain Python — tuples for ranges, lists for choices.
# swarmopt auto-detects the right sampling strategy from
# the param name (e.g. "lr" → log-scale, "n_layers" → integer).
search_space = {
    "lr": (1e-4, 1e-1),
    "hidden_dim": (32, 512),
    "activation": ["relu", "gelu", "silu"],
}

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}

def train_fn(config):
    # config has the keys you defined above — "lr", "hidden_dim", "activation"
    model = nn.Sequential(
        nn.Linear(784, config["hidden_dim"]),
        ACTIVATIONS[config["activation"]](),
        nn.Linear(config["hidden_dim"], 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_losses, val_losses = [], []
    for epoch in range(10):
        # ... your training loop, tracking per-epoch losses ...
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

    return {
        "score": val_losses[-1],          # required (lower is better)
        "train_losses": train_losses,     # optional but helps the LLM a lot
        "val_losses": val_losses,         # optional but helps the LLM a lot
    }
```

Run it:

```bash
swarmopt run train.py
```

Runs until Ctrl+C. Crash-safe, resumable. See [`examples/train_fashion.py`](examples/train_fashion.py) for a full working example.

## Or: just give it a model

Already have a PyTorch model? Define a `model` variable instead of `search_space`, and swarmopt walks the module tree to find what's tunable (activations, dropout, batch norm). It deep-copies your model each experiment, swaps in the modifications, and passes the modified copy to your `train_fn` as `config["model"]`.

```python
# train.py
import torch
import torch.nn as nn
import torchvision.models as models

# swarmopt will introspect this — finds 9 ReLU layers, 20 BatchNorm layers
model = models.resnet18(num_classes=10)

def train_fn(config):
    # config["model"] is a deep copy with modifications already applied
    # (e.g., all ReLUs swapped to GELUs, or BatchNorm toggled off)
    # config also has "lr", "wd", "optimizer" — always included
    m = config["model"].to("cuda")
    optimizer = torch.optim.Adam(m.parameters(), lr=config["lr"],
                                 weight_decay=config["wd"])

    train_losses, val_losses = [], []
    for epoch in range(5):
        # ... your training loop ...
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

    return {"score": val_losses[-1], "train_losses": train_losses,
            "val_losses": val_losses}
```

```bash
swarmopt run train.py
```

```
Introspected model (11,689,512 params):
  Activations: ReLU (9 layers)
  BatchNorm: 20 layers
  Search space: ['activation', 'use_batchnorm', 'lr', 'wd', 'optimizer']
```

See [`examples/train_resnet.py`](examples/train_resnet.py) for a full working version of this.

## CLI

```bash
swarmopt run train.py                  # run search (auto-detect backend)
swarmopt run train.py --backend claude # specific backend
swarmopt run train.py -b 5 --log out.jsonl  # 5 configs/batch, custom log
swarmopt inspect train.py              # show what it would search over
swarmopt results search.jsonl          # analyze a log file
swarmopt results search.jsonl --top 20 # show top 20 results
```

## What the LLM sees

Most tuning tools give the optimizer a single number: *"this config scored 0.85."* We show the full picture:

```
lr=0.05, activation=relu, use_residual=False:
  ep1:  train=2.30  val=2.28  acc=0.12
  ep2:  train=1.45  val=1.52  acc=0.41
  ep3:  train=0.82  val=1.35  acc=0.53
  ep4:  train=0.31  val=1.61  acc=0.48   ← val rising = overfitting
  ep5:  train=0.09  val=1.89  acc=0.45

lr=8.8e-4, activation=gelu, use_residual=True:
  ep1:  train=1.92  val=1.85  acc=0.28
  ep2:  train=1.01  val=0.98  acc=0.62
  ep3:  train=0.62  val=0.71  acc=0.74
  ep4:  train=0.41  val=0.52  acc=0.81   ← both dropping = good fit
  ep5:  train=0.33  val=0.43  acc=0.85
```

Plus pre-computed signals: `OVERFITTING: train 2.30→0.09, val 1.52→1.89, gap=1.80`.

## Python API

If you'd rather not use the CLI, the same functionality is available directly:

```python
from swarmopt import ArchSearch, LogUniform, Categorical

# With a manual search space (same train_fn and search_space as your script)
search = ArchSearch(
    train_fn=train_fn,
    search_space=search_space,
    backend="claude",
    log_path="search.jsonl",
)
search.run()

# Or from an existing model (introspects it for you)
search = ArchSearch.from_model(model, train_fn, backend="claude")
search.run()
```

## What happened overnight

<p align="center">
  <img src="assets/results.png" alt="Convergence plot showing LLM decisions over 1,239 experiments" width="800">
</p>

Starting from completely random architectures, the LLM read the training curves and narrowed in:

```
After    5 evals: exploring wildly — trying different activations, depths, optimizers
After   20 evals: noticed GELU + residual combos consistently had smoother val curves
After   50 evals: stopped using dropout — saw train-val gap was already small
After  100 evals: locked in on AdamW ~8.8e-4 — higher LRs showed oscillation in curves
After  500 evals: fine-tuning channel widths and growth rates
After 1239 evals: all top 10 configs converged on the same design pattern
```

What it learned (from reading the curves, not from being told):

| Decision | What it chose | What it saw in the curves |
|----------|--------------|--------------------------|
| Activation | GELU | ReLU configs had slower early-epoch convergence |
| Skip connections | Yes | Val loss plateaued past 4 blocks without them |
| BatchNorm | Yes | High-LR configs without BN showed loss spikes |
| Dropout | 0.0 | Adding dropout widened the train-val gap with no val improvement |
| Optimizer | AdamW | SGD configs needed 3x more LR tuning to match |

## LLM backends

Auto-detects in order. Set an API key and it works.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude (recommended, ~$0.20/night)
export OPENAI_API_KEY="sk-..."          # OpenAI

swarmopt run train.py --backend qwen    # local Qwen on CPU, no key needed
swarmopt run train.py --backend none    # random search baseline
```

## Search space types

| Class | Use case |
|-------|----------|
| `LogUniform(1e-4, 1e-1)` | Learning rates, weight decay |
| `Uniform(0.0, 0.5)` | Dropout, momentum |
| `IntUniform(2, 8)` | Layer counts, hidden sizes |
| `Categorical(["adam", "sgd"])` | Optimizer, activation |
| `Categorical([True, False])` | Toggles (residual, batch norm) |

## `train_fn` contract

Your function receives a config dict and returns a dict:

```python
def train_fn(config) -> dict:
    # Required: "score" (lower is better)
    # Optional but recommended for curve-aware search:
    #   "train_losses": [float]    per-epoch
    #   "val_losses": [float]      per-epoch
    #   "val_accuracies": [float]  per-epoch
    #   "accuracy": float          final
    #   "n_params": int            model size
```

When using `from_model`, you also get `config["model"]` — the modified model.

## Installation

```bash
pip install swarmopt                # core
pip install swarmopt[llm]           # + Claude API
pip install swarmopt[llm-openai]    # + OpenAI API
pip install swarmopt[llm-local]     # + local Qwen
pip install swarmopt[torch]         # + PyTorch
pip install swarmopt[all]           # everything
```

## Examples

| File | Description |
|------|------------|
| [`train_fashion.py`](examples/train_fashion.py) | CNN architecture search on FashionMNIST |
| [`train_resnet.py`](examples/train_resnet.py) | Optimize a ResNet with model introspection |

## License

MIT
