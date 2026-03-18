# swarmopt

<p align="center">
  <img src="assets/banner.png" alt="Three robot researchers designing neural network architectures" width="700">
</p>

<p align="center">
  <em>An LLM reads your training curves and designs your next experiment.</em>
</p>

---

Point it at a training script, let it run overnight. The LLM sees full per-epoch train/val curves, spots overfitting, and decides what to try next. We ran it for 8 hours on a MacBook — 1,239 experiments, zero human intervention — and it independently converged on GELU + residual + BatchNorm + AdamW + zero dropout. **90.9% on FashionMNIST**, 2M params.

## Quick start

```bash
pip install swarmopt[llm]
export ANTHROPIC_API_KEY="sk-ant-..."
```

Write a training script with `train_fn` + `search_space`:

```python
# train.py
from swarmopt import LogUniform, Categorical

search_space = {
    "lr": LogUniform(1e-4, 1e-1),
    "hidden_dim": IntUniform(32, 512),
    "activation": Categorical(["relu", "gelu", "silu"]),
    "optimizer": Categorical(["adam", "adamw", "sgd"]),
}

def train_fn(config):
    model = build_model(config["hidden_dim"], config["activation"])
    # ... train for N epochs ...
    return {
        "score": val_loss,
        "train_losses": [2.3, 1.1, 0.6],  # per-epoch
        "val_losses": [2.1, 1.0, 0.7],
        "val_accuracies": [0.2, 0.5, 0.7],
    }
```

Run it:

```bash
swarmopt run train.py
```

That's it. Runs until Ctrl+C. Crash-safe, resumable.

## Or: just give it a model

Don't want to define a search space? Pass an existing PyTorch model and we'll figure out what's tunable:

```python
# train.py
import torchvision.models as models

model = models.resnet18(num_classes=10)

def train_fn(config):
    m = config["model"].to("cuda")  # activations/dropout/BN already modified
    optimizer = torch.optim.Adam(m.parameters(), lr=config["lr"])
    # ... your normal training loop ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}
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

It finds every swappable activation, dropout layer, and batch norm — deep-copies the model each experiment so your original is never touched.

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

Same thing, no CLI:

```python
from swarmopt import ArchSearch, LogUniform, Categorical

# Manual search space
search = ArchSearch(
    train_fn=train_fn,
    search_space={"lr": LogUniform(1e-4, 1e-1), ...},
    backend="claude",
)
search.run()

# Or from a model
search = ArchSearch.from_model(my_model, train_fn, backend="claude")
search.run()
```

## What happened overnight

```
After    5 evals:  62.4% accuracy   (exploring)
After   20 evals:  70.6%            (found GELU + residual)
After  100 evals:  72.5%            (tuning lr/wd)
After 1239 evals:  90.9%            (final best)
```

Every top-10 architecture landed on the same pattern:

| Decision | What it chose | Why |
|----------|--------------|-----|
| Activation | GELU | Smoother gradients, faster convergence |
| Skip connections | Yes | Val loss plateaued without them |
| BatchNorm | Yes | Training unstable without it |
| Dropout | 0.0 | Train-val gap was already small |
| Optimizer | AdamW lr≈8.8e-4 | Higher diverged, lower too slow |

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
