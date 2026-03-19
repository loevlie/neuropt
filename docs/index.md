# neuropt

LLM-guided ML optimization. Point it at a training script, let it run overnight.

The LLM sees full per-epoch train/val curves, spots overfitting, and proposes what to try next. When it can't produce valid configs, the system silently falls back to random search. Everything is logged to JSONL — crash-safe and resumable.

## Install

```bash
pip install neuropt[llm]
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Two ways to use it

### Define a search space

```python
# train.py
search_space = {
    "lr": (1e-4, 1e-1),
    "hidden_dim": (32, 512),
    "activation": ["relu", "gelu", "silu"],
}

def train_fn(config):
    model = build_my_model(config["hidden_dim"], config["activation"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}
```

Tuples become ranges (log-scale for `lr`/`wd`, integer for `n_layers`/`hidden_dim`, uniform otherwise). Lists become categorical choices. See [Search space types](api.md#search-space-types) for details.

### Or give it a model

```python
# train.py
model = torchvision.models.resnet18(num_classes=10)

def train_fn(config):
    m = config["model"].to("cuda")
    optimizer = torch.optim.Adam(m.parameters(), lr=config["lr"])
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}
```

neuropt walks the module tree, finds activations/dropout/batch norm, and generates a search space automatically. Your model is deep-copied each experiment — the original is never touched.

### Run it

```bash
neuropt run train.py
neuropt run train.py --backend claude -n 50  # stop after 50 experiments
```

Or in a notebook:

```python
from neuropt import ArchSearch

search = ArchSearch(train_fn=train_fn, search_space=search_space, backend="claude")
search.run(max_evals=50)
```
