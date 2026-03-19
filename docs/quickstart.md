# Quick Start

Get neuropt running in under 2 minutes.

---

<div class="step" markdown>
<div class="step-number">01</div>
<div class="step-content" markdown>

### Install neuropt

```bash
pip install "neuropt[llm]"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

</div>
</div>

---

<div class="step" markdown>
<div class="step-number">02</div>
<div class="step-content" markdown>

### Write a training function

Your function trains a model and returns results. neuropt calls it with different configs each time.

```python
# train.py

def train_fn(config):
    model = build_model(config["hidden_dim"], config["activation"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_losses, val_losses = [], []
    for epoch in range(10):
        # ... train one epoch ...
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return {
        "score": val_losses[-1],        # required
        "train_losses": train_losses,    # helps the LLM spot overfitting
        "val_losses": val_losses,        # helps the LLM spot underfitting
        "accuracy": val_acc,             # any extra metrics you want
    }
```

!!! tip "Return per-epoch losses"
    The more you return, the smarter the LLM gets. Per-epoch curves let it reason about *why* a config worked — not just that it did.

</div>
</div>

---

<div class="step" markdown>
<div class="step-number">03</div>
<div class="step-content" markdown>

### Define what to search

```python
# train.py (continued)

search_space = {
    "lr": (1e-4, 1e-1),              # auto → log-scale
    "hidden_dim": (64, 512),          # auto → integer
    "dropout": (0.0, 0.5),            # auto → uniform
    "activation": ["relu", "gelu"],   # auto → categorical
}
```

Tuples become ranges, lists become choices. neuropt infers the right sampling strategy from the param name and value types.

</div>
</div>

---

<div class="step" markdown>
<div class="step-number">04</div>
<div class="step-content" markdown>

### Run it

=== "CLI"

    ```bash
    neuropt run train.py
    neuropt run train.py --backend claude -n 50
    ```

=== "Python"

    ```python
    from neuropt import ArchSearch

    search = ArchSearch(
        train_fn=train_fn,
        search_space=search_space,
        backend="claude",
    )
    search.run(max_evals=50)

    print(search.best_config)
    print(search.best_score)
    ```

</div>
</div>

---

## Or skip the search space

Give it a model — neuropt figures out what to tune.

```python
import torchvision
from neuropt import ArchSearch

model = torchvision.models.resnet18(num_classes=10)

def train_fn(config):
    m = config["model"].to("cuda")   # modified deep copy
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}

search = ArchSearch.from_model(model, train_fn)
search.run(max_evals=30)
```

Works with PyTorch models, XGBoost, LightGBM, Random Forest, and any sklearn estimator. See [Model Introspection](introspection.md) for details.

---

## Options

| Flag | What it does |
|------|-------------|
| `--backend claude` | Use Claude (default if `ANTHROPIC_API_KEY` is set) |
| `--backend none` | Random search — no API key needed |
| `-n 50` | Stop after 50 experiments |
| `--log results.jsonl` | Custom log file (crash-safe, resumable) |
| `--device cuda` | Force GPU device |
