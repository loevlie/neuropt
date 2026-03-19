# Python API

## ArchSearch

The main class. Use it directly in notebooks or scripts.

### From a search space

```python
from neuropt import ArchSearch

search = ArchSearch(
    train_fn=train_fn,
    search_space={
        "lr": (1e-4, 1e-1),
        "n_layers": (2, 8),
        "activation": ["relu", "gelu", "silu"],
        "use_bn": [True, False],
    },
    backend="claude",
)
search.run(max_evals=50)

print(search.best_config)
print(search.best_score)
```

### From a model

```python
from neuropt import ArchSearch

search = ArchSearch.from_model(
    model=my_model,
    train_fn=train_fn,
    backend="claude",
)
search.run(max_evals=50)
```

`from_model` introspects the module tree, finds activations/dropout/batch norm, generates a search space, and wraps your `train_fn` so `config["model"]` contains the modified deep copy.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_fn` | required | `config dict → result dict` |
| `search_space` | required | Dict of param names to ranges/choices |
| `backend` | `"auto"` | `"auto"`, `"claude"`, `"openai"`, `"qwen"`, `"none"` |
| `log_path` | `"search.jsonl"` | JSONL log file |
| `batch_size` | `3` | Configs per LLM call |
| `device` | `None` | Injected as `config["device"]` |
| `timeout` | `600` | Max seconds per experiment |
| `ml_context` | generic | Domain knowledge for the LLM |

### `run(max_evals=None)`

Runs the search loop. If `max_evals` is set, stops after that many experiments. Otherwise runs until Ctrl+C.

### Result attributes

After `run()` completes:

- `search.best_score` — lowest score seen
- `search.best_config` — config dict that produced it
- `search.best_accuracy` — accuracy of best config (if returned)
- `search.total_experiments` — total experiments run
- `search.llm_success` — LLM calls that produced valid configs
- `search.llm_fallback` — LLM calls that fell back to random

## train_fn contract

Your function receives a config dict and returns a result dict.

**Required return key:**

- `"score"` — float, lower is better

**Optional return keys (recommended):**

- `"train_losses"` — list of per-epoch training losses
- `"val_losses"` — list of per-epoch validation losses
- `"val_accuracies"` — list of per-epoch validation accuracies
- `"accuracy"` — final accuracy
- `"n_params"` — model parameter count

The per-epoch lists are what give the LLM its advantage — it can spot overfitting, underfitting, and learning rate issues from the curve shapes.

## Search space types

You can use plain Python types (auto-inferred) or explicit dimension objects.

### Auto-inference from tuples and lists

```python
search_space = {
    "lr": (1e-4, 1e-1),              # → LogUniform (name-based)
    "wd": (1e-6, 1e-2),              # → LogUniform (name-based)
    "dropout": (0.0, 0.5),           # → Uniform
    "n_layers": (2, 8),              # → IntUniform (name + int values)
    "hidden_dim": (32, 512),          # → IntUniform (name + int values)
    "activation": ["relu", "gelu"],   # → Categorical
    "use_bn": [True, False],          # → Categorical
}
```

Names like `lr`, `learning_rate`, `wd`, `weight_decay` automatically get log-scale sampling. Names like `n_layers`, `hidden_dim`, `num_heads` get integer sampling. Integer tuple values also trigger IntUniform.

### Explicit dimension objects

For full control over ranges and sampling:

```python
from neuropt import LogUniform, Uniform, IntUniform, Categorical

search_space = {
    "lr": LogUniform(1e-4, 1e-1),
    "momentum": Uniform(0.8, 0.99),
    "depth": IntUniform(2, 8),
    "optimizer": Categorical(["sgd", "adam", "adamw"]),
}
```
