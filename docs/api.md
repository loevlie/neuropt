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

`from_model` introspects the module tree — activations, dropout, norms, pooling — and generates a search space automatically. See [Model Introspection](introspection.md) for the full detection list. For pretrained models, it also adds freeze strategies, LR decay, and L2-SP — see [Fine-Tuning](fine-tuning.md).

Pass `pretrained=True` or `pretrained=False` to override auto-detection:

```python
search = ArchSearch.from_model(model, train_fn, pretrained=True)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_fn` | required | `config dict → result dict` |
| `search_space` | required | Dict of param names to ranges/choices |
| `backend` | `"auto"` | `"auto"`, `"claude"`, `"openai"`, `"openai-compatible"`, `"none"`, or a backend instance (`"qwen"` is deprecated) |
| `log_path` | `"search.jsonl"` | JSONL log file |
| `batch_size` | `3` | Configs per LLM call |
| `device` | `None` | Injected as `config["device"]` |
| `timeout` | `600` | Max seconds per experiment |
| `ml_context` | generic | Domain knowledge for the LLM |
| `minimize` | `True` | If `True`, lower scores are better (loss). Set `False` for accuracy/AUROC |
| `seed` | `42` | Seed for the random configs the search draws (random fallback, decision-model sampling) |
| `max_curve_points` | `None` | Show at most this many points per curve in the prompt (default all). `20` cut XGBoost runs' cost 8x with no measured loss in quality |
| `initial_configs` | `None` | Configs to evaluate before asking the model, e.g. your current settings. Snapped/clamped into the space; keys may be left out; not re-run on resume |

`from_model(..., start_from_model=True)` evaluates the model's own settings first: an estimator's
`get_params()` (params left at `None` keep the library default), or for a PyTorch model the model as
it is, trained with AdamW, lr 1e-3, weight decay 1e-4. Pass `start_from_model=False` to skip it. In the
benchmark harness, starting from a sensible config helped every neuropt arm on the CNN task and never
made one worse ([evaluations](evals.md)).

### Backends and models

| Backend | Selected when | Model |
|---|---|---|
| `openai-compatible` | `$NEUROPT_LLM_BASE_URL` is set (auto) | `$NEUROPT_MODEL` (required) |
| `claude` | `$ANTHROPIC_API_KEY` is set (auto), or `$NEUROPT_MODEL` starts with `claude` | `$NEUROPT_MODEL`, else Claude Haiku 5.5 |
| `openai` | `$OPENAI_API_KEY` is set (auto) | `$NEUROPT_MODEL`, else GPT-4o mini |

Pass a model directly with a backend instance:

```python
from neuropt.backends.claude_backend import ClaudeBackend
from neuropt.backends.openai_backend import OpenAIBackend

ArchSearch(train_fn, space, backend=ClaudeBackend(model="claude-haiku-5-5"))
ArchSearch(train_fn, space, backend=OpenAIBackend(model="qwen3.5:9b", base_url="http://localhost:11434/v1"))
```

The model's answer is constrained to a JSON schema built from the search space (structured outputs),
and the part of the prompt that never changes is sent as a cached system prompt. Servers that don't
support a schema get plain JSON instructions. A custom backend only needs `generate(prompt, max_tokens)`.

### `run(max_evals=None)`

Runs the search loop. If `max_evals` is set, stops after that many experiments. Otherwise runs until Ctrl+C.

### Result attributes

After `run()` completes:

- `search.best_score` — best score seen (lowest if `minimize=True`, highest if `minimize=False`)
- `search.best_config` — config dict that produced it
- `search.best_accuracy` — accuracy of best config (if returned)
- `search.total_experiments` — total experiments run
- `search.llm_success` — LLM calls that produced valid configs
- `search.llm_fallback` — LLM calls that fell back to random
- `search.sources` — evals by where their config came from (`llm`, `random`, `initial`); the summary warns when more than 10% of the model's configs were random
- `search.stats` — LLM calls, parse failures, errors, invalid and duplicate proposals

## DecisionSearch

*Experimental: still being evaluated.* Same interface as `ArchSearch`, but a [decision model](decision-models.md) (a Jev-style model that answers multiple-choice questions with probabilities) picks the configs.

```python
from neuropt import DecisionSearch

search = DecisionSearch(train_fn, search_space, backend="decision")
search.run(max_evals=50)
```

Extra parameters on top of `ArchSearch`'s:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"auto"` | `"auto"` (server if reachable, else random), `"decision"`, `"none"`, or a `BaseDecisionBackend` |
| `n_bins` | `8` | Ranges each numeric parameter is split into |
| `n_candidates` | `12` | Configs sampled from the answers before the joint pick |
| `explore` | `0.15` | Uniform probability mixed into each answer when sampling |
| `rerank` | `True` | Ask the model to pick jointly among the candidates |
| `max_curve_points` | `20` | Long curves (e.g. 500 boosting rounds) are shown at this many evenly spaced epochs; every question re-reads the state |

`search.llm_success` / `search.llm_fallback` count decision-model iterations that succeeded or fell back to random.

## train_fn contract

Your function receives a config dict and returns a result dict.

**Required return key:**

- `"score"` — float, lower is better by default (set `minimize=False` for accuracy/AUROC)

**Everything else is auto-detected.** Return any extra keys and the LLM will see them:

- **Scalars** (int, float, str, bool) → shown as columns in the history table
- **Lists of numbers** → shown as per-epoch curves

```python
return {
    "score": val_loss,
    "train_losses": epoch_train_losses,   # → curve
    "val_losses": epoch_val_losses,       # → curve
    "val_accuracies": epoch_val_accs,     # → curve
    "accuracy": final_acc,                # → scalar column
    "n_params": count_params(model),      # → scalar column
    "n_train": len(train_set),            # → scalar column
    "f1_macro": f1,                       # → scalar column
    "epoch_time": avg_epoch_secs,         # → scalar column
}
```

The per-epoch curves are what give the LLM its advantage — it can spot overfitting, underfitting, and learning rate issues from the curve shapes. Return whatever metrics matter for your problem; the LLM sees all of them.

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
