# Fine-Tuning Pretrained Models

!!! note "This is all under the hood"
    neuropt handles fine-tuning detection and strategy selection automatically. Just pass a pretrained model to `from_model()` — no configuration needed. This page explains what's happening inside.

When `from_model()` detects pretrained weights, it adds fine-tuning strategies to the search space alongside the standard [introspection params](introspection.md). The LLM uses domain knowledge about what works for fine-tuning — freeze strategies, learning rate decay, regularization toward pretrained weights — and decides what to try based on your training curves.

## Quick start

```python
import torchvision
from neuropt import ArchSearch

model = torchvision.models.resnet18(weights="DEFAULT")

def train_fn(config):
    m = config["model"].to("cuda")
    optimizer = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad],
        lr=config["lr"],
    )
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}

search = ArchSearch.from_model(model, train_fn, backend="claude")
search.run(max_evals=50)
```

```
Introspected PyTorch model (11,689,512 params):
  Activations: ReLU (9 layers)
  BatchNorm: 20 layers
  Pooling: 1 layers (current: avg)
  Pretrained: yes (fine-tuning strategies enabled)
  Head: fc
  Search space: ['activation', 'dropout', 'use_batchnorm', 'pool_type', 'lr',
    'wd', 'optimizer', 'freeze_strategy', 'lr_layer_decay', 'l2sp_regularization']
```

## How pretrained detection works

neuropt compares each parameter's variance to what PyTorch's default initialization would produce. Pretrained models that have been trained with weight decay have significantly lower variance than random init. This is automatic — no labels or metadata needed.

Override if the heuristic gets it wrong:

```python
search = ArchSearch.from_model(model, train_fn, pretrained=True)   # force on
search = ArchSearch.from_model(model, train_fn, pretrained=False)  # force off
```

neuropt also detects **layer groups** (repeating numbered blocks like `layer1.0`, `layer1.1`) and the **classification head** (last `nn.Linear`), which are used by the freeze strategies below.

## Fine-tuning search params

These three params are only added when pretrained weights are detected.

### `freeze_strategy`

Controls which layers are trainable. The LLM is given guidance about when each works best.

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| `full` | Train everything (no-op) | Large datasets, no forgetting risk |
| `head_only` | Freeze all, train only the classification head | Small datasets, safest option |
| `gradual_unfreeze` | Freeze all, unfreeze last ~1/3 of layer groups + head | Good default for medium datasets |
| `all_but_embeddings` | Freeze only `nn.Embedding` layers | NLP models where embeddings are well-trained |

### `lr_layer_decay`

Float in `[0.5, 1.0]`. Passed through as `config["lr_layer_decay"]` for you to implement layer-wise learning rate decay (LLRD) in your training loop.

- **Near 1.0** = uniform LR across all layers
- **Near 0.5** = aggressive decay (early layers learn much slower)

This is a pass-through value — neuropt puts it in config, you apply it:

```python
def train_fn(config):
    m = config["model"]
    decay = config.get("lr_layer_decay", 1.0)
    param_groups = []
    for i, (name, param) in enumerate(m.named_parameters()):
        if param.requires_grad:
            layer_lr = config["lr"] * (decay ** i)
            param_groups.append({"params": [param], "lr": layer_lr})
    optimizer = torch.optim.AdamW(param_groups)
    # ...
```

### `l2sp_regularization`

Boolean. When `True`, neuropt snapshots the pretrained weights before training and injects them as `config["pretrained_weights"]` — a dict of `{name: tensor}`. Use these to regularize toward pretrained weights instead of zero (L2-SP), which prevents catastrophic forgetting.

```python
def train_fn(config):
    m = config["model"].to("cuda")
    # ... forward pass ...
    loss = task_loss
    if "pretrained_weights" in config:
        l2sp_loss = sum(
            ((p - config["pretrained_weights"][n].to(p.device)) ** 2).sum()
            for n, p in m.named_parameters() if p.requires_grad
        )
        loss = loss + 0.01 * l2sp_loss
    # ...
```

## What the LLM knows

When a pretrained model is detected, the LLM's context includes fine-tuning-specific guidance:

- `head_only` is safest for small datasets
- `gradual_unfreeze` is a good default for medium datasets
- `full` is best with enough data but risks catastrophic forgetting
- `lr_layer_decay` near 0.5 means early layers barely update
- L2-SP prevents forgetting — worth trying if full fine-tuning overfits

This lets the LLM reason about your training curves in the context of fine-tuning: *"val loss is rising with `full` strategy → try `head_only` or increase L2-SP"* rather than generic overfitting advice.

## Early results

In a 10-eval benchmark fine-tuning ViT-B/16 on CIFAR-10, neuropt found a config achieving **0.142 val loss / 95.7% accuracy** — the LLM quickly converged on `last_blocks` freeze + AdamW + lr=3e-4. Optuna's best in the same budget was 0.195. The LLM's domain knowledge about freeze strategies gave it an edge that statistical surrogates can't match.
