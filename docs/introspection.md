# Auto Model Introspection

!!! note "This is all under the hood"
    You don't need to understand any of this to use neuropt. Just call `ArchSearch.from_model(model, train_fn)` and it handles everything. This page is for people who want to know what's happening inside.

With `ArchSearch.from_model()`, you don't specify any hyperparameters or search ranges — neuropt walks your model's module tree, discovers what's tunable, and builds a search space for you.

```python
from neuropt import ArchSearch

model = build_my_cnn()  # or load a pretrained model

def train_fn(config):
    m = config["model"].to("cuda")  # deep copy with modifications applied
    optimizer = torch.optim.Adam(m.parameters(), lr=config["lr"])
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}

search = ArchSearch.from_model(model, train_fn, backend="claude")
search.run(max_evals=50)
```

neuropt prints what it found and what it will search over:

```
Introspected PyTorch model (77,322 params):
  Activations: ReLU, Mish (2 layers)
  Dropout: 3 layers (rate=0.20)
  BatchNorm: 4 layers
  Pooling: 1 layers (current: avg)
  Search space: ['activation', 'dropout', 'use_batchnorm', 'pool_type',
    'lr', 'wd', 'optimizer']
```

Your original model is never modified — each experiment gets a deep copy with the LLM's proposed changes applied.

## How it works

1. **Walk the module tree** — `model.named_modules()` finds every layer
2. **Classify each module** — activations, dropout, norms, pooling, etc.
3. **Build a search space** — each detected component becomes a tunable parameter
4. **Wrap your train_fn** — deep-copies the model, applies config, passes it as `config["model"]`
5. **Detect pretrained weights** — if the model looks pretrained, add fine-tuning strategies (see [Fine-Tuning](fine-tuning.md))

The LLM gets context about what was detected ("4 BatchNorm layers", "attention and FF dropout found separately") and domain-specific guidance so it can reason about *why* to try a change, not just guess.

## What gets detected

| Component | What neuropt finds | Search param |
|-----------|-------------------|-------------|
| Activations | `ReLU`, `GELU`, `SiLU`, `Mish`, `Hardswish`, `PReLU`, `LeakyReLU` | `activation` (7-way swap) |
| Dropout | `Dropout`, `Dropout1d/2d/3d`, `AlphaDropout`, `FeatureAlphaDropout` | `dropout` (rate) |
| MHA dropout | `MultiheadAttention` internal `.dropout` float | `mha_dropout` (rate) |
| Per-path dropout | Groups by path name (`attn`, `ff`, `mlp`, `embed`) | `dropout_attn`, `dropout_ff`, etc. |
| BatchNorm | `BatchNorm1d/2d/3d` | `use_batchnorm` (on/off) |
| LayerNorm | `LayerNorm` | `use_layernorm` (on/off) |
| Pooling | `AdaptiveAvgPool`, `AdaptiveMaxPool` | `pool_type` (avg/max/attention) |
| Training | Always included | `lr`, `wd`, `optimizer` |

If pretrained weights are detected, additional fine-tuning params are added — see [Fine-Tuning Pretrained Models](fine-tuning.md).

## Activations

All detected activation layers are swapped together. The 7 options:

| Activation | Notes |
|-----------|-------|
| `relu` | Standard default |
| `gelu` | Common in transformers, often better than ReLU |
| `silu` | Smooth, used in EfficientNet and modern CNNs |
| `mish` | Competitive with SiLU, used in YOLOv4/v5 |
| `hardswish` | Cheaper SiLU approximation, good for mobile/edge |
| `leaky_relu` | Avoids dead neurons |
| `prelu` | Learnable negative slope (adds a small number of parameters) |

## Dropout

neuropt detects all PyTorch dropout variants — `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout`, and `FeatureAlphaDropout`. The rate is tuned as a single `dropout` param.

**Per-path dropout** — for transformers with dropout in distinct paths, neuropt groups by module path name and emits separate params so the LLM can tune attention and feedforward dropout independently:

- Path contains `attn`/`attention` → `dropout_attn`
- Path contains `ff`/`ffn`/`mlp` → `dropout_ff`
- Path contains `embed` → `dropout_embed`
- Falls back to single `dropout` if only one group exists

**MHA internal dropout** — `nn.MultiheadAttention` stores dropout as a plain float attribute, not a module. neuropt catches this and adds `mha_dropout` as a separate search param.

## Normalization

BatchNorm and LayerNorm can each be toggled off (replaced with `nn.Identity()`).

- **BatchNorm off** — sometimes helps very small models or when batch sizes vary widely
- **LayerNorm off** — rarely helpful in transformers, but the LLM can test it

## Pooling

When `AdaptiveAvgPool` or `AdaptiveMaxPool` layers are found, neuropt adds a three-way swap:

| Pool type | How it works |
|-----------|-------------|
| `avg` | Average over all spatial positions (standard default) |
| `max` | Keep the strongest activation per channel |
| `attention` | Learned weighted average — a small linear projection computes attention weights over spatial positions (~C extra params) |

Attention pooling is useful when not all spatial positions are equally important for the task — it learns where to look.
