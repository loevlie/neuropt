# swarmopt

<p align="center">
  <img src="assets/banner.png" alt="Three robot researchers designing neural network architectures" width="700">
</p>

<p align="center">
  <em>An LLM reads your training curves and designs your next experiment.</em>
</p>

---

We pointed an LLM at a CNN search space, gave it ML domain knowledge, and let it run overnight on a MacBook. It trained 1,239 models in 8 hours and independently converged on GELU + residual connections + BatchNorm + AdamW — the same recipe a senior ML engineer would pick. It also figured out that dropout was hurting performance on this dataset and stopped using it entirely.

**Best result: 90.9% accuracy on FashionMNIST**, 2M parameter model, zero human intervention.

## What happened overnight

The LLM sees full per-epoch training curves — not just final scores — so it can reason about *why* things work or don't.

```
After    5 evals:  62.4% accuracy   (exploring wildly)
After   20 evals:  70.6%            (narrowing in on GELU + residual)
After  100 evals:  72.5%            (tuning lr/wd around 8e-4)
After  500 evals:  72.6%            (micro-optimizing channel widths)
After 1239 evals:  90.9%            (final best)
```

Every one of the top 10 architectures landed on the same design pattern:

| Decision | What it chose | Why (from the curves) |
|----------|--------------|----------------------|
| Activation | GELU | Smoother gradients, faster convergence in early epochs |
| Skip connections | Yes | Val loss plateaued without them past 4 blocks |
| BatchNorm | Yes | Training was unstable without it at higher learning rates |
| Dropout | 0.0 | Train-val gap was already small — dropout just slowed learning |
| Optimizer | AdamW | SGD needed 3x more tuning to match; Adam had weight decay issues |
| LR | ~8.8e-4 | Higher diverged (saw it in curves), lower converged too slowly |

## Try it

```bash
pip install swarmopt[llm]
```

```python
from swarmopt import ArchSearch, LogUniform, Uniform, IntUniform, Categorical

search = ArchSearch(
    train_fn=my_train_fn,   # you provide this
    search_space={
        "lr": LogUniform(1e-4, 1e-1),
        "n_layers": IntUniform(2, 8),
        "hidden_dim": IntUniform(32, 256),
        "dropout": Uniform(0.0, 0.5),
        "activation": Categorical(["relu", "gelu", "silu"]),
        "optimizer": Categorical(["adam", "adamw", "sgd"]),
    },
    backend="claude",       # or "openai", "qwen", "none"
)
search.run()  # runs until Ctrl+C
```

Your `train_fn` takes a config dict, trains a model, returns a dict with at least `"score"`. For smarter LLM decisions, also return per-epoch metrics:

```python
def train_fn(config):
    model = build_model(config)
    # ... train for N epochs, tracking losses ...
    return {
        "score": val_loss,
        "accuracy": val_acc,
        "train_losses": [2.3, 1.1, 0.6, 0.4],   # per-epoch
        "val_losses": [2.1, 1.0, 0.7, 0.5],
        "val_accuracies": [0.2, 0.5, 0.7, 0.8],
    }
```

Everything logs to JSONL — crash-safe, resumable. See [`examples/llm_arch_search.py`](examples/llm_arch_search.py) for a full CNN architecture search on FashionMNIST.

## What the LLM actually sees

This is the core idea. Most hyperparameter tools give the optimizer a single number: *"this config scored 0.85."* We give the LLM the full picture:

```
4blk/64ch/relu/nores/sgd lr=0.05:
  ep1:  train=2.30  val=2.28  acc=0.12
  ep2:  train=1.45  val=1.52  acc=0.41
  ep3:  train=0.82  val=1.35  acc=0.53
  ep4:  train=0.31  val=1.61  acc=0.48   ← val rising while train drops
  ep5:  train=0.09  val=1.89  acc=0.45   ← overfitting, gap = 1.80

5blk/32ch/gelu/res/adamw lr=8.8e-4:
  ep1:  train=1.92  val=1.85  acc=0.28
  ep2:  train=1.01  val=0.98  acc=0.62
  ep3:  train=0.62  val=0.71  acc=0.74
  ep4:  train=0.41  val=0.52  acc=0.81   ← both dropping smoothly
  ep5:  train=0.33  val=0.43  acc=0.85   ← good fit, small gap
```

The system also pre-computes signals: `OVERFITTING: 4blk/64ch (train 2.30→0.09, val 1.52→1.89, gap=1.80)`.

A human researcher would look at those curves and say *"the first model is overfitting hard, try more regularization or a smaller model."* The LLM does the same thing.

## How it works

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│  LLM reads  │────▶│  Proposes    │────▶│  Train model │
│  history +  │     │  next batch  │     │  on GPU/MPS  │
│  curves     │     │  of configs  │     │              │
└─────────────┘     └─────────────┘     └──────┬───────┘
       ▲                                        │
       │            ┌─────────────┐             │
       └────────────│  Log results│◀────────────┘
                    │  (JSONL)    │
                    └─────────────┘
```

- **LLM consults take 1-3s** vs 20s per training eval — negligible overhead
- **If the LLM returns bad JSON**, the system silently falls back to random configs
- **Everything is logged** — you can resume after crashes, Ctrl+C, or reboots
- **Works on Apple Silicon** — training on MPS, LLM on CPU (if local), no contention

## Architecture search space

The script searches over both architecture and training hyperparameters:

| Parameter | Range | Type |
|-----------|-------|------|
| Conv blocks | 2–8 | int |
| Base channels | 16–128 | int |
| Channel growth | 1.0–2.5x per block | float |
| Kernel size | 3, 5 | choice |
| Activation | ReLU, GELU, LeakyReLU, SiLU | choice |
| Residual connections | yes/no | bool |
| Batch normalization | yes/no | bool |
| Dropout | 0.0–0.5 | float |
| Pooling interval | every 1–4 blocks | int |
| FC hidden layer | 0–512 units | int |
| Learning rate | 1e-4 to 0.1 | log float |
| Weight decay | 1e-6 to 0.01 | log float |
| Optimizer | SGD, Adam, AdamW | choice |

## LLM backends

Auto-detects in order. Set an API key and it just works.

```bash
# Claude (recommended — fast, cheap with Haiku, ~$0.20 per overnight run)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Local Qwen 2.5 1.5B on CPU (no API needed, ~3s per call)
python examples/llm_arch_search.py --backend qwen

# Random search baseline (no LLM)
python examples/llm_arch_search.py --backend none
```

## CLI options

```bash
python examples/llm_arch_search.py \
    --backend claude        # claude, openai, qwen, none
    --epochs 10             # training epochs per eval
    --batch-per-iter 3      # configs proposed per LLM call
    --subset-size 5000      # training subset size
    --device mps            # cuda, mps, cpu
    --log results.jsonl     # log file (supports resume)
```

---

## Also included: PSO hyperparameter tuner

A standalone particle swarm optimizer with an sklearn-style API. No LLM needed.

```python
from swarmopt import SwarmTuner, LogUniform

tuner = SwarmTuner(
    train_fn=train_fn,
    search_space={
        "lr": LogUniform(1e-4, 1e-1),
        "wd": LogUniform(1e-6, 1e-2),
    },
    n_particles=5,
    n_iterations=10,
    device="cuda",
)
tuner.fit()

print(tuner.best_params)  # {"lr": 0.023, "wd": 1.2e-5}
tuner.plot()               # convergence + trajectory plots
tuner.animate()            # particle trajectory GIF
```

There's also a hybrid PSO+LLM mode where PSO proposes candidates and the LLM refines them based on history. See `examples/llm_pso_fashion.py`.

### Search space types

| Class | Use case |
|-------|----------|
| `LogUniform(low, high)` | Learning rates, weight decay |
| `Uniform(low, high)` | Momentum, dropout |
| `IntUniform(low, high)` | Layer counts, hidden units |
| `Categorical(choices)` | Optimizer names, activation types |

## Installation

```bash
pip install swarmopt              # core PSO tuner
pip install swarmopt[torch]       # + PyTorch for examples
pip install swarmopt[llm]         # + Claude API for arch search
pip install swarmopt[all]         # everything
```

## Examples

| File | What it does |
|------|-------------|
| [`llm_arch_search.py`](examples/llm_arch_search.py) | Autonomous overnight CNN architecture search |
| [`fashion_mnist.py`](examples/fashion_mnist.py) | PSO hyperparameter tuning on FashionMNIST |
| [`llm_pso_fashion.py`](examples/llm_pso_fashion.py) | Hybrid PSO + LLM search |
| [`benchmark_llm_pso.ipynb`](examples/benchmark_llm_pso.ipynb) | LLM+PSO vs Bayesian vs Grid Search comparison |
| [`benchmark_pso_vs_bayes_vs_grid.ipynb`](examples/benchmark_pso_vs_bayes_vs_grid.ipynb) | PSO vs Optuna vs Grid Search |

## License

MIT
