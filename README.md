# swarmopt

<p align="center">
  <img src="assets/banner.png" alt="Three robot researchers designing neural network architectures" width="700">
</p>

PSO hyperparameter tuner for PyTorch, plus an LLM-guided architecture search that runs overnight and finds good CNN designs while you sleep.

## Installation

```bash
pip install swarmopt
```

With PyTorch support (for the examples):
```bash
pip install swarmopt[torch]
```

With an LLM backend for architecture search:
```bash
pip install swarmopt[llm]        # Claude API
pip install swarmopt[llm-openai] # OpenAI API
pip install swarmopt[llm-local]  # Local Qwen (runs on CPU)
pip install swarmopt[all]        # Everything
```

## PSO Hyperparameter Tuning

Define a training function, a search space, call `.fit()`.

```python
from swarmopt import SwarmTuner, LogUniform

def train_fn(params):
    lr, wd = params["lr"], params["wd"]
    # ... your training loop ...
    return val_loss

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

print(tuner.best_params)   # {"lr": 0.023, "wd": 1.2e-5}
print(tuner.best_score)    # 0.312
tuner.plot()               # 3-panel convergence figure
tuner.animate()            # particle trajectory GIF
```

Your `train_fn` can return a dict for richer tracking:

```python
def train_fn(params):
    # ... training ...
    return {"score": val_loss, "model": model.state_dict(), "accuracy": acc}

tuner.fit()
tuner.best_model  # state_dict of the best run
```

### Search Space Types

| Class | Maps to | Use case |
|-------|---------|----------|
| `Uniform(low, high)` | `[low, high]` | Bounded continuous (momentum, dropout) |
| `LogUniform(low, high)` | `[log10(low), log10(high)]` | Learning rates, weight decay |
| `IntUniform(low, high)` | `[low, high]` rounded | Discrete params (layers, units) |
| `Categorical(choices)` | Integer index | Architecture names, optimizer types |

## LLM-Guided Architecture Search

The more interesting part. Instead of PSO picking random points in a continuous space, an LLM reads the full experiment history — including per-epoch train/val loss curves — and decides what architecture to try next.

The LLM sees overfitting patterns, knows that GELU tends to outperform ReLU, understands that residual connections help deeper networks, and adjusts its suggestions based on what's actually working. When it can't produce valid configs (bad JSON, API timeout), the system silently falls back to random search.

### Running the architecture search

```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python examples/llm_arch_search.py
```

That's it. It runs until you Ctrl+C. Every experiment is logged to `arch_search.jsonl` so you can resume later and nothing is lost if it crashes.

The search space covers both architecture and training hyperparameters:

- **Architecture:** number of conv blocks, channel width, channel growth rate, kernel size, activation function (ReLU/GELU/LeakyReLU/SiLU), residual connections, batch normalization, dropout, pooling strategy, FC head size
- **Training:** learning rate, weight decay, optimizer (SGD/Adam/AdamW)

### What it found overnight

We ran it for ~8 hours on a single M1 MacBook (1,239 experiments, ~20s each). The LLM converged on:

| Parameter | Best value |
|-----------|-----------|
| Architecture | 7 blocks, 26 base channels, 1.6x growth |
| Activation | GELU |
| Residual connections | Yes |
| BatchNorm | Yes |
| Dropout | 0.0 |
| Optimizer | AdamW, lr=8.84e-4, wd=1.2e-4 |
| **Result** | **90.9% accuracy on FashionMNIST** (2M params) |

All 10 of the top architectures used GELU + residual + BatchNorm + AdamW + zero dropout. The LLM learned from the training curves that dropout was hurting small models on this dataset and stopped using it entirely after the first few iterations.

Convergence:
```
After   5 evals: 0.376 val loss
After  20 evals: 0.294
After 100 evals: 0.275
After 500 evals: 0.274
After 1239 evals: 0.256
```

### CLI options

```bash
python examples/llm_arch_search.py \
    --backend claude          # or openai, qwen, none (random baseline)
    --epochs 10               # training epochs per eval
    --batch-per-iter 3        # configs per LLM consultation
    --subset-size 5000        # training subset size
    --device mps              # cuda, mps, or cpu
    --log my_search.jsonl     # log file path (supports resume)
```

### Using different LLM backends

The system auto-detects available backends in order: Claude API → OpenAI API → local Qwen → random search fallback.

```bash
# Claude (recommended — fast, cheap with Haiku)
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/llm_arch_search.py --backend claude

# OpenAI
export OPENAI_API_KEY="sk-..."
python examples/llm_arch_search.py --backend openai

# Local Qwen 2.5 1.5B on CPU (no API key needed, ~3s per consultation)
python examples/llm_arch_search.py --backend qwen

# Pure random search (baseline comparison)
python examples/llm_arch_search.py --backend none
```

## PSO + LLM Hybrid

If you want PSO's exploration/exploitation dynamics as a prior but with LLM refinement, there's a hybrid mode too. PSO proposes particle positions, the LLM reads the history and modifies the configs before evaluation, then PSO updates its velocities based on actual results.

```bash
python examples/llm_pso_fashion.py --backend claude
```

See `examples/benchmark_llm_pso.ipynb` for a head-to-head comparison against plain PSO, Bayesian optimization (Optuna TPE), and grid search on the same evaluation budget.

## Examples

| File | What it does |
|------|-------------|
| `examples/fashion_mnist.py` | PSO hyperparameter search (lr, wd) on FashionMNIST |
| `examples/llm_arch_search.py` | Autonomous LLM-guided CNN architecture search |
| `examples/llm_pso_fashion.py` | PSO + LLM hybrid search |
| `examples/benchmark_pso_vs_bayes_vs_grid.ipynb` | PSO vs Optuna vs Grid Search comparison |
| `examples/benchmark_llm_pso.ipynb` | LLM+PSO vs PSO vs Bayesian vs Grid Search |

## How the LLM advisor works

The advisor builds a structured prompt with:

1. **Search space description** — parameter names, types, valid ranges
2. **Best result so far** — the target to beat
3. **Recent experiments table** — last 20 configs with results
4. **Per-epoch learning curves** — train loss, val loss, val accuracy at each epoch for recent experiments
5. **Pre-computed analysis** — overfitting/underfitting detection, trend direction, best hyperparameters seen

The LLM responds with a JSON array of configs. If parsing fails for any reason, the system falls back to PSO suggestions (hybrid mode) or random configs (pure LLM mode). No exceptions propagate, no crashes — just a fallback counter that ticks up.

## License

MIT
