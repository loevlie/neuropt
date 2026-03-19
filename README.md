# swarmopt

<p align="center">
  <img src="assets/banner.png" alt="Three robot researchers designing neural network architectures" width="700">
</p>

<p align="center">
  <em>An LLM reads your training curves and designs your next experiment.</em>
</p>

---

Point it at a training script, let it run overnight. The LLM sees full per-epoch train/val curves, spots overfitting, and proposes what to try next — like a research assistant who never sleeps and actually reads the loss plots.

We let it run for 8 hours on a MacBook. Starting from random configs, it independently converged on GELU + residual connections + BatchNorm + AdamW + zero dropout — the same recipe a senior ML engineer would pick. It noticed dropout was hurting performance from the training curves and stopped using it. 1,239 experiments, zero human intervention.

<p align="center">
  <img src="assets/convergence.png" alt="LLM convergence: loss 0.72 → 0.26 over 858 experiments" width="750">
</p>

### vs Optuna and random search (same 15-eval budget)

<p align="center">
  <img src="assets/benchmark.png" alt="Benchmark: LLM vs Optuna vs Random" width="650">
</p>

14-parameter CNN search space on FashionMNIST. Optuna TPE wasted 7 of 15 evals on configs that scored worse than random chance. The LLM already knew to use residual connections, reasonable learning rates, and modern activations — it didn't need to learn that from scratch.

> Local Qwen backend is experimental — works but has a ~40% JSON parse failure rate on complex search spaces. Claude API recommended.

## Quick start

```bash
pip install swarmopt[llm]
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option 1** — define what to search over:

```python
# train.py
search_space = {
    "lr": (1e-4, 1e-1),                    # auto-detects log-scale
    "hidden_dim": (32, 512),                # auto-detects integer
    "activation": ["relu", "gelu", "silu"], # categorical
}

def train_fn(config):
    model = build_my_model(config["hidden_dim"], config["activation"])
    # ... train, return per-epoch losses for smarter LLM decisions ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}
```

**Option 2** — just give it a model, we figure out the rest:

```python
# train.py
model = torchvision.models.resnet18(num_classes=10)  # swarmopt introspects this

def train_fn(config):
    m = config["model"].to("cuda")  # deep copy with modifications applied
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}
```

Then run:

```bash
swarmopt run train.py
```

Runs until Ctrl+C. Crash-safe, resumable. Works in notebooks too:

```python
from swarmopt import ArchSearch

search = ArchSearch(train_fn=train_fn, search_space=search_space, backend="claude")
search.run(max_evals=50)
```

## Documentation

See the [full documentation](https://loevlie.github.io/swarmopt/) for:

- [How it works](https://loevlie.github.io/swarmopt/how-it-works/) — what the LLM sees, training curve analysis
- [CLI reference](https://loevlie.github.io/swarmopt/cli/) — `swarmopt run`, `inspect`, `results`
- [Python API](https://loevlie.github.io/swarmopt/api/) — `ArchSearch`, `from_model`, search space types
- [Examples](https://loevlie.github.io/swarmopt/examples/) — CNN search, ResNet tuning, overnight results
- [Benchmarks](https://loevlie.github.io/swarmopt/benchmarks/) — vs Optuna, random search, grid search

## Installation

```bash
pip install swarmopt                # core
pip install swarmopt[llm]           # + Claude API (recommended)
pip install swarmopt[llm-openai]    # + OpenAI API
pip install swarmopt[all]           # everything
```

## License

MIT
