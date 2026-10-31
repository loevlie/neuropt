# neuropt

<p align="center">
  <a href="https://pypi.org/project/neuropt/"><img alt="PyPI" src="https://img.shields.io/pypi/v/neuropt"></a>
  <a href="https://github.com/loevlie/neuropt/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/loevlie/neuropt/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/loevlie/neuropt/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://pypi.org/project/neuropt/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/neuropt"></a>
  <a href="https://colab.research.google.com/github/loevlie/neuropt/blob/main/examples/neuropt_demo.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center">
  <img src="assets/banner.png" alt="Three robot researchers designing neural network architectures " width="700">
</p>

<p align="center">
  <em>An LLM reads your training curves and designs your next experiment.</em>
</p>

---

<p align="center">
  <a href="https://loevlie.github.io/neuropt/quickstart/"><strong>Get started with neuropt →</strong></a>
</p>

Point it at a training script, let it run overnight. The LLM sees full per-epoch train/val curves, spots overfitting, and proposes what to try next — like a research assistant who never sleeps and actually reads the loss plots.

### vs Optuna and random search

<p align="center">
  <img src="assets/benchmark.png" alt="Benchmark: neuropt vs Optuna vs Random" width="700">
</p>

Same 15-eval budget on two tasks: CNN architecture search (14 params) and XGBoost tuning (9 params, 7-class Covertype). These results use **Claude Haiku 4.5** (the smallest and cheapest of their 4.5 models). We expect even stronger results with Sonnet or Opus. Optuna's TPE was configured with `n_startup_trials=3` for a fair comparison (default is 10, which would make it purely random for most of the budget).

In a separate 200-eval run on the CNN task, neuropt again beat the others within 15 evals and kept improving — reaching 0.337 val loss by eval 200 (vs 0.454 for Optuna's best at 15). Local Qwen backend (experimental) also beat Optuna at 15 evals (0.440 vs 0.454) despite a 40% JSON parse failure rate.

<p align="center">
  <img src="assets/search_combined_15.png" alt="2-param search: neuropt vs Optuna vs Random" width="700">
</p>

## Quick start

```bash
pip install "neuropt[llm]"
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
model = torchvision.models.resnet18(num_classes=10)  # neuropt introspects this

def train_fn(config):
    m = config["model"].to("cuda")  # deep copy with modifications applied
    # ... train ...
    return {"score": val_loss, "train_losses": [...], "val_losses": [...]}
```

Then run:

```bash
neuropt run train.py
```

Runs until Ctrl+C. Crash-safe, resumable. Works in notebooks too:

```python
from neuropt import ArchSearch

search = ArchSearch(train_fn=train_fn, search_space=search_space, backend="claude")
search.run(max_evals=50)
```

### Or let a decision model choose (experimental)

> [!NOTE]
> **Still evaluating.** `DecisionSearch` works end to end, but it's new and the results below are preliminary: one seed per task. Its defaults may change as we benchmark it further.

`DecisionSearch` uses a [Jev](https://openrouter.ai/docs/guides/community/jev.md)-style **decision model** instead of a generative LLM. A decision model answers multiple-choice questions with calibrated probabilities instead of writing text. neuropt asks it one question per hyperparameter ("which `lr` range is most likely to beat the current best?"), samples candidates from its answers, then asks it to pick among them. The default is the open-weight [H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b), which runs locally through llama.cpp (Apple Silicon included). Any server with the same `/v1/systemone` API also works.

<p align="center">
  <img src="assets/benchmark_decision_cnn.png" alt="CNN benchmark: H2O-Lightning-4B decision model vs Claude Haiku 4.5, Optuna and random search" width="680">
</p>

| CNN search · 14 params · 15 evals | Best val loss | Best accuracy | Best after 5 evals |
|---|:---:|:---:|:---:|
| **neuropt · H2O-Lightning-4B decision model** | **0.363** | **86.9%** | **0.388** |
| neuropt · Claude Haiku 4.5 | 0.385 | 85.4% | 0.401 |
| Optuna TPE | 0.454 | 82.7% | 0.612 |
| Random search | 0.610 | 76.7% | 0.655 |

On the CNN task its first picks were textbook (residual connections, AdamW, batch norm, max pooling), and it was ahead from the second evaluation. On the 9-parameter XGBoost task it currently trails (0.593 vs 0.415 for Optuna) because it stayed with learning rates that were too low. That's the next thing we're working on.

```python
from neuropt import DecisionSearch

search = DecisionSearch(train_fn=train_fn, search_space=search_space, backend="decision")
search.run(max_evals=50)
```

```bash
neuropt run train.py --decision-url http://gpu-box:8741
```

See [Decision Models](https://loevlie.github.io/neuropt/decision-models/) for server setup and how it decides. To reproduce the benchmark: `python examples/benchmark.py --skip-claude --skip-qwen --decision-url http://127.0.0.1:8741`.

## Documentation

See the [full documentation](https://loevlie.github.io/neuropt/) for:

- [How it works](https://loevlie.github.io/neuropt/how-it-works/) — what the LLM sees, training curve analysis
- [CLI reference](https://loevlie.github.io/neuropt/cli/) — `neuropt run`, `inspect`, `results`
- [Python API](https://loevlie.github.io/neuropt/api/) — `ArchSearch`, `from_model`, search space types
- [Model Introspection](https://loevlie.github.io/neuropt/introspection/) — `from_model()` auto-detection, activations, dropout, pooling
- [Fine-Tuning](https://loevlie.github.io/neuropt/fine-tuning/) — pretrained models, freeze strategies, L2-SP
- [Decision Models](https://loevlie.github.io/neuropt/decision-models/) — `DecisionSearch` with H2O-Lightning-4B or any Jev-compatible server (experimental)
- [Examples](https://loevlie.github.io/neuropt/examples/) — CNN search, ResNet tuning
- [Benchmarks](https://loevlie.github.io/neuropt/benchmarks/) — vs Optuna, random search

## Installation

```bash
pip install neuropt                  # core
pip install "neuropt[llm]"           # + Claude API (recommended)
pip install "neuropt[llm-openai]"    # + OpenAI API
pip install "neuropt[all]"           # everything
```

## License

MIT
