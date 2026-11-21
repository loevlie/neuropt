# How it works

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

Each iteration:

1. Build a prompt with the search space, best result, last 20 experiments, and per-epoch training curves
2. Ask the LLM to propose a batch of configs as JSON
3. Validate and clamp the response (fall back to random on any parse failure)
4. Train each config, tracking per-epoch train loss, val loss, and val accuracy
5. Log everything to JSONL, update best
6. Repeat

## What the LLM sees

Most tuning tools give the optimizer a single number: *"this config scored 0.85."*

neuropt shows the full picture:

```
lr=0.05, activation=relu, use_residual=False:
  ep1:  train=2.30  val=2.28  acc=0.12
  ep2:  train=1.45  val=1.52  acc=0.41
  ep3:  train=0.82  val=1.35  acc=0.53
  ep4:  train=0.31  val=1.61  acc=0.48   ← val rising = overfitting
  ep5:  train=0.09  val=1.89  acc=0.45

lr=8.8e-4, activation=gelu, use_residual=True:
  ep1:  train=1.92  val=1.85  acc=0.28
  ep2:  train=1.01  val=0.98  acc=0.62
  ep3:  train=0.62  val=0.71  acc=0.74
  ep4:  train=0.41  val=0.52  acc=0.81   ← both dropping = good fit
  ep5:  train=0.33  val=0.43  acc=0.85
```

The system also pre-computes signals like `OVERFITTING: train 2.30→0.09, val 1.52→1.89, gap=1.80` so the LLM doesn't have to do the math.

## Why this works better than Bayesian optimization

Bayesian methods (Optuna TPE, Gaussian processes) build a statistical surrogate from `(config, score)` pairs. They learn correlations but have no domain knowledge — they don't know that high learning rates cause oscillation, or that dropout hurts underfitting models.

The LLM starts with ML knowledge and reads the training curves to understand *why* something worked or didn't. It can reason about interactions: "this config overfits because the model is too big for this dataset and has no regularization" rather than just "this config scored poorly."

## Fallback behavior

The model's answer must match a JSON schema built from the search space (structured outputs, where
the API supports it). Each config is still checked: numbers are clamped into their range, a number
between two numeric choices snaps to the nearest one (96 becomes 128 for `[64, 128, 256]`), and a
string that differs only in case matches its choice. A config that is still invalid is replaced by a
random one; the other configs in the answer are kept. If the whole answer is unusable, or the API is
down, all configs for that round are random.

Nothing raises: the search keeps going. But every eval's source (`llm`, `random`, `initial`) is
logged, and the end-of-run summary says how many configs were random and warns when it is more than
10%, so a broken key or model doesn't quietly turn into random search.

## LLM backends

Auto-detected in order:

| Backend | How | Notes |
|---------|-----|-------|
| OpenAI-compatible server | `NEUROPT_LLM_BASE_URL` + `NEUROPT_MODEL`, or `--base-url` / `--model` | Ollama, vLLM, llama.cpp, LM Studio, OpenRouter |
| Claude | `ANTHROPIC_API_KEY` | Claude Haiku 5.5 by default; about 4 s per call in the [harness](evals.md) |
| OpenAI | `OPENAI_API_KEY` | GPT-4o mini by default |
| Random search | `--backend none`, or nothing configured | Baseline; the run header says so |

`--model` (or `NEUROPT_MODEL`) picks the model; a `claude-*` model selects the Claude backend. The
old local Qwen 2.5 1.5B backend (`--backend qwen`) is deprecated and no longer auto-detected; run a
local model behind Ollama or vLLM instead.
