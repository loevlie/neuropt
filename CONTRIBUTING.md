# Contributing to neuropt

Thanks for your interest! Here are some ways to help:

## Good first contributions

- **Try it on a new dataset/model** and share your results in an issue
- **Add a new LLM backend** — see `neuropt/backends/` for the pattern (just implement `generate` and `is_available`)
- **Improve the prompt** — the system prompt in `arch_search.py` can always be better. If you find phrasing that gets better results, open a PR
- **Add tests** — especially for edge cases in config validation, dedup logic, and new model types

## Bigger ideas

- **Multi-objective optimization** — optimize for accuracy AND inference speed simultaneously
- **Architecture generation** — go beyond swapping components to designing new architectures from scratch
- **Better local LLM support** — the Qwen backend works but has a high parse failure rate on complex search spaces. Prompt engineering or constrained decoding could help

## Setup

```bash
git clone https://github.com/loevlie/neuropt.git
cd neuropt
uv sync
uv run neuropt --help
```

## Running tests

```bash
uv run pytest
```

## PR guidelines

- Keep changes focused — one thing per PR
- Add a test if you're changing logic
- Run `uv run neuropt run examples/train_fashion.py --backend none -n 3` to verify nothing is broken
