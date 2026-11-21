# CLI Reference

## `neuropt run`

Run LLM-guided search on a training script.

```bash
neuropt run train.py
neuropt run train.py --backend claude
neuropt run train.py --backend none -n 50 --log results.jsonl
neuropt run train.py --backend decision --decision-url http://gpu-box:8741
neuropt run train.py --model claude-haiku-5-5
neuropt run train.py --base-url http://localhost:11434/v1 --model qwen3.5:9b   # local model via Ollama
```

Your script must define:

- `train_fn(config)` — returns dict with at least `{"score": float}`
- `search_space` dict **or** `model` (nn.Module) — one of the two

Optional: `ml_context` string with domain knowledge for the LLM,
and `minimize = False` if higher scores are better.

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | auto | `auto`, `claude`, `openai`, `openai-compatible`, `none`, `decision` ([decision model](decision-models.md)); `qwen` is deprecated |
| `-m` / `--model` | `$NEUROPT_MODEL`, else the backend's default | Model id, e.g. `claude-haiku-5-5`, `gpt-4o-mini`, `qwen3.5:9b`. A `claude-*` model selects the Claude backend |
| `--base-url` | `$NEUROPT_LLM_BASE_URL` | OpenAI-compatible server (Ollama, vLLM, llama.cpp, LM Studio, OpenRouter); needs `--model` |
| `--decision-url` | `$NEUROPT_DECISION_URL` or `http://127.0.0.1:8741` | Decision server root (implies `--backend decision`) |
| `--log` | search.jsonl | Log file path (supports resume) |
| `-b` / `--batch-size` | 3 | Configs proposed per LLM call |
| `-n` / `--max-evals` | unlimited | Stop after N experiments |
| `--device` | auto | `cuda`, `mps`, `cpu` |
| `--timeout` | 600 | Max seconds per experiment |
| `--maximize` | off | Higher scores are better (accuracy, AUROC) |

## `neuropt inspect`

Show what neuropt would search over for a given model.

```bash
neuropt inspect train.py
```

```
Model: 11,689,512 parameters
  Activations: ReLU (9 layers)
  BatchNorm: 20 layers
  Pooling: 1 layers (current: avg)

Search space (7 params):
  activation: Categorical(['relu', 'gelu', 'silu', 'leaky_relu', 'mish', 'hardswish', 'prelu'])
  use_batchnorm: Categorical([True, False])
  pool_type: Categorical(['avg', 'max', 'attention'])
  lr: LogUniform(0.0001, 0.1)
  wd: LogUniform(1e-06, 0.01)
  optimizer: Categorical(['sgd', 'adam', 'adamw'])
```

Only works with scripts that define a `model` variable.

## `neuropt results`

Analyze a search log.

```bash
neuropt results search.jsonl
neuropt results search.jsonl --top 20
neuropt results search.jsonl --maximize   # if the search ran with --maximize
```

Shows total experiments, top N results with configs, and convergence over time.
