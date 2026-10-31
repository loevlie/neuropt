# Decision Models

!!! warning "Experimental: still evaluating"
    `DecisionSearch` works end to end, but it's new and has only been benchmarked on one seed per task.
    Defaults may change as we evaluate it further.

`DecisionSearch` lets a **decision model** pick what to try next instead of a generative LLM.

## What a decision model is

[Jev](https://openrouter.ai/docs/guides/community/jev.md), from TypeSafe AI, started this kind of model. It doesn't write text. You give it a `state` (the evidence) and typed questions. It answers each question with **calibrated probabilities** over the options you defined, in one forward pass, in tens of milliseconds:

```json
POST /v1/systemone
{"state": "...experiment history...",
 "questions": {"p0": {"type": "choice", "instructions": "Which lr range ...?",
                      "criteria": {"1.00e-04 to 2.37e-04": "...", "...": "..."}}}}

→ {"answers": {"p0": {"choice": "1.33e-03 to 3.16e-03",
                      "probabilities": {"1.00e-04 to 2.37e-04": 0.02, "...": 0.61}}}}
```

Because the model can only choose among options you give it, neuropt turns the search into questions.

## Default model: H2O-Lightning-4B

neuropt targets the TypeSafe `/v1/systemone` request format, so any compatible server works. The default is **[H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b)** (Apache-2.0, a Qwen3.5-4B fine-tune):

- On the independent [JevBench](https://github.com/fstandhartinger/jevbench) v1.6.1 board it has the best **Composite** score of any open-weight model: 72.52, against 71.49 for Jev 1.13.0 itself.
- At 4B parameters it fits on one GPU. Its `serve.sh` exposes `/v1/systemone` directly, with a 40k-token context and up to 255 options per question.

| Open alternative | Size | Why not the default |
|---|---|---|
| Quyet-1.0-Large | 31B | Highest raw capability, but needs an 80 GB GPU, allows at most 10 options per question, and has no server |
| decisio (Gemma-4-31B) | 31B | Strong, but far heavier |
| JevK5 / Kev | 4B, 0.8–9B | Rank much lower on JevBench (open rank 25 and 64 or lower) |
| Laya | 0.4B | Near chance outside its training domain |

## Start the server

You need a CUDA GPU with 24 GB or more for vLLM. The server doesn't have to run on the machine that trains: point neuropt at it over HTTP.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "vllm==0.30.0" "torchcodec==0.17.0" "transformers==5.20.2"
hf download h2oai/h2o-lightning-4b --local-dir h2o-lightning-4b
TWO_WAY=0 bash h2o-lightning-4b/serve.sh      # listens on http://127.0.0.1:8741
```

Check the [model card](https://huggingface.co/h2oai/h2o-lightning-4b) for the current serving instructions. For example, the default two-way mode needs H2O's vLLM fork.

**On a Mac or a CPU box**, llama.cpp build b11773 or later serves `/v1/systemone` natively for the [GGUF files](https://huggingface.co/h2oai/h2o-lightning-4b-gguf). Q8_0 is H2O's recommended size: 4.5 GB, and 99.6% of its answers match the full-precision model.

```bash
llama-server -hf h2oai/h2o-lightning-4b-gguf:Q8_0 --alias h2oai/h2o-lightning-4b \
    -c 40960 -np 1 --no-kv-unified --no-mmproj --host 127.0.0.1 --port 8741
```

On Apple Silicon the server and PyTorch share memory. If training is on the same machine, oversized configs can make it swap; one PyTorch run can then stall past its timeout. Capping PyTorch's share (for example `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25`) turns that into a quick out-of-memory failure instead.

## Use it

```python
from neuropt import DecisionSearch

search = DecisionSearch(
    train_fn,
    search_space={"lr": (1e-4, 1e-1), "hidden_dim": (32, 512), "act": ["relu", "gelu", "silu"]},
    backend="decision",              # or a DecisionBackend(...) instance
)
search.run(max_evals=50)
```

`DecisionSearch` takes everything `ArchSearch` does, including `from_model`, `log_path`, `batch_size`, `minimize` and `ml_context`. Logs and resume work the same way.

```bash
export NEUROPT_DECISION_URL=http://gpu-box:8741
neuropt run train.py --backend decision
neuropt run train.py --decision-url http://gpu-box:8741 -n 30
```

With `backend="auto"`, `DecisionSearch` uses the decision server if one answers at `$NEUROPT_DECISION_URL` (default `http://127.0.0.1:8741`). Otherwise it falls back to random search.

### Other servers

Change the URL (and model id) to use any other `/v1/systemone` server:

```python
import os

from neuropt.backends import DecisionBackend

DecisionBackend(base_url="http://localhost:8090", model="alibiserikbay/JevK5")   # jevk5-serve
DecisionBackend(base_url="https://openrouter.ai/api", model="typesafe/jev-1.14",
                api_key=os.environ["OPENROUTER_API_KEY"])                       # hosted Jev
```

Running the same search against hosted Jev is an easy way to compare the open model with the original.

## How it decides

Each iteration:

1. **State.** The model reads the same evidence the LLM backend gets: your `ml_context`, the config space, the best result, the last 20 experiments, per-epoch curves and overfitting/underfitting signals.
2. **One question per hyperparameter**, all in one request:
    - **Categorical** parameters offer every value.
    - **Numeric** parameters are split into `n_bins` ranges (default 8): log-spaced for `LogUniform`, one option per integer for small `IntUniform` ranges.
    - Every option says how often it was tried, its best score, and whether the current best config uses it.
3. **Candidates.** `n_candidates` configs (default 12) are sampled from the answers, with `explore` (default 0.15) of uniform probability mixed in. The first candidate takes the top answer for every parameter and keeps the best config's exact value wherever the model chose that value's range. Configs already tried are resampled.
4. **Joint pick** (`rerank=True`). The model chooses among the candidates as whole configs, which catches combinations the per-parameter answers can't see. The top `batch_size - 1` run, plus one sampled from the rest as exploration.

Each iteration prints the model's top answers, for example:

```
  decision: act=gelu (0.81) | lr=1.33e-03 to 3.16e-03 (0.62) | bn=true (0.58)
  joint pick: c3 (0.41), c1 (0.22), c9 (0.05)
```

If the server errors, that iteration falls back to random configs and the search keeps going.

## Results so far

Preliminary, one seed per task, with the README's benchmark scripts. The decision model (H2O-Lightning-4B, Q8_0 on llama.cpp) was run on an Apple M5 Pro. The other CNN numbers are from the original README run.

| 15 evals | Decision model | Claude Haiku 4.5 | Optuna TPE | Random |
|---|:---:|:---:|:---:|:---:|
| CNN, 14 params | **0.363** | 0.385 | 0.454 | 0.610 |
| XGBoost Covertype, 9 params | 0.593 | ~0.414 | **0.415** | 0.418 |

On XGBoost it never left low learning rates (at most 0.008), so every model was undertrained. The per-parameter questions currently favour the current best's range too strongly. Improving that is the next step.

## Limitations

- The per-parameter questions are answered independently, so they can't see each other's answers. The joint pick is what accounts for interactions between parameters.
- Numeric values are chosen at bin granularity and refined by sampling within the bin. Raise `n_bins` for finer control.
- A parameter with more than 255 options isn't asked about; it's sampled uniformly.
- Decision models are trained for classification-style judgments. How well they read training curves isn't something JevBench measures, so compare against `ArchSearch` on your own task.
