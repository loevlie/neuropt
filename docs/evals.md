# Evaluations

How neuropt is evaluated, what was decided before looking at results, and what was found, including
negative results. Everything here is produced by the harness in `benchmarks/`:

```bash
python -m benchmarks.run     # every task x method x seed; finished runs are skipped
python -m benchmarks.plot    # figures and tables from the cached runs
```

## Protocol (fixed 19 Nov 2026, before any harness run)

**Tasks.** Each has a fixed train / val / test split (data seed fixed, independent of the run seed).

| Task | Data | Train / val / test | Search space |
|---|---|---|---|
| `cnn` | FashionMNIST, 32x32 | 5,000 / 1,250 (both from the official training split) / 10,000 (official test split) | 14 params, architecture + training, 5 epochs |
| `xgb` | Covertype, 7 classes | 16,000 / 4,000 / 10,000 (disjoint random rows) | 9 XGBoost params |
| `mlp` | Higgs-small (OpenML 23512) | 20,000 / 5,000 / 20,000 (disjoint random rows) | 9 params, 10 epochs |

**What the optimizer sees.** Only train and validation information: the validation loss as the score,
validation accuracy, and per-epoch train / val curves. The test split is scored by the harness for
every trained config and written to the result file; it is never returned to an optimizer and never
appears in a prompt or a decision-model state. `tests/test_benchmarks.py` checks this with a canary.

**Reported number.** At each eval count, take the config with the best validation loss so far (the
incumbent) and report its **test** loss. Validation best-so-far is shown next to it.

**Methods.**

| Method | What it is |
|---|---|
| `random` | Uniform random search |
| `tpe` | Optuna TPE with Optuna's default settings |
| `neuropt-<model>` | `ArchSearch` with that LLM, batch size 3. Phase 1 uses the package's two API defaults, `haiku-4.5` and `gpt-4o-mini` (the second was added on 19 Nov 2026 after the classical arms had run, when the Anthropic account ran out of credit; no LLM results had been looked at for it) |
| `jev` | `DecisionSearch` with hosted Jev (`jev-latest`) |
| `<method>+default` | The same method, but its first eval is the task's default config |

The `+default` variants are the control from arXiv 2606.21641, which found that an LLM advisor's
advantage on tabular HPO came from the default config it started with. A `+default` LLM arm is
compared with `tpe+default` and `random+default`, never with an unseeded baseline.

**Default configs** (fixed here, not tuned): XGBoost's library defaults (regularisation terms that
default to 0 are set to the lower bound of their log range); for the CNN and MLP a plain textbook
setup (AdamW, lr 1e-3, ReLU, batch norm). The exact values are in `benchmarks/tasks.py`.

**Budget and fairness.**

- 30 evals per run, reported at 5, 10, 15 and 30. Every eval counts, including failed ones.
- A failed eval (error, non-finite loss, or over the time limit) scores as a uniform guess,
  `ln(n_classes)`. The same time limit applies to every method.
- 5 seeds (0 to 4) per method and task. The run seed sets the optimizer's randomness and the
  training seed of each eval, so every method trains with the same sequence of training seeds.
- Also recorded per run: wall-clock time, tokens, API cost, LLM latency, parse failures, LLM errors,
  duplicate configs, and how many evals fell back to random configs.

**Decision rule.** Variant A "helps" over B on a task only if both hold at 30 evals:

1. A's mean incumbent test loss over the 5 seeds is lower than B's, and
2. A is better than B on at least 4 of the 5 seeds (paired by seed).

Anything else is reported as "no measurable difference". A change goes into the package defaults only
if it helps on at least 2 of the 3 tasks and is not worse (by the same rule) on the third. Cost is
reported next to every score; a variant that helps but costs more is described as such.

## Phase 0: what the old benchmark showed (19 Nov 2026)

Re-running the README benchmark scripts (one seed, 15 evals, M1 Max) before building the harness:

| Best val loss after 15 evals | CNN | XGBoost Covertype |
|---|:---:|:---:|
| neuropt, Claude Haiku 4.5 | 0.371 (README: 0.385) | 0.425 (docs: about 0.414) |
| Optuna TPE (`n_startup_trials=3`) | 0.392 (README: 0.454) | 0.443 (docs: 0.415) |
| Random search | 0.624 (README: 0.610) | 0.418 (docs: 0.418) |
| neuropt, hosted Jev | 0.332 | 0.511 |

Problems with that benchmark, which the protocol above fixes:

1. No held-out test set. The CNN "validation" set was a subset of the FashionMNIST test split, and the
   score shown to the optimizer was the score reported.
2. One seed, and training itself was unseeded. Optuna moved from 0.454 to 0.392 between two runs of
   the same script, which is most of its published gap to Claude.
3. Failed evals were dropped from the curves, so "best after N" meant N successes, not N evals.
4. Only the neuropt arms had a time limit on the CNN task.
5. No default-seeded baseline.
6. The local Qwen backend failed on every call with transformers 4.53 (`dtype` argument) and ran as
   random search while still being labelled as Qwen.
7. On XGBoost, random search beat neuropt with Claude Haiku 4.5 in this run.

## Results

Not run yet.
