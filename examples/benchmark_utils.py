"""Shared helpers for the benchmark example scripts.

Each benchmark script defines its own data setup, ``evaluate(cfg)`` function
returning ``(loss, accuracy, elapsed, ...)``, and a neuropt-style search space
dict. Everything else — running each method, reading the search log, and
printing the result tables — lives here.
"""

import json
import os
import random
import time

from neuropt.arch_search import _normalize_search_space, _random_config
from neuropt.search_space import Categorical, IntUniform, LogUniform


def read_search_log(path):
    """Load an ArchSearch JSONL log → (losses, accuracies) for successful rows.

    Handles both the current format ({"score", "scalars": {"accuracy": ...}})
    and the legacy format ({"val_loss", "val_accuracy"}).
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    losses = [s for s in (r.get("score", r.get("val_loss")) for r in ok)
              if s is not None]
    accs = []
    for r in ok:
        scalars = r.get("scalars", {})
        acc = scalars.get("accuracy", scalars.get("val_accuracy", r.get("val_accuracy")))
        if acc:
            accs.append(acc)
    return losses, accs


def summarize(losses, accs, wall_time, **extra):
    """Standard per-method result dict used by the summary/convergence tables."""
    return {
        "scores": losses,
        "best_loss": min(losses) if losses else float("inf"),
        "best_acc": max(accs) if accs else 0,
        "wall_time": wall_time,
        "n_evals": len(losses),
        **extra,
    }


def run_neuropt(train_fn, space, backend_name, n_evals, log_path, **kwargs):
    """Run a neuropt ArchSearch and summarize its log."""
    from neuropt import ArchSearch

    if os.path.exists(log_path):
        os.remove(log_path)

    search = ArchSearch(train_fn=train_fn, search_space=space,
                        backend=backend_name, log_path=log_path,
                        batch_size=3, **kwargs)
    t0 = time.time()
    search.run(max_evals=n_evals)
    losses, accs = read_search_log(log_path)
    return summarize(losses, accs, time.time() - t0,
                     llm_success=search.llm_success,
                     llm_fallback=search.llm_fallback)


def _suggest_config(trial, space):
    """Map a normalized neuropt search space onto Optuna suggest calls."""
    cfg = {}
    for name, dim in space.items():
        if isinstance(dim, Categorical):
            cfg[name] = trial.suggest_categorical(name, dim.choices)
        elif isinstance(dim, IntUniform):
            cfg[name] = trial.suggest_int(name, dim.low, dim.high)
        elif isinstance(dim, LogUniform):
            cfg[name] = trial.suggest_float(name, dim.low, dim.high, log=True)
        else:
            cfg[name] = trial.suggest_float(name, dim.low, dim.high)
    return cfg


def run_optuna(evaluate_fn, space, n_evals, seed=42):
    """Optuna TPE over the same search space. evaluate_fn(cfg) -> (loss, acc, elapsed, ...)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = _normalize_search_space(space)
    losses, accs = [], []

    def objective(trial):
        cfg = _suggest_config(trial, space)
        loss, acc, elapsed = evaluate_fn(cfg)[:3]
        losses.append(loss)
        accs.append(acc)
        print(f"  Optuna [{len(losses)}/{n_evals}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)")
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3))
    t0 = time.time()
    study.optimize(objective, n_trials=n_evals)
    return summarize(losses, accs, time.time() - t0)


def run_random(evaluate_fn, space, n_evals, seed=42):
    """Random search over the same search space."""
    space = _normalize_search_space(space)
    rng = random.Random(seed)
    losses, accs = [], []
    t0 = time.time()
    for i in range(n_evals):
        cfg = _random_config(space, rng)
        loss, acc, elapsed = evaluate_fn(cfg)[:3]
        losses.append(loss)
        accs.append(acc)
        print(f"  Random [{i+1}/{n_evals}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)")
    return summarize(losses, accs, time.time() - t0)


def header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_summary(all_results, n_evals, show_time=False):
    header(f"RESULTS ({n_evals} evals each)")
    cols = f"{'Method':<22} {'Best Loss':>10} {'Best Acc':>10}"
    if show_time:
        cols += f" {'Wall Time':>10}"
    print(cols)
    print("-" * len(cols))
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["best_loss"]):
        line = f"{name:<22} {r['best_loss']:>10.4f} {r['best_acc']:>10.4f}"
        if show_time:
            line += f" {r['wall_time']:>9.1f}s"
        print(line)


def print_convergence(all_results, n_evals, col_width=22):
    print("\nConvergence (best-so-far):")
    milestones = [m for m in [5, 10, 15, 20, 25, 30, 40, 50] if m <= n_evals]
    print(f"{'Eval':>6}" + "".join(f"{name:>{col_width}}" for name in all_results))
    for m in milestones:
        line = f"{m:>6}"
        for r in all_results.values():
            s = r["scores"][:m]
            best = min(s) if s else float("inf")
            line += f"{best:>{col_width}.4f}"
        print(line)


def save_results(all_results, out_path):
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
