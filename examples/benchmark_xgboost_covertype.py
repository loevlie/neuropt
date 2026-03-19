"""
Benchmark: neuropt vs Optuna vs Random on XGBoost (Covertype).

7-class classification, 54 features, 20k samples.
Bad configs hit 50% accuracy, good ones hit 84% — big gap.

Usage:
    python examples/benchmark_xgboost_covertype.py
    python examples/benchmark_xgboost_covertype.py --n-evals 30
"""

import argparse
import json
import os
import random
import time

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ── Data ─────────────────────────────────────────────────────────────────

X, y = fetch_covtype(return_X_y=True)
y = LabelEncoder().fit_transform(y)

rng = np.random.default_rng(42)
idx = rng.choice(len(X), 20000, replace=False)
X_train, X_val, y_train, y_val = train_test_split(X[idx], y[idx], test_size=0.2, random_state=42)

print(f"Dataset: Covertype ({X_train.shape[0]:,} train, {X_val.shape[0]:,} val, "
      f"{len(set(y))} classes, {X.shape[1]} features)")


# ── Search space ─────────────────────────────────────────────────────────

SPACE = {
    "max_depth": (3, 12),
    "learning_rate": (1e-3, 0.3),
    "n_estimators": (50, 500),
    "min_child_weight": (1, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_alpha": (1e-5, 10.0),
    "reg_lambda": (1e-5, 10.0),
    "gamma": (1e-5, 10.0),
}


def evaluate(cfg):
    t0 = time.time()
    model = XGBClassifier(
        max_depth=int(cfg["max_depth"]),
        learning_rate=cfg["learning_rate"],
        n_estimators=int(cfg["n_estimators"]),
        min_child_weight=int(cfg["min_child_weight"]),
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        reg_alpha=cfg["reg_alpha"],
        reg_lambda=cfg["reg_lambda"],
        gamma=cfg["gamma"],
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    results = model.evals_result()
    tl = results["validation_0"]["mlogloss"]
    vl = results["validation_1"]["mlogloss"]
    acc = model.score(X_val, y_val)
    return vl[-1], acc, time.time() - t0, tl, vl


def random_config(rng):
    return {
        "max_depth": rng.randint(3, 12),
        "learning_rate": 10 ** rng.uniform(-3, np.log10(0.3)),
        "n_estimators": rng.randint(50, 500),
        "min_child_weight": rng.randint(1, 10),
        "subsample": rng.uniform(0.5, 1.0),
        "colsample_bytree": rng.uniform(0.5, 1.0),
        "reg_alpha": 10 ** rng.uniform(-5, 1),
        "reg_lambda": 10 ** rng.uniform(-5, 1),
        "gamma": 10 ** rng.uniform(-5, 1),
    }


# ── Methods ──────────────────────────────────────────────────────────────

def run_neuropt(backend_name, n_evals):
    from neuropt import ArchSearch

    log_path = f"/tmp/bench_covtype_{backend_name}.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    def train_fn(config):
        loss, acc, _, tl, vl = evaluate(config)
        return {"score": loss, "accuracy": acc, "train_losses": tl, "val_losses": vl}

    search = ArchSearch(
        train_fn=train_fn,
        search_space=SPACE,
        backend=backend_name,
        log_path=log_path,
        batch_size=3,
    )
    t0 = time.time()
    search.run(max_evals=n_evals)

    with open(log_path) as f:
        results = [json.loads(line) for line in f]

    scores = [r["val_loss"] for r in results if r.get("status") == "ok"]
    accs = [r.get("val_accuracy", 0) for r in results if r.get("status") == "ok"]
    return {"scores": scores, "best_loss": min(scores), "best_acc": max(accs),
            "wall_time": time.time() - t0}


def run_optuna(n_evals):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scores = []

    def objective(trial):
        cfg = {
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 10.0, log=True),
        }
        loss, acc, elapsed, _, _ = evaluate(cfg)
        scores.append({"val_loss": loss, "accuracy": acc})
        print(f"  Optuna [{len(scores)}/{n_evals}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)")
        return loss

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=3))
    t0 = time.time()
    study.optimize(objective, n_trials=n_evals)

    losses = [s["val_loss"] for s in scores]
    accs = [s["accuracy"] for s in scores]
    return {"scores": losses, "best_loss": min(losses), "best_acc": max(accs),
            "wall_time": time.time() - t0}


def run_random(n_evals):
    rng = random.Random(42)
    scores = []
    t0 = time.time()

    for i in range(n_evals):
        cfg = random_config(rng)
        loss, acc, elapsed, _, _ = evaluate(cfg)
        scores.append({"val_loss": loss, "accuracy": acc})
        print(f"  Random [{i+1}/{n_evals}] loss={loss:.4f} acc={acc:.4f} ({elapsed:.1f}s)")

    losses = [s["val_loss"] for s in scores]
    accs = [s["accuracy"] for s in scores]
    return {"scores": losses, "best_loss": min(losses), "best_acc": max(accs),
            "wall_time": time.time() - t0}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-evals", type=int, default=15)
    args = parser.parse_args()

    print(f"Budget: {args.n_evals} evaluations per method")
    print(f"Search space: 9 XGBoost hyperparameters\n")

    all_results = {}

    print("=" * 60)
    print("neuropt (Claude)")
    print("=" * 60)
    try:
        all_results["neuropt (Claude)"] = run_neuropt("claude", args.n_evals)
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n" + "=" * 60)
    print("Optuna TPE (n_startup_trials=3)")
    print("=" * 60)
    all_results["Optuna TPE"] = run_optuna(args.n_evals)

    print("\n" + "=" * 60)
    print("Random Search")
    print("=" * 60)
    all_results["Random"] = run_random(args.n_evals)

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS ({args.n_evals} evals each)")
    print("=" * 60)
    print(f"{'Method':<22} {'Best Loss':>10} {'Best Acc':>10}")
    print("-" * 45)
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["best_loss"]):
        print(f"{name:<22} {r['best_loss']:>10.4f} {r['best_acc']:>10.4f}")

    # Convergence
    print(f"\nConvergence (best-so-far):")
    milestones = [m for m in [5, 10, 15, 20, 25, 30, 50] if m <= args.n_evals]
    header = f"{'Eval':>6}" + "".join(f"{name:>22}" for name in all_results.keys())
    print(header)
    for m in milestones:
        line = f"{m:>6}"
        for name, r in all_results.items():
            s = r["scores"][:m]
            best = min(s) if s else float("inf")
            line += f"{best:>22.4f}"
        print(line)

    out_path = "benchmark_xgboost_covertype.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
