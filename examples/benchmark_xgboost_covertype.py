"""
Benchmark: neuropt vs Optuna vs Random on XGBoost (Covertype).

7-class classification, 54 features, 20k samples.
Bad configs hit 50% accuracy, good ones hit 84% — big gap.

Usage:
    python examples/benchmark_xgboost_covertype.py
    python examples/benchmark_xgboost_covertype.py --n-evals 30
"""

import argparse
import time

import numpy as np
from benchmark_utils import (
    header,
    print_convergence,
    print_summary,
    run_neuropt,
    run_optuna,
    run_random,
    save_results,
)
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


def train_fn(config):
    loss, acc, _, tl, vl = evaluate(config)
    return {"score": loss, "accuracy": acc, "train_losses": tl, "val_losses": vl}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-evals", type=int, default=15)
    args = parser.parse_args()

    print(f"Budget: {args.n_evals} evaluations per method")
    print("Search space: 9 XGBoost hyperparameters")

    all_results = {}

    header("neuropt (Claude)")
    try:
        all_results["neuropt (Claude)"] = run_neuropt(
            train_fn, SPACE, "claude", args.n_evals, "/tmp/bench_covtype_claude.jsonl")
    except Exception as e:
        print(f"  Skipped: {e}")

    header("Optuna TPE (n_startup_trials=3)")
    all_results["Optuna TPE"] = run_optuna(evaluate, SPACE, args.n_evals)

    header("Random Search")
    all_results["Random"] = run_random(evaluate, SPACE, args.n_evals)

    print_summary(all_results, args.n_evals)
    print_convergence(all_results, args.n_evals)
    save_results(all_results, "benchmark_xgboost_covertype.json")


if __name__ == "__main__":
    main()
