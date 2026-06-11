"""
Benchmark: neuropt (Claude) vs Optuna TPE vs Random on XGBoost.

Same evaluation budget, same search space, same data.

Usage:
    python examples/benchmark_xgboost.py
    python examples/benchmark_xgboost.py --n-evals 30
"""

import argparse
import time

from benchmark_utils import (
    header,
    print_convergence,
    print_summary,
    run_neuropt,
    run_optuna,
    run_random,
    save_results,
)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ── Data ─────────────────────────────────────────────────────────────────

X, y = load_breast_cancer(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Search space (shared) ────────────────────────────────────────────────

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
    """Train an XGBClassifier and return (logloss, accuracy, elapsed, train_losses, val_losses)."""
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
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    results = model.evals_result()
    train_losses = results["validation_0"]["logloss"]
    val_losses = results["validation_1"]["logloss"]
    accuracy = model.score(X_val, y_val)
    return val_losses[-1], accuracy, time.time() - t0, train_losses, val_losses


def train_fn(config):
    loss, acc, _, tl, vl = evaluate(config)
    return {"score": loss, "accuracy": acc, "train_losses": tl, "val_losses": vl}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark: neuropt vs Optuna vs Random (XGBoost)")
    parser.add_argument("--n-evals", type=int, default=15)
    args = parser.parse_args()

    print(f"Dataset: Breast Cancer ({X_train.shape[0]} train, {X_val.shape[0]} val)")
    print(f"Budget: {args.n_evals} evaluations per method")
    print("Search space: 9 XGBoost hyperparameters")

    all_results = {}

    header("neuropt (Claude)")
    try:
        all_results["neuropt (Claude)"] = run_neuropt(
            train_fn, SPACE, "claude", args.n_evals, "/tmp/bench_xgb_claude.jsonl")
    except Exception as e:
        print(f"  Skipped: {e}")

    header("Optuna TPE (n_startup_trials=3)")
    all_results["Optuna TPE"] = run_optuna(evaluate, SPACE, args.n_evals)

    header("Random Search")
    all_results["Random"] = run_random(evaluate, SPACE, args.n_evals)

    print_summary(all_results, args.n_evals)
    print_convergence(all_results, args.n_evals, col_width=18)
    save_results(all_results, "benchmark_xgboost_results.json")


if __name__ == "__main__":
    main()
