"""Run the evaluation harness: every task x method x seed, cached as JSONL.

    python -m benchmarks.run                              # everything (finished runs are skipped)
    python -m benchmarks.run --tasks xgb --methods random tpe+default --seeds 0 1
    python -m benchmarks.run --smoke                      # 1 seed, 4 evals, separate output folder

One file per run: ``benchmarks/runs/<task>/<method>/seed<k>.jsonl`` with one row per eval and a final
summary row. A run without a summary row is incomplete and is run again from scratch.
"""

import argparse
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

from benchmarks.methods import ALL_METHODS, LLMCallFailed, Unavailable, run_method
from benchmarks.tasks import TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
N_EVALS, SEEDS, TIMEOUT = 30, [0, 1, 2, 3, 4], 120


class _TimeLimit(Exception):
    pass


@contextmanager
def _time_limit(seconds):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise _TimeLimit()

    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


class EvalLog:
    """The only way a method can evaluate a config.

    ``log(config)`` trains the config and returns the public (train / val) result. The test metrics
    stay in ``log.rows``. Failed evals count against the budget and score ``task.fail_score``.
    """

    def __init__(self, task, evaluate, seed, n_evals, timeout=TIMEOUT, verbose=True):
        self.task, self.evaluate, self.seed = task, evaluate, seed
        self.n_evals, self.timeout, self.verbose = n_evals, timeout, verbose
        self.rows = []

    @property
    def n(self):
        return len(self.rows)

    def __call__(self, config, source=""):
        if self.n >= self.n_evals:
            raise RuntimeError(f"budget of {self.n_evals} evals exceeded")
        config = {k: v for k, v in config.items() if k != "device"}
        train_seed = self.seed * 1000 + self.n  # same sequence for every method
        start = time.time()
        status, error, public, private = "ok", "", {}, {}
        try:
            with _time_limit(self.timeout):
                public, private = self.evaluate(config, train_seed)
            if not math.isfinite(public["score"]):
                status = "diverged"
        except _TimeLimit:
            status, error = "timeout", f">{self.timeout}s"
        except Exception as e:
            status, error = "error", f"{type(e).__name__}: {e}"[:200]
        if status != "ok":
            public, private = {"score": self.task.fail_score, "status": status}, {}
        self.rows.append({
            "type": "eval", "eval": self.n + 1, "source": source, "config": config,
            "val_score": public["score"], "val_accuracy": public.get("accuracy"),
            "test_score": private.get("test_score", self.task.fail_score),
            "test_accuracy": private.get("test_accuracy"),
            "status": status, "error": error, "elapsed": time.time() - start,
        })
        if self.verbose:
            print(f"    eval {self.n:>2}/{self.n_evals} val={public['score']:.4f} "
                  f"{self.rows[-1]['elapsed']:.1f}s {status if status != 'ok' else ''}", flush=True)
        return public


def run_path(runs_dir, task, method, seed):
    return os.path.join(runs_dir, task, method, f"seed{seed}.jsonl")


def is_complete(path):
    if not os.path.exists(path):
        return False
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    return bool(lines) and json.loads(lines[-1]).get("type") == "summary"


def _versions():
    out = {"python": platform.python_version()}
    for name in ("neuropt", "torch", "xgboost", "optuna", "anthropic", "sklearn"):
        module = sys.modules.get(name)  # only what this run loaded: importing torch next to
        if module is not None:          # xgboost can crash on macOS
            out[name] = getattr(module, "__version__", "?")
    return out


def run_one(task, evaluate, method, seed, n_evals, runs_dir, timeout, device, backend=None):
    """Run one (task, method, seed) and write its result file. Returns the summary row."""
    path = run_path(runs_dir, task.name, method, seed)
    workdir = os.path.join(HERE, ".work", f"{task.name}-{method}-seed{seed}-{os.getpid()}")
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    log = EvalLog(task, evaluate, seed, n_evals, timeout)
    start = time.time()
    try:
        llm = run_method(method, task, log, seed, n_evals, workdir, backend=backend)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if log.n != n_evals:
        raise RuntimeError(f"{method} used {log.n} evals, expected {n_evals}")
    summary = {
        "type": "summary", "task": task.name, "method": method, "seed": seed, "n_evals": n_evals,
        "wall_time": time.time() - start, "train_time": sum(r["elapsed"] for r in log.rows),
        "llm": llm, "device": device, "host": platform.node(), "timeout": timeout,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": _versions(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for row in log.rows:
            f.write(json.dumps(row, default=str) + "\n")
        f.write(json.dumps(summary, default=str) + "\n")
    os.replace(path + ".tmp", path)
    return summary


def _default_device():
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--n-evals", type=int, default=N_EVALS)
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="seconds per eval, all methods")
    parser.add_argument("--device", default=None)
    parser.add_argument("--runs-dir", default=RUNS_DIR)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 4 evals, into runs-smoke/")
    args = parser.parse_args(argv)
    if args.smoke:
        args.seeds, args.n_evals = [0], 4
        args.runs_dir = os.path.join(HERE, "runs-smoke")

    if len(args.tasks) > 1:
        # One process per task: PyTorch and XGBoost in one process can crash on macOS.
        passthrough = [a for a in (argv if argv is not None else sys.argv[1:])]
        failed = 0
        for name in args.tasks:
            cmd = [sys.executable, "-m", "benchmarks.run", *passthrough, "--tasks", name]
            code = subprocess.call(cmd)
            if code != 0:
                failed += 1
                print(f"[{name}] FAILED: process exited with code {code}", flush=True)
        return 1 if failed else 0

    task = TASKS[args.tasks[0]]
    device = (args.device or _default_device()) if task.uses_torch else "cpu"
    todo = [(m, s) for m in args.methods for s in args.seeds
            if not is_complete(run_path(args.runs_dir, task.name, m, s))]
    print(f"[{task.name}] {len(todo)} run(s) to do on {device}, {args.n_evals} evals each", flush=True)
    if not todo:
        return 0
    evaluate = task.build(device)
    skipped = {}
    for method, seed in todo:
        if method in skipped:
            continue
        print(f"  {task.name} / {method} / seed {seed}", flush=True)
        try:
            s = run_one(task, evaluate, method, seed, args.n_evals, args.runs_dir, args.timeout, device)
        except (Unavailable, LLMCallFailed) as e:
            # Nothing is written for this run; the method's remaining seeds are skipped too.
            skipped[method] = str(e)
            print(f"  SKIPPED {method}: {e}", flush=True)
            continue
        cost = (s["llm"] or {}).get("cost_usd")
        print(f"  done in {s['wall_time']:.0f}s" + (f", ${cost:.4f}" if cost else ""), flush=True)
    if skipped:
        print(f"[{task.name}] skipped, not run: {skipped}", flush=True)
    return 2 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
