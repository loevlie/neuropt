"""Tests for the evaluation harness (benchmarks/): budget, default seeding, and no test leakage."""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import run as bench_run  # noqa: E402
from benchmarks.methods import RecordingDecision, RecordingLLM, run_method  # noqa: E402
from benchmarks.run import EvalLog, is_complete, run_one, run_path  # noqa: E402
from benchmarks.tasks import Task  # noqa: E402
from neuropt.arch_search import _normalize_search_space, _random_config  # noqa: E402
from neuropt.backends.base import BaseDecisionBackend, BaseLLMBackend  # noqa: E402

SPACE = {"lr": (1e-4, 1e-1), "width": (8, 64), "act": ["relu", "gelu", "tanh"]}
DEFAULT = {"lr": 1e-3, "width": 32, "act": "relu"}
TEST_CANARY = 7.7777  # every test score starts with these digits; val scores are below 1


def _build(device):
    def evaluate(cfg, seed):
        if cfg["act"] == "tanh":
            raise ValueError("tanh always fails in this fake task")
        val = abs(cfg["lr"] - 0.01) * 5 + 0.1
        public = {"score": val, "accuracy": 0.5, "train_losses": [val + 0.2, val + 0.1],
                  "val_losses": [val + 0.1, val]}
        return public, {"test_score": TEST_CANARY + seed * 1e-6, "test_accuracy": 0.91919}
    return evaluate


TASK = Task("fake", "fake task", SPACE, DEFAULT, n_classes=3, build=_build)


class FakeLLM(BaseLLMBackend):
    def __init__(self):
        super().__init__()
        self._rng = random.Random(0)

    def generate(self, prompt, max_tokens=1024):
        space = _normalize_search_space(SPACE)
        return json.dumps([_random_config(space, self._rng) for _ in range(3)])

    def is_available(self):
        return True


class FakeDecision(BaseDecisionBackend):
    def decide(self, state, questions):
        out = {}
        for qid, q in questions.items():
            keys = list(q["criteria"])
            out[qid] = {"choice": keys[0], "probabilities": {k: 1 / len(keys) for k in keys}}
        return out

    def is_available(self):
        return True


def _backend(method):
    if method.startswith("neuropt"):
        return RecordingLLM(FakeLLM(), keep_prompts=True)
    if method.startswith("jev"):
        return RecordingDecision(FakeDecision(), keep_prompts=True)
    return None


def _run(method, tmp_path, n_evals=7, seed=0):
    log = EvalLog(TASK, TASK.build("cpu"), seed, n_evals, verbose=False)
    backend = _backend(method)
    stats = run_method(method, TASK, log, seed, n_evals, str(tmp_path), backend=backend)
    return log, backend, stats


METHODS = ["random", "tpe", "neuropt-fake", "jev"]


@pytest.mark.parametrize("method", [m + s for m in METHODS for s in ("", "+default")])
def test_exact_budget_and_default_first(method, tmp_path, capsys):
    if method.startswith("tpe"):
        pytest.importorskip("optuna")
    log, _, _ = _run(method, tmp_path)
    assert log.n == 7
    assert [r["eval"] for r in log.rows] == list(range(1, 8))
    if method.endswith("+default"):
        assert log.rows[0]["config"] == DEFAULT
        assert log.rows[0]["source"] == "default"
    else:
        assert all(r["source"] != "default" for r in log.rows)


@pytest.mark.parametrize("method", ["neuropt-fake", "neuropt-fake+default", "jev", "jev+default"])
def test_test_metrics_never_reach_the_model(method, tmp_path, capsys):
    log, backend, _ = _run(method, tmp_path, n_evals=9)
    assert backend.prompts, "the model was never called"
    assert any(r["status"] == "ok" for r in log.rows)
    for prompt in backend.prompts:
        assert "7.77" not in prompt
        assert "0.919" not in prompt
        assert "test" not in prompt.lower()
    # The harness itself did record them
    assert all(str(r["test_score"]).startswith("7.77") for r in log.rows if r["status"] == "ok")


def test_public_result_has_no_test_keys():
    log = EvalLog(TASK, TASK.build("cpu"), 0, 2, verbose=False)
    public = log(dict(DEFAULT))
    assert not [k for k in public if "test" in k]


def test_failed_evals_count_and_score_as_uniform_guess():
    log = EvalLog(TASK, TASK.build("cpu"), 0, 2, verbose=False)
    public = log({"lr": 0.01, "width": 16, "act": "tanh"})
    assert public == {"score": TASK.fail_score, "status": "error"}
    assert log.rows[0]["status"] == "error" and log.rows[0]["test_score"] == TASK.fail_score
    log(dict(DEFAULT))
    with pytest.raises(RuntimeError, match="budget"):
        log(dict(DEFAULT))


def test_time_limit_applies(monkeypatch):
    import time

    def slow(cfg, seed):
        time.sleep(5)
        return {"score": 0.1}, {}

    log = EvalLog(TASK, slow, 0, 1, timeout=1, verbose=False)
    assert log(dict(DEFAULT))["status"] == "timeout"
    assert log.rows[0]["elapsed"] < 3


def test_every_method_gets_the_same_training_seeds(tmp_path, capsys):
    seen = {}

    def build(device):
        def evaluate(cfg, seed):
            seen.setdefault(current[0], []).append(seed)
            return {"score": 0.5}, {"test_score": 1.0}
        return evaluate

    task = Task("fake", "fake", SPACE, DEFAULT, 3, build)
    current = [None]
    for method in ("random", "random+default", "neuropt-fake"):
        current[0] = method
        log = EvalLog(task, task.build("cpu"), 3, 6, verbose=False)
        run_method(method, task, log, 3, 6, str(tmp_path), backend=_backend(method))
    assert seen["random"] == seen["random+default"] == seen["neuropt-fake"] == list(range(3000, 3006))


def test_run_one_writes_a_complete_resumable_file(tmp_path, capsys):
    runs = str(tmp_path / "runs")
    path = run_path(runs, "fake", "random+default", 1)
    assert not is_complete(path)
    summary = run_one(TASK, TASK.build("cpu"), "random+default", 1, 5, runs, 60, "cpu")
    assert is_complete(path)
    rows = [json.loads(line) for line in open(path)]
    assert [r["type"] for r in rows] == ["eval"] * 5 + ["summary"]
    assert summary["n_evals"] == 5 and summary["llm"] == {}
    # A truncated file (no summary row) counts as not done
    with open(path, "w") as f:
        f.write(json.dumps(rows[0]) + "\n")
    assert not is_complete(path)


def test_llm_stats_are_recorded(tmp_path, capsys):
    _, _, stats = _run("neuropt-fake", tmp_path, n_evals=6)
    assert stats["calls"] == 2 and stats["llm_calls"] == 2
    assert stats["parse_failures"] == 0 and stats["fallback_evals"] == 0
    assert stats["proposed_configs"] == 6
    assert stats["latency_mean"] is not None


def test_main_skips_finished_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(bench_run.TASKS, "fake", TASK)
    args = ["--tasks", "fake", "--methods", "random", "--seeds", "0", "--n-evals", "3",
            "--runs-dir", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", ["run"])
    parser_choices_patch = list(bench_run.TASKS)
    assert "fake" in parser_choices_patch
    assert bench_run.main(args) == 0
    assert "1 run(s) to do" in capsys.readouterr().out
    assert bench_run.main(args) == 0
    assert "0 run(s) to do" in capsys.readouterr().out
