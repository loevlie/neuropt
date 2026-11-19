"""Search methods compared by the harness.

Every method drives the same ``EvalLog`` callable: ``log(config) -> public result``. The log counts
the budget, applies the time limit and keeps the test metrics to itself, so no method can see them.

Method names: ``random``, ``tpe``, ``neuropt-<model>``, ``jev``; add ``+default`` to make the first
eval the task's default config.
"""

import os
import random
import time

from neuropt.arch_search import _JSONLLogger, _normalize_search_space, _random_config
from neuropt.backends.base import BaseDecisionBackend, BaseLLMBackend
from neuropt.search_space import Categorical, IntUniform, LogUniform

# method name -> (backend, model id)
LLM_MODELS = {
    "neuropt-haiku-4.5": ("claude", "claude-haiku-4-5-20251001"),  # package default until Phase 2
}
JEV_URL, JEV_MODEL = "https://api.typesafe.ai", "jev-latest"
JEV_USD_PER_MTOK = 0.042  # input tokens; output is free (docs.typesafe.ai/models, 19 Nov 2026)

BASE_METHODS = ["random", "tpe", *LLM_MODELS, "jev"]
ALL_METHODS = [m + suffix for m in BASE_METHODS for suffix in ("", "+default")]


class Unavailable(RuntimeError):
    """The method can't run here (missing API key or package)."""


# ── Backend wrappers that time every call ────────────────────────────────


class RecordingLLM(BaseLLMBackend):
    def __init__(self, inner, keep_prompts=False):
        super().__init__()
        self.inner, self.latencies, self.prompts = inner, [], []
        self._keep = keep_prompts

    def generate(self, prompt, max_tokens=1024):
        if self._keep:
            self.prompts.append(prompt)
        start = time.time()
        try:
            return self.inner.generate(prompt, max_tokens=max_tokens)
        finally:
            self.latencies.append(time.time() - start)
            self.total_input_tokens = self.inner.total_input_tokens
            self.total_output_tokens = self.inner.total_output_tokens

    def is_available(self):
        return self.inner.is_available()

    @property
    def name(self):
        return self.inner.name

    @property
    def total_cost(self):
        return self.inner.total_cost


class RecordingDecision(BaseDecisionBackend):
    def __init__(self, inner, usd_per_mtok=None, keep_prompts=False):
        super().__init__()
        self.inner, self.latencies, self.prompts = inner, [], []
        self._usd_per_mtok, self._keep = usd_per_mtok, keep_prompts

    def decide(self, state, questions):
        if self._keep:
            self.prompts.append(state + "\n" + repr(questions))
        start = time.time()
        try:
            return self.inner.decide(state, questions)
        finally:
            self.latencies.append(time.time() - start)
            self.total_input_tokens = self.inner.total_input_tokens
            self.total_output_tokens = self.inner.total_output_tokens

    def is_available(self):
        return self.inner.is_available()

    @property
    def name(self):
        return self.inner.name

    @property
    def total_cost(self):
        if self._usd_per_mtok is None:
            return None
        return self.total_input_tokens * self._usd_per_mtok / 1e6


# ── Methods ──────────────────────────────────────────────────────────────


def run_random(task, log, seed, n_evals, seeded):
    space = _normalize_search_space(task.space)
    rng = random.Random(seed)
    if seeded:
        log(dict(task.default), source="default")
    while log.n < n_evals:
        log(_random_config(space, rng), source="random")
    return {}


def run_tpe(task, log, seed, n_evals, seeded):
    try:
        import optuna
    except ImportError as e:
        raise Unavailable("optuna is not installed") from e
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = _normalize_search_space(task.space)

    def objective(trial):
        cfg = {}
        for name, dim in space.items():
            if isinstance(dim, Categorical):
                cfg[name] = trial.suggest_categorical(name, dim.choices)
            elif isinstance(dim, IntUniform):
                cfg[name] = trial.suggest_int(name, dim.low, dim.high)
            else:
                cfg[name] = trial.suggest_float(name, dim.low, dim.high,
                                                log=isinstance(dim, LogUniform))
        source = "default" if seeded and trial.number == 0 else "tpe"
        return log(cfg, source=source)["score"]  # failed evals come back as task.fail_score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    if seeded:
        study.enqueue_trial(dict(task.default))
    study.optimize(objective, n_trials=n_evals)
    return {}


def _run_search(search, backend, task, log, n_evals, seeded):
    """Run an ArchSearch / DecisionSearch and collect its LLM statistics."""
    if seeded:
        # Evaluate the default first and put it in the search log, so the model
        # sees it as experiment #1 exactly as it would after a resume.
        result = search._run_one(dict(task.default))
        log.rows[-1]["source"] = "default"
        _JSONLLogger(search.log_path).log(-1, dict(task.default), result, "default")
    search.run(max_evals=n_evals - log.n)

    # Which evals were proposed by the model and which were the random fallback
    for row, logged in zip(log.rows, _JSONLLogger(search.log_path).load_history()):
        row["source"] = logged.get("source", "")

    lat = backend.latencies
    return {
        "backend": backend.name,
        "calls": len(lat),
        "latency_mean": sum(lat) / len(lat) if lat else None,
        "latency_total": sum(lat),
        "tokens_in": backend.total_input_tokens,
        "tokens_out": backend.total_output_tokens,
        "cost_usd": backend.total_cost,
        "iterations_ok": search.llm_success,
        "iterations_fallback": search.llm_fallback,
        "fallback_evals": sum(r["source"] == "random" for r in log.rows),
        **search.stats,
    }


def _train_fn(log):
    return lambda config: log(config)


def run_neuropt(task, log, seed, n_evals, seeded, method, workdir, backend=None):
    from neuropt import ArchSearch

    if backend is None:
        kind, model = LLM_MODELS[method]
        if kind != "claude":
            raise Unavailable(f"unknown backend kind {kind!r}")
        from neuropt.backends.claude_backend import ClaudeBackend
        backend = ClaudeBackend(model=model)
        if not backend.is_available():
            raise Unavailable("ANTHROPIC_API_KEY is not set")
    backend = backend if isinstance(backend, RecordingLLM) else RecordingLLM(backend)
    search = ArchSearch(train_fn=_train_fn(log), search_space=task.space, backend=backend,
                        log_path=os.path.join(workdir, "search.jsonl"), batch_size=3, seed=seed,
                        timeout=log.timeout * 10)  # the harness enforces the real time limit
    return _run_search(search, backend, task, log, n_evals, seeded)


def run_jev(task, log, seed, n_evals, seeded, workdir, backend=None):
    from neuropt import DecisionSearch
    from neuropt.backends import DecisionBackend

    if backend is None:
        key = os.environ.get("NEUROPT_DECISION_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise Unavailable("NEUROPT_DECISION_API_KEY (or TYPESAFE_API_KEY) is not set")
        backend = RecordingDecision(
            DecisionBackend(base_url=JEV_URL, model=JEV_MODEL, api_key=key, timeout=120),
            usd_per_mtok=JEV_USD_PER_MTOK)
    elif not isinstance(backend, RecordingDecision):
        backend = RecordingDecision(backend)
    search = DecisionSearch(train_fn=_train_fn(log), search_space=task.space, backend=backend,
                            log_path=os.path.join(workdir, "search.jsonl"), batch_size=3, seed=seed,
                            timeout=log.timeout * 10)
    return _run_search(search, backend, task, log, n_evals, seeded)


def run_method(method, task, log, seed, n_evals, workdir, backend=None):
    """Run one method to exactly ``n_evals`` evals. Returns LLM stats ({} for classical methods)."""
    base, _, suffix = method.partition("+")
    if suffix not in ("", "default"):
        raise ValueError(f"unknown method {method!r}")
    seeded = suffix == "default"
    if base == "random":
        return run_random(task, log, seed, n_evals, seeded)
    if base == "tpe":
        return run_tpe(task, log, seed, n_evals, seeded)
    if base == "jev":
        return run_jev(task, log, seed, n_evals, seeded, workdir, backend)
    if base in LLM_MODELS or backend is not None:
        return run_neuropt(task, log, seed, n_evals, seeded, base, workdir, backend)
    raise ValueError(f"unknown method {method!r}")
