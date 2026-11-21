"""Search methods compared by the harness.

Every method drives the same ``EvalLog`` callable: ``log(config) -> public result``. The log counts
the budget, applies the time limit and keeps the test metrics to itself, so no method can see them.

Method names: ``random``, ``tpe``, ``neuropt-<model>``, ``jev``, ``hybrid-<model>``; add ``+default``
to make the first eval the task's default config.

``neuropt-haiku-4.5`` and ``neuropt-gpt-4o-mini`` are the Phase 1 arms, produced by neuropt 0.6.1
(whole batch discarded on one invalid config, full curves, no structured outputs). They are frozen:
their cached runs are kept and new runs are refused, so a newer package version can't be cached under
their names. ``neuropt-v2-<model>`` arms use this branch's ArchSearch.
"""

import math
import os
import random
import time

from neuropt.arch_search import _JSONLLogger, _normalize_search_space, _random_config
from neuropt.backends.base import BaseDecisionBackend, BaseLLMBackend
from neuropt.search_space import Categorical, IntUniform, LogUniform

# method name -> (backend, model id)
LLM_MODELS = {
    "neuropt-haiku-4.5": ("claude", "claude-haiku-4-5-20251001"),  # Phase 1 (0.6.1), frozen
    "neuropt-gpt-4o-mini": ("openai", "gpt-4o-mini"),  # Phase 1 (0.6.1), frozen
    "neuropt-v2-haiku-4.5": ("claude", "claude-haiku-4-5-20251001"),
    "neuropt-v2-haiku-5.5": ("claude", "claude-haiku-5-5"),
    "neuropt-v2-gpt-4o-mini": ("openai", "gpt-4o-mini"),
    # Local model behind an OpenAI-compatible server at $NEUROPT_LLM_BASE_URL (Ollama)
    "neuropt-v2-qwen3.5-9b": ("compatible", "qwen3.5:9b"),
    # The package default after Phase 2: Haiku 5.5 with every curve point
    "neuropt-v2-haiku-5.5-full": ("claude", "claude-haiku-5-5"),
}
# The Phase 2 v2 arms ran with at most 20 points per curve (then the package default); pinned so
# they stay reproducible. Only XGBoost's 500-round curves are long enough for this to matter.
CURVE_POINTS = {m: 20 for m in ("neuropt-v2-haiku-4.5", "neuropt-v2-haiku-5.5",
                                "neuropt-v2-gpt-4o-mini", "neuropt-v2-qwen3.5-9b")}
FROZEN = {"neuropt-haiku-4.5", "neuropt-gpt-4o-mini"}
# hybrid-<model>: that model's ArchSearch for the first evals, then Optuna TPE warm-started with them
HYBRID = {"hybrid-v2-haiku-5.5": "neuropt-v2-haiku-5.5"}
HYBRID_LLM_EVALS = 9  # 3 LLM calls (plus the default for +default)
JEV_URL, JEV_MODEL = "https://api.typesafe.ai", "jev-latest"
JEV_USD_PER_MTOK = 0.042  # input tokens; output is free (docs.typesafe.ai/models, 19 Nov 2026)

BASE_METHODS = ["random", "tpe", *LLM_MODELS, "jev", *HYBRID]
ALL_METHODS = [m + suffix for m in BASE_METHODS for suffix in ("", "+default")]


class Unavailable(RuntimeError):
    """The method can't run here (missing API key or package)."""


class Frozen(RuntimeError):
    """A Phase 1 arm whose code no longer exists in this version of the package."""


class LLMCallFailed(BaseException):
    """The model API failed after retries (outage, no credit, bad key).

    A BaseException on purpose: the search's own "fall back to random configs" handler must not
    swallow it, or the run would turn into random search and be cached under the model's name.
    """


RETRY_DELAYS = (2, 15)  # seconds before the 2nd and 3rd attempt


def _call(fn):
    error = None
    for delay in (0, *RETRY_DELAYS):
        time.sleep(delay)
        start = time.time()
        try:
            return fn(), time.time() - start
        except Exception as e:
            error = e
    raise LLMCallFailed(f"{type(error).__name__}: {str(error)[:300]}")


# ── Backend wrappers that time every call ────────────────────────────────


class RecordingLLM(BaseLLMBackend):
    def __init__(self, inner, keep_prompts=False):
        super().__init__()
        self.inner, self.latencies, self.prompts = inner, [], []
        self._keep = keep_prompts

    def generate(self, prompt, max_tokens=1024):
        return self.generate_structured(prompt, max_tokens=max_tokens)

    def generate_structured(self, prompt, system=None, schema=None, max_tokens=1024):
        if self._keep:
            self.prompts.append(prompt if system is None else system + "\n" + prompt)
        out, latency = _call(lambda: self.inner.generate_structured(
            prompt, system=system, schema=schema, max_tokens=max_tokens))
        self.latencies.append(latency)
        self.total_input_tokens = self.inner.total_input_tokens
        self.total_output_tokens = self.inner.total_output_tokens
        return out

    @property
    def structured(self):
        return getattr(self.inner, "_structured", None)

    @property
    def cache_tokens(self):
        return (getattr(self.inner, "cache_write_tokens", 0),
                getattr(self.inner, "cache_read_tokens", 0))

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
        out, latency = _call(lambda: self.inner.decide(state, questions))
        self.latencies.append(latency)
        self.total_input_tokens = self.inner.total_input_tokens
        self.total_output_tokens = self.inner.total_output_tokens
        return out

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
    """Run an ArchSearch / DecisionSearch and collect its LLM statistics.

    With ``seeded`` the search was built with ``initial_configs=[task.default]``,
    so the package itself evaluates the default first.
    """
    search.run(max_evals=n_evals)

    # Which evals were proposed by the model and which were the random fallback
    for row, logged in zip(log.rows, _JSONLLogger(search.log_path).load_history()):
        source = logged.get("source", "")
        row["source"] = "default" if source == "initial" else source

    lat = backend.latencies
    extra = {}
    if isinstance(backend, RecordingLLM):
        extra = {"structured_outputs": backend.structured,
                 "cache_write_tokens": backend.cache_tokens[0],
                 "cache_read_tokens": backend.cache_tokens[1]}
    return {
        **extra,
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
    search, backend = _arch_search(task, log, seed, seeded, method, workdir, backend)
    return _run_search(search, backend, task, log, n_evals, seeded)


def _make_llm(method):
    kind, model = LLM_MODELS[method]
    if kind == "claude":
        from neuropt.backends.claude_backend import ClaudeBackend
        backend = ClaudeBackend(model=model)
    elif kind == "openai":
        from neuropt.backends.openai_backend import OpenAIBackend
        backend = OpenAIBackend(model=model, base_url="")
    elif kind == "compatible":
        from neuropt.backends.openai_backend import OpenAIBackend
        if not os.environ.get("NEUROPT_LLM_BASE_URL"):
            raise Unavailable("NEUROPT_LLM_BASE_URL is not set (OpenAI-compatible server)")
        backend = OpenAIBackend(model=model)
    else:
        raise Unavailable(f"unknown backend kind {kind!r}")
    if not backend.is_available():
        raise Unavailable(f"no API key for the {kind} backend")
    return backend


def _arch_search(task, log, seed, seeded, method, workdir, backend):
    from neuropt import ArchSearch

    if backend is None:
        backend = _make_llm(method)
    backend = backend if isinstance(backend, RecordingLLM) else RecordingLLM(backend)
    search = ArchSearch(train_fn=_train_fn(log), search_space=task.space, backend=backend,
                        log_path=os.path.join(workdir, "search.jsonl"), batch_size=3, seed=seed,
                        timeout=log.timeout * 10,  # the harness enforces the real time limit
                        initial_configs=[dict(task.default)] if seeded else None,
                        max_curve_points=CURVE_POINTS.get(method))
    return search, backend


def run_hybrid(task, log, seed, n_evals, seeded, method, workdir, backend=None):
    """The LLM proposes the first evals, then Optuna TPE continues from all of them.

    TPE gets the LLM-phase trials through ``add_trial`` with their validation scores (the same
    public scores the LLM saw), so its model starts from them instead of random startup trials.
    """
    try:
        import optuna
    except ImportError as e:
        raise Unavailable("optuna is not installed") from e
    search, backend = _arch_search(task, log, seed, seeded, HYBRID[method], workdir, backend)
    llm_evals = min(n_evals, HYBRID_LLM_EVALS + int(seeded))
    stats = _run_search(search, backend, task, log, llm_evals, seeded)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = _normalize_search_space(task.space)
    dists = {name: _optuna_distribution(optuna, dim) for name, dim in space.items()}
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    for row in _JSONLLogger(search.log_path).load_history():
        params = {k: row["config"][k] for k in space}
        score = row["score"] if math.isfinite(row["score"]) else task.fail_score
        study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=score))

    def objective(trial):
        cfg = {name: (trial.suggest_categorical(name, dim.choices) if isinstance(dim, Categorical)
                      else trial.suggest_int(name, dim.low, dim.high) if isinstance(dim, IntUniform)
                      else trial.suggest_float(name, dim.low, dim.high,
                                               log=isinstance(dim, LogUniform)))
               for name, dim in space.items()}
        return log(cfg, source="tpe")["score"]

    study.optimize(objective, n_trials=n_evals - log.n)
    stats["llm_phase_evals"] = llm_evals
    return stats


def _optuna_distribution(optuna, dim):
    """The distribution suggest_* would record; Optuna < 3 (the Mac runs) has older class names."""
    d = optuna.distributions
    log = isinstance(dim, LogUniform)
    if isinstance(dim, Categorical):
        return d.CategoricalDistribution(dim.choices)
    if hasattr(d, "FloatDistribution"):
        if isinstance(dim, IntUniform):
            return d.IntDistribution(dim.low, dim.high)
        return d.FloatDistribution(dim.low, dim.high, log=log)
    if isinstance(dim, IntUniform):
        return d.IntUniformDistribution(dim.low, dim.high)
    return d.LogUniformDistribution(dim.low, dim.high) if log else d.UniformDistribution(dim.low, dim.high)


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
                            timeout=log.timeout * 10,
                            initial_configs=[dict(task.default)] if seeded else None)
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
    if base in FROZEN and backend is None:
        raise Frozen(f"{base} is a Phase 1 arm produced by neuropt 0.6.1; its cached runs are kept, "
                     "but this version of the package can't reproduce it. Use neuropt-v2-*.")
    if base == "jev":
        return run_jev(task, log, seed, n_evals, seeded, workdir, backend)
    if base in HYBRID:
        return run_hybrid(task, log, seed, n_evals, seeded, base, workdir, backend)
    if base in LLM_MODELS or backend is not None:
        return run_neuropt(task, log, seed, n_evals, seeded, base, workdir, backend)
    raise ValueError(f"unknown method {method!r}")
