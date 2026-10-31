"""Search driven by a decision model (Jev-style) instead of a generative LLM.

EXPERIMENTAL: still being evaluated. It works end to end, but it has only been
benchmarked on one seed per task (ahead of Claude, Optuna and random search on
the 14-param CNN benchmark, behind them on XGBoost), so defaults may change.

A decision model can't write a config. It answers typed multiple-choice
questions with calibrated probabilities. So each iteration:

1. The model reads the same evidence the LLM would (results table, per-epoch
   curves, overfitting/underfitting signals) and answers one ``choice``
   question per hyperparameter: which value (categoricals) or which range
   (numeric params, split into bins) is most likely to beat the current best.
   All questions go in one request.
2. Candidate configs are sampled from those probabilities, mixed with a little
   uniform noise for exploration. The first candidate takes the top answer for
   every parameter.
3. The model then picks among the candidates jointly (one more ``choice``
   question), which catches combinations the per-parameter answers can't see.
   The top-ranked configs run, plus one sampled from the rest as exploration.

Works with any ``BaseDecisionBackend``; the default is ``DecisionBackend``
talking to an H2O-Lightning-4B server (see ``docs/decision-models.md``).
"""

import json
import math

from neuropt.arch_search import (
    ArchSearch,
    _config_key,
    _fmt,
    _get_score,
    _random_config,
    _resolve_backend,
    _short_config,
    _validate_config,
)
from neuropt.backends.base import BaseDecisionBackend
from neuropt.search_space import Categorical, IntUniform, LogUniform

# Most decision servers cap a choice question at 255 options.
MAX_OPTIONS = 255


class DecisionSearch(ArchSearch):
    """Hyperparameter search where a decision model picks what to try next.

    .. note:: Experimental, still being evaluated; defaults may change.

    Same interface as :class:`ArchSearch` (``run``, ``from_model``, logging,
    resume); only how configs are chosen differs.

    Example::

        search = DecisionSearch(train_fn, {"lr": (1e-4, 1e-1), "act": ["relu", "gelu"]})
        search.run(max_evals=30)

    Args:
        train_fn, search_space: As for :class:`ArchSearch`.
        backend: ``"auto"`` (a :class:`~neuropt.backends.DecisionBackend` if a
            server answers at ``$NEUROPT_DECISION_URL``, else random search),
            ``"decision"`` (always use it), ``"none"``, or a
            :class:`~neuropt.backends.base.BaseDecisionBackend` instance.
        n_bins: Ranges each numeric parameter is split into for its question.
        n_candidates: Configs sampled from the answers before the joint pick.
        explore: Share of uniform probability mixed into each answer when
            sampling candidates (0 = trust the model completely).
        rerank: Ask the model to pick jointly among the candidates.
        max_curve_points: Long per-epoch curves (e.g. 500 boosting rounds)
            are shown at this many evenly spaced epochs, which keeps the
            state small: each question re-reads it. None shows every epoch.
        **kwargs: Passed to :class:`ArchSearch` (``log_path``, ``batch_size``,
            ``minimize``, ``ml_context``, ...).
    """

    _title = "Decision-Model-Guided Search (experimental)"
    _decider_label = "Decision model"
    MAX_RESAMPLE = 10

    def __init__(self, train_fn, search_space, backend="auto", *, n_bins: int = 8,
                 n_candidates: int = 12, explore: float = 0.15, rerank: bool = True,
                 max_curve_points: int | None = 20, **kwargs):
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2")
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1")
        if not 0.0 <= explore <= 1.0:
            raise ValueError("explore must be between 0 and 1")
        super().__init__(train_fn, search_space, backend=backend, **kwargs)
        self.n_bins = n_bins
        self.n_candidates = n_candidates
        self.explore = explore
        self.rerank = rerank
        self._max_curve_points = max_curve_points

    @classmethod
    def _resolve_backend_for(cls, backend):
        if backend in ("auto", "decision"):
            resolved = _resolve_backend("decision")
            if backend == "auto" and not resolved.is_available():
                print(f"No decision server answering at {resolved.base_url} "
                      "(set NEUROPT_DECISION_URL); falling back to random search.")
                return None
            return resolved
        resolved = _resolve_backend(backend)
        if resolved is not None and not isinstance(resolved, BaseDecisionBackend):
            raise TypeError(
                f"DecisionSearch needs a decision-model backend, got {type(resolved).__name__}. "
                "Use ArchSearch for LLM backends."
            )
        return resolved

    # ── Config generation ──────────────────────────────────────────────────

    def _get_configs(self, history, rng):
        if self._backend is None:
            return super()._get_configs(history, rng)

        state = self._build_state(history)
        params = self._build_params(history)
        asked = [p for p in params if p.question is not None]
        try:
            answers = self._backend.decide(state, {p.qid: p.question for p in asked}) if asked else {}
            probs = {p.name: _distribution(p.keys, answers[p.qid]) for p in asked}
        except Exception as e:
            self.llm_fallback += 1
            print(f"  [decision model error: {e}; using random]")
            return [_random_config(self.search_space, rng)
                    for _ in range(self.batch_size)], "random"

        self._print_decisions(asked, probs)
        pool = self._sample_pool(params, probs, history, rng)
        configs = self._pick(state, pool, rng)
        self.llm_success += 1
        return configs, "decision"

    def _build_params(self, history):
        """One _Param (options + question) per search-space dimension."""
        ok_rows = [r for r in history
                   if r.get("status") == "ok" and _get_score(r) is not None]
        params = []
        for i, (name, dim) in enumerate(self.search_space.items()):
            options = _options_for(dim, self.n_bins)
            param = _Param(name, dim, f"p{i}", options)
            if 2 <= len(options) <= MAX_OPTIONS:
                param.question = self._question_for(param, ok_rows)
            params.append(param)
        return params

    def _question_for(self, param, ok_rows):
        # Per-option history: how often tried, best score, does it hold the best config?
        tried = [0] * len(param.options)
        best = [None] * len(param.options)
        for row in ok_rows:
            idx = param.index_of(row.get("config", {}).get(param.name))
            if idx is None:
                continue
            score = _get_score(row)
            tried[idx] += 1
            if best[idx] is None or self._better(score, best[idx]):
                best[idx] = score
        best_idx = None
        if self.best_config is not None:
            best_idx = param.index_of(self.best_config.get(param.name))

        criteria = {}
        for idx, opt in enumerate(param.options):
            if tried[idx]:
                stats = f"tried {tried[idx]}x, best score {best[idx]:.4g}"
            else:
                stats = "not tried yet"
            desc = f"{param.name} {opt.describe()}; {stats}"
            if idx == best_idx:
                desc += "; the current best config uses this"
            criteria[opt.key] = desc

        single_values = all(getattr(o, "lo", 0) == getattr(o, "hi", 0) for o in param.options)
        kind = "value" if single_values else "range"
        direction = "lower" if self.minimize else "higher"
        if self.best_config is not None:
            goal = (f"most likely to give a {direction} score than the current best "
                    f"({self.best_score:.4g}) in the next experiment")
        else:
            goal = f"most promising for the next experiment ({direction} score is better)"
        instructions = (f"Which {param.name} {kind} is {goal}? Base the decision on the "
                        "experiment results and training curves.")
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    def _sample_pool(self, params, probs, history, rng):
        """Sample distinct candidate configs from the per-parameter answers."""
        seen = {_config_key(r.get("config", {})) for r in history}
        size = max(self.n_candidates, self.batch_size)
        pool = []
        for slot in range(size):
            cfg = None
            for attempt in range(self.MAX_RESAMPLE):
                greedy = slot == 0 and attempt == 0
                candidate = _validate_config(
                    {p.name: self._draw(p, probs.get(p.name), rng, greedy) for p in params},
                    self.search_space)
                if candidate is not None and _config_key(candidate) not in seen:
                    cfg = candidate
                    break
            if cfg is None:  # the model's answers keep hitting tried configs
                cfg = _random_config(self.search_space, rng)
            seen.add(_config_key(cfg))
            pool.append(cfg)
        return pool

    def _draw(self, param, p, rng, greedy):
        if p is None:  # parameter wasn't asked (one option, or too many)
            return param.options[rng.randrange(len(param.options))].sample(rng)
        if greedy:
            idx = max(range(len(p)), key=p.__getitem__)
            # Keep the best config's value when the model chose its bin.
            if self.best_config is not None:
                current = self.best_config.get(param.name)
                if current is not None and param.index_of(current) == idx:
                    return current
            return param.options[idx].center()
        k = len(p)
        mixed = [(1 - self.explore) * q + self.explore / k for q in p]
        idx = rng.choices(range(k), weights=mixed)[0]
        return param.options[idx].sample(rng)

    def _pick(self, state, pool, rng):
        """Choose batch_size configs from the pool, asking the model jointly."""
        n = self.batch_size
        if not self.rerank or len(pool) <= n:
            return pool[:n]

        keys = [f"c{i + 1}" for i in range(len(pool))]
        direction = "lowest" if self.minimize else "highest"
        question = {
            "type": "choice",
            "instructions": (f"Which of these candidate configs is most likely to reach the "
                             f"{direction} score in the next experiment? Base the decision on "
                             "the experiment results and training curves."),
            "criteria": {k: _short_config(cfg) for k, cfg in zip(keys, pool)},
        }
        try:
            answer = self._backend.decide(state, {"pick": question})["pick"]
            p = _distribution(keys, answer)
        except Exception as e:
            print(f"  [joint pick failed: {e}; using sampled order]")
            return pool[:n]

        order = sorted(range(len(pool)), key=lambda i: -p[i])
        chosen = order[:max(n - 1, 1)]
        if n > 1:
            # Last slot: sample from the rest in proportion to the model's
            # probabilities, so every batch has at least one exploratory config.
            rest = order[n - 1:]
            weights = [p[i] for i in rest]
            if sum(weights) <= 0:
                weights = [1.0] * len(rest)
            chosen.append(rng.choices(rest, weights=weights)[0])
        picks = ", ".join(f"{keys[i]} ({p[i]:.2f})" for i in chosen)
        print(f"  joint pick: {picks}")
        return [pool[i] for i in chosen]

    def _better(self, a, b):
        return a < b if self.minimize else a > b

    def _print_decisions(self, asked, probs, max_n=6):
        top = []
        for param in asked:
            p = probs[param.name]
            idx = max(range(len(p)), key=p.__getitem__)
            top.append((p[idx], param.name, param.options[idx].key))
        top.sort(reverse=True)
        if top:
            line = " | ".join(f"{name}={key} ({conf:.2f})" for conf, name, key in top[:max_n])
            print(f"  decision: {line}")


# ── Options: what a parameter's question offers ───────────────────────────

class _Param:
    def __init__(self, name, dim, qid, options):
        self.name = name
        self.dim = dim
        self.qid = qid
        self.options = options
        self.keys = [o.key for o in options]
        self.question = None

    def index_of(self, value):
        if value is None:
            return None
        for i, opt in enumerate(self.options):
            if opt.contains(value):
                return i
        return None


class _ValueOption:
    """One categorical value."""

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def describe(self):
        return f"= {self.key}"

    def contains(self, value):
        # Bools only match bools (True == 1 in Python, but not here).
        if isinstance(value, bool) != isinstance(self.value, bool):
            return False
        try:
            return bool(value == self.value)
        except Exception:
            return False

    def sample(self, rng):
        return self.value

    def center(self):
        return self.value


class _IntRangeOption:
    """Integers lo..hi inclusive."""

    def __init__(self, key, lo, hi):
        self.key = key
        self.lo = lo
        self.hi = hi

    def describe(self):
        return f"= {self.lo}" if self.lo == self.hi else f"from {self.lo} to {self.hi}"

    def contains(self, value):
        try:
            return self.lo <= value <= self.hi
        except TypeError:
            return False

    def sample(self, rng):
        return rng.randint(self.lo, self.hi)

    def center(self):
        return (self.lo + self.hi) // 2


class _FloatRangeOption:
    """A bin of a Uniform/LogUniform dimension, in its internal (log10) space."""

    def __init__(self, key, dim, lo, hi, last):
        self.key = key
        self.dim = dim
        self.lo = lo
        self.hi = hi
        self.last = last

    def describe(self):
        scale = " (log scale)" if isinstance(self.dim, LogUniform) else ""
        return f"between {self.key.replace(' to ', ' and ')}{scale}"

    def contains(self, value):
        try:
            x = self.dim.to_internal(float(value))
        except (TypeError, ValueError):
            return False
        return self.lo <= x < self.hi or (self.last and math.isclose(x, self.hi))

    def sample(self, rng):
        return self.dim.from_internal(rng.uniform(self.lo, self.hi))

    def center(self):
        return self.dim.from_internal((self.lo + self.hi) / 2)


def _options_for(dim, n_bins):
    if isinstance(dim, Categorical):
        keys = _unique([v if isinstance(v, str) else json.dumps(v, default=str)
                        for v in dim.choices])
        return [_ValueOption(k, v) for k, v in zip(keys, dim.choices)]

    if isinstance(dim, IntUniform):
        n_values = dim.high - dim.low + 1
        if n_values <= n_bins:
            return [_IntRangeOption(str(v), v, v) for v in range(dim.low, dim.high + 1)]
        edges = [dim.low + (n_values * i) // n_bins for i in range(n_bins + 1)]
        spans = [(edges[i], edges[i + 1] - 1) for i in range(n_bins)]
        keys = _unique([str(a) if a == b else f"{a} to {b}" for a, b in spans])
        return [_IntRangeOption(k, a, b) for k, (a, b) in zip(keys, spans)]

    lo, hi = dim.bounds()
    step = (hi - lo) / n_bins
    spans = [(lo + i * step, lo + (i + 1) * step) for i in range(n_bins)]
    keys = _unique([f"{_fmt(dim.from_internal(a))} to {_fmt(dim.from_internal(b))}"
                    for a, b in spans])
    return [_FloatRangeOption(k, dim, a, b, last=(i == n_bins - 1))
            for i, (k, (a, b)) in enumerate(zip(keys, spans))]


def _unique(keys):
    """Make option keys unique by suffixing repeats."""
    out, seen = [], set()
    for i, key in enumerate(keys):
        if key in seen:
            key = f"{key} [{i + 1}]"
        seen.add(key)
        out.append(key)
    return out


def _distribution(keys, answer):
    """Probabilities aligned with ``keys`` from a choice answer.

    Uses ``answer["probabilities"]`` when present (renormalised over the
    known keys), otherwise a one-hot on ``answer["choice"]``.
    """
    if not isinstance(answer, dict):
        raise ValueError(f"expected a choice answer object, got {answer!r}")
    probs = answer.get("probabilities")
    values = []
    if isinstance(probs, dict):
        for k in keys:
            try:
                v = float(probs.get(k, 0.0))
            except (TypeError, ValueError):
                v = 0.0
            values.append(v if math.isfinite(v) and v > 0 else 0.0)
    total = sum(values)
    if total > 0:
        return [v / total for v in values]
    choice = answer.get("choice")
    if choice in keys:
        return [1.0 if k == choice else 0.0 for k in keys]
    raise ValueError(f"answer has no usable probabilities or choice: {answer!r}")
