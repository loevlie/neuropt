"""LLM-guided hyperparameter and architecture search.

The LLM reads per-epoch training curves and proposes what to try next.
Runs until Ctrl+C. Everything is logged to JSONL for crash recovery.
"""

import json
import math
import os
import random
import signal
import sys
import threading
import time

from neuropt._parsing import extract_json
from neuropt.search_space import Categorical, IntUniform, LogUniform, Uniform

DEFAULT_ML_CONTEXT = """\
You are an ML researcher running hyperparameter and architecture search.
Analyze the experiment history and propose better configurations.

**Reading the training curves (this is your main signal):**
- Train loss dropping + val loss rising = OVERFITTING → add regularization, reduce size, more weight decay
- Train loss stuck high = UNDERFITTING → more capacity, less regularization, higher LR
- Train AND val both dropping smoothly = GOOD FIT → stay nearby, fine-tune
- Loss oscillating or exploding = LR too high → reduce by 2-5x
- Very slow convergence = LR too low → increase LR
- Train-val gap small but both plateaued = capacity ceiling → try different architecture

**Strategy:**
- Early on: explore broadly, try diverse configs
- After seeing patterns: exploit what works, refine around winning configs
- Always include at least 1 exploratory config that tries something new
"""


class ArchSearch:
    """LLM-guided search over any parameter space.

    **Simplest usage** — from an existing model::

        search = ArchSearch.from_model(my_resnet, train_fn)
        search.run()

    **Custom search space** — plain Python, we infer the types::

        search = ArchSearch(train_fn, {
            "lr": (1e-4, 1e-1),                    # auto → LogUniform
            "hidden_dim": (32, 512),                # auto → IntUniform
            "activation": ["relu", "gelu", "silu"], # auto → Categorical
            "use_bn": [True, False],                # auto → Categorical
        })
        search.run()

    **In a notebook** — stop after N experiments::

        search.run(max_evals=50)
    """

    # Labels used in console output; DecisionSearch overrides them.
    _title = "LLM-Guided Search"
    _decider_label = "LLM"
    # Show at most this many epochs per curve in the state (None = all).
    _max_curve_points = None

    @classmethod
    def from_model(cls, model, train_fn, backend="auto", pretrained=None, **kwargs):
        """Create a search by introspecting a model.

        Works with PyTorch nn.Module (swaps activations, dropout, batch norm,
        layer norm, and fine-tuning strategies for pretrained models) and
        sklearn-compatible models like XGBoost, LightGBM, sklearn estimators
        (discovers tunable params, asks the LLM for ranges).

        Your ``train_fn`` receives ``config["model"]`` — a modified copy of
        your model. Just train it and return results.

        Args:
            model: PyTorch nn.Module or sklearn-compatible estimator.
            train_fn: Training function — receives config dict with "model" key.
            backend: LLM backend (same as __init__).
            pretrained: Override pretrained detection for PyTorch models.
                If True/False, skips auto-detection. If None (default),
                auto-detects by checking parameter statistics.
            **kwargs: Passed to __init__ (log_path, batch_size, device, etc).
        """
        from neuropt.introspect import is_sklearn_compatible

        if is_sklearn_compatible(model):
            return cls._from_sklearn_model(model, train_fn, backend, **kwargs)
        else:
            return cls._from_pytorch_model(model, train_fn, backend, pretrained=pretrained, **kwargs)

    @classmethod
    def _from_pytorch_model(cls, model, train_fn, backend, pretrained=None, **kwargs):
        from neuropt.introspect import (
            build_ml_context,
            build_search_space,
            introspect,
            make_wrapped_train_fn,
        )

        info = introspect(model, pretrained=pretrained)
        search_space = build_search_space(info)
        ml_context = build_ml_context(info)
        wrapped_fn = make_wrapped_train_fn(model, train_fn, info)

        print(f"Introspected PyTorch model ({info['n_params']:,} params):")
        if info["activation_paths"]:
            print(f"  Activations: {', '.join(sorted(info['activations_found']))} "
                  f"({len(info['activation_paths'])} layers)")
        if info["has_dropout"]:
            groups = info.get("dropout_groups", {})
            non_default = {k: v for k, v in groups.items() if k != "default"}
            if len(non_default) >= 2:
                group_desc = ", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items()))
                print(f"  Dropout by path: {group_desc}")
            else:
                print(f"  Dropout: {len(info['dropout_paths'])} layers (rate={info['dropout_rate']:.2f})")
        if info["has_batchnorm"]:
            print(f"  BatchNorm: {len(info['batchnorm_paths'])} layers")
        if info.get("has_layernorm"):
            print(f"  LayerNorm: {len(info['layernorm_paths'])} layers")
        if info.get("has_pool"):
            print(f"  Pooling: {len(info['pool_paths'])} layers (current: {info.get('pool_type', '?')})")
        if info.get("mha_dropout_paths"):
            print(f"  MHA dropout: {len(info['mha_dropout_paths'])} layers")
        if info.get("is_pretrained"):
            print("  Pretrained: yes (fine-tuning strategies enabled)")
            if info.get("last_linear_path"):
                print(f"  Head: {info['last_linear_path']}")
        print(f"  Search space: {list(search_space.keys())}")
        print()

        return cls(
            train_fn=wrapped_fn,
            search_space=search_space,
            backend=backend,
            ml_context=ml_context,
            **kwargs,
        )

    @classmethod
    def _from_sklearn_model(cls, model, train_fn, backend, **kwargs):
        from neuropt.introspect import (
            _fallback_sklearn_search_space,
            build_sklearn_ml_context,
            build_sklearn_search_space_with_llm,
            introspect_sklearn,
            make_sklearn_wrapped_train_fn,
        )

        info = introspect_sklearn(model)
        resolved_backend = cls._resolve_backend_for(backend)

        if resolved_backend is not None and hasattr(resolved_backend, "generate"):
            print(f"Asking LLM for search ranges for {info['model_type']}...")
            search_space = build_sklearn_search_space_with_llm(info, resolved_backend)
        else:
            search_space = _fallback_sklearn_search_space(info)

        ml_context = build_sklearn_ml_context(info, search_space)
        wrapped_fn = make_sklearn_wrapped_train_fn(model, train_fn)

        print(f"Introspected {info['model_type']}:")
        print(f"  Tunable params: {len(info['tunable_params'])}")
        print(f"  Search space: {list(search_space.keys())}")
        print()

        return cls(
            train_fn=wrapped_fn,
            search_space=search_space,
            backend=resolved_backend,
            ml_context=ml_context,
            **kwargs,
        )

    def __init__(
        self,
        train_fn,
        search_space: dict,
        backend="auto",
        log_path: str = "search.jsonl",
        batch_size: int = 3,
        device: str | None = None,
        timeout: int = 600,
        ml_context: str | None = None,
        minimize: bool = True,
        seed: int | None = None,
    ):
        """
        Args:
            train_fn: Callable(config_dict) -> dict. Must return at least
                ``{"score": float}``. For curve-aware search, also return
                ``train_losses``, ``val_losses``, ``val_accuracies`` as lists.
            search_space: Dict mapping param names to values. Values can be:

                - A dimension object: ``LogUniform(1e-4, 1e-1)``
                - A tuple of two numbers: ``(1e-4, 1e-1)`` — auto-inferred
                - A list: ``["relu", "gelu"]`` — becomes Categorical
                - We pick LogUniform vs Uniform vs IntUniform based on
                  the param name and value types (see ``_infer_dim``).

            backend: "auto", "claude", "openai", "qwen", "none", or instance.
            log_path: JSONL log file path (supports resume).
            batch_size: Configs proposed per LLM call.
            device: Injected into config as ``config["device"]`` if set.
            timeout: Max seconds per experiment. Enforced with SIGALRM, so it
                only applies when ``run()`` is called from the main thread
                on a POSIX system, and it can't interrupt a long call into a
                C extension until that call returns to Python.
            ml_context: Domain knowledge for the LLM (appended to default
                prompt). Tell it about your dataset, architecture, etc.
            minimize: If True (default), lower scores are better (loss).
                If False, higher scores are better (accuracy, AUROC).
            seed: Seed for the random configs this search draws (random
                fallback, decision-model sampling). Default 42.
        """
        self.train_fn = train_fn
        self.search_space = _normalize_search_space(search_space)
        self.batch_size = batch_size
        self.device = device
        self.timeout = timeout
        self.log_path = log_path
        self.ml_context = ml_context or DEFAULT_ML_CONTEXT
        self.minimize = minimize
        self.seed = 42 if seed is None else seed

        self._backend = self._resolve_backend_for(backend)
        self._shutdown = False
        self._warned_no_timeout = False

        # State
        self.best_score = float("inf") if minimize else float("-inf")
        self.best_config = None
        self.best_accuracy = 0.0
        self.total_experiments = 0
        self.llm_success = 0
        self.llm_fallback = 0

    @classmethod
    def _resolve_backend_for(cls, backend):
        """Turn the ``backend`` argument into a backend instance (or None)."""
        resolved = _resolve_backend(backend)
        from neuropt.backends.base import BaseDecisionBackend
        if isinstance(resolved, BaseDecisionBackend):
            raise TypeError(
                f"{type(resolved).__name__} is a decision model: it answers multiple-choice "
                "questions and can't write configs. Use neuropt.DecisionSearch with it instead "
                "of ArchSearch."
            )
        return resolved

    def run(self, max_evals: int | None = None, resume: bool = True):
        """Run the search loop.

        Args:
            max_evals: Stop after this many experiments. If None, runs
                until Ctrl+C (useful for CLI / overnight). Set this for
                notebooks or scripted use.
            resume: If True (default), continue from existing log file.
                If False, start fresh by clearing any existing log.
        """
        self._shutdown = False
        previous_handlers = self._setup_signals()
        try:
            self._run_loop(max_evals, resume)
        finally:
            self._restore_signals(previous_handlers)

    def _run_loop(self, max_evals, resume):
        if not resume and os.path.exists(self.log_path):
            os.remove(self.log_path)

        logger = _JSONLLogger(self.log_path)
        history = logger.load_history()
        self.total_experiments = len(history)
        iteration = 0

        # Restore best from existing log (handles both old and new format)
        for row in history:
            if row.get("status") == "ok":
                score = _get_score(row)
                if score is not None and self._is_better(score):
                    self.best_score = score
                    self.best_config = row.get("config")
                    self.best_accuracy = _get_accuracy(row)

        backend_name = self._backend.name if self._backend else "None (random)"
        print("=" * 60)
        print(self._title)
        print(f"  Backend: {backend_name}")
        print(f"  Device: {self.device or 'auto'}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Log: {self.log_path}")
        if max_evals:
            print(f"  Max evals: {max_evals}")
        if self.total_experiments:
            print(f"  Resuming from {self.total_experiments} experiments")
            print(f"  Best so far: score={self.best_score:.4f} acc={self.best_accuracy:.4f}")
        if not max_evals:
            print("  Ctrl+C to stop")
        print("=" * 60)
        print()
        evals_at_start = self.total_experiments

        rng = random.Random(self.seed + self.total_experiments)

        while not self._shutdown:
            if max_evals and (self.total_experiments - evals_at_start) >= max_evals:
                break

            iter_start = time.time()

            configs, source = self._get_configs(history, rng)

            for idx, cfg in enumerate(configs):
                if self._shutdown:
                    break
                if max_evals and (self.total_experiments - evals_at_start) >= max_evals:
                    break

                run_cfg = dict(cfg)
                if self.device is not None:
                    run_cfg["device"] = self.device

                result = self._run_one(run_cfg)

                improved = ""
                if self._is_better(result["score"]):
                    self.best_score = result["score"]
                    self.best_config = cfg
                    self.best_accuracy = _get_accuracy(result)
                    improved = " *** NEW BEST ***"

                logger.log(iteration, cfg, result, source)
                history.append({
                    "id": self.total_experiments + 1,
                    "config": cfg,
                    "score": result["score"],
                    "scalars": result.get("scalars", {}),
                    "curves": result.get("curves", {}),
                    "status": result["status"],
                })
                self.total_experiments += 1

                # Compact status line — show score + up to 3 scalar extras
                extra_parts = _format_scalars(result.get("scalars", {}), max_n=3)
                extra = f" ({' '.join(extra_parts)})" if extra_parts else ""
                status_s = f" [{result['status']}]" if result["status"] != "ok" else ""
                cfg_s = _short_config(cfg)
                print(f"  [{iteration}.{idx}] {cfg_s} → {result['score']:.4f}"
                      f"{extra} {result['elapsed']:.1f}s [{source}]{status_s}{improved}")

            if self._shutdown:
                break

            print(f"  iter {iteration} done in {time.time() - iter_start:.1f}s | "
                  f"best: {self.best_score:.4f} | total: {self.total_experiments}")
            if self._backend:
                print(f"  {self._decider_label.lower()}: {self.llm_success} ok, "
                      f"{self.llm_fallback} fallback")
            print()
            iteration += 1

        self._print_summary(iteration)

    # ── Config generation ──────────────────────────────────────────────────

    MAX_RETRIES = 3

    def _get_configs(self, history, rng):
        """Ask LLM for configs with retry loop for duplicates."""
        if self._backend is None:
            return [_random_config(self.search_space, rng)
                    for _ in range(self.batch_size)], "random"

        prompt = self._build_prompt(history)

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._backend.generate(prompt, max_tokens=2048)
                configs = self._parse_response(response)
                if configs is not None:
                    # Top up a short (but valid) answer with random configs
                    while len(configs) < self.batch_size:
                        configs.append(_random_config(self.search_space, rng))

                if configs is None:
                    self.llm_fallback += 1
                    # Show first 200 chars of response for debugging
                    snippet = response[:200].replace('\n', ' ') if response else "(empty)"
                    print(f"  [LLM parse failed, using random] response preview: {snippet}")
                    break

                # Check for duplicates against history
                dupes = self._find_duplicates(configs, history)
                if not dupes:
                    self.llm_success += 1
                    return configs, "llm"

                # Tell the LLM what it repeated and ask for something new
                if attempt < self.MAX_RETRIES - 1:
                    print(f"  [retry {attempt+1}: {len(dupes)} duplicate configs, asking LLM to adjust]")
                    prompt = self._build_retry_prompt(configs, dupes, history)
                else:
                    # Last attempt — keep non-duplicate configs, replace dupes with random
                    print(f"  [still {len(dupes)} duplicates after {self.MAX_RETRIES} tries, "
                          f"replacing with random]")
                    for i in dupes:
                        configs[i] = _random_config(self.search_space, rng)
                    self.llm_success += 1
                    return configs, "llm"

            except Exception as e:
                self.llm_fallback += 1
                print(f"  [LLM error: {e}, using random]")
                break

        return [_random_config(self.search_space, rng)
                for _ in range(self.batch_size)], "random"

    def _find_duplicates(self, configs, history):
        """Return indices of configs that match history OR each other in the batch."""
        seen = set()
        for row in history:
            cfg = row.get("config", {})
            seen.add(_config_key(cfg))

        dupes = []
        for i, cfg in enumerate(configs):
            key = _config_key(cfg)
            if key in seen:
                dupes.append(i)
            else:
                seen.add(key)  # also dedup within the batch
        return dupes

    def _build_retry_prompt(self, configs, dupe_indices, history):
        """Build a follow-up prompt telling the LLM what it repeated."""
        parts = [self.ml_context.strip(), ""]

        # Search space (abbreviated)
        parts.append("## Config Space\n")
        for name, dim in self.search_space.items():
            parts.append(_describe_dim(name, dim))
        parts.append("")

        # What it proposed that was already tried
        parts.append("## You proposed configs that were already tried\n")
        parts.append("These configs have already been evaluated — don't propose them again:\n")

        for i in dupe_indices:
            cfg = configs[i]
            key = _config_key(cfg)
            # Find the matching result
            prev_result = None
            for row in reversed(history):
                if _config_key(row.get("config", {})) == key:
                    prev_result = row
                    break

            parts.append(f"Config {i+1}: {json.dumps(cfg, default=str)}")
            if prev_result:
                score = _get_score(prev_result)
                score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
                result_parts = [f"score={score_s}"]
                result_parts += _format_scalars(prev_result.get("scalars", {}), max_n=3)
                parts.append(f"  → Result: {', '.join(result_parts)}")
            parts.append("")

        # Best result for context
        if self.best_config is not None:
            parts.append(f"## Current best: score={self.best_score:.4f}")
            parts.append(f"Config: {json.dumps(self.best_config, default=str)}")
            parts.append("")

        parts.append("## Task\n")
        parts.append(
            f"Propose exactly {self.batch_size} NEW configs that are DIFFERENT from "
            "anything already tried. Focus on changes that could beat the current best. "
            "Try meaningfully different values, not tiny tweaks.\n"
            "Respond with ONLY a JSON array of config objects."
        )
        return "\n".join(parts)

    # ── Experiment runner ──────────────────────────────────────────────────

    def _run_one(self, cfg):
        """Run train_fn with timeout and error handling."""
        old_handler = None
        start = time.time()
        try:
            if _in_main_thread():
                try:
                    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                    signal.alarm(self.timeout)
                except (AttributeError, OSError, ValueError):
                    pass
            elif not self._warned_no_timeout:
                self._warned_no_timeout = True
                print("  [timeout disabled: run() is not on the main thread]")

            raw = self.train_fn(cfg)
            elapsed = time.time() - start

            if old_handler is not None:
                try:
                    signal.alarm(0)
                except (AttributeError, OSError):
                    pass

            if isinstance(raw, (int, float)):
                raw = {"score": float(raw)}

            worst = float("inf") if self.minimize else float("-inf")
            score = float(raw.get("score", worst))
            if math.isnan(score) or math.isinf(score):
                score = worst

            # Separate user-returned keys into scalars and curves (lists of numbers)
            _internal = {"score", "status", "error"}
            scalars = {}
            curves = {}
            for k, v in raw.items():
                if k in _internal:
                    continue
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    curves[k] = v
                elif isinstance(v, (int, float, str, bool)):
                    scalars[k] = v

            return {
                "score": score,
                "scalars": scalars,
                "curves": curves,
                "elapsed": elapsed,
                "status": raw.get("status", "ok"),
                "error": raw.get("error", ""),
            }

        except _Timeout:
            worst = float("inf") if self.minimize else float("-inf")
            return {"score": worst, "scalars": {}, "curves": {},
                    "elapsed": time.time() - start,
                    "status": "timeout", "error": f">{self.timeout}s"}
        except Exception as e:
            worst = float("inf") if self.minimize else float("-inf")
            return {"score": worst, "scalars": {}, "curves": {},
                    "elapsed": time.time() - start,
                    "status": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            if old_handler is not None:
                try:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                except (AttributeError, OSError, ValueError):
                    pass

    # ── Prompt building ────────────────────────────────────────────────────

    def _build_prompt(self, history):
        return self._build_state(history) + "\n" + self._task_text()

    def _build_state(self, history):
        """Render everything the model should know: context, space, history, curves.

        Shared by the LLM prompt and the decision-model state, so both see the
        same evidence.
        """
        parts = [self.ml_context.strip(), ""]

        # Search space
        parts.append("## Config Space\n")
        parts.append("Each config is a JSON object with these keys:")
        for name, dim in self.search_space.items():
            parts.append(_describe_dim(name, dim))
        parts.append("")

        direction = "lower is better" if self.minimize else "higher is better"
        parts.append(f"Score direction: {direction}\n")

        # Best result
        if self.best_config is not None:
            parts.append("## Best Result So Far")
            best_extra = ""
            if self.best_accuracy:
                best_extra = f", accuracy: {self.best_accuracy:.4f}"
            parts.append(f"Score: {self.best_score:.4f}{best_extra}")
            parts.append(f"Config: {json.dumps(self.best_config, default=str)}")
            parts.append("")

        # Recent history table
        recent = history[-20:]
        if recent:
            # Auto-detect scalar keys from history
            scalar_keys = _collect_scalar_keys(recent)
            curve_keys = _collect_curve_keys(recent)

            parts.append("## Recent Experiments (last 20)\n")
            cfg_keys = list(self.search_space.keys())
            header = ["#"] + cfg_keys + ["score"] + scalar_keys + ["status"]
            parts.append("| " + " | ".join(header) + " |")
            parts.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in recent:
                cfg = row.get("config", {})
                score = _get_score(row)
                score_s = "-" if score is None else f"{score:.4f}"
                vals = [str(row.get("id", ""))]
                for k in cfg_keys:
                    v = cfg.get(k, "")
                    if isinstance(v, float):
                        v = f"{v:.4g}"
                    vals.append(str(v))
                vals.append(score_s)
                scalars = row.get("scalars", {})
                for k in scalar_keys:
                    v = scalars.get(k, _compat_scalar(row, k))
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    elif v is not None:
                        vals.append(str(v))
                    else:
                        vals.append("-")
                vals.append(row.get("status", ""))
                parts.append("| " + " | ".join(vals) + " |")
            parts.append("")

            # Per-epoch curves — auto-detect from any list-of-numbers keys
            if curve_keys:
                ok_with_curves = [r for r in recent
                                  if r.get("status") == "ok"
                                  and any(_get_curve(r, k) for k in curve_keys)]
                if ok_with_curves:
                    parts.append("## Per-Epoch Curves (last 8 successful)\n")
                    for row in ok_with_curves[-8:]:
                        cfg = row["config"]
                        label = ", ".join(f"{k}={_fmt(cfg.get(k))}"
                                         for k in list(cfg.keys())[:5])
                        parts.append(f"#{row.get('id', '?')} {label}:")
                        # Gather all curves for this row
                        row_curves = {k: _get_curve(row, k) for k in curve_keys}
                        row_curves = {k: v for k, v in row_curves.items() if v}
                        max_epochs = max((len(v) for v in row_curves.values()), default=0)
                        for e in _curve_indices(max_epochs, self._max_curve_points):
                            line = f"  ep{e+1}:"
                            for k, v in row_curves.items():
                                if e < len(v):
                                    line += f" {k}={v[e]:.4f}"
                            parts.append(line)
                    parts.append("")

            # Overfitting/underfitting signals (uses train_losses/val_losses if available)
            signals = []
            for row in recent[-8:]:
                if row.get("status") != "ok":
                    continue
                tl = _get_curve(row, "train_losses")
                vl = _get_curve(row, "val_losses")
                if tl and vl and len(tl) >= 2 and len(vl) >= 2:
                    signal_name = _fit_signal(tl, vl)
                    if signal_name == "overfitting":
                        signals.append(f"- OVERFITTING: #{row.get('id', '?')} "
                                       f"{_short_config(row['config'])} "
                                       f"(gap={vl[-1] - tl[-1]:.3g})")
                    elif signal_name == "underfitting":
                        signals.append(f"- UNDERFITTING: #{row.get('id', '?')} "
                                       f"{_short_config(row['config'])} "
                                       f"(train {tl[0]:.3g} -> {tl[-1]:.3g})")
            if signals:
                parts.append("## Signals\n")
                parts.extend(signals)
                parts.append("")

        return "\n".join(parts)

    def _task_text(self):
        return "\n".join([
            "## Task\n",
            f"Propose exactly {self.batch_size} configs to try next.\n\n"
            "Rules:\n"
            "- Every config must be DIFFERENT from the history AND from each other\n"
            "- At least 1 config should explore something new (different activation, "
            "architecture size, optimizer, or a parameter you haven't varied recently)\n"
            "- Don't just repeat the best config with tiny tweaks — try to find a "
            "meaningfully better design\n"
            "- If the best configs are clustering, try something outside that cluster\n\n"
            "Respond with ONLY a JSON array of config objects. "
            "No explanation, no markdown fences."
        ])

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(self, response):
        """Parse LLM JSON into validated configs.

        Returns up to ``batch_size`` configs (extras are dropped), or None if
        there is no JSON array, it is empty, or any config is invalid.
        """
        configs = extract_json(response, "[")
        if not configs:
            return None
        configs = configs[:self.batch_size]

        validated = []
        for cfg in configs:
            cleaned = _validate_config(cfg, self.search_space)
            if cleaned is None:
                return None
            validated.append(cleaned)
        return validated

    # ── Signal handling ────────────────────────────────────────────────────

    def _is_better(self, score):
        if self.minimize:
            return score < self.best_score
        return score > self.best_score

    def _setup_signals(self):
        """Install Ctrl+C / SIGTERM handlers; return the previous ones.

        Signal handlers can only be set from the main thread, so elsewhere this
        is a no-op (Ctrl+C then interrupts the way it normally would).
        """
        if not _in_main_thread():
            return None

        def handler(signum, frame):
            if self._shutdown:
                print("\nForce exit.")
                sys.exit(1)
            print("\nStopping after current experiment...")
            self._shutdown = True

        previous = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        return previous

    @staticmethod
    def _restore_signals(previous):
        if not previous:
            return
        for sig, old in previous.items():
            if old is not None:
                signal.signal(sig, old)

    def _print_summary(self, iteration):
        print("\n" + "=" * 60)
        print("DONE")
        print(f"  Experiments: {self.total_experiments}")
        print(f"  Best score: {self.best_score:.4f}")
        if self.best_accuracy:
            print(f"  Best accuracy: {self.best_accuracy:.4f}")
        if self.best_config:
            print(f"  Best config: {json.dumps(self.best_config, default=str)}")
        if self._backend:
            print(f"  {self._decider_label}: {self.llm_success} ok, {self.llm_fallback} fallback")
            tokens_in = self._backend.total_input_tokens
            tokens_out = self._backend.total_output_tokens
            if tokens_in or tokens_out:
                print(f"  Tokens: {tokens_in:,} in + {tokens_out:,} out")
                cost = self._backend.total_cost
                if cost is not None:
                    print(f"  API cost: ${cost:.4f}")
        print("=" * 60)


# ── Helpers (module-level) ────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _curve_indices(n, max_points):
    """Epoch indices to show: all of them, or ``max_points`` evenly spaced
    ones that always include the first and last epoch."""
    if max_points is None or n <= max_points:
        return range(n)
    if max_points < 2:
        return [n - 1]
    return sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})


def _in_main_thread():
    return threading.current_thread() is threading.main_thread()


def _fit_signal(train_losses, val_losses):
    """Classify a pair of loss curves as "overfitting", "underfitting" or None.

    Uses relative thresholds so it works for any loss scale (cross-entropy,
    MSE, ...): overfitting = train still falling while val has climbed more
    than 5% off its minimum and sits well above train; underfitting = train
    loss improved by less than 10% over the run.
    """
    t0, t_end, v_end = train_losses[0], train_losses[-1], val_losses[-1]
    v_min = min(val_losses)
    gap = v_end - t_end
    if t_end < t0 and v_end > v_min + 0.05 * abs(v_min) and gap > 0.2 * abs(v_end):
        return "overfitting"
    if t0 > 0 and t_end > 0.9 * t0:
        return "underfitting"
    return None


class _JSONLLogger:
    def __init__(self, path):
        self.path = path
        self._counter = 0
        if os.path.exists(path):
            self._counter = len(self.load_history(warn=False))
            # A crash mid-write can leave a partial last line with no newline;
            # start a fresh line so the next row isn't glued onto it.
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() > 0:
                    f.seek(-1, os.SEEK_END)
                    needs_newline = f.read(1) != b"\n"
                else:
                    needs_newline = False
            if needs_newline:
                with open(path, "a") as f:
                    f.write("\n")

    def log(self, iteration, cfg, result, source):
        self._counter += 1
        entry = {
            "id": self._counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            "source": source,
            "config": cfg,
            "score": result.get("score"),
            "scalars": result.get("scalars", {}),
            "curves": result.get("curves", {}),
            "elapsed": result.get("elapsed"),
            "status": result.get("status"),
            "error": result.get("error", ""),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()

    def load_history(self, warn=True):
        """Read all rows, skipping lines that aren't valid JSON objects."""
        if not os.path.exists(self.path):
            return []
        rows = []
        skipped = 0
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    skipped += 1
        if skipped and warn:
            print(f"  [skipped {skipped} unreadable line(s) in {self.path}]")
        return rows


# Names that should use log-uniform sampling (span orders of magnitude)
_LOG_SCALE_NAMES = {
    "lr", "learning_rate", "wd", "weight_decay", "l2", "l2_reg",
    "eps", "epsilon", "min_lr", "max_lr", "base_lr",
}

# Names that are clearly integer-valued
_INT_NAMES = {
    "n_layers", "num_layers", "n_blocks", "num_blocks", "depth",
    "hidden_dim", "hidden_size", "n_hidden", "width", "n_heads",
    "num_heads", "n_units", "num_units", "embed_dim", "embedding_dim",
    "base_channels", "channels", "n_channels", "fc_hidden",
    "batch_size", "pool_every", "kernel_size", "epochs", "n_epochs",
}


def _infer_dim(name, value):
    """Infer a dimension type from a param name and raw value.

    Handles:
        - list → Categorical
        - tuple of 2 numbers → LogUniform/Uniform/IntUniform based on name
        - Already a dimension object → pass through
    """
    # Already a dimension object
    if isinstance(value, (LogUniform, Uniform, IntUniform, Categorical)):
        return value

    # List → Categorical
    if isinstance(value, list):
        return Categorical(value)

    # Tuple of (low, high) → infer numeric type. Precedence:
    #   1. int bounds + int-like name → IntUniform (even if the ratio is large)
    #   2. log-scale name, or ratio > 100 → LogUniform
    #   3. int bounds → IntUniform
    #   4. otherwise → Uniform
    if isinstance(value, tuple) and len(value) == 2:
        lo, hi = value
        both_int = isinstance(lo, int) and isinstance(hi, int)
        int_like_name = (name.lower() in _INT_NAMES
                         or name.startswith("n_") or name.startswith("num_"))

        if both_int and int_like_name:
            return IntUniform(lo, hi)

        lo, hi = float(lo), float(hi)
        log_like = name.lower() in _LOG_SCALE_NAMES or (lo > 0 and hi / lo > 100)
        if log_like and lo > 0:
            return LogUniform(lo, hi)

        if both_int:
            return IntUniform(int(lo), int(hi))

        return Uniform(lo, hi)

    raise ValueError(
        f"Can't infer dimension for '{name}': {value!r}. "
        f"Use a tuple (lo, hi), a list of choices, or a dimension object."
    )


def _normalize_search_space(space):
    """Convert a user-friendly search space dict to dimension objects.

    Accepts plain tuples, lists, or dimension objects.
    """
    normalized = {}
    for name, value in space.items():
        normalized[name] = _infer_dim(name, value)
    return normalized


def _resolve_backend(backend):
    if backend is None or backend == "none":
        return None
    if backend == "auto":
        from neuropt.backends import get_default_backend
        return get_default_backend()
    if isinstance(backend, str):
        from neuropt.backends import get_backend_by_name
        return get_backend_by_name(backend)
    return backend  # assume it's a backend instance


def _describe_dim(name, dim):
    if isinstance(dim, Categorical):
        choices = dim.choices
        if choices == [True, False] or choices == [False, True]:
            return f'- "{name}": true or false'
        return f'- "{name}": one of {choices}'
    elif isinstance(dim, LogUniform):
        return f'- "{name}": float [{dim.low}, {dim.high}] (log scale)'
    elif isinstance(dim, IntUniform):
        return f'- "{name}": integer [{dim.low}, {dim.high}]'
    elif isinstance(dim, Uniform):
        return f'- "{name}": float [{dim.low}, {dim.high}]'
    return f'- "{name}": unknown type'


def _random_config(search_space, rng=None):
    if rng is None:
        rng = random
    cfg = {}
    for name, dim in search_space.items():
        if isinstance(dim, Categorical):
            cfg[name] = rng.choice(dim.choices)
        elif isinstance(dim, LogUniform):
            lo, hi = dim.bounds()
            cfg[name] = dim.from_internal(rng.uniform(lo, hi))
        elif isinstance(dim, IntUniform):
            cfg[name] = rng.randint(dim.low, dim.high)
        elif isinstance(dim, Uniform):
            cfg[name] = rng.uniform(dim.low, dim.high)
    return cfg


def _validate_config(cfg, search_space):
    if not isinstance(cfg, dict):
        return None
    cleaned = {}
    for name, dim in search_space.items():
        if name not in cfg:
            return None
        val = cfg[name]

        if isinstance(dim, Categorical):
            # Handle bool coercion from JSON
            if dim.choices in ([True, False], [False, True]):
                if isinstance(val, str):
                    val = val.lower() in ("true", "1", "yes")
                else:
                    val = bool(val)
            choice = _match_choice(val, dim.choices)
            if choice is _NO_MATCH:
                return None
            cleaned[name] = choice
        elif isinstance(dim, IntUniform):
            try:
                val = int(round(float(val)))
            except (TypeError, ValueError):
                return None
            val = max(dim.low, min(val, dim.high))
            cleaned[name] = val
        elif isinstance(dim, (LogUniform, Uniform)):
            try:
                val = float(val)
            except (TypeError, ValueError):
                return None
            lo, hi = dim.bounds()
            internal = dim.to_internal(val)
            internal = max(lo, min(internal, hi))
            cleaned[name] = dim.from_internal(internal)
    return cleaned


_NO_MATCH = object()


def _match_choice(val, choices):
    """Return the element of ``choices`` that ``val`` refers to.

    Returns the choice object itself (so ``64.0`` for choices ``[32, 64]``
    comes back as the int ``64``). Bools only match bools, so ``True`` is not
    accepted for ``[0, 1]``. (For ``[True, False]`` the caller coerces JSON
    values like ``"true"`` or ``1`` to a bool first.)
    """
    for c in choices:
        if type(c) is type(val) and c == val:
            return c
    for c in choices:
        if isinstance(c, bool) or isinstance(val, bool):
            continue
        try:
            if c == val:
                return c
        except Exception:
            continue
    return _NO_MATCH


def _fmt(v):
    if isinstance(v, float):
        if v < 0.01 or v > 1000:
            return f"{v:.2e}"
        return f"{v:.3f}"
    return str(v)


def _config_key(cfg):
    """Hashable key for a config dict, rounding floats for fuzzy matching."""
    parts = []
    for k in sorted(cfg.keys()):
        if k in ("device", "model"):
            continue
        v = cfg[k]
        if isinstance(v, float) and math.isfinite(v):
            v = float(f"{v:.6g}")  # 6 significant figures, not 6 decimal places
        parts.append((k, v))
    return tuple(parts)


def _short_config(cfg):
    """One-line summary of a config dict."""
    parts = []
    for k, v in cfg.items():
        if k in ("device", "model"):
            continue
        parts.append(f"{k}={_fmt(v)}")
    return " ".join(parts)


# ── Auto-detect scalars and curves from history ──────────────────────────

def _get_score(row):
    """Get score from a history row (handles old and new log format)."""
    return row.get("score", row.get("val_loss"))


def _get_accuracy(row):
    """Get accuracy from a result/history row (handles old and new log format)."""
    scalars = row.get("scalars", {})
    return scalars.get(
        "accuracy", scalars.get("val_accuracy", row.get("val_accuracy", 0)))


def _format_scalars(scalars, max_n=3):
    """Format up to max_n scalars as ["k=v", ...] strings for display."""
    parts = []
    for k, v in list(scalars.items())[:max_n]:
        if isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, int):
            parts.append(f"{k}={v:,}")
        else:
            parts.append(f"{k}={v}")
    return parts


def _get_curve(row, key):
    """Get a curve list from a history row (handles old and new format)."""
    curves = row.get("curves", {})
    if key in curves:
        return curves[key]
    # Old format: curves were top-level keys
    v = row.get(key)
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return v
    return []


def _compat_scalar(row, key):
    """Get a scalar from old-format log rows (before scalars dict)."""
    # Old format stored these as top-level keys
    v = row.get(key)
    if isinstance(v, (int, float, str, bool)):
        return v
    # Also check common old-format key mappings
    if key == "accuracy":
        return row.get("val_accuracy")
    return None


def _collect_scalar_keys(history):
    """Auto-detect scalar keys returned by train_fn across history."""
    keys = set()
    for row in history:
        scalars = row.get("scalars", {})
        keys.update(scalars.keys())
        # Old format compat: check for known scalar top-level keys
        if not scalars:
            for k in ("accuracy", "val_accuracy", "n_params"):
                if row.get(k) is not None:
                    keys.add(k)
    # Stable ordering
    return sorted(keys)


def _collect_curve_keys(history):
    """Auto-detect per-epoch curve keys returned by train_fn across history."""
    keys = set()
    for row in history:
        curves = row.get("curves", {})
        keys.update(curves.keys())
        # Old format compat: check for known curve top-level keys
        if not curves:
            for k in ("train_losses", "val_losses", "val_accuracies"):
                v = row.get(k)
                if isinstance(v, list) and v:
                    keys.add(k)
    return sorted(keys)
