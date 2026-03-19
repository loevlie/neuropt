"""LLM-guided hyperparameter and architecture search.

The LLM reads per-epoch training curves and proposes what to try next.
Runs until Ctrl+C. Everything is logged to JSONL for crash recovery.
"""

import json
import math
import os
import random
import re
import signal
import sys
import time
import traceback

from neuropt.search_space import Categorical, IntUniform, LogUniform, Uniform


DEFAULT_ML_CONTEXT = """\
You are an ML researcher running hyperparameter and architecture search.
Analyze the experiment history and propose better configurations.

**Reading the training curves (this is your main signal):**
- Train loss dropping + val loss rising = OVERFITTING → add regularization, reduce model size, increase weight decay
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

    @classmethod
    def from_model(cls, model, train_fn, backend="auto", **kwargs):
        """Create a search by introspecting a model.

        Works with PyTorch nn.Module (swaps activations, dropout, batch norm)
        and sklearn-compatible models like XGBoost, LightGBM, sklearn estimators
        (discovers tunable params, asks the LLM for ranges).

        Your ``train_fn`` receives ``config["model"]`` — a modified copy of
        your model. Just train it and return results.

        Args:
            model: PyTorch nn.Module or sklearn-compatible estimator.
            train_fn: Training function — receives config dict with "model" key.
            backend: LLM backend (same as __init__).
            **kwargs: Passed to __init__ (log_path, batch_size, device, etc).
        """
        from neuropt.introspect import is_sklearn_compatible

        if is_sklearn_compatible(model):
            return cls._from_sklearn_model(model, train_fn, backend, **kwargs)
        else:
            return cls._from_pytorch_model(model, train_fn, backend, **kwargs)

    @classmethod
    def _from_pytorch_model(cls, model, train_fn, backend, **kwargs):
        from neuropt.introspect import (
            introspect, build_search_space, build_ml_context,
            make_wrapped_train_fn,
        )

        info = introspect(model)
        search_space = build_search_space(info)
        ml_context = build_ml_context(info)
        wrapped_fn = make_wrapped_train_fn(model, train_fn, info)

        print(f"Introspected PyTorch model ({info['n_params']:,} params):")
        if info["activation_paths"]:
            print(f"  Activations: {', '.join(sorted(info['activations_found']))} "
                  f"({len(info['activation_paths'])} layers)")
        if info["has_dropout"]:
            print(f"  Dropout: {len(info['dropout_paths'])} layers (rate={info['dropout_rate']:.2f})")
        if info["has_batchnorm"]:
            print(f"  BatchNorm: {len(info['batchnorm_paths'])} layers")
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
            introspect_sklearn, build_sklearn_search_space_with_llm,
            build_sklearn_ml_context, make_sklearn_wrapped_train_fn,
            _fallback_sklearn_search_space,
        )

        info = introspect_sklearn(model)
        resolved_backend = _resolve_backend(backend)

        if resolved_backend is not None:
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
            backend=backend,
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
            timeout: Max seconds per experiment.
            ml_context: Domain knowledge for the LLM (appended to default
                prompt). Tell it about your dataset, architecture, etc.
        """
        self.train_fn = train_fn
        self.search_space = _normalize_search_space(search_space)
        self.batch_size = batch_size
        self.device = device
        self.timeout = timeout
        self.log_path = log_path
        self.ml_context = ml_context or DEFAULT_ML_CONTEXT

        self._backend = _resolve_backend(backend)
        self._shutdown = False

        # State
        self.best_score = float("inf")
        self.best_config = None
        self.best_accuracy = 0.0
        self.total_experiments = 0
        self.llm_success = 0
        self.llm_fallback = 0

        # LLM memory — insights persisted across iterations
        self._insights_path = log_path.rsplit(".", 1)[0] + "_insights.json"
        self._insights = self._load_insights()

    def run(self, max_evals: int | None = None):
        """Run the search loop.

        Args:
            max_evals: Stop after this many experiments. If None, runs
                until Ctrl+C (useful for CLI / overnight). Set this for
                notebooks or scripted use.
        """
        self._setup_signals()

        logger = _JSONLLogger(self.log_path)
        history = logger.load_history()
        self.total_experiments = len(history)
        iteration = 0

        # Restore best from existing log
        for row in history:
            if row.get("status") == "ok":
                vl = row.get("val_loss", float("inf"))
                if vl < self.best_score:
                    self.best_score = vl
                    self.best_config = row.get("config")
                    self.best_accuracy = row.get("val_accuracy", 0)

        backend_name = self._backend.name if self._backend else "None (random)"
        print("=" * 60)
        print("LLM-Guided Search")
        print(f"  Backend: {backend_name}")
        print(f"  Device: {self.device or 'auto'}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Log: {self.log_path}")
        if max_evals:
            print(f"  Max evals: {max_evals}")
        if self.total_experiments:
            print(f"  Resuming from {self.total_experiments} experiments")
            print(f"  Best so far: loss={self.best_score:.4f} acc={self.best_accuracy:.4f}")
        if not max_evals:
            print("  Ctrl+C to stop")
        print("=" * 60)
        print()
        evals_at_start = self.total_experiments
        self._max_evals = max_evals

        rng = random.Random(42 + self.total_experiments)

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
                if result["score"] < self.best_score:
                    self.best_score = result["score"]
                    self.best_config = cfg
                    self.best_accuracy = result.get("accuracy", 0)
                    improved = " *** NEW BEST ***"

                logger.log(iteration, cfg, result, source)
                history.append({
                    "id": self.total_experiments + 1,
                    "config": cfg,
                    "val_loss": result["score"],
                    "val_accuracy": result.get("accuracy"),
                    "n_params": result.get("n_params"),
                    "train_losses": result.get("train_losses", []),
                    "val_losses": result.get("val_losses", []),
                    "val_accuracies": result.get("val_accuracies", []),
                    "status": result["status"],
                })
                self.total_experiments += 1

                # Compact status line
                acc_s = f"acc={result['accuracy']:.4f}" if result.get("accuracy") else ""
                params_s = f"{result['n_params']:,}p" if result.get("n_params") else ""
                extra = " ".join(filter(None, [acc_s, params_s]))
                if extra:
                    extra = f" ({extra})"
                status_s = f" [{result['status']}]" if result["status"] != "ok" else ""
                cfg_s = _short_config(cfg)
                print(f"  [{iteration}.{idx}] {cfg_s} → {result['score']:.4f}"
                      f"{extra} {result['elapsed']:.1f}s [{source}]{status_s}{improved}")

            if self._shutdown:
                break

            # Extract insights only when experiments are about to scroll
            # out of the 20-experiment prompt window
            if (self._backend and source == "llm"
                    and len(history) > 20 and len(history) % 20 == 0):
                self._extract_insights(history)

            print(f"  iter {iteration} done in {time.time() - iter_start:.1f}s | "
                  f"best: {self.best_score:.4f} | total: {self.total_experiments}")
            if self._backend:
                print(f"  llm: {self.llm_success} ok, {self.llm_fallback} fallback")
            if self._insights:
                print(f"  insights: {len(self._insights)}")
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
                    print(f"  [still {len(dupes)} duplicates after {self.MAX_RETRIES} tries, replacing with random]")
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
                vl = prev_result.get("val_loss", "?")
                va = prev_result.get("val_accuracy", "?")
                vl_s = f"{vl:.4f}" if isinstance(vl, (int, float)) else str(vl)
                va_s = f"{va:.4f}" if isinstance(va, (int, float)) else str(va)
                parts.append(f"  → Result: loss={vl_s}, accuracy={va_s}")
            parts.append("")

        # Best result for context
        if self.best_config is not None:
            parts.append(f"## Current best: loss={self.best_score:.4f}")
            parts.append(f"Config: {json.dumps(self.best_config, default=str)}")
            parts.append("")

        parts.append(f"## Task\n")
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
            try:
                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(self.timeout)
            except (AttributeError, OSError):
                pass

            raw = self.train_fn(cfg)
            elapsed = time.time() - start

            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

            if isinstance(raw, (int, float)):
                raw = {"score": float(raw)}

            score = float(raw.get("score", float("inf")))
            if math.isnan(score) or math.isinf(score):
                score = float("inf")

            return {
                "score": score,
                "accuracy": raw.get("accuracy", raw.get("val_accuracy")),
                "n_params": raw.get("n_params"),
                "train_losses": raw.get("train_losses", []),
                "val_losses": raw.get("val_losses", []),
                "val_accuracies": raw.get("val_accuracies", []),
                "elapsed": elapsed,
                "status": raw.get("status", "ok"),
                "error": raw.get("error", ""),
            }

        except _Timeout:
            return {"score": float("inf"), "elapsed": time.time() - start,
                    "status": "timeout", "error": f">{self.timeout}s",
                    "train_losses": [], "val_losses": [], "val_accuracies": []}
        except Exception as e:
            return {"score": float("inf"), "elapsed": time.time() - start,
                    "status": "error", "error": f"{type(e).__name__}: {e}",
                    "train_losses": [], "val_losses": [], "val_accuracies": []}
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, OSError):
                pass

    # ── Prompt building ────────────────────────────────────────────────────

    def _build_prompt(self, history):
        parts = [self.ml_context.strip(), ""]

        # Accumulated insights from previous iterations
        if self._insights:
            parts.append("## Key Insights (from previous iterations)\n")
            for insight in self._insights:
                parts.append(f"- {insight}")
            parts.append("")

        # Search space
        parts.append("## Config Space\n")
        parts.append("Each config is a JSON object with these keys:")
        for name, dim in self.search_space.items():
            parts.append(_describe_dim(name, dim))
        parts.append("")

        # Best result
        if self.best_config is not None:
            parts.append("## Best Result So Far")
            parts.append(f"Score: {self.best_score:.4f}"
                         + (f", accuracy: {self.best_accuracy:.4f}"
                            if self.best_accuracy else ""))
            parts.append(f"Config: {json.dumps(self.best_config, default=str)}")
            parts.append("")

        # Recent history table
        recent = history[-20:]
        if recent:
            parts.append("## Recent Experiments (last 20)\n")
            # Dynamic columns from config keys + result columns
            cfg_keys = list(self.search_space.keys())
            header = ["#"] + cfg_keys + ["score", "accuracy", "status"]
            parts.append("| " + " | ".join(header) + " |")
            parts.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in recent:
                cfg = row.get("config", {})
                vl = row.get("val_loss")
                va = row.get("val_accuracy")
                vl_s = f"{vl:.4f}" if vl is not None and vl != float("inf") else "inf"
                va_s = f"{va:.4f}" if va is not None else "-"
                vals = [str(row.get("id", ""))]
                for k in cfg_keys:
                    v = cfg.get(k, "")
                    if isinstance(v, float):
                        v = f"{v:.4g}"
                    vals.append(str(v))
                vals.extend([vl_s, va_s, row.get("status", "")])
                parts.append("| " + " | ".join(vals) + " |")
            parts.append("")

            # Per-epoch curves
            ok_with_curves = [r for r in recent
                              if r.get("status") == "ok" and r.get("train_losses")]
            if ok_with_curves:
                parts.append("## Per-Epoch Learning Curves (last 8 successful)\n")
                for row in ok_with_curves[-8:]:
                    cfg = row["config"]
                    label = ", ".join(f"{k}={_fmt(cfg.get(k))}"
                                     for k in list(cfg.keys())[:5])
                    parts.append(f"{label}:")
                    tl = row.get("train_losses", [])
                    vl = row.get("val_losses", [])
                    va = row.get("val_accuracies", [])
                    for e in range(max(len(tl), len(vl), len(va))):
                        line = f"  ep{e+1}:"
                        if e < len(tl): line += f" train={tl[e]:.4f}"
                        if e < len(vl): line += f" val={vl[e]:.4f}"
                        if e < len(va): line += f" acc={va[e]:.4f}"
                        parts.append(line)
                parts.append("")

            # Overfitting/underfitting signals
            ok_rows = [r for r in recent
                       if r.get("status") == "ok" and r.get("val_loss")]
            signals = []
            for row in ok_rows[-8:]:
                tl = row.get("train_losses", [])
                vl = row.get("val_losses", [])
                if len(tl) >= 2 and len(vl) >= 2:
                    gap = vl[-1] - tl[-1]
                    if tl[-1] < tl[0] and vl[-1] > min(vl) and gap > 0.3:
                        signals.append(f"- OVERFITTING: {_short_config(row['config'])} "
                                       f"(gap={gap:.2f})")
                    elif tl[-1] > 1.5:
                        signals.append(f"- UNDERFITTING: {_short_config(row['config'])} "
                                       f"(train={tl[-1]:.3f})")
            if signals:
                parts.append("## Signals\n")
                parts.extend(signals)
                parts.append("")

        # Task — budget-aware exploration/exploitation
        parts.append(f"## Task\n")

        budget_hint = ""
        if self._max_evals and self.total_experiments > 0:
            progress = self.total_experiments / self._max_evals
            remaining = self._max_evals - self.total_experiments
            if progress < 0.4:
                budget_hint = (
                    f"You are {self.total_experiments}/{self._max_evals} evals in "
                    f"(early phase). EXPLORE broadly — try diverse configs to map "
                    f"the search space. You have {remaining} evals left.\n"
                )
            elif progress < 0.75:
                budget_hint = (
                    f"You are {self.total_experiments}/{self._max_evals} evals in "
                    f"(mid phase). Mix exploration with exploitation — refine what "
                    f"works but still try 1 different approach per batch.\n"
                )
            else:
                budget_hint = (
                    f"You are {self.total_experiments}/{self._max_evals} evals in "
                    f"(final phase, {remaining} left). EXPLOIT — focus on fine-tuning "
                    f"the best config. Small, targeted changes to lr, wd, dropout.\n"
                )

        parts.append(
            f"Propose exactly {self.batch_size} configs to try next.\n\n"
            + (budget_hint + "\n" if budget_hint else "")
            + "Rules:\n"
            "- Every config must be DIFFERENT from the history AND from each other\n"
            "- Don't just repeat the best config with tiny tweaks\n\n"
            "Respond with ONLY a JSON array of config objects. "
            "No explanation, no markdown fences."
        )
        return "\n".join(parts)

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_response(self, response):
        """Parse LLM JSON into validated configs. Returns None on any failure."""
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if not match:
            return None
        try:
            configs = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        if not isinstance(configs, list) or len(configs) != self.batch_size:
            return None

        validated = []
        for cfg in configs:
            cleaned = _validate_config(cfg, self.search_space)
            if cleaned is None:
                return None
            validated.append(cleaned)
        return validated

    # ── Signal handling ────────────────────────────────────────────────────

    # ── LLM memory ──────────────────────────────────────────────────────

    def _load_insights(self):
        if os.path.exists(self._insights_path):
            with open(self._insights_path) as f:
                return json.load(f)
        return []

    def _save_insights(self):
        with open(self._insights_path, "w") as f:
            json.dump(self._insights, f, indent=2)

    def _extract_insights(self, history):
        """Ask the LLM to summarize what it learned from experiments that
        are about to scroll out of the prompt window.

        Only called when len(history) crosses a multiple of 20 — meaning
        old experiments are being dropped and their lessons would be lost.
        """
        try:
            # Show the experiments that are about to leave the window
            leaving = history[-40:-20] if len(history) >= 40 else history[:len(history)-20]
            if not leaving:
                return

            prompt = "These experiments are about to leave your visible history:\n\n"
            for row in leaving:
                cfg = row.get("config", {})
                vl = row.get("val_loss")
                vl_s = f"{vl:.4f}" if vl is not None else "inf"
                status = row.get("status", "ok")
                prompt += f"- {_short_config(cfg)} → loss={vl_s} [{status}]\n"

            if self._insights:
                prompt += "\nYou already have these notes:\n"
                for insight in self._insights:
                    prompt += f"- {insight}\n"

            prompt += (
                "\nBefore these experiments disappear from your context, extract "
                "any critical patterns you'd want to remember. Only write down "
                "things you're confident about based on multiple experiments — "
                "not guesses from a single result.\n\n"
                "Rules:\n"
                "- 1-3 short bullet points, or NONE if nothing worth saving\n"
                "- Don't repeat what you already know\n"
                "- Record SPECIFIC results: 'lr=0.05 with gelu scored 0.38, best so far' "
                "not 'gelu works well'\n"
                "- Record what DIDN'T work too: 'dropout > 0.3 hurt all 4 configs that tried it'\n"
                "- Only save patterns you saw in multiple experiments\n"
                "- Respond with ONLY the bullet points (starting with -) or NONE"
            )

            response = self._backend.generate(prompt, max_tokens=256)
            response = response.strip()

            if response.upper() == "NONE" or not response:
                return

            new_insights = []
            for line in response.split("\n"):
                line = line.strip().lstrip("•").lstrip("-").strip()
                if line and len(line) > 5 and len(line) < 200:
                    new_insights.append(line)

            if new_insights:
                existing = set(self._insights)
                added = [i for i in new_insights if i not in existing]
                if added:
                    self._insights.extend(added)
                    # Cap at 20 to keep prompt manageable
                    if len(self._insights) > 20:
                        self._insights = self._insights[-20:]
                    self._save_insights()
                    for insight in added:
                        print(f"  [insight] {insight}")

        except Exception:
            pass  # never crash over insight extraction

    def _setup_signals(self):
        def handler(signum, frame):
            if self._shutdown:
                print("\nForce exit.")
                sys.exit(1)
            print("\nStopping after current experiment...")
            self._shutdown = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

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
            print(f"  LLM: {self.llm_success} ok, {self.llm_fallback} fallback")
        print("=" * 60)


# ── Helpers (module-level) ────────────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


class _JSONLLogger:
    def __init__(self, path):
        self.path = path
        self._counter = 0
        if os.path.exists(path):
            with open(path) as f:
                self._counter = sum(1 for _ in f)

    def log(self, iteration, cfg, result, source):
        self._counter += 1
        entry = {
            "id": self._counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            "source": source,
            "config": cfg,
            "val_loss": result.get("score"),
            "val_accuracy": result.get("accuracy"),
            "n_params": result.get("n_params"),
            "train_losses": result.get("train_losses", []),
            "val_losses": result.get("val_losses", []),
            "val_accuracies": result.get("val_accuracies", []),
            "elapsed": result.get("elapsed"),
            "status": result.get("status"),
            "error": result.get("error", ""),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()

    def load_history(self):
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
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

    # Tuple of (low, high) → infer numeric type
    if isinstance(value, tuple) and len(value) == 2:
        lo, hi = value

        # Both ints and name suggests integer → IntUniform
        if isinstance(lo, int) and isinstance(hi, int):
            if name.lower() in _INT_NAMES or name.startswith("n_") or name.startswith("num_"):
                return IntUniform(lo, hi)

        lo, hi = float(lo), float(hi)

        # Name suggests log scale, or ratio > 100 with small values
        name_lower = name.lower()
        if name_lower in _LOG_SCALE_NAMES:
            if lo > 0:
                return LogUniform(lo, hi)

        # Large ratio with small min → probably log scale
        if lo > 0 and hi / lo > 100:
            return LogUniform(lo, hi)

        # Both ints → IntUniform
        if isinstance(value[0], int) and isinstance(value[1], int):
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
            if val not in dim.choices:
                return None
            cleaned[name] = val
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
        if isinstance(v, float):
            v = round(v, 6)
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
