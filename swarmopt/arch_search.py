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

from swarmopt.search_space import Categorical, IntUniform, LogUniform, Uniform


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

    Define a training function that returns per-epoch metrics, a search space
    using swarmopt dimension types, and let the LLM figure out what works.

    Example::

        from swarmopt import ArchSearch, LogUniform, Categorical

        search = ArchSearch(
            train_fn=my_train_fn,
            search_space={
                "lr": LogUniform(1e-4, 1e-1),
                "activation": Categorical(["relu", "gelu", "silu"]),
            },
            backend="claude",
        )
        search.run()
    """

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
            search_space: Dict mapping param names to dimension objects
                (LogUniform, Uniform, IntUniform, Categorical).
            backend: LLM backend — "auto", "claude", "openai", "qwen", "none",
                or a backend instance.
            log_path: JSONL log file path. Supports resume.
            batch_size: Configs proposed per LLM call.
            device: Injected into config as ``config["device"]`` if set.
            timeout: Max seconds per experiment.
            ml_context: Domain knowledge for the LLM. Overrides the default
                generic prompt. Use this to tell the LLM about your task,
                model architecture constraints, or dataset properties.
        """
        self.train_fn = train_fn
        self.search_space = search_space
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

    def run(self):
        """Run the search loop until Ctrl+C or SIGTERM."""
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
        if self.total_experiments:
            print(f"  Resuming from {self.total_experiments} experiments")
            print(f"  Best so far: loss={self.best_score:.4f} acc={self.best_accuracy:.4f}")
        print("  Ctrl+C to stop")
        print("=" * 60)
        print()

        rng = random.Random(42 + self.total_experiments)

        while not self._shutdown:
            iter_start = time.time()

            configs, source = self._get_configs(history, rng)

            for idx, cfg in enumerate(configs):
                if self._shutdown:
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

            print(f"  iter {iteration} done in {time.time() - iter_start:.1f}s | "
                  f"best: {self.best_score:.4f} | total: {self.total_experiments}")
            if self._backend:
                print(f"  llm: {self.llm_success} ok, {self.llm_fallback} fallback")
            print()
            iteration += 1

        self._print_summary(iteration)

    # ── Config generation ──────────────────────────────────────────────────

    def _get_configs(self, history, rng):
        """Ask LLM for configs, fall back to random."""
        if self._backend is not None:
            try:
                prompt = self._build_prompt(history)
                response = self._backend.generate(prompt, max_tokens=2048)
                configs = self._parse_response(response)
                if configs is not None:
                    self.llm_success += 1
                    return configs, "llm"
                else:
                    self.llm_fallback += 1
                    print(f"  [LLM parse failed, using random]")
            except Exception as e:
                self.llm_fallback += 1
                print(f"  [LLM error: {e}, using random]")

        return [_random_config(self.search_space, rng)
                for _ in range(self.batch_size)], "random"

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

        # Task
        parts.append(f"## Task\n")
        parts.append(
            f"Propose exactly {self.batch_size} configs to try next. "
            "Balance exploration with exploitation.\n"
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


def _resolve_backend(backend):
    if backend is None or backend == "none":
        return None
    if backend == "auto":
        from llm_pso.backends import get_default_backend
        return get_default_backend()
    if isinstance(backend, str):
        from llm_pso.backends import get_backend_by_name
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


def _short_config(cfg):
    """One-line summary of a config dict."""
    parts = []
    for k, v in cfg.items():
        if k == "device":
            continue
        parts.append(f"{k}={_fmt(v)}")
    return " ".join(parts[:6])  # cap at 6 for readability
