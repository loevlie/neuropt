"""LLM Advisor: builds prompts from experiment history and parses JSON configs."""

import json
import re

from swarmopt.search_space import Categorical, LogUniform


class LLMAdvisor:
    """Uses an LLM backend to refine PSO-suggested hyperparameter configs.

    On any failure (bad JSON, timeout, crash), silently falls back to
    the raw PSO suggestions.
    """

    def __init__(self, backend, search_space: dict):
        """
        Args:
            backend: An LLMBackend instance (has .generate()).
            search_space: Dict mapping param names to dimension objects.
        """
        self.backend = backend
        self.search_space = search_space
        self.fallback_count = 0
        self.success_count = 0

    def advise(self, pso_configs: list[dict], history: list[dict],
               best_result: dict | None = None) -> tuple[list[dict], str]:
        """Ask the LLM to refine PSO suggestions based on experiment history.

        Args:
            pso_configs: List of param dicts from PSO.
            history: List of experiment result dicts from LogStore.
            best_result: Dict with best params and score so far.

        Returns:
            Tuple of (configs, source) where source is 'llm' or 'pso'.
        """
        try:
            prompt = self._build_prompt(pso_configs, history, best_result)
            response = self.backend.generate(prompt, max_tokens=1024)
            configs = self._parse_response(response, len(pso_configs))

            if configs is not None:
                self.success_count += 1
                return configs, "llm"

        except Exception:
            pass

        self.fallback_count += 1
        return pso_configs, "pso"

    def _build_prompt(self, pso_configs, history, best_result):
        parts = []
        parts.append("You are a hyperparameter optimization advisor. "
                      "Analyze the experiment history and suggest better configs.\n")

        # Search space description
        parts.append("## Search Space\n")
        for name, dim in self.search_space.items():
            if isinstance(dim, Categorical):
                parts.append(f"- {name}: categorical, choices={dim.choices}")
            elif isinstance(dim, LogUniform):
                parts.append(f"- {name}: log-uniform [{dim.low}, {dim.high}]")
            else:
                lo, hi = dim.bounds()
                parts.append(f"- {name}: [{lo}, {hi}]")
        parts.append("")

        # Best result
        if best_result:
            parts.append(f"## Best Result So Far")
            parts.append(f"Score (lower is better): {best_result.get('score', '?')}")
            if best_result.get('params'):
                parts.append(f"Params: {json.dumps(best_result['params'], default=str)}")
            parts.append("")

        # Recent history as markdown table
        recent = history[-20:] if len(history) > 20 else history
        if recent:
            parts.append("## Recent Experiments (last 20)\n")
            cols = ["lr", "wd", "val_loss", "val_accuracy", "status", "source"]
            available_cols = [c for c in cols if any(r.get(c) for r in recent)]
            if available_cols:
                parts.append("| " + " | ".join(available_cols) + " |")
                parts.append("| " + " | ".join(["---"] * len(available_cols)) + " |")
                for row in recent:
                    vals = [str(row.get(c, "")) for c in available_cols]
                    parts.append("| " + " | ".join(vals) + " |")
                parts.append("")

            # Per-epoch learning curves for recent experiments
            curves = self._format_learning_curves(recent)
            if curves:
                parts.append("## Per-Epoch Learning Curves\n")
                parts.append("Each row shows train_loss / val_loss / val_acc per epoch. "
                             "Look for overfitting (train_loss drops but val_loss rises), "
                             "underfitting (both stay high), or divergence (losses explode).\n")
                parts.append(curves)
                parts.append("")

            # Pre-computed trends
            parts.append("## Trends\n")
            ok_rows = [r for r in recent if r.get("status") == "ok" and r.get("val_loss")]
            if len(ok_rows) >= 3:
                losses = []
                for r in ok_rows:
                    try:
                        losses.append(float(r["val_loss"]))
                    except (ValueError, TypeError):
                        pass
                if losses:
                    parts.append(f"- Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
                    if len(losses) >= 4:
                        first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
                        second_half = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
                        trend = "improving" if second_half < first_half else "stagnating"
                        parts.append(f"- Trend: {trend} (first half avg={first_half:.4f}, second half avg={second_half:.4f})")

                # LR analysis
                lrs = []
                for r in ok_rows:
                    try:
                        lrs.append((float(r["lr"]), float(r["val_loss"])))
                    except (ValueError, TypeError):
                        pass
                if lrs:
                    best_lr_row = min(lrs, key=lambda x: x[1])
                    parts.append(f"- Best lr seen: {best_lr_row[0]:.6f} (loss={best_lr_row[1]:.4f})")

                # Overfitting analysis
                overfit_analysis = self._analyze_overfitting(ok_rows)
                if overfit_analysis:
                    parts.append(overfit_analysis)
            parts.append("")

        # PSO suggestions
        parts.append("## PSO Suggested Configs (use as starting points)\n")
        parts.append("```json")
        # Strip non-serializable keys for the prompt
        clean_configs = []
        for cfg in pso_configs:
            clean = {k: v for k, v in cfg.items() if k != "device"}
            clean_configs.append(clean)
        parts.append(json.dumps(clean_configs, indent=2, default=str))
        parts.append("```\n")

        parts.append("## Instructions\n")
        parts.append(
            "Based on the history and trends, suggest improved configs. "
            "You may modify the PSO suggestions or propose entirely new values, "
            "but stay within the search space bounds. "
            f"Return exactly {len(pso_configs)} configs.\n"
            "Respond with ONLY a JSON array of config objects. No explanation, no markdown fences."
        )

        return "\n".join(parts)

    def _format_learning_curves(self, history: list[dict]) -> str:
        """Format per-epoch curves as compact text for the LLM prompt.

        Shows the last 10 experiments that have curve data.
        """
        rows_with_curves = []
        for row in history:
            tl = row.get("train_losses", [])
            vl = row.get("val_losses", [])
            va = row.get("val_accuracies", [])
            # Also handle JSON strings from LogStore reload
            if isinstance(tl, str):
                try:
                    tl = json.loads(tl) if tl else []
                except (json.JSONDecodeError, TypeError):
                    tl = []
            if isinstance(vl, str):
                try:
                    vl = json.loads(vl) if vl else []
                except (json.JSONDecodeError, TypeError):
                    vl = []
            if isinstance(va, str):
                try:
                    va = json.loads(va) if va else []
                except (json.JSONDecodeError, TypeError):
                    va = []
            if tl or vl:
                rows_with_curves.append((row, tl, vl, va))

        if not rows_with_curves:
            return ""

        # Show last 10 with curves
        recent_curves = rows_with_curves[-10:]
        lines = []
        for row, tl, vl, va in recent_curves:
            lr_str = row.get("lr", "?")
            try:
                lr_str = f"{float(lr_str):.4e}"
            except (ValueError, TypeError):
                pass
            wd_str = row.get("wd", "?")
            try:
                wd_str = f"{float(wd_str):.4e}"
            except (ValueError, TypeError):
                pass

            lines.append(f"lr={lr_str}, wd={wd_str}:")
            n_epochs = max(len(tl), len(vl), len(va))
            for e in range(n_epochs):
                parts = [f"  ep{e+1}:"]
                if e < len(tl):
                    parts.append(f"train_loss={tl[e]:.4f}")
                if e < len(vl):
                    parts.append(f"val_loss={vl[e]:.4f}")
                if e < len(va):
                    parts.append(f"val_acc={va[e]:.4f}")
                lines.append(" ".join(parts))

        return "\n".join(lines)

    def _analyze_overfitting(self, ok_rows: list[dict]) -> str:
        """Pre-compute overfitting signals for the LLM."""
        signals = []
        for row in ok_rows[-10:]:
            tl = row.get("train_losses", [])
            vl = row.get("val_losses", [])
            if isinstance(tl, str):
                try:
                    tl = json.loads(tl) if tl else []
                except (json.JSONDecodeError, TypeError):
                    tl = []
            if isinstance(vl, str):
                try:
                    vl = json.loads(vl) if vl else []
                except (json.JSONDecodeError, TypeError):
                    vl = []

            if len(tl) >= 2 and len(vl) >= 2:
                train_dropping = tl[-1] < tl[0]
                val_rising = vl[-1] > min(vl)
                gap = vl[-1] - tl[-1] if tl[-1] and vl[-1] else 0

                lr_str = row.get("lr", "?")
                try:
                    lr_str = f"{float(lr_str):.4e}"
                except (ValueError, TypeError):
                    pass

                if train_dropping and val_rising and gap > 0.3:
                    signals.append(f"  lr={lr_str}: OVERFITTING "
                                   f"(train {tl[0]:.3f}->{tl[-1]:.3f}, "
                                   f"val {vl[0]:.3f}->{vl[-1]:.3f}, "
                                   f"gap={gap:.3f})")
                elif not train_dropping and tl[-1] > 1.5:
                    signals.append(f"  lr={lr_str}: UNDERFITTING "
                                   f"(train still at {tl[-1]:.3f})")

        if signals:
            return "- Overfitting/underfitting signals:\n" + "\n".join(signals)
        return ""

    def _parse_response(self, response: str, expected_count: int) -> list[dict] | None:
        """Parse LLM response into validated config dicts.

        Returns None on any parse failure.
        """
        # Find JSON array in response
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if not match:
            return None

        try:
            configs = json.loads(match.group())
        except json.JSONDecodeError:
            return None

        if not isinstance(configs, list) or len(configs) != expected_count:
            return None

        # Validate and clamp each config
        validated = []
        param_names = set(self.search_space.keys())
        for cfg in configs:
            if not isinstance(cfg, dict):
                return None

            cleaned = {}
            for name, dim in self.search_space.items():
                if name not in cfg:
                    return None  # Missing key
                val = cfg[name]

                if isinstance(dim, Categorical):
                    if val not in dim.choices:
                        return None
                    cleaned[name] = val
                else:
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        return None
                    lo, hi = dim.bounds()
                    internal = dim.to_internal(val)
                    internal = max(lo, min(internal, hi))
                    cleaned[name] = dim.from_internal(internal)

            validated.append(cleaned)

        return validated
