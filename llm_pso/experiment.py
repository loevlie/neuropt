"""Experiment runner with timeout and rich metrics."""

import math
import signal
import time
import traceback


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Experiment timed out")


class ExperimentRunner:
    """Runs train_fn(config) with timeout enforcement and error handling.

    Returns a rich result dict regardless of success or failure.
    """

    def __init__(self, train_fn, timeout: int = 600):
        """
        Args:
            train_fn: Callable(params_dict) -> dict with at least 'score' key.
            timeout: Max seconds per experiment (Unix only, uses signal.alarm).
        """
        self.train_fn = train_fn
        self.timeout = timeout

    def run(self, params: dict) -> dict:
        """Run a single experiment.

        Returns:
            Dict with keys: score, val_accuracy, train_loss_final,
            train_losses, elapsed, status, error.
        """
        start = time.time()
        old_handler = None

        try:
            # Set timeout (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self.timeout)
            except (AttributeError, OSError):
                pass  # Windows or signal not available

            raw = self.train_fn(params)
            elapsed = time.time() - start

            # Cancel alarm
            try:
                signal.alarm(0)
            except (AttributeError, OSError):
                pass

            # Normalize return value
            if isinstance(raw, (int, float)):
                raw = {"score": float(raw)}

            score = float(raw.get("score", float("inf")))
            if math.isnan(score) or math.isinf(score):
                score = float("inf")

            return {
                "score": score,
                "val_accuracy": raw.get("accuracy", raw.get("val_accuracy", None)),
                "train_loss_final": raw.get("train_loss_final", None),
                "train_losses": raw.get("train_losses", []),
                "val_losses": raw.get("val_losses", []),
                "val_accuracies": raw.get("val_accuracies", []),
                "elapsed": elapsed,
                "status": "ok",
                "error": "",
            }

        except _Timeout:
            elapsed = time.time() - start
            return {
                "score": float("inf"),
                "val_accuracy": None,
                "train_loss_final": None,
                "train_losses": [],
                "elapsed": elapsed,
                "status": "timeout",
                "error": f"Exceeded {self.timeout}s timeout",
            }

        except Exception as e:
            elapsed = time.time() - start
            tb = traceback.format_exc()
            return {
                "score": float("inf"),
                "val_accuracy": None,
                "train_loss_final": None,
                "train_losses": [],
                "elapsed": elapsed,
                "status": "error",
                "error": f"{type(e).__name__}: {e}\n{tb}",
            }

        finally:
            # Restore signal handler
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, OSError):
                pass
