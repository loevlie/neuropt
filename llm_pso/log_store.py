"""Append-only TSV logger for experiment results."""

import json
import os
import time


COLUMNS = [
    "experiment_id", "timestamp", "iteration", "particle",
    "lr", "wd", "momentum", "architecture", "optimizer",
    "val_loss", "val_accuracy", "train_loss_final",
    "train_losses_json", "val_losses_json", "val_accuracies_json",
    "elapsed_seconds", "status", "error", "source",
]


class LogStore:
    """Append-only TSV logger. Survives crashes via flush-after-write."""

    def __init__(self, path: str):
        self.path = path
        self._counter = 0
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            with open(self.path, "w") as f:
                f.write("\t".join(COLUMNS) + "\n")
                f.flush()
        else:
            # Count existing rows to continue experiment_id
            with open(self.path, "r") as f:
                self._counter = max(0, sum(1 for _ in f) - 1)

    def log(self, iteration: int, particle: int, params: dict,
            result: dict, source: str = "pso"):
        """Append one experiment row.

        Args:
            iteration: PSO iteration number.
            particle: Particle index within the iteration.
            params: Dict of hyperparameters used.
            result: Dict with keys like score, val_accuracy, train_loss_final,
                    train_losses, elapsed, status, error.
            source: One of 'pso', 'llm', 'llm_modified'.
        """
        self._counter += 1

        train_losses = result.get("train_losses", [])
        train_losses_json = json.dumps(train_losses) if train_losses else ""
        val_losses = result.get("val_losses", [])
        val_losses_json = json.dumps(val_losses) if val_losses else ""
        val_accuracies = result.get("val_accuracies", [])
        val_accuracies_json = json.dumps(val_accuracies) if val_accuracies else ""

        row = {
            "experiment_id": self._counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "iteration": iteration,
            "particle": particle,
            "lr": params.get("lr", ""),
            "wd": params.get("wd", ""),
            "momentum": params.get("momentum", ""),
            "architecture": params.get("architecture", ""),
            "optimizer": params.get("optimizer_name", ""),
            "val_loss": result.get("score", ""),
            "val_accuracy": result.get("val_accuracy", ""),
            "train_loss_final": result.get("train_loss_final", ""),
            "train_losses_json": train_losses_json,
            "val_losses_json": val_losses_json,
            "val_accuracies_json": val_accuracies_json,
            "elapsed_seconds": f"{result.get('elapsed', 0):.1f}",
            "status": result.get("status", "ok"),
            "error": result.get("error", ""),
            "source": source,
        }

        line = "\t".join(str(row.get(c, "")) for c in COLUMNS)
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def load_history(self) -> list[dict]:
        """Load all rows as a list of dicts."""
        if not os.path.exists(self.path):
            return []
        rows = []
        with open(self.path, "r") as f:
            header = f.readline().strip().split("\t")
            for line in f:
                vals = line.strip().split("\t")
                row = dict(zip(header, vals))
                rows.append(row)
        return rows
