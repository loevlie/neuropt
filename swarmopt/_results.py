"""Result accumulator for SwarmTuner."""

import numpy as np


class ResultStore:
    """Accumulates per-particle results and builds a DataFrame."""

    def __init__(self, param_names):
        self._param_names = list(param_names)
        self._rows = []

    def add(self, iteration, particle, params_dict, score, elapsed, status="ok"):
        row = {
            "iteration": iteration,
            "particle": particle,
            **{k: params_dict[k] for k in self._param_names},
            "score": score,
            "elapsed": elapsed,
            "status": status,
        }
        self._rows.append(row)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self._rows)

    def get_best(self):
        """Return (params_dict, score, row_index) for the best result."""
        if not self._rows:
            return None, np.inf, -1
        best_idx = min(range(len(self._rows)), key=lambda i: self._rows[i]["score"])
        best = self._rows[best_idx]
        params = {k: best[k] for k in self._param_names}
        return params, best["score"], best_idx
