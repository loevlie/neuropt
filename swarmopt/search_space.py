"""Search space dimension classes for PSO hyperparameter tuning."""

import math


class Uniform:
    """Uniform distribution over [low, high]."""

    def __init__(self, low, high):
        self.low = float(low)
        self.high = float(high)

    def to_internal(self, value):
        return float(value)

    def from_internal(self, internal):
        return float(internal)

    def bounds(self):
        return (self.low, self.high)

    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class LogUniform:
    """Log-uniform distribution: internal space is [log10(low), log10(high)]."""

    def __init__(self, low, high):
        self.low = float(low)
        self.high = float(high)

    def to_internal(self, value):
        return math.log10(value)

    def from_internal(self, internal):
        return 10 ** internal

    def bounds(self):
        return (math.log10(self.low), math.log10(self.high))

    def __repr__(self):
        return f"LogUniform({self.low}, {self.high})"


class IntUniform:
    """Integer uniform distribution over [low, high]."""

    def __init__(self, low, high):
        self.low = int(low)
        self.high = int(high)

    def to_internal(self, value):
        return float(value)

    def from_internal(self, internal):
        return int(round(internal))

    def bounds(self):
        return (float(self.low), float(self.high))

    def __repr__(self):
        return f"IntUniform({self.low}, {self.high})"


class Categorical:
    """Categorical dimension: maps choices to integer indices in internal space."""

    def __init__(self, choices):
        self.choices = list(choices)

    def to_internal(self, value):
        return float(self.choices.index(value))

    def from_internal(self, internal):
        idx = int(round(internal))
        idx = max(0, min(idx, len(self.choices) - 1))
        return self.choices[idx]

    def bounds(self):
        return (0.0, float(len(self.choices) - 1))

    def __repr__(self):
        return f"Categorical({self.choices})"
