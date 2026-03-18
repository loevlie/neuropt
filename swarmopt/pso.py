"""Pure-numpy PSO engine with a stateful step() interface."""

import numpy as np


class PSOEngine:
    """Particle Swarm Optimization engine.

    Does not call the objective function — it receives costs and returns
    new positions. The caller drives the evaluation loop.
    """

    def __init__(self, n_dims, n_particles, bounds, w=0.7, c1=1.5, c2=1.5,
                 seed=42):
        self.n_dims = n_dims
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self._rng = np.random.default_rng(seed)

        self._lo = np.array([b[0] for b in bounds])
        self._hi = np.array([b[1] for b in bounds])

        self._x = None
        self._v = None
        self._pbest = None
        self._pbest_cost = None
        self._gbest = None
        self._gbest_cost = np.inf
        self._iteration = 0

    def initialize(self):
        """Initialize particles and return positions (n_particles, n_dims)."""
        lo, hi = self._lo, self._hi
        self._x = self._rng.uniform(lo, hi, (self.n_particles, self.n_dims))
        self._v = self._rng.uniform(-(hi - lo), hi - lo,
                                     (self.n_particles, self.n_dims))
        self._pbest = self._x.copy()
        self._pbest_cost = np.full(self.n_particles, np.inf)
        self._iteration = 0
        return self._x.copy()

    def step(self, costs):
        """Update pbest/gbest from costs, advance velocities, return new positions.

        Args:
            costs: array-like of shape (n_particles,) — objective values from
                   the most recent evaluation.

        Returns:
            New positions array of shape (n_particles, n_dims).
        """
        costs = np.asarray(costs, dtype=float)

        # Update personal bests
        improved = costs < self._pbest_cost
        self._pbest[improved] = self._x[improved].copy()
        self._pbest_cost[improved] = costs[improved]

        # Update global best
        best_idx = self._pbest_cost.argmin()
        self._gbest = self._pbest[best_idx].copy()
        self._gbest_cost = self._pbest_cost[best_idx]

        # Velocity & position update
        r1, r2 = self._rng.random((2, self.n_particles, self.n_dims))
        self._v = (self.w * self._v
                    + self.c1 * r1 * (self._pbest - self._x)
                    + self.c2 * r2 * (self._gbest - self._x))
        self._x = np.clip(self._x + self._v, self._lo, self._hi)

        self._iteration += 1
        return self._x.copy()

    @property
    def gbest(self):
        return self._gbest.copy() if self._gbest is not None else None

    @property
    def gbest_cost(self):
        return self._gbest_cost

    @property
    def iteration(self):
        return self._iteration
