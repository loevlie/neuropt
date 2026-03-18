"""SwarmTuner — PSO hyperparameter tuner with multiprocessing parallelism."""

import time
import warnings
import numpy as np

from swarmopt.pso import PSOEngine
from swarmopt._results import ResultStore
from swarmopt import visualize


class SwarmTuneError(Exception):
    """Raised when the tuner cannot produce any valid result."""


class SwarmTuner:
    """Particle Swarm Optimization hyperparameter tuner.

    Args:
        train_fn: callable that takes a dict of params and returns either
            a float (the score) or a dict with at least a "score" key.
        search_space: dict mapping param names to dimension objects
            (LogUniform, Uniform, IntUniform).
        n_particles: number of particles in the swarm.
        n_iterations: number of PSO iterations.
        n_workers: number of parallel workers (defaults to n_particles).
        w: PSO inertia weight.
        c1: PSO cognitive coefficient.
        c2: PSO social coefficient.
        seed: random seed.
        minimize: if True (default), lower scores are better.
        device: if set, injected into params dict as params["device"].
        verbose: if True, print progress.
    """

    def __init__(self, train_fn, search_space, n_particles=10, n_iterations=20,
                 n_workers=None, w=0.7, c1=1.5, c2=1.5, seed=42,
                 minimize=True, device=None, verbose=True):
        self.train_fn = train_fn
        self.search_space = dict(search_space)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.n_workers = n_workers if n_workers is not None else n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.minimize = minimize
        self.device = device
        self.verbose = verbose

        self._best_params = None
        self._best_score = np.inf
        self._best_model = None
        self._results_store = None
        self._results_df = None

    def _positions_to_params(self, positions):
        """Convert internal-space positions to list of user-space param dicts."""
        param_names = list(self.search_space.keys())
        dims = list(self.search_space.values())
        all_params = []
        for pos in positions:
            d = {}
            for i, (name, dim) in enumerate(zip(param_names, dims)):
                d[name] = dim.from_internal(pos[i])
            if self.device is not None:
                d["device"] = self.device
            all_params.append(d)
        return all_params

    def _evaluate_parallel(self, params_list, fn_bytes, pool):
        """Evaluate all particles in parallel using a pre-created pool."""
        from swarmopt._worker import worker_fn

        args = [(fn_bytes, p, i) for i, p in enumerate(params_list)]
        results = pool.starmap(worker_fn, args)
        return results

    def _evaluate_sequential(self, params_list):
        """Evaluate all particles sequentially (fallback)."""
        import math
        import traceback

        results = []
        for i, params in enumerate(params_list):
            try:
                raw = self.train_fn(params)
                if isinstance(raw, dict):
                    result = dict(raw)
                    if "score" not in result:
                        raise ValueError(
                            "train_fn returned a dict without a 'score' key")
                else:
                    result = {"score": float(raw)}
                if math.isnan(result["score"]):
                    result["score"] = float("inf")
                    result["error"] = "NaN score replaced with inf"
            except Exception:
                result = {"score": float("inf"), "error": traceback.format_exc()}
            results.append(result)
        return results

    def fit(self):
        """Run the PSO optimization loop.

        Returns:
            self (for chaining).
        """
        param_names = list(self.search_space.keys())
        dims = list(self.search_space.values())
        bounds = [d.bounds() for d in dims]
        n_dims = len(bounds)

        engine = PSOEngine(
            n_dims=n_dims, n_particles=self.n_particles, bounds=bounds,
            w=self.w, c1=self.c1, c2=self.c2, seed=self.seed,
        )
        store = ResultStore(param_names)
        self._results_store = store

        # Try to serialize train_fn for multiprocessing
        use_parallel = True
        fn_bytes = None
        try:
            import cloudpickle
            fn_bytes = cloudpickle.dumps(self.train_fn)
        except Exception as e:
            warnings.warn(
                f"Could not serialize train_fn with cloudpickle ({e}). "
                "Falling back to sequential evaluation.",
                stacklevel=2,
            )
            use_parallel = False

        positions = engine.initialize()
        total_evals = self.n_particles * (self.n_iterations + 1)
        eval_count = 0
        any_success = False

        if self.verbose:
            print(f"SwarmTuner: {self.n_particles} particles x "
                  f"{self.n_iterations} iterations "
                  f"({'parallel' if use_parallel else 'sequential'})")

        # Create pool once — spawning processes is expensive (torch imports, etc.)
        pool = None
        if use_parallel:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(self.n_workers)

        try:
            for iteration in range(self.n_iterations + 1):
                params_list = self._positions_to_params(positions)
                t0 = time.time()

                if use_parallel:
                    results = self._evaluate_parallel(params_list, fn_bytes,
                                                      pool)
                else:
                    results = self._evaluate_sequential(params_list)

                elapsed_iter = time.time() - t0
                costs = []

                for j, result in enumerate(results):
                    score = result["score"]
                    if not self.minimize:
                        cost = -score
                    else:
                        cost = score
                    costs.append(cost)

                    status = "error" if "error" in result else "ok"
                    if status == "ok":
                        any_success = True

                    # Strip device from params before storing
                    store_params = {k: params_list[j][k] for k in param_names}
                    store.add(iteration, j, store_params, score,
                              elapsed_iter / len(results), status)

                    if status == "error" and self.verbose:
                        warnings.warn(
                            f"Particle {j} iter {iteration}: "
                            f"{result['error'][:200]}",
                            stacklevel=2,
                        )

                    # Track best
                    if score < self._best_score:
                        self._best_score = score
                        self._best_params = store_params.copy()
                        self._best_model = result.get("model", None)

                eval_count += len(results)

                if self.verbose:
                    best_display = self._best_score
                    param_str = "  ".join(
                        f"{k}={self._best_params[k]:.2e}"
                        for k in param_names
                    ) if self._best_params else ""
                    label = "Init " if iteration == 0 else f"Iter {iteration:>2}"
                    print(f"  {label} | best={best_display:.4f}  {param_str}  "
                          f"({elapsed_iter:.1f}s)")

                if iteration < self.n_iterations:
                    positions = engine.step(np.array(costs))
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        if not any_success:
            raise SwarmTuneError(
                "All evaluations failed. Check your train_fn and search_space.")

        self._results_df = store.to_dataframe()

        if self.verbose:
            print(f"\nBest score: {self._best_score:.4f}")
            for k, v in self._best_params.items():
                print(f"  {k} = {v}")

        return self

    @property
    def best_params(self):
        return self._best_params

    @property
    def best_score(self):
        return self._best_score

    @property
    def best_model(self):
        return self._best_model

    @property
    def results(self):
        if self._results_df is None and self._results_store is not None:
            self._results_df = self._results_store.to_dataframe()
        return self._results_df

    def plot(self, save_path="pso_results.png"):
        """Generate a 3-panel static plot of the results."""
        return visualize.plot(self.results, self.search_space, save_path)

    def animate(self, save_path="pso_trajectories.gif", fps=2):
        """Generate an animated GIF of particle trajectories."""
        visualize.animate(self.results, self.search_space, save_path, fps)
