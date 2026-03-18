"""Autonomous PSO + LLM advisor runner: the never-stop loop."""

import signal
import sys
import time

import numpy as np

from swarmopt.pso import PSOEngine
from llm_pso.advisor import LLMAdvisor
from llm_pso.experiment import ExperimentRunner
from llm_pso.log_store import LogStore


class AutonomousRunner:
    """Runs PSO + LLM advisor in an autonomous loop until stopped.

    PSO proposes positions -> LLM refines -> train -> log -> feed costs -> repeat.
    """

    def __init__(
        self,
        train_fn,
        search_space: dict,
        log_path: str = "experiments.tsv",
        backend=None,
        n_particles: int = 3,
        timeout: int = 600,
        device: str | None = None,
        seed: int = 42,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        """
        Args:
            train_fn: Callable(params) -> dict with 'score' key.
            search_space: Dict mapping param names to dimension objects.
            log_path: Path for TSV log file.
            backend: LLM backend instance or None for pure PSO.
            n_particles: Number of particles (configs per iteration).
            timeout: Max seconds per experiment.
            device: Device string to inject into params.
            seed: Random seed for PSO.
        """
        self.search_space = search_space
        self.device = device
        self.n_particles = n_particles
        self._shutdown = False

        # Ordered param names and bounds
        self._param_names = list(search_space.keys())
        bounds = [search_space[name].bounds() for name in self._param_names]

        self.pso = PSOEngine(
            n_dims=len(self._param_names),
            n_particles=n_particles,
            bounds=bounds,
            w=w, c1=c1, c2=c2,
            seed=seed,
        )

        self.experiment = ExperimentRunner(train_fn, timeout=timeout)
        self.log_store = LogStore(log_path)

        if backend is not None:
            self.advisor = LLMAdvisor(backend, search_space)
        else:
            self.advisor = None

        # Track global best
        self._best_score = float("inf")
        self._best_params = None

    def _positions_to_configs(self, positions: np.ndarray) -> list[dict]:
        """Convert PSO internal positions to param dicts."""
        configs = []
        for i in range(positions.shape[0]):
            cfg = {}
            for j, name in enumerate(self._param_names):
                dim = self.search_space[name]
                cfg[name] = dim.from_internal(positions[i, j])
            if self.device is not None:
                cfg["device"] = self.device
            configs.append(cfg)
        return configs

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            if self._shutdown:
                print("\nForce exit.")
                sys.exit(1)
            print("\nShutdown requested. Finishing current experiment...")
            self._shutdown = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def run(self):
        """Run the autonomous loop until Ctrl+C or SIGTERM."""
        self._setup_signal_handlers()

        print("=" * 60)
        print("Autonomous PSO + LLM Research Loop")
        print(f"  Particles: {self.n_particles}")
        print(f"  LLM advisor: {self.advisor.backend.name if self.advisor else 'None (pure PSO)'}")
        print(f"  Log: {self.log_store.path}")
        print(f"  Device: {self.device or 'auto'}")
        print("  Press Ctrl+C to stop gracefully")
        print("=" * 60)

        positions = self.pso.initialize()
        history = self.log_store.load_history()
        total_experiments = 0
        iteration = 0

        while not self._shutdown:
            iter_start = time.time()
            pso_configs = self._positions_to_configs(positions)

            # Ask LLM advisor if available and enough history
            source = "pso"
            if (self.advisor is not None
                    and len(history) >= self.n_particles):
                best_info = None
                if self._best_params is not None:
                    best_info = {"score": self._best_score,
                                 "params": self._best_params}
                configs, source = self.advisor.advise(
                    pso_configs, history, best_info)
            else:
                configs = pso_configs

            # Run experiments sequentially
            costs = []
            for p_idx, cfg in enumerate(configs):
                if self._shutdown:
                    break

                result = self.experiment.run(cfg)
                costs.append(result["score"])

                # Update global best
                if result["score"] < self._best_score:
                    self._best_score = result["score"]
                    self._best_params = {k: v for k, v in cfg.items()
                                         if k != "device"}

                # Log
                self.log_store.log(iteration, p_idx, cfg, result, source)
                history.append({
                    "lr": str(cfg.get("lr", "")),
                    "wd": str(cfg.get("wd", "")),
                    "val_loss": str(result["score"]) if result["score"] != float("inf") else "",
                    "val_accuracy": str(result.get("val_accuracy", "")),
                    "train_losses": result.get("train_losses", []),
                    "val_losses": result.get("val_losses", []),
                    "val_accuracies": result.get("val_accuracies", []),
                    "status": result["status"],
                    "source": source,
                })
                total_experiments += 1

                # Per-experiment status
                acc_str = f", acc={result['val_accuracy']:.4f}" if result.get("val_accuracy") else ""
                status_str = f" [{result['status']}]" if result["status"] != "ok" else ""
                print(f"  [{iteration}:{p_idx}] loss={result['score']:.4f}{acc_str}"
                      f"  ({result['elapsed']:.1f}s) {source}{status_str}")

            if self._shutdown:
                break

            # Pad costs if we exited early
            while len(costs) < self.n_particles:
                costs.append(float("inf"))

            # Feed costs to PSO and get new positions
            positions = self.pso.step(np.array(costs))

            iter_elapsed = time.time() - iter_start
            print(f"  Iter {iteration} done in {iter_elapsed:.1f}s | "
                  f"Best: {self._best_score:.4f} | "
                  f"Total experiments: {total_experiments}")

            if self.advisor:
                fb = self.advisor.fallback_count
                sc = self.advisor.success_count
                print(f"  LLM: {sc} advised, {fb} fallbacks")

            print()
            iteration += 1

        # Final summary
        print("\n" + "=" * 60)
        print("SHUTDOWN SUMMARY")
        print(f"  Total iterations: {iteration}")
        print(f"  Total experiments: {total_experiments}")
        print(f"  Best score: {self._best_score:.4f}")
        if self._best_params:
            print(f"  Best params: {self._best_params}")
        if self.advisor:
            print(f"  LLM advised: {self.advisor.success_count}, "
                  f"fallbacks: {self.advisor.fallback_count}")
        print("=" * 60)
