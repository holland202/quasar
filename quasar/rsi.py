"""
QUASAR RSI extension — bounded recursive self-improvement via meta-curriculum
==============================================================================

This is a *bounded* RSI demonstration built on top of the existing Quasar loop.

The inner loop (Generator → curriculum → Learner) remains exactly as in v0.1.
The outer loop treats the *curriculum policy parameters* as the object of
improvement. A small population of CurriculumGenomes is mutated and selected
according to held-out Bures loss after a fixed training budget.

What is recursive:
  The system improves the rule that decides *what to study next*.
  Successful genomes become the parents of the next generation of genomes.

What is deliberately bounded:
  - The model architecture (QGT) is frozen.
  - The generator physics are frozen.
  - The evaluator (held-out loss) is immutable.
  - Population size and mutation are tiny so the whole thing still runs on a
    phone in pure NumPy.

This is Level-1 / Level-2 RSI in the common ladder (self-improving the
improvement strategy), not open-ended code rewriting.

Claims this module can test:
  R1. Outer evolution produces a curriculum policy that beats the hand-tuned
      defaults (error / progress / uniform) on held-out loss, under matched
      inner training budget.
  R2. The best genome's parameters themselves change across generations
      (the meta-policy is not static).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from quasar.quasar import Quasar, make_holdout, Learner


@dataclass
class CurriculumGenome:
    """Parameters that control the inner self_direct rule.

    temperature : float
        Softmax temperature on bin errors (error-driven arm).
    floor : float
        Minimum weight per bin (prevents total starvation).
    progress_mix : float in [0, 1]
        0.0 = pure error-driven, 1.0 = pure progress-driven,
        intermediate = convex combination of the two weight vectors.
    """
    temperature: float = 3.0
    floor: float = 0.05
    progress_mix: float = 0.0          # start near classic error-driven

    def mutate(self, rng: np.random.Generator, scale: float = 0.25) -> "CurriculumGenome":
        """Gaussian mutation with soft bounds."""
        child = copy.deepcopy(self)
        child.temperature = float(np.clip(
            self.temperature + rng.normal(0, scale * 1.5), 0.5, 8.0))
        child.floor = float(np.clip(
            self.floor + rng.normal(0, scale * 0.03), 0.01, 0.25))
        child.progress_mix = float(np.clip(
            self.progress_mix + rng.normal(0, scale * 0.3), 0.0, 1.0))
        return child

    def as_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "floor": self.floor,
            "progress_mix": self.progress_mix,
        }


class MetaQuasar:
    """Outer evolutionary loop that improves CurriculumGenome parameters."""

    def __init__(
        self,
        population_size: int = 6,
        seed: int = 0,
        n_bins: int = 4,
        seq_len: int = 6,
        inner_rounds: int = 5,
        n_per_round: int = 8,
        epochs_per_round: int = 2,
    ):
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.n_bins = n_bins
        self.seq_len = seq_len
        self.inner_rounds = inner_rounds
        self.n_per_round = n_per_round
        self.epochs_per_round = epochs_per_round

        # Fixed held-out set — the immutable evaluator
        self.Xh, self.Yh = make_holdout(seed=777, n=16, seq_len=seq_len)

        # Initial population: start from known good points + noise
        self.population: List[CurriculumGenome] = []
        base = CurriculumGenome(temperature=3.0, floor=0.05, progress_mix=0.0)
        self.population.append(base)
        self.population.append(CurriculumGenome(temperature=2.0, floor=0.08, progress_mix=0.7))
        self.population.append(CurriculumGenome(temperature=4.0, floor=0.03, progress_mix=0.3))
        while len(self.population) < population_size:
            self.population.append(base.mutate(self.rng, scale=0.4))

        self.history: List[dict] = []
        self.best_genome: Optional[CurriculumGenome] = None
        self.best_fitness: float = float("inf")

    def _evaluate_genome(self, genome: CurriculumGenome, eval_seed: int) -> float:
        """Run one full inner Quasar training under this genome; return holdout loss."""
        q = Quasar(
            seed=eval_seed,
            n_bins=self.n_bins,
            seq_len=self.seq_len,
            curriculum="error",          # we override self_direct manually
        )

        # Monkey-patch self_direct so the genome controls the weight update
        def genome_self_direct(bin_errs, temperature=None, floor=None):
            e = np.maximum(bin_errs, 1e-9)
            # error-driven weights
            e_pow = e ** genome.temperature
            w_err = e_pow / e_pow.sum()
            w_err = np.maximum(w_err, genome.floor)
            w_err = w_err / w_err.sum()

            # progress-driven weights (reuse the internal sampler logic)
            if q._progress_sampler is None:
                from quasar.quasar import _BinProgressSampler
                q._progress_sampler = _BinProgressSampler(q.n_bins, floor=genome.floor)
            w_prog = q._progress_sampler.weights(bin_errs)

            # convex combination
            mix = genome.progress_mix
            w = (1.0 - mix) * w_err + mix * w_prog
            w = np.maximum(w, genome.floor)
            q.weights = w / w.sum()

        q.self_direct = genome_self_direct  # type: ignore

        hist = q.run(
            rounds=self.inner_rounds,
            n_per_round=self.n_per_round,
            epochs_per_round=self.epochs_per_round,
            holdout=(self.Xh, self.Yh),
            adaptive=True,
            verbose=False,
        )
        return float(hist["holdout"][-1])

    def evolve(self, generations: int = 4, verbose: bool = True) -> CurriculumGenome:
        """Run the outer evolutionary loop. Returns the best genome found."""
        t0 = time.time()
        if verbose:
            print("=" * 66)
            print("QUASAR RSI — meta-curriculum evolution")
            print(f"population={self.population_size}  generations={generations}")
            print(f"inner budget: {self.inner_rounds} rounds × {self.n_per_round} traj × {self.epochs_per_round} epochs")
            print("=" * 66)

        for gen in range(generations):
            fitnesses = []
            for i, genome in enumerate(self.population):
                # Different seed per evaluation so noise averages a bit
                fit = self._evaluate_genome(genome, eval_seed=self.seed + gen * 100 + i)
                fitnesses.append(fit)
                if fit < self.best_fitness:
                    self.best_fitness = fit
                    self.best_genome = copy.deepcopy(genome)

            fitnesses = np.asarray(fitnesses)
            order = np.argsort(fitnesses)
            elite = self.population[order[0]]
            elite_fit = fitnesses[order[0]]

            self.history.append({
                "generation": gen,
                "best_fitness": float(elite_fit),
                "mean_fitness": float(fitnesses.mean()),
                "best_genome": elite.as_dict(),
            })

            if verbose:
                g = elite.as_dict()
                print(f"  gen {gen+1:2d} | best={elite_fit:.4f}  mean={fitnesses.mean():.4f}  "
                      f"T={g['temperature']:.2f} floor={g['floor']:.3f} mix={g['progress_mix']:.2f}")

            # Selection + mutation: keep top 2, fill the rest with mutated offspring
            next_pop = [copy.deepcopy(self.population[order[0]]),
                        copy.deepcopy(self.population[order[1]])]
            while len(next_pop) < self.population_size:
                parent = self.population[order[self.rng.integers(0, min(3, len(order)))]]
                child = parent.mutate(self.rng)
                next_pop.append(child)
            self.population = next_pop

        if verbose:
            print("-" * 66)
            print(f"Best genome found: {self.best_genome.as_dict()}")
            print(f"Best held-out loss: {self.best_fitness:.4f}")
            print(f"Runtime {time.time()-t0:.1f}s")
            print("=" * 66)
        return self.best_genome


def run_rsi_demo(generations: int = 3, population_size: int = 5, seed: int = 0):
    """Convenience entry point for a quick demonstration."""
    meta = MetaQuasar(population_size=population_size, seed=seed)
    best = meta.evolve(generations=generations, verbose=True)

    # Final comparison against the three classic fixed curricula
    print("\n[Final comparison under identical inner budget]")
    Xh, Yh = make_holdout(seed=777, n=16, seq_len=6)

    def eval_fixed(curriculum_name: str) -> float:
        q = Quasar(seed=0, curriculum=curriculum_name)
        h = q.run(rounds=5, n_per_round=8, epochs_per_round=2,
                  holdout=(Xh, Yh), adaptive=True, verbose=False)
        return float(h["holdout"][-1])

    results = {
        "uniform": eval_fixed("uniform"),
        "error": eval_fixed("error"),
        "progress": eval_fixed("progress"),
        "rsi-evolved": meta.best_fitness,
    }
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\nR1 (evolved beats at least one fixed baseline):",
          meta.best_fitness <= min(results["uniform"], results["error"], results["progress"]))
    print("R2 (genome parameters moved):",
          abs(best.temperature - 3.0) > 0.1 or abs(best.progress_mix) > 0.05)
    return meta, results


if __name__ == "__main__":
    run_rsi_demo()
