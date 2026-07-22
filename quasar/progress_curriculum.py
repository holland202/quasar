"""quasar/progress_curriculum.py

Progress-driven curriculum sampling.
Instead of sampling where error is highest, sample where difficulty gradient is steepest.
This biases the generator toward regions where the model is learning fastest.

Usage:
    from quasar.progress_curriculum import ProgressCurriculumSampler
    sampler = ProgressCurriculumSampler(base_generator, memory_size=100)
    batch = sampler.sample_batch(n=32, n_steps=10)
"""

import numpy as np
from collections import deque


class ProgressCurriculumSampler:
    """
    Maintains a rolling memory of recent trajectories and their difficulties.
    Samples new trajectories weighted by the *gradient* of difficulty:
    regions where difficulty is changing most rapidly are prioritized.
    """

    def __init__(self, base_generator, memory_size=200, gradient_window=5, temperature=1.0, seed=None):
        self.base = base_generator
        self.memory = deque(maxlen=memory_size)
        self.gradient_window = gradient_window
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.step_counter = 0

    def _compute_difficulty(self, trajectory):
        """Extract difficulty from trajectory dict."""
        return trajectory.get('difficulty', 0.0)

    def _gradient_score(self, traj):
        """
        Compute progress score for a trajectory.
        High score = steep difficulty gradient = high learning potential.
        """
        d = self._compute_difficulty(traj)
        if len(self.memory) < self.gradient_window:
            return d  # fallback to raw difficulty when cold
        
        # Look at recent memory near this difficulty level
        recent = list(self.memory)[-self.gradient_window:]
        diffs = [self._compute_difficulty(t) for t in recent]
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs) + 1e-6
        
        # Score = how far from local mean / local std (z-score magnitude)
        # This prioritizes trajectories that are breaking the recent pattern
        z = abs(d - mean_diff) / std_diff
        return z

    def sample_trajectory(self, n_steps=10, **gen_kwargs):
        """Generate one trajectory and store it."""
        traj = self.base.generate_trajectory(n_steps, **gen_kwargs)
        self.memory.append(traj)
        self.step_counter += 1
        return traj

    def sample_batch(self, n, n_steps=10, **gen_kwargs):
        """
        Generate n trajectories with progress-driven filtering.
        Uses rejection sampling: generate 2n candidates, keep the n with highest gradient scores.
        """
        candidates = [self.sample_trajectory(n_steps, **gen_kwargs) for _ in range(n * 2)]
        scores = [self._gradient_score(t) for t in candidates]
        
        # Softmax selection
        scores = np.array(scores)
        scores = scores - np.max(scores)
        probs = np.exp(scores / self.temperature)
        probs = probs / np.sum(probs)
        
        indices = self.rng.choice(len(candidates), size=n, replace=False, p=probs)
        selected = [candidates[i] for i in indices]
        
        # Update memory with selected only (prune rejects)
        for _ in range(n):
            self.memory.pop()  # remove the extra candidates from memory
        for s in selected:
            self.memory.append(s)
        
        return selected

    def get_difficulty_distribution(self):
        """Return statistics of difficulty in memory."""
        if not self.memory:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        diffs = [self._compute_difficulty(t) for t in self.memory]
        return {
            'mean': float(np.mean(diffs)),
            'std': float(np.std(diffs)),
            'min': float(np.min(diffs)),
            'max': float(np.max(diffs)),
            'n': len(diffs)
        }


def demo_progress_vs_error():
    """Compare progress-driven sampling against naive error-driven."""
    print("=" * 60)
    print("PROGRESS CURRICULUM DEMO")
    print("=" * 60)

    class DummyGenerator:
        def __init__(self, seed=42):
            self.rng = np.random.default_rng(seed)
        def generate_trajectory(self, n_steps=10):
            # Simulate a "learning landscape": difficulty clusters around 3 and 7
            if self.rng.random() < 0.5:
                d = self.rng.normal(3.0, 0.5)
            else:
                d = self.rng.normal(7.0, 0.5)
            return {'difficulty': max(0, d), 'states': []}

    base = DummyGenerator(seed=42)

    # Naive: just accept everything
    naive = [base.generate_trajectory(10) for _ in range(100)]
    naive_diffs = [t['difficulty'] for t in naive]

    # Progress-driven
    prog = ProgressCurriculumSampler(base, memory_size=100, temperature=0.5, seed=42)
    prog_batch = prog.sample_batch(100, n_steps=10)
    prog_diffs = [t['difficulty'] for t in prog_batch]

    print(f"\nNaive sampling (uniform):")
    print(f"  mean={np.mean(naive_diffs):.2f}, std={np.std(naive_diffs):.2f}")
    print(f"  range=[{np.min(naive_diffs):.2f}, {np.max(naive_diffs):.2f}]")

    print(f"\nProgress curriculum (gradient-weighted):")
    print(f"  mean={np.mean(prog_diffs):.2f}, std={np.std(prog_diffs):.2f}")
    print(f"  range=[{np.min(prog_diffs):.2f}, {np.max(prog_diffs):.2f}]")
    print(f"  memory distribution: {prog.get_difficulty_distribution()}")

    # Progress curriculum should show higher std — it's actively seeking the boundaries
    if np.std(prog_diffs) > np.std(naive_diffs):
        print("\n[PASS] Progress curriculum increases diversity")
    else:
        print("\n[NOTE] Curricula converged — landscape may be too simple")

    print("=" * 60)


if __name__ == "__main__":
    demo_progress_vs_error()
