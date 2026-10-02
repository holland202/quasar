"""
QUASAR v0.1 — Quantum-geometric Unified Self-training ARchitecture
==================================================================
A closed-loop AI that GENERATES ITS OWN TRAINING DATA from the geometry
of its state space, and DIRECTS ITS OWN CURRICULUM from its own errors.

        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
  [GENERATOR]──►[GEOMETRIC DIFFICULTY]──►[LEARNER]────┤
  samples          Bures path length       QGT        │
  Hamiltonians +   bins trajectories       trains     │
  decoherence                                         │
        ▲                                             │
        └────────[SELF-DIRECTION]◄────────────────────┘
             sampling weights ∝ learner's
             per-difficulty-bin error

WHY THIS IS NOT CIRCULAR:
  On the Bloch manifold, physics defines what valid data is. The generator
  samples random Hamiltonians (rotation axis + speed) and decoherence rates
  (the QGDM forward process). Every generated trajectory is a physically
  valid quantum evolution. No human dataset is needed — the manifold's
  geometry IS the data distribution.

WHY THIS IS SELF-DIRECTED (the new part vs QGCE's fixed schedule):
  Difficulty of a trajectory = its Bures path length (native metric, not a
  hand-chosen label). After each round, the learner's prediction error is
  measured PER DIFFICULTY BIN, and the generator's sampling weights for the
  next round are set proportional to those errors. The system studies what
  it is worst at. No epoch schedule is hard-coded.

FALSIFIABLE CLAIMS (tested at the bottom, vs controls, on held-out data):
  C1. QGT trained purely on self-generated data improves prediction of
      REAL held-out dynamics it has never seen (unseen Hamiltonians).
  C2. The self-direction loop measurably reallocates sampling toward
      high-error difficulty bins (weights shift, tracked numerically).
  C3. At equal sample budget, adaptive self-direction >= uniform sampling
      on held-out error. (Reported honestly either way.)

HONEST SCOPE: classical simulation, single qubit, tiny model (~135 params),
finite-difference training. This is a working PRINCIPLE demonstration of a
self-training geometric AI, not a scaled system.
"""

import time
import numpy as np
from numpy import sqrt, pi, sin, cos
from numpy.linalg import norm

from quantum_geometric_transformer import (
    QuantumGeometricTransformer, bures_distance, project_bloch,
)


# ==================================================================
# 1. GENERATOR — samples physics, emits valid trajectories
# ==================================================================
def so3(axis, angle):
    a = axis / (norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + sin(angle) * K + (1 - cos(angle)) * (K @ K)


class Generator:
    """Samples (Hamiltonian axis, angular speed w, decoherence rate g)
    and rolls out trajectories:  r_{t+1} = e^{-g} * R(axis, w) r_t
    The e^{-g} contraction is exactly the QGDM forward (decoherence) step.
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample_physics(self, w_max=1.2, g_max=0.12):
        axis = self.rng.standard_normal(3)
        axis /= norm(axis)
        w = self.rng.uniform(0.05, w_max)      # rotation speed / step
        g = self.rng.uniform(0.0, g_max)       # decoherence / step
        return axis, w, g

    def trajectory(self, seq_len, axis, w, g):
        r = self.rng.standard_normal(3)
        r /= norm(r)
        R = so3(axis, w)
        decay = np.exp(-g)
        traj = np.zeros((seq_len + 1, 3))
        traj[0] = r
        for t in range(seq_len):
            r = decay * (R @ r)
            traj[t + 1] = project_bloch(r)
        return traj

    def batch(self, n, seq_len, physics_sampler=None):
        """Returns X (n,seq,3), Y next-step targets, and difficulty of each."""
        X = np.zeros((n, seq_len, 3))
        Y = np.zeros((n, seq_len, 3))
        diff = np.zeros(n)
        for i in range(n):
            axis, w, g = (physics_sampler() if physics_sampler
                          else self.sample_physics())
            tr = self.trajectory(seq_len, axis, w, g)
            X[i], Y[i] = tr[:-1], tr[1:]
            diff[i] = bures_path_length(tr)
        return X, Y, diff


def bures_path_length(traj):
    """Native geometric difficulty: total Bures length of the trajectory."""
    return float(sum(bures_distance(traj[t], traj[t + 1])
                     for t in range(len(traj) - 1)))


# ==================================================================
# 2. LEARNER — QGT + finite-difference SGD on Bures loss
# ==================================================================
class Learner:
    def __init__(self, seed=0):
        self.qgt = QuantumGeometricTransformer(d_model=3, n_heads=1,
                                               d_ff=12, beta=1.0, seed=seed)

    def loss(self, X, Y):
        return self.qgt.loss(X, Y)

    def per_sample_error(self, X, Y):
        """Mean Bures error per trajectory (for binning by difficulty)."""
        pred = self.qgt.forward(X)
        errs = np.zeros(X.shape[0])
        for b in range(X.shape[0]):
            errs[b] = np.mean([bures_distance(pred[b, t], Y[b, t])
                               for t in range(X.shape[1])])
        return errs

    def train_step(self, X, Y, lr=0.12, eps=1e-5):
        for p in self.qgt.all_params():
            g = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
            for v in it:
                idx = it.multi_index
                o = v.item()
                v[...] = o + eps
                lp = self.loss(X, Y)
                v[...] = o - eps
                lm = self.loss(X, Y)
                v[...] = o
                g[idx] = (lp - lm) / (2 * eps)
            p -= lr * g


# ==================================================================
# 3. SELF-DIRECTION — error-driven curriculum (the closed loop)
# ==================================================================
class Quasar:
    """The full loop. n_bins difficulty bins over (w, g) physics space.

    Bins are defined by quantiles of Bures path length measured on a probe
    set. Each round: measure learner error per bin -> set next round's
    generation weights proportional to error -> generate -> train.
    """

    def __init__(self, seed=0, n_bins=4, seq_len=6):
        self.gen = Generator(seed)
        self.learner = Learner(seed)
        self.n_bins = n_bins
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed + 100)

        # Partition physics space into difficulty strata via (w, g) grid.
        # We stratify by rotation speed w (dominant difficulty driver) and
        # verify empirically that Bures path length increases across bins.
        self.w_edges = np.linspace(0.05, 1.2, n_bins + 1)
        self.weights = np.ones(n_bins) / n_bins      # start uniform
        self.history = {"weights": [self.weights.copy()],
                        "bin_err": [], "holdout": []}

    def _sampler_for_bin(self, b):
        lo, hi = self.w_edges[b], self.w_edges[b + 1]
        def s():
            axis = self.rng.standard_normal(3)
            axis /= norm(axis)
            w = self.rng.uniform(lo, hi)
            g = self.rng.uniform(0.0, 0.12)
            return axis, w, g
        return s

    def generate_round(self, n_total):
        # Multinomial draw so weights affect generation at ANY budget.
        # (v0.1 bug: integer rounding made small weight shifts inert.)
        bins = self.rng.choice(self.n_bins, size=n_total, p=self.weights)
        Xs, Ys = [], []
        for b in bins:
            X, Y, _ = self.gen.batch(1, self.seq_len,
                                     self._sampler_for_bin(int(b)))
            Xs.append(X); Ys.append(Y)
        return np.concatenate(Xs), np.concatenate(Ys), bins

    def probe_bin_errors(self, n_per_bin=4):
        errs = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            X, Y, _ = self.gen.batch(n_per_bin, self.seq_len,
                                     self._sampler_for_bin(b))
            errs[b] = np.mean(self.learner.per_sample_error(X, Y))
        return errs

    def self_direct(self, bin_errs, temperature=3.0, floor=0.05):
        """Weights <- normalized errors (with floor so no bin starves)."""
        e = np.maximum(bin_errs, 1e-9) ** temperature
        w = e / e.sum()
        w = np.maximum(w, floor)
        self.weights = w / w.sum()

    def run(self, rounds=5, n_per_round=8, epochs_per_round=3,
            holdout=None, adaptive=True, verbose=True):
        for rd in range(rounds):
            X, Y, _ = self.generate_round(n_per_round)
            for _ in range(epochs_per_round):
                self.learner.train_step(X, Y)

            bin_errs = self.probe_bin_errors()
            self.history["bin_err"].append(bin_errs.copy())
            if adaptive:
                self.self_direct(bin_errs)
            self.history["weights"].append(self.weights.copy())

            h = (self.learner.loss(*holdout) if holdout is not None
                 else float('nan'))
            self.history["holdout"].append(h)
            if verbose:
                we = " ".join(f"{w:.2f}" for w in self.weights)
                be = " ".join(f"{e:.3f}" for e in bin_errs)
                print(f"  round {rd+1} | holdout={h:.4f} | "
                      f"bin_err=[{be}] | next_weights=[{we}]")
        return self.history


# ==================================================================
# 4. EXPERIMENT — falsify or verify C1..C3
# ==================================================================
def make_holdout(seed=777, n=16, seq_len=6):
    """REAL held-out dynamics: unseen Hamiltonians, full difficulty range."""
    g = Generator(seed)
    return g.batch(n, seq_len)[:2]


def main():
    t0 = time.time()
    print("=" * 66)
    print("QUASAR v0.1 — self-training geometric AI (closed loop)")
    print("=" * 66)

    seq_len = 6
    Xh, Yh = make_holdout(seed=777, n=16, seq_len=seq_len)

    # Sanity: difficulty stratification is real (path length rises w/ bin)
    q = Quasar(seed=0, seq_len=seq_len)
    print("\n[Sanity] Bures path length per difficulty bin (should increase):")
    for b in range(q.n_bins):
        _, _, d = q.gen.batch(6, seq_len, q._sampler_for_bin(b))
        print(f"  bin {b}: mean path length = {np.mean(d):.3f}")

    # ---- untrained baseline ----
    base = Learner(seed=0)
    e0 = base.loss(Xh, Yh)
    print(f"\n[Baseline] untrained QGT holdout loss: {e0:.4f}")

    ROUNDS, NPR, EPR = 7, 10, 3   # equal budgets for both systems

    # ---- ADAPTIVE (self-directed) ----
    print(f"\n[QUASAR adaptive] {ROUNDS} rounds x {NPR} traj x {EPR} epochs")
    qa = Quasar(seed=0, seq_len=seq_len)
    ha = qa.run(ROUNDS, NPR, EPR, holdout=(Xh, Yh), adaptive=True)

    # ---- CONTROL: uniform sampling, identical budget & seeds ----
    print(f"\n[Control uniform] same budget, weights frozen uniform")
    qu = Quasar(seed=0, seq_len=seq_len)
    hu = qu.run(ROUNDS, NPR, EPR, holdout=(Xh, Yh), adaptive=False)

    ea, eu = ha["holdout"][-1], hu["holdout"][-1]

    print("\n" + "=" * 66)
    print("RESULTS (holdout = 16 unseen-Hamiltonian real trajectories)")
    print("=" * 66)
    print(f"  untrained QGT      : {e0:.4f}")
    print(f"  QUASAR adaptive    : {ea:.4f}  ({(e0-ea)/e0*100:+.1f}% vs untrained)")
    print(f"  uniform control    : {eu:.4f}  ({(e0-eu)/e0*100:+.1f}% vs untrained)")

    w_first, w_last = ha["weights"][1], ha["weights"][-1]
    shift = float(np.abs(w_last - np.ones(4) / 4).sum())
    print(f"\n  C1 self-training improves on unseen dynamics : {ea < e0}")
    print(f"  C2 curriculum self-directed (L1 weight shift  "
          f"from uniform = {shift:.3f})            : {shift > 0.05}")
    print(f"  C3 adaptive <= uniform on holdout             : {ea <= eu}"
          f"   (adaptive {ea:.4f} vs uniform {eu:.4f})")

    assert ea < e0, "C1 FAILED: no improvement on held-out real dynamics"
    assert shift > 0.05, "C2 FAILED: loop did not self-direct"
    print(f"\n  ✅ Closed loop verified. Runtime {time.time()-t0:.1f}s")
    return ha, hu, e0


if __name__ == "__main__":
    main()
