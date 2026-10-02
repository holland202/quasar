"""
QUASAR v0.2 — M3: Articulation experiment
=========================================
Idea (adapted from Anthropic's counterfactual-reflection result, and the
one salvageable strategy from an external proposal): train the QGT to
ARTICULATE the physics that generated each trajectory — rotation axis,
angular speed w, decoherence g — alongside next-state prediction.

Why this is the right M3 task, from geometry:
  One transition r0 -> r1 = e^{-g} R(axis,w) r0 gives 3 equations for
  4 unknowns (axis 2dof, w, g): UNDERDETERMINED. Two transitions determine
  the physics. So the articulation head provably requires integrating
  information across >= 3 sequence positions — exactly the cross-position
  demand that F2 showed our attention never faced (broadcast ratio 0.001).

FALSIFIABLE PREDICTIONS (either outcome goes in FINDINGS):
  P-A  Articulation training raises the attention broadcast ratio
       (information must flow into the final position).
  P-B  (external claim) Articulation improves holdout NEXT-STATE error
       on unseen Hamiltonians vs a matched-budget state-only control.

Protocol: identical data streams, seeds, step counts, learning rate.
Only difference: joint loss  L = L_state + lam * MSE(physics).
"""

import time
import numpy as np
from numpy.linalg import norm

from quantum_geometric_transformer import QuantumGeometricTransformer
from quasar import Generator
from backprop import (T, add, matmul, tanh, softmax_rows, pbloch,
                      bures_scores, bures_mean_loss, QGTAnalytical)


# ---- two extra primitives -----------------------------------------
def take_last(a):
    """Slice [:, -1, :]."""
    out = T(a.data[:, -1, :], (a,))
    def bw(g):
        full = np.zeros_like(a.data)
        full[:, -1, :] = g
        a._acc(full)
    out._bw = bw
    return out


def mse_to_const(pred, target):
    diff = pred.data - target
    out = T(np.mean(diff ** 2), (pred,))
    n = diff.size
    def bw(g):
        pred._acc(g * 2.0 * diff / n)
    out._bw = bw
    return out


def combine(La, Lb, lam):
    out = T(La.data + lam * Lb.data, (La, Lb))
    def bw(g):
        La._acc(g)
        Lb._acc(lam * g)
    out._bw = bw
    return out


# ---- articulated model on the AD engine ---------------------------
class ArticulatedAnalytical(QGTAnalytical):
    """QGT + physics head reading the final position's pre-output state.
    Physics target: [axis_x, axis_y, axis_z, w, g] (raw MSE; (axis,w) is
    identifiable since w > 0 by construction)."""

    def __init__(self, qgt, seed=7):
        super().__init__(qgt)
        rng = np.random.default_rng(seed)
        self.Wp = np.ascontiguousarray(rng.normal(0, 0.1, (3, 5)))
        self.bp = np.ascontiguousarray(rng.normal(0, 0.01, 5))
        self.p["Wp"] = T(self.Wp)
        self.p["bp"] = T(self.bp)

    def _graph(self, x):
        qgt = self.qgt
        pos = qgt.pos_enc.encode(x.shape[1])
        s0 = pbloch(T(x + pos[None], requires_grad=False))
        s0.requires_grad = False
        Q = matmul(s0, self.p["Wq"]); K = matmul(s0, self.p["Wk"])
        V = matmul(s0, self.p["Wv"])
        attn = softmax_rows(bures_scores(Q, K, qgt.attn.beta, causal=True))
        s1 = pbloch(add(s0, pbloch(matmul(matmul(attn, V), self.p["Wo"]))))
        h = tanh(add(matmul(s1, self.p["W1"]), self.p["b1"]))
        s2 = pbloch(add(s1, pbloch(add(matmul(h, self.p["W2"]), self.p["b2"]))))
        out = pbloch(add(matmul(s2, self.p["Wout"]), self.p["bout"]))
        return s2, out

    def joint_step(self, x, y, phys, lam=0.5, lr=0.12, articulate=True):
        for t in self.p.values():
            t.grad = None
        s2, out = self._graph(x)
        L_state = bures_mean_loss(out, y)
        physpred = add(matmul(take_last(s2), self.p["Wp"]), self.p["bp"])
        L_phys = mse_to_const(physpred, phys)
        L = combine(L_state, L_phys, lam) if articulate else L_state
        L.backward()
        for name, t in self.p.items():
            if t.grad is not None:
                t.data -= lr * t.grad
        # keep qgt's arrays in sync (they are the same objects for the
        # original params; Wp/bp live only here)
        return float(L_state.data), float(L_phys.data)

    def physics_error(self, x, phys):
        for t in self.p.values():
            t.grad = None
        s2, _ = self._graph(x)
        pred = take_last(s2).data @ self.p["Wp"].data + self.p["bp"].data
        return float(np.mean((pred - phys) ** 2))


# ---- data with physics labels --------------------------------------
def batch_with_physics(gen, n, seq_len):
    X = np.zeros((n, seq_len, 3)); Y = np.zeros((n, seq_len, 3))
    P = np.zeros((n, 5))
    for i in range(n):
        axis, w, g = gen.sample_physics()
        tr = gen.trajectory(seq_len, axis, w, g)
        X[i], Y[i] = tr[:-1], tr[1:]
        P[i] = np.concatenate([axis, [w, g]])
    return X, Y, P


# ---- broadcast ratio (F2 instrument, inlined) -----------------------
def broadcast_ratio(qgt, gen_seed=123, n=30, seq_len=6):
    from quantum_geometric_transformer import bures_distance, project_bloch
    g = Generator(gen_seed)
    rng = np.random.default_rng(0)
    XA, _, _ = g.batch(n, seq_len)
    XB, _, _ = g.batch(n, seq_len)
    pos = qgt.pos_enc.encode(seq_len)

    def s0_of(x):
        return project_bloch(x + pos[None])

    def fwd(s0):
        s1 = project_bloch(s0 + qgt.attn.forward(s0))
        s2 = project_bloch(s1 + qgt.ffn.forward(s1))
        return project_bloch(s2 @ qgt.W_out + qgt.b_out)

    direct, bcast = [], []
    for i in range(n):
        sA = s0_of(XA[i:i+1]); sB = s0_of(XB[i:i+1])
        outB = fwd(sB)
        t = rng.integers(0, seq_len - 1)
        sp = sB.copy(); sp[0, t] = sA[0, t]
        outP = fwd(sp)
        d = [bures_distance(outP[0, k], outB[0, k]) for k in range(seq_len)]
        direct.append(d[t])
        bcast.append(np.mean([d[k] for k in range(seq_len) if k != t]))
    return float(np.mean(bcast)) / max(float(np.mean(direct)), 1e-12)


# ---- the experiment -------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 66)
    print("M3 — ARTICULATION EXPERIMENT (matched-budget control)")
    print("=" * 66)
    seq_len, steps, bsz, lam = 6, 400, 8, 0.5

    # identical data stream for both arms
    gen = Generator(seed=11)
    batches = [batch_with_physics(gen, bsz, seq_len) for _ in range(steps)]

    # holdout: unseen Hamiltonians
    ghold = Generator(seed=999)
    Xh, Yh, Ph = batch_with_physics(ghold, 32, seq_len)

    # physics-error floor: always predict the training-mean physics vector
    Pmean = np.mean(np.concatenate([b[2] for b in batches]), axis=0)
    phys_floor = float(np.mean((Ph - Pmean) ** 2))

    arms = {}
    for name, artic in [("control (state only)", False),
                        ("articulated", True)]:
        qgt = QuantumGeometricTransformer(d_model=3, n_heads=1, d_ff=12, seed=0)
        model = ArticulatedAnalytical(qgt, seed=7)
        for (x, y, p) in batches:
            model.joint_step(x, y, p, lam=lam, articulate=artic)
        arms[name] = {
            "holdout_state": qgt.loss(Xh, Yh),
            "holdout_phys": model.physics_error(Xh, Ph),
            "broadcast": broadcast_ratio(qgt),
        }

    base = QuantumGeometricTransformer(d_model=3, n_heads=1, d_ff=12, seed=0)
    r0 = broadcast_ratio(base)

    print(f"\n  physics-error floor (predict mean): {phys_floor:.4f}")
    print(f"  broadcast ratio, untrained        : {r0:.5f}\n")
    print(f"  {'arm':<22}{'holdout state':>14}{'holdout phys':>14}"
          f"{'broadcast':>11}")
    for name, r in arms.items():
        print(f"  {name:<22}{r['holdout_state']:>14.4f}"
              f"{r['holdout_phys']:>14.4f}{r['broadcast']:>11.5f}")

    c, a = arms["control (state only)"], arms["articulated"]
    print("\n  P-A broadcast ratio raised by articulation : "
          f"{a['broadcast'] > 3 * c['broadcast']}  "
          f"({c['broadcast']:.5f} -> {a['broadcast']:.5f})")
    print("  P-B articulation improves holdout state    : "
          f"{a['holdout_state'] < c['holdout_state']}  "
          f"({c['holdout_state']:.4f} vs {a['holdout_state']:.4f})")
    print("      physics learned (vs predict-mean floor): "
          f"{a['holdout_phys'] < phys_floor}  "
          f"({a['holdout_phys']:.4f} vs floor {phys_floor:.4f})")
    print(f"\n  runtime {time.time()-t0:.1f}s "
          f"({2*steps} analytical steps total)")


if __name__ == "__main__":
    main()
