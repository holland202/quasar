"""
QUASAR v0.2 — M1: Analytical backprop through the full QGT
===========================================================
A minimal reverse-mode autodiff engine (pure NumPy) with fused primitives
for the quantum-geometric ops, plus an AD re-implementation of the QGT
forward pass that SHARES the original model's parameter arrays.

Verification standard (all run in main(), nothing asserted that isn't):
  V1  Forward equivalence: AD forward reproduces qgt.loss() to ~1e-12.
  V2  Gradient correctness: AD gradients vs central finite differences
      over ALL parameters of the model. Report max abs / rel error.
  V3  Speed: wall-clock per training step, finite-diff vs analytical.
  V4  Sanity: a short analytical-gradient training run decreases loss.

Gradient-safety conventions (mirroring the verified bures_distance_gradient):
  d(arccos√F)/dF is masked to 0 where F ≥ 0.999999 or F ≤ 1e-9, exactly as
  the original hand gradient zeroes near-identical states.
"""

import time
import numpy as np
from numpy.linalg import norm

from quantum_geometric_transformer import QuantumGeometricTransformer


# ==================================================================
# Micro reverse-mode autodiff
# ==================================================================
class T:
    """Tensor node: data + grad + backward closure."""
    __slots__ = ("data", "grad", "_bw", "_parents", "requires_grad")

    def __init__(self, data, parents=(), bw=None, requires_grad=True):
        self.data = np.asarray(data, dtype=float)
        self.grad = None
        self._parents = parents
        self._bw = bw
        self.requires_grad = requires_grad

    def backward(self):
        # topological order
        topo, seen = [], set()
        def build(t):
            if id(t) in seen or not t.requires_grad:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)
        build(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(topo):
            if t._bw is not None:
                t._bw(t.grad)

    def _acc(self, g):
        if self.grad is None:
            self.grad = g.copy()
        else:
            self.grad += g


def _unbroadcast(g, shape):
    """Sum gradient g down to `shape` (reverse of numpy broadcasting)."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, s in enumerate(shape):
        if s == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


def add(a, b):
    out = T(a.data + b.data, (a, b))
    def bw(g):
        if a.requires_grad: a._acc(_unbroadcast(g, a.data.shape))
        if b.requires_grad: b._acc(_unbroadcast(g, b.data.shape))
    out._bw = bw
    return out


def matmul(a, b):
    """Batched matmul via np.matmul semantics."""
    out = T(a.data @ b.data, (a, b))
    def bw(g):
        if a.requires_grad:
            ga = g @ np.swapaxes(b.data, -1, -2)
            a._acc(_unbroadcast(ga, a.data.shape))
        if b.requires_grad:
            gb = np.swapaxes(a.data, -1, -2) @ g
            b._acc(_unbroadcast(gb, b.data.shape))
    out._bw = bw
    return out


def tanh(a):
    y = np.tanh(a.data)
    out = T(y, (a,))
    def bw(g):
        a._acc(g * (1 - y * y))
    out._bw = bw
    return out


def softmax_rows(a):
    """Fused softmax along last axis."""
    m = np.max(a.data, axis=-1, keepdims=True)
    e = np.exp(a.data - m)
    y = e / e.sum(axis=-1, keepdims=True)
    out = T(y, (a,))
    def bw(g):
        dot = np.sum(g * y, axis=-1, keepdims=True)
        a._acc(y * (g - dot))
    out._bw = bw
    return out


def pbloch(a):
    """Fused project_bloch. y = r/|r| where |r|>1 else r."""
    mag = norm(a.data, axis=-1, keepdims=True)
    mask = mag > 1
    safe = np.where(mag > 1e-12, mag, 1.0)
    y = np.where(mask, a.data / safe, a.data)
    out = T(y, (a,))
    def bw(g):
        yhat = a.data / safe
        gp = (g - np.sum(g * yhat, axis=-1, keepdims=True) * yhat) / safe
        a._acc(np.where(mask, gp, g))
    out._bw = bw
    return out


def _dd_dF(F):
    """d(arccos √F)/dF with the same safety masking as the verified
    hand gradient: zero outside (1e-9, 0.999999)."""
    ok = (F > 1e-9) & (F < 0.999999)
    Fs = np.where(ok, F, 0.5)
    d = -1.0 / (2.0 * np.sqrt(Fs) * np.sqrt(1.0 - Fs))
    return np.where(ok, d, 0.0)


def bures_scores(Q, K, beta, causal=True):
    """Fused pairwise Bures scores: (b,s,s), scores = -beta*d_B(Q_i,K_j),
    causal cells replaced with -1e9 (matching original forward exactly)."""
    q, k = Q.data, K.data                          # (b,s,3)
    dots = q @ np.swapaxes(k, -1, -2)              # (b,s,s)
    mq = 1 - np.sum(q * q, axis=-1)                # (b,s)
    mk = 1 - np.sum(k * k, axis=-1)
    u = mq[:, :, None] * mk[:, None, :]            # (b,s,s)
    upos = np.maximum(0, u)
    s_term = np.sqrt(upos)
    F = np.clip(0.5 * (1 + dots + s_term), 0, 1)
    D = np.arccos(np.sqrt(F))
    b, s, _ = D.shape
    cmask = np.triu(np.ones((s, s), dtype=bool), k=1)[None] if causal else \
        np.zeros((1, s, s), dtype=bool)
    scores = np.where(cmask, -1e9, -beta * D)
    out = T(scores, (Q, K))
    # cache for backward
    inv_s = np.where(s_term > 1e-12, 1.0 / np.where(s_term > 1e-12, s_term, 1.0), 0.0)
    def bw(g):
        gD = np.where(cmask, 0.0, -beta * g)       # through the mask
        # clip mask: dF zero where F was clipped (F==0 or F==1 handled by _dd_dF)
        gF = gD * _dd_dF(F)                        # (b,s,s)
        if Q.requires_grad:
            # dF/dq_i = 0.5*k_j - 0.5*q_i*mk_j*inv_s   (second term iff u>0)
            t1 = 0.5 * (gF @ k)                    # (b,s,3)
            w = gF * np.where(u > 0, mk[:, None, :] * inv_s, 0.0)
            t2 = -0.5 * np.sum(w, axis=-1, keepdims=True) * q
            Q._acc(t1 + t2)
        if K.requires_grad:
            gFT = np.swapaxes(gF, -1, -2)
            t1 = 0.5 * (gFT @ q)
            w = gFT * np.where(np.swapaxes(u, -1, -2) > 0,
                               mq[:, None, :] * np.swapaxes(inv_s, -1, -2), 0.0)
            t2 = -0.5 * np.sum(w, axis=-1, keepdims=True) * k
            K._acc(t1 + t2)
    out._bw = bw
    return out


def bures_mean_loss(pred, target):
    """Fused mean Bures distance to a constant target. Scalar output."""
    p, t = pred.data, target                       # (b,s,3)
    dots = np.sum(p * t, axis=-1)
    mp = 1 - np.sum(p * p, axis=-1)
    mt = 1 - np.sum(t * t, axis=-1)
    u = mp * mt
    upos = np.maximum(0, u)
    s_term = np.sqrt(upos)
    F = np.clip(0.5 * (1 + dots + s_term), 0, 1)
    D = np.arccos(np.sqrt(F))
    out = T(np.mean(D ** 2), (pred,))
    inv_s = np.where(s_term > 1e-12, 1.0 / np.where(s_term > 1e-12, s_term, 1.0), 0.0)
    n = D.size
    def bw(g):
        gF = (g * 2.0 * D / n) * _dd_dF(F)         # through mean(D^2)
        dF = 0.5 * t - 0.5 * np.where((u > 0), mt * inv_s, 0.0)[..., None] * p
        pred._acc(gF[..., None] * dF)
    out._bw = bw
    return out


# ==================================================================
# QGT forward in the engine (shares qgt's parameter arrays)
# ==================================================================
class QGTAnalytical:
    def __init__(self, qgt):
        self.qgt = qgt
        # wrap params: T objects VIEW the same numpy arrays
        a = qgt.attn
        f = qgt.ffn
        self.p = {name: T(arr) for name, arr in [
            ("Wq", a.W_q), ("Wk", a.W_k), ("Wv", a.W_v), ("Wo", a.W_o),
            ("W1", f.W1), ("b1", f.b1), ("W2", f.W2), ("b2", f.b2),
            ("Wout", qgt.W_out), ("bout", qgt.b_out)]}

    def loss(self, x, y):
        """Build graph, return (loss_value, grads dict). x,y: (b,s,3)."""
        for t in self.p.values():
            t.grad = None
            t.data = np.asarray(t.data)            # stay in sync
        qgt = self.qgt
        pos = qgt.pos_enc.encode(x.shape[1])
        s0 = pbloch(T(x + pos[None], requires_grad=False))
        # s0 is constant w.r.t. params, but pbloch needs a T; mark const:
        s0.requires_grad = False

        Q = matmul(s0, self.p["Wq"])
        K = matmul(s0, self.p["Wk"])
        V = matmul(s0, self.p["Wv"])
        scores = bures_scores(Q, K, qgt.attn.beta, causal=True)
        attn = softmax_rows(scores)
        ao = pbloch(matmul(matmul(attn, V), self.p["Wo"]))
        s1 = pbloch(add(s0, ao))

        h = tanh(add(matmul(s1, self.p["W1"]), self.p["b1"]))
        fo = pbloch(add(matmul(h, self.p["W2"]), self.p["b2"]))
        s2 = pbloch(add(s1, fo))

        logits = add(matmul(s2, self.p["Wout"]), self.p["bout"])
        out = pbloch(logits)
        L = bures_mean_loss(out, y)
        L.backward()
        grads = {k: (t.grad if t.grad is not None else np.zeros_like(t.data))
                 for k, t in self.p.items()}
        return float(L.data), grads

    def train_step(self, x, y, lr=0.12):
        L, g = self.loss(x, y)
        order = ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "Wout", "bout"]
        arrays = (self.qgt.attn.params() + self.qgt.ffn.params()
                  + [self.qgt.W_out, self.qgt.b_out])
        for name, arr in zip(order, arrays):
            arr -= lr * g[name]
            self.p[name].data = arr
        return L


# ==================================================================
# Verification
# ==================================================================
def main():
    print("=" * 64)
    print("M1 — ANALYTICAL BACKPROP THROUGH THE FULL QGT")
    print("=" * 64)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 6, 3))
    x = x / norm(x, axis=-1, keepdims=True)
    y = rng.standard_normal((4, 6, 3))
    y = y / norm(y, axis=-1, keepdims=True) * 0.9

    qgt = QuantumGeometricTransformer(d_model=3, n_heads=1, d_ff=12, seed=0)
    ad = QGTAnalytical(qgt)

    # V1: forward equivalence
    L_ref = qgt.loss(x, y)
    L_ad, grads = ad.loss(x, y)
    print(f"\n[V1] forward equivalence: |L_AD - L_ref| = {abs(L_ad-L_ref):.2e}"
          f"   (L = {L_ref:.6f})")
    assert abs(L_ad - L_ref) < 1e-10

    # V2: full finite-difference gradient check over ALL parameters
    print("\n[V2] gradient check vs central finite differences (eps=1e-6):")
    eps = 1e-6
    order = ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "Wout", "bout"]
    arrays = (qgt.attn.params() + qgt.ffn.params() + [qgt.W_out, qgt.b_out])
    worst_abs, worst_rel, n_par = 0.0, 0.0, 0
    for name, arr in zip(order, arrays):
        fd = np.zeros_like(arr)
        it = np.nditer(arr, flags=['multi_index'], op_flags=['readwrite'])
        for v in it:
            idx = it.multi_index
            o = v.item()
            v[...] = o + eps; Lp = qgt.loss(x, y)
            v[...] = o - eps; Lm = qgt.loss(x, y)
            v[...] = o
            fd[idx] = (Lp - Lm) / (2 * eps)
        aerr = np.abs(grads[name] - fd)
        scale = np.maximum(np.abs(fd), 1e-8)
        rerr = aerr / scale
        worst_abs = max(worst_abs, aerr.max())
        worst_rel = max(worst_rel, rerr.max())
        n_par += arr.size
        print(f"   {name:>5}: max|Δ| = {aerr.max():.2e}   max rel = {rerr.max():.2e}")
    print(f"   -> {n_par} parameters checked | worst abs {worst_abs:.2e} | "
          f"worst rel {worst_rel:.2e}")
    assert worst_abs < 1e-6, "Gradient check FAILED"

    # V3: speed benchmark
    print("\n[V3] wall-clock per training step (batch 4, seq 6):")
    def fd_train_step(m, x, y, lr=0.12, eps=1e-5):
        for p in m.all_params():
            g = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
            for v in it:
                idx = it.multi_index
                o = v.item()
                v[...] = o + eps; Lp = m.loss(x, y)
                v[...] = o - eps; Lm = m.loss(x, y)
                v[...] = o
                g[idx] = (Lp - Lm) / (2 * eps)
            p -= lr * g
    qgt_fd = QuantumGeometricTransformer(d_model=3, n_heads=1, d_ff=12, seed=0)
    t0 = time.time()
    for _ in range(3):
        fd_train_step(qgt_fd, x, y)
    t_fd = (time.time() - t0) / 3
    qgt_ad = QuantumGeometricTransformer(d_model=3, n_heads=1, d_ff=12, seed=0)
    ad2 = QGTAnalytical(qgt_ad)
    t0 = time.time()
    for _ in range(3):
        ad2.train_step(x, y, lr=0.12)
    t_ad = (time.time() - t0) / 3
    print(f"   finite-difference step: {t_fd*1000:8.1f} ms")
    print(f"   analytical step       : {t_ad*1000:8.1f} ms")
    print(f"   speedup               : {t_fd/t_ad:8.1f}x")

    # V4: training sanity with analytical gradients
    print("\n[V4] 30 analytical-gradient steps:")
    L0 = ad2.train_step(x, y)
    for _ in range(28):
        ad2.train_step(x, y)
    L1 = ad2.train_step(x, y)
    print(f"   loss {L0:.4f} -> {L1:.4f}  (decreased: {L1 < L0})")
    assert L1 < L0

    print("\n✅ M1 VERIFIED: analytical backprop matches finite differences "
          "across all parameters,\n   at a "
          f"{t_fd/t_ad:.0f}x per-step speedup. Capacity scale-up is unblocked.")
    return t_fd / t_ad, worst_abs


if __name__ == "__main__":
    main()
