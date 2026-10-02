#!/usr/bin/env python3
"""
F18 — Bures-metric attention vs dot-product attention, n=1 and n=2
==================================================================
Tests the QGT's founding premise (never registered in F1-F16) and M4's
scaling question in one design. See F18_PREREG.md — predictions were
frozen BEFORE this script's first run.

Two arms identical in architecture, parameter shapes, init (same seed
draws the same matrices), optimizer, budget. ONLY the attention score
differs: GEO uses -beta*d(Q,K) on the (super)fidelity angle; DOT uses
scaled dot-product.

Exit code: 1 if any harness selftest (H1-H5) fails or P0 declares the
instrument uninformative; 0 otherwise. P1-P3 outcomes are scientific
results either way and are printed as verdicts, failures kept.

Dynamics are noiseless, so the oracle ceiling is exactly 0 by
construction; the informative brackets are the two floors. Stated here
so the 0.000000 line below is not mistaken for a measurement.
"""
import sys
import numpy as np
from numpy.linalg import norm, eigvalsh, qr

from backprop import (T, add, matmul, tanh, softmax_rows,
                      bures_scores, bures_mean_loss, _dd_dF)

# =================================================================
# Generalized (cap, denom) differentiable ops
#   n=1: cap=1, denom=2  -> exact single-qubit Bures fidelity
#   n=2: cap=3, denom=4  -> superfidelity G (lucid2q convention)
# =================================================================

def gen_scores(Q, K, beta, cap, denom, causal=True):
    q, k = Q.data, K.data
    dots = q @ np.swapaxes(k, -1, -2)
    mq = cap - np.sum(q * q, axis=-1)
    mk = cap - np.sum(k * k, axis=-1)
    u = mq[:, :, None] * mk[:, None, :]
    upos = np.maximum(0, u)
    s_term = np.sqrt(upos)
    F = np.clip((1 + dots + s_term) / denom, 0, 1)
    D = np.arccos(np.sqrt(F))
    b, s, _ = D.shape
    cmask = (np.triu(np.ones((s, s), dtype=bool), k=1)[None] if causal
             else np.zeros((1, s, s), dtype=bool))
    scores = np.where(cmask, -1e9, -beta * D)
    out = T(scores, (Q, K))
    inv_s = np.where(s_term > 1e-12,
                     1.0 / np.where(s_term > 1e-12, s_term, 1.0), 0.0)
    def bw(g):
        gD = np.where(cmask, 0.0, -beta * g)
        gF = gD * _dd_dF(F)
        if Q.requires_grad:
            t1 = (gF @ k) / denom
            w = gF * np.where(u > 0, mk[:, None, :] * inv_s, 0.0)
            t2 = -(np.sum(w, axis=-1, keepdims=True) / denom) * q
            Q._acc(t1 + t2)
        if K.requires_grad:
            gFT = np.swapaxes(gF, -1, -2)
            t1 = (gFT @ q) / denom
            w = gFT * np.where(np.swapaxes(u, -1, -2) > 0,
                               mq[:, None, :] * np.swapaxes(inv_s, -1, -2),
                               0.0)
            t2 = -(np.sum(w, axis=-1, keepdims=True) / denom) * k
            K._acc(t1 + t2)
    out._bw = bw
    return out


def dot_scores(Q, K, causal=True):
    q, k = Q.data, K.data
    d = q.shape[-1]
    raw = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d)
    b, s, _ = raw.shape
    cmask = (np.triu(np.ones((s, s), dtype=bool), k=1)[None] if causal
             else np.zeros((1, s, s), dtype=bool))
    scores = np.where(cmask, -1e9, raw)
    out = T(scores, (Q, K))
    def bw(g):
        gm = np.where(cmask, 0.0, g) / np.sqrt(d)
        if Q.requires_grad:
            Q._acc(gm @ k)
        if K.requires_grad:
            K._acc(np.swapaxes(gm, -1, -2) @ q)
    out._bw = bw
    return out


def gen_mean_loss(pred, target, cap, denom):
    p, t = pred.data, target
    dots = np.sum(p * t, axis=-1)
    mp = cap - np.sum(p * p, axis=-1)
    mt = cap - np.sum(t * t, axis=-1)
    u = mp * mt
    upos = np.maximum(0, u)
    s_term = np.sqrt(upos)
    F = np.clip((1 + dots + s_term) / denom, 0, 1)
    D = np.arccos(np.sqrt(F))
    out = T(np.mean(D ** 2), (pred,))
    inv_s = np.where(s_term > 1e-12,
                     1.0 / np.where(s_term > 1e-12, s_term, 1.0), 0.0)
    n = D.size
    def bw(g):
        gF = (g * 2.0 * D / n) * _dd_dF(F)
        dF = (t - np.where((u > 0), mt * inv_s, 0.0)[..., None] * p) / denom
        pred._acc(gF[..., None] * dF)
    out._bw = bw
    return out


def gen_pcap(a, cap):
    """Project to |x| <= sqrt(cap). Generalizes backprop.pbloch (cap=1)."""
    c = np.sqrt(cap)
    mag = norm(a.data, axis=-1, keepdims=True)
    mask = mag > c
    safe = np.where(mag > 1e-12, mag, 1.0)
    u = a.data / safe
    y = np.where(mask, c * u, a.data)
    out = T(y, (a,))
    def bw(g):
        gp = c * (g - np.sum(g * u, axis=-1, keepdims=True) * u) / safe
        a._acc(np.where(mask, gp, g))
    out._bw = bw
    return out


def loss_np(p, t, cap, denom):
    """Numpy-only mean d^2 for floors/oracle (no autodiff)."""
    dots = np.sum(p * t, axis=-1)
    mp = np.maximum(0, cap - np.sum(p * p, axis=-1))
    mt = np.maximum(0, cap - np.sum(t * t, axis=-1))
    F = np.clip((1 + dots + np.sqrt(mp * mt)) / denom, 0, 1)
    return float(np.mean(np.arccos(np.sqrt(F)) ** 2))


# =================================================================
# Physical channels: c -> gamma * R_U c  (unitary conjugation in the
# Pauli basis composed with depolarizing; CP by construction)
# =================================================================

s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI1 = np.array([sx, sy, sz])                       # (3,2,2)
PAULI2 = np.array([np.kron(a, b) for a in (s0, sx, sy, sz)
                   for b in (s0, sx, sy, sz)][1:])    # (15,4,4)


def haar_unitary(rng, d):
    z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    q, r = qr(z)
    return q * (np.diagonal(r) / np.abs(np.diagonal(r)))


def channel(rng, dim):
    P, dH = (PAULI1, 2) if dim == 3 else (PAULI2, 4)
    U = haar_unitary(rng, dH)
    R = np.real(np.einsum('aij,jk,bkl,li->ab',
                          P, U, P, U.conj().T)) / dH
    gamma = rng.uniform(0.7, 0.95)
    return gamma * R


def rand_state(rng, dim):
    P, dH = (PAULI1, 2) if dim == 3 else (PAULI2, 4)
    A = rng.standard_normal((dH, dH)) + 1j * rng.standard_normal((dH, dH))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    return np.real(np.einsum('kl,alk->a', rho, P))


def make_batch(rng, nb, seq, dim, M_fixed=None):
    """Original registered task: fresh channel per trajectory (M_fixed
    None). Amendment 1 task: one shared channel per seed (M_fixed set);
    holdout = unseen initial states of the same channel."""
    X = np.zeros((nb, seq, dim))
    Y = np.zeros((nb, seq, dim))
    Ms = []
    for i in range(nb):
        M = channel(rng, dim) if M_fixed is None else M_fixed
        x = rand_state(rng, dim)
        traj = [x]
        for _ in range(seq):
            traj.append(M @ traj[-1])
        traj = np.array(traj)
        X[i], Y[i] = traj[:-1], traj[1:]
        Ms.append(M)
    return X, Y, Ms


def choi_min_eig(M):
    """Choi min eigenvalue of the unital map c -> M c (n=2 only)."""
    def apply(Xmat):
        c0 = np.trace(Xmat)
        x = np.einsum('kl,alk->a', Xmat, PAULI2)
        y = M @ np.real(x) + 1j * (M @ np.imag(x))
        return c0 * np.eye(4, dtype=complex) / 4 + \
            np.einsum('a,akl->kl', y, PAULI2) / 4
    C = np.zeros((16, 16), dtype=complex)
    for i in range(4):
        for j in range(4):
            E = np.zeros((4, 4), dtype=complex)
            E[i, j] = 1
            C[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = apply(E)
    return float(eigvalsh(C).min().real)


# =================================================================
# Model: one causal attention layer + FFN + head, arms differ ONLY
# in the score function
# =================================================================

class Arm:
    def __init__(self, dim, seed, geo, cap, denom, beta=2.0):
        rng = np.random.default_rng(seed)      # SAME seed both arms
        h = 2 * dim
        def p(*shape):
            return T(rng.normal(0, 0.1, shape))
        self.Wq, self.Wk, self.Wv, self.Wo = (p(dim, dim) for _ in range(4))
        self.W1, self.b1 = p(dim, h), p(h)
        self.W2, self.b2 = p(h, dim), p(dim)
        self.Wout, self.bout = p(dim, dim), p(dim)
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo, self.W1,
                       self.b1, self.W2, self.b2, self.Wout, self.bout]
        self.geo, self.cap, self.denom, self.beta = geo, cap, denom, beta

    def forward(self, Xnp):
        x = T(Xnp, requires_grad=False)
        Q, K, V = (matmul(x, W) for W in (self.Wq, self.Wk, self.Wv))
        S = (gen_scores(Q, K, self.beta, self.cap, self.denom)
             if self.geo else dot_scores(Q, K))
        A = softmax_rows(S)
        o = matmul(matmul(A, V), self.Wo)
        x1 = gen_pcap(add(T(Xnp, requires_grad=False), o), self.cap)
        hdn = tanh(add(matmul(x1, self.W1), self.b1))
        f = add(matmul(hdn, self.W2), self.b2)
        x2 = gen_pcap(add(x1, f), self.cap)
        return gen_pcap(add(matmul(x2, self.Wout), self.bout), self.cap)

    def step(self, Xnp, Ynp, lr):
        for pr in self.params:
            pr.grad = None
        L = gen_mean_loss(self.forward(Xnp), Ynp, self.cap, self.denom)
        L.backward()
        for pr in self.params:
            pr.data -= lr * pr.grad
        return float(L.data)

    def eval_loss(self, Xnp, Ynp):
        return loss_np(self.forward(Xnp).data, Ynp, self.cap, self.denom)


# =================================================================
# Harness selftests H1-H5 — must pass before results mean anything
# =================================================================

def selftests():
    ok = True
    rng = np.random.default_rng(0)
    q = rng.normal(0, 0.4, (2, 5, 3))
    k = rng.normal(0, 0.4, (2, 5, 3))
    t = rng.normal(0, 0.4, (2, 5, 3))

    # H1: gen_scores == backprop.bures_scores at (1,2), fwd + bwd
    Qa, Ka = T(q.copy()), T(k.copy())
    Qb, Kb = T(q.copy()), T(k.copy())
    Sa = gen_scores(Qa, Ka, 2.0, 1, 2)
    Sb = bures_scores(Qb, Kb, 2.0)
    fwd = np.max(np.abs(Sa.data - Sb.data))
    g = rng.normal(0, 1, Sa.data.shape)
    Sa._bw(g); Sb._bw(g)
    bwd = max(np.max(np.abs(Qa.grad - Qb.grad)),
              np.max(np.abs(Ka.grad - Kb.grad)))
    h1 = fwd <= 1e-10 and bwd <= 1e-10
    ok &= h1
    print(f"[{'PASS' if h1 else 'FAIL'}] H1 gen_scores == bures_scores "
          f"at (1,2): fwd {fwd:.2e}, bwd {bwd:.2e}")

    # H2: gen_mean_loss == bures_mean_loss at n=1
    Pa, Pb = T(q.copy()), T(q.copy())
    La = gen_mean_loss(Pa, t, 1, 2)
    Lb = bures_mean_loss(Pb, t)
    La._bw(np.ones(())); Lb._bw(np.ones(()))
    fwd = abs(float(La.data) - float(Lb.data))
    bwd = np.max(np.abs(Pa.grad - Pb.grad))
    h2 = fwd <= 1e-10 and bwd <= 1e-10
    ok &= h2
    print(f"[{'PASS' if h2 else 'FAIL'}] H2 gen_mean_loss == "
          f"bures_mean_loss at n=1: fwd {fwd:.2e}, bwd {bwd:.2e}")

    # H3: end-to-end analytic grad vs central finite differences
    worst = 0.0
    for dim, cap, denom in ((3, 1, 2), (15, 3, 4)):
        for geo in (True, False):
            rngb = np.random.default_rng(7)
            X, Y, _ = make_batch(rngb, 4, 5, dim)
            arm = Arm(dim, seed=11, geo=geo, cap=cap, denom=denom)
            for pr in arm.params:
                pr.grad = None
            L = gen_mean_loss(arm.forward(X), Y, cap, denom)
            L.backward()
            pr = arm.Wq
            idx = [(0, 0), (dim - 1, dim - 1), (0, dim - 1)]
            eps = 1e-6
            for (i, j) in idx:
                keep = pr.data[i, j]
                pr.data[i, j] = keep + eps
                lp = float(gen_mean_loss(arm.forward(X), Y,
                                         cap, denom).data)
                pr.data[i, j] = keep - eps
                lm = float(gen_mean_loss(arm.forward(X), Y,
                                         cap, denom).data)
                pr.data[i, j] = keep
                fd = (lp - lm) / (2 * eps)
                worst = max(worst, abs(fd - pr.grad[i, j]))
    h3 = worst <= 1e-3
    ok &= h3
    print(f"[{'PASS' if h3 else 'FAIL'}] H3 analytic vs finite-diff "
          f"gradients, both dims/arms: worst {worst:.2e} (tol 1e-3)")

    # H4: generated n=2 channels are physical (Choi PSD)
    rngc = np.random.default_rng(21)
    mins = [choi_min_eig(channel(rngc, 15)) for _ in range(20)]
    h4 = min(mins) >= -1e-9
    ok &= h4
    print(f"[{'PASS' if h4 else 'FAIL'}] H4 Choi min-eig over 20 random "
          f"n=2 channels: {min(mins):.2e} (bound -1e-9)")

    # H6: generalized F vs superfidelity from raw density matrices (n=2)
    # — the check H1/H2 cannot perform, added after the (cap+...)/denom
    # bug: at n=1 cap==1 masked the wrong additive constant entirely.
    rngf = np.random.default_rng(33)
    worst6 = 0.0
    for _ in range(50):
        c1, c2 = rand_state(rngf, 15), rand_state(rngf, 15)
        rho1 = np.eye(4, dtype=complex) / 4 + \
            np.einsum('a,akl->kl', c1, PAULI2) / 4
        rho2 = np.eye(4, dtype=complex) / 4 + \
            np.einsum('a,akl->kl', c2, PAULI2) / 4
        g_true = np.real(np.trace(rho1 @ rho2)) + np.sqrt(
            max(0, 1 - np.real(np.trace(rho1 @ rho1)))
            * max(0, 1 - np.real(np.trace(rho2 @ rho2))))
        g_form = (1 + c1 @ c2 + np.sqrt(
            max(0, 3 - c1 @ c1) * max(0, 3 - c2 @ c2))) / 4
        worst6 = max(worst6, abs(g_true - g_form))
    h6 = worst6 <= 1e-10
    ok &= h6
    print(f"[{'PASS' if h6 else 'FAIL'}] H6 generalized F vs density-"
          f"matrix superfidelity at n=2: worst {worst6:.2e}")

    # H5: fixed seed => bitwise-identical short run
    def short():
        rngd = np.random.default_rng(5)
        arm = Arm(3, seed=5, geo=True, cap=1, denom=2)
        for _ in range(20):
            X, Y, _ = make_batch(rngd, 4, 5, 3)
            L = arm.step(X, Y, 0.05)
        return L
    h5 = short() == short()
    ok &= h5
    print(f"[{'PASS' if h5 else 'FAIL'}] H5 determinism: identical "
          f"seeds give bitwise-identical final loss")
    return ok


# =================================================================
# The experiment
# =================================================================

SEQ, STEPS, BS, LR, SEEDS, NHOLD = 6, 1500, 8, 0.05, 5, 64


def run_dim(dim, cap, denom):
    print(f"\n----- n={'1' if dim == 3 else '2'}  (dim {dim}) -----")
    gaps = []
    wins = 0
    p0_arm_beats_mean = True
    p0_any_beats_last = False
    for seed in range(SEEDS):
        rho = np.random.default_rng(1000 + seed)
        geo = Arm(dim, seed=seed, geo=True, cap=cap, denom=denom)
        dot = Arm(dim, seed=seed, geo=False, cap=cap, denom=denom)
        M_seed = channel(rho, dim)          # Amendment 1: one channel/seed
        # identical data stream for both arms
        batches = [make_batch(rho, BS, SEQ, dim, M_seed)[:2]
                   for _ in range(STEPS)]
        for X, Y in batches:
            geo.step(X, Y, LR)
        for X, Y in batches:
            dot.step(X, Y, LR)
        rhold = np.random.default_rng(9000 + seed)
        Xh, Yh, Mh = make_batch(rhold, NHOLD, SEQ, dim, M_seed)
        lg = geo.eval_loss(Xh, Yh)
        ld = dot.eval_loss(Xh, Yh)
        fmean = loss_np(np.zeros_like(Yh), Yh, cap, denom)
        flast = loss_np(Xh, Yh, cap, denom)
        oracle = loss_np(np.array([[M @ x for x in Xh[i]]
                                   for i, M in enumerate(Mh)]),
                         Yh, cap, denom)
        gap = (ld - lg) / ld
        gaps.append(gap)
        wins += gap > 0.02
        p0_arm_beats_mean &= (lg < fmean and ld < fmean)
        p0_any_beats_last |= (lg < flast or ld < flast)
        print(f"seed {seed}: GEO {lg:.6f} | DOT {ld:.6f} | "
              f"floor-mean {fmean:.6f} | floor-last {flast:.6f} | "
              f"oracle {oracle:.6f} | rel gap {gap:+.4f}")
    mg = float(np.mean(gaps))
    print(f"mean relative gap (DOT-GEO)/DOT: {mg:+.4f} | "
          f"seeds with gap > 2%: {wins}/{SEEDS}")
    return {"gaps": gaps, "mean_gap": mg, "wins": wins,
            "p0_mean": p0_arm_beats_mean, "p0_last": p0_any_beats_last}


def main():
    print("=" * 66)
    print("F18 harness selftests")
    print("=" * 66)
    if not selftests():
        print("\nHARNESS SELFTEST FAILED — no experimental claim is made.")
        sys.exit(1)

    print("\n" + "=" * 66)
    print(f"F18 experiment: {SEEDS} seeds x 2 arms x 2 dims, "
          f"{STEPS} steps, batch {BS}, seq {SEQ}, lr {LR}")
    print("=" * 66)
    r1 = run_dim(3, 1, 2)
    r2 = run_dim(15, 3, 4)

    print("\n" + "=" * 66)
    print("Registered verdicts (see F18_PREREG.md)")
    print("=" * 66)
    p0 = (r1["p0_mean"] and r2["p0_mean"]
          and r1["p0_last"] and r2["p0_last"])
    print(f"[{'PASS' if p0 else 'FAIL'}] P0 anti-vacuity: both arms beat "
          f"floor-mean and some arm beats floor-last, both dims")
    if not p0:
        print("\nP0 FAILED — instrument uninformative; no architecture "
              "claim may be made in either direction.")
        sys.exit(1)
    p1 = r1["wins"] >= 4
    p2 = r2["wins"] >= 4
    p3 = r2["mean_gap"] >= r1["mean_gap"]
    print(f"[{'PASS' if p1 else 'FAIL'}] P1 n=1: GEO > DOT by >2% in "
          f">=4/5 seeds (got {r1['wins']}/5, mean gap "
          f"{r1['mean_gap']:+.4f})")
    print(f"[{'PASS' if p2 else 'FAIL'}] P2 n=2: same criterion (got "
          f"{r2['wins']}/5, mean gap {r2['mean_gap']:+.4f})")
    print(f"[{'PASS' if p3 else 'FAIL'}] P3 scaling: n=2 mean gap >= n=1 "
          f"mean gap ({r2['mean_gap']:+.4f} vs {r1['mean_gap']:+.4f})")
    print("\nP4 (n=3, 63-dim) deliberately left unrun — the door.")
    print("Failures above, if any, are KEPT results, not errors.")
    sys.exit(0)


if __name__ == "__main__":
    main()
