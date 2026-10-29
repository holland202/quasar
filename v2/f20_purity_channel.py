#!/usr/bin/env python3
"""
F20 — Bures vs dot-product where the geometry CAN differ
========================================================
See F20_PREREG.md; predictions frozen before this script first ran.

Two measurements motivated this design:
  (1) attention in F18/F19 was DEAD -- entropy 99.9-100.0% of uniform max,
      caused by projection scale collapse (|Q| ~ 0.057 at d_model=3).
  (2) at equal |q|,|k| the Bures score is a strictly monotone function of
      the dot product (Kendall tau = 1.000000), so the ONLY term carrying
      extra information is sqrt((1-|q|^2)(1-|k|^2)) -- a PURITY channel.

So: fix the scale defect, gate on live attention (P0b), and split the
mixture into a DIRECTION regime (channels differ by rotation axis) and a
PURITY regime (channels differ by contraction gamma). Purity decay is the
channel signature only in the second.

Exit 1 if any harness selftest fails, or if P0a/P0b void the run.
P1-P3 outcomes are results either way; failures are kept.
"""
import sys
import numpy as np
from numpy.linalg import norm, eigvalsh

from backprop import T, add, matmul, tanh, softmax_rows, _dd_dF

CAP = DENOM = None  # unit ball, exact Bures at d=3: (1 + q.k + s)/2


# ------------------------------------------------------------- ops
def gen_scores(Q, K, beta, causal=True):
    q, k = Q.data, K.data
    dots = q @ np.swapaxes(k, -1, -2)
    mq = 1 - np.sum(q * q, axis=-1)
    mk = 1 - np.sum(k * k, axis=-1)
    u = mq[:, :, None] * mk[:, None, :]
    s_term = np.sqrt(np.maximum(0, u))
    F = np.clip((1 + dots + s_term) / 2, 0, 1)
    D = np.arccos(np.sqrt(F))
    b, s, _ = D.shape
    cmask = (np.triu(np.ones((s, s), dtype=bool), k=1)[None] if causal
             else np.zeros((1, s, s), dtype=bool))
    out = T(np.where(cmask, -1e9, -beta * D), (Q, K))
    inv_s = np.where(s_term > 1e-12,
                     1.0 / np.where(s_term > 1e-12, s_term, 1.0), 0.0)

    def bw(g):
        gF = np.where(cmask, 0.0, -beta * g) * _dd_dF(F)
        if Q.requires_grad:
            t1 = (gF @ k) / 2
            w = gF * np.where(u > 0, mk[:, None, :] * inv_s, 0.0)
            Q._acc(t1 - (np.sum(w, axis=-1, keepdims=True) / 2) * q)
        if K.requires_grad:
            gFT = np.swapaxes(gF, -1, -2)
            t1 = (gFT @ q) / 2
            w = gFT * np.where(np.swapaxes(u, -1, -2) > 0,
                               mq[:, None, :] * np.swapaxes(inv_s, -1, -2),
                               0.0)
            K._acc(t1 - (np.sum(w, axis=-1, keepdims=True) / 2) * k)
    out._bw = bw
    return out


def dot_scores(Q, K, beta, causal=True):
    q, k = Q.data, K.data
    raw = beta * (q @ np.swapaxes(k, -1, -2))
    b, s, _ = raw.shape
    cmask = (np.triu(np.ones((s, s), dtype=bool), k=1)[None] if causal
             else np.zeros((1, s, s), dtype=bool))
    out = T(np.where(cmask, -1e9, raw), (Q, K))

    def bw(g):
        gm = np.where(cmask, 0.0, g) * beta
        if Q.requires_grad:
            Q._acc(gm @ k)
        if K.requires_grad:
            K._acc(np.swapaxes(gm, -1, -2) @ q)
    out._bw = bw
    return out


def mean_loss(pred, target):
    p, t = pred.data, target
    dots = np.sum(p * t, axis=-1)
    mp = 1 - np.sum(p * p, axis=-1)
    mt = 1 - np.sum(t * t, axis=-1)
    u = mp * mt
    s_term = np.sqrt(np.maximum(0, u))
    F = np.clip((1 + dots + s_term) / 2, 0, 1)
    D = np.arccos(np.sqrt(F))
    out = T(np.mean(D ** 2), (pred,))
    inv_s = np.where(s_term > 1e-12,
                     1.0 / np.where(s_term > 1e-12, s_term, 1.0), 0.0)
    n = D.size

    def bw(g):
        gF = (g * 2.0 * D / n) * _dd_dF(F)
        dF = (t - np.where(u > 0, mt * inv_s, 0.0)[..., None] * p) / 2
        pred._acc(gF[..., None] * dF)
    out._bw = bw
    return out


def pball(a, r=1.0):
    mag = norm(a.data, axis=-1, keepdims=True)
    mask = mag > r
    safe = np.where(mag > 1e-12, mag, 1.0)
    u = a.data / safe
    out = T(np.where(mask, r * u, a.data), (a,))

    def bw(g):
        gp = r * (g - np.sum(g * u, axis=-1, keepdims=True) * u) / safe
        a._acc(np.where(mask, gp, g))
    out._bw = bw
    return out


def loss_np(p, t):
    dots = np.sum(p * t, axis=-1)
    mp = np.maximum(0, 1 - np.sum(p * p, axis=-1))
    mt = np.maximum(0, 1 - np.sum(t * t, axis=-1))
    F = np.clip((1 + dots + np.sqrt(mp * mt)) / 2, 0, 1)
    return float(np.mean(np.arccos(np.sqrt(F)) ** 2))


# --------------------------------------------------------- physics
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
P3 = np.array([sx, sy, sz])


def rot(axis, th):
    a = np.asarray(axis, float)
    a = a / norm(a)
    U = (np.cos(th / 2) * np.eye(2, dtype=complex)
         - 1j * np.sin(th / 2) * (a[0] * sx + a[1] * sy + a[2] * sz))
    return np.real(np.einsum('aij,jk,bkl,li->ab',
                             P3, U, P3, U.conj().T)) / 2


def dictionary(rng, K, regime):
    """DIRECTION: shared gamma, different axes.
       PURITY:    shared axis, different gamma."""
    out = []
    if regime == "direction":
        g = 0.90
        for _ in range(K):
            v = rng.standard_normal(3)
            out.append(g * rot(v, rng.uniform(0.6, 2.4)))
    else:
        axis, th = np.array([0.0, 0.0, 1.0]), 1.2
        R = rot(axis, th)
        for j in range(K):
            out.append((0.70 + 0.28 * j / max(1, K - 1)) * R)
    return out


def rand_state(rng):
    A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = A @ A.conj().T
    rho /= np.trace(rho).real
    return np.real(np.einsum('kl,alk->a', rho, P3))


def choi_min_eig(M):
    def apply(X):
        c0 = np.trace(X)
        xr = np.real(np.einsum('kl,alk->a', X, P3))
        xi = np.imag(np.einsum('kl,alk->a', X, P3))
        y = M @ xr + 1j * (M @ xi)
        return c0 * np.eye(2, dtype=complex) / 2 + \
            np.einsum('a,akl->kl', y, P3) / 2
    C = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2), dtype=complex)
            E[i, j] = 1
            C[i::2, j::2] = apply(E)
    return float(eigvalsh(C).min().real)


def make_batch(rng, nb, seq, dic):
    X = np.zeros((nb, seq, 3))
    Y = np.zeros((nb, seq, 3))
    for i in range(nb):
        M = dic[rng.integers(0, len(dic))]
        x = rand_state(rng)
        traj = [x]
        for _ in range(seq):
            traj.append(M @ traj[-1])
        traj = np.array(traj)
        X[i], Y[i] = traj[:-1], traj[1:]
    return X, Y


# ----------------------------------------------------------- model
QK_SCALE = 3.0   # fixes the F19 scale collapse: O(1) scores at init


class Arm:
    def __init__(self, seed, mode, beta=6.0):
        rng = np.random.default_rng(seed)
        d, h = 3, 12

        def p(*shape, s=0.1):
            return T(rng.normal(0, s, shape))
        # projections deliberately larger so scores start with real range
        self.Wq, self.Wk = p(d, d, s=QK_SCALE), p(d, d, s=QK_SCALE)
        self.Wv, self.Wo = p(d, d), p(d, d)
        self.W1, self.b1 = p(d, h), p(h)
        self.W2, self.b2 = p(h, d), p(d)
        self.Wout, self.bout = p(d, d), p(d)
        self.beta = T(np.array(float(beta)))       # learnable temperature
        self.params = [self.Wq, self.Wk, self.Wv, self.Wo, self.W1,
                       self.b1, self.W2, self.b2, self.Wout, self.bout]
        self.mode = mode

    def _qk(self, Xnp):
        x = T(Xnp, requires_grad=False)
        Q = pball(matmul(x, self.Wq), r=0.999)
        K = pball(matmul(x, self.Wk), r=0.999)
        return x, Q, K

    def scores(self, Xnp):
        _, Q, K = self._qk(Xnp)
        b = float(self.beta.data)
        return (gen_scores(Q, K, b) if self.mode == "geo"
                else dot_scores(Q, K, b))

    def forward(self, Xnp):
        x = T(Xnp, requires_grad=False)
        if self.mode == "blind":
            o = matmul(matmul(x, self.Wv), self.Wo)
        else:
            _, Q, K = self._qk(Xnp)
            V = matmul(x, self.Wv)
            b = float(self.beta.data)
            S = (gen_scores(Q, K, b) if self.mode == "geo"
                 else dot_scores(Q, K, b))
            o = matmul(matmul(softmax_rows(S), V), self.Wo)
        x1 = pball(add(T(Xnp, requires_grad=False), o))
        hdn = tanh(add(matmul(x1, self.W1), self.b1))
        x2 = pball(add(x1, add(matmul(hdn, self.W2), self.b2)))
        return pball(add(matmul(x2, self.Wout), self.bout))

    def step(self, Xnp, Ynp, lr):
        for pr in self.params:
            pr.grad = None
        L = mean_loss(self.forward(Xnp), Ynp)
        L.backward()
        for pr in self.params:
            if pr.grad is not None:
                pr.data -= lr * pr.grad
        return float(L.data)

    def eval_loss(self, Xnp, Ynp):
        return loss_np(self.forward(Xnp).data, Ynp)


def attn_entropy(arm, Xnp):
    """Mean entropy of the final-position attention row, and the uniform
    maximum it is measured against."""
    if arm.mode == "blind":
        return None, None
    S = arm.scores(Xnp).data
    s = S.shape[-1]
    row = S[:, -1, :]
    A = np.exp(row - row.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    ent = float(np.mean(-(A * np.log(A + 1e-12)).sum(-1)))
    return ent, float(np.log(s))


def purity_spread(arm, Xnp):
    _, Q, K = arm._qk(Xnp)
    return float(np.mean(np.std(1 - np.sum(K.data ** 2, -1), axis=-1)))


# ------------------------------------------------------- selftests
def selftests():
    from scipy.stats import kendalltau
    ok = True
    rng = np.random.default_rng(0)

    # H1: at EQUAL norms, geo score is a monotone transform of dot
    taus = []
    for _ in range(200):
        v = rng.standard_normal((2, 8, 3))
        v /= norm(v, axis=-1, keepdims=True)
        q, k = v[0] * 0.5, v[1] * 0.5
        F = (1 + q @ k.T + np.sqrt(np.outer(1 - np.sum(q * q, 1),
                                            1 - np.sum(k * k, 1)))) / 2
        g = -np.arccos(np.sqrt(np.clip(F, 0, 1))).ravel()
        taus.append(kendalltau(g, (q @ k.T).ravel()).statistic)
    h1 = min(taus) > 0.99999
    ok &= h1
    print(f"[{'PASS' if h1 else 'FAIL'}] H1 equal-norm monotone "
          f"equivalence: min Kendall tau {min(taus):.6f} (derivation "
          f"encoded as a test)")

    # H2: analytic vs finite differences
    worst = 0.0
    for regime in ("direction", "purity"):
        dic = dictionary(np.random.default_rng(4), 4, regime)
        X, Y = make_batch(np.random.default_rng(7), 4, 6, dic)
        for mode in ("geo", "dot", "blind"):
            arm = Arm(11, mode)
            for pr in arm.params:
                pr.grad = None
            mean_loss(arm.forward(X), Y).backward()
            pr = arm.Wv if mode == "blind" else arm.Wq
            for (i, j) in [(0, 0), (2, 2), (0, 2)]:
                keep, eps = pr.data[i, j], 1e-6
                pr.data[i, j] = keep + eps
                lp = float(mean_loss(arm.forward(X), Y).data)
                pr.data[i, j] = keep - eps
                lm = float(mean_loss(arm.forward(X), Y).data)
                pr.data[i, j] = keep
                worst = max(worst, abs((lp - lm) / (2 * eps)
                                       - pr.grad[i, j]))
    h2 = worst <= 1e-3
    ok &= h2
    print(f"[{'PASS' if h2 else 'FAIL'}] H2 analytic vs finite-diff, "
          f"3 modes x 2 regimes: worst {worst:.2e}")

    # H3: channels physical + identity control (kept from F19)
    ident = choi_min_eig(np.eye(3))
    mins = []
    for regime in ("direction", "purity"):
        mins += [choi_min_eig(M) for M in
                 dictionary(np.random.default_rng(21), 8, regime)]
    h3 = abs(ident) <= 1e-12 and min(mins) >= -1e-9
    ok &= h3
    print(f"[{'PASS' if h3 else 'FAIL'}] H3 physical channels: identity "
          f"control {ident:.2e}, worst {min(mins):.2e}")

    # H4: determinism
    def short():
        r = np.random.default_rng(5)
        dic = dictionary(np.random.default_rng(3), 4, "purity")
        arm = Arm(5, "geo")
        for _ in range(20):
            X, Y = make_batch(r, 4, 6, dic)
            L = arm.step(X, Y, 0.05)
        return L
    h4 = short() == short()
    ok &= h4
    print(f"[{'PASS' if h4 else 'FAIL'}] H4 determinism")

    # H5: context replacement moves attention arms, not blind
    dic = dictionary(np.random.default_rng(9), 4, "purity")
    X, _ = make_batch(np.random.default_rng(13), 4, 6, dic)
    X2, _ = make_batch(np.random.default_rng(99), 4, 6, dic)
    Xp = X.copy()
    Xp[:, :5] = X2[:, :5]
    dl = {}
    for mode in ("blind", "geo", "dot"):
        arm = Arm(11, mode)
        dl[mode] = float(np.max(np.abs(arm.forward(X).data[:, -1]
                                       - arm.forward(Xp).data[:, -1])))
    h5 = dl["blind"] <= 1e-12 and min(dl["geo"], dl["dot"]) > 1e-6
    ok &= h5
    print(f"[{'PASS' if h5 else 'FAIL'}] H5 context replacement moves "
          f"geo {dl['geo']:.2e} / dot {dl['dot']:.2e}, blind "
          f"{dl['blind']:.2e}")

    # H6: the ENTROPY INSTRUMENT itself must move
    def ent_of(row):
        A = np.exp(row - row.max(-1, keepdims=True))
        A /= A.sum(-1, keepdims=True)
        return float(np.mean(-(A * np.log(A + 1e-12)).sum(-1)))
    unif = ent_of(np.zeros((4, 8)))
    onehot = ent_of(np.eye(8)[:4] * 50.0)
    h6 = abs(unif - np.log(8)) <= 1e-9 and onehot <= 1e-6
    ok &= h6
    print(f"[{'PASS' if h6 else 'FAIL'}] H6 entropy instrument: uniform "
          f"{unif:.6f} (max {np.log(8):.6f}), one-hot {onehot:.2e} — the "
          f"gate can move")
    return ok


# ------------------------------------------------------ experiment
SEQ, STEPS, BS, LR, SEEDS, NHOLD, K = 8, 2500, 16, 0.05, 5, 96, 8


def run(regime):
    print(f"\n----- regime: {regime.upper()} -----")
    gaps, wins, ents, spreads = [], 0, [], []
    blind_beaten = True
    for seed in range(SEEDS):
        rd = np.random.default_rng(700 + seed)
        dic = dictionary(rd, K, regime)
        arms = {m: Arm(seed, m) for m in ("geo", "dot", "blind")}
        batches = [make_batch(rd, BS, SEQ, dic) for _ in range(STEPS)]
        for m in ("geo", "dot", "blind"):
            for X, Y in batches:
                arms[m].step(X, Y, LR)
        Xh, Yh = make_batch(np.random.default_rng(9700 + seed),
                            NHOLD, SEQ, dic)
        lg, ld, lb = (arms[m].eval_loss(Xh, Yh)
                      for m in ("geo", "dot", "blind"))
        eg, emax = attn_entropy(arms["geo"], Xh)
        ed, _ = attn_entropy(arms["dot"], Xh)
        sp = purity_spread(arms["geo"], Xh)
        gap = (ld - lg) / ld
        gaps.append(gap)
        wins += gap > 0.02
        ents += [eg / emax, ed / emax]
        spreads.append(sp)
        blind_beaten &= (min(lg, ld) < lb * 0.95)
        print(f"seed {seed}: GEO {lg:.6f} | DOT {ld:.6f} | BLIND {lb:.6f}"
              f" | entropy geo {eg / emax:.3f} dot {ed / emax:.3f} of max"
              f" | purity spread {sp:.4f} | gap {gap:+.4f}")
    mg = float(np.mean(gaps))
    print(f"mean gap {mg:+.4f} | wins {wins}/{SEEDS} | mean entropy "
          f"{np.mean(ents):.3f} of max | mean purity spread "
          f"{np.mean(spreads):.4f}")
    return {"mean_gap": mg, "wins": wins, "blind": blind_beaten,
            "ent": float(np.mean(ents)), "gaps": gaps,
            "spreads": spreads}


def main():
    print("=" * 68)
    print("F20 harness selftests")
    print("=" * 68)
    if not selftests():
        print("\nHARNESS SELFTEST FAILED — no claim is made.")
        sys.exit(1)
    print("\n" + "=" * 68)
    print(f"F20: {SEEDS} seeds x 3 arms x 2 regimes, K={K}, {STEPS} steps")
    print("=" * 68)
    d = run("direction")
    p = run("purity")

    print("\n" + "=" * 68)
    print("Registered verdicts (see F20_PREREG.md)")
    print("=" * 68)
    p0a = d["blind"] and p["blind"]
    p0b = d["ent"] < 0.90 and p["ent"] < 0.90
    print(f"[{'PASS' if p0a else 'FAIL'}] P0a context-blind beaten by >5%"
          f" in both regimes")
    print(f"[{'PASS' if p0b else 'FAIL'}] P0b LIVE ATTENTION: mean entropy"
          f" < 90% of uniform max (direction {d['ent']:.3f}, purity "
          f"{p['ent']:.3f})")
    if not (p0a and p0b):
        print("\nRUN VOID — attention not demonstrably alive and/or not "
              "load-bearing. No metric claim is made in either "
              "direction.")
        sys.exit(1)
    p1 = abs(d["mean_gap"]) < 0.02
    p2 = p["wins"] >= 4
    allg = d["gaps"] + p["gaps"]
    alls = d["spreads"] + p["spreads"]
    corr = float(np.corrcoef(allg, alls)[0, 1])
    p3 = corr > 0
    print(f"[{'PASS' if p1 else 'FAIL'}] P1 DIRECTION regime ties "
          f"(|gap| < 2%): {d['mean_gap']:+.4f}")
    print(f"[{'PASS' if p2 else 'FAIL'}] P2 PURITY regime GEO > DOT by "
          f">2% in >=4/5 seeds: {p['wins']}/5, mean {p['mean_gap']:+.4f}")
    print(f"[{'PASS' if p3 else 'FAIL'}] P3 gap correlates with purity "
          f"spread: r = {corr:+.3f}")
    print("\nP4 (multi-head) left unrun — the door.")
    print("Failures above are KEPT results, not errors.")
    sys.exit(0)


if __name__ == "__main__":
    main()
