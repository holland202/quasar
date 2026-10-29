"""
F24 — LEAK-FREE CANDIDATES + TEMPERATURE-MATCHED SCORING
==========================================================
Registered 2026-08-07, before running. Correction pass on F21/F22/F23
after two defects were caught in review of the F22 code.

DEFECT 1 (data leakage). F22's disjoint_maps built chunks while
lo+CHUNK <= t. With CHUNK=2 the most recent chunk covers transitions
(t-2)->(t-1) and (t-1)->(t) -- and (t-1)->(t) is exactly the transition
used to SCORE candidate quality. Verified by trace: at t=4,6,8 the
scored transition sits inside a candidate's own training window; only
t=5,7 were clean. 3 of 5 query points per trajectory scored partly on
memorization. Fix: require lo+CHUNK <= t-1, so no candidate's training
data contains the scored transition. (The EVALUATED transition t->t+1
was never leaked in any version.)

DEFECT 2 (temperature confound -- the larger one). F21/F22 applied the
same BETA=6.0 to two score functions on different scales: bures^2 spans
~[0,2.47], euclidean^2 spans ~[0,4.0]. So the two arms never ran at a
matched temperature, and their attention distributions had visibly
different sharpness. Across all four previously-reported runs, the arm
with LOWER entropy won -- 4 for 4:
    F21 direction  geo 0.738 / dot 0.942  -> GEO won (geo sharper)
    F21 purity     geo 1.005 / dot 1.282  -> GEO won (geo sharper)
    F22 direction  geo 0.971 / dot 0.826  -> DOT won (dot sharper)
    F22 purity     geo 0.981 / dot 0.917  -> DOT won (dot sharper)
Both the F21 "Bures wins both regimes" surprise and the F22 reversal are
therefore confounded with sharpness. Fix: calibrate BETA per arm on a
held-out set so both arms sit at the SAME mean normalized entropy.

Also raised CHUNK 2 -> 4. A 3x3 map is 9 parameters; CHUNK=2 gives 6
scalar equations, underdetermined and leaning on the ridge prior.
CHUNK=4 gives 12.

REGISTERED:
  P0  calibration must converge: both arms within 0.02 of the target
      normalized entropy on the calibration set. If not, no comparison.
  P1  anti-vacuity: best-live beats blind by >=5%.
  P2  THE TEST. With temperature matched and leakage removed, the
      geo-vs-dot relative gap should COLLAPSE. Registered threshold:
      |geo-dot| relative < 3%. Prior gaps were 9.2% (F21 direction) and
      6.1% (F22 direction). If the gap collapses, the earlier verdicts
      were sharpness artifacts and neither metric is doing work. If a
      gap >=3% survives, the metric choice is real after all and F21 or
      F22 was measuring something genuine.
"""
import numpy as np

def so3(axis, w):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3) + np.sin(w)*K + (1-np.cos(w))*(K @ K)

def project_bloch(r):
    n = np.linalg.norm(r, axis=-1, keepdims=True)
    return np.where(n > 1, r / np.maximum(n, 1e-12), r)

def bures_distance(r1, r2):
    dot = np.sum(r1*r2, axis=-1)
    m1 = np.linalg.norm(r1, axis=-1); m2 = np.linalg.norm(r2, axis=-1)
    F = 0.5*(1 + dot + np.sqrt(np.maximum(0, (1-m1**2)*(1-m2**2))))
    return np.arccos(np.sqrt(np.clip(F, 0, 1)))

def softmax(x):
    e = np.exp(x - np.max(x)); return e / e.sum()

def entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-np.sum(p*np.log(p)))

K, SEQ_LEN, CHUNK, LAM = 6, 20, 4, 0.3
N_TRIALS, SEEDS, TARGET_H = 40, 5, 0.60   # target normalized entropy

def make_channels(regime, seed=999):
    rng = np.random.default_rng(seed); chans = []
    if regime == "direction":
        for k in range(K):
            ax = rng.standard_normal(3); ax /= np.linalg.norm(ax)
            chans.append((ax, rng.uniform(0.5, 1.2), 0.05))
    else:
        for g in np.linspace(0.02, 0.30, K):
            chans.append((np.array([0.,0.,1.]), 0.8, g))
    return chans

def trajectory(rng, axis, w, g, n):
    r = rng.standard_normal(3); r /= np.linalg.norm(r)
    R = so3(axis, w); decay = np.exp(-g)
    tr = np.zeros((n+1,3)); tr[0] = r
    for t in range(n):
        r = decay*(R@r); tr[t+1] = project_bloch(r)
    return tr

def ridge_M(X, Y, lam=LAM):
    return Y.T @ X @ np.linalg.inv(X.T @ X + lam*np.eye(3))

def candidates(traj, t):
    """LEAK-FREE: no chunk may contain the scored transition (t-1)->(t)."""
    Ms, lo = [], 0
    while lo + CHUNK <= t - 1:
        Ms.append(ridge_M(traj[lo:lo+CHUNK], traj[lo+1:lo+CHUNK+1]))
        lo += CHUNK
    return Ms

def raw_score(M, x_prev, x_now, arm):
    pred = project_bloch(M @ x_prev)
    if arm == "geo":
        return -float(bures_distance(pred, x_now)**2)
    return -float(np.sum((pred - x_now)**2))

def gather(regime, seed, n_trials):
    chans = make_channels(regime); rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_trials):
        k = rng.integers(0, K)
        traj = trajectory(rng, *chans[k], SEQ_LEN)
        for t in range(2*CHUNK+1, SEQ_LEN):
            c = candidates(traj, t)
            if len(c) >= 2:
                out.append((c, traj[t-1], traj[t], traj[t+1]))
    return out

def mean_norm_entropy(beta, arm, data):
    vals = []
    for c, xp, xn, _ in data:
        raw = np.array([raw_score(M, xp, xn, arm) for M in c])
        vals.append(entropy(softmax(beta*raw))/np.log(len(c)))
    return float(np.mean(vals))

def calibrate(arm, data, target=TARGET_H):
    lo, hi = 1e-3, 1e5
    for _ in range(60):
        mid = np.sqrt(lo*hi)
        if mean_norm_entropy(mid, arm, data) > target: lo = mid
        else: hi = mid
    beta = np.sqrt(lo*hi)
    return beta, mean_norm_entropy(beta, arm, data)

print("="*76)
print("F24 -- leak-free candidates + temperature-matched scoring (registered)")
print("="*76)
for regime in ("direction", "purity"):
    cal = gather(regime, 7777, 30)
    b_geo, h_geo = calibrate("geo", cal)
    b_dot, h_dot = calibrate("dot", cal)
    print(f"\n--- {regime} regime ---")
    print(f"  calibration (target normalized entropy {TARGET_H:.2f}): "
          f"BETA_geo={b_geo:.3f} -> H={h_geo:.4f} | BETA_dot={b_dot:.3f} -> H={h_dot:.4f}")
    p0 = abs(h_geo-TARGET_H) < 0.02 and abs(h_dot-TARGET_H) < 0.02
    print(f"  P0 calibration converged (both within 0.02 of target): {p0}")
    if not p0:
        print("     -> NO COMPARISON MADE."); continue
    g_all, d_all, b_all = [], [], []
    for s in range(SEEDS):
        data = gather(regime, 1000+s, N_TRIALS)
        ge, de, bl = [], [], []
        for c, xp, xn, xnext in data:
            Ms = np.stack(c)
            rg = np.array([raw_score(M, xp, xn, "geo") for M in c])
            rd = np.array([raw_score(M, xp, xn, "dot") for M in c])
            Mg = np.tensordot(softmax(b_geo*rg), Ms, axes=(0,0))
            Md = np.tensordot(softmax(b_dot*rd), Ms, axes=(0,0))
            Mb = Ms.mean(axis=0)
            for arr, M in ((ge,Mg),(de,Md),(bl,Mb)):
                arr.append(bures_distance(project_bloch(M@xn), xnext)**2)
        g_all.append(np.mean(ge)); d_all.append(np.mean(de)); b_all.append(np.mean(bl))
    g, d, b = np.array(g_all), np.array(d_all), np.array(b_all)
    print(f"  blind : {b.mean():.6f} +- {b.std():.6f}")
    print(f"  geo   : {g.mean():.6f} +- {g.std():.6f}")
    print(f"  dot   : {d.mean():.6f} +- {d.std():.6f}")
    margin = (b.mean() - min(g.mean(), d.mean()))/b.mean()
    print(f"  P1 anti-vacuity (best-live beats blind >=5%): {margin>=0.05}  ({100*margin:+.1f}%)")
    gap = abs(g.mean()-d.mean())/min(g.mean(), d.mean())
    winner = "geo" if g.mean() < d.mean() else "dot"
    print(f"  P2 geo-vs-dot relative gap: {100*gap:.2f}%  (registered: collapses below 3%)")
    print(f"     -> {'GAP COLLAPSED: earlier verdicts were sharpness artifacts' if gap < 0.03 else f'GAP SURVIVES ({winner} ahead): metric choice is real'}")
