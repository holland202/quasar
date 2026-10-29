"""
F22 — CONTROLLED CANDIDATES + MAP-QUALITY SCORING
====================================================
Registered 2026-08-07, before running. Follow-up to F21, prompted by
Graeme's two-part critique.

NOTE ON SCOPE: Graeme's "single value per channel, controlled
arbitration" doesn't map cleanly onto this task's actual structure --
within one trajectory there is only ever one true channel, so there is
no live K-way choice for a single sequence's positions to arbitrate
over (K only varies ACROSS trials). The closest honest operationalization
of his intent: replace F21's overlapping trailing windows (every
position i gets its own window ending at i, so adjacent windows share
most of their data and are highly correlated) with disjoint
(non-overlapping) chunks -- a small set of genuinely independent local
map estimates instead of many redundant ones.

His second point is unambiguous and implemented exactly as stated:
score each candidate by MAP QUALITY -- how well it would have predicted
the most recent observed transition -- not by state similarity between
x_t and x_i. This also folds in his separate point about residual norm
as a free quality signal (that's exactly what map-quality scoring is).

TWO VARIABLES CHANGE AT ONCE (disjoint candidates AND quality scoring).
This experiment cannot cleanly isolate which one drives any shift from
F21's numbers -- only whether Bures' edge survives both changes
together. A clean 2x2 decomposition is future work, not run here.

REGISTERED:
  P0a liveness: as F21, entropy < 90% of uniform max.
  P0b anti-vacuity: best-live beats blind (uniform avg of the disjoint
      candidates) by >=5%, matched data/seeds.
  P1  does Bures' win survive this redesign? Registered as genuinely
      open -- both directions are informative. If it survives, that's
      evidence for Graeme's "score-to-weight mapping" hypothesis
      (though confounded with the candidate-set change). If it
      reverses, F21's Bures win may have depended on the specific
      overlapping-window redundancy structure, not on Bures itself.
  P2  margin vs F21's 27%/32% -- reported as a fact, not a hypothesis
      (not registered which direction, since two variables moved).
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
    e = np.exp(x - np.max(x))
    return e / e.sum()

def entropy(p):
    p = np.clip(p, 1e-12, 1)
    return float(-np.sum(p * np.log(p)))

K, SEQ_LEN, CHUNK, LAM, BETA = 6, 9, 2, 0.3, 6.0   # CHUNK=2: disjoint pairs
N_TRIALS, SEEDS = 60, 5

def make_channels(regime, seed=999):
    rng = np.random.default_rng(seed)
    chans = []
    if regime == "direction":
        for k in range(K):
            axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
            chans.append((axis, rng.uniform(0.5, 1.2), 0.05))
    else:
        gs = np.linspace(0.02, 0.30, K)
        for k in range(K):
            chans.append((np.array([0., 0., 1.]), 0.8, gs[k]))
    return chans

def trajectory(rng, axis, w, g, seq_len):
    r = rng.standard_normal(3); r /= np.linalg.norm(r)
    R = so3(axis, w); decay = np.exp(-g)
    traj = np.zeros((seq_len+1, 3)); traj[0] = r
    for t in range(seq_len):
        r = decay * (R @ r); traj[t+1] = project_bloch(r)
    return traj

def ridge_M(X, Y, lam=LAM):
    return Y.T @ X @ np.linalg.inv(X.T @ X + lam*np.eye(3))

def disjoint_maps(traj, t):
    """Non-overlapping CHUNK-sized windows within [0, t]. Each chunk ->
    one independent map estimate. Returns list of M's (no reuse of data
    across chunks, unlike F21's overlapping trailing windows)."""
    Ms = []
    lo = 0
    while lo + CHUNK <= t:
        Xw = traj[lo:lo+CHUNK]; Yw = traj[lo+1:lo+CHUNK+1]
        Ms.append(ridge_M(Xw, Yw))
        lo += CHUNK
    return Ms

def quality_score(M, x_prev, x_now, metric):
    pred = project_bloch(M @ x_prev)
    if metric == "geo":
        return -BETA * bures_distance(pred, x_now)**2
    else:
        return -BETA * float(np.sum((pred - x_now)**2))

def run_regime(regime, seed):
    chans = make_channels(regime, seed=999)
    rng = np.random.default_rng(1000 + seed)
    errs = {"geo": [], "dot": [], "blind": []}
    ent_geo, ent_dot = [], []
    for _ in range(N_TRIALS):
        k = rng.integers(0, K)
        traj = trajectory(rng, *chans[k], SEQ_LEN)
        for t in range(2*CHUNK, SEQ_LEN):       # need >=2 disjoint chunks
            cands = disjoint_maps(traj, t)
            if len(cands) < 2:
                continue
            x_prev, x_now, x_next = traj[t-1], traj[t], traj[t+1]
            geo_s = np.array([quality_score(M, x_prev, x_now, "geo") for M in cands])
            dot_s = np.array([quality_score(M, x_prev, x_now, "dot") for M in cands])
            wg, wd = softmax(geo_s), softmax(dot_s)
            ent_geo.append(entropy(wg)); ent_dot.append(entropy(wd))
            Mstack = np.stack(cands)
            Mg = np.tensordot(wg, Mstack, axes=(0,0))
            Md = np.tensordot(wd, Mstack, axes=(0,0))
            Mb = Mstack.mean(axis=0)
            for arm, M in (("geo", Mg), ("dot", Md), ("blind", Mb)):
                pred = project_bloch(M @ x_now)
                errs[arm].append(bures_distance(pred, x_next)**2)
    out = {a: float(np.mean(v)) for a, v in errs.items()}
    out["n"] = len(errs["geo"])
    out["ent_geo"] = float(np.mean(ent_geo)) if ent_geo else float('nan')
    out["ent_dot"] = float(np.mean(ent_dot)) if ent_dot else float('nan')
    out["ent_max"] = float(np.log(max(2, (SEQ_LEN-2*CHUNK)//CHUNK + 2)))
    return out

print("="*74)
print("F22 -- disjoint candidates + map-quality scoring, registered before running")
print("="*74)
F21_REF = {"direction": 27.0, "purity": 32.4}
for regime in ("direction", "purity"):
    rows = [run_regime(regime, s) for s in range(SEEDS)]
    geo = np.array([r["geo"] for r in rows]); dot = np.array([r["dot"] for r in rows])
    bl = np.array([r["blind"] for r in rows])
    eg = np.nanmean([r["ent_geo"] for r in rows]); ed = np.nanmean([r["ent_dot"] for r in rows])
    emax = rows[0]["ent_max"]
    print(f"\n--- {regime} regime ({SEEDS} seeds, ~{rows[0]['n']} samples/seed) ---")
    print(f"  blind (uniform avg)  : {bl.mean():.6f} +- {bl.std():.6f}")
    print(f"  geo (map-quality)    : {geo.mean():.6f} +- {geo.std():.6f}")
    print(f"  dot (map-quality)    : {dot.mean():.6f} +- {dot.std():.6f}")
    print(f"  attention entropy: geo {eg:.3f}, dot {ed:.3f}  (uniform max ~{emax:.3f})")
    p0a = (eg < 0.9*emax) and (ed < 0.9*emax)
    best_live = min(geo.mean(), dot.mean())
    margin = (bl.mean() - best_live) / bl.mean() if bl.mean() > 0 else 0
    p0b = margin >= 0.05
    print(f"  P0a liveness: {p0a}")
    print(f"  P0b anti-vacuity (best-live beats blind by >=5%): {p0b}  ({100*margin:+.1f}%)")
    if p0b:
        verdict = "GEO WINS" if geo.mean() < dot.mean()-1e-6 else ("DOT WINS" if dot.mean() < geo.mean()-1e-6 else "TIE")
        print(f"  P1 geo vs dot: {verdict}  (geo {geo.mean():.6f} vs dot {dot.mean():.6f})")
        print(f"  P2 margin vs F21 ({F21_REF[regime]:.1f}%): {100*margin:.1f}% "
              f"({'larger' if 100*margin>F21_REF[regime] else 'smaller'})")
    else:
        print("  P1/P2 not evaluated -- P0b failed, no claim in either direction")
