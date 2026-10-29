"""
F21 — TOKENS CARRY TRANSITIONS, NOT STATES
============================================
Registered 2026-08-07, before running.

F20's diagnosis: attention returns a convex combination of VALUES. When
values are raw states, no combination of them identifies the linear map
that generated the trajectory (states carry position, not relation).
This experiment changes what a value IS: each past position i carries a
LOCAL MAP ESTIMATE M_i, fit by ridge regression over a trailing window of
W consecutive transitions ending at i. Attention now averages map
estimates. If the diagnosis is right, that average should be informative
where averaging states never could be.

Query/Key scoring is unchanged from the F18-F20 line (Bures vs dot,
scored on the states x_t, x_i) so this isolates the one variable F20
named. beta is fixed (not trained) since this design is closed-form,
not gradient-trained — so the P0a liveness question is "does the
scoring function produce non-uniform weight by construction", not a
training-dynamics question.

REGISTERED:
  P0a liveness: mean attention entropy at the last query position must
      sit materially below ln(t_max) (uniform max) -- below 90% of it.
  P0b anti-vacuity: blind (uniform-averaged M_i) control must be beaten
      by >=5% relative holdout Bures^2 by the best live-attention arm,
      matched data/seeds. Below 5%, no claim in either direction.
  P1  if P0 passes: is the reduction large relative to F20's STATE-VALUE
      numbers (recorded: direction blind 0.046562 vs best-attention
      0.046524 essentially tied; purity blind 0.000908 vs 0.000890)?
      Different code, not bit-comparable -- contrast of SCALE is the
      registered signal, not a direct ratio.
  P2  GEO vs DOT: registered GEO ~= DOT in the direction regime (axis
      differences are captured by plain dot product of directions);
      GEO could beat DOT in the purity regime (Bures carries the extra
      sqrt((1-|q|^2)(1-|k|^2)) purity term aligned with contraction).
Floor: blind control. No oracle computed here -- see verdicts for why.
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

K, SEQ_LEN, W, LAM, BETA = 6, 8, 3, 0.3, 6.0
N_TRIALS, SEEDS = 60, 5

def make_channels(regime, seed=999):
    rng = np.random.default_rng(seed)
    chans = []
    if regime == "direction":
        for k in range(K):
            axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
            chans.append((axis, rng.uniform(0.5, 1.2), 0.05))
    else:  # purity
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

def local_maps(traj):
    """M_i for i=1..seq_len, ridge over trailing window ending at i."""
    Ms = []
    for i in range(1, len(traj)):
        lo = max(1, i - W + 1)
        Xw = traj[lo-1:i]; Yw = traj[lo:i+1]
        Ms.append(ridge_M(Xw, Yw))
    return Ms  # Ms[i-1] = M_i

def run_regime(regime, seed):
    chans = make_channels(regime, seed=999)
    rng = np.random.default_rng(1000 + seed)
    errs = {"geo": [], "dot": [], "blind": []}
    ent_geo, ent_dot = [], []
    for _ in range(N_TRIALS):
        k = rng.integers(0, K)
        traj = trajectory(rng, *chans[k], SEQ_LEN)
        Ms = local_maps(traj)  # Ms[i-1] for i=1..SEQ_LEN
        for t in range(W, SEQ_LEN):           # need >=W past transitions
            xt, xt1 = traj[t], traj[t+1]
            cand_i = list(range(1, t+1))       # positions with an M_i
            Mstack = np.stack([Ms[i-1] for i in cand_i])
            geo_s = np.array([-BETA*bures_distance(xt, traj[i]) for i in cand_i])
            dot_s = np.array([ BETA*np.dot(xt, traj[i]) for i in cand_i])
            wg, wd = softmax(geo_s), softmax(dot_s)
            ent_geo.append(entropy(wg)); ent_dot.append(entropy(wd))
            Mg = np.tensordot(wg, Mstack, axes=(0,0))
            Md = np.tensordot(wd, Mstack, axes=(0,0))
            Mb = Mstack.mean(axis=0)
            for arm, M in (("geo", Mg), ("dot", Md), ("blind", Mb)):
                pred = project_bloch(M @ xt)
                errs[arm].append(bures_distance(pred, xt1)**2)
    out = {a: float(np.mean(v)) for a, v in errs.items()}
    out["ent_geo"] = float(np.mean(ent_geo)); out["ent_dot"] = float(np.mean(ent_dot))
    out["ent_max"] = float(np.log(SEQ_LEN - 1))  # loosest case, t=SEQ_LEN-1
    return out

print("="*74)
print("F21 -- transition-token attention, registered before running")
print("="*74)
for regime in ("direction", "purity"):
    rows = [run_regime(regime, s) for s in range(SEEDS)]
    geo = np.array([r["geo"] for r in rows]); dot = np.array([r["dot"] for r in rows])
    bl = np.array([r["blind"] for r in rows])
    eg = np.mean([r["ent_geo"] for r in rows]); ed = np.mean([r["ent_dot"] for r in rows])
    emax = rows[0]["ent_max"]
    print(f"\n--- {regime} regime ({SEEDS} seeds, {N_TRIALS} trials/seed) ---")
    print(f"  blind (uniform M avg)   : {bl.mean():.6f} +- {bl.std():.6f}")
    print(f"  geo   (Bures-scored)    : {geo.mean():.6f} +- {geo.std():.6f}")
    print(f"  dot   (dot-scored)      : {dot.mean():.6f} +- {dot.std():.6f}")
    print(f"  attention entropy: geo {eg:.3f}, dot {ed:.3f}  (uniform max ~{emax:.3f})")
    p0a = (eg < 0.9*emax) and (ed < 0.9*emax)
    best_live = min(geo.mean(), dot.mean())
    margin = (bl.mean() - best_live) / bl.mean() if bl.mean() > 0 else 0
    p0b = margin >= 0.05
    print(f"  P0a liveness (entropy < 90% of uniform max): {p0a}")
    if not p0a:
        print("       -> WARNING: scoring is close to uniform; P0b result is suspect")
    print(f"  P0b anti-vacuity (best-live beats blind by >=5%): {p0b}  ({100*margin:+.1f}%)")
    if p0b:
        verdict = "GEO WINS" if geo.mean() < dot.mean()-1e-6 else ("DOT WINS" if dot.mean() < geo.mean()-1e-6 else "TIE")
        print(f"  P2 geo vs dot: {verdict}  (geo {geo.mean():.6f} vs dot {dot.mean():.6f})")
    else:
        print("  P2 not evaluated -- P0b failed, no claim in either direction")
