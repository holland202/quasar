"""
F23 — BURES CONFINED TO EVALUATION, NOT SCORING
==================================================
Registered 2026-08-07, before running. Direct response to Graeme's
follow-up on F22: stop testing Bures as a scoring mechanism (F22 already
showed it losing there in both regimes). Confine it strictly to the
output/evaluation metric. Scoring becomes ONE method only -- plain
negative squared error, no geometry anywhere in the weighting. The only
remaining question: does judging the SAME reconstructions by Bures^2
instead of Euclidean^2 change how much attention beats the blind control.

Candidates unchanged from F22: disjoint (non-overlapping) chunk-based
local map estimates -- that part of the design isn't in question here.

REGISTERED:
  P0a liveness: as before, entropy < 90% of uniform max.
  P0b anti-vacuity: attention (single scoring method) must beat blind
      by >=5% on AT LEAST ONE of the two evaluation metrics for a claim
      to be made on that metric. Each metric's margin is judged on its
      own terms -- passing on Euclidean and failing on Bures (or vice
      versa) is itself the informative outcome, not a failure of the
      experiment.
  P1  does the anti-vacuity margin differ meaningfully between the two
      evaluation metrics (Bures^2 vs Euclidean^2), holding scoring
      fixed? Registered as genuinely open. If margins are close, Bures
      does not "earn its keep" as an evaluation lens either -- the
      geometry would show no measurable role anywhere in this task at
      this scale. If Bures shows a clearly larger margin, that is
      evidence for Graeme's reframing: geometry matters for judging
      quality, not for weighting candidates.
DEFERRED, not built here: the two-stage LUCID-coarse-then-attention-fine
pipeline Graeme also described. That is a separate architectural change
and needs its own registration, not folded into this one.
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

K, SEQ_LEN, CHUNK, LAM, BETA = 6, 9, 2, 0.3, 6.0
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
    Ms = []
    lo = 0
    while lo + CHUNK <= t:
        Xw = traj[lo:lo+CHUNK]; Yw = traj[lo+1:lo+CHUNK+1]
        Ms.append(ridge_M(Xw, Yw))
        lo += CHUNK
    return Ms

def run_regime(regime, seed):
    chans = make_channels(regime, seed=999)
    rng = np.random.default_rng(1000 + seed)
    errs_bures = {"live": [], "blind": []}
    errs_eucl  = {"live": [], "blind": []}
    ents = []
    for _ in range(N_TRIALS):
        k = rng.integers(0, K)
        traj = trajectory(rng, *chans[k], SEQ_LEN)
        for t in range(2*CHUNK, SEQ_LEN):
            cands = disjoint_maps(traj, t)
            if len(cands) < 2:
                continue
            x_prev, x_now, x_next = traj[t-1], traj[t], traj[t+1]
            # SCORING: plain negative squared error only. No Bures here.
            scores = np.array([
                -BETA * float(np.sum((project_bloch(M @ x_prev) - x_now)**2))
                for M in cands])
            w = softmax(scores)
            ents.append(entropy(w))
            Mstack = np.stack(cands)
            M_live = np.tensordot(w, Mstack, axes=(0,0))
            M_blind = Mstack.mean(axis=0)
            for arm, M in (("live", M_live), ("blind", M_blind)):
                pred = project_bloch(M @ x_now)
                errs_bures[arm].append(bures_distance(pred, x_next)**2)
                errs_eucl[arm].append(float(np.sum((pred - x_next)**2)))
    out = {
        "bures_live": float(np.mean(errs_bures["live"])),
        "bures_blind": float(np.mean(errs_bures["blind"])),
        "eucl_live": float(np.mean(errs_eucl["live"])),
        "eucl_blind": float(np.mean(errs_eucl["blind"])),
        "ent": float(np.mean(ents)),
        "ent_max": float(np.log(max(2, (SEQ_LEN-2*CHUNK)//CHUNK + 2))),
    }
    return out

print("="*74)
print("F23 -- Bures confined to evaluation only, registered before running")
print("="*74)
for regime in ("direction", "purity"):
    rows = [run_regime(regime, s) for s in range(SEEDS)]
    bl_l = np.array([r["bures_live"] for r in rows]); bl_b = np.array([r["bures_blind"] for r in rows])
    eu_l = np.array([r["eucl_live"] for r in rows]);  eu_b = np.array([r["eucl_blind"] for r in rows])
    ent = np.mean([r["ent"] for r in rows]); emax = rows[0]["ent_max"]
    p0a = ent < 0.9*emax
    m_bures = (bl_b.mean() - bl_l.mean()) / bl_b.mean() if bl_b.mean() > 0 else 0
    m_eucl  = (eu_b.mean() - eu_l.mean()) / eu_b.mean() if eu_b.mean() > 0 else 0
    print(f"\n--- {regime} regime ({SEEDS} seeds, ~{'?'} samples/seed) ---")
    print(f"  attention entropy: {ent:.3f}  (uniform max ~{emax:.3f})  P0a liveness: {p0a}")
    print(f"  [Bures^2 eval]     live {bl_l.mean():.6f} +- {bl_l.std():.6f}   "
          f"blind {bl_b.mean():.6f} +- {bl_b.std():.6f}   margin {100*m_bures:+.1f}%  "
          f"P0b: {m_bures >= 0.05}")
    print(f"  [Euclidean^2 eval] live {eu_l.mean():.6f} +- {eu_l.std():.6f}   "
          f"blind {eu_b.mean():.6f} +- {eu_b.std():.6f}   margin {100*m_eucl:+.1f}%  "
          f"P0b: {m_eucl >= 0.05}")
    diff = 100*(m_bures - m_eucl)
    print(f"  P1 metric-choice effect on margin: {diff:+.1f} percentage points "
          f"({'Bures shows a bigger win' if diff>1 else 'Euclidean shows a bigger win' if diff<-1 else 'no meaningful difference'})")
