"""
N2 — THE CAPACITY RETEST: does C3 flip at n = 2?
=================================================
C3 (public, v0.1): adaptive error-driven curriculum TIED with uniform
sampling (-0.3%). Diagnosis (F4/C3): a 135-param learner has FLAT
COMPETENCE across difficulty, so error-proportional resampling has no
gradient to exploit. That diagnosis made a FALSIFIABLE PREDICTION:
    "adaptive self-direction wins once the learner can specialize."
Here we test it on the n=2 substrate (225-param channel learner, 15-dim).

ANTI-VACUITY GUARD (F1 rule), REGISTERED FIRST:
  N2-0  Competence must NOT be flat: the spread of per-bin error must
        materially exceed the C3 regime's. If competence is flat again,
        adaptive CANNOT win and the experiment is UNINFORMATIVE — we
        report that, and no curriculum claim may be made either way.
REGISTERED:
  N2-a  At matched budget, adaptive beats uniform by > 1% relative on a
        uniformly-drawn holdout.
  N2-b  Direction of the C3 prediction is confirmed (adaptive >= uniform).
Floor: uniform sampling at equal budget. Both arms: identical init, same
number of gradient steps, same batch size, same optimizer.
"""
import os
import numpy as np
import lucid2q as L2                      # runs T1-T6 on import (verified)
from backprop import T, matmul, bures_mean_loss

seq, STEPS, BS, LR, NBINS = 6, 3000, 8, 0.08, 5
rng = np.random.default_rng(3)

def sb2(a, b):
    return float(np.arccos(np.sqrt(L2.G_fid(a, b))) ** 2)

def make(n, wlo, whi):
    """Trajectories with generator strength (difficulty) in [wlo, whi]."""
    X = np.zeros((n, seq, 15)); Y = np.zeros((n, seq, 15))
    for i in range(n):
        w = rng.uniform(wlo, whi)
        M = L2.make_channel(True, scale=w, gamma=0.06)[0]
        tr = L2.trajectory(M, seq, product_init=False)
        X[i], Y[i] = tr[:-1], tr[1:]
    return X, Y

# difficulty axis = generator strength omega; 5 bins
EDGES = np.linspace(0.3, 1.8, NBINS + 1)
BINS = [(EDGES[i], EDGES[i+1]) for i in range(NBINS)]
HOLD = [make(24, lo, hi) for lo, hi in BINS]        # uniform holdout

def bin_err(M):
    out = []
    for (Xh, Yh) in HOLD:
        out.append(np.mean([[sb2(M @ Xh[b, t], Yh[b, t]) for t in range(seq)]
                            for b in range(len(Xh))]))
    return np.array(out)

def train(adaptive, seed=7):
    r = np.random.default_rng(seed)
    M = T(r.normal(0, 0.05, (15, 15)))               # 225 params
    w = np.ones(NBINS) / NBINS                        # sampling weights
    hist = []
    for s in range(STEPS):
        if adaptive and s % 200 == 0 and s > 0:
            e = bin_err(M.data.T)
            w = e / e.sum()                           # error-driven
        k = int(np.random.default_rng(s).choice(NBINS, p=w))
        x, y = make(BS, *BINS[k])
        M.grad = None
        bures_mean_loss(matmul(T(x, requires_grad=False), M), y).backward()
        M.data -= LR * M.grad
        if s % 500 == 0:
            hist.append(bin_err(M.data.T))
    return M.data.T, np.array(hist), w

print("=" * 70)
print("N2 — capacity retest of C3 at n = 2 (225-param learner)")
print("=" * 70)

# ---- N2-0: anti-vacuity. Is competence flat? ----
Mu, hu, _ = train(adaptive=False)
eu = bin_err(Mu)
spread = eu.std() / eu.mean()
print(f"\nN2-0 ANTI-VACUITY — per-bin holdout error (uniform-trained):")
for i, (lo, hi) in enumerate(BINS):
    bar = "#" * int(60 * eu[i] / eu.max())
    print(f"   omega [{lo:.2f},{hi:.2f}]  {eu[i]:.4f}  {bar}")
print(f"   relative spread (std/mean) = {spread:.3f}")
flat = spread < 0.15
if flat:
    print("   -> COMPETENCE IS FLAT. Adaptive sampling has no gradient to "
          "exploit. TEST IS UNINFORMATIVE; no curriculum claim admissible.")
else:
    print(f"   -> COMPETENCE IS UNEVEN (spread {spread:.2f}). The learner "
          f"CAN specialize, so adaptive sampling has something to exploit.")

# ---- N2-a/b: matched-budget comparison ----
Ma, ha, wa = train(adaptive=True)
ea = bin_err(Ma)
hu_m, ha_m = eu.mean(), ea.mean()
rel = (hu_m - ha_m) / hu_m
print(f"\nmatched budget: {STEPS} steps, bs={BS}, lr={LR}, identical init")
print(f"   uniform  holdout (mean over bins): {hu_m:.5f}")
print(f"   adaptive holdout (mean over bins): {ha_m:.5f}")
print(f"   relative improvement: {100*rel:+.2f}%")
print(f"   final adaptive weights: {np.round(wa, 3)}  (uniform = "
      f"{np.round(np.ones(NBINS)/NBINS, 3)})")
print(f"   L1 shift from uniform: {np.abs(wa - 1/NBINS).sum():.3f}")
print(f"\nN2-a adaptive beats uniform by > 1% relative: {rel > 0.01}  "
      f"({100*rel:+.2f}%)")
print(f"N2-b C3 prediction direction confirmed (adaptive >= uniform): "
      f"{ha_m <= hu_m}")
print(f"\nC3 (135 params, n=1) was: TIED (-0.3%).  N2 (225 params, n=2) "
      f"is: {100*rel:+.2f}%")
np.save(os.path.join(os.environ.get("TMPDIR", "/tmp"), "n2.npy"), np.array([eu, ea, wa]))
