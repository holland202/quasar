"""
N2' — WHY error-driven curricula fail, and what fixes them.
===========================================================
N2 refuted the C3 prediction: with a learner that CAN specialize
(competence spread 0.56), error-proportional sampling was 9.45% WORSE
than uniform. Hypothesis: ERROR != LEARNABILITY. The hardest bin carries
large IRREDUCIBLE error; error-proportional sampling pours the budget
into it and starves the bins where learning is achievable.
REGISTERED:
  N2'-0  ANTI-VACUITY / mechanism check: measure LEARNABILITY per bin
         (error reduction from init to convergence under uniform training).
         If learnability is proportional to error, the hypothesis is dead.
  N2'-a  A PROGRESS-proportional curriculum (weights ~ recent error
         REDUCTION, not error level) beats uniform by > 1% relative.
  N2'-b  It also beats the error-proportional curriculum.
Same learner, same budget, same init, same holdout as N2.
"""
import os
import numpy as np
import lucid2q as L2
from backprop import T, matmul, bures_mean_loss
from n2_capacity import (seq, STEPS, BS, LR, NBINS, BINS, HOLD, make,
                         bin_err, sb2, train)

print("\n" + "="*70)
print("N2' — learnability vs error; progress-driven curriculum")
print("="*70)

# ---- N2'-0: measure LEARNABILITY per bin ----
r = np.random.default_rng(7)
M0 = r.normal(0, 0.05, (15, 15)).T
e_init = bin_err(M0)
Mu, _, _ = train(adaptive=False)
e_final = bin_err(Mu)
learn = e_init - e_final                      # absolute error reduction
print("\nN2'-0  per-bin: initial error -> final error  (LEARNABILITY)")
for i, (lo, hi) in enumerate(BINS):
    print(f"   omega [{lo:.2f},{hi:.2f}]  {e_init[i]:.4f} -> {e_final[i]:.4f}"
          f"   learned {learn[i]:+.4f}   ({100*learn[i]/e_init[i]:5.1f}% of "
          f"its error was reducible)")
corr = np.corrcoef(e_final, learn)[0, 1]
print(f"   correlation(final error, learnability) = {corr:+.3f}")
print(f"   -> {'ERROR TRACKS LEARNABILITY (hypothesis dead)' if corr > 0.8 else 'ERROR DOES NOT TRACK LEARNABILITY: high-error bins are NOT the most learnable. Error-proportional sampling misallocates.'}")

# ---- N2'-a: progress-proportional curriculum ----
def train_progress(seed=7):
    rr = np.random.default_rng(seed)
    M = T(rr.normal(0, 0.05, (15, 15)))
    w = np.ones(NBINS) / NBINS
    prev = bin_err(M.data.T)
    for s in range(STEPS):
        if s % 200 == 0 and s > 0:
            cur = bin_err(M.data.T)
            prog = np.clip(prev - cur, 0, None)     # RECENT progress
            w = (prog + 1e-4) / (prog + 1e-4).sum() # sample where we're LEARNING
            prev = cur
        k = int(np.random.default_rng(1000 + s).choice(NBINS, p=w))
        x, y = make(BS, *BINS[k])
        M.grad = None
        bures_mean_loss(matmul(T(x, requires_grad=False), M), y).backward()
        M.data -= LR * M.grad
    return M.data.T, w

Mp, wp = train_progress()
Ma, _, wa = train(adaptive=True)
eu, ea, ep = bin_err(Mu).mean(), bin_err(Ma).mean(), bin_err(Mp).mean()
print(f"\nmatched budget ({STEPS} steps, identical init):")
print(f"   uniform               : {eu:.5f}")
print(f"   error-proportional    : {ea:.5f}   ({100*(eu-ea)/eu:+.2f}% vs uniform)")
print(f"   PROGRESS-proportional : {ep:.5f}   ({100*(eu-ep)/eu:+.2f}% vs uniform)")
print(f"   progress weights: {np.round(wp,3)}")
print(f"   error    weights: {np.round(wa,3)}")
print(f"\nN2'-a progress beats uniform by > 1%: {(eu-ep)/eu > 0.01}  "
      f"({100*(eu-ep)/eu:+.2f}%)")
print(f"N2'-b progress beats error-proportional: {ep < ea}  "
      f"({100*(ea-ep)/ea:+.2f}%)")
np.save(os.path.join(os.environ.get("TMPDIR", "/tmp"), "n2b.npy"), np.array([e_init, e_final, learn, wa, wp,
                                  [eu, ea, ep, 0, 0]]))
