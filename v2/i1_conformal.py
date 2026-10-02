"""VERA I1 — Conformal certificates on LUCID predictions.
Split conformal, per-position (handles the position-dependent error the
in-context gate creates). Nonconformity = Bures distance. Registered:
  I1-a  90% nominal -> per-position empirical coverage in [87%, 93%]
        on 500 fresh unseen-Hamiltonian trajectories.
  I1-b  Certified radius SHRINKS with position (in-context inference,
        now with a guarantee attached).
  I1-c  Under distribution shift (w in [1.3, 2.0], beyond training
        range [0.05, 1.2]) coverage drops below 85% — the certificate
        can fail, hence is informative (anti-vacuity, per F1 rule),
        and doubles as an out-of-distribution alarm."""
import numpy as np
from quasar import Generator
from quantum_geometric_transformer import bures_distance
from backprop import T, matmul, add, pbloch, bures_mean_loss, softmax_rows
from m3_articulation import batch_with_physics
from lucid_v1 import mul, stack_last, select_last, bures_sq

seq, K, beta_g, lr = 6, 6, 30.0, 0.1
A = np.zeros((seq, seq-1))
for t in range(1, seq): A[t, :t] = 1.0/t
A_T = T(A.T, requires_grad=False)
rng = np.random.default_rng(9)
atoms = [T(rng.normal(0, 0.4, (3,3))) for _ in range(K)]
prior = T(np.zeros(K))

def forward(x):
    xT = T(x, requires_grad=False)
    preds = [pbloch(matmul(xT, Mk)) for Mk in atoms]
    errs = [bures_sq(T(p.data[:, :-1, :], (p,),
                       lambda g, p=p: p._acc(np.pad(g,((0,0),(0,1),(0,0))))),
                     x[:, 1:, :]) for p in preds]
    cums = [matmul(e, A_T) for e in errs]
    logits = add(stack_last([mul(c, T(np.array(-beta_g), requires_grad=False))
                             for c in cums]), prior)
    G = softmax_rows(logits)
    yhat = None
    for k in range(K):
        term = mul(select_last(G, k), preds[k])
        yhat = term if yhat is None else add(yhat, term)
    return yhat

print("[train] LUCID v1.1 mixture (600 steps)...")
gen = Generator(seed=11)
for _ in range(600):
    x, y, _ = batch_with_physics(gen, 8, seq)
    for p in atoms + [prior]: p.grad = None
    bures_mean_loss(forward(x), y).backward()
    for p in atoms + [prior]:
        if p.grad is not None: p.data -= lr * p.grad

def scores(gsrc, n, sampler=None):
    if sampler is None:
        X, Y, _ = batch_with_physics(gsrc, n, seq)
    else:
        X = np.zeros((n, seq, 3)); Y = np.zeros((n, seq, 3))
        for i in range(n):
            ax, w, gam = sampler()
            tr = gsrc.trajectory(seq, ax, w, gam)
            X[i], Y[i] = tr[:-1], tr[1:]
    P = forward(X).data
    return np.array([[bures_distance(P[b,t], Y[b,t]) for t in range(seq)]
                     for b in range(len(X))])

# --- calibrate (per position) --------------------------------------
S_cal = scores(Generator(seed=313), 300)
n = S_cal.shape[0]
k = int(np.ceil((n + 1) * 0.9))
q = np.sort(S_cal, axis=0)[k - 1]          # per-position 90% radius

# --- in-distribution test -------------------------------------------
S_in = scores(Generator(seed=2027), 500)
cov_in = (S_in <= q[None]).mean(axis=0)

# --- shifted test (w beyond training range) --------------------------
gsh = Generator(seed=404)
def shifted():
    ax = gsh.rng.standard_normal(3); ax /= np.linalg.norm(ax)
    return ax, gsh.rng.uniform(1.3, 2.0), gsh.rng.uniform(0.0, 0.12)
S_sh = scores(gsh, 500, shifted)
cov_sh = (S_sh <= q[None]).mean(axis=0)

print("\npos              :", "     ".join(str(t) for t in range(seq)))
print("certified radius :", " ".join(f"{v:.3f}" for v in q))
print("coverage in-dist :", " ".join(f"{v:.3f}" for v in cov_in))
print("coverage shifted :", " ".join(f"{v:.3f}" for v in cov_sh))
print(f"\nI1-a all in-dist coverage in [0.87,0.93] : "
      f"{bool(np.all((cov_in>=0.87)&(cov_in<=0.93)))}  "
      f"(mean {cov_in.mean():.3f})")
print(f"I1-b radius shrinks pos0 -> pos5         : {q[5] < q[0]}  "
      f"({q[0]:.3f} -> {q[5]:.3f})")
print(f"I1-c shifted coverage < 0.85 (informative): "
      f"{cov_sh.mean() < 0.85}  (mean {cov_sh.mean():.3f})")
