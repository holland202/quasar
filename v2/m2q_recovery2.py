"""LUCID-2Q v1c — symmetry breaking by DATA-DRIVEN INIT.
Identical to m2q_recovery except: each atom initialized by ridge
least-squares on ONE random trajectory's transitions (the world's own
diversity seeds the dictionary). Registered: DR-1/2/3 flip to pass."""
import numpy as np
import lucid2q as L2
from backprop import T, matmul, add, softmax_rows
from lucid_v1 import mul, stack_last, select_last
from m2q_mixture import sb_sq, sb2

seq, K, lr, steps = 6, 8, 0.05, 4000
rng = np.random.default_rng(77)
Mtrue = [L2.make_channel(True, gamma=0.06)[0] for _ in range(K)]

def batch(n):
    X = np.zeros((n, seq, 15)); Y = np.zeros((n, seq, 15)); js = []
    for i in range(n):
        j = rng.integers(0, K)
        tr = L2.trajectory(Mtrue[j], seq, product_init=False)
        X[i], Y[i] = tr[:-1], tr[1:]; js.append(j)
    return X, Y, np.array(js)

# --- data-driven init: one ridge LS fit per atom, one trajectory each
def ridge_fit(x, y, lam=0.5):
    return y.T @ x @ np.linalg.inv(x.T @ x + lam*np.eye(15))
atoms = []
for k in range(K):
    x, y, _ = batch(1)
    atoms.append(T(np.ascontiguousarray(ridge_fit(x[0], y[0]).T)))
init_atoms = [a.data.T.copy() for a in atoms]
prior = T(np.zeros(K))

A = np.zeros((seq, seq-1))
for t in range(1, seq): A[t, :t] = 1.0/t
A_T = T(A.T, requires_grad=False)

def forward(x, beta):
    xT = T(x, requires_grad=False)
    preds = [matmul(xT, Mk) for Mk in atoms]
    errs = [sb_sq(T(p.data[:, :-1, :], (p,),
                    lambda g, p=p: p._acc(np.pad(g,((0,0),(0,1),(0,0))))),
                  x[:, 1:, :]) for p in preds]
    cums = [matmul(e, A_T) for e in errs]
    logits = add(stack_last([mul(c, T(np.array(-beta), requires_grad=False))
                             for c in cums]), prior)
    G = softmax_rows(logits)
    yhat = None
    for k in range(K):
        term = mul(select_last(G, k), preds[k])
        yhat = term if yhat is None else add(yhat, term)
    return yhat, G

print(f"[train] data-init mixture: K={K}, {steps} steps, beta=25")
for i in range(steps):
    x, y, _ = batch(8)
    for p in atoms + [prior]: p.grad = None
    L2.sb_mean_loss(forward(x, 25.0)[0], y).backward()
    for p in atoms + [prior]:
        if p.grad is not None: p.data -= lr * p.grad

M1 = T(np.random.default_rng(4).normal(0, 0.05, (15,15)))
for i in range(steps):
    x, y, _ = batch(8)
    M1.grad = None
    L2.sb_mean_loss(matmul(T(x, requires_grad=False), M1), y).backward()
    M1.data -= 0.1 * M1.grad
Msingle = M1.data.T

Xh, Yh, jh = batch(64)
yhat, G = forward(Xh, 60.0)
pp = lambda M: np.mean([[sb2(M(b,t), Yh[b,t]) for t in range(seq)]
                        for b in range(64)])
mix = pp(lambda b,t: yhat.data[b,t])
sin = pp(lambda b,t: Msingle @ Xh[b,t])
ora = pp(lambda b,t: Mtrue[jh[b]] @ Xh[b,t])
learned = [a.data.T for a in atoms]
rec = [min(np.linalg.norm(Lk - Mt) for Lk in learned) for Mt in Mtrue]
floor = [min(np.linalg.norm(Ik - Mt) for Ik in init_atoms) for Mt in Mtrue]
amap = [int(np.argmin([np.linalg.norm(Lk - Mt) for Mt in Mtrue]))
        for Lk in learned]
pick = np.argmax(G.data[:, -1, :], axis=-1)
acc = float(np.mean([amap[pick[b]] == jh[b] for b in range(64)]))

print(f"\nholdout mean d_SB^2: single {sin:.4f} | MIXTURE {mix:.4f} | "
      f"oracle {ora:.6f}")
print("recovery per channel:", " ".join(f"{v:.2f}" for v in rec),
      f"(init floor {np.mean(floor):.2f})")
print(f"\nDR-1 mixture beats single      : {mix < sin}  ({mix:.4f} vs {sin:.4f})")
print(f"DR-2 dictionary recovered      : {np.mean(rec) < 0.3*np.mean(floor)}  "
      f"(mean {np.mean(rec):.2f})")
print(f"DR-3 identification acc >= 0.80: {acc >= 0.80}  (acc {acc:.3f})")
