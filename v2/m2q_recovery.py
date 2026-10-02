"""LUCID-2Q v1b — DICTIONARY RECOVERY under a finite channel ensemble.
Environment = 8 fixed true channels (entangling, gamma=0.06). Train K=8
atoms with moderate gating (beta_train=10). Registered: DR-1 mixture <
single-channel holdout; DR-2 recovery: per true channel, min_k ||M_k -
M_true||_F far below random-init floor; DR-3 identification acc >= 0.80."""
import numpy as np
import lucid2q as L2
from backprop import T, matmul, add, softmax_rows
from lucid_v1 import mul, stack_last, select_last
from m2q_mixture import sb_sq, sb2

seq, K, lr, steps = 6, 8, 0.05, 4000
Mtrue = [L2.make_channel(True, gamma=0.06)[0] for _ in range(K)]

def batch(n):
    X = np.zeros((n, seq, 15)); Y = np.zeros((n, seq, 15)); js = []
    for i in range(n):
        j = np.random.default_rng().integers(0, K)
        tr = L2.trajectory(Mtrue[j], seq, product_init=False)
        X[i], Y[i] = tr[:-1], tr[1:]; js.append(j)
    return X, Y, np.array(js)

A = np.zeros((seq, seq-1))
for t in range(1, seq): A[t, :t] = 1.0/t
A_T = T(A.T, requires_grad=False)
r2 = np.random.default_rng(21)
atoms = [T(r2.normal(0, 0.05, (15,15))) for _ in range(K)]
init_atoms = [a.data.T.copy() for a in atoms]
prior = T(np.zeros(K))

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

print(f"[train] finite-ensemble mixture: K={K}, {steps} steps, beta=10")
for i in range(steps):
    x, y, _ = batch(8)
    for p in atoms + [prior]: p.grad = None
    L2.sb_mean_loss(forward(x, 10.0)[0], y).backward()
    for p in atoms + [prior]:
        if p.grad is not None: p.data -= lr * p.grad

# single-channel control, same data distribution & budget
M1 = T(np.random.default_rng(4).normal(0, 0.05, (15,15)))
for i in range(steps):
    x, y, _ = batch(8)
    M1.grad = None
    L2.sb_mean_loss(matmul(T(x, requires_grad=False), M1), y).backward()
    M1.data -= 0.1 * M1.grad
Msingle = M1.data.T

Xh, Yh, jh = batch(64)
yhat, G = forward(Xh, 60.0)                    # sharp gate at eval
mix = np.mean([[sb2(yhat.data[b,t], Yh[b,t]) for t in range(seq)]
               for b in range(64)])
sin = np.mean([[sb2(Msingle @ Xh[b,t], Yh[b,t]) for t in range(seq)]
               for b in range(64)])
ora = np.mean([[sb2(Mtrue[jh[b]] @ Xh[b,t], Yh[b,t]) for t in range(seq)]
               for b in range(64)])

# recovery: match learned atoms to true channels
learned = [a.data.T for a in atoms]
rec = [min(np.linalg.norm(Lk - Mt) for Lk in learned) for Mt in Mtrue]
floor = [min(np.linalg.norm(Ik - Mt) for Ik in init_atoms) for Mt in Mtrue]

# identification: gate's argmax atom -> nearest true channel label
amap = [int(np.argmin([np.linalg.norm(Lk - Mt) for Mt in Mtrue]))
        for Lk in learned]
pick = np.argmax(G.data[:, -1, :], axis=-1)
acc = float(np.mean([amap[pick[b]] == jh[b] for b in range(64)]))

print(f"\nholdout mean d_SB^2: single {sin:.4f} | MIXTURE {mix:.4f} | "
      f"oracle {ora:.6f}")
print(f"gate entropy final pos: {-(G.data[:,-1]*np.log(G.data[:,-1]+1e-12)).sum(-1).mean():.2f}")
print(f"recovery ||learned-true||_F per channel: "
      + " ".join(f"{v:.2f}" for v in rec) + f"  (init floor ~{np.mean(floor):.2f})")
print(f"\nDR-1 mixture beats single      : {mix < sin}  ({mix:.4f} vs {sin:.4f})")
print(f"DR-2 dictionary recovered      : {np.mean(rec) < 0.3*np.mean(floor)}  "
      f"(mean {np.mean(rec):.2f} vs floor {np.mean(floor):.2f})")
print(f"DR-3 identification acc >= 0.80: {acc >= 0.80}  (acc {acc:.3f})")
