"""LUCID-2Q v1 — TRAINED mixture of 15-dim channels, causal evidence gating.
The R1 rematch on n=2 ground. Registered:
  TR-1 trained mixture beats single SB-trained channel on holdout (mean d_SB^2)
  TR-2 mixture error decreases with position (in-context inference)
  TR-3 learned atoms individually near-CPTP (mean Choi min-eig >= -0.05)
Floor = single channel (F8's M_sb); ceiling = oracle true channel."""
import numpy as np
import lucid2q as L2                      # runs its verified suite on import
from backprop import T, matmul, add, softmax_rows
from lucid_v1 import mul, stack_last, select_last

seq, K, beta_g, lr, steps = 6, 8, 60.0, 0.05, 4000

def _dd_dF(F):
    ok = (F > 1e-9) & (F < 0.999999)
    Fs = np.where(ok, F, 0.5)
    return np.where(ok, -1.0/(2*np.sqrt(Fs)*np.sqrt(1-Fs)), 0.0)

def sb_sq(pred, target):
    """per-element d_SB^2 (b,s) with backward to pred; 15-dim c-vectors."""
    p, t = pred.data, target
    dots = np.sum(p*t, -1); mp = 3-np.sum(p*p,-1); mt = 3-np.sum(t*t,-1)
    u = mp*mt; st = np.sqrt(np.maximum(0,u))
    F = np.clip((1+dots+st)/4, 0, 1); D = np.arccos(np.sqrt(F))
    out = T(D**2, (pred,))
    inv = np.where(st>1e-12, 1.0/np.where(st>1e-12,st,1.0), 0.0)
    def bw(g):
        gF = g*2*D*_dd_dF(F)
        dF = 0.25*t - 0.25*np.where(u>0, mt*inv, 0.0)[...,None]*p
        pred._acc(gF[...,None]*dF)
    out._bw = bw
    return out

A = np.zeros((seq, seq-1))
for t in range(1, seq): A[t, :t] = 1.0/t
A_T = T(A.T, requires_grad=False)
r2 = np.random.default_rng(21)
atoms = [T(r2.normal(0, 0.05, (15,15))) for _ in range(K)]
prior = T(np.zeros(K))

def forward(x):
    xT = T(x, requires_grad=False)
    preds = [matmul(xT, Mk) for Mk in atoms]          # store M^T in atoms
    errs = [sb_sq(T(p.data[:, :-1, :], (p,),
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
    return yhat, G

print(f"[train] n=2 mixture: K={K}, {steps} steps...")
for i in range(steps):
    x, y = L2.batch(8, seq)
    for p in atoms + [prior]: p.grad = None
    L2.sb_mean_loss(*(forward(x)[:1]+ (y,))) .backward()
    for p in atoms + [prior]:
        if p.grad is not None: p.data -= lr * p.grad

# holdout with oracle channels retained
nH = 64
Xh = np.zeros((nH, seq, 15)); Yh = np.zeros((nH, seq, 15)); Mtrue = []
for i in range(nH):
    M, _ = L2.make_channel(True)
    tr = L2.trajectory(M, seq, product_init=False)
    Xh[i], Yh[i] = tr[:-1], tr[1:]; Mtrue.append(M)

def sb2(a, b):
    return float(np.arccos(np.sqrt(L2.G_fid(a, b)))**2)

yhat, G = forward(Xh)
mix_pp = np.array([[sb2(yhat.data[b,t], Yh[b,t]) for t in range(seq)]
                   for b in range(nH)])
sin_pp = np.array([[sb2(L2.M_sb @ Xh[b,t], Yh[b,t]) for t in range(seq)]
                   for b in range(nH)])
ora_pp = np.array([[sb2(Mtrue[b] @ Xh[b,t], Yh[b,t]) for t in range(seq)]
                   for b in range(nH)])
choi = [L2.choi_min_eig(Mk.data.T, np.zeros(15)) for Mk in atoms]

print(f"\nholdout mean d_SB^2: single {sin_pp.mean():.4f} | "
      f"MIXTURE {mix_pp.mean():.4f} | oracle {ora_pp.mean():.6f}")
print("error by pos  single :", " ".join(f"{v:.3f}" for v in sin_pp.mean(0)))
print("error by pos  mixture:", " ".join(f"{v:.3f}" for v in mix_pp.mean(0)))
ge = -(G.data*np.log(G.data+1e-12)).sum(-1).mean(0)
print("gate entropy by pos  :", " ".join(f"{v:.2f}" for v in ge),
      f"(max {np.log(K):.2f})")
print(f"atom Choi min-eigs   : " + " ".join(f"{c:+.3f}" for c in choi))
print(f"\nTR-1 mixture beats single : {mix_pp.mean() < sin_pp.mean()}  "
      f"({mix_pp.mean():.4f} vs {sin_pp.mean():.4f})")
print(f"TR-2 error decreases t0->t5: {mix_pp.mean(0)[5] < mix_pp.mean(0)[0]}  "
      f"({mix_pp.mean(0)[0]:.3f} -> {mix_pp.mean(0)[5]:.3f})")
print(f"TR-3 atoms near-CPTP (mean >= -0.05): {np.mean(choi) >= -0.05}  "
      f"(mean {np.mean(choi):+.3f})")
