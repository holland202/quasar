"""LUCID v1 — mixture of quantum channels with causal Bures-evidence gating.
Atoms: K learnable Bloch maps M_k. At position t, gate weights =
softmax(-beta_g * mean_{s<t} d_B^2(M_k x_s, x_{s+1})): causal, geometric,
and itself a statement of evidence. Output = sum_k w_k * pbloch(M_k x_t)
(a convex mixture of channels IS a channel -> CPTP preserved by design).
Registered: (R1) beats single-channel floor 0.0566 on identical holdout;
(R2) mixture error DECREASES with position (in-context inference), single
channel's stays flat; (R3) bracketed by oracle (true per-trajectory channel)."""
import numpy as np
from numpy.linalg import norm
from quasar import Generator
from quantum_geometric_transformer import project_bloch, bures_distance
from backprop import T, matmul, add, pbloch, bures_mean_loss, softmax_rows, _unbroadcast
from m3_articulation import batch_with_physics

# --- three tiny new primitives ---------------------------------
def mul(a, b):
    out = T(a.data * b.data, (a, b))
    def bw(g):
        if a.requires_grad: a._acc(_unbroadcast(g * b.data, a.data.shape))
        if b.requires_grad: b._acc(_unbroadcast(g * a.data, b.data.shape))
    out._bw = bw
    return out

def stack_last(ts):
    out = T(np.stack([t.data for t in ts], axis=-1), tuple(ts))
    def bw(g):
        for k, t in enumerate(ts):
            if t.requires_grad: t._acc(g[..., k])
    out._bw = bw
    return out

def select_last(a, k):
    out = T(a.data[..., k:k+1], (a,))
    def bw(g):
        full = np.zeros_like(a.data); full[..., k:k+1] = g; a._acc(full)
    out._bw = bw
    return out

def bures_sq(pred, target):          # per-element d_B^2, (b,s)
    p, t = pred.data, target
    dots = np.sum(p*t, -1); mp = 1-np.sum(p*p,-1); mt = 1-np.sum(t*t,-1)
    u = mp*mt; st = np.sqrt(np.maximum(0,u))
    F = np.clip(0.5*(1+dots+st), 0, 1); D = np.arccos(np.sqrt(F))
    out = T(D**2, (pred,))
    inv = np.where(st>1e-12, 1.0/np.where(st>1e-12,st,1.0), 0.0)
    from backprop import _dd_dF
    def bw(g):
        gF = g*2.0*D*_dd_dF(F)
        dF = 0.5*t - 0.5*np.where(u>0, mt*inv, 0.0)[...,None]*p
        pred._acc(gF[...,None]*dF)
    out._bw = bw
    return out

# --- data: identical to F4/F5 -----------------------------------
seq = 6
gen = Generator(seed=11)
batches = [batch_with_physics(gen, 8, seq) for _ in range(600)]
gh = Generator(seed=999); Xh, Yh, Ph = batch_with_physics(gh, 32, seq)

# causal cumulative-average matrix A: (seq, seq-1), A[t,s]=1/t for s<t
A = np.zeros((seq, seq-1))
for t in range(1, seq): A[t, :t] = 1.0/t
A_T = T(A.T, requires_grad=False)

K, beta_g, lr = 6, 30.0, 0.1
rng = np.random.default_rng(9)
atoms = [T(rng.normal(0, 0.4, (3,3))) for _ in range(K)]   # store M_k^T

def forward(x):
    xT = T(x, requires_grad=False)
    preds = [pbloch(matmul(xT, Mk)) for Mk in atoms]
    errs  = [bures_sq(T(p.data[:, :-1, :], (p,),
                        lambda g, p=p: p._acc(np.pad(g,((0,0),(0,1),(0,0))))),
                      x[:, 1:, :]) for p in preds]
    cums  = [matmul(e, A_T) for e in errs]                 # (b,seq)
    logits = stack_last([mul(cum, T(np.array(-beta_g), requires_grad=False))
                         for cum in cums])                 # (b,seq,K)
    G = softmax_rows(logits)
    yhat = None
    for k in range(K):
        term = mul(select_last(G, k), preds[k])
        yhat = term if yhat is None else add(yhat, term)
    return yhat, G

print("[train] mixture, 600 steps...")
for (x, y, _) in batches:
    for Mk in atoms: Mk.grad = None
    yhat, _ = forward(x)
    L = bures_mean_loss(yhat, y)
    L.backward()
    for Mk in atoms:
        if Mk.grad is not None: Mk.data -= lr * Mk.grad

# --- evaluation ---------------------------------------------------
yhat, G = forward(Xh)
mix_pp = np.array([[bures_distance(yhat.data[b,t], Yh[b,t])**2
                    for t in range(seq)] for b in range(32)])
# single-channel arm (retrain quickly, same recipe as v0)
M1 = T(np.random.default_rng(4).normal(0,0.1,(3,3))); c1 = T(np.random.default_rng(4).normal(0,0.01,3))
for (x, y, _) in batches[:400]:
    M1.grad=None; c1.grad=None
    p = pbloch(add(matmul(T(x,requires_grad=False), M1), c1))
    bures_mean_loss(p, y).backward()
    M1.data -= 0.12*M1.grad; c1.data -= 0.12*c1.grad
ps = project_bloch(Xh @ M1.data + c1.data)
sin_pp = np.array([[bures_distance(ps[b,t], Yh[b,t])**2 for t in range(seq)]
                   for b in range(32)])
# oracle ceiling: true channel per trajectory
def so3(a, w):
    Kx = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(w)*Kx+(1-np.cos(w))*(Kx@Kx)
ora_pp = np.zeros((32, seq))
for b in range(32):
    Mt = np.exp(-Ph[b,4]) * so3(Ph[b,:3], Ph[b,3])
    po = project_bloch(Xh[b] @ Mt.T)
    ora_pp[b] = [bures_distance(po[t], Yh[b,t])**2 for t in range(seq)]

print(f"\n  holdout Bures^2:  single {sin_pp.mean():.4f} | "
      f"MIXTURE {mix_pp.mean():.4f} | oracle {ora_pp.mean():.4f}")
print(f"  R1 mixture beats single-channel floor: {mix_pp.mean() < sin_pp.mean()}")
print("  R2 error vs position (in-context signature):")
print("     pos      :", "  ".join(f"{t}" for t in range(seq)))
print("     single   :", " ".join(f"{v:.3f}" for v in sin_pp.mean(0)))
print("     mixture  :", " ".join(f"{v:.3f}" for v in mix_pp.mean(0)))
print("     oracle   :", " ".join(f"{v:.3f}" for v in ora_pp.mean(0)))
gate_ent = -(G.data*np.log(G.data+1e-12)).sum(-1).mean(0)
print("  gate entropy by pos:", " ".join(f"{v:.2f}" for v in gate_ent),
      f"(max {np.log(K):.2f})")
print("  atom singular values (tomography readout):")
for k, Mk in enumerate(atoms):
    print(f"    atom {k}: {np.linalg.svd(Mk.data.T, compute_uv=False).round(3)}")
