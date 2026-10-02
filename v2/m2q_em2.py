"""LUCID-2Q v1e — EM + restarts + split-merge. Registered: DR-1/2/3 pass."""
import numpy as np
import lucid2q as L2
from m2q_mixture import sb2

seq, K, N = 6, 8, 400
rng = np.random.default_rng(77)
Mtrue = [L2.make_channel(True, gamma=0.06)[0] for _ in range(K)]

def make(n):
    X = np.zeros((n, seq, 15)); Y = np.zeros((n, seq, 15)); js = []
    for i in range(n):
        j = rng.integers(0, K)
        tr = L2.trajectory(Mtrue[j], seq, product_init=False)
        X[i], Y[i] = tr[:-1], tr[1:]; js.append(j)
    return X, Y, np.array(js)

X, Y, jtrue = make(N)
Xf = X.reshape(N, seq, 15); Yf = Y.reshape(N, seq, 15)

def ridge(x, y, lam=0.5):
    return y.T @ x @ np.linalg.inv(x.T @ x + lam*np.eye(15))

def errs_matrix(atoms):
    E = np.zeros((N, len(atoms)))
    for k, M in enumerate(atoms):
        P = np.einsum('ab,ntb->nta', M, Xf)
        for i in range(N):
            E[i,k] = np.mean([sb2(P[i,t], Yf[i,t]) for t in range(seq)])
    return E

def em(seed, rounds=10):
    r = np.random.default_rng(seed)
    atoms = [ridge(X[i], Y[i]) for i in r.choice(N, K, replace=False)]
    for _ in range(rounds):
        E = errs_matrix(atoms)
        assign = np.argmin(E, axis=1)
        sizes = np.bincount(assign, minlength=K)
        for k in range(K):
            idx = np.where(assign == k)[0]
            if sizes[k] < 8:                     # starving: SPLIT the fattest
                big = int(np.argmax(sizes))
                members = np.where(assign == big)[0]
                worst = members[int(np.argmax(E[members, big]))]
                atoms[k] = ridge(X[worst], Y[worst])
            else:
                atoms[k] = ridge(X[idx].reshape(-1,15), Y[idx].reshape(-1,15))
    E = errs_matrix(atoms)
    assign = np.argmin(E, axis=1)
    return atoms, assign, float(E[np.arange(N), assign].mean())

best = min((em(s) for s in range(10)), key=lambda t: t[2])
atoms, assign, tot = best
print(f"best-of-10 restarts: total assignment error {tot:.5f}")

rec = [min(np.linalg.norm(Lk - Mt) for Lk in atoms) for Mt in Mtrue]
amap = [int(np.argmin([np.linalg.norm(Lk - Mt) for Mt in Mtrue]))
        for Lk in atoms]
acc = float(np.mean([amap[assign[i]] == jtrue[i] for i in range(N)]))
choi = [L2.choi_min_eig(a, np.zeros(15)) for a in atoms]

Xh, Yh, jh = make(64)
M_ls = ridge(X.reshape(-1,15), Y.reshape(-1,15))
mix_e, sin_e = [], []
for b in range(64):
    cum = np.zeros(K)
    for t in range(seq):
        if t > 0:
            for k in range(K):
                cum[k] += sb2(atoms[k] @ Xh[b,t-1], Xh[b,t])
        khat = int(np.argmin(cum)) if t > 0 else 0
        mix_e.append(sb2(atoms[khat] @ Xh[b,t], Yh[b,t]))
        sin_e.append(sb2(M_ls @ Xh[b,t], Yh[b,t]))
mix, sin = np.mean(mix_e), np.mean(sin_e)

print(f"recovery per channel:", " ".join(f"{v:.2f}" for v in rec))
print(f"atom Choi min-eigs  :", " ".join(f"{c:+.3f}" for c in choi))
print(f"holdout: global-LS {sin:.4f} | EM-gated {mix:.4f} | oracle 0")
print(f"\nDR-1 EM-gated beats global-LS: {mix < sin}  ({mix:.4f} vs {sin:.4f})")
print(f"DR-2 dictionary recovered    : {np.mean(rec) < 1.0}  "
      f"(mean {np.mean(rec):.3f})")
print(f"DR-3 assignment acc >= 0.90  : {acc >= 0.90}  (acc {acc:.3f})")
