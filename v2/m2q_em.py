"""LUCID-2Q v1d — EM dictionary learning (closed-form M-step).
E-step: assign each trajectory to its best-evidence atom. M-step: ridge
refit each atom on ALL its assigned transition pairs. Registered:
DR-1 gated predictor beats GLOBAL least-squares single map;
DR-2 recovery mean ||learned-true|| < 0.3 * init floor;
DR-3 assignment accuracy >= 0.90 (matched mapping)."""
import numpy as np
import lucid2q as L2
from m2q_mixture import sb2

seq, K, N, rounds = 6, 8, 400, 8
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

def ridge(x, y, lam=0.5):           # x,y: (m,15) -> M (15,15)
    return y.T @ x @ np.linalg.inv(x.T @ x + lam*np.eye(15))

# init: single-trajectory fits (diverse but partial)
atoms = [ridge(X[i], Y[i]) for i in rng.choice(N, K, replace=False)]
init_atoms = [a.copy() for a in atoms]

def traj_err(M, i):
    return np.mean([sb2(M @ X[i,t], Y[i,t]) for t in range(seq)])

for r in range(rounds):
    # E-step
    E = np.array([[traj_err(atoms[k], i) for k in range(K)]
                  for i in range(N)])
    assign = np.argmin(E, axis=1)
    # M-step (closed form per atom)
    for k in range(K):
        idx = np.where(assign == k)[0]
        if len(idx) == 0:            # dead atom: reseed from worst-fit traj
            i = int(np.argmax(E.min(axis=1)))
            atoms[k] = ridge(X[i], Y[i]); continue
        xs = X[idx].reshape(-1, 15); ys = Y[idx].reshape(-1, 15)
        atoms[k] = ridge(xs, ys)
    sizes = np.bincount(assign, minlength=K)
    print(f"  round {r+1}: cluster sizes {sizes.tolist()}")

# --- evaluation --------------------------------------------------
rec = [min(np.linalg.norm(Lk - Mt) for Lk in atoms) for Mt in Mtrue]
floor = [min(np.linalg.norm(Ik - Mt) for Ik in init_atoms) for Mt in Mtrue]
amap = [int(np.argmin([np.linalg.norm(Lk - Mt) for Mt in Mtrue]))
        for Lk in atoms]
acc = float(np.mean([amap[assign[i]] == jtrue[i] for i in range(N)]))

# holdout: gated prediction (causal evidence, sharp) vs global-LS single
Xh, Yh, jh = make(64)
M_ls = ridge(X.reshape(-1,15), Y.reshape(-1,15))
mix_e, sin_e, ora_e = [], [], []
for b in range(64):
    cum = np.zeros(K)
    for t in range(seq):
        if t > 0:
            for k in range(K):
                cum[k] += sb2(atoms[k] @ Xh[b,t-1], Xh[b,t])
        khat = int(np.argmin(cum)) if t > 0 else 0
        mix_e.append(sb2(atoms[khat] @ Xh[b,t], Yh[b,t]))
        sin_e.append(sb2(M_ls @ Xh[b,t], Yh[b,t]))
        ora_e.append(sb2(Mtrue[jh[b]] @ Xh[b,t], Yh[b,t]))
mix, sin, ora = map(np.mean, (mix_e, sin_e, ora_e))

# tomography of one recovered atom: CPTP?
choi = [L2.choi_min_eig(a, np.zeros(15)) for a in atoms]

print(f"\nholdout mean d_SB^2: global-LS single {sin:.4f} | "
      f"EM-gated {mix:.4f} | oracle {ora:.6f}")
print("recovery per channel:", " ".join(f"{v:.2f}" for v in rec),
      f"(init floor {np.mean(floor):.2f})")
print("atom Choi min-eigs  :", " ".join(f"{c:+.3f}" for c in choi))
print(f"\nDR-1 EM-gated beats global-LS  : {mix < sin}  ({mix:.4f} vs {sin:.4f})")
print(f"DR-2 dictionary recovered      : {np.mean(rec) < 0.3*np.mean(floor)}  "
      f"(mean {np.mean(rec):.2f} vs floor {np.mean(floor):.2f})")
print(f"DR-3 assignment acc >= 0.90    : {acc >= 0.90}  (acc {acc:.3f})")
