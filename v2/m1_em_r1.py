"""R1 REMATCH (n=1) — EM dictionary + aggregate cold-start.
F6 left R1 OPEN: gated mixture 0.0695 vs single channel 0.0566, lost
entirely at position 0. F9 gave EM; F5 gave the aggregate channel.
HARD PART: the n=1 ensemble is CONTINUOUS (random axis/omega/gamma), so
K atoms must QUANTIZE a continuum — this can legitimately fail.
REGISTERED:
  R1-a  EM-gated + aggregate cold-start beats single-channel 0.0566
  R1-b  position-0 error ~ single channel's 0.074 (cold start fixed)
  R1-c  every learned atom CPTP-certified (Choi >= -0.02)
Floor: single trained channel. Ceiling: oracle true channel."""
import numpy as np
from quasar import Generator
from quantum_geometric_transformer import project_bloch, bures_distance
from backprop import T, matmul, add, pbloch, bures_mean_loss
from m3_articulation import batch_with_physics

seq, N, K, ROUNDS, RESTARTS = 6, 500, 12, 10, 8

def choi_min_eig(M):
    """1-qubit Choi from Bloch map (no affine part)."""
    s = [np.eye(2, dtype=complex),
         np.array([[0,1],[1,0]], dtype=complex),
         np.array([[0,-1j],[1j,0]], dtype=complex),
         np.array([[1,0],[0,-1]], dtype=complex)]
    J = np.zeros((4,4), dtype=complex)
    for k in range(2):
        for l in range(2):
            E = np.zeros((2,2), dtype=complex); E[k,l] = 1
            cE = np.array([np.trace(E @ s[a]) for a in (1,2,3)])
            tr = 1.0 if k == l else 0.0
            out = tr*np.eye(2)/2 + sum((M @ cE)[a]*s[a+1] for a in range(3))/2
            J[2*k:2*k+2, 2*l:2*l+2] = out
    return float(np.min(np.linalg.eigvalsh((J + J.conj().T)/2)))

def so3(a, w):
    Kx = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3) + np.sin(w)*Kx + (1-np.cos(w))*(Kx@Kx)

def d2(a, b): return bures_distance(a, b)**2
def ridge(x, y, lam=0.3):
    return y.T @ x @ np.linalg.inv(x.T @ x + lam*np.eye(3))

# ---- data: IDENTICAL generators to F5/F6 ----
gen = Generator(seed=11)
X = np.zeros((N, seq, 3)); Y = np.zeros((N, seq, 3))
for i in range(N):
    x, y, _ = batch_with_physics(gen, 1, seq)
    X[i], Y[i] = x[0], y[0]
gh = Generator(seed=999); Xh, Yh, Ph = batch_with_physics(gh, 32, seq)

# ---- floor: single trained channel (F5 recipe) ----
r = np.random.default_rng(4)
M1 = T(r.normal(0,0.1,(3,3))); c1 = T(r.normal(0,0.01,3))
g2 = Generator(seed=11)
for _ in range(400):
    x, y, _ = batch_with_physics(g2, 8, seq)
    M1.grad=None; c1.grad=None
    bures_mean_loss(pbloch(add(matmul(T(x,requires_grad=False),M1),c1)), y).backward()
    M1.data -= 0.12*M1.grad; c1.data -= 0.12*c1.grad
Ms = M1.data.T; cs = c1.data
sin_pp = np.array([[d2(project_bloch(Xh[b,t] @ Ms.T + cs), Yh[b,t])
                    for t in range(seq)] for b in range(32)])

# ---- aggregate channel (F5): Bures-optimal isotropic contraction ----
M_agg = 0.925 * np.eye(3)

# ---- EM dictionary (F9 recipe: restarts + starve-split) ----
def traj_err(M, i):
    return np.mean([d2(project_bloch(M @ X[i,t]), Y[i,t]) for t in range(seq)])

def em(seed):
    rr = np.random.default_rng(seed)
    atoms = [ridge(X[i], Y[i]) for i in rr.choice(N, K, replace=False)]
    for _ in range(ROUNDS):
        E = np.array([[traj_err(a, i) for a in atoms] for i in range(N)])
        asg = np.argmin(E, axis=1); sizes = np.bincount(asg, minlength=K)
        for k in range(K):
            if sizes[k] < max(4, N//(6*K)):
                big = int(np.argmax(sizes)); mem = np.where(asg==big)[0]
                worst = mem[int(np.argmax(E[mem,big]))]
                atoms[k] = ridge(X[worst], Y[worst])
            else:
                idx = np.where(asg==k)[0]
                atoms[k] = ridge(X[idx].reshape(-1,3), Y[idx].reshape(-1,3))
    E = np.array([[traj_err(a, i) for a in atoms] for i in range(N)])
    return atoms, float(E.min(axis=1).mean())

atoms, tot = min((em(s) for s in range(RESTARTS)), key=lambda t: t[1])
print(f"[EM] K={K}, best-of-{RESTARTS} restarts, assignment err {tot:.5f}")

# ---- evaluate: causal evidence gate, AGGREGATE at cold start ----
mix_pp = np.zeros((32, seq)); ora_pp = np.zeros((32, seq))
for b in range(32):
    Mt = np.exp(-Ph[b,4]) * so3(Ph[b,:3], Ph[b,3])
    cum = np.zeros(K)
    for t in range(seq):
        if t == 0:
            pred = project_bloch(M_agg @ Xh[b,0])       # no evidence yet
        else:
            for k in range(K):
                cum[k] += d2(project_bloch(atoms[k] @ Xh[b,t-1]), Xh[b,t])
            pred = project_bloch(atoms[int(np.argmin(cum))] @ Xh[b,t])
        mix_pp[b,t] = d2(pred, Yh[b,t])
        ora_pp[b,t] = d2(project_bloch(Mt @ Xh[b,t]), Yh[b,t])

choi = [choi_min_eig(a) for a in atoms]
print(f"\nholdout mean Bures^2: single {sin_pp.mean():.4f} | "
      f"EM+agg MIXTURE {mix_pp.mean():.4f} | oracle {ora_pp.mean():.4f}")
print("by pos  single :", " ".join(f"{v:.3f}" for v in sin_pp.mean(0)))
print("by pos  mixture:", " ".join(f"{v:.3f}" for v in mix_pp.mean(0)))
print(f"atom Choi min-eigs: {' '.join(f'{c:+.3f}' for c in choi)}")
print(f"\nR1-a beats single-channel floor : {mix_pp.mean() < sin_pp.mean()}  "
      f"({mix_pp.mean():.4f} vs {sin_pp.mean():.4f})")
print(f"R1-b cold start fixed (t0 <= 1.1x single) : "
      f"{mix_pp.mean(0)[0] <= 1.1*sin_pp.mean(0)[0]}  "
      f"({mix_pp.mean(0)[0]:.3f} vs {sin_pp.mean(0)[0]:.3f})")
print(f"R1-c all atoms CPTP (>= -0.02)  : {min(choi) >= -0.02}  "
      f"(worst {min(choi):+.3f})")

# ================= R1-d: PHYSICALITY AS A REGULARIZER =================
# Project each atom onto the CP cone: build Choi J(M) (affine in M),
# clip its negative eigenvalues, map back by least-squares. Registered:
#   R1-d  all atoms CPTP after projection AND holdout does not worsen.
print("\n" + "="*58 + "\nR1-d: CP-projection of atoms\n" + "="*58)
sg = [np.eye(2, dtype=complex),
      np.array([[0,1],[1,0]], dtype=complex),
      np.array([[0,-1j],[1j,0]], dtype=complex),
      np.array([[1,0],[0,-1]], dtype=complex)]

def choi_of(M):
    J = np.zeros((4,4), dtype=complex)
    for k in range(2):
        for l in range(2):
            E = np.zeros((2,2), dtype=complex); E[k,l] = 1
            cE = np.array([np.trace(E @ sg[a]) for a in (1,2,3)])
            tr = 1.0 if k == l else 0.0
            out = tr*np.eye(2)/2 + sum((M @ cE)[a]*sg[a+1] for a in range(3))/2
            J[2*k:2*k+2, 2*l:2*l+2] = out
    return J

J0 = choi_of(np.zeros((3,3)))                      # constant part
basis = []                                          # dJ/dM_ab
for a in range(3):
    for b in range(3):
        Eab = np.zeros((3,3)); Eab[a,b] = 1
        basis.append(choi_of(Eab) - J0)
Amat = np.stack([np.concatenate([B.real.ravel(), B.imag.ravel()])
                 for B in basis], axis=1)          # (32, 9)

def cp_project(M, iters=20):
    for _ in range(iters):
        J = choi_of(M)
        w, V = np.linalg.eigh((J + J.conj().T)/2)
        if w.min() >= -1e-12:
            break
        Jp = V @ np.diag(np.clip(w, 0, None)) @ V.conj().T
        tgt = np.concatenate([(Jp - J0).real.ravel(), (Jp - J0).imag.ravel()])
        M = np.linalg.lstsq(Amat, tgt, rcond=None)[0].reshape(3,3)
    return M

atoms_cp = [cp_project(a) for a in atoms]
choi_cp = [choi_min_eig(a) for a in atoms_cp]
mix_cp = np.zeros((32, seq))
for b in range(32):
    cum = np.zeros(K)
    for t in range(seq):
        if t == 0:
            pred = project_bloch(M_agg @ Xh[b,0])
        else:
            for k in range(K):
                cum[k] += d2(project_bloch(atoms_cp[k] @ Xh[b,t-1]), Xh[b,t])
            pred = project_bloch(atoms_cp[int(np.argmin(cum))] @ Xh[b,t])
        mix_cp[b,t] = d2(pred, Yh[b,t])
print(f"atom Choi AFTER projection: {' '.join(f'{c:+.3f}' for c in choi_cp)}")
print(f"holdout: raw-EM {mix_pp.mean():.4f} | CP-projected "
      f"{mix_cp.mean():.4f} | single {sin_pp.mean():.4f}")
print(f"\nR1-d all atoms CPTP after projection : {min(choi_cp) >= -1e-6}  "
      f"(worst {min(choi_cp):+.6f})")
print(f"     holdout not worsened (<= raw+0.002): "
      f"{mix_cp.mean() <= mix_pp.mean() + 0.002}  "
      f"(delta {mix_cp.mean()-mix_pp.mean():+.5f})")
print(f"     PHYSICALITY AS REGULARIZER (strictly better): "
      f"{mix_cp.mean() < mix_pp.mean()}")
