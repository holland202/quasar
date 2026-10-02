"""LUCID v0 — a network layer that IS a quantum channel.
Registered predictions (before running):
  L1  Trained Bloch map M converges to the mean environment channel,
      which by symmetry is depolarizing: M ~ s*I, s = E[e^-g](1+2E[cos w])/3.
  L2  Emergent physicality: learned M is approximately CPTP (Choi min
      eigenvalue >= -eps) though nothing enforces it.
  L3  A 9-param channel layer matches the 135-param QGT on holdout
      (0.1464 from F4, identical data/seeds) — explaining F2.
Alternative for L1: trajectory state-channel correlations bias M away
from s*I toward the pooled-regression solution (computed as ceiling)."""
import numpy as np
from numpy.linalg import norm
from quasar import Generator
from quantum_geometric_transformer import project_bloch
from backprop import T, matmul, add, pbloch, bures_mean_loss
from m3_articulation import batch_with_physics

# --- closed-form prediction -------------------------------------
Eg = (1 - np.exp(-0.12)) / 0.12
Ec = (np.sin(1.2) - np.sin(0.05)) / (1.2 - 0.05)
s_pred = Eg * (1 + 2 * Ec) / 3
print(f"[theory] predicted depolarizing strength s = {s_pred:.4f}")

# --- Monte-Carlo mean channel (ground truth for L1) --------------
g = Generator(seed=51)
Ms = np.zeros((3, 3))
N = 200000
for _ in range(N):
    axis, w, gam = g.sample_physics()
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + np.sin(w)*K + (1-np.cos(w))*(K@K)
    Ms += np.exp(-gam) * R
Mbar = Ms / N
print(f"[MC]     mean channel: diag = {np.diag(Mbar).round(4)}, "
      f"offdiag max |.| = {np.abs(Mbar - np.diag(np.diag(Mbar))).max():.4f}")

# --- same data as F4 (seed 11 train, 999 holdout) -----------------
gen = Generator(seed=11)
batches = [batch_with_physics(gen, 8, 6) for _ in range(400)]
gh = Generator(seed=999)
Xh, Yh, _ = batch_with_physics(gh, 32, 6)

# pooled least-squares ceiling (what regression would give)
Xp = np.concatenate([b[0].reshape(-1,3) for b in batches])
Yp = np.concatenate([b[1].reshape(-1,3) for b in batches])
M_reg, *_ = np.linalg.lstsq(Xp, Yp, rcond=None)
M_reg = M_reg.T

# --- train the channel layer: y_pred = pbloch(M r + c) ------------
rng = np.random.default_rng(4)
MT = T(rng.normal(0, 0.1, (3, 3)))   # stores M^T
c  = T(rng.normal(0, 0.01, (3,)))
M_init = MT.data.T.copy()
for (x, y, _) in batches:
    MT.grad = None; c.grad = None
    pred = pbloch(add(matmul(T(x, requires_grad=False), MT), c))
    L = bures_mean_loss(pred, y)
    L.backward()
    MT.data -= 0.12 * MT.grad
    c.data  -= 0.12 * c.grad
M = MT.data.T

# --- holdout loss (Bures^2, same metric as qgt.loss) ---------------
from quantum_geometric_transformer import bures_distance
pred = project_bloch(Xh @ M.T + c.data)
hold = np.mean([[bures_distance(pred[b,t], Yh[b,t])**2 for t in range(6)]
                for b in range(32)])

# --- CPTP check via Choi matrix ------------------------------------
sig = [np.array([[0,1],[1,0]],complex), np.array([[0,-1j],[1j,0]]),
       np.array([[1,0],[0,-1]],complex)]
def channel(rho):
    r = np.real([np.trace(rho @ s) for s in sig])
    rp = M @ r + c.data
    out = 0.5*np.eye(2, dtype=complex)
    for i in range(3): out += 0.5*rp[i]*sig[i]
    # trace part: TP by construction of Bloch affine map
    return out * np.trace(rho).real + (np.trace(rho)==0)*0
E = [[np.zeros((2,2),complex) for _ in range(2)] for _ in range(2)]
J = np.zeros((4,4), complex)
for k in range(2):
    for l in range(2):
        Ekl = np.zeros((2,2),complex); Ekl[k,l]=1
        # affine map on non-density basis: use linearity of PTM rep
        # Phi(Ekl) via decomposition Ekl = a*I/2 + sum b_i sig_i/... do directly:
        tr = np.trace(Ekl)
        r = np.array([np.trace(Ekl @ s) for s in sig])
        rp = M @ r + c.data * tr
        out = 0.5*tr*np.eye(2,dtype=complex)
        for i in range(3): out += 0.5*rp[i]*sig[i]
        J[2*0:,:][k*2:(k+1)*2, l*2:(l+1)*2] = out
choi_min = float(np.min(np.linalg.eigvalsh((J+J.conj().T)/2)))
# random-matrix control for L2
rand_viol = []
for _ in range(50):
    Mr = np.random.default_rng().normal(0,0.5,(3,3)); cr=np.zeros(3)
    Jr = np.zeros((4,4),complex)
    for k in range(2):
        for l in range(2):
            Ekl=np.zeros((2,2),complex);Ekl[k,l]=1
            tr=np.trace(Ekl); r=np.array([np.trace(Ekl@s) for s in sig])
            rp = Mr@r
            out=0.5*tr*np.eye(2,dtype=complex)
            for i in range(3): out+=0.5*rp[i]*sig[i]
            Jr[k*2:(k+1)*2, l*2:(l+1)*2]=out
    rand_viol.append(np.min(np.linalg.eigvalsh((Jr+Jr.conj().T)/2)))

print("\nRESULTS")
print(f"  learned M diag        : {np.diag(M).round(4)}  (predicted s={s_pred:.4f})")
print(f"  learned offdiag max   : {np.abs(M-np.diag(np.diag(M))).max():.4f}   |c| = {norm(c.data):.4f}")
print(f"  ||M - s_pred*I||_F    : {norm(M - s_pred*np.eye(3)):.4f}")
print(f"  ||M - Mbar||_F        : {norm(M - Mbar):.4f}   (init floor {norm(M_init-Mbar):.4f})")
print(f"  ||M - M_regression||_F: {norm(M - M_reg):.4f}   (regression vs sI: {norm(M_reg - s_pred*np.eye(3)):.4f})")
print(f"  L3 holdout Bures^2    : {hold:.4f}   (QGT-135param, same data: 0.1464)")
print(f"  L2 Choi min eig       : {choi_min:+.4f}   (random-M control mean: {np.mean(rand_viol):+.4f})")
