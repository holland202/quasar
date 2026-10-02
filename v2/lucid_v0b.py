"""LUCID v0b — resolve the L1 surprise.
Registered: (1) MSE-trained channel -> diag ~ 0.796 (mean channel);
(2) Bures-optimal isotropic contraction eta* (direct MC scan) ~ observed 0.928.
If both hold: 'Bures learning is purity-preserving; Euclidean learning is
direction-averaging' — a real property of learning under quantum metrics."""
import numpy as np
from numpy.linalg import norm
from quasar import Generator
from quantum_geometric_transformer import project_bloch, bures_distance
from backprop import T, matmul, add, pbloch, bures_mean_loss
from m3_articulation import batch_with_physics, mse_to_const

gen = Generator(seed=11)
batches = [batch_with_physics(gen, 8, 6) for _ in range(400)]
gh = Generator(seed=999); Xh, Yh, _ = batch_with_physics(gh, 32, 6)

def train(loss_kind):
    rng = np.random.default_rng(4)
    MT = T(rng.normal(0, 0.1, (3, 3))); c = T(rng.normal(0, 0.01, (3,)))
    for (x, y, _) in batches:
        MT.grad = None; c.grad = None
        pred = pbloch(add(matmul(T(x, requires_grad=False), MT), c))
        L = bures_mean_loss(pred, y) if loss_kind=="bures" else mse_to_const(pred, y)
        L.backward()
        MT.data -= 0.12 * MT.grad; c.data -= 0.12 * c.grad
    return MT.data.T, c.data

def hold_bures(M, c):
    p = project_bloch(Xh @ M.T + c)
    return np.mean([[bures_distance(p[b,t], Yh[b,t])**2 for t in range(6)]
                    for b in range(32)])

M_b, c_b = train("bures"); M_m, c_m = train("mse")
print(f"Bures-trained diag: {np.diag(M_b).round(4)}  holdout(B2)={hold_bures(M_b,c_b):.4f}")
print(f"MSE-trained   diag: {np.diag(M_m).round(4)}  holdout(B2)={hold_bures(M_m,c_m):.4f}")

# direct scan: eta* = argmin E[d_B^2(eta*r, true next)]  (isotropic family)
g2 = Generator(seed=77)
Xs, Ys, _ = batch_with_physics(g2, 400, 6)
Xf, Yf = Xs.reshape(-1,3), Ys.reshape(-1,3)
etas = np.linspace(0.70, 1.00, 61)
def risk(eta, metric):
    P = project_bloch(eta * Xf)
    if metric=="bures":
        return np.mean([bures_distance(P[i], Yf[i])**2 for i in range(len(P))])
    return np.mean(np.sum((P - Yf)**2, axis=1))
rb = [risk(e,"bures") for e in etas]; rm = [risk(e,"mse") for e in etas]
print(f"eta* (Bures risk) = {etas[int(np.argmin(rb))]:.3f}   "
      f"eta* (MSE risk) = {etas[int(np.argmin(rm))]:.3f}")
print(f"reference: E[e^-g]={((1-np.exp(-0.12))/0.12):.4f}, mean-channel s=0.7960")
