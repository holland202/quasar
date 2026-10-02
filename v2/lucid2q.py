"""
LUCID-2Q v0 — the capacity wall experiment (n = 2 qubits, 15-dim Bloch)
=======================================================================
Everything that failed at n=1 (C3, F2, F4) traced to the 3-dim channel.
This file generalizes the verified stack to two qubits and tests whether
the n=1 discoveries are laws or artifacts.

METRIC HONESTY: for d>2 the exact Uhlmann fidelity needs matrix square
roots. We use SUPERFIDELITY (Miszczak et al. 2009):
    G(rho,sigma) = Tr(rho sigma) + sqrt((1-Tr rho^2)(1-Tr sigma^2))
which EQUALS the exact fidelity at n=1 (it is the formula this project
has used all along) and upper-bounds it for d>2. We define
    d_SB = arccos(sqrt(G))   ("super-Bures")
and phrase every n=2 claim in d_SB. In generalized Bloch coordinates
(rho = I/4 + (1/4) sum_a c_a P_a, P_a = 15 two-qubit Pauli products):
    Tr(rho sigma) = (1 + c.c')/4,   1 - Tr rho^2 = (3 - |c|^2)/4
    G = [1 + c.c' + sqrt((3-|c|^2)(3-|c'|^2))]/4     (n=1: /2 with 1-r^2)

REGISTERED CLAIMS (verified or falsified below, nothing else):
  T1  Pauli basis: Tr(P_a P_b) = 4 delta_ab exactly.
  T2  PTM action == direct unitary conjugation (|err| < 1e-10).
  T3  Superfidelity limits: G(psi,psi)=1; orthogonal pure states G=0.
  T4  Choi test validity: PSD (min eig > -1e-9) on a TRUE channel,
      strongly negative on a random linear map (anti-vacuity, F1 rule).
  T5  Negativity witness: product-H trajectories from product inits have
      negativity < 1e-9; entangling-H trajectories develop > 0.02.
  T6  Group theory: H-ensemble is unitarily invariant => E[T_U] = mu*I
      (adjoint rep of SU(4) irreducible). MC check: ||E[T]-mu I|| small.
  N2-1  Purity-preserving law generalizes: MSE-trained M -> mu_bar*I;
        super-Bures-trained M -> eta**I with eta* (independent risk scan)
        > mu_bar. (F5's discovery tested as a law.)
  N2-2  Emergent CPTP at n=2: Choi(learned) min eig >= -0.05 and
        exceeds random-map control by > 0.3.
  N2-3  Evidence gating in 15-dim: oracle dictionary of 8 true channels
        (4 entangling, 4 product); trajectory identification accuracy at
        the last position >= 0.80 (floor 0.125); entangling-vs-product
        class accuracy >= 0.90.
"""

import time
import numpy as np
from numpy.linalg import norm, eigh, eigvalsh

from backprop import T, matmul, add
from m3_articulation import mse_to_const

t_start = time.time()
rng = np.random.default_rng(0)

# ==================================================================
# Two-qubit Pauli machinery
# ==================================================================
s0 = np.eye(2, dtype=complex)
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
paulis1 = [s0, sx, sy, sz]
P = [np.kron(paulis1[i], paulis1[j])
     for i in range(4) for j in range(4)][1:]          # 15, skip I(x)I
P = np.array(P)                                        # (15,4,4)

def c_of_rho(rho):
    return np.real(np.einsum('kl,alk->a', rho, P))     # Tr(rho P_a)

def rho_of_c(c):
    return np.eye(4, dtype=complex) / 4 + np.einsum('a,akl->kl', c, P) / 4

def random_H(scale):
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    H = (A + A.conj().T) / 2
    return H * scale / norm(H)

def U_of_H(H):
    w, V = eigh(H)
    return V @ np.diag(np.exp(-1j * w)) @ V.conj().T

def ptm(U):
    UP = np.einsum('kl,alm,mn->akn', U, P, U.conj().T)  # U P_a U^dag
    return np.real(np.einsum('akn,bnk->ab', UP, P)) / 4  # T_ba? fix below

def ptm_correct(U):
    # (T)_{ab} = (1/4) Tr(P_a U P_b U^dag)
    UPU = np.einsum('kl,blm,mn->bkn', U, P, U.conj().T)  # U P_b U^dag
    return np.real(np.einsum('akl,blk->ab', P, UPU)) / 4

def random_pure(product=False):
    if product:
        a = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        b = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        psi = np.kron(a / norm(a), b / norm(b))
    else:
        psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        psi = psi / norm(psi)
    return np.outer(psi, psi.conj())

def negativity(rho):
    """Min eigenvalue of the partial transpose (subsystem B). PPT: >=0
    for all separable states of 2x2 systems (Peres-Horodecki, exact)."""
    r = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    return float(np.min(eigvalsh((r + r.conj().T) / 2)))

# ---- T1..T3 -------------------------------------------------------
gram = np.real(np.einsum('akl,blk->ab', P, P))
assert np.allclose(gram, 4 * np.eye(15), atol=1e-12), "T1 FAIL"
print("T1 PASS  Pauli basis orthonormal (Tr P_a P_b = 4 delta_ab)")

H = random_H(0.8); U = U_of_H(H); TU = ptm_correct(U)
rho = random_pure(); c = c_of_rho(rho)
err = norm(c_of_rho(U @ rho @ U.conj().T) - TU @ c)
assert err < 1e-10, f"T2 FAIL err={err}"
orth = norm(TU @ TU.T - np.eye(15))
print(f"T2 PASS  PTM action == conjugation (err {err:.1e}); "
      f"orthogonality dev {orth:.1e}")

def G_fid(c1, c2):
    t1 = max(0.0, 3 - float(c1 @ c1)); t2 = max(0.0, 3 - float(c2 @ c2))
    return np.clip((1 + float(c1 @ c2) + np.sqrt(t1 * t2)) / 4, 0, 1)

psi1 = random_pure(); c1 = c_of_rho(psi1)
w, V = eigh(psi1); psi_orth = np.outer(V[:, 0], V[:, 0].conj())
assert abs(G_fid(c1, c1) - 1) < 1e-10, "T3a FAIL"
assert G_fid(c1, c_of_rho(psi_orth)) < 1e-10, "T3b FAIL"
print("T3 PASS  superfidelity: G(psi,psi)=1, orthogonal G=0")

# ---- Choi test + T4 ----------------------------------------------
def choi_min_eig(M, shift):
    """Affine Bloch map c -> M c + shift*Tr. CP iff Choi >= 0."""
    J = np.zeros((16, 16), dtype=complex)
    for k in range(4):
        for l in range(4):
            E = np.zeros((4, 4), dtype=complex); E[k, l] = 1
            cE = np.array([np.trace(E @ P[a]) for a in range(15)])
            tr = 1.0 if k == l else 0.0
            out = (tr * np.eye(4, dtype=complex) / 4
                   + np.einsum('a,akl->kl', M @ cE + shift * tr, P) / 4)
            J[4*k:4*k+4, 4*l:4*l+4] = out    # block (k,l) = Phi(E_kl)
    # J = sum_kl E_kl (x) Phi(E_kl); CP <=> J >= 0 (identity-channel
    # hand-check: blocks E_kl give d|Omega><Omega| >= 0)
    Jh = (J + J.conj().T) / 2
    return float(np.min(eigvalsh(Jh)))

gam = 0.08
true_min = choi_min_eig(np.exp(-gam) * TU, np.zeros(15))
rand_min = choi_min_eig(rng.standard_normal((15, 15)) * 0.4, np.zeros(15))
assert true_min > -1e-9 and rand_min < -0.2, \
    f"T4 FAIL true={true_min:.2e} rand={rand_min:.2f}"
print(f"T4 PASS  Choi test: true channel {true_min:+.1e}, "
      f"random map {rand_min:+.3f}")

# ---- physics generator + T5 ---------------------------------------
def make_channel(entangling, scale=0.8, gamma=None):
    if entangling:
        Hc = random_H(scale)
    else:
        A = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        HA = (A + A.conj().T) / 2
        B = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        HB = (B + B.conj().T) / 2
        Hc = np.kron(HA, s0) + np.kron(s0, HB)
        Hc = Hc * scale / norm(Hc)
    g = rng.uniform(0.0, 0.12) if gamma is None else gamma
    return np.exp(-g) * ptm_correct(U_of_H(Hc)), g

def trajectory(M, seq, product_init=True):
    c = c_of_rho(random_pure(product=product_init))
    out = np.zeros((seq + 1, 15)); out[0] = c
    for t in range(seq):
        c = M @ c
        out[t + 1] = c
    return out

neg_prod, neg_ent = [], []
for _ in range(20):
    Mp, _ = make_channel(False); Me, _ = make_channel(True)
    tp = trajectory(Mp, 6); te = trajectory(Me, 6)
    neg_prod.append(min(negativity(rho_of_c(tp[t])) for t in range(7)))
    neg_ent.append(min(negativity(rho_of_c(te[t])) for t in range(7)))
neg_prod, neg_ent = np.array(neg_prod), np.array(neg_ent)
assert neg_prod.min() > -1e-9, "T5a FAIL"
assert np.mean(neg_ent < -0.02) > 0.7, f"T5b FAIL {neg_ent}"
print(f"T5 PASS  negativity: product min {neg_prod.min():+.1e}; "
      f"entangling develop < -0.02 in {np.mean(neg_ent < -0.02):.0%}")

# ---- T6: mean channel is mu*I (irreducibility) ---------------------
NMC = 4000
acc = np.zeros((15, 15)); Eg = 0.0
for _ in range(NMC):
    Mk, g = make_channel(True)
    acc += Mk; Eg += np.exp(-g)
Mbar = acc / NMC
mu = np.trace(Mbar) / 15
offI = norm(Mbar - mu * np.eye(15))
assert offI < 0.08, f"T6 FAIL offI={offI}"
print(f"T6 PASS  E[channel] = mu*I: mu = {mu:.4f}, ||E-mu I||_F = {offI:.3f}")

# ==================================================================
# N2-1 + N2-2: train 15x15 map under both losses
# ==================================================================
def _dd_dF(F):
    ok = (F > 1e-9) & (F < 0.999999)
    Fs = np.where(ok, F, 0.5)
    return np.where(ok, -1.0 / (2 * np.sqrt(Fs) * np.sqrt(1 - Fs)), 0.0)

def sb_mean_loss(pred, target):
    """Fused mean d_SB^2 to constant target; c-vectors (b,s,15)."""
    p, t = pred.data, target
    dots = np.sum(p * t, -1)
    mp = 3 - np.sum(p * p, -1); mt = 3 - np.sum(t * t, -1)
    u = mp * mt; st = np.sqrt(np.maximum(0, u))
    F = np.clip((1 + dots + st) / 4, 0, 1)
    D = np.arccos(np.sqrt(F))
    out = T(np.mean(D ** 2), (pred,))
    inv = np.where(st > 1e-12, 1.0 / np.where(st > 1e-12, st, 1.0), 0.0)
    n = D.size
    def bw(g):
        gF = (g * 2 * D / n) * _dd_dF(F)
        dF = 0.25 * t - 0.25 * np.where(u > 0, mt * inv, 0.0)[..., None] * p
        pred._acc(gF[..., None] * dF)
    out._bw = bw
    return out

def batch(n, seq):
    X = np.zeros((n, seq, 15)); Y = np.zeros((n, seq, 15))
    for i in range(n):
        M, _ = make_channel(True)
        tr = trajectory(M, seq, product_init=False)
        X[i], Y[i] = tr[:-1], tr[1:]
    return X, Y

seq, steps = 6, 6000
data = [batch(8, seq) for _ in range(steps)]

def train(loss):
    r2 = np.random.default_rng(4)
    MT = T(r2.normal(0, 0.05, (15, 15)))
    for (x, y) in data:
        MT.grad = None
        pred = matmul(T(x, requires_grad=False), MT)
        (sb_mean_loss(pred, y) if loss == "sb"
         else mse_to_const(pred, y)).backward()
        MT.data -= 0.1 * MT.grad
    return MT.data.T

t0 = time.time()
M_sb = train("sb"); M_ms = train("mse")
t_train = time.time() - t0

def diagI(M):
    d = np.trace(M) / 15
    return d, norm(M - d * np.eye(15))

d_sb, off_sb = diagI(M_sb); d_ms, off_ms = diagI(M_ms)
mu_bar = mu * (Eg / NMC) / np.mean([np.exp(-0)  # mu already includes e^-g
                                    ]) if False else mu
# (make_channel already multiplied e^-g into each sample; mu IS mu_bar)

# independent eta* scan under d_SB risk
Xs, Ys = batch(400, seq)
Xf, Yf = Xs.reshape(-1, 15), Ys.reshape(-1, 15)
def sb_risk(eta):
    Pd = eta * Xf
    dots = np.sum(Pd * Yf, -1)
    u = np.maximum(0, (3 - np.sum(Pd*Pd, -1)) * (3 - np.sum(Yf*Yf, -1)))
    F = np.clip((1 + dots + np.sqrt(u)) / 4, 0, 1)
    return np.mean(np.arccos(np.sqrt(F)) ** 2)
etas = np.linspace(0.5, 1.0, 101)
eta_star = etas[int(np.argmin([sb_risk(e) for e in etas]))]
def ms_risk(eta): return np.mean(np.sum((eta * Xf - Yf) ** 2, -1))
eta_ms = etas[int(np.argmin([ms_risk(e) for e in etas]))]

print(f"\nN2-1  mean-channel mu = {mu:.4f} | MSE-scan eta = {eta_ms:.3f} | "
      f"SB-scan eta* = {eta_star:.3f}")
print(f"      MSE-trained : diag {d_ms:.4f}  off-I {off_ms:.3f}")
print(f"      SB-trained  : diag {d_sb:.4f}  off-I {off_sb:.3f}")
law = (abs(d_ms - mu) < 0.05 and abs(d_sb - eta_star) < 0.05
       and eta_star > mu + 0.03)
print(f"      purity-preserving law generalizes to n=2: {law}")

ch_sb = choi_min_eig(M_sb, np.zeros(15))
ch_rn = np.mean([choi_min_eig(np.random.default_rng(i).normal(0, 0.25, (15, 15)),
                              np.zeros(15)) for i in range(10)])
print(f"N2-2  Choi min eig: SB-learned {ch_sb:+.4f} | random ctrl {ch_rn:+.4f}"
      f"  -> emergent CPTP: {ch_sb >= -0.05 and ch_sb > ch_rn + 0.3}")

# ==================================================================
# N2-3: oracle-dictionary evidence gating in 15-dim
# ==================================================================
Jn = 8
dict_ch, dict_ent = [], []
for j in range(Jn):
    Mj, _ = make_channel(entangling=(j < 4), gamma=0.06)
    dict_ch.append(Mj); dict_ent.append(j < 4)

def d_sb2(c1, c2):
    return float(np.arccos(np.sqrt(G_fid(c1, c2))) ** 2)

trials, beta = 400, 60.0
correct = np.zeros(seq); cls_ok = 0
err_gate, err_oracle = [], []
for _ in range(trials):
    j = rng.integers(0, Jn)
    tr = trajectory(dict_ch[j], seq, product_init=True)
    x, y = tr[:-1], tr[1:]
    cum = np.zeros(Jn)
    for t in range(seq):
        if t > 0:
            for k in range(Jn):
                cum[k] += d_sb2(dict_ch[k] @ x[t-1], x[t])
        w = np.exp(-beta * cum / max(t, 1)); w /= w.sum()
        khat = int(np.argmax(w))
        correct[t] += (khat == j)
        if t == seq - 1:
            cls_ok += (dict_ent[khat] == dict_ent[j])
            pred = sum(w[k] * (dict_ch[k] @ x[t]) for k in range(Jn))
            err_gate.append(d_sb2(pred, y[t]))
            err_oracle.append(d_sb2(dict_ch[j] @ x[t], y[t]))
correct /= trials
print(f"\nN2-3  identification acc by position: "
      + " ".join(f"{v:.2f}" for v in correct) + f"  (floor {1/Jn:.3f})")
print(f"      final-pos accuracy {correct[-1]:.3f} (>=0.80: "
      f"{correct[-1] >= 0.80}) | entangling-vs-product "
      f"{cls_ok/trials:.3f} (>=0.90: {cls_ok/trials >= 0.90})")
print(f"      last-step d_SB^2: gate {np.mean(err_gate):.5f} vs oracle "
      f"{np.mean(err_oracle):.5f}")

print(f"\nruntime: total {time.time()-t_start:.1f}s "
      f"(training both arms {t_train:.1f}s, {steps} steps x 225 params each)")
