"""quasar/multi_qubit_tomography.py

2-qubit tomography extension.
- 15-dim generalized Bloch vector (SU(4) Gell-Mann basis)
- Pauli tensor-product measurements (15 nontrivial observables)
- Linear inversion + MLE projection onto physical states
- Superfidelity metric: F_super = Tr(rho*sigma) + sqrt((1-Tr(rho^2))(1-Tr(sigma^2)))

Self-contained; no modifications to quasar.py needed.
"""

import numpy as np
from itertools import product

# --- SU(4) generalized Gell-Mann basis ---
def _pauli():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return {'I': I, 'X': X, 'Y': Y, 'Z': Z}

_PAULI = _pauli()

# 15 nontrivial two-qubit Pauli operators (exclude II)
_TWO_QUBIT_PAULI = []
_TWO_QUBIT_LABELS = []
for a, b in product(['I', 'X', 'Y', 'Z'], repeat=2):
    if a == 'I' and b == 'I':
        continue
    op = np.kron(_PAULI[a], _PAULI[b])
    _TWO_QUBIT_PAULI.append(op)
    _TWO_QUBIT_LABELS.append(a + b)

# Normalization: Tr(P_i P_j) = 4 delta_ij, so divide by 2 for HS norm 1
_BASIS_OPS = [P / 2.0 for P in _TWO_QUBIT_PAULI]


def rho_to_bloch15(rho):
    """4x4 density matrix -> 15-dim Bloch vector."""
    return np.array([np.trace(rho @ B).real for B in _BASIS_OPS], dtype=float)


def bloch15_to_rho(r):
    """15-dim Bloch vector -> 4x4 density matrix."""
    r = np.asarray(r, dtype=float)
    return (np.eye(4, dtype=complex) + sum(ri * Bi for ri, Bi in zip(r, _BASIS_OPS))) / 4.0


def is_physical(rho, tol=1e-9):
    """Check Hermitian, trace-1, positive-semidefinite."""
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    if not np.isclose(np.trace(rho), 1.0, atol=tol):
        return False
    ev = np.linalg.eigvalsh(rho)
    return np.all(ev >= -tol)


def project_physical(rho):
    """Project onto physical states via eigenvalue clipping."""
    rho = (rho + rho.conj().T) / 2.0
    w, v = np.linalg.eigh(rho)
    w = np.maximum(w, 0)
    w = w / np.sum(w)
    return v @ np.diag(w) @ v.conj().T


def tr_rho2(rho):
    return np.trace(rho @ rho).real


def superfidelity(rho, sigma):
    """Superfidelity — lower bound on Uhlmann-Jozsa fidelity."""
    tr = np.trace(rho @ sigma).real
    r2 = tr_rho2(rho)
    s2 = tr_rho2(sigma)
    return tr + np.sqrt(max(0.0, (1.0 - r2) * (1.0 - s2)))


def bures_distance2q(rho, sigma):
    """Bures distance for 2-qubit states."""
    f = superfidelity(rho, sigma)
    f = min(f, 1.0)
    return np.sqrt(2.0 - 2.0 * np.sqrt(f))


def measure_pauli_expectations(rho, shots, rng=None):
    """Simulate finite-shot measurement of all 15 Pauli observables."""
    if rng is None:
        rng = np.random.default_rng()
    expectations = []
    for P in _TWO_QUBIT_PAULI:
        ev = np.trace(rho @ P).real
        p = (1 + ev) / 2
        counts = rng.binomial(shots, p)
        est = 2 * (counts / shots) - 1
        expectations.append(est)
    return np.array(expectations, dtype=float)


def linear_inversion_2q(expectations):
    """Reconstruct rho from 15 Pauli expectation values."""
    rho = np.eye(4, dtype=complex) / 4.0
    for e, P in zip(expectations, _TWO_QUBIT_PAULI):
        rho += e * P / 4.0
    return rho


def mle_reconstruction_2q(expectations, max_iter=100, tol=1e-9):
    """Iterative MLE for 2-qubit state (Hradil-style)."""
    rho = project_physical(linear_inversion_2q(expectations))
    for _ in range(max_iter):
        R = sum((e / max(np.trace(rho @ P).real, 1e-12)) * P for e, P in zip(expectations, _TWO_QUBIT_PAULI))
        R = R / 4.0
        rho_new = project_physical(R @ rho @ R)
        if np.linalg.norm(rho_new - rho, 'fro') < tol:
            break
        rho = rho_new
    return rho


class Tomographic2QGenerator:
    """Finite-shot tomography wrapper for 2-qubit trajectories."""

    def __init__(self, base_generator, shots=2048, method='mle', seed=None):
        self.base = base_generator
        self.shots = shots
        self.method = method
        self.rng = np.random.default_rng(seed)

    def generate_trajectory(self, n_steps=10):
        exact = self.base.generate_trajectory(n_steps)
        exact_rhos = exact['states']
        recon_rhos = []
        errors = []
        fids = []
        for rho in exact_rhos:
            exps = measure_pauli_expectations(rho, self.shots, self.rng)
            if self.method == 'linear':
                r = linear_inversion_2q(exps)
            else:
                r = mle_reconstruction_2q(exps)
            r = project_physical(r)
            recon_rhos.append(r)
            errors.append(bures_distance2q(rho, r))
            fids.append(superfidelity(rho, r))
        exact_bloch = [rho_to_bloch15(r) for r in exact_rhos]
        recon_bloch = [rho_to_bloch15(r) for r in recon_rhos]
        difficulty = sum(bures_distance2q(recon_rhos[i], recon_rhos[i+1]) for i in range(len(recon_rhos)-1))
        return {
            'states': recon_bloch,
            'exact_states': exact_bloch,
            'exact_rhos': exact_rhos,
            'reconstructed_rhos': recon_rhos,
            'difficulty': difficulty,
            'reconstruction_errors': errors,
            'fidelities': fids,
            'shots': self.shots,
        }

    def generate_batch(self, n, n_steps=10):
        return [self.generate_trajectory(n_steps) for _ in range(n)]


def _random_physical_2q(rng):
    """Generate random physical 2-qubit state."""
    psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    p = rng.random() * 0.3
    return (1 - p) * rho + p * np.eye(4) / 4


def self_test():
    print("=" * 60)
    print("MULTI-QUBIT TOMOGRAPHY — SELF-TEST")
    print("=" * 60)
    rng = np.random.default_rng(42)
    passed = 0
    total = 0

    total += 1
    rho = _random_physical_2q(rng)
    r = rho_to_bloch15(rho)
    rho2 = bloch15_to_rho(r)
    if np.allclose(rho, rho2, atol=1e-10):
        print("[PASS] Bloch15 roundtrip")
        passed += 1
    else:
        print("[FAIL] Bloch15 roundtrip")

    total += 1
    if is_physical(bloch15_to_rho(np.zeros(15))):
        print("[PASS] Maximally mixed is physical")
        passed += 1
    else:
        print("[FAIL] Maximally mixed physicality")

    total += 1
    rho = _random_physical_2q(rng)
    sigma = _random_physical_2q(rng)
    f = superfidelity(rho, sigma)
    if 0 <= f <= 1 + 1e-9:
        print("[PASS] Superfidelity in [0,1]")
        passed += 1
    else:
        print(f"[FAIL] Superfidelity = {f}")

    total += 1
    exps = [np.trace(rho @ P).real for P in _TWO_QUBIT_PAULI]
    rho_lin = linear_inversion_2q(exps)
    if np.allclose(rho, rho_lin, atol=1e-10):
        print("[PASS] Linear inversion exact recovery")
        passed += 1
    else:
        print("[FAIL] Linear inversion")

    total += 1
    exps = measure_pauli_expectations(rho, 4096, rng)
    rho_rec = mle_reconstruction_2q(exps)
    rho_rec = project_physical(rho_rec)
    err = bures_distance2q(rho, rho_rec)
    if err < 0.15:
        print(f"[PASS] MLE reconstruction error = {err:.4f}")
        passed += 1
    else:
        print(f"[FAIL] MLE reconstruction error = {err:.4f}")

    total += 1
    errs = []
    shots_list = [256, 512, 1024, 2048, 4096]
    for s in shots_list:
        exps = measure_pauli_expectations(rho, s, rng)
        r = mle_reconstruction_2q(exps)
        errs.append(bures_distance2q(rho, project_physical(r)))
    slope = np.polyfit(np.log(shots_list), np.log(errs), 1)[0]
    if slope < -0.4:
        print(f"[PASS] Scaling slope = {slope:.3f}")
        passed += 1
    else:
        print(f"[FAIL] Scaling slope = {slope:.3f}")

    print(f"\nRESULT: {passed}/{total} suites passed")
    print("=" * 60)


if __name__ == "__main__":
    self_test()
