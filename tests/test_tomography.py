import unittest
import numpy as np

from quasar.finite_shot_tomography import (
    bloch_to_rho,
    rho_to_bloch,
    fidelity,
    bures_distance as bures_distance_dm,
    MeasurementSimulator,
    StateReconstructor,
    run_all_tests,
)


class TestTomography(unittest.TestCase):
    def test_bloch_roundtrip(self):
        for _ in range(100):
            r = np.random.randn(3)
            r = r / np.linalg.norm(r) * np.random.random()
            rho = bloch_to_rho(r)
            r2 = rho_to_bloch(rho)
            np.testing.assert_allclose(r, r2, atol=1e-12)

    def test_fidelity_pure_orthogonal(self):
        p0 = np.array([1, 0], dtype=complex)
        p1 = np.array([0, 1], dtype=complex)
        rho0 = np.outer(p0, p0.conj())
        rho1 = np.outer(p1, p1.conj())
        self.assertAlmostEqual(fidelity(rho0, rho1), 0.0, places=10)

    def test_fidelity_pure_same(self):
        p0 = np.array([1, 0], dtype=complex)
        rho0 = np.outer(p0, p0.conj())
        self.assertAlmostEqual(fidelity(rho0, rho0), 1.0, places=10)

    def test_measurement_z_basis(self):
        sim = MeasurementSimulator(shots=100000, seed=42)
        rho = bloch_to_rho(np.array([0, 0, 1]))
        c, p = sim.measure(rho, np.eye(2, dtype=complex))
        self.assertAlmostEqual(p[0], 1.0, places=12)
        self.assertAlmostEqual(p[1], 0.0, places=12)

    def test_linear_inversion_exact(self):
        bx = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        by = np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2)
        bz = np.eye(2, dtype=complex)
        rec = StateReconstructor(method='linear')
        for _ in range(50):
            r = np.random.randn(3)
            r = r / np.linalg.norm(r) * np.random.random()
            rho = bloch_to_rho(r)
            sim = MeasurementSimulator(shots=100000, seed=None)
            d = {'bases': [bx, by, bz], 'counts': [], 'probs': [], 'shots': sim.shots}
            for b in [bx, by, bz]:
                c, p = sim.measure(rho, b)
                d['counts'].append(c)
                d['probs'].append(p)
            np.testing.assert_allclose(rec.linear_inversion(d), r, atol=0.02)

    def test_mle_vs_linear(self):
        rl = StateReconstructor(method='linear')
        rm = StateReconstructor(method='mle')
        sim = MeasurementSimulator(shots=512, seed=42)
        n = 0
        for _ in range(100):
            r = np.random.randn(3)
            r = r / np.linalg.norm(r) * np.random.random()
            rho = bloch_to_rho(r)
            d = sim.measure_random_bases(rho, 6)
            r_mle = rm.reconstruct(d)
            self.assertLessEqual(np.linalg.norm(r_mle), 1.0 + 1e-10)
            if bures_distance_dm(bloch_to_rho(r_mle), rho) <= bures_distance_dm(bloch_to_rho(rl.reconstruct(d)), rho) + 1e-10:
                n += 1
        self.assertGreaterEqual(n, 40)

    def test_scaling(self):
        r = np.array([0.3, -0.5, 0.7])
        rho = bloch_to_rho(r)
        e = []
        shots_list = [64, 128, 256, 512, 1024, 2048]
        for shots in shots_list:
            sim = MeasurementSimulator(shots=shots, seed=42)
            rec = StateReconstructor(method='mle')
            e.append(rec.reconstruction_error(sim.measure_random_bases(rho, 6), rho))
        slope = np.polyfit(np.log(shots_list), np.log(e), 1)[0]
        self.assertGreaterEqual(slope, -0.7)
        self.assertLess(slope, -0.3)

    def test_self_test_suite(self):
        self.assertTrue(run_all_tests())


if __name__ == "__main__":
    unittest.main()
