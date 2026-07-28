import unittest
import numpy as np
from numpy.linalg import norm

from quasar.core import project_bloch, bures_distance
from quasar.quantum_geometric_transformer import (
    QuantumGeometricTransformer,
    BuresAttention,
    run_self_test,
)


class TestTransformer(unittest.TestCase):
    def test_bloch_constraint_attention(self):
        attn = BuresAttention(d_model=3, n_heads=1, beta=1.0, seed=0)
        x = np.random.default_rng(42).normal(0, 0.1, (2, 4, 3))
        x = project_bloch(x)
        out = attn.forward(x, causal=True)
        max_norm = np.max(norm(out, axis=-1))
        self.assertLessEqual(max_norm, 1.0 + 1e-6)

    def test_bloch_constraint_full_forward(self):
        qgt = QuantumGeometricTransformer(seed=0)
        x = np.random.default_rng(42).normal(0, 0.1, (2, 8, 3))
        x = project_bloch(x)
        pred = qgt.forward(x)
        max_norm = np.max(norm(pred, axis=-1))
        self.assertLessEqual(max_norm, 1.0 + 1e-6)

    def test_training_convergence(self):
        qgt = QuantumGeometricTransformer(seed=0)
        rng = np.random.default_rng(42)
        X = np.zeros((4, 8, 3))
        Y = np.zeros((4, 8, 3))
        for b in range(4):
            r0 = rng.standard_normal(3)
            r0 = r0 / norm(r0)
            axis = rng.standard_normal(3)
            axis = axis / norm(axis)
            for t in range(8):
                from quasar.core import so3
                R = so3(axis, 0.2 * t)
                X[b, t] = project_bloch(R @ r0)
                R_next = so3(axis, 0.2 * (t + 1))
                Y[b, t] = project_bloch(R_next @ r0)

        loss_before = qgt.loss(X, Y)
        for _ in range(5):
            qgt.train_step(X, Y, lr=0.1, eps=1e-5)
        loss_after = qgt.loss(X, Y)
        self.assertLess(loss_after, loss_before)
        improvement = (loss_before - loss_after) / loss_before * 100
        self.assertGreater(improvement, 5.0)

    def test_self_test_runs(self):
        self.assertTrue(run_self_test(verbose=False))


if __name__ == "__main__":
    unittest.main()
