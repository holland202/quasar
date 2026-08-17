import unittest
import numpy as np
from numpy import sqrt, pi, cos, sin
from numpy.linalg import norm

from quasar import (
    project_bloch,
    bures_distance,
    bures_distance_gradient,
    so3,
    softmax,
    bures_path_length,
)


class TestCoreMath(unittest.TestCase):
    def test_project_bloch_inside(self):
        v = np.array([0.3, 0.4, 0.2])
        p = project_bloch(v)
        np.testing.assert_allclose(p, v)

    def test_project_bloch_outside(self):
        v = np.array([2.0, 0, 0])
        p = project_bloch(v)
        np.testing.assert_allclose(p, [1.0, 0, 0])

    def test_bures_pure_states(self):
        r1 = np.array([1, 0, 0])
        r2 = np.array([cos(pi / 3), sin(pi / 3), 0])
        d = bures_distance(r1, r2)
        expected = np.arccos(sqrt((1 + cos(pi / 3)) / 2))
        self.assertAlmostEqual(d, expected, places=10)

    def test_bures_gradient(self):
        r1 = np.array([0.5, 0.3, 0.1])
        r2 = np.array([0.2, 0.4, 0.6])
        grad_analytical = bures_distance_gradient(r1, r2)
        eps = 1e-5
        grad_fd = np.zeros(3)
        for i in range(3):
            rp = r1.copy(); rp[i] += eps
            rm = r1.copy(); rm[i] -= eps
            grad_fd[i] = (bures_distance(rp, r2) - bures_distance(rm, r2)) / (2 * eps)
        diff = norm(grad_analytical - grad_fd)
        self.assertLess(diff, 1e-3)

    def test_so3_orthogonal(self):
        for _ in range(20):
            axis = np.random.randn(3)
            axis = axis / norm(axis)
            angle = np.random.uniform(0, 2 * pi)
            R = so3(axis, angle)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=10)

    def test_softmax(self):
        x = np.array([1.0, 2.0, 3.0])
        s = softmax(x)
        self.assertAlmostEqual(s.sum(), 1.0)
        self.assertTrue(np.all(s >= 0))

    def test_bures_path_length(self):
        traj = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        length = bures_path_length(traj)
        self.assertGreater(length, 0)


if __name__ == "__main__":
    unittest.main()
