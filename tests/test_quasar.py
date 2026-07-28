import unittest
import numpy as np

from quasar.quasar import Quasar, Learner, Generator, make_holdout


class TestQuasarLoop(unittest.TestCase):
    def test_generator_batch_shape(self):
        gen = Generator(seed=0)
        X, Y, diff = gen.batch(10, seq_len=6)
        self.assertEqual(X.shape, (10, 6, 3))
        self.assertEqual(Y.shape, (10, 6, 3))
        self.assertEqual(diff.shape, (10,))

    def test_generator_bloch_constraint(self):
        gen = Generator(seed=0)
        X, Y, _ = gen.batch(5, seq_len=6)
        from numpy.linalg import norm
        self.assertTrue(np.all(norm(X, axis=-1) <= 1.0 + 1e-10))
        self.assertTrue(np.all(norm(Y, axis=-1) <= 1.0 + 1e-10))

    def test_difficulty_stratification(self):
        q = Quasar(seed=0, seq_len=6, curriculum='uniform')
        means = []
        for b in range(q.n_bins):
            _, _, d = q.gen.batch(6, 6, q._sampler_for_bin(b))
            means.append(np.mean(d))
        # Difficulty should increase across bins
        for i in range(len(means) - 1):
            self.assertLess(means[i], means[i + 1])

    def test_c1_self_training_improves(self):
        Xh, Yh = make_holdout(seed=777, n=16, seq_len=6)
        base = Learner(seed=0)
        e0 = base.loss(Xh, Yh)

        qa = Quasar(seed=0, seq_len=6, curriculum='error')
        ha = qa.run(3, 8, 2, holdout=(Xh, Yh), adaptive=True, verbose=False)
        ea = ha["holdout"][-1]
        self.assertLess(ea, e0, "C1 FAILED: no improvement on held-out dynamics")

    def test_c2_curriculum_self_directs(self):
        Xh, Yh = make_holdout(seed=777, n=16, seq_len=6)
        qa = Quasar(seed=0, seq_len=6, curriculum='error')
        ha = qa.run(3, 8, 2, holdout=(Xh, Yh), adaptive=True, verbose=False)
        w_first, w_last = ha["weights"][1], ha["weights"][-1]
        shift = float(np.abs(w_last - np.ones(4) / 4).sum())
        self.assertGreater(shift, 0.05, "C2 FAILED: loop did not self-direct")

    def test_progress_curriculum_mode(self):
        q = Quasar(seed=0, seq_len=6, curriculum='progress')
        self.assertIsNotNone(q._progress_sampler)
        X, Y, _ = q.generate_round(4)
        self.assertEqual(X.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
