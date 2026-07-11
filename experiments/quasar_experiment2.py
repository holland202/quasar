import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
QUASAR Experiment 2 — When does self-direction actually matter?
================================================================
Hypothesis (from Exp 1): adaptive curriculum ties with uniform when the
learner's error is flat across difficulty. It should WIN when competence
is uneven. We create that regime by pretraining ONLY on easy (bin-0) data,
then racing adaptive vs uniform from identical lopsided checkpoints.

Holdout is skewed toward hard dynamics (where the pretrained learner is
weak) so the metric is sensitive to whether the gap gets closed.
"""
import copy
import numpy as np
from quasar.quasar import Quasar, Generator, make_holdout

def lopsided_pretrain(q, rounds=4, n=10, epochs=3):
    """Train only on the easiest bin -> strong on easy, weak on hard."""
    for _ in range(rounds):
        X, Y, _ = q.gen.batch(n, q.seq_len, q._sampler_for_bin(0))
        for _ in range(epochs):
            q.learner.train_step(X, Y)

def hard_holdout(seed=555, n=16, seq_len=6):
    """Held-out REAL dynamics drawn from the harder half of physics space."""
    g = Generator(seed)
    def hard_sampler():
        axis = g.rng.standard_normal(3); axis /= np.linalg.norm(axis)
        w = g.rng.uniform(0.65, 1.2)          # bins 2-3 range
        gam = g.rng.uniform(0.0, 0.12)
        return axis, w, gam
    X, Y, _ = g.batch(n, seq_len, hard_sampler)
    return X, Y

def main():
    seq_len = 6
    Xh, Yh = hard_holdout(seed=555, n=16, seq_len=seq_len)

    # Build ONE lopsided checkpoint, then clone for a fair race
    q0 = Quasar(seed=0, seq_len=seq_len)
    print("[Pretraining] easy-only (bin 0) to create uneven competence...")
    lopsided_pretrain(q0)
    e_pre = q0.learner.loss(Xh, Yh)
    errs = q0.probe_bin_errors(n_per_bin=6)
    print(f"  hard-holdout loss after easy-only pretrain: {e_pre:.4f}")
    print(f"  per-bin error: {' '.join(f'{e:.3f}' for e in errs)}"
          f"  (uneven? {errs[-2:].mean() > errs[:2].mean()})")

    qa = copy.deepcopy(q0)   # adaptive
    qu = copy.deepcopy(q0)   # uniform control

    R, NPR, EPR = 6, 10, 3
    print(f"\n[Adaptive] {R} rounds x {NPR} traj")
    ha = qa.run(R, NPR, EPR, holdout=(Xh, Yh), adaptive=True)
    print(f"\n[Uniform control] same budget")
    hu = qu.run(R, NPR, EPR, holdout=(Xh, Yh), adaptive=False)

    ea, eu = ha["holdout"][-1], hu["holdout"][-1]
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS (hard held-out dynamics)")
    print("=" * 60)
    print(f"  after easy-only pretrain : {e_pre:.4f}")
    print(f"  adaptive self-direction  : {ea:.4f}  ({(e_pre-ea)/e_pre*100:+.1f}%)")
    print(f"  uniform control          : {eu:.4f}  ({(e_pre-eu)/e_pre*100:+.1f}%)")
    print(f"\n  adaptive beats uniform: {ea < eu}"
          f"  (margin {(eu-ea)/eu*100:+.2f}%)")

if __name__ == "__main__":
    main()
