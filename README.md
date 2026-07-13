# QUASAR — Quantum-geometric Unified Self-training ARchitecture

[![verify-all-claims](https://github.com/holland202/quasar/actions/workflows/tests.yml/badge.svg)](https://github.com/holland202/quasar/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![deps](https://img.shields.io/badge/dependencies-numpy%20only-brightgreen)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
![runs on](https://img.shields.io/badge/runs%20on-an%20Android%20phone-orange)

**A closed-loop AI that generates its own training data from the geometry of
its state space, and directs its own curriculum from its own errors.**
Pure NumPy. No GPU. Verified end-to-end on an Android phone.

```
[GENERATOR] → [GEOMETRIC DIFFICULTY] → [LEARNER (QGT)] → errors
     ▲            Bures path length                        │
     └────────── sampling weights ∝ error ◄────────────────┘
```

The generator samples random Hamiltonians and decoherence channels — every
trajectory is a physically valid quantum evolution, so **no human dataset is
needed**. Difficulty is measured natively as Bures path length. The learner's
per-difficulty error sets what gets generated next. No hand-designed schedule.

## Components

| Module | What it is |
|---|---|
| `quasar/quantum_geometric_transformer.py` | **QGT** — transformer with Bures-metric attention on Bloch vectors, replacing dot-product similarity with quantum state distinguishability. 7-suite self-test. |
| `quasar/quantum_geometric_rl.py` | **QGRL** — QGT as a control policy, bracketed between a random floor and the analytical-optimal ceiling. |
| `quasar/quasar.py` | **The closed loop** — generator + geometric difficulty + error-driven self-direction, with a uniform-sampling control at equal budget. |
| `experiments/quasar_experiment2.py` | Uneven-competence experiment (honest negative result, below). |

## Results

All numbers are printed by the code itself. Nothing here that
`run_all_tests.py` doesn't reproduce.

**QGRL — quantum control** (held-out: 200 unseen start states)

| Policy | mean final Bures distance | normalised return |
|---|---|---|
| Random | 0.74 | 0.00 (floor) |
| **QGT, learned** | **0.033** | **+0.93** |
| Analytical optimal | 0.00 | 1.00 (ceiling) |

**QUASAR closed loop** (held-out: 16 unseen-Hamiltonian trajectories)

| | holdout Bures loss |
|---|---|
| Untrained QGT | 0.2406 |
| **Trained purely on self-generated data** | **0.2090 (−13%)** |

- **C1 ✅** Self-training transfers to dynamics never seen in training.
- **C2 ✅** The curriculum self-directs: generation weights track the
  learner's per-bin error (L1 shift from uniform ≈ 0.22).
- **C3 ❌ (kept on purpose)** Adaptive self-direction **tied** with uniform
  sampling (−0.3%). Diagnosis: a 135-parameter learner has flat competence
  across difficulty — no gradient for self-direction to exploit. A second
  experiment engineered to create uneven competence confirmed the model is
  too small to specialize. The loop steers correctly; the ship is too small
  to turn.

Negative results are the point. A self-training system whose advantage
appears only under stated, testable conditions is science; one that always
wins is marketing.

## Run it (desktop or Termux)

```bash
pip install numpy                                # only dependency
python -m quasar.quantum_geometric_transformer   # ~5 s   7 test suites
python -m quasar.quantum_geometric_rl            # ~60 s  RL experiment
python -m quasar.quasar                          # ~40-90 s  closed loop
python experiments/quasar_experiment2.py         # ~120 s
python run_all_tests.py                          # everything
```

### Termux (Android)

```bash
pkg update && pkg install -y python git python-numpy
git clone https://github.com/holland202/quasar && cd quasar
python run_all_tests.py
```

Verified on-device (aarch64, Python 3.14): **all suites pass, closed loop
38.4 s** — faster than the x86 sandbox it was built in.

## Reproduced it?

Open a [reproduction issue](https://github.com/holland202/quasar/issues/new?template=reproduction.md) with your `run_all_tests.py`
output and timings. Independent reproductions (or failures!) are the most
valuable contribution this repo can receive.

## The C3 prediction was tested. It was WRONG.

This README previously predicted that adaptive self-direction would win
"once the learner can specialize," and promised the result either way.
Here it is.

Retested on a 225-parameter learner (2 qubits, 15-dim), where competence
is genuinely uneven (per-bin error 0.052 → 0.346, spread 0.56 — the
learner *can* specialize, and the sampler *did* reallocate 37% of its
budget to the hardest bin):

| curriculum | holdout (5 seeds) | vs uniform | seeds won |
|---|---|---|---|
| uniform | 0.19056 ± 0.00264 | — | — |
| **error-proportional** | 0.20595 ± 0.00179 | **−8.1%** | **0/5** |
| **progress-proportional** | 0.18428 ± 0.00249 | **+3.3%** | **5/5** |

Adaptive sampling didn't merely fail to win — **it was actively harmful.**

**The mechanism:** `corr(final error, learnability) = −0.999`. Measuring how
much of each bin's error is actually *reducible*: the easiest bin, 87.2%;
the hardest, 14.8%. **Error does not track learnability.** Error-driven
sampling pours the budget into precisely the region where learning is
least possible. Hard ≠ learnable.

**The fix:** weight by *learning progress* (recent error **reduction**),
not error level. It wins 5/5, beats the error-driven curriculum by 12.5%,
and discovers an easy-to-hard schedule unprompted (early weights
0.276 → 0.121, relaxing to uniform as progress equalizes).

This also retro-explains C3 itself: the original tie was **luck**. A
flat-competence learner cannot be harmed by a bad signal. Give it capacity,
and the bad signal hurts you.

*Prior art, stated honestly:* learning-progress curricula are known
(Graves et al., *Automated Curriculum Learning*, 2017). What is new here is
the mechanism measured in channel space at corr = −0.999.

**Consequence: the error-driven curriculum in this repository is wrong by
design.** It is left in place, unchanged, because it is the experiment that
produced the finding — and because a repository that quietly deletes its
refuted claims is not a scientific record.

## Roadmap

1. ~~**Analytical backprop** through Bures attention~~ — **DONE.** Micro
   reverse-mode AD engine; gradients verified to 8.4e-11 against central
   finite differences across all 135 parameters; ~600–680× speedup
   (load-dependent).
2. ~~**Scale capacity** and retest C3~~ — **DONE, and the prediction was
   REFUTED.** See above.
3. ~~**Multi-qubit extension** (d_model = 4ⁿ − 1)~~ — **DONE.** Two-qubit
   substrate (15-dim generalized Bloch, superfidelity metric); the
   purity-preserving learning law replicates at the new scale.
4. **Next:** replace the curriculum signal with learning progress; finite-shot
   tomography (raw measurement outcomes rather than exact states).

v0.2 is developed privately and released when its claims are verified.

## Scope — read before hyping

Classical simulation of single-qubit geometry. The Bures metric, SO(3)
actions, and decoherence channel are genuine quantum-information objects;
there is **no quantum hardware and no quantum-advantage claim**. This is a
working principle demonstration of self-training geometric AI at tiny scale,
with every claim tested against controls and ground truth — including the
ones that failed.

## License

MIT — see [LICENSE](LICENSE).


