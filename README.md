# QUASAR — Quantum-geometric Unified Self-training ARchitecture

A closed-loop AI that **generates its own training data** from the geometry
of its state space (the Bloch manifold) and **directs its own curriculum**
from its own per-difficulty errors. Pure NumPy. Runs on a phone.

```
[GENERATOR] → [GEOMETRIC DIFFICULTY] → [LEARNER (QGT)] → errors
     ▲            Bures path length                        │
     └──────────── sampling weights ∝ error ◄──────────────┘
```

## Components
| Module | What it is |
|---|---|
| `quasar/quantum_geometric_transformer.py` | QGT: Bures-metric attention on Bloch vectors. 7-suite self-test. |
| `quasar/quantum_geometric_rl.py` | QGRL: QGT as control policy, bracketed vs random floor / analytical-optimal ceiling. |
| `quasar/quasar.py` | The closed loop: generator + native geometric difficulty + error-driven self-direction. |
| `experiments/quasar_experiment2.py` | Uneven-competence experiment (honest negative result, see below). |

## Verified claims (run them yourself)
- **C1** Trained purely on self-generated data, holdout error on 16
  never-seen Hamiltonians drops ~13% (0.2406 → 0.2090).
- **C2** The curriculum self-directs: generation weights measurably track
  the learner's per-bin error (L1 shift from uniform ≈ 0.22).
- **QGRL** learned policy reaches normalised return +0.93 on 200 unseen
  start states (0 = random, 1 = analytical optimum).

## Honest negative results (kept on purpose)
- **C3** Adaptive self-direction did **not** beat uniform sampling at this
  scale (tie, −0.3%). Diagnosis: a 135-parameter learner has flat
  competence across difficulty — no gradient for self-direction to
  exploit. Roadmap: analytical backprop → scale capacity → retest.
- Finite-difference training is the current bottleneck (O(params) loss
  evals per step).

## Run it (desktop or Termux)
```bash
pip install -r requirements.txt        # numpy only
python -m quasar.quantum_geometric_transformer   # ~5 s,  7 test suites
python -m quasar.quantum_geometric_rl            # ~60 s, RL experiment
python -m quasar.quasar                          # ~90 s, closed loop
python experiments/quasar_experiment2.py         # ~120 s
python run_all_tests.py                          # everything
```
Verified on Android/Termux (aarch64, Python 3.14): all suites pass, closed loop 38.4s — faster than the x86 sandbox it was built in.

## Termux quickstart
```bash
pkg update && pkg install -y python git
pip install numpy
git clone <your-repo-url> && cd quasar
python run_all_tests.py
```

## Scope (read before hyping)
Classical simulation of single-qubit geometry. The Bures metric, SO(3)
actions, and decoherence channel are genuine quantum-information objects;
there is no quantum hardware and no quantum-advantage claim. This is a
working principle demonstration of self-training geometric AI at tiny
scale, with every claim tested against controls and ground truth.

## License
MIT
