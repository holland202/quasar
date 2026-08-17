# QUASAR — a learner that generates its own training data

**The problem it addresses:** human-authored training material is finite. QUASAR
is a closed loop — **generator → curriculum → learner** — with no human corpus
anywhere in it. The generator invents trajectories, the curriculum decides which
kinds to invent more of, and the learner trains on what comes out.

The test domain is single-qubit channel dynamics. That choice is deliberate and
unglamorous: it is exhaustively checkable in NumPy on a phone, and it has a real
difficulty axis, so "the curriculum shifted toward harder material" is a
measurable claim rather than a vibe.

Pure NumPy. No GPU. No quantum hardware.

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Status, honestly

**Fixed 2026-08-17.** From 2026-07-26 until that date this repo did not import
on a cold clone, and it was archived on 2026-08-04 with a notice saying so.
The cause was smaller than the notice implied:

`quasar/__init__.py` imported six names from `quasar.core`, `quasar.gradients`
and `quasar.experiment_logger` — **modules that were never written in this
line.** Six of the seven real modules ran fine standalone the whole time; one
unreachable import took the package down with them. The symbols it wanted
(`project_bloch`, `bures_distance`, `so3`, `softmax`, …) had always existed,
just in `quantum_geometric_transformer.py` and `quasar.py`. `__init__.py` was
written against a planned refactor that never landed.

The same commit — labelled *"v0.2.0 production package: installable API, full
test suite"* — also added a `tests/` directory written against that same
unbuilt API, and CI that died at `pip install -e .` before it could reveal
that four of nine suites failed. **The install failure was hiding the test
failures.**

What was done, all of it repointing rather than inventing:

- `__init__.py` now exports from the modules that actually define the symbols
- added `pyproject.toml` — there was no packaging file at all, so `pip install -e .` could never have worked
- `tests/` imports repointed; `bures_distance_dm` → `bures_distance`
- `Quasar` gained a real `curriculum=` switch (`uniform` | `error` | `progress`) that the tests had always called and the class never had
- `train_step` moved from `Learner` onto `QuantumGeometricTransformer`, where it already operated; `Learner` delegates. Behaviour identical
- `run_self_test` gained the `verbose=` argument the tests had always passed
- `tomography_bridge`'s `DummyGenerator` gained `sample_physics`, an interface method it was missing

**Measured after:** `pip install -e .` succeeds in a clean venv, `import quasar`
works, and `python run_all_tests.py` reports **ALL SUITES PASSED, exit 0** across
all nine suites. Sabotage-checked: corrupting one assertion in the QGT self-test
gives `SOME SUITES FAILED`, exit 1 — the runner can still fail.

---

## Quick Start

```
pip install -e .
python run_all_tests.py                          # all 9 suites
python -m quasar.quantum_geometric_transformer   # QGT self-test, 6 suites
python -m quasar.quasar                          # the closed loop
python -m quasar.quantum_geometric_rl            # QGT as control policy
```

---

## The loop

| Module | Role |
|---|---|
| `quasar/quasar.py` | **The loop.** Generator, Learner, difficulty binning, curriculum |
| `quasar/quantum_geometric_transformer.py` | The learner, plus the geometry it runs on |
| `quasar/quantum_geometric_rl.py` | Same model used as a control policy |
| `quasar/progress_curriculum.py` | Learning-progress sampler |
| `quasar/finite_shot_tomography.py` | Born-rule simulator, MLE reconstruction |
| `quasar/multi_qubit_tomography.py` | 15-dim generalized Bloch, superfidelity |
| `quasar/tomography_bridge.py` | Tomography wrapper for the generator |

Difficulty is stratified into bins by rotation speed, and the stratification is
*verified* — `test_difficulty_stratification` asserts measured Bures path length
increases across bins, so "harder bin" means something.

---

## Claims

**C1 ✅ Self-training transfers to unseen dynamics.** The learner improves on a
held-out set it never generated. `test_c1_self_training_improves`.

**C2 ✅ The curriculum self-directs.** Generation weights move away from uniform
by more than 0.05 in L1 without being told to. `test_c2_curriculum_self_directs`.

**C3 ❌ (kept) Adaptive sampling ties uniform at demo scale.** Measured in
`experiments/curriculum_scale.py`: error-driven wins 1/5 seeds at −0.08% vs
uniform; progress wins 2/5 at −0.21%. Neither beats the control at this size.
This is registered as a failure and kept.

### What the successor repo refuted

[quasar-v2](https://github.com/holland202/quasar-v2) carries findings F1–F18,
two of which bear directly on this README's own framing:

- **F18 refuted the founding premise.** Bures-metric attention does **not** beat
  plain dot-product attention: 0/5 seeds, mean gap +0.0007. The geometry is real
  mathematics; it is not the reason anything works here.
- **F16: error is not learnability** (corr = −0.999). Error-driven sampling
  measured 8.08% worse than progress-driven, 0/5. C3 above is the demo-scale
  shadow of that result.

Both are kept refutations. They are the most load-bearing things in this line.

---

## Scope

Classical simulation of single-qubit geometry. **No quantum hardware, no
quantum-advantage claim.** This is a principle demonstration: can a learner
manufacture its own curriculum and improve on dynamics it was never shown? C1
and C2 say yes at this scale. C3 says the adaptive part doesn't yet beat
uniform sampling at this scale.

v0.1 is the historical line and now runs. Active development is in
[quasar-v2](https://github.com/holland202/quasar-v2).

## License

MIT

*Vincit Omnia Veritas.*
