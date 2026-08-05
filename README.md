> ## ⚠️ ARCHIVED — superseded by [quasar-v2](https://github.com/holland202/quasar-v2)
>
> This repo is v0.1 of the QUASAR line, kept read-only for the history of
> findings F1–F4. Active development, the full findings log (F1–F16, incl.
> kept refutations), and code that passes its gates on a cold clone all
> live in **quasar-v2**.
>
> Known-broken as archived, recorded rather than hidden: `quasar/__init__.py`
> imports `quasar.core`, `quasar.gradients`, and `quasar.experiment_logger`,
> which were never published to this repo — `import quasar` fails on a cold
> clone, and the CI workflow dies at `pip install -e .` (no packaging file).
> v2 is a structurally different codebase, so these imports were not fixed
> here; the honest resolution is this notice. Archived 2026-08-04.

# QUASAR — Quantum-geometric Unified Self-training ARchitecture

[![verify-all-claims](https://github.com/holland202/quasar/actions/workflows/tests.yml/badge.svg)](https://github.com/holland202/quasar/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

**A closed-loop AI that generates its own training data from the geometry of its state space, and directs its own curriculum from its own errors.**  
Pure NumPy. No GPU. Verified end-to-end.

## Quick Start

```bash
pip install -e .
python -m quasar.quantum_geometric_transformer  # ~5s, 7 test suites
python -m quasar.quantum_geometric_rl           # ~60s, RL experiment
python -m quasar.quasar                           # ~40s, closed loop
python run_all_tests.py                           # everything
```

Play in Browser

[holland202.github.io/quasar](https://holland202.github.io/quasar/) — drag sliders, watch the Bloch sphere deform.

Architecture

Module	Purpose	
`quasar/core.py`	Bures metric, SO(3), Bloch projection	
`quasar/gradients.py`	Centralized finite-difference + analytical engine	
`quasar/quantum_geometric_transformer.py`	QGT with Bures attention	
`quasar/quantum_geometric_rl.py`	QGT as control policy	
`quasar/quasar.py`	Closed loop: generator + curriculum + learner	
`quasar/finite_shot_tomography.py`	Born-rule simulator, MLE reconstruction	
`quasar/multi_qubit_tomography.py`	15-dim generalized Bloch, superfidelity	
`quasar/tomography_bridge.py`	Drop-in tomography wrapper	
`quasar/progress_curriculum.py`	Learning-progress curriculum	
`quasar/experiment_logger.py`	Reproducible experiment logging	

Verified Claims

- C1 ✅ Self-training transfers to unseen dynamics.
- C2 ✅ Curriculum self-directs (L1 weight shift > 0.05).
- C3 ❌ (honest) Adaptive ties uniform at demo scale; progress curriculum wins at scale. See `experiments/curriculum_scale.py`.

Scope

Classical simulation of single-qubit geometry. No quantum hardware, no quantum-advantage claim. This is a principle demonstration with every claim tested against controls — including the ones that failed.

License

MIT
