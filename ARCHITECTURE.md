# QUASAR Architecture

## Design Principles

1. **Manifold-native**: Every operation respects the Bloch ball constraint.
2. **Metric-native**: Loss is Bures distance, not MSE.
3. **Self-contained**: Pure NumPy, zero external ML dependencies.
4. **Honest**: Negative results are first-class artifacts.

## Module Dependencies

```

quasar/core.py
↓
quasar/gradients.py
↓
quasar/quantum_geometric_transformer.py
↓
quasar/quantum_geometric_rl.py
quasar/quasar.py
quasar/finite_shot_tomography.py
quasar/multi_qubit_tomography.py
quasar/tomography_bridge.py
quasar/progress_curriculum.py
quasar/experiment_logger.py

```

## Gradient Engine

`quasar/gradients.py` centralizes all gradient computation. Currently uses
central finite differences (verified correct). The `GradientEngine` class
allows swapping to analytical mode once full architecture backprop is
released.
