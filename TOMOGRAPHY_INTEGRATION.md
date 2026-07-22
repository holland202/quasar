# Integration Guide: Finite-Shot Tomography into QUASAR

## Quick wire

```python
from quasar.finite_shot_tomography import TomographicTrajectoryGenerator

tomogen = TomographicTrajectoryGenerator(
    base_generator=your_generator,
    shots=512,
    n_bases=6,
    reconstructor_method='mle',
)

result = tomogen.generate_trajectory(n_steps=10)
states = result['reconstructed_states']  # list of (3,) Bloch vectors
Performance
 
MLE: ~12ms/state (512 shots, 6 bases)
 
Linear inversion: ~0.1ms/state (faster, can be non-physical)
 
Android Termux: ~50ms/state
Verified metricsScaling: ~1/sqrt(shots)
