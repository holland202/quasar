"""
QUASAR: Quantum-geometric Unified Self-training ARchitecture

A closed-loop AI that generates its own training data from the geometry
of its state space, and directs its own curriculum from its own errors.

Pure NumPy. No GPU. Verified end-to-end.
"""

from quasar.core import (
    project_bloch,
    bures_distance,
    bures_distance_gradient,
    bures_path_length,
    so3,
    softmax,
)
from quasar.quantum_geometric_transformer import (
    QuantumGeometricTransformer,
    BuresAttention,
    QuantumGeometricFFN,
    GoldenRatioPositionalEncoding,
)
from quasar.quasar import Quasar, Generator, Learner, make_holdout
from quasar.quantum_geometric_rl import (
    BlochControlEnv,
    QGTPolicy,
    train_policy,
    evaluate_policy,
)
from quasar.finite_shot_tomography import (
    MeasurementSimulator,
    StateReconstructor,
    TomographicTrajectoryGenerator,
)
from quasar.tomography_bridge import TomographicQuasarGenerator
from quasar.gradients import GradientEngine
from quasar.progress_curriculum import ProgressCurriculumSampler
from quasar.experiment_logger import ExperimentLogger

__version__ = "0.2.0"
__all__ = [
    "project_bloch",
    "bures_distance",
    "bures_distance_gradient",
    "bures_path_length",
    "so3",
    "softmax",
    "QuantumGeometricTransformer",
    "BuresAttention",
    "QuantumGeometricFFN",
    "GoldenRatioPositionalEncoding",
    "Quasar",
    "Generator",
    "Learner",
    "make_holdout",
    "BlochControlEnv",
    "QGTPolicy",
    "train_policy",
    "evaluate_policy",
    "MeasurementSimulator",
    "StateReconstructor",
    "TomographicTrajectoryGenerator",
    "TomographicQuasarGenerator",
    "GradientEngine",
    "ProgressCurriculumSampler",
    "ExperimentLogger",
]
