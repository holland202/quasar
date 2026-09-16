"""
QUASAR v0.1 -- Quantum-geometric Unified Self-training ARchitecture.

A closed-loop learner that GENERATES ITS OWN TRAINING DATA and directs its
own curriculum, for the case where human-authored training material runs out.
Generator -> curriculum -> learner, with no human corpus in the loop. The test
domain is single-qubit channel dynamics, chosen because it is exhaustively
checkable in NumPy on a phone and has a real difficulty axis.

Pure NumPy. No GPU. See README for what is verified and what was refuted.

Note on this file: until 2026-08-17 it imported from quasar.core,
quasar.gradients and quasar.experiment_logger -- modules that were never
written in this line. That single unreachable import made `import quasar`
fail on a cold clone even though six of seven modules run fine standalone.
The symbols below live where they always lived.

RSI extension (branch rsi-meta-curriculum):
  CurriculumGenome + MetaQuasar provide a bounded outer evolutionary loop
  that improves the curriculum policy parameters themselves.
"""

from quasar.quantum_geometric_transformer import (
    softmax,
    project_bloch,
    density_matrix,
    fidelity_bures,
    bures_distance,
    bures_distance_gradient,
    BuresAttention,
    QuantumGeometricFFN,
    GoldenRatioPositionalEncoding,
    QuantumGeometricTransformer,
)
from quasar.quasar import (
    so3,
    bures_path_length,
    Generator,
    Learner,
    Quasar,
    make_holdout,
)
from quasar.quantum_geometric_rl import (
    BlochControlEnv,
    QGTPolicy,
    evaluate_policy,
    train,
)
from quasar.rsi import (
    CurriculumGenome,
    MetaQuasar,
    run_rsi_demo,
)

__version__ = "0.1.1+rsi"
