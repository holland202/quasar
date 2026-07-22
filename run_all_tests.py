"""Run every verification suite. Exit nonzero on any failure."""
import sys, subprocess
suites = [
    [sys.executable, "-m", "quasar.quantum_geometric_transformer"],
    [sys.executable, "-m", "quasar.quantum_geometric_rl"],
    [sys.executable, "-m", "quasar.quasar"],
]
for s in suites:
    print("\n" + "="*66 + f"\nRUNNING: {' '.join(s[1:])}\n" + "="*66)
    r = subprocess.run(s)
    if r.returncode != 0:
        sys.exit(f"FAILED: {s}")
print("\nALL SUITES PASSED")

# Finite-shot tomography
print("\n[4/4] Finite-shot tomography...")
from quasar.finite_shot_tomography import run_all_tests as tom_tests
tom_ok = tom_tests()
all_ok = all_ok and tom_ok
