#!/usr/bin/env python3
"""
QUASAR — Run all verification suites.
Usage: python run_all_tests.py
"""

import subprocess
import sys


def run_suite(name, module):
    print("")
    print("=" * 66)
    print(f"RUNNING: {module}")
    print("=" * 66)
    print("")
    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR running {module}: {e}")
        return False


if __name__ == "__main__":
    suites = [
        ("Core Math", "tests.test_core"),
        ("Transformer", "tests.test_transformer"),
        ("QUASAR Closed Loop", "tests.test_quasar"),
        ("Tomography", "tests.test_tomography"),
        ("QGT Self-Test", "quasar.quantum_geometric_transformer"),
        ("QGRL", "quasar.quantum_geometric_rl"),
        ("Finite-Shot Tomography", "quasar.finite_shot_tomography"),
        ("Multi-Qubit Tomography", "quasar.multi_qubit_tomography"),
        ("Curriculum Scale", "experiments.curriculum_scale"),
    ]

    all_ok = True
    for name, module in suites:
        ok = run_suite(name, module)
        all_ok = all_ok and ok

    print("")
    print("=" * 66)
    if all_ok:
        print("ALL SUITES PASSED")
        sys.exit(0)
    else:
        print("SOME SUITES FAILED")
        sys.exit(1)
