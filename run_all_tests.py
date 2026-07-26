#!/usr/bin/env python3
"""
QUASAR - Run all test suites
Usage: python run_all_tests.py
"""

import subprocess
import sys

def run_suite(name, module):
    print("")
    print("=" * 66)
    print("RUNNING: -m " + module)
    print("=" * 66)
    print("")
    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print("ERROR running " + module + ": " + str(e))
        return False

if __name__ == "__main__":
    suites = [
        ("Quantum Geometric Transformer", "quasar.quantum_geometric_transformer"),
        ("Quantum Geometric RL", "quasar.quantum_geometric_rl"),
        ("QUASAR Closed Loop", "quasar.quasar"),
        ("Finite-shot Tomography", "quasar.finite_shot_tomography"),
        ("Multi-qubit Tomography", "quasar.multi_qubit_tomography"),
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
