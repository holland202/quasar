"""
Gate test for rsi_p0_precondition.py — proves the verdict fires BOTH ways.

Why this exists
---------------
The P0 sweep has only ever been observed returning `null` at every budget,
so its exit path has only ever been exercised in one direction. Nobody has
watched it return 0 for a passing budget. A gate seen firing in only one
direction is undemonstrated, not vacuous — but the distinction is invisible
until someone tries to make it fire the other way. This does that.

It does not modify the instrument. It imports the real `paired_verdict` and
calls the real `main()`, patching only module-level constants.

Runtime: a few seconds. No meaningful training happens — the budget is
patched down to (1, 2, 1), which is deliberately too small to learn
anything. This test measures the GATE, not the substrate.

SAFETY: `main()` writes `rsi_p0_results.json` into the current directory.
The committed artifact from the real device run lives at that exact path.
G0 below hashes it before and after and fails if it changed.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np

import rsi_p0_precondition as P

FAILURES: list[str] = []
PASSES = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSES
    if condition:
        PASSES += 1
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


def sha(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --- G0: the committed artifact must not be touched by this test -----------
ARTIFACT = "rsi_p0_results.json"
artifact_before = sha(ARTIFACT)


# --- V1..V4: the verdict function, driven with synthetic arrays ------------
print("\n[V] paired_verdict — synthetic arms")

base = np.full(8, 1.0)

# V1 clear PASS: candidate 5% better on every seed, tight spread
cand = np.full(8, 0.95) + np.linspace(-0.001, 0.001, 8)
v = P.paired_verdict(base, cand)
check("V1 clear win -> beats_bar", v["beats_bar"] and v["resolvable"],
      f"rel={v['mean_rel_improvement']:.4f} wins={v['wins']} "
      f"sem={v['sem_rel']:.5f}")

# V2 clear null: 0.2% better, well resolved. This is the shape of the real
# 240-unit result.
cand = np.full(8, 0.998) + np.linspace(-0.0005, 0.0005, 8)
v = P.paired_verdict(base, cand)
check("V2 tiny effect -> null, still resolvable",
      (not v["beats_bar"]) and v["resolvable"],
      f"rel={v['mean_rel_improvement']:.4f} sem={v['sem_rel']:.5f}")

# V3 VOID: big mean improvement but the spread swamps it. Must be flagged
# unresolvable, NOT reported as a win.
rng = np.random.default_rng(0)
cand = np.full(8, 0.95) + rng.normal(0, 0.15, 8)
v = P.paired_verdict(base, cand)
check("V3 noisy arm -> not resolvable (VOID)", not v["resolvable"],
      f"sem={v['sem_rel']:.5f} bar={P.SEM_BAR}")

# V4 sign test governs: 5% mean improvement carried by a few seeds only.
cand = np.array([0.55, 0.60, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02])
v = P.paired_verdict(base, cand)
check("V4 magnitude without sign majority -> no pass",
      not v["beats_bar"],
      f"rel={v['mean_rel_improvement']:.4f} wins={v['wins']}/8")

# V5 direction: lower loss must read as positive improvement.
v = P.paired_verdict(np.full(8, 1.0), np.full(8, 0.9))
check("V5 lower loss reads as positive improvement",
      v["mean_rel_improvement"] > 0,
      f"rel={v['mean_rel_improvement']:.4f}")


# --- E1/E2: the real main() exit path, both directions ---------------------
print("\n[E] main() exit code — both directions")

TINY = [(1, 2, 1)]          # too small to learn; we are testing the gate
saved = (P.BUDGETS, P.REL_BAR, P.TEST_SEEDS)
workdir = tempfile.mkdtemp(dir=os.path.expanduser("~"), prefix="p0gate_")
cwd = os.getcwd()

try:
    os.chdir(workdir)

    # E1 — bar unreachable: no budget can pass, main must return 1.
    P.BUDGETS = TINY
    P.REL_BAR = 0.99
    P.TEST_SEEDS = [9001, 9002, 9003, 9004]
    rc_fail = P.main()
    check("E1 no budget passes -> main() returns 1", rc_fail == 1,
          f"got {rc_fail}")

    # E2 — bar set below any achievable contrast: something must pass, and
    # main must return 0. Same code path, opposite verdict.
    P.REL_BAR = -1.0
    P.SIGN_BAR = 0
    rc_pass = P.main()
    check("E2 a budget passes -> main() returns 0", rc_pass == 0,
          f"got {rc_pass}")

    check("E1/E2 differ (gate is not constant)", rc_fail != rc_pass,
          f"{rc_fail} vs {rc_pass}")
finally:
    os.chdir(cwd)
    P.BUDGETS, P.REL_BAR, P.TEST_SEEDS = saved
    for f in os.listdir(workdir):
        os.remove(os.path.join(workdir, f))
    os.rmdir(workdir)


# --- S1: the __main__ guard actually wires main() to the process exit ------
print("\n[S] source wiring")
src = open(P.__file__, encoding="utf-8").read()
check("S1 __main__ guard calls sys.exit(main())",
      "sys.exit(main())" in src)


# --- G0 concluded ----------------------------------------------------------
print("\n[G] artifact protection")
check("G0 committed rsi_p0_results.json unchanged",
      sha(ARTIFACT) == artifact_before,
      "the real device artifact was overwritten — investigate before commit")


# --- verdict ---------------------------------------------------------------
print("\n" + "=" * 58)
if FAILURES:
    print(f"FAILED: {len(FAILURES)} of {PASSES + len(FAILURES)} — "
          + ", ".join(FAILURES))
    print("=" * 58)
    sys.exit(1)
print(f"ALL {PASSES} GATE CHECKS PASSED")
print("The P0 verdict has now been demonstrated in both directions.")
print("=" * 58)
sys.exit(0)
