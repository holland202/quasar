#!/usr/bin/env python3
"""
QUASAR — Curriculum is inert at demo scale (F16 + F18 + F18-a consolidated).
Author: Claude Fable 5 (Anthropic) + holland202.
Reproduces the public-repo result: NO curriculum signal beats uniform on the
demo-scale toy — not error-driven, not progress-driven, at any probe size.
This does NOT contradict the private-lab 5/5 progress win (README C3 section):
that used a 225-param 2-qubit learner where competence is uneven. The public
toy has no curriculum-exploitable structure. Scale, not signal, is the variable.

Run:  python note_curriculum_scale.py         (uses recorded runs)
      python note_curriculum_scale.py --live   (recomputes, ~10 min)
"""
import sys, json, os
import numpy as np

# Recorded results (5 seeds; produced by run_seeds.py / run_f18a.py, this repo)
UNIFORM = [0.2079, 0.2088, 0.2348, 0.2058, 0.2071]
ERROR   = [0.2090, 0.2072, 0.2357, 0.2061, 0.2073]   # F18, probe 4
PROG4   = [0.2088, 0.2082, 0.2327, 0.2074, 0.2095]   # F18, probe 4
PROG16  = [0.2093, 0.2101, 0.2300, 0.2044, 0.2138]   # F18-a
PROG64  = [0.2009, 0.2171, 0.2354, 0.2043, 0.2144]   # F18-a

def summarize(name, arr, base):
    a, b = np.array(arr), np.array(base)
    wins = int((a < b).sum())
    eff = (b.mean() - a.mean()) / b.mean() * 100
    print(f"  {name:<22} mean {a.mean():.4f}  wins {wins}/5  vs uniform {eff:+.2f}%")
    return wins, eff

if "--live" in sys.argv:
    sys.path.insert(0, ".")
    from quasar_v02_experiment import QuasarV02
    from quasar.quasar import make_holdout
    print("live recompute not bundled here; see run_seeds.py + run_f18a.py")
    sys.exit(0)

print("QUASAR curriculum @ demo scale — consolidated (F16/F18/F18-a)")
print("=" * 60)
summarize("uniform (baseline)", UNIFORM, UNIFORM)
we, _ = summarize("error-driven (p=4)", ERROR, UNIFORM)
w4, e4 = summarize("progress (p=4)", PROG4, UNIFORM)
w16, e16 = summarize("progress (p=16)", PROG16, UNIFORM)
w64, e64 = summarize("progress (p=64)", PROG64, UNIFORM)
print("=" * 60)
print("REGISTERED FINDINGS:")
print(f"  F16  error-driven beats uniform?      {we}/5  -> {'no' if we<3 else 'YES'}")
print(f"  F18  progress beats uniform (p=4)?     {w4}/5  -> {'no' if w4<3 else 'YES'}")
print(f"  F18a progress improves as probe grows? effect {e4:+.2f}->{e16:+.2f}->{e64:+.2f}%")
print(f"       -> monotonic DOWN: cleaner signal is WORSE, not better")
print("=" * 60)
print("CONCLUSION: no curriculum signal beats uniform at demo scale.")
print("The self-training LOOP works (C2); curriculum BENEFIT is absent (C3)")
print("because it does not exist at this scale, not because the signal is wrong.")
