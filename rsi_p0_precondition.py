"""
RSI-1 / P0 — SUBSTRATE PRECONDITION SWEEP
=========================================

Registered BEFORE running (2026-09-16).

Question this answers, and nothing else:
    At what inner training budget, if any, does a FIXED curriculum contrast
    become detectable? The RSI outer loop evolves curriculum parameters. If
    no fixed curriculum beats uniform at a given budget, there is no signal
    for the meta-layer to find and any positive RSI result at that budget is
    a selection artifact.

This is the precondition. It is NOT an RSI experiment and makes no RSI claim.

REGISTERED PREDICTIONS
    P0a (anti-vacuity / instrument sanity): at every budget, the paired
        standard error must be small enough to resolve a 3% relative effect.
        Concretely: sem_rel < 0.01. If this fails the budget is UNDERPOWERED
        and its verdict is VOID, not NULL.
    P0b (the precondition itself): at some budget in the ladder, a fixed
        curriculum (error or progress) beats uniform, paired on matched
        seeds, by > 3% relative, with a sign test at >= 6 of 8 seeds.
    P0c (direction, from quasar-v2 F16): if any budget passes P0b, the
        winner is `progress`, not `error`. F16 measured error-driven as
        8.08% WORSE than uniform at 135 params, 0/5 seeds.

A budget that passes P0a and fails P0b is a real null at that budget.
A budget that fails P0a tells us nothing at all.

Exit code: 0 if at least one budget passes P0b. 1 if none do.
A nonzero exit means: DO NOT RUN THE OUTER LOOP. There is nothing to find.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time

import numpy as np

from quasar.quasar import Quasar, make_holdout

# --- Seed discipline -------------------------------------------------------
# SELECT seeds are what any future outer loop is allowed to see.
# TEST seeds are disjoint and reserved. This sweep uses TEST seeds only,
# because it is measuring the substrate, not selecting anything.
SELECT_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
TEST_SEEDS = [9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008]
assert not set(SELECT_SEEDS) & set(TEST_SEEDS), "seed sets must be disjoint"

SEED_HASH = hashlib.sha256(
    json.dumps({"select": SELECT_SEEDS, "test": TEST_SEEDS},
               sort_keys=True).encode()
).hexdigest()[:16]

REL_BAR = 0.03          # P0b: 3% relative improvement over uniform
SIGN_BAR = 6            # P0b: wins on >= 6 of 8 paired seeds
SEM_BAR = 0.01          # P0a: paired sem must resolve better than 1% relative

# rounds, n_per_round, epochs_per_round
BUDGETS = [
    (5, 8, 2),      # the budget MetaQuasar actually used: 80 work units
    (10, 12, 2),    # 3x
    (15, 16, 3),    # 9x
]

ARMS = ("uniform", "error", "progress")


def evaluate(curriculum: str, seed: int, budget) -> float:
    rounds, n_per_round, epochs = budget
    Xh, Yh = make_holdout(seed=777, n=16, seq_len=6)
    q = Quasar(seed=seed, n_bins=4, seq_len=6, curriculum=curriculum)
    hist = q.run(rounds=rounds, n_per_round=n_per_round,
                 epochs_per_round=epochs, holdout=(Xh, Yh),
                 adaptive=True, verbose=False)
    loss = float(hist["holdout"][-1])
    if not np.isfinite(loss):
        raise AssertionError(f"non-finite loss: {curriculum} seed={seed}")
    return loss


def paired_verdict(base: np.ndarray, cand: np.ndarray) -> dict:
    """base = uniform, cand = the challenger. Lower loss is better."""
    d = cand - base                      # negative means candidate wins
    rel = -d / base                       # positive means candidate wins
    n = d.size
    sd = float(np.std(rel, ddof=1))
    sem = sd / np.sqrt(n)
    wins = int((d < 0).sum())
    return {
        "n": n,
        "mean_rel_improvement": float(rel.mean()),
        "sd_rel": sd,
        "sem_rel": float(sem),
        "wins": wins,
        "resolvable": bool(sem < SEM_BAR),          # P0a
        "beats_bar": bool(rel.mean() > REL_BAR and wins >= SIGN_BAR),  # P0b
    }


def main() -> int:
    t0 = time.time()
    record = {
        "experiment": "RSI-1/P0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_hash": SEED_HASH,
        "select_seeds": SELECT_SEEDS,
        "test_seeds": TEST_SEEDS,
        "bars": {"rel": REL_BAR, "sign": SIGN_BAR, "sem": SEM_BAR},
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "budgets": [],
    }

    any_pass = False
    for budget in BUDGETS:
        units = budget[0] * budget[1] * budget[2]
        print(f"\n=== budget rounds={budget[0]} n={budget[1]} "
              f"epochs={budget[2]}  ({units} work units) ===", flush=True)
        losses = {}
        for arm in ARMS:
            losses[arm] = np.array([evaluate(arm, s, budget)
                                    for s in TEST_SEEDS])
            print(f"  {arm:9s} mean={losses[arm].mean():.5f} "
                  f"min={losses[arm].min():.5f}", flush=True)

        entry = {"budget": list(budget), "work_units": units,
                 "losses": {k: v.tolist() for k, v in losses.items()},
                 "contrasts": {}}
        for arm in ("error", "progress"):
            v = paired_verdict(losses["uniform"], losses[arm])
            entry["contrasts"][f"{arm}_vs_uniform"] = v
            flag = ("VOID (underpowered)" if not v["resolvable"]
                    else "PASS" if v["beats_bar"] else "null")
            print(f"    {arm:9s} vs uniform: rel={v['mean_rel_improvement']:+.4f} "
                  f"sem={v['sem_rel']:.4f} wins={v['wins']}/{v['n']} -> {flag}",
                  flush=True)
            if v["resolvable"] and v["beats_bar"]:
                any_pass = True
        record["budgets"].append(entry)

    record["P0b_any_budget_passes"] = any_pass
    record["runtime_s"] = round(time.time() - t0, 1)
    with open("rsi_p0_results.json", "w") as f:
        json.dump(record, f, indent=2)

    print("\n" + "=" * 60)
    print(f"P0b (a fixed curriculum beats uniform by >{REL_BAR:.0%} at some "
          f"budget): {any_pass}")
    if not any_pass:
        print("VERDICT: no curriculum headroom at any tested budget.")
        print("The RSI outer loop has nothing to optimize. Do not run it.")
    print(f"seed_hash={SEED_HASH}  runtime={record['runtime_s']}s")
    print("=" * 60)
    return 0 if any_pass else 1


if __name__ == "__main__":
    sys.exit(main())
