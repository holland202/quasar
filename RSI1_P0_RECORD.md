# RSI-1 / P0 — Substrate Qualification Record

**Status: NOT ADMISSIBLE (VOID). The RSI-1 outer loop was not executed.**

This is not a refutation of recursive self-improvement, and not a refutation
of RSI-1's claims R1/R2. R1 and R2 were never tested. The experiment failed
its registered prerequisite and was stopped before the outer loop ran.

---

## What failed, first

`P0b` — the precondition that a *fixed* curriculum beats uniform by more than
3% at some training budget — returned **False at every budget tested**.

The RSI-1 outer loop evolves curriculum policy parameters. If no fixed
curriculum beats uniform at a given budget, there is no signal for the
meta-layer to find, and any positive RSI result at that budget would be a
selection artifact rather than an improvement. The prerequisite exists to
catch exactly that. It did.

---

## Provenance

| Field | Value |
|---|---|
| Experiment | RSI-1 / P0 |
| Instrument | `rsi_p0_precondition.py` |
| Repository | holland202/quasar |
| Branch | `rsi-meta-curriculum` |
| Commit under test | `061368f1eb6a776dc824c03255026928d4c533d7` |
| Seed hash | `10c3814e9f7c178b` |
| SELECT seeds (reserved, unused) | 1001–1008 |
| TEST seeds (used) | 9001–9008 |
| Runtime | 2594.6 s |
| Device | Samsung Galaxy S25 Ultra, Termux, aarch64 |
| Environment detail | recorded in `rsi_p0_results.json` |
| Artifact | `rsi_p0_results.json` |
| Log | `rsi_p0_device.log` |

Registered bars, fixed before the run: relative improvement > 3%, sign test
≥ 6 of 8 seeds, paired SEM < 0.01 for the contrast to be resolvable at all.

---

## Results — all three budgets

Lower holdout loss is better. Positive `rel` means the challenger beats uniform.

### 80 work units (rounds=5, n=8, epochs=2) — the budget MetaQuasar used

```
  uniform   mean=0.23433 min=0.22135
  error     mean=0.23497 min=0.22028
  progress  mean=0.23444 min=0.21921
    error     vs uniform: rel=-0.0027 sem=0.0013 wins=1/8 -> null
    progress  vs uniform: rel=-0.0005 sem=0.0024 wins=4/8 -> null
```

### 240 work units (rounds=10, n=12, epochs=2)

```
  uniform   mean=0.21208 min=0.19925
  error     mean=0.21165 min=0.19868
  progress  mean=0.21172 min=0.19997
    error     vs uniform: rel=+0.0020 sem=0.0007 wins=7/8 -> null
    progress  vs uniform: rel=+0.0016 sem=0.0021 wins=5/8 -> null
```

### 720 work units (rounds=15, n=16, epochs=3)

```
  uniform   mean=0.18027 min=0.16832
  error     mean=0.18067 min=0.16950
  progress  mean=0.18078 min=0.16780
    error     vs uniform: rel=-0.0022 sem=0.0014 wins=2/8 -> null
    progress  vs uniform: rel=-0.0027 sem=0.0020 wins=3/8 -> null
```

### Verdict

```
P0b (a fixed curriculum beats uniform by >3% at some budget): False
VERDICT: no curriculum headroom at any tested budget.
The RSI outer loop has nothing to optimize. Do not run it.
seed_hash=10c3814e9f7c178b  runtime=2594.6s
```

---

## This is a resolved null, not an underpowered one

`P0a` is the anti-vacuity control: the paired SEM must be small enough to
resolve a 3% relative effect, i.e. below 0.01. Every contrast cleared it by a
wide margin — SEMs ran 0.0007 to 0.0024, between 4× and 14× below the bar.

The instrument could have detected a 3% effect. There wasn't one. That
distinction is the whole reason `P0a` exists: a budget that fails it would
have been marked VOID and told us nothing, rather than being reported as a
null.

The substrate itself is learning. Uniform holdout loss falls cleanly with
budget: 0.23433 → 0.21208 → 0.18027. The task is not saturated and the model
is not broken. What is absent is *curriculum contrast*, specifically.

---

## Scope of this result

Established:

> Under the registered P0 protocol, substrate performance continued improving
> with increased budget, but neither tested curriculum produced the
> predefined >3% improvement over uniform at any tested budget. The RSI-1
> outer-loop experiment therefore has no qualifying optimization headroom
> under this substrate and protocol, and is not admissible to run.

Not established:

- that recursive self-improvement is impossible
- that RSI-1's claims R1 or R2 are false — they were not tested
- that curriculum learning does not work in general
- that a larger budget, higher model capacity, or a different task would
  also yield no headroom. Only three budgets on one substrate were tested.

## Relation to the earlier RSI demo run

`quasar/rsi.py`'s `run_rsi_demo()` prints `R1 ... True`. That output is not
evidence and should not be cited. `MetaQuasar.best_fitness` is a running
minimum over 15 genome evaluations at varying seeds, while each fixed
curriculum is evaluated once at `seed=0`. Comparing a min-of-15 against a
single draw inflates the apparent winner by the spread of the evaluation
noise, which here is far larger than any curriculum effect. `R2`'s threshold
is likewise satisfied by the hand-seeded starting population before any
evolution occurs.

Independently of P0, that comparison does not support R1.

## Unresolved

**Direction reversal across budget regimes.** The `error` contrast runs
+0.0020 (7/8 wins) at 240 units and −0.0022 (2/8 wins) at 720 units. Both
sit roughly 15× below the 3% decision bar, six contrasts were computed across
the sweep, and neither is far from its own SEM. Recorded as an observation
with no mechanism attached. It is not a finding and should not be cited as
one. Cross-comparison with quasar-v2's F16 (error-driven measured 8.08% worse
than uniform at 135 parameters, 0/5 seeds) is pending and should not be
narrated into agreement or disagreement without an experiment designed to
settle it.

**Higher-capacity substrate untested.** F16 found its curriculum effect at
225 parameters, not 135. This sweep varied training budget only, holding
model capacity fixed. A capacity sweep is the obvious unrun door.

## Open prediction (unrun)

> **P0d:** at 720 work units with model capacity raised to the F16 regime,
> `progress` beats `uniform` by more than 3% paired on the TEST seeds.

If P0d passes, RSI-1 becomes admissible at that configuration, and must then
run with SELECT seeds 1001–1008 only — the TEST seeds stay reserved.

---

## Note on capturing the verdict

The gate returns a nonzero exit code when `P0b` is False. Capturing it
through a pipe discards it:

```bash
python rsi_p0_precondition.py | tee log.txt; echo $?   # reports tee's status
```

Use one of:

```bash
set -o pipefail
python rsi_p0_precondition.py | tee log.txt; echo "rc=$?"
# or
python rsi_p0_precondition.py | tee log.txt; echo "rc=${PIPESTATUS[0]}"
```

The first device run reported `rc=0` for this reason while the instrument had
correctly returned 1. The verdict was right; the capture was not.

---

## Contributors

Chad Holland — direction, device execution, environment.
Claude (Anthropic) — P0 protocol design, instrument implementation,
container pre-run. Grok, ChatGPT — earlier framing of the RSI-1 experiment
and the evidence-record structure.
