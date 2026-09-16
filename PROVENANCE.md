# Provenance

## Identity

- **Repository**: holland202/quasar
- **Canonical URL**: https://github.com/holland202/quasar
- **Purpose**: Implementation and experimentation with a closed-loop self-training architecture (generator → curriculum → learner) on single-qubit channel dynamics. Pure NumPy. No quantum hardware.
- **License**: MIT (see LICENSE)

## Origin

Chad Holland initiated the work. The earliest public repository history and commit record support this attribution. No earlier independent public origin is documented in this repository.

## Contributions

See AUTHORS.md for the category breakdown. All primary categories currently map to Chad Holland on the basis of repository history.

## Research Status Distinction

| Status        | Meaning in this repository                                      |
|---------------|-----------------------------------------------------------------|
| IMPLEMENTED   | Code and tests exist and can be executed                        |
| EXPERIMENTAL  | Experiments have been run under the conditions described        |
| VERIFIED      | Not claimed here; requires independent verification             |
| REPRODUCED    | No independent reproduction is recorded in this repository      |
| REFUTED       | Specific claims that failed are retained (see below)            |
| UNRESOLVED    | Open questions remain                                           |
| NOT TESTED    | Explicitly out of scope or not yet attempted                    |

“Implemented” or “experimental” must not be read as “scientifically validated.”

## Experimental Lineage

Significant experiments and results recorded in this repository (and its successor line) include:

- Self-training transfer (C1) — status recorded as supported under the tests present in this repository.
- Curriculum self-direction (C2) — status recorded as supported under the tests present in this repository.
- Adaptive sampling vs uniform at demo scale (C3) — recorded as failure; retained.
- References to successor findings (quasar-v2 F16, F18) that refute or qualify earlier framing are retained rather than removed.

### RSI-1 — bounded recursive self-improvement via meta-curriculum

Branch `rsi-meta-curriculum`. An outer evolutionary loop over curriculum
policy parameters (`CurriculumGenome`: temperature, floor, progress_mix)
around a frozen QGT, a frozen generator, and a held-out Bures evaluator.

Registered claims R1 and R2 are **NOT TESTED**. The experiment was stopped
at its prerequisite and the outer loop was never run.

```
RSI-1
  └── P0 substrate qualification  (rsi_p0_precondition.py)
       ├──  80 work units  -> resolved null
       ├── 240 work units  -> resolved null
       └── 720 work units  -> resolved null
              |
              v
         P0b = FALSE
              |
              v
       RSI-1 = NOT ADMISSIBLE (VOID)
              |
              +-- R1 = NOT TESTED
              +-- R2 = NOT TESTED
```

| Field | Value |
|---|---|
| Candidate revision | `061368f` (`rsi-meta-curriculum`) |
| Record commit | `894955b` |
| Evaluator | held-out Bures loss, `make_holdout(seed=777, n=16)`, unmodified |
| Seed hash | `10c3814e9f7c178b` |
| Seeds | SELECT 1001-1008 reserved and unused; TEST 9001-9008 used; disjoint |
| Budgets | (5,8,2), (10,12,2), (15,16,3) work units 80 / 240 / 720 |
| Environment | Python 3.14.6, NumPy 2.4.4, Android-16-aarch64 |
| Runtime | 2594.6 s |
| Artifacts | `RSI1_P0_RECORD.md`, `rsi_p0_results.json`, `rsi_p0_precondition.py` |
| Status | NOT ADMISSIBLE (VOID) |

P0a, the anti-vacuity control, passed at every budget: paired SEMs of 0.0007
to 0.0024 against a 0.01 resolution bar. These are resolved nulls, not
underpowered measurements. Uniform holdout loss fell 0.23433 -> 0.21208 ->
0.18027 across the ladder, so the substrate learns; curriculum contrast
specifically is absent.

Scope: this establishes that the RSI-1 substrate lacks the curriculum
headroom the registered outer-loop test requires, at the three budgets tested
and at fixed model capacity. It does not establish anything about recursive
self-improvement in general, and it does not refute R1 or R2.

Unresolved and carried forward: an unexplained direction reversal in the
`error` contrast between 240 and 720 work units, logged as an observation
with no mechanism; and one unrun prediction, P0d, raising model capacity to
the quasar-v2 F16 regime before re-testing the precondition.

Detail in `RSI1_P0_RECORD.md` on the `rsi-meta-curriculum` branch.

Detailed per-experiment fields (exact parent/candidate revisions, evaluator hashes, full seed lists, artifact hashes) are not yet centralized in a machine-readable manifest in this repository. Where present, they appear in experiment scripts, test outputs, or the successor repository.

## Independent Reproduction

None recorded in this repository.

## Refutation / Failure Record

Failures are first-class:

- C3 (adaptive sampling does not beat uniform at the recorded demo scale) is kept.
- References to F18 (Bures-metric attention does not beat plain attention under the recorded conditions) and F16 (error-driven vs progress-driven) from the successor line are retained as load-bearing negative results.

Original claims, observed limitations, and subsequent corrections remain visible in the history.

## Corrections

Methodological and packaging corrections (including the 2026-08-17 restoration of importability and test alignment) are documented in the README and commit history. Corrections are additive; prior failing states are not erased.

## Scope of Evidence

This repository documents implementation and experimentation on a classical simulation of single-qubit geometry.

It does **not** establish:

- recursive self-improvement
- general intelligence
- autonomy
- quantum advantage
- performance on quantum hardware
- independent scientific validation of the recorded results
- generalization beyond the described experimental conditions

Successful execution of the code and tests does not by itself establish any of the above.

## Relationship to Other Repositories

This repository is an instrument. Cross-repository relationships, promotion gates, and higher-level research-record functions (if any) are intended to live in a separate research-operating layer rather than inside this codebase.
