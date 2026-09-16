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
