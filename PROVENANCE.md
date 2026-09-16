# Provenance

## Identity

- **Repository**: holland202/quasar
- **Canonical URL**: https://github.com/holland202/quasar
- **Purpose**: Implementation and experimentation on a closed-loop learner that generates its own training data (generator → curriculum → learner) in the single-qubit channel dynamics domain.
- **License**: MIT

## Origin

Initiated by Chad Holland. Earliest identifiable repository history supports origin and primary implementation by the same individual.

## Contributions

See `AUTHORS.md` for category-level attribution. All primary categories currently map to Chad Holland on the basis of repository history.

## Research Status Distinction

| Status | Meaning in this repository |
|--------|----------------------------|
| IMPLEMENTED | Code and tests exist and run |
| EXPERIMENTAL | Results obtained under registered conditions |
| VERIFIED | Not claimed; requires independent verification protocol |
| REPRODUCED | No independent third-party reproductions recorded |
| REFUTED | Explicitly recorded failures retained |
| UNRESOLVED | Open questions remain |
| NOT TESTED | Items outside current experimental scope |

**Implemented does not mean scientifically validated.**

## Experimental Lineage

### Claim / Experiment C1 — Self-training transfers to unseen dynamics

- **Status**: EXPERIMENTAL (registered positive under local tests)
- **Parent revision**: historical line culminating in current main
- **Result summary**: Learner improves on held-out set it never generated (`test_c1_self_training_improves`)
- **Notes**: Local test suite evidence only. Not independently reproduced in this record.

### Claim / Experiment C2 — Curriculum self-directs

- **Status**: EXPERIMENTAL (registered positive under local tests)
- **Result summary**: Generation weights move away from uniform by more than 0.05 in L1 without external instruction (`test_c2_curriculum_self_directs`)
- **Notes**: Local test suite evidence only.

### Claim / Experiment C3 — Adaptive sampling vs uniform at demo scale

- **Status**: REFUTED (kept)
- **Result summary**: Error-driven wins 1/5 seeds at −0.08% vs uniform; progress wins 2/5 at −0.21%. Neither beats control at this scale.
- **Failure mode**: Adaptive curricula do not outperform uniform at the registered demo scale.
- **Evidence**: `experiments/curriculum_scale.py` and associated measurements.
- **Affected conclusion**: Adaptive sampling advantage is not established at this scale.
- **Correction / subsequent**: Retained as failure. Related stronger negative results appear in successor repository (see below).

### Related results from successor line (quasar-v2)

- **F18**: Bures-metric attention does not beat plain dot-product attention (0/5 seeds). Founding geometric premise refuted for the measured regime. Status: REFUTED (kept).
- **F16**: Error-driven sampling measured worse than progress-driven. Status: REFUTED for the error-as-learnability framing (kept).

These are referenced, not absorbed. Full artifacts live in the successor repository.

## Independent Reproduction

No independent third-party reproduction records are present in this repository.

## Refutation / Failure Record

Failures are first-class:

- C3 retained as registered failure.
- Packaging/import failures (2026-07-26 to 2026-08-17) documented in README and fixed by repointing rather than invention.
- Successor-line refutations (F18, F16) explicitly preserved and linked.

## Corrections

- 2026-08-17: Import and packaging repairs. Cause was an `__init__.py` written against a planned refactor that never landed. Tests and packaging were aligned to existing modules. No experimental results were altered.

## Scope of Evidence

This repository documents implementation and local experimentation in classical simulation of single-qubit geometry.

It does **not** establish:

- recursive self-improvement
- general intelligence
- autonomy
- quantum advantage
- human-level novelty
- results that survive independent reproduction or multi-regime generalization

Successful execution of the test suite and registered experiments does not by itself establish any of the above.

Active development and stronger experimental claims (including retained refutations) are tracked in the successor repository [quasar-v2](https://github.com/holland202/quasar-v2).
