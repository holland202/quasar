# QUASAR RSI Extension — Bounded Recursive Self-Improvement

> **STATUS: NOT ADMISSIBLE (VOID). The outer loop has not been run.**
> The registered prerequisite P0 failed on device: no fixed curriculum
> beats uniform by >3% at any tested budget (80 / 240 / 720 work units),
> with paired SEMs 4x-14x below the resolution bar. R1 and R2 below are
> registered claims that were **never tested**, not results. See
> [RSI1_P0_RECORD.md](RSI1_P0_RECORD.md).

This branch adds a **meta-curriculum evolutionary loop** on top of the existing Quasar closed loop.

## What was added

| File | Role |
|------|------|
| `quasar/rsi.py` | `CurriculumGenome` + `MetaQuasar` outer evolutionary loop |
| `README_RSI.md` | This document |

The original `quasar/quasar.py` loop is **unchanged**. The RSI layer sits outside it.

## Design (honest scope)

**Inner loop** (unchanged):
```
Generator → difficulty bins → Learner (QGT) → error / progress weights → next generation
```

**Outer loop** (new):
```
Population of CurriculumGenomes
  ↓ evaluate each under fixed training budget on held-out dynamics
  ↓ select elites + mutate parameters (temperature, floor, progress_mix)
  ↓ next generation of genomes
```

A genome controls how the inner self-direction rule converts per-bin errors into sampling weights. By evolving those parameters, the system improves *the rule that decides what to study next*.

This is **bounded RSI**:
- Architecture frozen
- Generator physics frozen
- Evaluator (held-out Bures loss) immutable
- Only the curriculum policy parameters are improved

It is *not* open-ended code rewriting or architecture search. Those can be added later.

## Quick start

```bash
pip install -e .
python -m quasar.rsi                 # short demo (3 gens, pop=5)
```

Expected output ends with a comparison table:

```
[Final comparison under identical inner budget]
  uniform     : 0.xxxx
  error       : 0.xxxx
  progress    : 0.xxxx
  rsi-evolved : 0.xxxx
```

## Claims this extension can test

- **R1** Evolved curriculum policy beats at least one of the fixed baselines (uniform / error / progress) under matched inner budget.
- **R2** The best genome’s parameters actually move away from the hand-chosen defaults.

Both are checked at the end of `run_rsi_demo()`.

## Relation to the original claims

C1–C3 from the main README still stand for the inner loop.  
The RSI layer is a separate, higher-order experiment that asks whether *the way we do self-direction* can itself be improved by the same geometric substrate.

## Next possible steps (not implemented yet)

1. Allow genomes to control more of the inner loop (number of bins, learning rate schedule, …).
2. Represent the self_direct function itself as a small editable expression / AST and mutate the code.
3. Add a frozen “red-team” evaluator that rejects genomes that overfit the training probe.
4. Promote the best genome into the default curriculum of a future Quasar version only after multi-seed confirmation.

*Vincit Omnia Veritas.*
