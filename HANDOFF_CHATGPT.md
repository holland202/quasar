# Handoff Document — QUASAR RSI Work

**Date:** 2026-09-16  
**Repo:** https://github.com/holland202/quasar  
**Branch:** `rsi-meta-curriculum`  
**Pull Request:** https://github.com/holland202/quasar/pull/1  
**Author of this handoff:** Grok (xAI) working with Chad Holland (holland202)

---

## 1. What the original project is

QUASAR is a pure-NumPy closed-loop learner that generates its own training data for single-qubit channel dynamics:

```
Generator → difficulty bins (Bures path length) → Learner (Quantum Geometric Transformer) → curriculum weights → next generation
```

Key properties:
- No human-authored corpus.
- Difficulty is native geometric (Bures distance).
- Curriculum can be `uniform` | `error` | `progress`.
- Extremely small model (~135 params), finite-difference training, fully checkable on a phone.
- Honest science culture: negative results are kept and registered (see C3 and the findings in the successor repo quasar-v2).

Main claims of the original system (C1–C3) are documented in the root README.

---

## 2. What we just built (the RSI layer)

We added a **bounded recursive self-improvement** layer that sits *outside* the original loop.

### Core idea
The inner Quasar loop is left completely unchanged.  
The outer loop treats the *curriculum policy parameters* as the object of improvement:

- `temperature` — how sharply error is turned into sampling weight
- `floor` — minimum weight per difficulty bin
- `progress_mix` — mixture between pure error-driven and pure progress-driven weighting

A population of `CurriculumGenome` objects is evolved with mutation + elitist selection.  
Fitness = held-out Bures loss after a fixed training budget (the evaluator is immutable).

This is **Level-1 / Level-2 RSI** in the common ladder:
- The system improves the rule that decides *what to study next*.
- It does **not** rewrite its own architecture, generator physics, or code.
- It is deliberately bounded and honest about scope.

### Files added on this branch
| File | Purpose |
|------|---------|
| `quasar/rsi.py` | `CurriculumGenome`, `MetaQuasar`, `run_rsi_demo()` |
| `README_RSI.md` | Design notes, claims R1/R2, how to run |
| `HANDOFF_CHATGPT.md` | This document |
| Updates to `quasar/__init__.py` and root `README.md` | Exports + pointers |

### How to run the RSI demo
```bash
git checkout rsi-meta-curriculum
pip install -e .
python -m quasar.rsi
```

It will:
1. Evolve a small population for a few generations.
2. Print the best genome found.
3. Compare it against the three classic fixed curricula under identical inner budget.
4. Report whether R1 and R2 hold for that run.

---

## 3. Current status

- Branch `rsi-meta-curriculum` is live.
- Pull Request #1 is open against `main`.
- Everything remains pure NumPy, zero external ML dependencies beyond the original project.
- The original test suite and claims are untouched.

---

## 4. Suggested next steps (in rough priority order)

1. **Merge the PR** (or iterate on it) after a quick local run of `python -m quasar.rsi`.
2. Add a formal, deterministic test (or multi-seed check) for R1/R2 and register the results the same way C1–C3 were registered.
3. Expand the genome (learning-rate schedule, number of bins, epoch counts, etc.).
4. Move from parameter mutation to mutating a small editable expression / AST that implements `self_direct`.
5. Add a promotion gate: only accept a genome into the default curriculum after it beats baselines on a larger held-out set across multiple seeds.
6. Consider a “red-team” evaluator that rejects genomes that look good on the training probe but fail transfer.

---

## 5. Design principles we are following

- Keep the inner loop frozen and well-tested.
- Keep the evaluator immutable.
- Prefer small, checkable experiments over ambitious but un-auditable ones.
- Register predictions before running; keep negative results.
- Stay pure NumPy so the whole system remains inspectable and runnable on very modest hardware.

---

## 6. Key links

- Repo: https://github.com/holland202/quasar
- RSI branch: https://github.com/holland202/quasar/tree/rsi-meta-curriculum
- Pull Request: https://github.com/holland202/quasar/pull/1
- Successor experiment repo (quasar-v2): https://github.com/holland202/quasar-v2
- Original README claims: C1, C2, C3
- New RSI claims: R1, R2 (see README_RSI.md)

---

## 7. One-sentence summary for a new session

> We took the existing pure-NumPy self-generating curriculum system (QUASAR) and added a bounded outer evolutionary loop that improves the curriculum policy parameters themselves; the PR is open and the next useful work is multi-seed confirmation of the meta-improvement claims and possible expansion of the genome.

*Vincit Omnia Veritas.*
