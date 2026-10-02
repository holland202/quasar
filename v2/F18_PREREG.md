# F18 (pre-registration) — Does Bures-metric attention actually beat dot-product, and does the answer survive n=1 → n=2?

**Status:** REGISTERED, NOT YET RUN. This file is written before the first
training run. Predictions below are frozen; results will be appended to
FINDINGS.md as F18 with any failures kept.

**Motivation, stated plainly:** the QGT header claims a "genuinely new AI
architecture replacing dot-product attention," but FINDINGS F1–F16 contain
no registered Bures-vs-dot comparison. F18 tests the founding premise at
n=1 and whether it scales to n=2 (M4's open question). A dot-product win
at either dimension is a kept refutation of the premise, not a discarded
run.

## Design

Two arms, identical in every respect except the attention score:

- **GEO:** score(i,j) = −β · d(Q_i, K_j), d = arccos √F,
  F = (cap + q·k + √((cap−|q|²)(cap−|k|²))) / denom
  with (cap, denom) = (1, 2) at n=1 (exact single-qubit Bures fidelity)
  and (3, 4) at n=2 (superfidelity G, per lucid2q; a proxy, exact for
  pure states, upper-bounds true fidelity — same choice F8 made).
- **DOT:** score(i,j) = (Q_i · K_j) / √dim (standard scaled dot-product).

Shared: single attention layer (causal) + tanh FFN + linear head, all
outputs norm-capped to |x| ≤ √cap (n=2 cap is a superset of the true
state body — necessary-condition projection, same as the repo's 15-dim
code); identical parameter shapes and count; identical init (same seed
draws the same matrices for both arms); same optimizer (plain SGD), LR,
batch, steps; loss = mean d² to target.

Data: trajectories x_{t+1} = M x_t from PHYSICAL channels only —
M = γ · R_U where R_U is unitary conjugation expressed in the Pauli
basis (R_ab = Tr(P_a U P_b U†)/dim_H) and γ ∈ [0.7, 0.95] depolarizing.
Unitary conjugation + depolarizing is CP by construction at both
dimensions (harness still verifies Choi PSD at n=2). Held-out
evaluation uses unseen channels (fresh U, γ) — in-context dynamics
inference, matching quasar.py's task style.

Floors / ceiling, per dimension:
- floor-mean: predict the maximally mixed state (0 vector)
- floor-last: predict x_t (identity; strong under contraction)
- ceiling: oracle true-M one-step prediction

## Registered predictions

- **P0 (anti-vacuity, gates everything):** on holdout, BOTH trained arms
  beat floor-mean, and at least one arm beats floor-last. Otherwise the
  task is uninformative and NO architecture claim may be made in either
  direction.
- **P1:** at n=1, GEO beats DOT by > 2% relative mean d² on holdout, in
  ≥ 4 of 5 seeds. Failure with DOT ahead = premise refuted at n=1, kept.
- **P2:** same criterion at n=2.
- **P3 (scaling):** the relative GEO-over-DOT gap at n=2 is ≥ the gap at
  n=1 (geometry matters more in higher dimension). Registered knowing it
  may fail; a shrinking gap would itself be an informative negative.
- **P4 (the door, deliberately left unrun):** n=3, 63-dim generalized
  Bloch — does the trend from P3 extrapolate?

## Harness selftests (must pass BEFORE the experiment is interpreted)

- H1: generalized score fn == backprop.bures_scores at (cap,denom)=(1,2),
  agreement ≤ 1e-10 on random batches (fwd) and gradients (bwd).
- H2: generalized loss == backprop.bures_mean_loss at n=1, same bound.
- H3: analytic gradients vs central finite differences ≤ 1e-3 (repo
  standard), both dims, both arms.
- H4: every generated n=2 channel has Choi min-eig ≥ −1e-9 (physical).
- H5: fixed seed ⇒ bitwise-identical run (determinism).

Budget: 5 seeds × 2 arms × 2 dims; seq 6, batch 8, matched step count.
Numbers in the eventual F18 entry are pasted from
`f18_bures_vs_dot.py` output verbatim.

Registered 2026-08-04 (container), holland202 + Claude (Anthropic).
Vincit omnia veritas.

---

## Amendment 1 — recorded 2026-08-04, after the first run, before any claim

The first run of the registered design produced two recorded outcomes:

1. **P0 null at n=1 (kept):** with a fresh channel per trajectory and 6
   ticks of context, both arms converged to the floor-mean predictor
   (e.g. seed 0: GEO 0.065427, DOT 0.065427 vs floor-mean 0.064995) with
   arm gap +0.0000 across all 5 seeds. The registered in-context task is
   uninformative at this scale — the anti-vacuity gate fired as designed.
   No architecture claim from that run.
2. **Harness bug at n=2 (kept, fixed):** the generalized fidelity used
   (cap + c·d + s)/denom; the correct superfidelity constant is 1 at both
   dimensions: (1 + c·d + s)/denom. At n=2 the wrong constant forced
   F ≡ 1 → distance ≡ 0 → zero gradient; the n=2 "training" trained
   nothing and its zeros were fake. H1–H3 could not catch this: cap = 1
   masks the bug at n=1, and H3 verified gradients of the wrong formula
   self-consistently. **H6 added:** the generalized F is now checked
   against superfidelity computed independently from raw density
   matrices, at n=2, tolerance 1e-10. (Same lesson as arch_map P10b: the
   instrument contained the defect class it was built to find.)

**Task revision:** one channel per seed, shared by both arms; training
batches and holdout draw fresh initial states, holdout states unseen.
This makes the dynamics learnable by a single-layer model while keeping
the arms' comparison exact. All P0–P4 criteria unchanged. n=1 numbers
from the first run are recorded above; n=2 first-run numbers are void
(nothing trained) and are superseded by the H6-verified rerun.

---

## First complete run — container, 2026-08-04 (pending device reproduction)

Harness: H1–H6 all PASS (H6 worst 0.00e+00-class agreement; see script
output). Verdict block pasted verbatim from `f18_bures_vs_dot.py`:

    [PASS] P0 anti-vacuity: both arms beat floor-mean and some arm beats floor-last, both dims
    [FAIL] P1 n=1: GEO > DOT by >2% in >=4/5 seeds (got 0/5, mean gap +0.0007)
    [FAIL] P2 n=2: same criterion (got 0/5, mean gap +0.0003)
    [FAIL] P3 scaling: n=2 mean gap >= n=1 mean gap (+0.0003 vs +0.0007)

**Reading, stated carefully:** at matched budget, identical init, and a
learnable single-channel dynamics task, Bures-metric attention is
indistinguishable from dot-product attention at BOTH n=1 and n=2 — mean
relative gaps +0.0007 and +0.0003, zero seeds past the 2% criterion.
P1–P3 are refuted as registered and KEPT.

Scope of the refutation (mechanism candidate, not yet tested): under
Amendment 1's shared channel, the FFN + head can learn the dynamics map
directly and attention need not carry information — n=1 lands near the
oracle (e.g. 0.000068 vs floor-mean 0.067866), consistent with attention
being a passthrough for both arms. The discriminating experiment is one
where attention is load-bearing AND learnable: K-channel mixtures with
in-context identification at higher capacity. That is the real door
behind P4, and it remains open.

Per house rule these are container numbers until reproduced on device;
no FINDINGS.md entry until then.
