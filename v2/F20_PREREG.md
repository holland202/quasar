# F20 (pre-registration) — Attention was never alive. Now test where the geometry can actually differ.

**Status:** REGISTERED, NOT YET RUN. Frozen before the first training run.

## What changed since F19

Two measurements, both made before this file was written:

**1. The attention in F18/F19 was dead.** Attention entropy at the last
position, measured on the trained F19 arms (d=3, K=8): **2.0782 (geo) and
2.0794 (dot) against a uniform maximum of ln 8 = 2.0794** — 99.9% and
100.0% of maximum. Max attention weight 0.1327 and 0.1252, where uniform
is 0.1250. Both arms compute a flat average over the causal prefix.

Cause is scale collapse: after 2500 steps the projections sit at
|Q| = 0.0571 ± 0.0320 and |K| = 0.0891 ± 0.0554, so dot scores have
spread 0.00249. Softmax over near-constant scores is uniform, and flat
attention returns almost no gradient to sharpen the projections — the
state is self-reinforcing. At d_model = 3 with σ = 0.1 init the scores
are orders of magnitude too small to bite; standard transformers escape
this only because d_model is 64–512.

This single defect is consistent with F2 (broadcast ratio 0.001),
F4 (physics head at the predict-mean floor), F18 (+0.0007 tie), and
F19 (P0 null). Those findings measured downstream symptoms; none
measured the attention distribution itself.

**2. Bures attention = dot-product attention + a purity channel.** At
equal query/key norms |q| = |k| = r,

    F = (1 + q·k + (1 − r²)) / 2

is strictly monotone increasing in q·k, so −β·arccos√F preserves the
dot-product **ordering exactly**; only sharpness differs, and β absorbs
that. Measured: Kendall τ = 1.000000, perfect agreement in 300/300 draws
at zero purity spread. Agreement decays with spread — τ = 0.919 (0.05),
0.703 (0.20), 0.368 (0.50), 0.309 (0.70).

The only term carrying information beyond the dot product is
√((1−|q|²)(1−|k|²)). **The geometry's one possible advantage is reading
purity.**

## The prediction that follows

Contraction is purity decay: |x_{t+1}| / |x_t| ≈ γ. So in a mixture whose
channels differ by **contraction**, purity is the channel signature and
Bures can read it where dot product structurally cannot. Where channels
differ only by **rotation axis**, direction carries everything and the
two scores should tie.

## Design

F19's mixture task, with the scale defect fixed: query/key projections
initialized so score spread is O(1) at init, and β learnable. Everything
else — arms, budget, seeds, the context-blind control — unchanged from
F19.

Two channel regimes, matched difficulty:
- **DIRECTION:** all channels share γ, differ by rotation axis.
- **PURITY:** all channels share axis, differ by γ.

## Registered predictions

- **P0a (anti-vacuity, unchanged from F19):** the context-blind control
  is beaten by an attention arm by > 5% relative, in both regimes.
- **P0b (LIVE ATTENTION, new — the gate F18/F19 lacked):** mean attention
  entropy at the final position must be **< 90% of the uniform maximum**
  in both arms, both regimes. If attention is uniform, the run is VOID
  and no metric claim may be made in either direction. Dead attention must
  never again be mistaken for a metric result.
- **P1:** in the DIRECTION regime, |GEO − DOT| < 2% relative — they tie,
  as the monotone-equivalence argument predicts.
- **P2:** in the PURITY regime, GEO beats DOT by > 2% relative in ≥ 4 of
  5 seeds. **This is the whole thesis of the architecture, stated in the
  one place it could hold.** If P2 fails with attention verifiably live,
  the Bures metric buys nothing on this substrate and the header claim
  should be retired, not rephrased.
- **P3:** the GEO−DOT gap correlates positively with measured purity
  spread across keys (the mechanism, not just the outcome).
- **P4 (door, unrun):** multi-head — several parallel score paths, per
  Graeme Randle's lift metaphor. Deliberately deferred: heads cannot help
  while every head is uniform, so this question only becomes meaningful
  once P0b passes.

## Harness selftests

- H1: at equal norms, generalized score == a monotone transform of dot
  product (Kendall τ = 1.0 exactly) — encodes the derivation as a test.
- H2: analytic gradients vs central finite differences ≤ 1e-3, all arms.
- H3: generated channels physical at d=3 (Choi min-eig ≥ −1e-9), with an
  identity-channel control at exactly 0 (kept from F19's convention bug).
- H4: determinism — same seed, bitwise-identical loss.
- H5: context replacement moves the attention arms and leaves the blind
  arm at exactly 0 (kept from F19).
- H6: **entropy instrument sanity** — a hand-built uniform score matrix
  measures at exactly the uniform maximum, and a hand-built one-hot score
  matrix measures near 0. The instrument that gates the experiment must
  itself be shown to move.

Numbers pasted verbatim from `f20_purity_channel.py`.

Registered 2026-08-05 (container), holland202 + Claude (Anthropic).
Diagnosis prompted by Graeme Randle's external critique.
Vincit omnia veritas.

---

## First complete run — container, 2026-08-05

Harness H1–H6 all PASS, including H1 confirming the derivation
(min Kendall τ = 1.000000 at equal norms) and H6 showing the entropy
gate can move (uniform 2.079442 = max; one-hot ≈ 0).

    [FAIL] P0a context-blind beaten by >5% in both regimes
    [PASS] P0b LIVE ATTENTION: mean entropy < 90% of uniform max
           (direction 0.705, purity 0.725)

**P0b PASSED — the dead-attention defect is fixed and demonstrated.**
Mean final-position entropy fell from 99.9–100.0% of uniform (F19) to
**70.5% and 72.5%**. The scale correction works and the instrument
proves it. That is a real, verified result and it stands independently.

**P0a FAILED, so the run is VOID and no metric claim is made.** With
attention now genuinely attending, deleting it *still* costs almost
nothing: direction regime BLIND 0.046562 vs best attention 0.046524;
purity regime BLIND 0.000908 vs 0.000890. P1–P3 were computed and are
void; the gap figures (−0.0008 direction, +0.0021 purity, 0/5 seeds
either way) are recorded only so the next design can be checked against
them, not as results.

**Diagnosis, and it is structural rather than a tuning matter.**
Identifying a linear map M from a trajectory requires relating
consecutive states — the sufficient statistic is built from products
like x_{t+1} x_tᵀ. Attention returns a convex combination of values
V = xW_v, i.e. a weighted average of individual states. **No convex
combination of states identifies the map that generated them.** The
architecture cannot form the statistic the task requires, however alive
its attention is, and however good its metric.

Second, smaller defect: the purity regime is too easy. Losses near
0.0009 with the blind arm at 0.000908 mean a shared axis plus a mean-γ
guess nearly solves it — there is no headroom for context to win.

**The next registration follows from the first paragraph, not from
tuning this one:** tokens must carry transitions, not states — attend
over (x_t, x_{t+1}) pairs or their deltas, so that a weighted average
*is* an estimate of the map. Then purity spread and the Bures term have
something to act on. Predictions get written down first, as always.

Container numbers. Nothing enters FINDINGS.md before device
reproduction.
