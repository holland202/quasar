# QUASAR v0.2 — Findings Log

Every entry: claim, method, number, status. Newest last.
LEDGER INTEGRITY NOTE (2026-07-12): the on-device FINDINGS.md was
corrupted (null bytes) by an interrupted write on Android/Termux, the
same event that corrupted .git/index and refs/heads/main. This file is
a faithful reconstruction from the full working-session log; every
number below is printed by code in this repository.

## F1 — METHODOLOGY TRAP: full-state patching is vacuous (2026-07-11)
First mediation probe transplanted ENTIRE intermediate states and
measured mediation = +1.000 at every stage. Mathematically forced: in a
strictly sequential network with no bypass pathway, the output is a
deterministic function of any full intermediate state — the probe could
not fail. Rule adopted: every intervention probe must have a
configuration in which it CAN return a null result, demonstrated before
its numbers are trusted. mediation_probe.py kept as the documented trap;
broadcast_probe.py is the corrected instrument.

## F2 — Bures attention broadcast ratio ~ 0.001 (2026-07-11)
Single-position pre-attention state transplant between inputs from
distinct Hamiltonians (broadcast_probe.py, n=30). Output shift at the
patched position 0.0947; at other positions 0.0001; ratio 0.001. The
v0.1-trained QGT's attention carries essentially NO cross-position
information — next-step prediction on these dynamics is positionally
local, so attention was never forced to earn its keep. Broadcast ratio
adopted as a standing diagnostic.

## F3 — M1 COMPLETE: analytical backprop, 633x speedup (2026-07-11)
Micro reverse-mode AD engine (backprop.py) with fused geometric
primitives (pbloch, vectorized pairwise Bures scores, softmax, Bures
loss), sharing the original model's parameter arrays.
V1 forward equivalence 5.6e-17 (machine precision); V2 gradients vs
central finite differences across ALL 135 parameters, worst abs 8.4e-11;
V3 633x per-step speedup (428 ms -> 0.7 ms), growing with param count;
V4 training decreases loss (0.2947 -> 0.2536 / 30 steps).
Bug caught by V1: the loss is mean SQUARED Bures distance — the initial
reimplementation omitted the square (0.245 discrepancy, flagged
instantly). M2 (capacity scale-up) unblocked.

## F4 — Articulation refuted at single-qubit scale; bottleneck localized (2026-07-11)
Auxiliary head predicting the generating physics (axis, w, g); geometry
requires integrating >= 3 positions, so this was the F2-mandated
cross-position task. Matched-budget experiment (m3_articulation.py,
400 steps/arm, 0.8 s total on analytical gradients):
P-A broadcast ratio unchanged (0.0007); P-B no state-prediction gain
(0.1464 control vs 0.1471 articulated); physics head AT the
predict-mean floor (0.2224 vs floor 0.2191) — learned nothing.
Diagnosis test (m3b_widereadout.py): a 12-dim readout ALSO at floor
(0.2230) -> readout-bottleneck hypothesis REFUTED. Localized cause: all
cross-position information transits ONE 3-dim linear attention mix,
which cannot carry the >= 6 numbers of relational structure physics
inference needs. Converges with C3 + F2: the binding constraint is
residual width (d_model = 3) and attention depth, NOT d_ff.
External proposal scorecard: strategies 1, 2, 4, 5 broken/obsolete/
premature; strategy 3 tested here and refuted at this scale.

## F5 — LUCID v0: channel layers, emergent CPTP, purity-preserving learning (2026-07-12)
A 9-parameter Bloch-map layer (lucid_v0.py, lucid_v0b.py), identical
data/budget to F4.
(1) EMERGENT PHYSICALITY: the learned map is genuinely CPTP (Choi min
eigenvalue +0.031) with nothing enforcing it; random-matrix control
-0.458.
(2) BEATS THE TRANSFORMER: holdout Bures^2 0.0566 vs 0.1464 for the
135-param QGT on identical data — explaining F2 (the QGT only ever
needed the per-position map, and learned it worse).
(3) LOSS GEOMETRY DETERMINES THE LEARNED AGGREGATE: closed-form mean
channel is depolarizing with s = 0.7960 (Monte Carlo confirmed 0.7956).
MSE training converges there (diag 0.808; scan eta*_MSE = 0.795). Bures
training instead converges to the purity-preserving contraction (diag
0.925-0.93; scan eta*_Bures = 0.925). New quantitative result:
Bures learning preserves purity statistics; Euclidean learning averages
directions. Three independent methods agree at each optimum.

## F6 — LUCID v1: in-context channel inference CONFIRMED; aggregate win open (2026-07-12)
Mixture of K=6 channel atoms, causal Bures-evidence gating (lucid_v1.py).
R2 CONFIRMED: gate entropy collapses 1.79 -> 0.43 after ONE observed
transition; the mixture beats the single channel at every position >= 1
(0.028 vs 0.042 at t=5, -33%). The in-context capability the transformer
lacked (F4) exists in a channel dictionary with ~55 params.
R1 OPEN: average holdout (0.0695; 0.0625 with learnable prior; 0.0691
with a seeded aggregate atom) loses to single-channel 0.0566 purely at
position 0 — cold-start credit assignment: prior gradients flow through
only 1/6 of loss terms; even the seeded aggregate atom drifts. Next
attacks: position-weighted prior loss, or hierarchical confidence gate
(aggregate fallback outside the softmax). Oracle ceiling 0.0000 — large
headroom.

## VERA — unification thesis (registered 2026-07-12)
Refound, don't merge: VERA (Verifiable Evidence-gated Runtime
Architecture) = LUCID substrate + QUASAR knowledge loop + conformal
uncertainty (SENTINEL lineage) + Veritas thermal governance +
SIC/VEST provenance + SECP-pattern response gate refounded on
certificates instead of heuristics. Each layer ships only behind a
finding. Leave-behind: QOLAS LLM backbone (never trained), SKN
prototype, demo theater; SECP's formal-verification/Byzantine claims
until run. One-sentence thesis: every layer can be interrogated at
runtime and must answer with evidence — mechanism by tomography, data
by physical law, uncertainty by coverage, operation by thermodynamics,
history by authentication.

## F7 — VERA I1: conformal certificates VERIFIED; gate power curve (2026-07-12)
Per-position split conformal on LUCID v1.1, nonconformity = Bures
distance (i1_conformal.py).
I1-a: first run missed the registered band [0.87, 0.93] at ONE position
by 0.002 (0.932, n_cal = 300) — consistent with conditional-coverage
variance (sd ~ 1.7%). High-power rerun (n_cal = 1000, n_test = 2000):
ALL positions in band, 0.904-0.929, mean 0.914. VERIFIED.
I1-b VERIFIED: certified radius contracts 0.531 -> 0.284 along the
sequence — in-context inference now carries a coverage guarantee.
I1-c VERIFIED: shifted physics (w in [1.3, 2.0]) collapses coverage to
0.394 — the certificate can fail, hence is informative.
VERA GATE: mean-score detector FAILED registered TPR >= 0.80 (0.682);
diagnosis — the mixture partially ADAPTS to OOD physics in-context
(shifted coverage rises 0.31 -> 0.48 with position), camouflaging late
positions. Adaptation-detection tension noted as a general safety
observation for adaptive systems. Max-exceedance detector (any position
breaching its certified radius): FPR 0.091; power curve monotone in
shift magnitude: 0.583 / 0.730 / 0.790 / 0.860 / 0.892 for w-bands
[1.25-1.4] .. [2.0-2.5]. A single-number TPR claim was ill-posed; the
power curve is the correct characterization.

## F8 — LUCID-2Q: capacity wall down; the purity law is a LAW (2026-07-12)
15-dim generalized Bloch formalism (lucid2q.py); metric = superfidelity
d_SB (exact Bures at n=1 — it is the formula this project has used all
along — and a rigorous computable bound for d > 2; named honestly).
Self-tests T1-T6 all pass, including a Choi-convention bug CAUGHT BY T4
(transposed blocks = the SWAP operator, min eig -0.904; fixed, true
channel +0.019, random map -2.435) and a group-theory prediction
confirmed: the unitarily invariant H-ensemble forces E[channel] = mu*I
(irreducibility of the SU(4) adjoint rep); MC mu = 0.8003, residual
0.029. Negativity witness: product-H trajectories exactly PPT;
entangling-H develop negativity in 100% of samples.
N2-1 VERIFIED (after a diagnosed-then-fixed optimization-budget
artifact: the mse primitive normalizes per-element, giving 5x smaller
n=2 gradients; convergence targets mu / eta* were training-independent):
MSE-trained -> 0.8033*I ~ mu; SB-trained -> 0.9271*I ~ scan
eta* = 0.930. PURITY-PRESERVING LEARNING GENERALIZES: two scales, three
independent methods at each.
N2-2 VERIFIED: emergent CPTP at 16x16 Choi, -0.0164 vs random -1.35;
the violation SHRINKS with convergence (-0.077 -> -0.016): physicality
is an attractor of training on physical data.
N2-3 VERIFIED: 15-dim evidence gating identifies which of 8 physics
generated a trajectory with 100% accuracy from ONE observed transition
(floor 12.5%), including perfect entangling-vs-product discrimination —
the model reads whether its physics ENTANGLES. The n=1 in-context
mechanism (F6 R2) sharpens with dimension.
ON-DEVICE VERIFICATION: full suite reproduced on Galaxy S25
(aarch64/Termux) with identical digits (T2 err 8.9e-16 vs 5.0e-16
float-order difference only), runtime 5.1 s vs 9.4 s x86 sandbox — the
phone outran the workstation again.

## Milestones
M1 Analytical backprop — DONE (F3).
M2 Capacity: redefined by F4 as depth + d_model, not d_ff; n=2
   substrate DONE (F8); trained-mixture-at-n=2 and C3 retest NEXT.
M3 Cross-position tasks — articulation refuted at n=1 (F4); broadcast
   ratio standing diagnostic (F2); n=2 gating perfect (F8).
M4 Multi-qubit — substrate landed (F8); multi-layer / n>2 open.
VERA I1 conformal — DONE (F7). I2 Veritas thermal governance on-device
   — OPEN (requires S25 sysfs). I3 tomographic commit-gate — OPEN.
   I4 SIC/VEST provenance — OPEN. Q-SEED camera entropy — OPEN.

Release rule: nothing leaves this repo until its claims pass the v0.1
verification standard (floor, ceiling where provable, matched-budget
control, held-out data).

## F9 — Dictionary learning: gradient gating fails, EM recovers everything (2026-07-12)
Five controlled runs (m2q_*.py), each failure diagnosed into the next:
(1) TRAINED mixture, continuous ensemble: COLLAPSE — gate entropy 0.00
from t=1, t>=1 errors equal the single channel to 3 decimals; winner-
take-all starves all but the aggregate atom. TR-2 held (0.151->0.063).
(2) Finite ensemble (8 true channels): SYMMETRY NON-BREAKING — atoms
never differentiate (recovery ~1.96 from floor 3.64, ID 0.109 ~ chance);
uniform gates give every atom the same aggregate-pull gradient.
(3) Data-driven init (per-atom single-trajectory ridge): still fails
(ID 0.203) — one trajectory spans only ~5 of 15 dims, so seeded atoms
cannot route other trajectories of their own channel.
(4) EM (closed-form ridge M-step): DR-2 PASSES partially — four channels
recovered to 0.05-0.31 Frobenius (their Choi eigs +0.007/+0.008: the
recovered atoms ARE physical channels); two true-channel pairs merged
(cluster sizes [3,46,95,2,110,53,43,48], classic local optimum).
(5) EM + 10 restarts + split-merge: TOTAL RECOVERY — all 8 channels to
mean 0.050 Frobenius; assignment accuracy 1.000/400; every atom CPTP
(Choi +0.007..+0.013); EM-gated prediction 0.0295 vs global-LS 0.0793
(2.7x), residual = t=0 cold start; oracle 0.
Findings: (a) evidence GATING is solved (F8) but dictionary LEARNING via
gated gradients is blocked by symmetry non-breaking — a channel-space
demonstration of mixture-of-experts collapse; (b) closed-form EM with
restarts+split solves it completely; (c) PHYSICALITY AS CHECKSUM: Choi
positivity of learned atoms tracked recovery quality through every run —
the substrate certifies its own learning. Next: streaming/online EM,
cold-start via aggregate-of-atoms at t=0, and R1(n=1) revisit with EM.

## F10 note — QSLEUTH v0.1 released public (2026-07-12)
F9's recovery machinery productized: K-discovery (validation-residual +
ghost-prune + duplicate-merge), EM recovery, Choi certification,
entanglement witness, conformal coverage, intruder gate, hash-chain
audit. 9/9 registered self-tests. Two pre-release bugs caught by the
tests themselves (Choi transpose, mismatched-index seed) — documented.
Public timestamp: github.com/holland202/qsleuth.

## F11 — VERITAS RUNTIME v0: VERA Governance layer, runnable (2026-07-12)
Fusion of verified cores (SLC/Veritas thermal governor, SECP Sentinel +
hash-chain audit, self-healing rollback) wrapped around real LUCID Bures
training. Sandbox claims V1-V4 ALL PASS: thermal envelope held (42.1C vs
ungoverned 50.0C on the same RC model); converges under 50% duty cycle
(0.302 -> 0.056); gradient bomb (x1e6 @ step 250) detected and recovered
to 0.056 — IDENTICAL to the clean run; audit chain tamper-detected.
BUG CAUGHT BY V3: first guard checkpointed AFTER the update and judged on
the PRE-update loss -> it snapshotted the poisoned weights and every
rollback restored TO the poison (0.826). Rule: never checkpoint state you
have not verified. Fix = forward-only PROBE on fixed holdout, judged
post-update; rollback on probe divergence; checkpoint verified state only.
Anti-theater: SysfsThermal FAILS LOUD without real sensors (the QOLAS
`except: return 45.0` pattern is a governor that cooks the machine while
reporting safe). I2 sandbox half DONE; live half still needs S25 sysfs.

## F12 — R1 CLOSED: EM dictionary + aggregate cold-start; physicality is a REGULARIZER, not just a checksum (2026-07-12)
The oldest open claim (F6 R1) is dead. n=1, CONTINUOUS ensemble (K atoms
must quantize a continuum — the hard case, unlike F9's finite set).
R1-a PASS: EM (K=12, 8 restarts, starve-split) + causal Bures gating, with
the F5 aggregate channel (eta*=0.925) serving position 0 OUTSIDE the gate:
holdout 0.0306 vs single-channel floor 0.0566 (1.85x); oracle 0.0000.
R1-b PASS: cold start exactly repaired — pos-0 error 0.074, identical to
the single channel (was 0.223 in F6's softmax-only mixture).
R1-c FAIL (informative): 2 of 12 atoms non-physical (Choi -0.039, -0.024).
Diagnosis: atoms fit CLUSTERS of nearby channels, and pooled least squares
overshoots the CP cone — the checksum located exactly the boundary-
straddling atoms.
R1-d PASS: projecting each atom onto the CP cone (Choi eigenvalue clip +
least-squares pullback) makes ALL atoms CPTP (worst -0.000000) AND holdout
gets STRICTLY BETTER (-0.00004). Small magnitude, but the SIGN is the
result: enforcing physicality costs nothing and pays. CP is not a tax on
the learner; it is free information. Physicality: checksum (F9) ->
regularizer (F12).

## F3 AMENDMENT — speedup restated honestly (2026-07-12)
The "633x" autodiff speedup was ONE wall-clock sample quoted to three
significant figures. Repeated runs: 597 / 633 / 657 / 665 / 680x. Honest
claim: ~600-680x, load- and machine-dependent. Correctness figures
(5.6e-17 forward, 8.4e-11 gradients) are exact and stand unchanged.

## F14 — N1: CP projection is a DATA-SCARCITY regularizer (2026-07-12)
Registered follow-on to F12. Project the Bloch map onto the CP cone after
EVERY gradient step; matched init, data stream, lr, steps; 8 paired seeds
(both init AND data vary per seed — an earlier version varied only the
init, caught by an implausible std of exactly 0.00000 across "independent"
runs).
N1-0 ANTI-VACUITY (F1 rule) PASSES: the control genuinely leaves the CP
cone (23% / 69% / 46% of steps across regimes), so projection is not a
no-op and a win is attributable.
RESULTS (paired delta, holdout Bures^2):
  standard   (400 steps, bs=8) : -0.00001, 6/8 wins -> NOT ESTABLISHED
  low-data   (120 steps, bs=2) : -0.00114, 8/8 wins -> PASS
  aggressive (lr=0.5)          : -0.00013, 7/8 wins -> PASS
N1-c REFUTED as registered: the effect does NOT track how hard the
constraint binds (aggressive binds 46% and gains ~nothing; low-data binds
69% and gains most). The controlling variable is DATA SCARCITY.
STATEMENT: CP projection is a free regularizer that helps exactly where
regularizers help — scarce data, noisy optimizer. At convergence with
abundant data the learner reaches the cone unaided (F5) and projection is
a no-op. Physicality: checksum (F9) -> post-hoc regularizer (F12) ->
in-training regularizer, conditionally (F14).

## F13 (in progress) — device thermal calibration
Cause of the failed live run: max() over ALL sysfs zones. The S25 exposes
68; modem/battery/skin zones idle at 60C, so a 42C envelope (tuned on the
synthetic RC model) pauses forever. The governor was CORRECT and
MISCALIBRATED. Fix: --zones discovery (name + temp), single-zone
selection, and an envelope set relative to the MEASURED idle baseline
(T_high = idle + delta). Awaiting on-device duty-cycle curve.

## F15 — N7: VERA SURVIVES REAL TOMOGRAPHY (2026-07-13)
The program's largest gap, closed. Exact Bloch vectors replaced with
simulated finite-shot Pauli tomography (binomial outcomes, error ~1/sqrt(S)).
N7-0 anti-vacuity PASS: noise degrades the pipeline (S=32 holdout 0.121 vs
exact 0.058).
N7-a REGISTERED CLAIM WAS ILL-POSED and is rewritten: "recovery" matched
noisy-EM atoms to exact-EM atoms, but BOTH carry clustering noise, so the
metric saturates (~S^-0.04) and cannot see convergence. Same failure class
as F7's withdrawn single-number TPR. Correct metric = holdout vs the
exact-data baseline: 2.06x at S=32 -> 1.25x at 128 -> 1.03x at 1024 ->
1.00x at 4096. GRACEFUL, MONOTONE, converged by ~1024 shots.
N7-b: raw EM atoms are slightly non-physical at EVERY shot count INCLUDING
EXACT (quantization artifact, F12 — not a noise artifact). CP projection
(F14) repairs them at every S: worst Choi -0.070 -> -0.000.
N7-c HEADLINE: CONFORMAL COVERAGE HOLDS AT EVERY SHOT COUNT (0.891-0.923,
nominal 0.90) EVEN WHERE THE MODEL IS TWICE AS BAD. The guarantee is
distribution-free; the certificates simply WIDEN. VERA's uncertainty layer
is real-world-ready even when every other layer degrades.
N7-d: 31% of raw estimated states land OUTSIDE the Bloch ball at S=32 and
STILL 10% at S=4096 (pure states sit ON the ball; noise pushes half out).
On real data physicality is an OPERATION, not merely an audit.

## F16 — N2: THE C3 PREDICTION IS REFUTED; error-driven curricula are ACTIVELY HARMFUL; progress-driven curricula work (2026-07-13)
C3/F4 diagnosed the tied curriculum as flat competence and predicted
adaptive sampling would WIN once the learner could specialize. FALSIFIED.
N2-0 anti-vacuity PASS: at n=2 with 225 params competence is strongly
uneven (per-bin error 0.052 -> 0.346, spread 0.56). The learner CAN
specialize; the sampler DID reallocate (weights 0.059 -> 0.368, L1 0.46).
N2-a/b FAIL: error-proportional sampling is 8.08% WORSE than uniform,
losing 0/5 seeds (0.20595 +- 0.00179 vs 0.19056 +- 0.00264).
MECHANISM (the finding): corr(final error, LEARNABILITY) = -0.999. The
easiest bin had 87.2% of its error reducible; the hardest, 14.8%. ERROR
DOES NOT TRACK LEARNABILITY — error-driven sampling pours the budget into
precisely the bin where learning is least possible. Hard != learnable.
N2'-a/b PASS: PROGRESS-proportional sampling (weights ~ recent error
REDUCTION) beats uniform by +3.30%, winning 5/5 seeds (0.18428 +- 0.00249),
and beats error-proportional by 12.5%. Its weights genuinely deviate
(L1 up to 1.17) and it DISCOVERS an easy-to-hard schedule unprompted
(early weights 0.276 -> 0.121), relaxing to uniform as progress equalizes.
PRIOR ART, stated honestly: learning-progress as a curriculum signal is
known (Graves et al., Automated Curriculum Learning, 2017). What is new
here is the mechanism measured cleanly in channel space at corr = -0.999.
CONSEQUENCE: the QUASAR loop's error-driven curriculum is WRONG BY DESIGN
and must be replaced with a progress-driven one. This retro-explains C3:
the tie was luck — a flat-competence learner cannot be harmed by a bad
signal. Give it capacity and the bad signal HURTS.

## F17 — VERA TRANSFERS OFF THE BLOCH BALL: certificates on real SCADA data (2026-08-04)

First application of the F7/F15 certificate layer to data this program did not
generate. Target: BATADAL, the public water-distribution attack benchmark
(Taormina et al. 2018), 43 standardized channels, 2,089 test records, 407
attack ticks. Method: ridge dynamics x_(t+1) = A x_t + b in place of the Bloch
map, split conformal on clean dataset03 (60/40 temporal), attacks flagged as
certificate breaches. Predictions registered before the first run.
P0 ANTI-VACUITY PASS: attack ticks breach the 90% certificate at 54.5% vs 30.3%
for clean ticks — the instrument separates, so the run is informative.
P1 REFUTED (kept): clean-test coverage 0.697 against nominal 0.90. This is F7
I1-c appearing on real infrastructure — the certificate did not fail, it
reported that the plant drifted between the calibration year and the test year.
The guarantee is distribution-free per split, not across a distribution shift.
P2 REFUTED (kept): attack-to-clean exceedance 1.8x against a registered 3x.
P3: tick F1 0.437 at the 99% certificate. HONEST SCOPE: tick-level only, no
episode detection, no TTD, and NOT scored on BATADAL S — the benchmark ranks on
S, not F1, so this number does not rank the method against any published entry.
Until it runs through batadal_score.py it is an experiment, not a benchmark
result.
REPRODUCTION: digit-for-digit on the Galaxy S25 (aarch64/Termux) and x86 —
q90 0.6298, q99 0.8052, breach 30.3%/54.5%, coverage 0.697, F1 0.437, all four
verdicts identical. Code: github.com/holland202/sentinel-batadal-validation,
vera_batadal.py. NEXT: EM dynamics dictionary (F9/F12 machinery, multi-mode),
drift-aware calibration for P1, and the official S score.

## F18 — THE FOUNDING PREMISE IS REFUTED: Bures attention does not beat dot-product (2026-08-05)

The QGT header has claimed since v0.1 that Bures-metric attention replaces
dot-product attention. F1-F17 never registered that comparison. F18 runs it.
Two arms identical in parameter shapes, init (same seed draws the same
matrices), data stream, optimizer, LR, batch and step count; ONLY the
attention score differs. GEO = -beta*arccos(sqrt(F)) on the fidelity angle;
DOT = scaled dot-product. Task: in-context dynamics prediction, one physical
channel per seed (gamma * unitary conjugation, CP by construction), holdout
on unseen initial states. Floors: predict-mixed and predict-last. Registered
before running.
P0 ANTI-VACUITY PASS: both arms beat both floors at both dimensions, so the
task is learnable and the comparison is informative.
P1 REFUTED (kept): at n=1, GEO beat DOT by >2% in 0 of 5 seeds; mean relative
gap +0.0007.
P2 REFUTED (kept): at n=2 (15-dim, superfidelity), 0 of 5 seeds; mean +0.0003.
P3 REFUTED (kept): the gap did not grow with dimension (+0.0003 at n=2 vs
+0.0007 at n=1).
STATEMENT: at matched budget on this task the metric is worth a rounding
error. The geometry is not buying what the architecture claimed.
SCOPE, stated so the refutation is not overread: with one channel per seed the
FFN and head can learn the map alone and attention need not carry information
(n=1 lands near the oracle, 0.000068 vs floor 0.0679), consistent with
attention being a passthrough for BOTH arms. This does not rescue the claim -
it says the claim was never tested where it could matter. F2 measured the same
thing (broadcast ratio 0.001) and F4 localized it (one 3-dim linear mix).
HARNESS BUG CAUGHT AND KEPT: the generalized fidelity first used
(cap + c.d + s)/denom; the correct superfidelity constant is 1 at both
dimensions. At n=2 the wrong constant forced F == 1, distance == 0, zero
gradient - the n=2 arm trained nothing and its zeros were fake. H1-H3 could
not catch it (cap=1 masks it at n=1; finite differences verified the wrong
formula self-consistently). H6 added: generalized F checked against
superfidelity computed independently from raw density matrices, tol 1e-10.
The instrument contained the defect class it was built to find - same lesson
as F1, F8 T4, and arch_map P10b.
P4 (n=3, 63-dim) deliberately left unrun.
REPRODUCTION: Galaxy S25 (aarch64/Termux) matches the container digit for
digit - +0.0007, +0.0003, 0/5 both dims, all four verdicts identical.
Code: f18_bures_vs_dot.py, predictions in F18_PREREG.md.
