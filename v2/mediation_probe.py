"""
QGT Causal Mediation Probe v0.1  (inspired by Anthropic's J-lens paper,
"Verbalizable Representations Form a Global Workspace in LMs", July 2026)

Their core protocol: swap an intermediate internal representation between
two contexts and measure whether the OUTPUT follows the transplant. We run
the exact analog on the QGT, whose internal states are Bloch vectors —
interpretable-by-construction physical quantum states.

HONEST FRAMING: a 1-layer, 135-param QGT cannot have a "global workspace"
(that requires selectivity among many parallel processes). What we CAN
measure rigorously is the CAUSAL MEDIATION PROFILE: at each internal stage,
how much of the final output is determined by the information in that
stage's state, when transplanted into a different context?

Mediation index at stage k:
    run A clean -> out_A ; run B clean -> out_B
    run B but transplant A's stage-k state -> out_P
    m_k = ( d_B(out_P, out_B) - d_B(out_P, out_A) ) / d_B(out_A, out_B)
    m_k = +1  -> output fully follows the TRANSPLANT (stage k carries it all)
    m_k = -1  -> output ignores the transplant (residual context dominates)
Averaged over many (A, B) pairs, per token position.
"""

import numpy as np
from quantum_geometric_transformer import (QuantumGeometricTransformer,
                                           bures_distance, project_bloch)
from quasar import Quasar

STAGES = ["post-position", "post-attention", "post-FFN"]


def forward_capture(qgt, x):
    """Standard forward, returning output and all intermediate states."""
    seq_len = x.shape[1]
    pos = qgt.pos_enc.encode(seq_len)
    s0 = project_bloch(x + pos[np.newaxis, :, :])
    s1 = project_bloch(s0 + qgt.attn.forward(s0))
    s2 = project_bloch(s1 + qgt.ffn.forward(s1))
    out = project_bloch(s2 @ qgt.W_out + qgt.b_out)
    return out, [s0, s1, s2]


def forward_patched(qgt, x, stage, patched_state):
    """Forward on x, but at `stage` replace the state with patched_state."""
    seq_len = x.shape[1]
    pos = qgt.pos_enc.encode(seq_len)
    s0 = project_bloch(x + pos[np.newaxis, :, :])
    if stage == 0:
        s0 = patched_state
    s1 = project_bloch(s0 + qgt.attn.forward(s0))
    if stage == 1:
        s1 = patched_state
    s2 = project_bloch(s1 + qgt.ffn.forward(s1))
    if stage == 2:
        s2 = patched_state
    return project_bloch(s2 @ qgt.W_out + qgt.b_out)


def mean_bures(a, b):
    return float(np.mean([[bures_distance(a[i, t], b[i, t])
                           for t in range(a.shape[1])]
                          for i in range(a.shape[0])]))


def main():
    print("=" * 62)
    print("QGT CAUSAL MEDIATION PROBE (J-lens-style patching)")
    print("=" * 62)

    # Train a learner briefly so its internals compute something real
    print("[setup] training QGT in the QUASAR loop (3 rounds)...")
    q = Quasar(seed=0, seq_len=6)
    q.run(rounds=3, n_per_round=8, epochs_per_round=3,
          holdout=None, adaptive=True, verbose=False)
    qgt = q.learner.qgt

    # Sample (A, B) input pairs from distinct random Hamiltonians
    n_pairs = 20
    XA, _, _ = q.gen.batch(n_pairs, q.seq_len)
    XB, _, _ = q.gen.batch(n_pairs, q.seq_len)

    print(f"[probe] {n_pairs} (A,B) pairs, transplant at each of 3 stages\n")
    results = {}
    rng = np.random.default_rng(42)
    for stage in range(3):
        ms, ctrl = [], []
        for i in range(n_pairs):
            a = XA[i:i+1]; b = XB[i:i+1]
            out_A, states_A = forward_capture(qgt, a)
            out_B, _ = forward_capture(qgt, b)
            dAB = mean_bures(out_A, out_B)
            if dAB < 1e-6:
                continue
            # transplant A's stage state into B's forward
            out_P = forward_patched(qgt, b, stage, states_A[stage])
            m = (mean_bures(out_P, out_B) - mean_bures(out_P, out_A)) / dAB
            ms.append(m)
            # control: transplant a RANDOM valid state (same shape)
            rnd = rng.standard_normal(states_A[stage].shape)
            rnd = rnd / np.linalg.norm(rnd, axis=-1, keepdims=True)
            out_R = forward_patched(qgt, b, stage, rnd * 0.7)
            mc = (mean_bures(out_R, out_B) - mean_bures(out_R, out_A)) / dAB
            ctrl.append(mc)
        results[stage] = (np.mean(ms), np.std(ms), np.mean(ctrl))
        print(f"  stage {stage} ({STAGES[stage]:>14}): "
              f"mediation = {np.mean(ms):+.3f} ± {np.std(ms):.3f}"
              f"   | random-state control = {np.mean(ctrl):+.3f}")

    print("\nInterpretation (+1 = output fully follows transplanted state,")
    print("                -1 = transplant ignored, context dominates):")
    m0, m1, m2 = (results[s][0] for s in range(3))
    print(f"  Information concentration deepens through the network: "
          f"{'YES' if m2 > m1 > m0 else 'NO'} "
          f"({m0:+.2f} -> {m1:+.2f} -> {m2:+.2f})")
    print(f"  Content-specific vs random control gap at final stage: "
          f"{m2 - results[2][2]:+.3f}")
    print("\nHONEST SCOPE: this measures causal mediation in a 1-layer,")
    print("135-param model. It is a METHOD seed for v0.2 (multi-layer,")
    print("scaled QGT), not a workspace discovery claim.")


if __name__ == "__main__":
    main()
