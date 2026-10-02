"""Broadcast probe: transplant ONE position's pre-attention state from A
into B. Bures attention is the only pathway that can carry that change to
OTHER positions. Measure output shift at the patched position (direct) vs
at all other positions (broadcast). Nontrivial: could be ~0 if attention
weights ignore the patched key/value."""
import numpy as np
from quantum_geometric_transformer import bures_distance, project_bloch
from quasar import Quasar
from mediation_probe import forward_capture

q = Quasar(seed=0, seq_len=6)
q.run(rounds=3, n_per_round=8, epochs_per_round=3, holdout=None,
      adaptive=True, verbose=False)
qgt = q.learner.qgt

def fwd_from_s0(s0):
    s1 = project_bloch(s0 + qgt.attn.forward(s0))
    s2 = project_bloch(s1 + qgt.ffn.forward(s1))
    return project_bloch(s2 @ qgt.W_out + qgt.b_out)

n = 30; rng = np.random.default_rng(0)
XA,_,_ = q.gen.batch(n, 6); XB,_,_ = q.gen.batch(n, 6)
direct, bcast, base = [], [], []
for i in range(n):
    _, sA = forward_capture(qgt, XA[i:i+1])
    outB, sB = forward_capture(qgt, XB[i:i+1])
    t = rng.integers(0, 5)          # patch a non-final position
    s0p = sB[0].copy(); s0p[0, t] = sA[0][0, t]
    outP = fwd_from_s0(s0p)
    d = [bures_distance(outP[0,k], outB[0,k]) for k in range(6)]
    direct.append(d[t])
    bcast.append(np.mean([d[k] for k in range(6) if k != t]))
    # baseline: typical output distance between unrelated states
    base.append(np.mean([bures_distance(outB[0,k], outP[0,(k+1)%6]) for k in range(6)]))
print(f"single-position transplant, n={n}:")
print(f"  output shift at PATCHED position : {np.mean(direct):.4f}")
print(f"  output shift at OTHER positions  : {np.mean(bcast):.4f}  (attention-mediated broadcast)")
print(f"  broadcast / direct ratio         : {np.mean(bcast)/np.mean(direct):.3f}")
