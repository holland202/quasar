"""M3b: test the bottleneck diagnosis. Same experiment, but the physics
head reads the 12-dim FFN hidden h[:, -1] instead of the 3-dim s2[:, -1].
Cross-position info still must arrive via attention (FFN is per-position).
If physics error drops below floor -> readout bottleneck confirmed."""
import numpy as np
from quantum_geometric_transformer import QuantumGeometricTransformer
from quasar import Generator
from backprop import (T, add, matmul, tanh, softmax_rows, pbloch,
                      bures_scores, bures_mean_loss)
from m3_articulation import (take_last, mse_to_const, combine,
                             batch_with_physics, broadcast_ratio,
                             ArticulatedAnalytical)

class WideReadout(ArticulatedAnalytical):
    def __init__(self, qgt, seed=7):
        super().__init__(qgt, seed)
        rng = np.random.default_rng(seed)
        self.p["Wp"] = T(np.ascontiguousarray(rng.normal(0, 0.1, (12, 5))))
        self.p["bp"] = T(np.ascontiguousarray(rng.normal(0, 0.01, 5)))

    def _graph_h(self, x):
        qgt = self.qgt
        pos = qgt.pos_enc.encode(x.shape[1])
        s0 = pbloch(T(x + pos[None], requires_grad=False)); s0.requires_grad = False
        Q = matmul(s0, self.p["Wq"]); K = matmul(s0, self.p["Wk"])
        V = matmul(s0, self.p["Wv"])
        attn = softmax_rows(bures_scores(Q, K, qgt.attn.beta, causal=True))
        s1 = pbloch(add(s0, pbloch(matmul(matmul(attn, V), self.p["Wo"]))))
        h = tanh(add(matmul(s1, self.p["W1"]), self.p["b1"]))
        s2 = pbloch(add(s1, pbloch(add(matmul(h, self.p["W2"]), self.p["b2"]))))
        out = pbloch(add(matmul(s2, self.p["Wout"]), self.p["bout"]))
        return h, out

    def joint_step(self, x, y, phys, lam=0.5, lr=0.12, articulate=True):
        for t in self.p.values(): t.grad = None
        h, out = self._graph_h(x)
        L_state = bures_mean_loss(out, y)
        physpred = add(matmul(take_last(h), self.p["Wp"]), self.p["bp"])
        L_phys = mse_to_const(physpred, phys)
        L = combine(L_state, L_phys, lam) if articulate else L_state
        L.backward()
        for name, t in self.p.items():
            if t.grad is not None: t.data -= lr * t.grad
        return float(L_state.data), float(L_phys.data)

    def physics_error(self, x, phys):
        for t in self.p.values(): t.grad = None
        h, _ = self._graph_h(x)
        pred = take_last(h).data @ self.p["Wp"].data + self.p["bp"].data
        return float(np.mean((pred - phys) ** 2))

seq_len, steps, bsz = 6, 400, 8
gen = Generator(seed=11)
batches = [batch_with_physics(gen, bsz, seq_len) for _ in range(steps)]
gh = Generator(seed=999); Xh, Yh, Ph = batch_with_physics(gh, 32, seq_len)
Pmean = np.mean(np.concatenate([b[2] for b in batches]), axis=0)
floor = float(np.mean((Ph - Pmean) ** 2))
print(f"floor (predict mean): {floor:.4f}")
for name, artic in [("control", False), ("articulated-wide", True)]:
    qgt = QuantumGeometricTransformer(3, 1, 12, seed=0)
    m = WideReadout(qgt, seed=7)
    for (x, y, p) in batches: m.joint_step(x, y, p, articulate=artic)
    print(f"{name:>18}: holdout state {qgt.loss(Xh, Yh):.4f} | "
          f"holdout phys {m.physics_error(Xh, Ph):.4f} | "
          f"broadcast {broadcast_ratio(qgt):.5f}")
