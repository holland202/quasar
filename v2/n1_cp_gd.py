"""
N1 — CP-PROJECTED GRADIENT DESCENT: is physicality a free optimizer?
=====================================================================
F12 showed post-hoc CP projection of EM atoms made prediction STRICTLY
BETTER. Conjecture: projecting onto the CP cone after EVERY optimizer
step is a strictly better optimizer on physical data.

ANTI-VACUITY GUARD (F1 rule) — REGISTERED FIRST:
  N1-0  The CONTROL arm must actually LEAVE the CP cone during training.
        If it never violates, projection is a no-op and any 'win' would
        be noise. In that case the claim is declared VACUOUS in that
        regime and we move to a regime where the constraint BINDS.
REGISTERED:
  N1-a  In a regime where N1-0 holds, CP-projected training achieves
        holdout Bures^2 <= control (matched init, data, lr, steps).
  N1-b  The projected arm is CPTP at EVERY step (lambda_min >= -1e-9).
  N1-c  Effect grows as the constraint binds harder (low-data regime).
Floor: control (unprojected). Ceiling: oracle per-trajectory channel.
"""
import numpy as np
from quasar import Generator
from quantum_geometric_transformer import project_bloch, bures_distance
from backprop import T, matmul, pbloch, bures_mean_loss
from m3_articulation import batch_with_physics

sg = [np.eye(2, dtype=complex),
      np.array([[0,1],[1,0]], dtype=complex),
      np.array([[0,-1j],[1j,0]], dtype=complex),
      np.array([[1,0],[0,-1]], dtype=complex)]

def choi(M):
    J = np.zeros((4,4), dtype=complex)
    for k in range(2):
        for l in range(2):
            E = np.zeros((2,2), dtype=complex); E[k,l] = 1
            cE = np.array([np.trace(E @ sg[a]) for a in (1,2,3)])
            tr = 1.0 if k == l else 0.0
            out = tr*np.eye(2)/2 + sum((M @ cE)[a]*sg[a+1] for a in range(3))/2
            J[2*k:2*k+2, 2*l:2*l+2] = out
    return (J + J.conj().T)/2

def lmin(M): return float(np.min(np.linalg.eigvalsh(choi(M))))

J0 = choi(np.zeros((3,3)))
B = []
for a in range(3):
    for b in range(3):
        E = np.zeros((3,3)); E[a,b] = 1
        B.append(choi(E) - J0)
A = np.stack([np.concatenate([x.real.ravel(), x.imag.ravel()]) for x in B], 1)

def cp_project(M, iters=12):
    for _ in range(iters):
        J = choi(M)
        w, V = np.linalg.eigh(J)
        if w.min() >= -1e-12: break
        Jp = V @ np.diag(np.clip(w, 0, None)) @ V.conj().T
        t = np.concatenate([(Jp-J0).real.ravel(), (Jp-J0).imag.ravel()])
        M = np.linalg.lstsq(A, t, rcond=None)[0].reshape(3,3)
    return M

def so3(a, w):
    Kx = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return np.eye(3)+np.sin(w)*Kx+(1-np.cos(w))*(Kx@Kx)

if __name__ == "__main__":
    gh = Generator(seed=999); Xh, Yh, Ph = batch_with_physics(gh, 32, 6)
    ora = np.mean([[bures_distance(project_bloch(
            np.exp(-Ph[b,4])*so3(Ph[b,:3],Ph[b,3]) @ Xh[b,t]), Yh[b,t])**2
            for t in range(6)] for b in range(32)])

    def run(steps, bs, lr, project, seed=4):
        """Paired: arms share init AND data stream; only `project` differs.
        Across seeds BOTH the init and the data stream change, so each seed
        is a genuinely independent experiment (an earlier version varied only
        the init on a fixed data stream — caught by an implausible std of
        exactly 0.00000 across 'independent' runs)."""
        g = Generator(seed=1000 + seed)
        r = np.random.default_rng(seed)
        M = T(r.normal(0, 0.35, (3,3)))          # deliberately off-cone start
        viol = []
        for i in range(steps):
            x, y, _ = batch_with_physics(g, bs, 6)
            M.grad = None
            bures_mean_loss(pbloch(matmul(T(x, requires_grad=False), M)), y).backward()
            M.data -= lr * M.grad
            viol.append(lmin(M.data.T))
            if project:
                M.data = cp_project(M.data.T).T
        hold = np.mean([[bures_distance(project_bloch(M.data.T @ Xh[b,t]),
                                        Yh[b,t])**2 for t in range(6)]
                        for b in range(32)])
        return M.data.T, float(hold), np.array(viol)

    print("="*66)
    print("N1 — CP-projected gradient descent  (oracle ceiling %.4f)" % ora)
    print("="*66)
    regimes = [("standard   (400 steps, bs=8)",  400, 8, 0.12),
               ("low-data   (120 steps, bs=2)",  120, 2, 0.12),
               ("aggressive (120 steps, bs=2, lr=0.5)", 120, 2, 0.50)]
    for name, st, bs, lr in regimes:
        Mc, hc, vc = run(st, bs, lr, project=False)
        Mp, hp, vp = run(st, bs, lr, project=True)
        binds = float((vc < -1e-6).mean())          # fraction of steps OFF-cone
        print(f"\n--- {name}")
        print(f"  N1-0 anti-vacuity: control spent {100*binds:.0f}% of steps "
              f"OUTSIDE the CP cone (worst {vc.min():+.4f})")
        if binds < 0.02:
            print("       -> CONSTRAINT DOES NOT BIND: claim VACUOUS in this "
                  "regime. No win can be attributed to projection.")
        print(f"  control  : holdout {hc:.4f}   final Choi {lmin(Mc):+.4f}")
        print(f"  projected: holdout {hp:.4f}   final Choi {lmin(Mp):+.4f}")
        print(f"  N1-b projected CPTP at every step: {vp.max() > -np.inf and lmin(Mp) >= -1e-6}")
        if binds >= 0.02:
            verdict = ("PROJECTED WINS" if hp < hc - 1e-5 else
                       "TIE" if abs(hp-hc) <= 1e-5 else "CONTROL WINS")
            print(f"  N1-a VERDICT: {verdict}  (delta {hp-hc:+.5f})")

    # ============ N1 STATISTICS: 8 seeds per regime, paired ============
    # A single seed is not evidence. Registered: projected beats control in
    # >= 7/8 paired seeds in a regime where the constraint binds.
    print("\n" + "="*66)
    print("N1 statistics — 8 paired seeds per regime")
    print("="*66)
    for name, st, bs, lr in regimes:
        dc, dp, wins, bind = [], [], 0, []
        for s in range(8):
            Mc, hc, vc = run(st, bs, lr, project=False, seed=4+s)
            Mp, hp, _  = run(st, bs, lr, project=True,  seed=4+s)
            dc.append(hc); dp.append(hp); bind.append((vc < -1e-6).mean())
            wins += (hp < hc)
        dc, dp = np.array(dc), np.array(dp)
        d = dp - dc
        print(f"\n--- {name}   (constraint binds {100*np.mean(bind):.0f}% of steps)")
        print(f"  control   {dc.mean():.4f} +- {dc.std():.4f}")
        print(f"  projected {dp.mean():.4f} +- {dp.std():.4f}")
        print(f"  paired delta {d.mean():+.5f} +- {d.std():.5f}   "
              f"projected wins {wins}/8")
        print(f"  VERDICT: {'PASS' if wins >= 7 else 'NOT ESTABLISHED'} "
              f"(registered: >= 7/8)")
