"""
Quantum Geometric Reinforcement Learning (QGRL) v1.0
====================================================
RL agent for single-qubit control. The QGT is the policy network.

HONEST SCOPE (read this before believing any hype):
  This is a CLASSICAL SIMULATION of single-qubit unitary control on the
  Bloch sphere. It is "quantum" in the same sense any single-qubit gate
  synthesis problem is quantum: the state space is a real quantum state
  space (the Bloch ball), the metric is the genuine Bures/Fubini-Study
  metric, and the actions are genuine SU(2)/SO(3) rotations. It does NOT
  run on quantum hardware and claims no quantum computational advantage.

  What it DOES demonstrate:
    - The QGT backbone can act as a policy that maps a quantum state to a
      control action (rotation axis + angle).
    - Direct policy search (finite-difference gradient ascent, the same
      no-autodiff method the QGT self-test uses) measurably improves the
      policy's return.
    - We bracket the learned policy between a RANDOM policy (floor) and the
      ANALYTICAL OPTIMAL controller (ceiling), so "it works" is a
      quantitative statement, not a vibe.

State:  Bloch vector r ∈ S² (we use pure states, |r| = 1)
Action: rotation axis n ∈ S² and angle θ ∈ [0, π]
Reward: -d_B(r', target)²   (negative squared Bures distance after the move)
Target: fixed = |0⟩ = [0, 0, 1]  (north pole)
"""

import numpy as np
from numpy import sqrt, pi, sin, cos
from numpy.linalg import norm

from quantum_geometric_transformer import (
    QuantumGeometricTransformer,
    bures_distance,
    project_bloch,
)


# ------------------------------------------------------------------
# Rotation utility (Rodrigues' formula on SO(3))
# ------------------------------------------------------------------
def rotation_matrix(axis, angle):
    """SO(3) rotation about `axis` by `angle` (radians)."""
    a = axis / (norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return np.eye(3) + sin(angle) * K + (1 - cos(angle)) * (K @ K)


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
class BlochControlEnv:
    """Drive a random pure single-qubit state to a fixed target state."""

    def __init__(self, target=None, horizon=3, tol=0.01, seed=0):
        self.target = (np.array([0.0, 0.0, 1.0]) if target is None
                       else np.asarray(target, float))
        self.target = self.target / norm(self.target)
        self.horizon = horizon
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.state = None
        self.t = 0

    def reset(self, state=None):
        if state is None:
            r = self.rng.standard_normal(3)
            self.state = r / norm(r)
        else:
            self.state = np.asarray(state, float)
            self.state = self.state / norm(self.state)
        self.t = 0
        return self.state.copy()

    def step(self, axis, angle):
        R = rotation_matrix(axis, angle)
        nxt = R @ self.state
        nxt = nxt / (norm(nxt) + 1e-12)
        d = bures_distance(nxt, self.target)
        reward = -d ** 2
        self.state = nxt
        self.t += 1
        done = (d < self.tol) or (self.t >= self.horizon)
        return nxt.copy(), reward, done, {"bures": d}


# ------------------------------------------------------------------
# Policies
# ------------------------------------------------------------------
class QGTPolicy:
    """QGT backbone as a deterministic control policy: state -> (axis, angle)."""

    def __init__(self, seed=0, action_scale=1.0):
        self.qgt = QuantumGeometricTransformer(d_model=3, n_heads=1,
                                               d_ff=12, beta=1.0, seed=seed)
        self.action_scale = action_scale

    def act(self, state):
        out = self.qgt.forward(state[np.newaxis, np.newaxis, :])[0, 0]
        mag = norm(out)
        if mag < 1e-9:
            return np.array([0.0, 0.0, 1.0]), 0.0
        axis = out / mag
        angle = np.clip(mag * self.action_scale * pi, 0, pi)
        return axis, angle

    def params(self):
        return self.qgt.all_params()


def analytical_optimal_action(state, target):
    """Ground-truth one-step controller: rotate `state` onto `target`.

    Axis = state x target (normalised), angle = arccos(state . target).
    Reaches Bures distance 0 in a single step (up to numerical error).
    """
    s = state / norm(state)
    t = target / norm(target)
    c = np.clip(np.dot(s, t), -1, 1)
    angle = np.arccos(c)
    axis = np.cross(s, t)
    na = norm(axis)
    if na < 1e-9:                      # already aligned or antipodal
        if c > 0:
            return np.array([0.0, 0.0, 1.0]), 0.0
        # antipodal: any perpendicular axis, rotate by pi
        perp = np.cross(s, np.array([1.0, 0.0, 0.0]))
        if norm(perp) < 1e-9:
            perp = np.cross(s, np.array([0.0, 1.0, 0.0]))
        return perp / norm(perp), pi
    return axis / na, angle


def random_action(rng):
    axis = rng.standard_normal(3)
    axis = axis / norm(axis)
    angle = rng.uniform(0, pi)
    return axis, angle


# ------------------------------------------------------------------
# Rollout + objective
# ------------------------------------------------------------------
def rollout(env, action_fn, start_state):
    """Run one episode; return (total_reward, final_bures)."""
    s = env.reset(start_state)
    total = 0.0
    last_b = bures_distance(s, env.target)
    done = False
    while not done:
        axis, angle = action_fn(s)
        s, r, done, info = env.step(axis, angle)
        total += r
        last_b = info["bures"]
    return total, last_b


def evaluate_policy(env, policy, start_states):
    """Mean total reward and mean final Bures distance over fixed starts."""
    tot, fin = [], []
    for s0 in start_states:
        T, B = rollout(env, policy.act, s0)
        tot.append(T)
        fin.append(B)
    return float(np.mean(tot)), float(np.mean(fin))


# ------------------------------------------------------------------
# Direct policy search (finite-difference gradient ascent)
# ------------------------------------------------------------------
def train(env, policy, start_states, iters=15, lr=0.15, eps=1e-4, verbose=True):
    """Maximise mean total reward via finite-difference gradient ascent.

    No autodiff — identical estimation style to the QGT self-test's Test 6.
    Deterministic objective (fixed start set), so the gradient is well-defined.
    """
    def J():
        return evaluate_policy(env, policy, start_states)[0]

    history = []
    J0 = J()
    history.append(J0)
    if verbose:
        _, b0 = evaluate_policy(env, policy, start_states)
        print(f"  iter  0 | return={J0:+.4f} | mean final d_B={b0:.4f}")

    for it in range(1, iters + 1):
        for p in policy.params():
            g = np.zeros_like(p)
            nd = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
            for val in nd:
                idx = nd.multi_index
                orig = val.item()
                val[...] = orig + eps
                Jp = J()
                val[...] = orig - eps
                Jm = J()
                val[...] = orig
                g[idx] = (Jp - Jm) / (2 * eps)
            p += lr * g          # ASCENT (maximising reward)
        Ji = J()
        history.append(Ji)
        if verbose:
            _, bi = evaluate_policy(env, policy, start_states)
            print(f"  iter {it:2d} | return={Ji:+.4f} | mean final d_B={bi:.4f}")
    return history


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------
def main():
    print("=" * 64)
    print("QUANTUM GEOMETRIC REINFORCEMENT LEARNING (QGRL) v1.0")
    print("=" * 64)
    print("Task: drive random pure states to |0> = [0,0,1]")
    print("Metric: Bures distance | Actions: SO(3) rotations\n")

    H = 3
    env = BlochControlEnv(horizon=H, tol=0.01, seed=1)

    # Fixed evaluation / training start set (deterministic objective)
    srng = np.random.default_rng(7)
    start_states = []
    for _ in range(24):
        r = srng.standard_normal(3)
        start_states.append(r / norm(r))

    # ---- Baselines --------------------------------------------------
    print("[Baselines] (mean over 24 fixed start states, horizon=%d)" % H)

    # Random policy
    rrng = np.random.default_rng(99)
    class RandPol:
        def act(self, s): return random_action(rrng)
    rand_T, rand_B = evaluate_policy(env, RandPol(), start_states)
    print(f"  Random  : return={rand_T:+.4f} | mean final d_B={rand_B:.4f}")

    # Analytical optimal
    class OptPol:
        def act(self, s): return analytical_optimal_action(s, env.target)
    opt_T, opt_B = evaluate_policy(env, OptPol(), start_states)
    print(f"  Optimal : return={opt_T:+.4f} | mean final d_B={opt_B:.4f}")

    # ---- Learn the QGT policy --------------------------------------
    print("\n[Training] QGT policy via finite-difference policy search")
    policy = QGTPolicy(seed=0, action_scale=1.0)
    hist = train(env, policy, start_states, iters=15, lr=0.15, eps=1e-4)

    learn_T, learn_B = evaluate_policy(env, policy, start_states)

    # ---- Verdict ----------------------------------------------------
    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    print(f"  Random policy     : return={rand_T:+.4f}  final d_B={rand_B:.4f}")
    print(f"  QGT (start)       : return={hist[0]:+.4f}  ")
    print(f"  QGT (learned)     : return={learn_T:+.4f}  final d_B={learn_B:.4f}")
    print(f"  Analytical optimal: return={opt_T:+.4f}  final d_B={opt_B:.4f}")

    # Normalised score: 0 = random floor, 1 = optimal ceiling
    span = (opt_T - rand_T)
    score0 = (hist[0] - rand_T) / span if span > 1e-9 else float('nan')
    score1 = (learn_T - rand_T) / span if span > 1e-9 else float('nan')
    print(f"\n  Normalised return (0=random, 1=optimal):")
    print(f"    before training: {score0:+.3f}")
    print(f"    after  training: {score1:+.3f}")

    improved = learn_T > hist[0] + 1e-6
    beats_random = learn_T > rand_T
    print("\n  Checks:")
    print(f"    training improved return : {improved}")
    print(f"    learned beats random     : {beats_random}")

    assert improved, "Policy search did not improve return!"
    assert beats_random, "Learned policy no better than random!"
    print("\n  ✅ QGRL learning loop verified: return increased and beats random.")
    return hist, (rand_T, learn_T, opt_T)


if __name__ == "__main__":
    main()
