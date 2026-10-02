"""
VERITAS RUNTIME v0 — VERA's Governance layer (I2), runnable.
=============================================================
Fusion of verified lineages: SLC/Veritas thermal governor (Schmitt-
trigger duty cycle), SECP Sentinel gate + SHA-256 hash-chain audit,
self-healing rollback — wrapped around REAL QUASAR/LUCID training
(Bures-loss channel learning on physics trajectories).

Two modes:
  python veritas_runtime.py            # sandbox self-test (synthetic
                                       # thermal physics), claims V1-V4
  python veritas_runtime.py --live 90  # on-device: real sysfs thermal
                                       # zones, 90 s governed training

REGISTERED (sandbox):
  V1  governed run: T never exceeds T_MAX + 0.3; ungoverned control
      exceeds T_MAX by > 3 deg on the same thermal model.
  V2  governed training still converges: final loss < 0.5 * initial.
  V3  injected gradient bomb (step 250, grads x 1e6) is detected and
      rolled back; final loss still < 0.5 * initial.
  V4  audit chain verifies end-to-end; single-field tamper detected.
On-device (V5, completes VERA I2): live sysfs run prints the measured
duty-cycle curve; paste it into FINDINGS.
"""
import sys, time, json, glob, hashlib
import numpy as np
from quasar import Generator
from backprop import T as Tn, matmul, add, pbloch, bures_mean_loss
from m3_articulation import batch_with_physics

T_MAX, T_HIGH, T_LOW, T_AMB = 42.0, 42.0, 38.0, 30.0

# ---------------- thermal sources ----------------
class SimThermal:
    """Honest RC model: heats while computing, Newton-cools while idle."""
    def __init__(self):
        self.T = T_AMB
    def step(self, running, dt=1.0):
        heat = 1.6 if running else 0.0
        self.T += dt * (heat - 0.08 * (self.T - T_AMB))
        return self.T
    def read(self):
        return self.T

class SysfsThermal:
    """Real sensors only. FAILS LOUD if it cannot read them.
    (Anti-pattern this exists to avoid: `except: return 45.0` — a
    governor that fabricates a safe temperature is worse than none.)"""
    def __init__(self):
        self.zones = [z for z in
                      glob.glob("/sys/class/thermal/thermal_zone*/temp")]
        live = []
        for z in self.zones:
            try:
                int(open(z).read().strip())
                live.append(z)
            except Exception:
                pass
        self.zones = live
        if not self.zones:
            raise RuntimeError(
                "REFUSING TO RUN: no readable /sys/class/thermal zones.\n"
                "  Crostini/containers often expose none. A thermal governor\n"
                "  without a thermometer is theater. Run --sim here, and run\n"
                "  --live on hardware with real sensors (e.g. the S25).")
    def read(self):
        vals = []
        for z in self.zones:
            try:
                vals.append(int(open(z).read().strip()) / 1000.0)
            except Exception:
                pass
        if not vals:
            raise RuntimeError("thermal read failed mid-run — halting "
                               "rather than guessing a temperature")
        return max(vals)

# ---------------- audit chain (SECP lineage) ----------------
class Audit:
    def __init__(self):
        self.chain = []
    def log(self, event, **payload):
        prev = self.chain[-1]["hash"] if self.chain else "GENESIS"
        rec = {"t": round(time.time(), 3), "event": event,
               "payload": payload, "prev": prev}
        rec["hash"] = hashlib.sha256(
            json.dumps(rec, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.chain.append(rec)
    def verify(self):
        prev = "GENESIS"
        for r in self.chain:
            body = {k: r[k] for k in r if k != "hash"}
            if body["prev"] != prev or hashlib.sha256(
                json.dumps(body, sort_keys=True, default=str).encode()
            ).hexdigest() != r["hash"]:
                return False
            prev = r["hash"]
        return True

# ---------------- the governed workload ----------------
class Workload:
    """Real physics training: LUCID channel under Bures loss."""
    def __init__(self, seed=11):
        self.gen = Generator(seed=seed)
        r = np.random.default_rng(4)
        self.M = Tn(r.normal(0, 0.1, (3, 3)))
        self.c = Tn(r.normal(0, 0.01, (3,)))
        self.bomb_at = None
        self.k = 0
        hg = Generator(seed=999)
        self._hx, self._hy, _ = batch_with_physics(hg, 16, 6)
    def loss_step(self, lr=0.12):
        x, y, _ = batch_with_physics(self.gen, 8, 6)
        self.M.grad = None; self.c.grad = None
        L = bures_mean_loss(pbloch(add(matmul(
            Tn(x, requires_grad=False), self.M), self.c)), y)
        L.backward()
        g = 1e6 if self.bomb_at is not None and self.k == self.bomb_at else 1.0
        self.M.data -= lr * g * self.M.grad
        self.c.data -= lr * g * self.c.grad
        self.k += 1
        return float(L.data)
    def probe(self):
        """Forward-only loss on a FIXED holdout: the guard's evidence.
        Never used for updates — only to decide trust in current state."""
        x, y = self._hx, self._hy
        L = bures_mean_loss(pbloch(add(matmul(
            Tn(x, requires_grad=False), self.M), self.c)), y)
        return float(L.data)

    def snapshot(self):
        return (self.M.data.copy(), self.c.data.copy())
    def restore(self, snap):
        self.M.data, self.c.data = snap[0].copy(), snap[1].copy()

# ---------------- the governor ----------------
def govern(steps, thermal, workload, audit, ckpt_every=50, live=False,
           budget_s=None):
    paused, best, snap = False, float("inf"), workload.snapshot()
    last_ckpt = 0
    losses, temps, duty = [], [], []
    t0 = time.time()
    i = 0
    while i < steps:
        if budget_s and time.time() - t0 > budget_s:
            break
        Tnow = thermal.read()
        # Schmitt trigger (Veritas)
        if not paused and Tnow >= T_HIGH:
            paused = True
            audit.log("PAUSE", temp=round(Tnow, 2), step=i)
        elif paused and Tnow <= T_LOW:
            paused = False
            audit.log("RESUME", temp=round(Tnow, 2), step=i)
        if paused:
            duty.append(0)
            if isinstance(thermal, SimThermal):
                thermal.step(False)
            else:
                time.sleep(1.0)
            temps.append(thermal.read())
            continue
        duty.append(1)
        L = workload.loss_step()
        if isinstance(thermal, SimThermal):
            thermal.step(True)
        temps.append(thermal.read())
        # Sentinel: judge the state AFTER the update, on held-out evidence.
        # (A guard that trusts the pre-update loss checkpoints poison.)
        p = workload.probe()
        if not np.isfinite(p) or p > 3.0 * max(best, 0.02):
            workload.restore(snap)
            audit.log("ROLLBACK", probe=round(min(p, 1e9), 3), step=i,
                      restored_to=last_ckpt)
            losses.append(workload.probe())
        else:
            best = min(best, p)
            losses.append(p)
            if i % ckpt_every == 0:          # checkpoint VERIFIED state only
                snap = workload.snapshot()
                last_ckpt = i
                audit.log("CKPT", step=i, probe=round(p, 4))
        if live and i % 20 == 0:
            dc = 100 * np.mean(duty[-60:]) if duty else 100
            print(f"  step {i:4d}  T={Tnow:5.1f}C  loss={L:.4f}  "
                  f"duty={dc:5.1f}%  {'PAUSED' if paused else 'RUN'}")
        i += 1
    return np.array(losses), np.array(temps), np.array(duty)

# ---------------- sandbox self-test ----------------
def self_test():
    print("=" * 64)
    print("VERITAS RUNTIME — registered claims V1-V4 (synthetic thermal)")
    print("=" * 64)
    # ungoverned control: same thermal model, no gating
    th = SimThermal(); w = Workload(); L0 = w.loss_step()
    peak_un = T_AMB
    for _ in range(400):
        w.loss_step(); peak_un = max(peak_un, th.step(True))
    # governed run
    th2, w2, aud = SimThermal(), Workload(), Audit()
    aud.log("INIT", mode="sim", T_MAX=T_MAX)
    losses, temps, duty = govern(600, th2, w2, aud)
    v1 = temps.max() <= T_MAX + 0.3 and peak_un > T_MAX + 3
    v2 = losses[-1] < 0.5 * losses[0]
    print(f"V1 thermal envelope: governed peak {temps.max():.1f}C "
          f"(<= {T_MAX}+0.3) vs ungoverned {peak_un:.1f}C -> "
          f"{'PASS' if v1 else 'FAIL'}")
    print(f"V2 converges under gating: {losses[0]:.3f} -> {losses[-1]:.3f} "
          f"(duty {100*duty.mean():.0f}%) -> {'PASS' if v2 else 'FAIL'}")
    # V3 gradient bomb
    th3, w3, aud3 = SimThermal(), Workload(), Audit()
    w3.bomb_at = 250
    losses3, _, _ = govern(600, th3, w3, aud3)
    rb = any(r["event"] == "ROLLBACK" for r in aud3.chain)
    v3 = rb and losses3[-1] < 0.5 * losses3[0]
    print(f"V3 bomb@250 -> rollback fired: {rb}, recovered to "
          f"{losses3[-1]:.3f} -> {'PASS' if v3 else 'FAIL'}")
    ok = aud3.verify()
    aud3.chain[1]["payload"]["step"] = 9999
    v4 = ok and not aud3.verify()
    print(f"V4 audit chain verifies + tamper detected -> "
          f"{'PASS' if v4 else 'FAIL'}")
    assert v1 and v2 and v3 and v4
    print("\nALL 4 SANDBOX CLAIMS PASS.  On-device half (V5):")
    print("  python veritas_runtime.py --live 90")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        secs = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        th = SysfsThermal()
        print(f"[live] {len(th.zones)} thermal zones, start "
              f"{th.read():.1f}C, budget {secs}s, envelope "
              f"{T_LOW}-{T_HIGH}C")
        aud = Audit(); aud.log("INIT", mode="sysfs", zones=len(th.zones))
        losses, temps, duty = govern(10**9, th, Workload(), aud,
                                     live=True, budget_s=secs)
        print(f"\n[live verdict] steps {len(losses)} | loss "
              f"{losses[0]:.3f} -> {losses[-1]:.3f} | T range "
              f"{temps.min():.1f}-{temps.max():.1f}C | duty "
              f"{100*duty.mean():.0f}% | audit {aud.verify()} | "
              f"events {len(aud.chain)}")
    else:
        self_test()
