import numpy as np, time, json, sys, os
sys.path.insert(0, ".")
from quasar_v02_experiment import QuasarV02
from quasar.quasar import make_holdout

class QV02n(QuasarV02):
    """probe size configurable; progress vs uniform only."""
    def __init__(self, *a, probe_n=4, **k):
        super().__init__(*a, **k); self.probe_n = probe_n
    def run_mode(self, mode, rounds, npr, epr, holdout):
        for _ in range(rounds):
            X, Y, _ = self.generate_round(npr)
            for _ in range(epr):
                self.learner.train_step(X, Y)
            be = self.probe_bin_errors(n_per_bin=self.probe_n)
            if mode == "progress":
                self.self_direct_progress(be)
        return self.learner.loss(*holdout)

ROUNDS, NPR, EPR, SEQ = 7, 10, 3, 6
Xh, Yh = make_holdout(seed=777, n=16, seq_len=SEQ)
probe = int(sys.argv[1]); seeds = [int(s) for s in sys.argv[2:]]
fn = f"f18a_probe{probe}.json"
out = json.load(open(fn)) if os.path.exists(fn) else {"uniform": {}, "progress": {}}
for seed in seeds:
    for mode in ("uniform", "progress"):
        if str(seed) in out[mode]: continue
        t = time.time()
        q = QV02n(seed=seed, seq_len=SEQ, probe_n=probe)
        out[mode][str(seed)] = q.run_mode(mode, ROUNDS, NPR, EPR, (Xh, Yh))
        json.dump(out, open(fn, "w"))
        print(f"probe{probe} seed{seed} {mode}: {out[mode][str(seed)]:.4f} ({time.time()-t:.0f}s)", flush=True)
