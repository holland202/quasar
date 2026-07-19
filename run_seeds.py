import numpy as np, time, json, sys, os
sys.path.insert(0, ".")
from quasar_v02_experiment import QuasarV02
from quasar.quasar import make_holdout
ROUNDS, NPR, EPR, SEQ = 7, 10, 3, 6
Xh, Yh = make_holdout(seed=777, n=16, seq_len=SEQ)
seeds = [int(s) for s in sys.argv[1:]]
out = json.load(open("v02_results.json")) if os.path.exists("v02_results.json") else {"uniform": {}, "error": {}, "progress": {}}
for seed in seeds:
    for mode in ("uniform", "error", "progress"):
        if str(seed) in out[mode]:
            continue
        t = time.time()
        q = QuasarV02(seed=seed, seq_len=SEQ)
        out[mode][str(seed)] = q.run_mode(mode, ROUNDS, NPR, EPR, (Xh, Yh))
        json.dump(out, open("v02_results.json", "w"))
        print(f"seed {seed} {mode}: {out[mode][str(seed)]:.4f} ({time.time()-t:.0f}s)", flush=True)
