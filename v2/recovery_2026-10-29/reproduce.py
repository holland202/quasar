#!/usr/bin/env python3
"""Re-run recovered F20-F24 scripts and compare each output with the committed container output.

Run from v2/:
    python3 recovery_2026-10-29/reproduce.py f24_leakfree_temp_matched      # one script
    python3 recovery_2026-10-29/reproduce.py --all                          # all six (F20 needs SciPy, ~1 min)
    python3 recovery_2026-10-29/reproduce.py --all --label device           # writes <script>_device.txt
    python3 recovery_2026-10-29/reproduce.py --sabotage f21_transition_tokens   # must report DIFFERENT, exit 1

For each script it:
  1. checks the script's sha256 against SHA256SUMS (the bytes recovered from Google Drive);
  2. runs `python -B <script>.py` from v2/, stdout and stderr together, and appends "exit N";
  3. writes recovery_2026-10-29/<script>_<label>.txt;
  4. compares it with <script>_container.txt, ignoring only wall-clock lines (compare_outputs.py).
Exit 0 only if every script's bytes match and every output is identical.
"""
import hashlib
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from compare_outputs import compare  # noqa: E402

SCRIPTS = ["f20_purity_channel", "f21_transition_tokens", "f22_disjoint_quality",
           "f23_metric_at_evaluation", "f24_leakfree_temp_matched", "demo_quasar_method"]


def recorded_hashes():
    out = {}
    with open(os.path.join(HERE, "SHA256SUMS"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                digest, name = line.split(None, 1)
                out[name.strip().lstrip("*")] = digest
    return out


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def run(name):
    p = subprocess.run([sys.executable, "-B", name + ".py"], cwd=V2,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode("utf-8", errors="replace") + f"exit {p.returncode}\n"


def main(argv):
    label, sabotage, names = "container", False, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--all":
            names = list(SCRIPTS)
        elif a == "--label":
            i += 1
            label = argv[i]
        elif a == "--sabotage":
            sabotage = True
        elif a in SCRIPTS:
            names.append(a)
        else:
            sys.exit(f"unknown argument {a!r}\n\n" + __doc__)
        i += 1
    if not names:
        sys.exit(__doc__)
    if sabotage:
        label = "sabotage"  # never overwrite the container reference with a perturbed run
    try:
        import numpy
        npv = numpy.__version__
    except ImportError:
        npv = "missing"
    print(f"environment: {platform.machine()} | Python {platform.python_version()} | NumPy {npv}")
    hashes = recorded_hashes()
    all_ok = True
    for name in names:
        script = os.path.join(V2, name + ".py")
        got, want = sha256(script), hashes.get("v2/" + name + ".py")
        if got != want:
            print(f"{name}: SCRIPT BYTES DIFFER from SHA256SUMS ({got[:16]} vs {str(want)[:16]}); not run")
            all_ok = False
            continue
        text = run(name)
        if sabotage:
            # Change the first digit found after the banner, so the comparison must fail.
            pos = next(k for k, ch in enumerate(text) if ch.isdigit() and k > 200)
            text = text[:pos] + str((int(text[pos]) + 1) % 10) + text[pos + 1:]
        if label != "container":
            with open(os.path.join(HERE, f"{name}_{label}.txt"), "w", encoding="utf-8") as fh:
                fh.write(text)
            with open(os.path.join(HERE, f"{name}_container.txt"), encoding="utf-8") as fh:
                ref = fh.read()
            d = compare(ref, text)
            verdict = "IDENTICAL" if not d else f"DIFFERENT ({len(d)} line(s); first: line {d[0][0]})"
        else:
            with open(os.path.join(HERE, f"{name}_container.txt"), "w", encoding="utf-8") as fh:
                fh.write(text)
            verdict = "written as the container reference"
        last = [ln for ln in text.splitlines() if ln.startswith("exit ")][-1]
        print(f"{name}: bytes match recorded sha256 | {last} | {verdict}")
        all_ok &= verdict.startswith(("IDENTICAL", "written"))
    print("ALL IDENTICAL" if all_ok else "NOT ALL IDENTICAL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
