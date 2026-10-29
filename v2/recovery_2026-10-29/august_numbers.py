#!/usr/bin/env python3
"""Check that the numbers recorded in August 2026 appear, character for character, in a run's output.

The August numbers below were copied from the owner's session notes of 2026-08-05..07 (not from
this repository, which held no F19-F24 record until this recovery). Each must appear verbatim in
the output of the matching script. Passing shows the recovered code prints what was reported in
August; it does not show the August code was these exact bytes.

Usage:
    python3 august_numbers.py OUTPUT_DIR                  # reads <script>_container.txt
    python3 august_numbers.py OUTPUT_DIR --label device   # reads <script>_device.txt
    python3 august_numbers.py --selftest                  # shows the check can fail
"""
import os
import sys

AUGUST = {
    "f20_purity_channel": ["0.046562", "0.046524", "0.000908", "0.000890", "-0.0008", "+0.0021",
                           "wins 0/5", "0.705", "0.725", "[FAIL] P0a", "[PASS] P0b", "1.000000"],
    "f21_transition_tokens": ["0.064336", "0.046980", "0.051280", "0.738", "0.942", "1.946", "+27.0%",
                              "0.038029", "0.025716", "0.028993", "1.005", "1.282", "+32.4%"],
    "f22_disjoint_quality": ["0.056234", "0.053425", "0.050373", "0.971", "0.826", "1.386", "+10.4%",
                             "0.032902", "0.030943", "0.029256", "0.981", "0.917", "+11.1%"],
    "f23_metric_at_evaluation": ["+10.4%", "+10.8%", "+11.1%", "+11.0%", "0.826", "0.917"],
    "f24_leakfree_temp_matched": ["215.876", "59.049", "1072.682", "280.256", "0.23%", "0.66%",
                                  "0.008369", "0.004987", "0.004999", "+40.4%", "0.005251",
                                  "0.002813", "0.002832", "+46.4%"],
    "demo_quasar_method": ["0.008197", "0.005078", "0.005090", "+38.0%", "0.23%", "0.006781",
                           "0.003582", "0.003604", "+47.2%", "0.62%", "197.807", "53.957",
                           "1027.688", "268.495"],
}


def check(texts):
    """texts: {script: output text}. Returns (found, total, missing list)."""
    found, total, missing = 0, 0, []
    for script, tokens in AUGUST.items():
        txt = texts.get(script, "")
        for t in tokens:
            total += 1
            if t in txt:
                found += 1
            else:
                missing.append((script, t))
    return found, total, missing


def selftest():
    full = {s: " ".join(toks) for s, toks in AUGUST.items()}
    f, n, m = check(full)
    ok1 = (f == n and not m)
    one_off = dict(full)
    one_off["f24_leakfree_temp_matched"] = one_off["f24_leakfree_temp_matched"].replace("0.66%", "0.67%")
    f2, n2, m2 = check(one_off)
    ok2 = (m2 == [("f24_leakfree_temp_matched", "0.66%")])
    ok3 = check({})[0] == 0
    for name, ok in [("all tokens present -> pass", ok1), ("one digit changed -> exactly that token missing", ok2),
                     ("no output -> nothing found", ok3)]:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok1 and ok2 and ok3


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    if not args:
        sys.exit(__doc__)
    label = args[args.index("--label") + 1] if "--label" in args else "container"
    suffix = f"_{label}.txt"
    d = args[0]
    texts = {}
    for s in AUGUST:
        p = os.path.join(d, s + suffix)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                texts[s] = fh.read()
    for s, toks in AUGUST.items():
        miss = [t for t in toks if t not in texts.get(s, "")]
        state = "no output file" if s not in texts else f"{len(toks) - len(miss)}/{len(toks)}"
        print(f"{s + suffix:40s} {state}" + (f"  MISSING {miss}" if miss and s in texts else ""))
    f, n, m = check(texts)
    print(f"AUGUST NUMBERS FOUND: {f}/{n}")
    sys.exit(0 if f == n else 1)
