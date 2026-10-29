#!/usr/bin/env python3
"""Compare two runs of a recovered QUASAR script, ignoring wall-clock lines only.

Usage:
    python3 compare_outputs.py REFERENCE.txt NEW.txt     # exit 0 = identical, 1 = differ
    python3 compare_outputs.py --selftest                 # shows the check can fail

Lines dropped before comparing (they measure the machine, not the result):
  - `real`/`user`/`sys` lines from the shell's `time`
  - the demo's "runtime N.Ns" line
Nothing else is normalised: every other character, including every digit, must match.
"""
import re
import sys

WALL_CLOCK = re.compile(r"^(real|user|sys)\s+\d+m[\d.]+s$|^\s*runtime \d+(\.\d+)?s\b")


def lines(text):
    return [ln.rstrip("\n") for ln in text.splitlines() if not WALL_CLOCK.search(ln.strip())]


def compare(ref_text, new_text):
    a, b = lines(ref_text), lines(new_text)
    diffs = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b):
        diffs.append((min(len(a), len(b)) + 1, f"<{len(a)} lines>", f"<{len(b)} lines>"))
    return diffs


def selftest():
    ref = "seed 0: GEO 0.046529 | DOT 0.046524\nreal\t1m5.401s\n  runtime 2.4s, pure NumPy\nexit 1\n"
    same = "seed 0: GEO 0.046529 | DOT 0.046524\nreal\t0m58.000s\n  runtime 1.8s, pure NumPy\nexit 1\n"
    one_digit = "seed 0: GEO 0.046529 | DOT 0.046525\nreal\t1m5.401s\n  runtime 2.4s, pure NumPy\nexit 1\n"
    dropped = "seed 0: GEO 0.046529 | DOT 0.046524\n"
    checks = [
        ("wall-clock lines differ only -> identical", compare(ref, same) == []),
        ("one digit changed -> differ", len(compare(ref, one_digit)) == 1),
        ("exit line missing -> differ", compare(ref, dropped) != []),
    ]
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(ok for _, ok in checks)


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as f:
        ref = f.read()
    with open(sys.argv[2], encoding="utf-8") as f:
        new = f.read()
    d = compare(ref, new)
    for n, x, y in d[:20]:
        print(f"line {n}:\n  reference: {x}\n  new:       {y}")
    print("IDENTICAL (wall-clock lines ignored)" if not d else f"DIFFERENT: {len(d)} line(s)")
    sys.exit(0 if not d else 1)
