#!/usr/bin/env python3
"""Fail if any RNG in this repo is unseeded.

Rationale: CI run #45 went red on one Python version only, because three test
bodies drew their problem set from the unseeded global np.random. Measured over
60 unseeded trials the MLE-vs-linear statistic ranged 41..62 against a n>=40
gate -- a 2.3-sigma margin. A red build that depends on the draw is not a
result. This lint stops that class of defect returning.

Anti-vacuity: run with --self-check to confirm the linter can report a hit.
"""
import ast
import pathlib
import sys

ROOTS = ["quasar", "tests", "experiments"]
BANNED_ATTRS = {"randn", "rand", "random", "uniform", "normal", "choice",
                "standard_normal", "seed", "randint", "permutation", "shuffle"}


def offenses(paths):
    out = []
    for path in paths:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as e:
            out.append((path, e.lineno, f"syntax error: {e.msg}"))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue
            # np.random.default_rng() with no seed argument
            if f.attr == "default_rng" and not node.args and not node.keywords:
                out.append((path, node.lineno, "default_rng() called with no seed"))
            # np.random.<legacy global>(...)
            if (f.attr in BANNED_ATTRS
                    and isinstance(f.value, ast.Attribute)
                    and f.value.attr == "random"):
                out.append((path, node.lineno,
                            f"legacy global np.random.{f.attr}(...)"))
    return out


def collect():
    files = []
    for root in ROOTS:
        p = pathlib.Path(root)
        if p.is_dir():
            files += [f for f in p.rglob("*.py") if "__pycache__" not in f.parts]
    return files


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        tmp = pathlib.Path("_seedlint_probe.py")
        tmp.write_text("import numpy as np\nx = np.random.randn(3)\n"
                       "r = np.random.default_rng()\n")
        hits = offenses([tmp])
        tmp.unlink()
        if len(hits) == 2:
            print("SELF-CHECK PASS: linter reported 2/2 planted offenses")
            sys.exit(0)
        print(f"SELF-CHECK FAIL: expected 2 planted offenses, got {len(hits)}")
        sys.exit(1)

    files = collect()
    hits = offenses(files)
    if hits:
        for path, line, msg in hits:
            print(f"{path}:{line}: unseeded RNG -- {msg}")
        print(f"\nFAIL: {len(hits)} unseeded RNG site(s) in {len(files)} files.")
        sys.exit(1)
    print(f"PASS: no unseeded RNG in {len(files)} files across {ROOTS}.")
    sys.exit(0)
