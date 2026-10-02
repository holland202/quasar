"""Repair FINDINGS.md: drop duplicate sections, fix the split F12 header.
Keeps the FIRST occurrence of each section title. Prints a diff summary.
Writes only if something changed; makes a .bak first."""
import re, shutil, sys

p = "FINDINGS.md"
src = open(p).read()
shutil.copy(p, p + ".bak")

# merge the accidental two-line F12 header into one
src = src.replace(
    "## F12 — R1 CLOSED: EM dictionary + aggregate cold-start; physicality is a\n"
    "## REGULARIZER, not just a checksum (2026-07-12)",
    "## F12 — R1 CLOSED: EM dictionary + aggregate cold-start; "
    "physicality is a REGULARIZER, not just a checksum (2026-07-12)")

lines = src.split("\n")
heads = [i for i, l in enumerate(lines) if l.startswith("## ")]
preamble = lines[:heads[0]] if heads else lines
secs = []
for n, i in enumerate(heads):
    j = heads[n + 1] if n + 1 < len(heads) else len(lines)
    secs.append((lines[i], lines[i:j]))

seen, kept, dropped = set(), [], []
for title, body in secs:
    key = title.split("(")[0].strip()
    if key in seen:
        dropped.append(title[:60])
        continue
    seen.add(key)
    kept.append(body)

out = "\n".join(preamble + [l for b in kept for l in b]).rstrip() + "\n"
if out == src:
    print("no changes needed")
    sys.exit()
open(p, "w").write(out)
print("dropped duplicate sections:")
for d in dropped:
    print("  -", d)
print("\nfinal ledger:")
for l in out.split("\n"):
    if l.startswith("## "):
        print("  " + l[:72])
