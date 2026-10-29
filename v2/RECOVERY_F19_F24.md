# F19–F24: recovery record (2026-10-29)

**Status:** recovered and re-run in a container (SELF-TESTED). Device reproduction of these exact bytes: NOT RUN.
This record adds no entry to [FINDINGS.md](FINDINGS.md). F20's registration sets the rule: "Nothing enters
FINDINGS.md before device reproduction."

F19–F24 were run on 2026-08-05..07, after F18, and never committed. The code lived only in the owner's Google
Drive. This file says what was recovered, what the recovered code prints today, and what is still unknown.

## What could have gone wrong, first

1. **F19 is not recovered.** No F19 script or registration was found in the owner's Drive (searched 2026-10-29).
   What is known about F19 comes from the "What changed since F19" section of [F20_PREREG.md](F20_PREREG.md). It
   cannot be re-run from this repository.
2. **Registration order cannot be checked.** None of these files has commit history. F21–F24 each say "Registered
   2026-08-07, before running", but nothing here can confirm it. F20's registration and script reached Drive 12
   seconds apart. The recovered `F20_PREREG.md` is the post-run copy: its status line still reads "REGISTERED, NOT
   YET RUN", and the run's results are appended below a rule in the same file. Whether anything above that rule
   changed after the run cannot be checked.
3. **Four Drive copies were modified after upload**: F21 about 6 minutes later, F22 about 3, F23 about 2 and F24
   about 1. What changed is unknown. Every number the owner's August notes recorded for these runs (72 in all) is
   printed by the recovered code, so any change did not alter a recorded number.
4. **Two filenames were chosen at recovery.** Drive titled each file after its first line. `F20_PREREG.md` names
   `f20_purity_channel.py`, and the owner's notes name `f23_metric_at_evaluation.py`,
   `f24_leakfree_temp_matched.py` and `demo_quasar_method.py`. `f21_transition_tokens.py` and
   `f22_disjoint_quality.py` are new names.
5. **Several registered predictions cannot be precisely wrong.** F21's P1 and P2, F22's P1 and P2 and F23's P1 have
   no numeric pass criterion. F21's P2 expected GEO ≈ DOT in the direction regime with no tolerance; F22's P1 and
   F23's P1 are registered as "genuinely open"; F22's P2 is "reported as a fact". Only F20's and F24's predictions
   have numeric criteria throughout.
6. **F24 changed five settings at once**: the leak fix, per-arm temperature calibration, CHUNK 2 → 4, SEQ_LEN 9 → 20
   (from F22) and trials per seed 60 → 40. Its registered P2 (gap under 3%) held. But the collapse of the gap cannot
   be attributed to temperature alone. The "sharper arm won 4 of 4" pattern that motivated F24 was noticed after
   those runs; a coin flip would give 4 of 4 one time in 16. The demo's closing text ("The choice of geometry does
   NOTHING once temperature is matched") states more than F24 isolates.
7. **F23 was not re-run leak-free.** F24's docstring says F23 inherits F22's leak. F23's null has not been
   repeated with F24's fixes.
8. **Device evidence is reported, not logged.** The owner reported that F21, F22 and the demo matched the
   container digit for digit on the Galaxy S25 on 2026-08-07. No device log was recovered, and whether the device
   ran these exact bytes is unknown. F20, F23 and F24 were never run on the device.
9. **Self-tested.** Claude wrote these scripts in August and re-ran them here. A deterministic re-run shows what
   this code prints; it is not an independent check of what the numbers mean.
10. **Kept defects.** F23 prints "~? samples/seed": a placeholder that was never filled. F20's appended result
    quotes seed 0's losses (BLIND 0.046562 against DOT 0.046524), not means, and F21's P1 repeats them as F20's
    figures; the F20 script prints per-seed losses and mean gaps only.

## What was recovered

Each file was downloaded from Drive as raw bytes and committed unchanged. Full hashes are in
[recovery_2026-10-29/SHA256SUMS](recovery_2026-10-29/SHA256SUMS).

| Run | File | Drive upload (UTC) | Bytes | sha256 | Container 2026-10-29 | Device |
|---|---|---|---|---|---|---|
| F20 registration | `F20_PREREG.md` | 2026-08-05 11:48:00 | 7357 | `05abb0b3…` | — | — |
| F20 | `f20_purity_channel.py` | 2026-08-05 11:48:12 | 17124 | `49176243…` | exit 1, VOID as registered; 12/12 August numbers | NOT RUN |
| F21 | `f21_transition_tokens.py` | 2026-08-07 01:24:45 | 7073 | `1ab64a74…` | 13/13 | owner-reported match, 2026-08-07 |
| F22 | `f22_disjoint_quality.py` | 2026-08-07 10:51:26 | 7768 | `df8a2bd2…` | 13/13 | owner-reported match, 2026-08-07 |
| F23 | `f23_metric_at_evaluation.py` | 2026-08-07 18:30:15 | 6840 | `4031c129…` | 6/6 | NOT RUN |
| F24 | `f24_leakfree_temp_matched.py` | 2026-08-07 18:30:22 | 7676 | `5adb883b…` | 14/14 | NOT RUN |
| Demo | `demo_quasar_method.py` | 2026-08-07 20:33:14 | 8356 | `d6526e29…` | 14/14 | owner-reported match, 2026-08-07 |

"12/12 August numbers" means each number recorded in the owner's notes for that run appears verbatim in the
2026-10-29 output ([august_numbers.py](recovery_2026-10-29/august_numbers.py) lists them). The full outputs are
`recovery_2026-10-29/*_container.txt`; a second run was identical, and a run with one digit changed was caught
([checks_container.txt](recovery_2026-10-29/checks_container.txt)). Environment: x86_64, Python 3.13.18, NumPy
2.5.3. A run under Python 3.11.18 and NumPy 2.4.6, the versions CI uses, was also identical. F20 also needs SciPy
(1.18.1 here, 1.17.1 under 3.11) for its H1 selftest. Both runs were on x86_64; no aarch64 run has been made.

## Results, as registered

Pasted from `recovery_2026-10-29/*_container.txt`. Entropies are of the attention distribution; "max" is uniform.

| Run | Regime | Blind | GEO | DOT | Attention vs blind | Verdict |
|---|---|---|---|---|---|---|
| F20 | direction | — | — | — | mean gap -0.0008, wins 0/5 | P0a FAIL, P0b PASS (entropy 0.705 of max): **VOID** |
| F20 | purity | — | — | — | mean gap +0.0021, wins 0/5 | (entropy 0.725 of max) |
| F21 | direction | 0.064336 | 0.046980 | 0.051280 | +27.0% | P0a, P0b pass; GEO wins. **Voided by F24** |
| F21 | purity | 0.038029 | 0.025716 | 0.028993 | +32.4% | P0a, P0b pass; GEO wins. **Voided by F24** |
| F22 | direction | 0.056234 | 0.053425 | 0.050373 | +10.4% | P0a, P0b pass; DOT wins. **Voided by F24** |
| F22 | purity | 0.032902 | 0.030943 | 0.029256 | +11.1% | P0a, P0b pass; DOT wins. **Voided by F24** |
| F24 | direction | 0.008369 | 0.004987 | 0.004999 | +40.4% | P0, P1 pass; P2 HELD, gap 0.23% |
| F24 | purity | 0.005251 | 0.002813 | 0.002832 | +46.4% | P0, P1 pass; P2 HELD, gap 0.66% |

F23 uses one scoring method and two evaluation metrics. Margins over blind: Bures² +10.4% and Euclidean² +10.8%
(direction), Bures² +11.1% and Euclidean² +11.0% (purity); P1 metric effect -0.4 and +0.0 points. Its "live" Bures²
losses (0.050373, 0.029256) equal F22's DOT arm exactly, so its only new information is the Euclidean² column.

The demo re-runs the F24 design with fewer samples (15 calibration and 25 test trajectories per seed, 4 seeds):
+38.0% and +47.2% over blind, gaps 0.23% and 0.62%. It is a walkthrough, not a separate experiment.

F21's and F22's GEO–DOT verdicts ran both arms at one BETA (6.0) on score functions with different ranges. F24's
calibration put the arms 3.7× apart in BETA (215.876 against 59.049 in the direction regime). F22's candidates
also included the scored transition at 3 of 5 query points.

## What this shows and does not show

- **Shows:** the recovered bytes print the numbers reported in August (container only). Under F24's settings,
  GEO and DOT differ by under 1% while attention beats the blind control by 40–46%.
- **Does not show:** that these bytes reproduce on the device; that temperature alone explains F21's and F22's
  opposite verdicts (point 6); anything new about F19.
- **F18 stands as written.** F20 measured F19's trained attention arms as flat (entropy 99.9–100.0% of the
  uniform maximum) and argues that the same defect explains F18's tie. That narrows what F18 tested. Whether to
  amend F18's entry is the owner's call, after device reproduction.

## Device reproduction (pending)

On the S25, in Termux, one command at a time. A fresh clone avoids the Android download traps and leaves any
existing checkout alone.

1. `git clone https://github.com/holland202/quasar ~/quasar_f19_f24`
2. `cd ~/quasar_f19_f24 && git checkout records/f19-f24-recovery-20261029` (skip once merged)
3. `cd ~/quasar_f19_f24/v2 && python3 recovery_2026-10-29/reproduce.py f21_transition_tokens f22_disjoint_quality f23_metric_at_evaluation f24_leakfree_temp_matched demo_quasar_method --label device`
4. `python3 recovery_2026-10-29/reproduce.py f20_purity_channel --label device` (needs SciPy; about a minute in the container)
5. `python3 recovery_2026-10-29/august_numbers.py recovery_2026-10-29 --label device`

Each run writes `recovery_2026-10-29/<script>_device.txt`, checks the script's bytes against `SHA256SUMS`, and
compares the output with the container reference, ignoring only wall-clock lines. If the device still holds the
August copies, hashing them shows whether the August device runs used these bytes:
`find ~ \( -name 'f2[0-4]_*.py' -o -name 'demo_quasar_method.py' \) -exec sha256sum {} +`

## Doors (unrun)

- F23 repeated with F24's leak-free candidates.
- F24 with one change only: temperature matching, everything else as F22. This would isolate temperature.
- The doors the files already name: F20's P4 (multi-head) and F22's deferred queue (CP projection of candidate
  maps, multi-scale windows, a LUCID-coarse then attention-fine pipeline).

## Credit and provenance

- **Credit.** Graeme Randle gave the first external critique and drove the QUASAR geometry follow-ups. F20's
  registration, F22 and F23 say they respond to his critiques; the owner's notes say F19's task came from his
  diagnosis too. He approved that credit line on 2026-10-03 (relayed by Chad Holland). It approves the credit, not
  the work or the results. None of F19–F24 is an independent replication.
- **AI participation.** Claude (Anthropic) wrote F20–F24 and the demo in sessions Chad Holland directed in August
  2026, according to the owner's session notes; F20's registration names "holland202 + Claude (Anthropic)". The
  files do not record the model version. Claude (Opus 5.5) recovered the files, re-ran them, wrote the
  three check scripts and wrote this record on 2026-10-29.
- **Human validation.** Chad Holland reported the S25 runs of F21, F22 and the demo in August (no log recovered).
  This record: direction only, not reviewed line by line.
- **Human editing/curation:** none.
- **Responsibility:** Chad Holland.
