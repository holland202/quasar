# Phase 2C Governance Toolkit — Master Guide
**Complete Implementation of Sovereign Logic Core on Galaxy S25 Ultra**

---

## 🎯 What This Is

A **complete, production-ready governance layer** for Sovereign Logic Core. Sits between GGUF inference and SIC state modification, enforcing topological safety via two audits:

1. **PreInferenceGate** — Predictive risk evaluation (rejects obvious bad inferences early)
2. **Commit Gate** — Five topological/thermodynamic audits (prevents corruption of SIC manifold)

**Result:** Fully governed, audited inference pipeline with live monitoring on S25 Ultra.

---

## 📚 Documentation Structure

This toolkit includes comprehensive documentation at multiple levels:

| Level | File | Time | Purpose |
|-------|------|------|---------|
| **START HERE** | **QUICK_START.md** | 3 min | One-page commands + quick fixes |
| **OVERVIEW** | **TOOLKIT_SUMMARY.md** | 5 min | Package contents + quick reference |
| **SETUP** | **setup_phase2c.sh** | 5 min | Automated installation (execute once) |
| **COMPREHENSIVE** | **PHASE_2C_TOOLKIT_README.md** | 20 min | Everything: setup, running, tuning, troubleshooting |
| **MONITORING** | **MONITOR_GOVERNANCE_README.md** | 15 min | Dashboard interpretation + advanced monitoring |
| **INTEGRATION** | **MONITOR_INTEGRATION.md** | 10 min | Workflow examples + multi-process setup |
| **DEEP DIVE** | **IMPLEMENTATION_NOTES.md** | 30 min | Design decisions, math, tuning rationale |
| **MASTER** | **README.md** (this file) | 10 min | Architecture overview + navigation guide |

---

## 🚀 Quick Start (Choose Your Path)

### **Path 1: Just Run It** (5 minutes)
For people who want results immediately:

```bash
bash setup_phase2c.sh
./run_phase2c.sh 300 0.85
# Watch monitor + metrics appear
```

**Then:** QUICK_START.md for command reference.

---

### **Path 2: Understand First** (30 minutes)
For people who want to understand what they're running:

1. Read this file (Architecture section)
2. Read PHASE_2C_TOOLKIT_README.md (Concepts section)
3. Read QUICK_START.md (Commands)
4. Then run setup + calibration

**Then:** Run with monitor, adjust thresholds based on dashboard.

---

### **Path 3: Deep Dive** (2+ hours)
For people building on top of Phase 2C or deploying in production:

1. Read this file (entire Architecture section)
2. Read IMPLEMENTATION_NOTES.md (design rationale + math)
3. Read PHASE_2C_TOOLKIT_README.md (all sections)
4. Read source code comments in Python files
5. Run calibration with monitoring
6. Experiment with threshold tuning

**Then:** Production deployment with custom parameters.

---

## 🏗️ Architecture

### **The One-Way Crystallization Membrane**

Phase 2C enforces a hard rule: **LLM output never touches SIC directly.**

Instead, all writes go through a chain of audits:

```
User Prompt
    ↓
[STEP 4] PreInferenceGate
    • Evaluates: prompt entropy, SIC distance, cognitive pressure, history variance
    • Decides: should we even call GGUF? (early rejection saves inference time)
    • Output: pass/reject decision + risk score
    ├─ REJECT → record deferral, skip GGUF
    ↓ PASS
[STEP 5] GGUF Inference
    • Generates output with logprobs
    • Extracts logit_variance (Fisher information)
    ├─ FAIL → record failure
    ↓ SUCCESS
[STEP 6a] Draft Delta
    • Estimate what SIC state change would be
    • Compute topological strain (dry-run, no modification yet)
    ├─ FAIL → reject
    ↓ SUCCESS
[STEP 6b] Commit Gate Audit (5 checks)
    1. Fisher Sharpness: logit_variance ≥ 0.85?
    2. Spectral Norm: ‖UV^T‖₂ ≤ 2.0?
    3. Rank Preservation: rank unchanged?
    4. Geodesic Distance: Stiefel distance ≤ 0.15?
    5. Thermal Coupling: effective temperature ≥ threshold?
    • Decision: ALL checks must pass (AND logic)
    ├─ ANY FAIL → reject
    ↓ ALL PASS
[STEP 7] SIC.update() ← ONE GATE
    • Irreversible write to manifold
    • Record in CrystallizationMemory
    ↓
[STEP 8+] Continue cycle (UME, SMA, etc.)
```

**Guarantee:** SIC is modified ONLY if ALL gates pass.

---

### **Key Components**

#### **1. SIC Enhancement** (`core/sic.py`)
Added two methods:
- `get_state_for_gate()` — Lightweight snapshot for gates to read (U, V, rank, spectral_norm_U)
- `compute_scar_cost()` — Estimate topological strain before actual update (dry-run)

**Why:** Gates need manifold state without deep coupling or modifying it.

#### **2. CrystallizationMemory** (`crystallization_memory.py`)
Tracks history of crystallization events:
- `delta_history` — Logit variance from each scar
- `pressure_history` — Topological/thermal strain per event
- `rejection_history` — Which events were rejected

**Why:** PreInferenceGate uses history (variance, pressure) to predict future risk.

#### **3. PreInferenceGate** (`pre_inference_gate.py`)
4-factor predictive risk evaluation BEFORE GGUF:
1. **Prompt Entropy** (0.25 weight) — Is input OOD/adversarial?
2. **Identity Distance** (0.30 weight) — Is SIC state drifting?
3. **Cognitive Pressure** (0.25 weight) — Is system tired/over-loaded?
4. **Variance Term** (0.20 weight) — Has history been unstable?

**Output:** Soft probability (sigmoid boundary, not hard threshold) + risk breakdown.

**Why:** Prevents expensive GGUF calls when manifest risk is high.

#### **4. TransferController** (`transfer_controller.py`)
One-way membrane + Commit Gate audit:
- `draft_delta()` — Estimate proposed SIC change (dry-run)
- `commit_gate_audit()` — 5-check topological audit
- `crystallize()` — Actual SIC.update() call (only if audit passes)

**Why:** All SIC modifications go through here. Clean audit trail.

#### **5. Monitor Dashboard** (`monitor_governance.py`)
Real-time Termux-friendly display of:
- Hardware (temperature, thermal state)
- PreGate metrics (pass rate, risk score)
- CommitGate metrics (Fisher, geodesic, acceptance rate)
- Crystallization stats (success rate, scars)
- Memory state (variance, pressure)

**Why:** Instant visual feedback on governance health. Enables quick tuning.

#### **6. Test/Calibration Scripts**
- `smoke_test_2c.py` — Validate setup (no GGUF needed, uses mock engine)
- `calibrate_governance.py` — Main loop with tuning capability

**Why:** Smoke test = confidence gates work. Calibration = find optimal thresholds.

---

### **Engine Integration** (`core/engine.py`)

The 10-step cycle now includes governance:

```
OLD (Phase 1)              NEW (Phase 1 + Phase 2)
1. Thermal (Veritas)       1. Thermal (unchanged)
2. VEST auth               2. VEST auth (unchanged)
3. SMA optimization        3. SMA optimization (unchanged)
[no governance]            4. PreInferenceGate [NEW]
[direct inference]         5. GGUF inference [NEW]
[direct SIC write]         6. Commit Gate audit [NEW]
7. Scar formation          7. Scar formation (after gate)
8. UME dynamics            8. UME dynamics (unchanged)
9. SMA flow                9. SMA flow (unchanged)
10. Telemetry              10. Telemetry (unchanged)
```

Steps 1–3, 7–10 are **unchanged**. Steps 4–6 are **new governance**.

---

## 🔧 Governance Parameters

### **PreInferenceGate**

```
--pre-gate-threshold FLOAT      (default: 0.65)
  Risk midpoint for pass/fail decision
  Lower = more permissive (more inferences pass PreGate)
  Target: 60–80% pass rate
  Tuning: Adjust ±0.05 per run

--pre-gate-weights ...          (not exposed in CLI, edit source)
  [entropy, distance, pressure, variance] weights
  Default: [0.25, 0.30, 0.25, 0.20]
  Higher weight = more influence on risk score
```

### **TransferController (Commit Gate)**

```
--fisher-threshold FLOAT        (default: 0.85)
  Min GGUF logit confidence required
  Lower = accept lower-confidence outputs
  Target: CommitGate pass rate 70–85%
  Tuning: Adjust ±0.05–0.10 per run

--geodesic-distance-max FLOAT   (default: 0.15)
  Max allowed Stiefel distance of proposed delta
  Higher = allow larger SIC changes
  Target: <0.10 in actual metrics
  Tuning: Adjust ±0.05 per run

--thermal-multiplier FLOAT      (default: 1.0)
  Scale effective temperature (>1 = more permissive when tired)
  Higher = allow updates even when system exhausted
  Target: Keep pressure <0.7
  Tuning: Lower to 0.8 if pressure stuck high
```

### **Runtime**

```
--cycles INT                    (default: 300)
  Number of inference cycles

--mode {mock,engine}            (default: engine)
  mock = use MockGGUFEngine (no real inference)
  engine = use real GGUF + governance loop

--log-file PATH                 (required)
  JSON lines file where metrics are written
  Default: /tmp/governance_metrics.jsonl

--gpu-layers INT                (default: 33 on S25 Ultra)
  Number of GPU layers (higher = faster but hotter)
  Reduce to 16 if thermal throttling occurs
```

---

## 📊 Understanding the Dashboard

The monitor shows real-time metrics. Here's what each means:

### **HARDWARE**
```
Temperature:    42.5°C  State: OK
```
- **Green** (<40°C): Normal
- **Yellow** (40–45°C): Warm, acceptable
- **Red** (>45°C): Hot, monitor closely; abort >48°C
- **State**: "OK" or "THROTTLE"

### **PRE-INFERENCE GATE**
```
Pass Rate:      75.3%  (847 evals)
Risk Score:     0.542  (0.0–1.0)
```
- **Pass Rate**: % of inferences that passed PreGate (target: 60–80%)
  - Too low (<40%): Lower `pre_gate_threshold`
  - Too high (>95%): Raise `pre_gate_threshold`
- **Risk Score**: Current manifold risk (0.0 = safe, 1.0 = dangerous)

### **COMMIT GATE**
```
Pass Rate:      82.1%  (847 audits)
Fisher Sharp.:  0.881  (≥0.85 required)
Geodesic Dist.: 0.048  (≤0.15 target)
```
- **Pass Rate**: % of PreGate passers that clear Commit Gate (target: 70–85%)
  - Too low: Lower `fisher_threshold` or raise `geodesic_distance_max`
- **Fisher Sharpness**: GGUF output confidence (0.0–1.0)
  - <0.85: Check GGUF quality or lower threshold
- **Geodesic Distance**: Proposed SIC change magnitude
  - <0.10: Very safe
  - 0.10–0.15: Acceptable
  - >0.15: Risky, raise threshold or lower geodesic_max

### **CRYSTALLIZATION**
```
Success Rate:   67.2%  (scar admitted)
Total Scars:    423
```
- **Success Rate**: % of PreGate passers that resulted in SIC.update() (target: 50–70%)
- **Total Scars**: Cumulative SIC modifications

### **CRYSTALLIZATION MEMORY**
```
Variance Term:  0.187  (history stability)
Pressure:       0.342  (thermal burden)
```
- **Variance**: 0.0 = stable, 1.0 = unstable
  - High variance → PreGate will reject more (conservative)
- **Pressure**: 0.0 = fresh, 1.0 = exhausted
  - High pressure → few updates (system tired)
  - Decays exponentially, resets with low crystallization

---

## 🎮 Running Phase 2C

### **Option 1: Two Tabs** (Recommended)

```bash
# Tab 1: Monitor (foreground, updates every 1–2 seconds)
python3 monitor_governance.py --metrics-file /tmp/governance_metrics.jsonl

# Tab 2: Calibration (foreground, runs to completion)
python3 calibrate_governance.py --mode engine --cycles 300 \
    --fisher-threshold 0.85 --log-file /tmp/governance_metrics.jsonl
```

**Pros:** Easy to switch between monitor and logs  
**Cons:** Requires tab management

---

### **Option 2: Background + Monitor**

```bash
# Start test in background
python3 calibrate_governance.py --mode engine --cycles 300 \
    --fisher-threshold 0.85 --log-file /tmp/governance_metrics.jsonl &

# Monitor in foreground
python3 monitor_governance.py --metrics-file /tmp/governance_metrics.jsonl
```

**Pros:** Single terminal command  
**Cons:** Mixed output, Ctrl+C kills monitor

---

### **Option 3: Launcher Script** (Fully automated)

```bash
./run_phase2c.sh 300 0.85
# Starts monitor + calibration, cleans up on exit
```

**Pros:** One command, clean shutdown  
**Cons:** Less flexibility for live tuning

---

## 🧪 Tuning Workflow

### **Step 1: Baseline (Default Parameters)**
```bash
python3 calibrate_governance.py --mode engine --cycles 300 \
    --fisher-threshold 0.85 --pre-gate-threshold 0.65 \
    --log-file /tmp/governance_metrics.jsonl
```

Monitor and note metrics:
- PreGate pass rate: X%
- CommitGate pass rate: Y%
- Crystallization: Z%

### **Step 2: Identify Issues**

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| PreGate <40% | Too strict | Lower `pre_gate_threshold` to 0.60 |
| CommitGate <50% | Fisher too strict | Lower `fisher_threshold` to 0.75 |
| Crystallization <30% | Both gates too strict | Lower both thresholds |
| Pressure >0.8 | System exhausted | Lower `thermal_multiplier` to 0.8 |

### **Step 3: Tune One Parameter**
```bash
# Example: CommitGate was 45%, too strict
# Fix: Lower fisher_threshold
python3 calibrate_governance.py --mode engine --cycles 300 \
    --fisher-threshold 0.75 \
    --log-file /tmp/governance_metrics.jsonl
```

### **Step 4: Verify**
Watch dashboard. Did CommitGate pass rate improve?
- Yes → Move to Step 5
- No → Adjust more, re-run Step 3

### **Step 5: Iterate**
If other metrics are still out of range, adjust another threshold and repeat Step 3–4.

### **Step 6: Validate**
Once all metrics in healthy ranges, run a longer final validation:
```bash
python3 calibrate_governance.py --mode engine --cycles 500 \
    --fisher-threshold 0.75 --pre-gate-threshold 0.65 \
    --log-file /tmp/governance_metrics.jsonl
```

### **Step 7: Document**
Save final thresholds:
```bash
cp /tmp/governance_metrics.jsonl ~/results/final_phase2c_metrics.jsonl
echo "Final params: fisher=0.75, pre_gate=0.65" > ~/results/phase2c_params.txt
```

---

## 📈 Expected Performance

### **Hardware: Galaxy S25 Ultra (Snapdragon 8 Elite)**

| Metric | Typical | Range |
|--------|---------|-------|
| **Temperature (idle)** | 35°C | 33–38°C |
| **Temperature (GGUF)** | 43°C | 40–45°C |
| **Temperature (peak, sustained)** | 44°C | 42–46°C |
| **Temperature (throttle)** | 48°C | (auto-abort) |
| **Memory usage** | 450 MB | 400–550 MB |
| **CPU utilization** | 70% | 60–80% |
| **Duration (300 cycles)** | 7 min | 5–10 min |

### **Governance Metrics (300 cycles)**

| Metric | Typical | Range |
|--------|---------|-------|
| **PreGate pass rate** | 72% | 60–85% (depends on SIC state) |
| **CommitGate rate** | 78% | 70–85% (depends on thresholds) |
| **Crystallization rate** | 55% | 40–70% (depends on thresholds) |
| **Scars admitted** | 200 | 150–250 |
| **Total rejections** | 120 | 80–180 |
| **Mean Fisher sharpness** | 0.87 | 0.75–0.95 |
| **Mean geodesic distance** | 0.08 | 0.05–0.15 |

---

## 🐛 Troubleshooting Matrix

| Problem | Cause | Solution |
|---------|-------|----------|
| **Monitor shows "No metrics file"** | File not written yet (normal) | Wait 2–3 sec; check test is running |
| **Monitor shows "Temperature: unavailable"** | Thermal sysfs not exposed | OK, monitor works without it; try `--thermal-zone 1` |
| **CommitGate pass rate <50%** | Fisher threshold too high | Lower to `0.75` or `0.70` |
| **PreGate pass rate <40%** | PreGate threshold too high | Lower to `0.60` |
| **Crystallization <30%** | Both gates too strict | Lower both thresholds |
| **Temperature >46°C** | Device hot, may throttle | Pause, cool; next run: reduce `--gpu-layers 16` |
| **Ctrl+C doesn't stop test** | Signal handling in loop | Press Ctrl+C again or `pkill calibrate_governance.py` |
| **"GGUF engine not set"** | Using `--mode engine` without GGUF | Use `--mode mock` or ensure GGUF is installed |
| **Pressure stuck >0.8** | System exhausted | Lower `--thermal-multiplier 0.8` |
| **Fisher stuck at 0.80 exactly** | Logprobs extraction not working | See MONITOR_GOVERNANCE_README.md § "One honest caveat" |

See PHASE_2C_TOOLKIT_README.md § Common Issues & Fixes for full details.

---

## 📖 Deep Dives

For understanding design decisions and math:

### **Stiefel Manifold & Geodesic Distance**
See: IMPLEMENTATION_NOTES.md § 3 (TransferController § Geodesic Distance Approximation)

### **Fisher Information & Sharpness**
See: IMPLEMENTATION_NOTES.md § 3 (TransferController § Five Commit Gate Checks)

### **Thermal Governance & Exponential Decay**
See: IMPLEMENTATION_NOTES.md § 3 (TransferController § Thermal Coupling Deep Dive)

### **Prompt Entropy & Rank Deficiency**
See: IMPLEMENTATION_NOTES.md § 2 (PreInferenceGate § Identity Distance Approximation)

### **Complete Architecture Rationale**
See: IMPLEMENTATION_NOTES.md § 1–4 (all sections)

---

## 🎯 Common Workflows

### **"I just want to validate the setup"**
```bash
python3 smoke_test_2c.py --cycles 20 --log-file /tmp/governance_metrics.jsonl
# <30 seconds, shows governance layer works
```

### **"I want to find the right thresholds for my GGUF"**
```bash
# Run 1: Default
./run_phase2c.sh 100 0.85

# Observe CommitGate rate
# If <50%: Run 2 with lower Fisher
./run_phase2c.sh 100 0.75

# If still <50%: Run 3 with even lower
./run_phase2c.sh 100 0.70

# Once acceptable: final validation
./run_phase2c.sh 500 0.70
```

### **"I want to understand the governance math"**
1. Read IMPLEMENTATION_NOTES.md
2. Read source code comments in `.py` files
3. Run with monitor, observe behavior matching theory

### **"I want to deploy to production"**
1. Run Phase 2B calibration (500+ cycles)
2. Document final threshold values
3. Document thermal conditions (ambient temp, USB power, etc.)
4. Use these exact thresholds for production
5. Monitor first 100 production cycles to confirm behavior matches calibration

---

## 🔗 Complete File Reference

### **Documentation**
| File | Purpose | Read Time |
|------|---------|-----------|
| README.md | This file — architecture overview | 10 min |
| QUICK_START.md | One-page commands + fixes | 3 min |
| TOOLKIT_SUMMARY.md | Package overview | 5 min |
| PHASE_2C_TOOLKIT_README.md | Comprehensive guide (everything) | 20 min |
| MONITOR_GOVERNANCE_README.md | Dashboard reference + tuning | 15 min |
| MONITOR_INTEGRATION.md | Multi-process workflow examples | 10 min |
| IMPLEMENTATION_NOTES.md | Design decisions + math deep dive | 30 min |

### **Scripts**
| File | Purpose | Usage |
|------|---------|-------|
| setup_phase2c.sh | Installation (one-time) | `bash setup_phase2c.sh` |
| run_phase2c.sh | Automated launcher | `./run_phase2c.sh [cycles] [fisher]` |
| smoke_test_2c.py | Quick validation | `python3 smoke_test_2c.py --cycles 20` |
| calibrate_governance.py | Main calibration loop | `python3 calibrate_governance.py --mode engine ...` |
| monitor_governance.py | Live metrics dashboard | `python3 monitor_governance.py` |

### **Core Modules**
| File | Purpose |
|------|---------|
| core/sic.py | Scarred Identity Chronicle (+ get_state_for_gate, compute_scar_cost) |
| core/engine.py | 10-step orchestrator (Steps 4–6 now have governance) |
| core/crystallization_memory.py | History buffer for governance decisions |
| core/pre_inference_gate.py | 4-factor predictive risk evaluator |
| core/transfer_controller.py | One-way membrane + Commit Gate audit |

---

## ✅ Getting Started

### **For Complete Beginners**
1. Read QUICK_START.md (3 min)
2. `bash setup_phase2c.sh`
3. `./run_phase2c.sh 300 0.85`
4. Watch monitor, save results

### **For Software Engineers**
1. Read this README (Architecture section, 10 min)
2. Read IMPLEMENTATION_NOTES.md (30 min)
3. Review `.py` source code comments
4. Run with monitor, experiment with thresholds

### **For Researchers**
1. Read IMPLEMENTATION_NOTES.md § 1–8 (full deep dive)
2. Read PHASE_2C_TOOLKIT_README.md § Advanced Topics
3. Run calibration with monitoring
4. Analyze metrics in post-run JSON

### **For Production Deployment**
1. Read this entire README
2. Run Phase 2B calibration (500+ cycles)
3. Document system behavior (temperature, thresholds, metrics)
4. Use documented parameters in production
5. Monitor production runs against calibration baseline

---

## 📞 Support

| Question | Answer | File |
|----------|--------|------|
| "How do I run this?" | QUICK_START.md (commands) | QUICK_START.md |
| "What does metric X mean?" | Dashboard interpretation | MONITOR_GOVERNANCE_README.md |
| "Why is Y happening?" | Troubleshooting matrix | PHASE_2C_TOOLKIT_README.md |
| "How does this work?" | Architecture overview | README.md (this file) |
| "Why was Z designed this way?" | Design rationale + math | IMPLEMENTATION_NOTES.md |

---

## 🚀 Next Steps

1. **Read:** QUICK_START.md (3 min)
2. **Setup:** `bash setup_phase2c.sh` (5 min)
3. **Test:** `python3 smoke_test_2c.py --cycles 20` (<1 min)
4. **Run:** `./run_phase2c.sh 300 0.85` (7 min with monitor)
5. **Observe:** Watch dashboard, note metrics
6. **Tune:** Adjust thresholds, re-run (5 min per iteration)
7. **Validate:** Final 500-cycle run with tuned params (10 min)
8. **Deploy:** Use documented thresholds in production

---

## 📊 Key Takeaways

✓ **One-way membrane:** LLM output never touches SIC directly (gates audit all writes)  
✓ **Predictive governance:** PreGate prevents wasted inference on risky inputs  
✓ **Topological safety:** Commit Gate enforces 5 mathematical invariants  
✓ **Memory-aware:** History shapes future decisions (variance, pressure)  
✓ **Live monitoring:** Real-time dashboard shows governance health  
✓ **Fully documented:** 4000+ lines of guides + deep design rationale  
✓ **Production-ready:** Complete tuning workflow + validation process  

---

**Phase 2C is ready. Start with QUICK_START.md, run setup_phase2c.sh, and go.** 🎯

---

## Table of Contents

- 🎯 [What This Is](#-what-this-is)
- 📚 [Documentation Structure](#-documentation-structure)
- 🚀 [Quick Start (Choose Your Path)](#-quick-start-choose-your-path)
- 🏗️ [Architecture](#-architecture)
- 🔧 [Governance Parameters](#-governance-parameters)
- 📊 [Understanding the Dashboard](#-understanding-the-dashboard)
- 🎮 [Running Phase 2C](#-running-phase-2c)
- 🧪 [Tuning Workflow](#-tuning-workflow)
- 📈 [Expected Performance](#-expected-performance)
- 🐛 [Troubleshooting Matrix](#-troubleshooting-matrix)
- 📖 [Deep Dives](#-deep-dives)
- 🎯 [Common Workflows](#-common-workflows)
- 🔗 [Complete File Reference](#-complete-file-reference)
- ✅ [Getting Started](#-getting-started)
- 📞 [Support](#-support)
- 🚀 [Next Steps](#-next-steps)
- 📊 [Key Takeaways](#-key-takeaways)

---

**All documentation files included. Start anywhere, reference as needed. Good luck.** 🎯
