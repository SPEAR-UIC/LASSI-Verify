
# LASSI-Verify Pipeline Overview

**Version:** v4 (develop-runtime branch, March 2026)
**Target GPU:** NVIDIA RTX 5000 Ada (sm_89); also supports AMD ROCm via HIP
**Primary benchmark collection:** HeCBench (CUDA C++ kernels)

---

## What LASSI-Verify Does

LASSI-Verify is an LLM-assisted empirical verification framework for parallel scientific GPU code transformations. Its goal is to determine whether an LLM-generated refactoring of a CUDA kernel (optimized for energy efficiency or runtime) is functionally equivalent to the original source and genuinely better-performing.

It wraps around a multi-iteration LLM self-improvement loop ("LASSI") with three formal-methods-grounded verification tiers that together provide correctness evidence without requiring formal proofs or exhaustive testing.

**Research context:** Formal grounding is drawn from:
- Tier 1 → Owicki-Gries auxiliary variable assertions
- Tier 2 → Necula forward simulation / observational equivalence
- Tier 3 → Benton's Relational Hoare Logic; metamorphic testing (Chen et al. 1998, 2019)

---

## Key Data Structures

### `code_blocks` (list of strings)
Tracks all code versions produced during a pipeline run:
- `[0]` — **SOURCE_orig**: the original unmodified source file (no instrumentation)
- `[1]` — **SOURCE_verify**: Tier 1 + Tier 2 instrumented version of the source (observation points inserted by LLM)
- `[2]` — **Standalone Base LLM**: zero-shot vanilla LLM output (generated from SOURCE_orig, no instrumentation)
- `[3+]` — **LASSI v{n}**: successive LASSI-generated refactorings (generated from SOURCE_verify)

### History arrays (parallel to `code_blocks`)
- `runtime_history[i]` — wall-clock execution time (seconds) for code version i
- `energy_history[i]` — total GPU energy consumed (joules)
- `power_history[i]` — average GPU power (watts)
- `valid_code_history[i]` — "VALID" / "INVALID" / "SKIP" (judge + tier override verdict)

### `tracking_best_version`
Index into `code_blocks` pointing to the current best-performing valid code.
- Initialized to 1 (SOURCE_verify) at the start of the LASSI loop
- Updated to `len(code_blocks)-1` each time a new code wins on the optimization metric

### `verification_log` (dict)
Accumulated Tier 2 and Tier 3 results, judge discrepancies, and config fields.
Saved as a JSON sidecar file (`verification_log_<name>_Trial<N>.json`) at pipeline end.

---

## Pipeline Flow (Stage by Stage)

```
Source file (.cu)
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 0a: Test SOURCE_orig                                      │
│  Compile + execute original source; collect baseline metrics.   │
│  Captures: duration, energy, stdout → stored as code_blocks[0] │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLM CODE ANALYSIS (optional, run_code_analysis flag)            │
│  LLM reads source; produces refactoring opportunity summary.    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1 INSTRUMENTATION                                          │
│  LLM reads source + Invariant_Assertions_Catalog.md;            │
│  inserts runtime assertion blocks into source.                  │
│  Result: SOURCE_verify (code_blocks[1])                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2 INSTRUMENTATION                                          │
│  LLM reads SOURCE_verify + passthrough catalog;                 │
│  inserts observation point (passthrough) blocks at key          │
│  computation checkpoints (preserving Tier 1 blocks).            │
│  Written to disk as *_verify.cu                                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 0b: Test SOURCE_verify                                    │
│  Compile + execute *_verify.cu; collect instrumented baseline.  │
│  Captures: stdout with PASSTHROUGH blocks → tier2_spec_output   │
│  (This becomes the Tier 2 reference for all generated codes.)   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: METAMORPHIC CONTRACT GENERATION + SOURCE FILTER         │
│  (if run_tier3=True)                                            │
│  1. LLM reads SOURCE_verify + Metamorphic_Relations_Catalog.md; │
│     generates a metamorphic contract (JSON) specifying which    │
│     relations apply, with input variables and observation pts.  │
│  2. Source filter: for each relation in the contract, generate  │
│     a variant of SOURCE_verify, compile, execute, and check.    │
│     Relations that FAIL for source are marked inactive.         │
│     Result: tier3_active_relations list.                        │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
         ┌───────────────────────────────────────────────┐
         │         MAIN WHILE LOOP (iterative)           │
         │                                               │
         │  ┌─────────────────────────────────────────┐  │
         │  │ STAGE 1: Zero-Shot (Vanilla LLM)        │  │
         │  │  Generates one code from SOURCE_orig     │  │
         │  │  (no instrumentation, one attempt only). │  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ LASSI GENERATION                        │  │
         │  │  LLM generates from SOURCE_verify using │  │
         │  │  metrics feedback + self-improvement.   │  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ COMPILE                                 │  │
         │  │  nvcc (or hipcc) with -O3 -arch=sm_89   │  │
         │  │  Compile error → self-correction LLM.   │  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ EXECUTE + METRICS                       │  │
         │  │  Wall-clock time (+ GPU power if EE).   │  │
         │  │  Execute error → self-correction LLM.   │  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ TIER 1 CHECK (LASSI codes only)         │  │
         │  │  Parse stdout for ASSERTIONS PASS/FAIL. │  │
         │  │  FAIL → self-correction (up to 3x/code).│  │
         │  │  Exhausted → INVALID + continue         │  │
         │  │  (skips Tier 2, Tier 3, judge entirely).│  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ TIER 2 CHECK (LASSI codes only)         │  │
         │  │  Parse PASSTHROUGH blocks from stdout.  │  │
         │  │  Compare labeled blocks to spec         │  │
         │  │  (tier2_spec_output from SOURCE_verify).│  │
         │  │  FAIL → self-correction (up to 5x/code).│  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ TIER 3 CHECK (if run_tier3=True)        │  │
         │  │  For each active relation (fail-fast):  │  │
         │  │   - Generate MT variant of this code    │  │
         │  │   - Compile + execute variant           │  │
         │  │   - Check relation against this code's  │  │
         │  │     own Tier 2 passthrough output       │  │
         │  │  FAIL → self-correction (up to 3x/code).│  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ LLM JUDGE                               │  │
         │  │  Judge reads both codes + stdouts;       │  │
         │  │  returns VALID/INVALID/SKIP verdict.    │  │
         │  │  Three-tier overrides applied (see §).  │  │
         │  └──────────────┬──────────────────────────┘  │
         │                 │                              │
         │  ┌──────────────▼──────────────────────────┐  │
         │  │ METRICS FEEDBACK + WINNER SELECTION     │  │
         │  │  VALID + better than best → new winner. │  │
         │  │  Stagnation check → early exit.         │  │
         │  └─────────────────────────────────────────┘  │
         └───────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ POST-LOOP: BEST CLEAN RUN                                       │
│  Strip all LASSI-VERIFY blocks from best code (regex);          │
│  recompile; execute uninstrumented for fair performance          │
│  measurement. Captures lassiee_clean_{status, runtime_s,        │
│  runtime_diff, energy_J, energy_diff}.                          │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ FINAL LLM CODE COMPARISON                                       │
│  LLM compares SOURCE_orig vs best clean code with metrics.      │
│  Returns [OPTIMIZATION LIST] tag for CSV capture.               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
             CSV row + JSON sidecar + metadata .txt

```

---

## Tier 1: Invariant Assertion Checking

**Formal grounding:** Owicki-Gries auxiliary variable assertions.

**What it does:** The LLM inserts runtime assertion checks directly into the source C++ code as labeled comment blocks. These blocks execute inline during each run and print `PASS` or `FAIL` to stdout for each assertion.

**Block format (in C++ source):**
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
{
    // [assertion_label]: description
    bool _assertion = (condition);
    printf("LASSI-VERIFY ASSERTIONS [assertion_label]: %s\n", _assertion ? "PASS" : "FAIL");
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

**Catalog:** `Invariant_Assertions_Catalog.md` — provides a 3-layer framework (universal, domain, problem-specific) with examples for GPU kernels. All examples use the printf PASS/FAIL pattern (not `assert()`), matching what `check_tier1_assertions()` parses from stdout.

**When it runs (instrumentation):** Applied once at pipeline start to produce SOURCE_verify. All LASSI-generated codes inherit the assertion blocks because they are generated from SOURCE_verify.

**When it runs (per-iteration checking — Option B):** Inside the main while loop, after each generated code executes, the pipeline parses its stdout for all `LASSI-VERIFY ASSERTIONS [label]: PASS/FAIL` lines via `check_tier1_assertions()`. This gives full visibility into every violated assertion before deciding to retry.

**Self-correction:** On FAIL, the failing assertion labels and summary are fed back to the LLM with the instruction to fix the code without removing any LASSI-VERIFY blocks. Up to 3 retries per code (per-code counter resets on fresh generation). When the per-code budget exhausts, the code is immediately marked INVALID and the pipeline `continue`s — skipping Tier 2, Tier 3, and the judge entirely. This fail-fast behavior avoids unnecessary overhead when Tier 1 has already determined the code is incorrect.

---

## Tier 2: Empirical Equivalence Checking

**Formal grounding:** Necula forward simulation / observational equivalence.

**What it does:** The LLM inserts labeled observation point blocks at key computation checkpoints. Each block captures aggregate statistics (min, max, mean) and a sample of values from an output buffer immediately after a kernel execution. When the generated code runs, its passthrough output is compared block-by-block against the SOURCE_verify reference output (`tier2_spec_output`).

**Block format (in C++ source, printed to stdout):**
```
=== BEGIN LASSI-VERIFY PASSTHROUGH [label] (do not edit or delete this block) >>>
Buffer: <buffer_name>
Elements: <N>
Statistics: min=%.17g, max=%.17g, mean=%.17g
Sample_first10: %.17g %.17g ...
=== END LASSI-VERIFY PASSTHROUGH <<<
```

**Comparison:** Deterministic first (`|a-b| ≤ atol + rtol×max(|a|,|b|)` with `atol = |spec_max|×rtol`). Falls back to LLM parse if deterministic parsing fails. Default `tier2_tolerance = 1e-6`.

**Self-correction:** On FAIL, the raw stdout diff (SOURCE_verify reference + generated output) is fed back to the LLM as self-correction context. Up to 5 retries per code, 15 total across the run.

**Stacking:** Tier 1 instrumentation is applied first; Tier 2 is applied on top. The LLM prompt for Tier 2 explicitly preserves existing ASSERTIONS blocks.

---

## Tier 3: Metamorphic Testing

**Formal grounding:** Benton's Relational Hoare Logic; metamorphic testing (Chen 1998, Murphy et al.).

**Key architectural principle:** Tier 3 uses **separate variant binaries** that are never part of the timed performance run. This ensures Tier 3 never contaminates runtime/energy measurements.

**Self-referential testing:** Each generated code is tested against **itself** under input transformation — not against SOURCE_verify. The reference ("Y") is the generated code's own Tier 2 passthrough output. The variant ("Z") runs a transformed input. This requires only one additional compile+run per relation per code.

### Stage 0 Source Filter
Run once at pipeline start. Determines which relations from the contract are applicable:
- Relations that PASS for SOURCE_verify → **active** (applied to all generated codes)
- Relations that FAIL for SOURCE_verify → **inactive** (filtered out — inapplicable to this problem)
- Relations that SKIP → inactive

### Main Loop Check
For each generated code, for each active relation (fail-fast order — cheapest first):
1. LLM generates a metamorphic variant of the generated code
2. Compile variant to a separate binary (`*_mt_{relation_name}.cu`)
3. Execute variant (subprocess, no power monitoring)
4. `check_metamorphic_relation()` compares variant stdout to the code's passthrough output

On FAIL: self-correction (up to 3 retries per code, sharing global counter).

### Implemented Relation Types
| Type | Check | Block Fields |
|------|-------|--------------|
| **SCALING** | Z_stat ≈ factor × Y_stat; Z_sample[i] ≈ factor × Y_sample[i] | Factor, Statistics, Sample_first10 |
| **SYMMETRY** | Same as SCALING with factor=−1 (min/max swap) | Factor, Statistics, Sample_first10 |
| **PERMUTATIVE** | Step A: Z_stat ≈ Y_stat (invariance, factor=1.0); Step B: Z[idx1] ≈ Y[idx2] and Z[idx2] ≈ Y[idx1] (equivariance, fires when indices in first10/last10 range) | Swap_idx1, Swap_idx2, Sample_at_idx1, Sample_at_idx2, Statistics, Sample_first10 |
| **CONSERVATION** | \|sum_after − sum_before\| / \|sum_before\| < rtol | Sum (before + after blocks) |
| **MONOTONICITY** | residual[k] ≥ residual[k+1] − rtol×\|residual[k]\| | Iteration, Residual (per-iter blocks) |

**Catalog:** `Metamorphic_Relations_Catalog.md` — formal 3-tuples, examples, contraindications, and implementation notes for each relation type.

**Default tolerance:** `tier3_tolerance = 1e-5` (slightly looser than Tier 2).

**Relation cost:** All currently implemented relation types (CONSERVATION, SYMMETRY, SCALING, PERMUTATIVE, ROTATION, MONOTONICITY) require 1 extra compile + run (labeled `"cheap"` in the catalog). The theoretical zero-extra-run optimization for CONSERVATION/SYMMETRY is not yet implemented.

---

## LLM Judge + Three-Tier Override System

The LLM judge reads both codes and their stdout outputs and returns a functional equivalence verdict (VALID/INVALID/SKIP). For LASSI-generated codes, deterministic tier results take priority over the judge via four override rules applied in order:

| Override | Condition | Final verdict | Logged as |
|----------|-----------|---------------|-----------|
| **Tier 1 exhausted** (inline) | Per-code Tier 1 budget exhausted | **INVALID** + `continue` (skips judge entirely) | — |
| **Override 1** | Tier 2 budget exhausted | **INVALID** (overrides judge) | `false_positive` |
| **Override 2** | Tier 1 PASS/SKIP + Tier 2 PASS + Tier 3 PASS/SKIP | **VALID** (overrides judge) | `false_negative` |
| **Override 3** (last word) | Tier 3 budget exhausted | **INVALID** (overrides Override 2) | `false_positive` |

Override 3 runs after Override 2 and has final authority: a code that exhausts Tier 3 retries is INVALID regardless of any prior verdict.

The zero-shot (vanilla LLM) code always goes through the judge — it has no Tier 1/2/3 instrumentation, so the judge is its only correctness gate. No overrides apply to zero-shot.

Discrepancies are logged in `verification_log["judge_discrepancies"]` and CSV columns `judge_false_positives` / `judge_false_negatives`.

**Tier 2 coverage note:** `sample_first10` AND `sample_last10` are both captured and compared, so both boundary regions are covered. The weakness is specifically the **interior** (elements 10 through N-11 for large N). For convergence-sensitive solvers (e.g., Jacobi), interior divergence can be masked if boundary elements happen to look correct. This is precisely what Tier 3 PERMUTATIVE testing is designed to catch.

---

## Self-Correction Architecture

| Counter | Scope | Max | Trigger | Exhaustion behavior |
|---------|-------|-----|---------|---------------------|
| `self_correction_tier1_counter_code` | Per generated code (resets on fresh generation) | 3 | Tier 1 FAIL | Inline: `valid_code_flag=INVALID` + `continue` (skips Tier 2/3/judge) |
| `self_correction_verify_counter_code` | Per generated code (resets on fresh generation) | 5 | Tier 2 FAIL | Override 1: code marked INVALID; fresh generation next |
| Per-code Tier 3 counter | Per generated code | 3 | Tier 3 FAIL | Override 3: code marked INVALID (last word) |
| `self_correction_verify_counter` | Global, whole pipeline run | 15 | Tier 1 FAIL + Tier 2 FAIL + Tier 3 FAIL | Pipeline safety net; halts verification loop |

**Tier 1 exhaustion note:** When the per-code Tier 1 budget exhausts, the pipeline `continue`s immediately — Tier 2 and Tier 3 are never run for that code. This avoids overhead when Tier 1 has already determined the code is incorrect. The exhausted code stays in `code_blocks` as an artifact (no `pop()`).

**Tier 2 exhaustion note:** Code stays in `code_blocks` as an artifact. `compile_result = "energy_efficiency"` drives fresh generation on the next iteration.

---

## Task Modes

Configured by two notebook flags:

| Mode | `power_measure` | `yes_power_feedback` | Optimization metric |
|------|----------------|---------------------|-------------------|
| Energy Efficiency (EE) | `True` | `True` | GPU energy (joules) |
| Runtime Performance | `False` | `True` | Wall-clock time (s) |
| Translation (one-shot) | `False` | `False` | N/A (no loop) |

---

## Output Artifacts

| File | Contents |
|------|----------|
| `lassi-verify_pipeline_data.csv` | One row per trial; all `experiment_data` fields |
| `verification_log_<name>_Trial<N>.json` | Full Tier 2/3 results, judge discrepancies, config |
| `<name>_Trial<N>.cu` | Each generated code version |
| `<name>_verify.cu` | Instrumented SOURCE_verify (Tier 1 + Tier 2) |
| `<name>_best_clean.cu` | Best LASSI code stripped of LASSI-VERIFY blocks |
| `<name>_mt_<relation>.cu` | Tier 3 metamorphic variant sources |
| `<name>_Trial<N>.txt` | Full metadata, prompts, stdout, comparison response |

### Key CSV columns
- `source_v0_runtime_s` — uninstrumented source baseline
- `lassiee_runtime_s` — best LASSI instrumented runtime
- `lassiee_runtime_diff` — % change vs source (instrumented — includes overhead)
- `lassiee_clean_runtime_s` — best LASSI uninstrumented runtime (fair comparison)
- `lassiee_clean_runtime_diff` — % change vs source (uninstrumented — paper-reportable)
- `lassiee_clean_status` — `success` / `compile_error` / `execute_error` / `skipped`
- `tier2_pass_count`, `tier2_fail_total`
- `tier3_pass_count`, `tier3_fail_total`, `tier3_active_relations`
- `judge_false_positives`, `judge_false_negatives`

---

## Key Configuration Parameters (notebook `experimental_setup()`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `run_tier2` | `True` | Enable Tier 2 passthrough equivalence checking |
| `tier2_tolerance` | `1e-6` | Relative FP tolerance for Tier 2 block comparison |
| `run_tier3` | `True` | Enable Tier 3 metamorphic testing |
| `tier3_tolerance` | `1e-5` | Relative FP tolerance for MT relation checks |
| `run_llm_judge` | `True` | Enable LLM judge (always on for zero-shot) |
| `run_code_analysis` | `False` | Pre-loop LLM rubric step (pre-Tier-1/2 stop-gap) |
| `power_measure` | `False` | GPU power/energy monitoring (requires pynvml) |
| `yes_power_feedback` | `True` | Run iterative self-improvement loop |
| `vary_llm_temp` | `True` | Vary LLM temperature across iterations |

---

## Implementation Files

| File | Role |
|------|------|
| `lassi/pipeline_v4.py` | Main pipeline orchestration (~1700 lines) |
| `lassi/verify.py` | Tier 1, 2, and 3 instrumentation + checking functions |
| `lassi/code_manager.py` | Compile, execute, GPU vendor detection |
| `lassi/pipeline_methods.py` | Stage 0, code analysis, metrics helpers |
| `lassi/llm_methods.py` | LLM API calls (Argo/Ollama/Cerebras) |
| `lassi/prompt_dictionary.py` | All LLM prompt templates + catalog loading |
| `lassi/context/Invariant_Assertions_Catalog.md` | Tier 1 assertion reference |
| `lassi/context/Metamorphic_Relations_Catalog.md` | Tier 3 relation reference |
| `LASSI_power_pipeline_v4.ipynb` | Jupyter notebook: experiment configuration + run |
