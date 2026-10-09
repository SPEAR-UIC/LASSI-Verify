
# LASSI-Verify Pipeline Overview

**Version:** v4 — post-hardening (branch `verify-hardening`, updated 2026-07-23)
**Target GPU:** NVIDIA RTX 5000 Ada (sm_89); also supports AMD ROCm via HIP
**Primary benchmark collection:** HeCBench (CUDA C++ kernels)

> **Changelog — 2026-07-23 accuracy pass (sync doc to committed code through `7aaff6c`).**
> Sections touched in THIS pass (new or corrected material): (1) new **"Final Witness Chain"**
> section; (2) witness step added to the **Pipeline Flow** diagram and the **Tier 2** section;
> (3) **Tier 3 "Implemented Relation Types"** table extended with INVARIANCE, SENSITIVITY, and
> ADDITIVITY (was 6 types, now 9) + Mode-2 canonical-transform note; (4) **Self-Correction
> Architecture** budgets corrected (per-code Tier-2 = 3, global safety net = 45); (5)
> **zero-shot** path rewritten (single combined T1+T2 verify binary); (6) **Key Configuration
> Parameters** adds `run_witness_chain`; (7) **Implementation Files** line count + witness
> references. Everything NOT listed here is June-vintage (≤ `34ba419`) prose that was
> re-verified against the code this pass and left unchanged — most notably the judge-as-baseline
> model, `code_blocks` layout, and the Tier 1/2 core mechanics. Design provenance for the new
> material: `05abd7a`/`802e38d`/`30d1c69` (witness chain), `d837910`/`f6c918c`/`668de6e`/`7aaff6c`
> (INVARIANCE/SENSITIVITY + Mode-2), `014dfe1`/`e094788` (zero-shot consolidation),
> `bc1521d`/`71f96d9` (Stage-0 retry loops), `1d3733b` (determinism gate).

---

## What LASSI-Verify Does

LASSI-Verify is an adaptive empirical verification framework for parallel scientific GPU code transformations. Its goal is to determine whether an LLM-generated refactoring of a CUDA kernel (optimized for energy efficiency or runtime) is functionally equivalent to the original source and genuinely better-performing. The framework is *adaptive* in that verification properties are drawn from structured catalogs and instantiated relative to each specific transformation, while the three-tier architecture and mandatory coverage requirements remain fixed.

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
│ WITNESS-CHAIN INSTRUMENTATION (if run_witness_chain=True)       │
│  Separate LLM insertion step (Witness_Chain_Harness.md) wraps   │
│  the timed loop with a sentinel-write → kernel → chain-update    │
│  harness; captures final chain as a Tier 2 passthrough block    │
│  [witness_chain_state]. Requires run_tier2. A per-trial 64-bit  │
│  LASSI_VERIFY_SEED is injected at runtime. See "Final Witness   │
│  Chain" section. (Re-inserted deterministically into each variant.)│
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 0b: Test SOURCE_verify                                    │
│  Compile + execute *_verify.cu; collect instrumented baseline.  │
│  Captures: stdout with PASSTHROUGH blocks → tier2_spec_output   │
│  (This becomes the Tier 2 reference for all generated codes.)   │
│  A tolerance-aware determinism gate re-runs SOURCE_verify and    │
│  requires self-consistent output before it is accepted as spec.  │
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
         │  │  Includes [witness_chain_state] (exact  │  │
         │  │  equality) — a divergence here is the   │  │
         │  │  cheat signal, surfaced as tier2 FAIL.  │  │
         │  │  FAIL → self-correction (up to 3x/code).│  │
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

**What it does:** The LLM inserts runtime assertion checks directly into the source C++ code as labeled comment blocks. These blocks execute inline during each run and always print exactly one `PASS` or `FAIL` line per block to stdout, plus a single `tier1_complete` sentinel that confirms instrumentation ran to completion. Blocks are placed OUTSIDE solver/compute loops (one print per block per program execution).

**Block format — two types (in C++ source):**
```cpp
// Type 1: per-observation-point check (always prints one PASS or FAIL line)
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - [description]
{
    bool _t1_pass = (condition);
    printf("LASSI-VERIFY ASSERTIONS [assertion_label]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<

// Type 2: completion sentinel (ONE per function, AFTER all Type 1 blocks, OUTSIDE all loops)
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - tier1_complete (sentinel)
{
    printf("LASSI-VERIFY ASSERTIONS [tier1_complete]: PASS\n"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

**Catalog:** `Invariant_Assertions_Catalog.md` — provides a 3-layer framework (universal, domain, problem-specific) with examples for GPU kernels. All examples use the always-print `PASS/FAIL` printf pattern (not `assert()`). The sentinel pattern is required and documented in Pattern 4.

**When it runs (instrumentation):** Applied once at pipeline start to produce SOURCE_verify. All LASSI-generated codes inherit the assertion blocks because they are generated from SOURCE_verify.

**When it runs (per-iteration checking — Option B):** Inside the main while loop, after each generated code executes, the pipeline parses its stdout via `check_tier1_assertions()`. FAIL lines → FAIL verdict; sentinel present + all PASS → PASS; no assertion output (blocks stripped) → SKIP. The always-print format makes blocks look like essential output (similar to Tier 2 passthrough), reducing LLM stripping.

**Self-correction:** On FAIL, the failing assertion labels and summary are fed back to the LLM with the instruction to fix the code without removing any LASSI-VERIFY blocks. Up to 3 retries per code (per-code counter resets on fresh generation). When the per-code budget exhausts, the code is immediately marked INVALID and the pipeline `continue`s — skipping Tier 2, Tier 3, and the judge entirely. This fail-fast behavior avoids unnecessary overhead when Tier 1 has already determined the code is incorrect.

---

## Tier 2: Empirical Equivalence Checking

**Formal grounding:** Necula forward simulation / observational equivalence.

**What it does:** The LLM inserts labeled observation point blocks at key computation checkpoints. Each block captures aggregate statistics (min, max, mean) and a sample of values from an output buffer immediately after a kernel execution. When the generated code runs, its passthrough output is compared block-by-block against the SOURCE_verify reference output (`tier2_spec_output`).

**Block format (in C++ source, printed to stdout):**
```
=== BEGIN LASSI-VERIFY PASSTHROUGH [label] (VERIFICATION BLOCK — do not rename, edit, or delete this block; this [label] must remain character-for-character identical) >>>
Buffer: <buffer_name>
Type: <dtype_name>
Elements: <N>
Statistics: min=%.17g, max=%.17g, mean=%.17g
Sample_first10:  %.17g %.17g ...  (up to 10 values; indices 0–9)
Sample_last10:   %.17g %.17g ...  (10 values at N-10..N-1; omitted if N ≤ 10)
Sample_spread10: %.17g %.17g ...  (10 interior values at (k+1)*N/11, k=0..9; omitted if N ≤ 10)
=== END LASSI-VERIFY PASSTHROUGH <<<
```

`Sample_spread10` was added to provide guaranteed interior coverage. `Sample_first10`/`Sample_last10` only cover array boundaries (indices 0–9 and N-10..N-1). For convergence-sensitive solvers, interior divergence can be masked if boundary elements happen to look correct — notably observed in Jacobi when a pointer-swap bug left boundary rows intact while the interior diverged. `Sample_spread10` at evenly-spaced indices `(k+1)*N/11` catches interior bugs regardless of array layout.

**Comparison:** Deterministic first (`|a-b| ≤ atol + rtol×max(|a|,|b|)` with `atol = |spec_max|×rtol`). Falls back to LLM parse if deterministic parsing fails. Default `tier2_tolerance = 1e-6`. FP comparisons additionally support a **ULP-aware tolerance** path for integer-typedef / exact-FP cases (see `lassi/verify.py` tolerance helper). The largest observed per-block relative error is recorded as `tier2_max_relative_error` in the attempt log.

**Witness-chain block:** When `run_witness_chain=True`, the harness's final chain state is captured as an ordinary Tier 2 passthrough block labeled `[witness_chain_state]` and compared with **exact-string equality** (not the FP tolerance path). A mismatch therefore produces a Tier 2 `FAIL` with `witness_chain_state` in `tier2_failed_labels` — the witness chain has **no separate verdict of its own**; it rides on Tier 2. See the "Final Witness Chain" section for the mechanism.

**Self-correction:** On FAIL, the raw stdout diff (SOURCE_verify reference + generated output) is fed back to the LLM as self-correction context. Up to **3 retries per code** (`self_correction_verify_counter_code_max`); a global safety-net counter (`self_correction_verify_counter_max = 45`) bounds the whole run. When the per-code witness block is the failing label, a **cheat-detection-specific** feedback message is used instead of the generic "fix the passthrough" guidance.

**Stacking:** Tier 1 instrumentation is applied first; Tier 2 is applied on top; the witness-chain harness is a further, separate insertion step. The LLM prompts for Tier 2 and the witness step explicitly preserve existing ASSERTIONS blocks. Both the Tier 1 and Tier 2 Stage-0 instrumentation steps run **feedback-driven retry loops** (re-prompt on compile/parse failure) before the baseline is accepted.

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
| **ADDITIVITY** | Z_stat ≈ Y_stat + c (uniform additive shift, factor=0.0 convention) | Factor, Statistics, Sample_first10 |
| **PERMUTATIVE / ROTATION** | Step A: Z_stat ≈ Y_stat (invariance, factor=1.0); Step B: Z[idx1] ≈ Y[idx2] and Z[idx2] ≈ Y[idx1] (equivariance, fires when indices fall in first10 or last10 range) | Swap_idx1, Swap_idx2, Sample_at_idx1, Sample_at_idx2, Statistics, Sample_first10 |
| **CONSERVATION** | \|sum_after − sum_before\| / \|sum_before\| < rtol | Sum (before + after blocks) |
| **MONOTONICITY** | residual[k] ≥ residual[k+1] − rtol×\|residual[k]\| | Iteration, Residual (per-iter blocks) |
| **INVARIANCE** | Z_stat ≈ Y_stat under an input transform that must NOT change the observable (identity replay; factor conventionally 1.0). For categorical/equality-based kernels and co-normalized-scaling (z-score) kernels. | Statistics, Sample_first10 |
| **SENSITIVITY** | Inverse of INVARIANCE: a relevant input change **MUST** change the observable — "no response" = FAIL. Probed with a **multi-perturbation set** (`p_small` minimal bump + `p_outlier` generic out-of-range outlier); a variant is VALID iff it responds to **every** perturbation the SOURCE responded to (AND-all over source-active perturbations). Optional local-response probe via `canonical_target_index`. | Statistics, Sample_first10 (per perturbation) |

**Catalog:** `Metamorphic_Relations_Catalog.md` — formal 3-tuples, examples, contraindications, and implementation notes for each relation type.

**Mode-2 canonical-transform enforcement (SENSITIVITY / INVARIANCE soundness fix, `668de6e`/`7aaff6c`):** the input transform is wrapped in `MT_TRANSFORM` delimiters, captured byte-exact from the source at the Stage 0 filter, and re-inserted **byte-identically** into every variant (`_enforce_canonical_transform`). This makes the Stage-0 "source responds" guarantee transfer to each variant — the variant LLM controls only harness *placement*, never the stimulus content. It closes both the SENSITIVITY false-reject (variant used a weaker perturbation than the source was filtered with) and the INVARIANCE false-accept (variant used a weaker/no-op transform). `p_outlier` additionally carries a domain-aware guardrail so the outlier stays inside any hard input-domain constraint the kernel actually processes.

**Null-factor relations — Phase 3 future work (NOT yet implemented):**

> **Clarification (2026-07-23):** "null-factor" here means specifically the **element-wise
> buffer permutation / set-equivalence** comparison described below — that remains unimplemented.
> It does NOT refer to INVARIANCE or SENSITIVITY, which ARE implemented (above) as their own
> typed statistical checks (INVARIANCE via a factor=1.0 stat comparison, SENSITIVITY via the
> response/no-response perturbation probe) rather than via an element-wise buffer read.

All currently-implemented factor-based relation types require a concrete numeric `factor` in the contract JSON. The checker computes `Z_stat ≈ factor × Y_stat` and `Z_sample[i] ≈ factor × Y_sample[i]` for each block field. This is sufficient for SCALING, SYMMETRY (factor=−1), and PERMUTATIVE (factor=1.0, with index-swap equivariance checks).

A **null-factor** relation is one where no fixed scalar relates the variant output to the baseline — instead, the check requires element-wise array comparison between two specific output buffers (e.g., "the sorted output is always a permutation of the input"). These checks cannot be expressed as `Z_stat ≈ factor × Y_stat` on aggregate statistics and require full buffer reads and sorted-match or set-equivalence logic.

**Why deferred:** Implementing null-factor requires a distinct comparison code path (buffer-level permutation/equality rather than scalar scaling), a new passthrough field format that preserves enough elements for meaningful comparison, and a new variant-generation prompt that communicates the element-wise expectation to the LLM. This is non-trivial and would significantly expand the checker. The Phase 1/2 design deliberately excluded null-factor to ship a working scalar checker first.

**Importance:** Moderate. Null-factor relations would add coverage for PERMUTATIVE relations where sorted output is compared to sorted input (sortedness invariant), and for CONSERVATION checks where exact element preservation (not just sum) is the claim. However, the current PERMUTATIVE check (invariant stats + equivariant index swap) already provides structural coverage for most permutation-like operations. Null-factor becomes most valuable for kernels whose correctness claim is purely "output is a rearrangement of input with no scaling" — relatively rare in numerical HPC kernels.

**Anti-pattern:** Do NOT propose null-factor relations in the metamorphic contract (i.e., `factor: null` or omitting `factor`). The pipeline will SKIP them without any verification benefit, wasting a contract slot. Null-factor support will be added in Phase 3.

**Default tolerance:** `tier3_tolerance = 1e-5` (slightly looser than Tier 2).

**Relation cost:** All currently implemented relation types (SCALING, SYMMETRY, ADDITIVITY, PERMUTATIVE, ROTATION, CONSERVATION, MONOTONICITY, INVARIANCE, SENSITIVITY) require 1 extra compile + run per relation (labeled `"cheap"` in the catalog); SENSITIVITY runs one extra binary per perturbation in its set. ROTATION shares the same checker and variant-generation code path as PERMUTATIVE (both are handled by the `elif relation_type in ("PERMUTATIVE", "ROTATION"):` branch). The theoretical zero-extra-run optimization for CONSERVATION/SYMMETRY is not yet implemented. **Note (Tier 3 vs witness):** the witness-chain harness is stripped from the code before Tier 3 variant generation — Tier 3 does not compare chain state.

---

## Final Witness Chain (Cheat-Resistant Loop-Deletion Detection)

**What it is.** A resource-tracking instrumentation harness — the sibling of Tier 1 — that
deterministically detects transformations which preserve output equivalence while silently
*dropping computational work*. The canonical target is the timing-loop-deletion cheat: deleting a
benchmark's `for (i<repeat)` loop and post-dividing elapsed time by `repeat` to fabricate a speedup,
which every output-only check passes because the loop body is idempotent. **Enabled by default**
(`run_witness_chain=True`, `pipeline_v4.py:261`); **requires `run_tier2=True`** because it reports
through Tier 2.

**Catalog / authoritative spec:** `lassi/context/Witness_Chain_Harness.md` (matches the implemented
code; the "Decision Tree cases 1–8" section within it is explicitly marked *superseded*).

**Mechanism (final, as built — differs from the June-18 proposal):**
1. A per-trial 64-bit runtime seed `LASSI_VERIFY_SEED = 0x{secrets.randbits(64)}` is injected
   (`pipeline_v4.py:363`) — unknown at LLM code-generation time, so the expected chain value cannot
   be precomputed offline.
2. Each timed kernel launch is wrapped: a **pre-call sentinel write** (`_t1_chain_write_sentinel`)
   writes an 8-byte sentinel into the first 8 bytes of the kernel's output buffer, and a
   **post-call chain update** (`_t1_chain_update`) folds a signal into the running chain state. The
   sentinel write is a **type-agnostic 8-byte write** (reinterpret as `unsigned long long*`),
   which **supersedes** the older per-output-type sentinel Decision Tree.
3. **The fold is value-AGNOSTIC and BINARY** (job (ii), `30d1c69`): the chain update folds
   *whether* the kernel overwrote the 8-byte tripwire this iteration (`observed != sentinel → 1`,
   else `0`), **not what value** replaced it. If the kernel ran it overwrote the sentinel
   (signal=1); if it was skipped/no-op the sentinel persists (signal=0), diverging the chain. This
   deliberately makes the witness insensitive to benign output differences (FP rounding, alternate
   correct paths) — **content/value correctness is Tier 2's job; the witness chain is purely
   operational ("did the kernel actually run each iteration?").**
4. After the timed loop, the final chain state is emitted as the Tier 2 passthrough block
   `[witness_chain_state]` and compared with exact-string equality against SOURCE_verify's value
   under the same seed.

**Verdict locus — this is three tiers, not four.** The witness chain has **no independent verdict
column**. A chain mismatch surfaces as `tier2_verdict=FAIL` with `witness_chain_state` among
`tier2_failed_labels`. The framework verdict remains `T1 ∧ T2 ∧ T3` with the witness folded into T2.

**Determinism / canonicalization safeguards (post-proposal, prevent instrumentation-consistency
false-rejects):**
- `_enforce_canonical_witness_init` — exactly one continuous chain init placed before the
  *outermost* timed loop (never re-seeded inside a loop), so source and every variant accumulate a
  comparable single chain. Fixed the layout/segment-reduce init-placement over-strictness.
- `_enforce_canonical_witness_device_fns` — the framework deterministically re-derives both witness
  device-kernel bodies after insertion, guaranteeing byte-identical fold semantics for source and
  every variant regardless of LLM re-insertion variance.
- The harness is **re-inserted per variant** (a variant may have restructured/fused/graph-captured
  its loop), so a chain mismatch genuinely means the variant performed fewer real kernel
  invocations than the source — a dropped-work cheat — not an instrumentation artifact.

**Known limitation (open):** the pre-kernel sentinel write corrupts kernels that read-modify-write
their output buffer **in place** (confirmed on matrix-rotate; that kernel is held out of the current
campaign pending a fix). Tracked separately; do not describe the witness chain as universally safe
for in-place-mutation kernels.

**Logged fields:** `witness_chain_status` (which insertion path was taken:
`applied_decision_tree_case_N` / `soft_flag_no_safe_site` / `not_applied_no_timing_loop`) and
`witness_chain_mismatches` (count of LASSI variants whose chain diverged from source).

---

## LLM Judge as Pre-Tier Baseline Measurement

The LLM judge is a **baseline comparator** representing current LLM-only practice — it is NOT a framework tier. It runs immediately after compile+execute succeeds and BEFORE Tier 1/2/3, so it sees only what the executed code actually produced (the generated code and its stdout) and nothing from tier analysis. The judge returns a functional-equivalence verdict (VALID or INVALID); any other response is normalized to INVALID and counted against a 3-strike `judge_api_error_count` safety net that halts the pipeline on persistent API errors.

The judge verdict is recorded as `judge_raw_verdict` (stored in Table 2 as `judge_baseline_verdict`) and **does not participate** in deciding the framework verdict. The framework verdict (`valid_code_flag`, written to Table 2 as `final_verdict`) is derived from tier verdicts alone:

| Path | Condition | Framework verdict | `override_applied` tag |
|------|-----------|-------------------|------------------------|
| All tiers PASS | Tier 1 PASS + Tier 2 PASS + Tier 3 PASS-or-inactive | **VALID** | `none` |
| Tier 1 exhausted (inline) | Per-code T1 budget exhausted | **INVALID** + `continue` | `tier1_exhausted` |
| Tier 2 exhausted | Per-code T2 budget exhausted | **INVALID** | `tier2_exhausted` |
| Tier 3 exhausted | Per-code T3 budget exhausted | **INVALID** | `tier3_exhausted` |

For the zero-shot (vanilla LLM) code, the empirical verdict from a **single combined T1+T2 verify binary** (`014dfe1`: Tier-1 assertions and Tier-2 passthroughs are instrumented into one retrofit binary and run once — not the LASSI codes' separate multi-attempt T1→T2→T3 loop) IS the framework verdict when `verify_zeroshot=True`; these appear in the attempt log as `generation_type` zero-shot with `zs_tier1_verdict` / `zs_tier2_verdict` fields. A zero-shot variant that fails to compile/run still emits a Table 2 row with `final_verdict=INVALID` (`e094788`) rather than being dropped. The judge baseline is recorded separately with no override semantics. When `verify_zeroshot=False`, the judge baseline is the framework verdict (Mode A "vanilla LLM + judge only" baseline).

The per-attempt (judge_baseline, tier-final) pair is logged to `verification_log["judge_discrepancies"]` and used downstream to compute `judge_false_positives` (judge VALID + framework INVALID) and `judge_false_negatives` (judge INVALID + framework VALID) per trial — these are now derived directly from the raw verdict pair, not from a "type" tag set at override time.

**Tier 2 coverage note:** Three sample fields are captured: `Sample_first10` (indices 0–9), `Sample_last10` (N-10..N-1), and `Sample_spread10` (10 evenly-spaced interior indices `(k+1)*N/11`). Together these provide boundary + interior coverage. The original weakness — interior divergence masked by intact boundary elements (observed in a Jacobi pointer-swap bug) — is addressed by `Sample_spread10`. Tier 3 PERMUTATIVE testing provides additional structural coverage for cases where element-level sampling may still miss permutations.

---

## Self-Correction Architecture

| Counter | Scope | Max | Trigger | Exhaustion behavior |
|---------|-------|-----|---------|---------------------|
| `self_correction_tier1_counter_code` | Per generated code (resets on fresh generation) | 3 | Tier 1 FAIL | Inline: `valid_code_flag=INVALID` + `continue` (skips Tier 2/3/judge) |
| `self_correction_verify_counter_code` | Per generated code (resets on fresh generation) | 3 | Tier 2 FAIL (incl. `witness_chain_state`) | Code marked INVALID; fresh generation next |
| `self_correction_tier3_counter_code` | Per generated code | 3 | Tier 3 FAIL | Code marked INVALID (last word) |
| `self_correction_verify_counter` | Global, whole pipeline run | 45 | Tier 1 FAIL + Tier 2 FAIL + Tier 3 FAIL | Pipeline safety net; halts verification loop |

> **Note:** all three per-code budgets are **3** (`pipeline_v4.py:296,302,309`). A tier catch that
> is repaired within its 3-retry budget overwrites the failing `.cu` and leaves no on-disk artifact;
> only *budget exhaustion* preserves the failing file with `final_verdict=INVALID`. This is why most
> tier catches are invisible to the post-hoc oracle analysis (see `planning/oracle_analysis_runbook.md`
> Step B.5). The earlier "Tier 2 = 5, global = 15" figures were pre-`verify-hardening` values.

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
| `lassi-verify_pipeline_data.csv` | **Table 1** — one row per trial; all `experiment_data` fields (final outcomes, aggregated counts) |
| `lassi-verify_attempt_log.csv` | **Table 2** — one row per code attempt within a trial; per-attempt `judge_baseline_verdict`, `judge_response_text`, `final_verdict`, `tier1/2/3_verdict`, `override_applied`, `is_best_code`, runtime/energy |
| `verification_log_<name>_Trial<N>.json` | Full Tier 2/3 results, judge discrepancies, config |
| `<name>_Trial<N>.cu` | Each generated code version |
| `<name>_verify.cu` | Instrumented SOURCE_verify (Tier 1 + Tier 2) |
| `<name>_best_clean.cu` | Best LASSI code stripped of LASSI-VERIFY blocks |
| `<name>_mt_<relation>.cu` | Tier 3 metamorphic variant sources |
| `<name>_Trial<N>.txt` | Full metadata, prompts, stdout, comparison response |
| `LASSI_RESULT` artifact | Canonical winner record emitted when a validated LASSI winner beats the source (`3314363`). Note: the winning code file is named `_v{lassiee_version}_Trial<N>.cu` (the versioned winner), not plain `_Trial<N>.cu`. |

### Key CSV columns — Table 1 (`lassi-verify_pipeline_data.csv`)
- `lassiee_valid` — tier-dominant verdict for the best LASSI code (`VALID` / `INVALID` / `SOURCE`); used for Variant 1 (verified-pass@k) in pass@k analysis
- `base_v1_valid` — raw LLM judge verdict for the zero-shot code; used for Variant 3 (judge-pass@k zero-shot)
- `base_llm_verified_valid` — tier-only verdict for the zero-shot code from post-pipeline verification (True/False/NA); used for Variant 4 (verified-pass@k zero-shot)
- `source_v0_runtime_s` — uninstrumented source baseline
- `lassiee_runtime_s` — best LASSI instrumented runtime
- `lassiee_runtime_diff` — % change vs source (instrumented — includes overhead)
- `lassiee_clean_runtime_s` — best LASSI uninstrumented runtime (fair comparison)
- `lassiee_clean_runtime_diff` — % change vs source (uninstrumented — paper-reportable)
- `lassiee_clean_status` — `success` / `compile_error` / `execute_error` / `skipped`
- `tier2_pass_count`, `tier2_fail_total`
- `tier3_pass_count`, `tier3_fail_total`, `tier3_active_relations`
- `judge_false_positives`, `judge_false_negatives`

### Key CSV columns — Table 2 (`lassi-verify_attempt_log.csv`)
- `judge_baseline_verdict` — raw LLM judge output captured as a pre-tier baseline measurement; never overridden by tier verdicts; used for Variant 2 (judge-pass@k LASSI) on the `is_best_code=True` row. Older CSVs use the legacy column name `judge_verdict`; the analysis data loader transparently maps it onto this canonical name.
- `judge_response_text` — verbatim LLM judge prose (the reasoning the judge produced when it issued its VALID/INVALID verdict). Populated on every attempt that reached the judge, including SC intermediates, so cross-attempt judge-verdict shifts and the *why* behind any single verdict are queryable from the data. Empty on attempts where the judge wasn't invoked (compile/execute failed before the judge stage). Multi-line; CSV writers use `QUOTE_MINIMAL`. Also persisted into the `judge_discrepancies` entries in the sidecar JSON for grep-style access.
- `final_verdict` — framework verdict for the attempt (`VALID` / `INVALID`). Derived from tier verdicts alone for LASSI codes; equals the empirical verdict (when `verify_zeroshot=True`) or the judge baseline (when `verify_zeroshot=False`) for zero-shot codes. Drives winner selection and self-correction.
- `override_applied` — exhaustion / pipeline-budget tag for the attempt: `none` (clean tier-derived verdict), `tier1_exhausted`, `tier2_exhausted`, `tier3_exhausted`, `exhausted_t1_sc_budget`, `exhausted_sc_budget`, `global_sc_budget_exhausted`. Tier-vs-judge override tags from earlier schemas (`judge_override_*`) are no longer produced under the judge-as-baseline design.
- `tier1_verdict`, `tier2_verdict`, `tier3_verdict` — individual tier verdicts per attempt: `PASS` / `FAIL` / `SKIP`. All tiers are fully deterministic; parse errors in Tier 2 output produce `FAIL` and trigger self-correction. Budget exhaustion is encoded in `override_applied`, not in the verdict string.
- `is_best_code` — True on exactly one row per trial (the winning code's final attempt row)
- `is_final_attempt` — True for the definitive verdict row per code version; False for mid-retry self-correction rows
- `code_block_index` — index into `code_blocks` (2 = zero-shot, 3+ = LASSI v{n})

---

## Key Configuration Parameters (notebook `experimental_setup()`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `run_tier2` | `True` | Enable Tier 2 passthrough equivalence checking |
| `tier2_tolerance` | `1e-6` | Relative FP tolerance for Tier 2 block comparison |
| `run_tier3` | `True` | Enable Tier 3 metamorphic testing |
| `tier3_tolerance` | `1e-5` | Relative FP tolerance for MT relation checks |
| `run_witness_chain` | `True` | Enable the cheat-resistant witness-chain harness (requires `run_tier2`; reports via the `[witness_chain_state]` Tier 2 passthrough) |
| `run_llm_judge` | `True` | Enable LLM judge (always on for zero-shot) |
| `run_code_analysis` | `False` | Pre-loop LLM rubric step (pre-Tier-1/2 stop-gap) |
| `power_measure` | `False` | GPU power/energy monitoring (requires pynvml) |
| `yes_power_feedback` | `True` | Run iterative self-improvement loop |
| `vary_llm_temp` | `True` | Vary LLM temperature across iterations |

---

## Implementation Files

| File | Role |
|------|------|
| `lassi/pipeline_v4.py` | Main pipeline orchestration (~3960 lines) |
| `lassi/verify.py` | Tier 1, 2, 3 + witness-chain instrumentation, checking, strip, and canonicalization functions |
| `lassi/code_manager.py` | Compile, execute, GPU vendor detection, witness-chain seed |
| `lassi/pipeline_methods.py` | Stage 0, code analysis, metrics helpers |
| `lassi/llm_methods.py` | LLM API calls (Argo/Ollama/OpenAI-compatible) |
| `lassi/prompt_dictionary.py` | All LLM prompt templates + catalog loading (`get_codeknowledge`) |
| `lassi/context/Invariant_Assertions_Catalog.md` | Tier 1 assertion reference |
| `lassi/context/Metamorphic_Relations_Catalog.md` | Tier 3 relation reference |
| `lassi/context/Witness_Chain_Harness.md` | Witness-chain harness insertion spec (loaded for the dedicated witness step) |
| `run_lassi.py` / `lassiapp.py` | CLI entry point (`config.yaml`) / GUI launcher |
| `LASSI_power_pipeline_v4.ipynb` | Jupyter notebook: experiment configuration + run |
