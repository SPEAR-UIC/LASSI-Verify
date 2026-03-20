
---

# Passthrough Data Block Catalog for GPU Code Empirical Equivalence Checking

## What Passthrough Data Blocks Are

Passthrough data blocks are **observation points** inserted into the source code that capture
and print computational state to stdout in a structured, labeled format. They implement the
source-as-specification premise: the source code's observable output at these points defines
the correctness criterion against which any refactored version is evaluated.

Both the source (SOURCE_verify) and any LASSI-generated refactored code carry these blocks
unaltered through the transformation, ensuring that both produce directly comparable output
streams. A refactored code is empirically equivalent if its passthrough output matches the
source's output at every observation point, within floating-point tolerance.

Passthrough blocks are the Tier 2 empirical analog of observation points in translation
validation (Pnueli et al.), where source and target state must correspond at designated
control locations. They are also analogous to Necula's observational equivalence criterion:
two executions are equivalent if they produce the same sequence of observable events.

---

## Critical Distinction: What to Capture vs. What to Avoid

### ✅ Capture: Kernel-written output at the application's own output boundary

**Anchor to the application's own output transfers.** The source developer already marked
where the computed result lives: wherever they perform their final D→H `cudaMemcpy` back to
the host (to print, validate, or return results to the caller). That is the canonical output
location. Place the passthrough at or immediately before that existing D→H transfer — using
the same host buffer if one already exists, or inserting a new D→H copy only of the same
device buffer.

**Before placing each passthrough, confirm kernel ownership.** For every device buffer you
plan to capture, trace backward through the code from the passthrough location and answer:
*"Was this device buffer written by a GPU kernel call (`<<<...>>>`) at any point between the
last H→D copy of this buffer and this passthrough?"* If the answer is NO — the buffer still
holds the original host-initialized values and has never been modified by a kernel — then you
are capturing INPUT data, not computed output. Move the passthrough later in the code to a
point where the kernel has already written to the buffer.

What to capture:
- The device buffer that was the **output (destination) argument** of the final kernel call
- Final solution vectors, matrices, or field arrays at program completion
- Scalar results that reflect computed values (not just iteration counts or diagnostics)
- Intermediate arrays at iteration boundaries (every N steps for iterative algorithms)

### ❌ Do NOT Capture

**Scatter-write / thread-index-dependent buffers — the most dangerous false-positive source.**
Any buffer populated by writes of the form `buffer[f(thread_id, input_value)] = flag` records
*which data values a particular thread happened to observe*, not what the algorithm computed.
This is implementation-specific: a valid optimization that changes launch geometry (block count,
block size, warp assignment) places different threads at different array positions, so different
input values get flagged → the buffer changes → Tier 2 FAIL on every optimized version, even
ones that produce the identical algorithmic result. Example:
```cpp
// INSIDE the kernel — BAD passthrough target
if (tx == 11) {
    int bufIndex = gpuSrc[xidx];          // xidx depends on blockIdx.x * blockDim.x
    outputBuffer[bufIndex] = 1;            // sets a different bit for every launch geometry
}
```
A refactored code that changes grid dimensions writes a different set of `1` flags → `max_err=1.0`
against the spec even though the DP output is identical. This false-failure blocks ALL optimization
in that trial, draining the entire self-correction budget.

**Rule:** never instrument a buffer whose contents encode "which threads read which input values."
Only instrument buffers whose contents encode "what the algorithm computed."

---

Other values to avoid:
- Intermediate GPU kernel state (not host-accessible without an extra D→H copy)
- Timing values, performance counters, or power readings (transformation-dependent)
- Thread-local or block-local variables (not visible from host)
- **Solver convergence diagnostics: residuals, error norms, iteration counts, convergence
  scalars** — these are implementation-internal metrics that legitimately differ between
  correct implementations using different FP evaluation orders or algorithmic strategies
  (e.g., a swap-kernel implementation and a pointer-swap ping-pong implementation converge
  via different floating-point trajectories, producing different residual values at exit
  even though both produce the same correct output field). Tier 1 assertions check that
  the solver terminated correctly (finite value, correct sign, sentinel PASS); Tier 2
  checks the actual output the solver produced. Capture the **primary output array** (the
  converged field, solution vector, output buffer) — not the diagnostic that describes
  how hard the solver worked to get there.
- Values that legitimately change between source and refactored code due to
  floating-point reordering (prefer statistics like min/max/mean which are more robust)
- Raw binary data — always convert to human-readable text format

---

## Structured Output Format

Every passthrough block MUST use this exact labeled format so the deterministic comparison
parser can extract and match blocks between source and refactored output streams.

**Printf output template (appears in program stdout):**
```
=== BEGIN LASSI-VERIFY PASSTHROUGH [label] ===
Buffer: <buffer_name>
Type: <dtype>
Elements: <N>
Statistics: min=<val>, max=<val>, mean=<val>
Sample_first10: <v0> <v1> <v2> <v3> <v4> <v5> <v6> <v7> <v8> <v9>
Sample_last10: <vN-10> <vN-9> <vN-8> <vN-7> <vN-6> <vN-5> <vN-4> <vN-3> <vN-2> <vN-1>
Sample_spread10: <v[N/11]> <v[2N/11]> ... <v[10N/11]>
=== END LASSI-VERIFY PASSTHROUGH [label] ===
```

**Why no `sum`:** `sum = mean × N` is redundant given the `mean` and `Elements` fields already
in the block. For large N (e.g., 16K×16K = 268M elements), even sub-epsilon per-element
floating-point differences from operation reordering accumulate into a large absolute sum
difference — making `sum` an unreliable correctness criterion. Every statistic in this format
is checked by the comparison parser; including an unchecked or unreliable field would be
misleading. Use `min`, `max`, `mean`, and samples — these provide sufficient correctness
evidence without N-scaling artifacts.

**Why `Sample_spread10`:** `Sample_first10` covers indices 0–9 and `Sample_last10` covers
indices N-10 to N-1. For row-major GPU output buffers, these elements are often boundary
cells that remain unchanged even when interior computation is wrong. `Sample_spread10`
samples 10 elements at evenly-spaced interior indices `(k+1)*N/11` for k=0..9, giving
guaranteed interior coverage independent of array layout. All three sample fields are checked
element-wise by the Tier 2 deterministic parser.

**Label naming convention:** Use descriptive labels that identify the observation point.

- **Single-location block:** the label must be unique across all passthrough blocks (each
  identifies a distinct observation point). Examples:
  - `[final_solution]` — the primary output array at program completion
  - `[post_kernel_fieldA]` — field array after a named kernel completes

**Floating-point precision:** Always use `%.17g` format for float/double values to preserve
full IEEE 754 double precision. This ensures the deterministic comparison parser can detect
meaningful differences rather than rounding artifacts.

```cpp
// Correct: full precision for doubles
printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n",
       min_val, max_val, mean_val);

// Correct: sample values at full precision
printf("Sample_first10: %.17g %.17g %.17g\n", host_buf[0], host_buf[1], host_buf[2]);
```

**Integer buffers:** Use `%d` (int) or `%ld` (long), no precision issue.
**Complex buffers:** Print real and imaginary components separately, one component per line.

---

## Protective Comment Markers

The C++ protective comment wrapper ensures LLMs performing subsequent transformation steps
do not remove or modify the passthrough block:

```cpp
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [label] (VERIFICATION BLOCK — do not rename, edit, or delete this block; this [label] must remain character-for-character identical) >>>

// ... D→H copy and printf statements ...

// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

The label in the protective comment should match the label used in the printf output.
When multiple passthrough blocks are present, each has its own protective wrapper with
a unique label, so the LLM can distinguish them.

**IMPORTANT for multi-block codes:** Preserve ALL existing `LASSI-VERIFY ASSERTIONS` blocks
and ALL existing `LASSI-VERIFY PASSTHROUGH` blocks without modification. Only ADD new
passthrough blocks; do not alter or merge existing ones.

---

## Observation Point Selection Principles

The LLM selects observation points based on the code's computational structure. Work through
these priorities in order, stopping when you have sufficient coverage:

### Priority 1: Program Completion (Always Required)
Insert at least one passthrough block capturing the primary output buffer(s) at program
completion, immediately before memory deallocations. This is the minimum requirement for
every code.

### Priority 2: After Major Kernel Completions
Insert after any kernel that produces a primary output or significantly transforms the data
state. Place AFTER cudaDeviceSynchronize() (or equivalent) and AFTER the D→H copy that
brings results to the host.

### Priority 3: Post-Loop Final State (Iterative and Parametric Codes)
For codes with outer loops (iterative solvers, parametric sweeps, time-stepping algorithms),
insert a block **AFTER the loop exits**, capturing the final output state.

**⛔ NEVER insert a passthrough block inside any loop body** — even a small fixed-count
loop. A block inside a loop executes once per iteration, producing one labeled output block
per iteration. For a loop with N=2001 iterations, this creates 2001 output blocks, making
the comparison impractical and potentially causing context-window overflow in the LLM judge
(which receives the full stdout). The final post-loop state provides the correctness evidence
needed; per-iteration granularity adds overhead without benefit. If Tier 1 invariant
assertions are needed inside a loop, use the Tier 1 assertion format (not passthrough) —
assertions print only PASS/FAIL without data volume.

Examples of correct placement:
- Iterative solver (Jacobi, CG): capture final solution array **after** the iteration loop
- Parametric sweep: capture final output array **after** the sweep loop
- Time-stepping: capture final field state **after** the time loop

#### ⚠️ Ping-Pong Buffer / Pointer-Swap Patterns

Many GPU iterative algorithms use double-buffering where two device pointers alternate roles
each iteration via a pointer swap:
```cpp
// Typical ping-pong pattern (pathfinder, stencil codes, cellular automata)
for (int t = 0; t < NUM_ITER; t++) {
    kernel<<<grid, block>>>(d_src, d_dst, ...);  // reads src, writes dst
    cudaDeviceSynchronize();
    int* temp = d_dst; d_dst = d_src; d_src = temp;  // swap pointer values
}
// After the loop: d_src points to the LAST-WRITTEN buffer (the final result)
//                 d_dst points to the SECOND-TO-LAST buffer
```

**Critical rule:** After a loop with N pointer swaps, capture from the variable that was the
**kernel's output argument in the last iteration** — this is always `d_src` after the final
swap (since the swap at the end of iteration N moves the new result into `d_src`). Do NOT
assume that `d_src` always holds input data or that `d_dst` always holds output data — these
roles alternate every iteration, and the variable names are misleading after the loop.

**Step-by-step procedure for ping-pong patterns:**

1. Identify which variable the kernel WRITES to inside the loop (typically the second device
   pointer argument, labeled "dst" or "result" or similar)
2. Note that the pointer swap at the END of each iteration moves that written buffer into
   `d_src` (or equivalent) — so AFTER the loop, `d_src` holds the final computed result
3. Place the passthrough block AFTER the loop's pointer swap has executed, capturing from
   `d_src` (the variable that received the last-written buffer via the swap)
4. Do NOT capture from `d_dst` after the loop — it holds the second-to-last iteration's
   result (or the original source data for even-count loops)

**⚠️ Verification step:** After writing the passthrough instrumentation, mentally trace one
full iteration of the loop to confirm: "If this was the last iteration, which variable
pointer ends up with the newly-written data after the swap?" Capture from that variable.

**Wrong pattern (do NOT do this):**
```cpp
for (int t = 0; t < NUM_ITER; t++) {
    kernel<<<g,b>>>(d_src, d_dst);
    int* tmp = d_dst; d_dst = d_src; d_src = tmp;  // swap inside loop
}
// BAD: capturing d_dst here — this is the second-to-last result, NOT the final result
// >>> LASSI-VERIFY PASSTHROUGH [final_result] <<<
{ cudaMemcpy(lv_buf, d_dst, ...); /* WRONG: d_dst has wrong data after loop */ }
```

**Correct pattern:**
```cpp
for (int t = 0; t < NUM_ITER; t++) {
    kernel<<<g,b>>>(d_src, d_dst);
    int* tmp = d_dst; d_dst = d_src; d_src = tmp;  // swap inside loop
}
// CORRECT: d_src now points to the buffer last written by the kernel
// >>> LASSI-VERIFY PASSTHROUGH [final_result] <<<
{ cudaMemcpy(lv_buf, d_src, ...); /* CORRECT: d_src has final result */ }
```

### Priority 4: Multi-Phase Pipeline Boundaries
For codes with multiple distinct computation phases (e.g., preprocessing → main solve →
postprocessing), insert a checkpoint at each phase boundary.

**Guideline on count:** Aim for 2-5 observation points total. More is not always better —
excessive passthrough output increases comparison cost and can obscure meaningful differences.

---

## Complete Code Template

```cpp
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [final_solution] (VERIFICATION BLOCK — do not rename, edit, or delete this block; this [final_solution] must remain character-for-character identical) >>>
{
    // --- LASSI-VERIFY PASSTHROUGH: capture final output buffer for empirical equivalence ---
    // Allocate temporary host buffer for D→H transfer
    double* lv_host_buf = (double*) malloc(N * sizeof(double));
    cudaMemcpy(lv_host_buf, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute statistics
    double lv_min = lv_host_buf[0], lv_max = lv_host_buf[0], lv_sum = 0.0;
    for (int lv_i = 0; lv_i < N; lv_i++) {
        if (lv_host_buf[lv_i] < lv_min) lv_min = lv_host_buf[lv_i];
        if (lv_host_buf[lv_i] > lv_max) lv_max = lv_host_buf[lv_i];
        lv_sum += lv_host_buf[lv_i];
    }
    double lv_mean = lv_sum / N;

    // Print structured passthrough output
    printf("\n=== BEGIN LASSI-VERIFY PASSTHROUGH [final_solution] ===\n");
    printf("Buffer: d_output\n");
    printf("Type: double\n");
    printf("Elements: %d\n", N);
    printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n",
           lv_min, lv_max, lv_mean);

    int lv_sample = (N < 10) ? N : 10;
    printf("Sample_first10:");
    for (int lv_i = 0; lv_i < lv_sample; lv_i++)
        printf(" %.17g", lv_host_buf[lv_i]);
    printf("\n");

    if (N > 10) {
        printf("Sample_last10:");
        for (int lv_i = N - 10; lv_i < N; lv_i++)
            printf(" %.17g", lv_host_buf[lv_i]);
        printf("\n");
    }

    // Sample_spread10: 10 evenly-spaced interior elements at indices (k+1)*N/11.
    // Covers interior of the buffer regardless of array layout, catching bugs
    // that Sample_first10/last10 miss when boundary elements are unchanged.
    if (N > 10) {
        printf("Sample_spread10:");
        for (int lv_k = 0; lv_k < 10; lv_k++) {
            int lv_spread_idx = (lv_k + 1) * N / 11;
            printf(" %.17g", lv_host_buf[lv_spread_idx]);
        }
        printf("\n");
    }

    printf("=== END LASSI-VERIFY PASSTHROUGH [final_solution] ===\n");
    free(lv_host_buf);
}
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

**Variable naming:** Use the `lv_` prefix for all passthrough-local variables (lv_host_buf,
lv_min, lv_max, etc.) to avoid shadowing existing code variables.

---

## Domain-Specific Guidance

### Linear Solvers (Jacobi, CG, GMRES, etc.)
Key observation points:
- Final solution vector `x` at program completion: `[final_solution]`
- Right-hand side `b` before solve (validates input unchanged): `[rhs_pre_solve]`

⛔ **Do NOT capture residual norms, convergence scalars, or error metrics** as passthrough
observation points. These are solver diagnostics — implementation-specific measures of
convergence progress that legitimately differ between the reference implementation and an
optimized one (the final residual value changes when the FP evaluation order of the stencil
or reduction changes, e.g., a swap-kernel vs. pointer-swap ping-pong optimization). Capturing
these causes systematic Tier 2 false-failures that are unfixable by self-correction, burning
through the SC budget without any prospect of resolution. Use Tier 1 assertions instead to
check that the solver terminated (is the residual finite? non-negative? below the tolerance
threshold?).

### Stencil Computations (CFD, heat equation, wave equation)
Key observation points:
- Output grid/field at program completion: `[final_field]`
- Field at intermediate time steps (e.g., every T/4 steps): `[field_step_N]`
- Boundary values if they must be preserved: `[boundary_values]`

### Graph Algorithms (BFS, SSSP, PageRank)
Key observation points:
- Output result array (distances, ranks, labels) at completion: `[final_result]`
- Convergence metric (delta, residual) if iterative: `[convergence_delta]`

### Monte Carlo / Stochastic Methods
Key observation points:
- Statistical moments of the output (mean, variance) at completion: `[output_moments]`
- Do NOT capture individual sample values (non-deterministic between runs)
- Capture the accumulated result array, not intermediate random draws

### Dense Linear Algebra (GEMM, reduction, etc.)
Key observation points:
- Output matrix/vector at completion: `[final_output]`
- For multi-kernel pipelines: after each major kernel: `[post_kernel_A]`, `[post_kernel_B]`

### Hash / Indexing Computations
Key observation points:
- Output hash/index array at completion: `[final_hashes]`
- Statistics (min, max, mean) are meaningful even when individual values are hash-dependent

### Parametric Sweep Loops and Iterative Solvers
Codes that run the same kernel for multiple parameter values (segment sizes, tile sizes,
grid levels, bandwidth steps) or iterative solvers (Jacobi, CG, GMRES, time-stepping).

Key observation points:
- Final output buffer **after** the loop completes: `[post_sweep_result]`, `[final_solution]`
- Final aggregate result at program completion: `[final_aggregate]`

**⛔ Do NOT insert passthrough blocks inside any loop.** Place all blocks after the loop exits.
A single post-loop capture of the final output state is sufficient for correctness verification.

```cpp
// CORRECT — block AFTER the sweep loop
for (int seg_size : seg_sizes) {
    launch_kernel(..., seg_size, ...);
    cudaDeviceSynchronize();
}
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [post_sweep_result] >>>
{ /* D→H copy of final output, stats, printf */ }
// <<< END LASSI-VERIFY PASSTHROUGH <<<

// WRONG — block INSIDE loop; produces N output blocks (one per iteration)
for (int seg_size : seg_sizes) {
    launch_kernel(..., seg_size, ...);
    cudaDeviceSynchronize();
    // >>> BEGIN LASSI-VERIFY PASSTHROUGH [post_sweep_result] >>>  ← NEVER DO THIS
    { /* ... */ }
    // <<< END LASSI-VERIFY PASSTHROUGH <<<
}
```

---

## Anti-Patterns: What NOT to Do

### ❌ WRONG: Capture inside device kernel
```cpp
// BAD — cannot printf from device kernel context in this pattern
__global__ void kernel(...) {
    printf("value = %f\n", d_data[threadIdx.x]);  // Non-deterministic, not structured
}
```
**CORRECT:** Copy to host after kernel + cudaDeviceSynchronize(), then printf.

### ❌ WRONG: Missing label (breaks parser)
```cpp
printf("=== BEGIN PASSTHROUGH DATA ===\n");  // No label — parser cannot match
```
**CORRECT:** Always include `[label]` in the BEGIN/END markers.

### ❌ WRONG: Insufficient precision
```cpp
printf("Statistics: min=%f, max=%f\n", min_val, max_val);  // %f loses precision
```
**CORRECT:** Use `%.17g` for all float/double values.

### ❌ WRONG: Capturing solver convergence diagnostics as passthrough observation points
```cpp
// BAD — the final error scalar is a solver diagnostic, not a computational result.
// Its FP value legitimately changes when the optimization alters the evaluation order
// (e.g., swap-kernel → pointer-swap), causing permanent Tier 2 failures that self-
// correction cannot resolve because no code change can reduce a FP-order difference.
printf("\n=== BEGIN LASSI-VERIFY PASSTHROUGH [final_error_scalar] ===\n");
printf("Buffer: error\nType: float\nElements: 1\n");
printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n", error, error, error);
printf("Sample_first10: %.17g\n", (double)error);
printf("=== END LASSI-VERIFY PASSTHROUGH [final_error_scalar] ===\n");
```
**CORRECT:** Use a Tier 1 assertion to check that the solver terminated correctly, and a
passthrough block capturing the primary output array (the converged field):
```cpp
// Tier 1: check solver terminated correctly
{
    bool _t1_pass = (std::isfinite(error) && error >= 0.0f);
    printf("LASSI-VERIFY ASSERTIONS [post_solve_error_valid]: %s\n",
           _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// Tier 2: capture the actual computational result
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [final_solution] >>>
{ /* D→H copy of h_f, stats, printf of field array */ }
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

### ❌ WRONG: Shadowing existing variable names
```cpp
double* host_buf = ...;  // May shadow an existing host_buf in the outer scope
```
**CORRECT:** Use the `lv_` prefix for all passthrough-local variables.

### ❌ WRONG: Modifying algorithm flow
```cpp
// BAD — adding an extra D→H copy that changes synchronization behavior
cudaMemcpy(h_intermediate, d_intermediate, ...);  // Extra sync point mid-kernel-sequence
// ... then continue launching more kernels
```
**CORRECT:** Only add passthrough blocks at natural boundaries where D→H copies already
exist or where adding one after cudaDeviceSynchronize() does not change the algorithm flow.

### ❌ WRONG: Inserting a passthrough block inside any loop
```cpp
// BAD — block inside loop produces N output blocks (one per iteration)
// This inflates comparison overhead and can cause context-window overflow in the LLM judge.
for (int seg_size : seg_sizes) {
    launch_kernel(..., seg_size, ...);
    cudaDeviceSynchronize();
    // >>> BEGIN LASSI-VERIFY PASSTHROUGH [post_sweep_result] >>>  ← NEVER DO THIS
    { /* runs N times — produces N labeled output blocks */ }
    // <<< END LASSI-VERIFY PASSTHROUGH <<<
}
```
**CORRECT:** Place the block **after** the loop, capturing the final output state once:
```cpp
for (int seg_size : seg_sizes) {
    launch_kernel(..., seg_size, ...);
    cudaDeviceSynchronize();
}
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [post_sweep_result] >>>
{ /* single capture of final state */ }
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

---

## Notes on Implementation

1. **All passthrough blocks use standard library functions only:**
   - `<cstdio>` / `<stdio.h>` for printf
   - `<cstdlib>` / `<stdlib.h>` for malloc/free
   - `<cuda_runtime.h>` for cudaMemcpy (already included in CUDA source)

2. **Block scope:** Wrap each passthrough block in `{ }` to limit local variable scope.

3. **Output order:** Always print the header line (`=== BEGIN ...`), then Buffer/Type/Elements,
   then Statistics, then Sample_first10, then Sample_last10 (if N > 10), then Sample_spread10
   (if N > 10), then the footer. Consistent ordering is required for the deterministic parser.

4. **HIP adaptation:** Replace `cudaMemcpy`/`cudaMemcpyDeviceToHost` with
   `hipMemcpy`/`hipMemcpyDeviceToHost`. All other printf logic is identical.

5. **Fortran adaptation:** Use `write(*,*)` statements with equivalent format specifiers.
   The `=== BEGIN/END ===` delimiter strings must be identical to the C++ format.

---
