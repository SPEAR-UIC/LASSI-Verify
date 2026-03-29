
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

## ⚠️ CORE PRINCIPLE: Final Output Only

**Passthrough blocks observe the program's INTERFACE (final output), not its IMPLEMENTATION
(intermediate state).** An optimization is explicitly allowed to change how intermediate
results are computed — different buffer layouts, fused kernels, reordered reductions. Only
the final output that the program produces must be preserved.

**One passthrough block per primary output buffer, placed at program completion.** In most
codes this means exactly one block. Two blocks are appropriate only when the program produces
two genuinely independent output buffers (e.g., two separate result arrays). More than two
is almost never justified.

### The one placement rule:
> Place each passthrough block **after the final `cudaMemcpy` (D→H) of the primary output
> buffer**, immediately before memory deallocations. This is the only correct location.

### What this means in practice:
- Do NOT place passthrough blocks at intermediate pipeline stages (preprocessing, key packing,
  input validation, intermediate kernel outputs)
- Do NOT capture host-side CPU reference or verification data — only GPU-computed output
- Do NOT capture "phase boundaries" between preprocessing → compute → postprocessing
- Do NOT add extra D→H copies mid-computation just to observe an intermediate buffer

---

## What to Capture

**Anchor to the application's own output transfers.** The source developer already marked
where the computed result lives: wherever they perform their final D→H `cudaMemcpy` back to
the host (to print, validate, or return results to the caller). That is the canonical output
location. Place the passthrough at or immediately after that existing D→H transfer — using
the same host buffer that already exists.

**Before placing each passthrough, confirm kernel ownership.** For every device buffer you
plan to capture, trace backward through the code from the passthrough location and answer:
*"Was this device buffer written by a GPU kernel call (`<<<...>>>`) at any point between the
last H→D copy of this buffer and this passthrough?"* If the answer is NO — the buffer still
holds the original host-initialized values and has never been modified by a kernel — then you
are capturing INPUT data, not computed output. Do not place a passthrough here.

What to capture:
- The device buffer that was the **output (destination) argument** of the final kernel call
- Final solution vectors, matrices, or field arrays at program completion
- Scalar results that reflect computed values (not just iteration counts or diagnostics)

## What NOT to Capture

**Intermediate pipeline stages — the most important rule.** An optimization can legitimately
compute intermediate results differently (fused kernels, different buffer layouts, reordered
reductions). Observing intermediate state causes Tier 2 FAIL on correct optimized codes.
Never place a passthrough at any of these locations:
- After input preprocessing or data packing steps (before the main kernel)
- After intermediate kernels that are not the final computation
- At "phase boundaries" between preprocessing → compute → postprocessing
- On host-side CPU reference or verification data (e.g., a CPU reference hash computed for
  validation purposes — capture the GPU output, not the CPU check)

**Scatter-write / thread-index-dependent buffers — the most dangerous false-positive source.**
Any buffer populated by writes of the form `buffer[f(thread_id, input_value)] = flag` records
*which data values a particular thread happened to observe*, not what the algorithm computed.
A valid optimization that changes launch geometry places different threads at different array
positions → different buffer contents → Tier 2 FAIL even though the algorithmic result is
identical.

**Rule:** never instrument a buffer whose contents encode "which threads read which input
values." Only instrument buffers whose contents encode "what the algorithm computed."

Other values to avoid:
- Timing values, performance counters, or power readings (transformation-dependent)
- Thread-local or block-local variables (not visible from host)
- **Solver convergence diagnostics: residuals, error norms, iteration counts, convergence
  scalars** — these are implementation-internal metrics that legitimately differ between
  correct implementations using different FP evaluation orders or algorithmic strategies.
  Use Tier 1 assertions to check that the solver terminated correctly (finite value, correct
  sign, below tolerance); use passthrough only for the primary output array (the converged
  field, solution vector, output buffer).
- Values that legitimately change between source and refactored code due to
  floating-point reordering (prefer statistics like min/max/mean which are more robust)

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
in the block. For large N, even sub-epsilon per-element floating-point differences accumulate
into a large absolute sum difference — making `sum` an unreliable correctness criterion. Use
`min`, `max`, `mean`, and samples — these provide sufficient correctness evidence without
N-scaling artifacts.

**Why `Sample_spread10`:** `Sample_first10` covers indices 0–9 and `Sample_last10` covers
indices N-10 to N-1. For row-major GPU output buffers, these are often boundary elements that
remain unchanged even when interior computation is wrong. `Sample_spread10` samples 10
elements at evenly-spaced interior indices `(k+1)*N/11` for k=0..9, giving guaranteed
interior coverage independent of array layout.

**Label naming convention:** Use a descriptive label that identifies the output buffer.
Examples: `[final_solution]`, `[final_hashes]`, `[final_field]`, `[output_result]`.
Each label must be unique across all passthrough blocks in the file.

**Floating-point precision:** Always use `%.17g` format for float/double values to preserve
full IEEE 754 double precision.

```cpp
printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n", lv_min, lv_max, lv_mean);
printf("Sample_first10: %.17g %.17g %.17g\n", host_buf[0], host_buf[1], host_buf[2]);
```

**Integer buffers:** Use `%d` (int) or `%ld` (long).
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

**IMPORTANT for multi-block codes:** Preserve ALL existing `LASSI-VERIFY ASSERTIONS` blocks
and ALL existing `LASSI-VERIFY PASSTHROUGH` blocks without modification. Only ADD new
passthrough blocks; do not alter or merge existing ones.

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
    // Use long long for lv_N to prevent integer overflow in index calculations
    // when total elements exceeds ~200M (e.g., N=16384 → N*N=268M for a 2D grid).
    long long lv_N = (long long)N;
    double lv_min = lv_host_buf[0], lv_max = lv_host_buf[0], lv_sum = 0.0;
    for (long long lv_i = 0; lv_i < lv_N; lv_i++) {
        if (lv_host_buf[lv_i] < lv_min) lv_min = lv_host_buf[lv_i];
        if (lv_host_buf[lv_i] > lv_max) lv_max = lv_host_buf[lv_i];
        lv_sum += lv_host_buf[lv_i];
    }
    double lv_mean = lv_sum / (double)lv_N;

    // Print structured passthrough output
    printf("\n=== BEGIN LASSI-VERIFY PASSTHROUGH [final_solution] ===\n");
    printf("Buffer: d_output\n");
    printf("Type: double\n");
    printf("Elements: %lld\n", lv_N);
    printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n",
           lv_min, lv_max, lv_mean);

    int lv_sample = (lv_N < 10) ? (int)lv_N : 10;
    printf("Sample_first10:");
    for (int lv_i = 0; lv_i < lv_sample; lv_i++)
        printf(" %.17g", lv_host_buf[lv_i]);
    printf("\n");

    if (lv_N > 10) {
        printf("Sample_last10:");
        for (long long lv_i = lv_N - 10; lv_i < lv_N; lv_i++)
            printf(" %.17g", lv_host_buf[lv_i]);
        printf("\n");
    }

    // Sample_spread10: 10 evenly-spaced interior elements at indices (k+1)*lv_N/11.
    // Uses long long arithmetic to prevent overflow when lv_N > ~200M elements
    // (e.g., a 16384×16384 2D grid has 268M elements; (k+1)*N overflows int at k=7).
    if (lv_N > 10) {
        printf("Sample_spread10:");
        for (int lv_k = 0; lv_k < 10; lv_k++) {
            long long lv_spread_idx = (long long)(lv_k + 1) * lv_N / 11;
            printf(" %.17g", lv_host_buf[lv_spread_idx]);
        }
        printf("\n");
    }

    printf("=== END LASSI-VERIFY PASSTHROUGH [final_solution] ===\n");
    free(lv_host_buf);
}
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

**Variable naming:** Use the `lv_` prefix for all passthrough-local variables (`lv_host_buf`,
`lv_min`, `lv_max`, etc.) to avoid shadowing existing code variables.

**If the primary output is already on the host** (i.e., a D→H copy already exists just before
or at this location), read directly from that host buffer — do not malloc a redundant copy:

```cpp
// When d_out (host buffer) is already populated by an existing cudaMemcpy:
cudaMemcpy(d_out, dev_out, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [final_hashes] >>>
{
    long long lv_N = (long long)N;  // long long prevents overflow in spread index for large N
    uint64_t lv_min = d_out[0], lv_max = d_out[0];
    double lv_sum = 0.0;
    for (long long lv_i = 0; lv_i < lv_N; lv_i++) {
        if (d_out[lv_i] < lv_min) lv_min = d_out[lv_i];
        if (d_out[lv_i] > lv_max) lv_max = d_out[lv_i];
        lv_sum += (double)d_out[lv_i];
    }
    double lv_mean = lv_sum / (double)lv_N;
    printf("\n=== BEGIN LASSI-VERIFY PASSTHROUGH [final_hashes] ===\n");
    printf("Buffer: d_out\nType: uint64_t\nElements: %lld\n", lv_N);
    printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n",
           (double)lv_min, (double)lv_max, lv_mean);
    int lv_s = (lv_N < 10) ? (int)lv_N : 10;
    printf("Sample_first10:");
    for (int lv_i = 0; lv_i < lv_s; lv_i++) printf(" %.17g", (double)d_out[lv_i]);
    printf("\n");
    if (lv_N > 10) {
        printf("Sample_last10:");
        for (long long lv_i = lv_N - 10; lv_i < lv_N; lv_i++) printf(" %.17g", (double)d_out[lv_i]);
        printf("\n");
        printf("Sample_spread10:");
        for (int lv_k = 0; lv_k < 10; lv_k++) printf(" %.17g", (double)d_out[(long long)(lv_k+1)*lv_N/11]);
        printf("\n");
    }
    printf("=== END LASSI-VERIFY PASSTHROUGH [final_hashes] ===\n");
}
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

---

## Special Cases

### Post-Loop Final State (Iterative and Parametric Codes)

For codes with outer loops (iterative solvers, parametric sweeps, time-stepping), insert the
passthrough block **AFTER the loop exits**, capturing the final output state once.

**⛔ NEVER insert a passthrough block inside any loop body.** A block inside a loop produces
one labeled output block per iteration. For a loop with N=2001 iterations, this creates 2001
output blocks, making comparison impractical. The final post-loop state provides all the
correctness evidence needed.

```cpp
// CORRECT — block AFTER the loop
for (int iter = 0; iter < max_iter; iter++) {
    kernel<<<grid, block>>>(d_src, d_dst, ...);
    cudaDeviceSynchronize();
}
cudaMemcpy(h_result, d_src, N * sizeof(double), cudaMemcpyDeviceToHost);
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [final_solution] >>>
{ /* stats and printf from h_result */ }
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

### Ping-Pong Buffer / Pointer-Swap Patterns

Many GPU iterative algorithms alternate two device pointers via pointer swap:
```cpp
for (int t = 0; t < NUM_ITER; t++) {
    kernel<<<grid, block>>>(d_src, d_dst, ...);  // reads src, writes dst
    cudaDeviceSynchronize();
    int* temp = d_dst; d_dst = d_src; d_src = temp;  // swap
}
// After the loop: d_src points to the LAST-WRITTEN buffer (final result)
//                 d_dst points to the second-to-last buffer
```

**Critical rule:** After N pointer swaps, capture from `d_src` (the variable that received
the last-written buffer via the final swap). Do NOT capture from `d_dst` after the loop.

**Step-by-step:**
1. Identify which variable the kernel WRITES to inside the loop (the "dst" argument)
2. The pointer swap at the END of each iteration moves that written buffer into `d_src`
3. Place the passthrough AFTER the loop exits, capturing from `d_src`

**Wrong:**
```cpp
for (int t = 0; t < NUM_ITER; t++) {
    kernel<<<g,b>>>(d_src, d_dst);
    int* tmp = d_dst; d_dst = d_src; d_src = tmp;
}
// BAD: d_dst here is the second-to-last result, NOT the final result
cudaMemcpy(lv_buf, d_dst, ...);  // WRONG
```

**Correct:**
```cpp
for (int t = 0; t < NUM_ITER; t++) {
    kernel<<<g,b>>>(d_src, d_dst);
    int* tmp = d_dst; d_dst = d_src; d_src = tmp;
}
// CORRECT: d_src now points to the buffer last written by the kernel
cudaMemcpy(lv_buf, d_src, ...);  // CORRECT
```

---

## Domain-Specific Guidance

### Linear Solvers (Jacobi, CG, GMRES, etc.)
- Capture the final solution vector `x` (or `h_x`) after program completion: `[final_solution]`
- ⛔ Do NOT capture residual norms, convergence scalars, or error metrics — these are
  solver diagnostics that legitimately differ between implementations with different FP
  evaluation order. Use Tier 1 assertions to check termination (finite, non-negative, below
  tolerance); use passthrough only for the converged solution array.

### Stencil Computations (CFD, heat equation, wave equation)
- Capture the output field/grid at program completion: `[final_field]`
- ⛔ Do NOT capture field state at intermediate time steps — an optimized implementation
  can legitimately produce different intermediate states while converging to the same final result.

### Hash / Indexing Computations
- Capture the output hash/index array after the final D→H copy: `[final_hashes]`
- Read directly from the existing host buffer (no redundant malloc/copy needed)

### Graph Algorithms (BFS, SSSP, PageRank)
- Capture the output result array (distances, ranks, labels) at completion: `[final_result]`
- ⛔ Do NOT capture convergence deltas or iteration-count scalars.

### Dense Linear Algebra (GEMM, reduction, etc.)
- Capture the output matrix/vector at completion: `[final_output]`

### Monte Carlo / Stochastic Methods
- Capture the accumulated result array at completion: `[output_moments]`
- Do NOT capture individual sample values (non-deterministic between runs)

---

## Anti-Patterns: What NOT to Do

### ❌ WRONG: Observing intermediate pipeline stages
```cpp
// BAD — capturing host-side preprocessing data (key packing, prefix sums, CPU reference)
// before the GPU kernel has run. This is INPUT data, not computed output.
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [packed_keys] >>>
{ /* stats on d_keys after packing but before kernel launch */ }  // WRONG

// BAD — capturing CPU reference verification data (not GPU output)
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [cpu_reference] >>>
{ /* stats on CPU-computed reference hashes */ }  // WRONG
```
**CORRECT:** Place one passthrough block after the final D→H copy of the GPU output buffer.
The intermediate steps (packing, preprocessing, CPU verification) are implementation details
that an optimization is allowed to change.

### ❌ WRONG: Capturing solver convergence diagnostics as passthrough observation points
```cpp
// BAD — the final error scalar is a solver diagnostic, not a computational result.
// Its FP value legitimately changes when the optimization alters the evaluation order.
printf("\n=== BEGIN LASSI-VERIFY PASSTHROUGH [final_error_scalar] ===\n");
printf("Buffer: error\nType: float\nElements: 1\n");
printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n", error, error, error);
printf("=== END LASSI-VERIFY PASSTHROUGH [final_error_scalar] ===\n");
```
**CORRECT:** Use a Tier 1 assertion to check that the solver terminated correctly, and a
passthrough block capturing the primary output array:
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

### ❌ WRONG: Capture inside device kernel
```cpp
__global__ void kernel(...) {
    printf("value = %f\n", d_data[threadIdx.x]);  // Non-deterministic, not structured
}
```
**CORRECT:** Copy to host after kernel + cudaDeviceSynchronize(), then printf.

### ❌ WRONG: Inserting a passthrough block inside any loop
```cpp
for (int seg_size : seg_sizes) {
    launch_kernel(..., seg_size, ...);
    cudaDeviceSynchronize();
    // >>> BEGIN LASSI-VERIFY PASSTHROUGH [post_sweep_result] >>>  ← NEVER DO THIS
    { /* runs N times — produces N labeled output blocks */ }
    // <<< END LASSI-VERIFY PASSTHROUGH <<<
}
```
**CORRECT:** Place the block after the loop.

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

### ❌ WRONG: Shadowing existing variable names
```cpp
double* host_buf = ...;  // May shadow an existing host_buf in the outer scope
```
**CORRECT:** Use the `lv_` prefix for all passthrough-local variables.

---

## Notes on Implementation

1. **Always declare `long long lv_N` for element counts:** Never compute the element count
   as `int N * int N` inline in an index expression. For a 2D grid of side N=16384,
   `N*N = 268,435,456` fits in `int`, but `(k+1)*N*N` at k=7 overflows `INT_MAX` and
   produces a negative array index → segfault. Always declare:
   ```cpp
   long long lv_N = (long long)dim1 * (long long)dim2;  // or just (long long)flat_count
   ```
   and use `lv_N` (not a re-cast `int`) in all loop bounds and index expressions.

2. **All passthrough blocks use standard library functions only:**
   - `<cstdio>` / `<stdio.h>` for printf
   - `<cstdlib>` / `<stdlib.h>` for malloc/free (only when a new host buffer is needed)
   - `<cuda_runtime.h>` for cudaMemcpy (already included in CUDA source)

2. **Block scope:** Wrap each passthrough block in `{ }` to limit local variable scope.

3. **Output order:** Always print the header line (`=== BEGIN ...`), then Buffer/Type/Elements,
   then Statistics, then Sample_first10, then Sample_last10 (if N > 10), then Sample_spread10
   (if N > 10), then the footer. Consistent ordering is required for the deterministic parser.

4. **HIP adaptation:** Replace `cudaMemcpy`/`cudaMemcpyDeviceToHost` with
   `hipMemcpy`/`hipMemcpyDeviceToHost`. All other printf logic is identical.

---
