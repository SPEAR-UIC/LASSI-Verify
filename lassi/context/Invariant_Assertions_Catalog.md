
---

# Invariant Assertion Catalog for GPU Code Instrumentation

## Critical Distinction: Type A vs Type B Assertions

### ✅ Type A: Invariant Assertions (REQUIRED - INSERT THESE)
Properties that remain true regardless of parallelization strategy or execution order:
- Input validity constraints
- Mathematical invariants (conservation laws, symmetry, physical laws)
- Output validity and correctness
- Numerical properties that must hold
- **These survive ANY refactoring, optimization, or parallelization change**

### ❌ Type B: Implementation-Specific Assertions (FORBIDDEN - NEVER INSERT THESE)
Properties tied to specific execution patterns:
- Thread execution order assumptions
- Intermediate state checks assuming serial or specific parallel execution
- Timing or performance metrics
- Thread/block/warp-specific checks, including thread block dimensions (`blockDim.x`, `blockDim.y`), grid dimensions, or warp size assumptions
- Algorithm implementation details that change with optimization, including number of CUDA streams, shared memory tile sizes, reduction strategies, and memory layout choices
- Hardware tuning parameters that control how the computation is mapped to hardware but do not affect what is computed

**YOU MUST INSERT ONLY TYPE A ASSERTIONS.**

### The One-Sentence Decision Test
> **Ask: "If this property were violated, would the code produce a wrong answer?"**
> - If YES → it is a semantic invariant (Type A) and may be an assertion candidate.
> - If NO → it is a hardware tuning or implementation parameter (Type B) and must never be asserted.

**Example:** A thread block of 16×16 vs 32×8 both compute the same result. Violating `blockDim.x == 16 && blockDim.y == 16` does NOT produce wrong output — it is a valid optimization target. Do not assert it. Tier 2 confirms numerical equivalence independently.

---

---

## ⚠️ SCOPE BOUNDARY: What Tier 1 Is and Is Not

**Tier 1 assertions check that the program ran correctly in a structural sense — not that the computed values are correct.**

### Tier 1 IS:
- Pointer and allocation validity (did memory get allocated?)
- Size and dimension constraints (are problem-size counts, array dimensions, or index bounds in range?)
- Numerical validity of inputs and outputs (are floats finite? no NaN or Inf?)
- Plausible output range (is the output value within the physically/mathematically possible range for this type?)
- Structural properties of data layouts (prefix sums non-decreasing, probabilities sum to 1, etc.)
- Algorithmic parameters that control correctness (convergence tolerance > 0, max iterations > 0, grid covers the full domain)
- Pre-kernel launch configuration: only check that device pointers are non-null and problem-size counts are positive — do NOT assert specific block/grid dimensions

### Tier 1 is NOT:
- **Checking whether the computed output values are numerically correct** — do not compare GPU output to any reference, expected, or re-computed value
- **Verifying that GPU and CPU produce the same result** — this is handled separately and is not a Tier 1 concern
- **Checking specific domain-correct values** — do not assert that a field equals a known boundary condition value, a known analytical solution, or any particular number
- **Re-running the computation to verify it** — do not add a serial CPU fallback and compare to GPU output

### The one-sentence rule:
> **Check that values exist, are finite, and are in a plausible range. Do NOT check that they equal anything specific.**

If your assertion contains a comparison to an expected numerical value (e.g., `== 0.0`, `== ustar`, `>= ref_result`, `!= 0` after knowing what the result should be), it is a correctness check and should be removed or replaced with a range/finiteness check.

### ⚠️ IMPORTANT EXCEPTION — Problem Specification Constants (Type A, not Type B):

**Problem-size parameters that are fixed in the source code ARE correct targets for exact-equality assertions.** These are NOT hardware tuning parameters — they define WHAT problem is being solved. Changing them does not optimize the computation; it computes an answer to a different problem entirely.

**How to identify a problem specification constant:**
- It appears as a `#define`, `const int`, a hardcoded literal, or a value read from a fixed command-line argument that sets the domain size, number of elements, grid dimensions, or problem extent (e.g., `N`, `M`, `nx`, `ny`, `num_particles`, `width`, `height`, `num_keys`, etc.)
- It controls the size of the primary allocation passed to `cudaMalloc` and to the kernel as the problem extent
- **Ask: if this constant changed, would the program compute the answer to a different problem?** If yes, it is a specification constant and should be asserted.

**Contrast with Type B (do NOT assert):** Block dimensions, number of streams, tile sizes, shared memory sizes, reduction strategies — these change HOW the computation is mapped to hardware, not WHAT is computed. A valid optimization may change any of these while computing the identical answer. Tier 2 handles numerical equivalence.

**What to assert:** Immediately after the problem-size variable is initialized (from its define, literal, or argument), assert it equals the value found in the source code. The expected value comes directly from reading the source — do not guess it. Pattern:
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
{
    // Source specifies <parameter_name> = <source_value> (hardcoded / fixed constant).
    // A refactored code that changes this value solves a different problem.
    bool _t1_pass = (<parameter_name> == <source_value>);
    printf("LASSI-VERIFY ASSERTIONS [problem_spec_<parameter_name>_unchanged]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

**This is a Layer 1 Universal Invariant** for any kernel where the problem size or data extent is specified as a fixed constant in the source. Read the source, find the constant's value, and assert it. If the kernel reads its size from command-line arguments at runtime (not hardcoded), assert that the argument is still being passed and is positive — but do not assert a specific value, since runtime arguments are legitimately variable.

---

## Assertion Selection Framework

When analyzing code, work through these layers in order. Stop when you have sufficient coverage — do not add assertions from deeper layers simply because they exist in the catalog.

### Layer 1: Universal Invariants (Always Applicable)
Apply to **all** GPU codes before selecting anything else:
- **1.1** Pointer validity (input and output pointers non-null)
- **1.2** Array dimension constraints (sizes positive and non-overflowing)
- **1.4** Numerical validity of inputs (NaN/Inf checks before H→D transfer)
- **6.3** Numerical validity of outputs (finite-value check after D→H transfer)

### Layer 2: Domain Invariants (Conditionally Applicable)
Apply when your analysis identifies the computational domain:
- **Linear Algebra / Solvers**: matrix symmetry (3.1), conditioning (6.4), convergence residual (6.5)
- **CFD / Physics simulations**: conservation laws (2.1, 2.3), positivity constraints (5.1, 5.2)
- **Molecular Dynamics**: particle count conservation (2.2), energy conservation (2.1), temperature bounds (5.5)
- **Iterative Solvers**: final convergence criterion (6.5) — check the residual after all iterations, not intermediate monotonicity
- **Stochastic / Monte Carlo**: probability conservation (2.5), non-negativity (5.2)
- **Transforms / Permutations**: element count conservation (2.2), norm conservation if unitary (2.4)

### Layer 3: Problem-Specific Invariants (Rarely Applicable)
Apply only when the specific problem structure clearly requires it:
- Sparsity pattern preservation (7.1) — only for sparse matrix operations
- Diagonal dominance (7.2) — only if the algorithm requires it and the original code enforces it
- Orthogonality (7.4) — only for orthogonalization algorithms
- Symmetry under permutation (3.2, 3.3, 3.4) — only when the problem has explicit symmetry

**When in doubt about any Layer 3 assertion, omit it.** A false positive assertion — one that fails in the original code due to an incorrect assumption — is worse than a missing assertion.

---

## GPU-Specific Considerations

### Host vs Device Code
- **Host code assertions:** Place assertions in host functions before kernel launches and after device-to-host transfers
- **LASSI-VERIFY assertion blocks are HOST-SIDE ONLY.** Place them in host functions only. Transfer device data to host first (D→H) for validation.
- **Preferred approach:** Validate invariants on **host side** after transferring results from device

### Memory Transfer Points (Critical)
Insert assertions at these strategic locations:
1. **Before host-to-device transfer:** Validate input data
2. **After device-to-host transfer:** Validate output data
3. **Before kernel launch:** Validate kernel parameters
4. **After kernel completion:** Validate results

### Example GPU Code Instrumentation Pattern:
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Input finite check before GPU transfer
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!isfinite(h_input[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [input_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<

cudaMemcpy(d_input, h_input, N * sizeof(double), cudaMemcpyHostToDevice);

kernel<<<blocks, threads>>>(d_input, d_output, N);
cudaDeviceSynchronize();

cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Output finite check after GPU transfer
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!isfinite(h_output[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [output_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - tier1_complete (sentinel — confirms all assertion blocks were reached)
{
    printf("LASSI-VERIFY ASSERTIONS [tier1_complete]: PASS\n"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Required Block Format

Every assertion you insert **must** use two block types: **Type 1** (always-print check, one per observation point) and **Type 2** (completion sentinel, one per function). Together they tell the checker: "instrumentation ran and here are the results."

The pipeline parses `LASSI-VERIFY ASSERTIONS [label]: PASS` and `LASSI-VERIFY ASSERTIONS [label]: FAIL` lines, plus the `[tier1_complete]: PASS` sentinel. Any other format (including bare `assert()` calls) will be silently ignored by `check_tier1_assertions()`.

### Rules
- Label MUST be **globally unique within the file** — no two assertion blocks may share the same label string. The parser counts each `PASS`/`FAIL` line as a separate assertion instance; duplicate labels make the failure count misleading and inflate `tier1_assertions_failed`. For code with **multiple kernel launches** checking the same invariant, use a numeric or semantic suffix: `kernel_params_valid_k1`, `kernel_params_valid_k2`, `kernel_params_valid_k3` — not `kernel_params_valid` repeated. (e.g., `input_ptrs_valid`, `array_dims_valid`, `input_finite`, `output_finite`, `matrix_sym`, `conservation_check`)
- Label MUST be a **literal string with no `%` format specifiers** — never use `%s`, `%d`, `%f`, or any `%` character in the label. A label like `kernel_valid_%s` causes the `%s` to consume a printf argument, producing unpredictable text in stdout that the parser cannot match. Use a fixed literal suffix instead: `kernel_valid_k1`, `kernel_valid_k2`.
- `#include <cstdio>` — not `<assert.h>`
- **Type 1 blocks always print exactly one line** (`PASS` or `FAIL`) — this line is essential pipeline output and must never be removed or hidden inside a conditional
- **NEVER place assertion blocks inside solver loops or compute loops** — place them BEFORE and AFTER loops at observation points (function entry, pre-H→D transfer, post-D→H transfer, function exit). One print per block per program execution is the target — not one print per loop iteration.
- **Type 2 sentinel (`[tier1_complete]`) must appear ONCE, AFTER all Type 1 blocks, OUTSIDE any loops**
- LASSI-VERIFY assertion blocks are HOST-SIDE ONLY — never in `__global__` kernel functions
- **NEVER modify `__global__` kernel function signatures or bodies.** Do not add, remove, or rename parameters. Do not rewrite, simplify, or replace the kernel body in any way. If you need data from inside the kernel, add a D→H copy in host code after the kernel completes and validate there.
- Code runs to completion — never use `assert()`, `exit()`, or `abort()`

### Pattern 1: Single-condition check
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - [Brief description of what is being checked]
{
    bool _t1_pass = (your_condition_here);
    printf("LASSI-VERIFY ASSERTIONS [unique_label]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### Pattern 2: Loop/array check (accumulate pattern)
The loop must be a standard 3-part `for` with exactly one semicolon between init, condition, and increment. The boolean accumulator `_t1_pass` is **not** part of the for-header — it lives in the loop body.
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - [Brief description]
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {           // ← 3 parts: init ; condition ; increment
        if (!(per_element_condition_on_i)) { _t1_pass = false; break; }  // ← flag set in body
    }
    printf("LASSI-VERIFY ASSERTIONS [unique_label]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```
**NEVER write:** `for (int i = 0; i < N; _t1_pass; i++)` — placing `_t1_pass` as the for-header increment expression is a syntax error in C/C++ (the increment position does not accept a boolean expression as a statement in this context and produces a compiler error).

### Pattern 3: Before/after conservation check
Declare capture variable at function scope OUTSIDE any block, before computation:
```cpp
// Declare capture variable at function scope, before computation:
double _t1_initial_sum = 0.0;
for (int i = 0; i < N; i++) _t1_initial_sum += array[i];

// ... kernel execution (no LASSI-VERIFY block here) ...

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Verify conservation after computation
{
    double _t1_final_sum = 0.0;
    for (int i = 0; i < N; i++) _t1_final_sum += array[i];
    double _t1_denom = (fabs(_t1_initial_sum) > 1e-300) ? fabs(_t1_initial_sum) : 1.0;
    bool _t1_pass = (fabs(_t1_final_sum - _t1_initial_sum) / _t1_denom < 1e-6);
    printf("LASSI-VERIFY ASSERTIONS [conservation_check]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### Pattern 4: Completion sentinel (REQUIRED — one per function, after all Type 1 blocks)
Place this **once**, AFTER all Type 1 assertion blocks and OUTSIDE any loops. It confirms to the
checker that all assertion blocks were reached and no failures were detected.
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - tier1_complete (sentinel — confirms all assertion blocks were reached)
{
    printf("LASSI-VERIFY ASSERTIONS [tier1_complete]: PASS\n"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Your Instructions

### Step 1: Analyze the Code
Identify:
- Kernel functions and host functions
- Input/output data structures
- Memory transfer points (cudaMemcpy, hipMemcpy)
- Mathematical operations performed
- Physical domain and expected invariants

### Step 2: Select Minimal Relevant Assertions
- Choose from the catalog below
- Focus on properties that validate correctness
- Prefer host-side assertions over device-side
- Consider computational cost (avoid expensive checks in hot loops)

### Step 3: Insert Assertions at Strategic Points
- **Host function start:** Input validation
- **Before device transfer:** Data validity
- **After device transfer:** Output validation
- **Host function end:** Final correctness checks

### Step 4: Mark All Insertions
```cpp
// LASSI_INSTRUMENTATION - [Brief description of what you're checking]
[assertion code here]
```

### Step 5: Preserve Original Code
- Insert assertions around original code
- Do NOT modify the original algorithm
- Do NOT change kernel implementations — this means **zero changes** to `__global__` function signatures (parameters), bodies (logic), or call sites. Any modification to a kernel during assertion insertion corrupts the ground-truth spec and invalidates all downstream Tier 2 comparisons.
- Only ADD assertion checks in host code

---

## ❌ ANTI-PATTERNS: What NOT to Do

These are examples of **INCORRECT** assertion insertions that violate the invariant-only rule:

### ❌ WRONG: Hardware Tuning Parameters (Thread Block Dimensions, Stream Counts, Tile Sizes)
```cpp
// BAD — asserts specific thread block geometry sourced from original code
dim3 threads(blockDim.x, blockDim.y);
// WRONG: 32×8 blocks are a valid optimization of 16×16 blocks; both produce the same result
{
    bool _t1_pass = (blockDim.x == 16 && blockDim.y == 16);
    printf("LASSI-VERIFY ASSERTIONS [block_dims_valid]: %s\n", _t1_pass ? "PASS" : "FAIL");
}

// ALSO BAD — asserts a specific stream count
{
    bool _t1_pass = (num_streams == 4);  // WRONG: 1, 2, or 8 streams compute the same result
    printf("LASSI-VERIFY ASSERTIONS [stream_count_valid]: %s\n", _t1_pass ? "PASS" : "FAIL");
}
```
**WHY THIS IS WRONG:** Thread block dimensions, stream counts, shared memory tile sizes, and reduction strategies are hardware mapping decisions — they control how the GPU schedules work, not what is computed. Changing them is the primary mechanism by which an LLM optimization can improve energy efficiency or runtime. An assertion on these values directly forbids the optimization the framework is designed to find.

Applying the decision test: if `blockDim.x` changes from 16 to 32, does the code compute a wrong answer? No — Tier 2 confirms numerical equivalence independently. Therefore it is a Type B parameter and must never be asserted.

**CORRECT:** Assert only that the problem-size count is positive and the device pointer is non-null:
```cpp
// GOOD — checks only that the kernel can legally launch (problem-size constraint)
{
    bool _t1_pass = (d_input != nullptr && N > 0);
    printf("LASSI-VERIFY ASSERTIONS [launch_preconditions]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
```

### ❌ WRONG: Thread Execution Order Assumptions
```cpp
// BAD - Assumes specific thread execution order
__global__ void bad_kernel(double* data, int N) {
    int tid = threadIdx.x;
    // WRONG: This assumes thread 0 executes before thread 1
    if (tid > 0) {
        assert(data[tid-1] < data[tid]); // IMPLEMENTATION-SPECIFIC
    }
}
```

### ❌ WRONG: Device-Side Assertions for Expensive Checks
```cpp
// BAD - Expensive check in device code
__global__ void bad_kernel(double* matrix, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // WRONG: O(N) check per thread = O(N²) total cost
        for (int j = 0; j < N; j++) {
            assert(isfinite(matrix[i*N + j]));
        }
    }
}
```
**CORRECT:** Validate matrix on host after transfer

### ❌ WRONG: Intermediate Algorithm State
```cpp
// BAD - Checks implementation-specific intermediate state
for (int iter = 0; iter < max_iter; iter++) {
    jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x, d_x_new, N);

    // WRONG: Monotonicity assumption won't hold for all parallel strategies
    cudaMemcpy(h_x_new, d_x_new, N*sizeof(double), cudaMemcpyDeviceToHost);
    double new_error = compute_error(h_x_new, N);
    assert(new_error < prev_error); // IMPLEMENTATION-SPECIFIC
    prev_error = new_error;
}
```
**CORRECT:** Only check final convergence after all iterations

### ❌ WRONG: Performance Metrics
```cpp
// BAD - Timing assertions are implementation-specific
auto start = std::chrono::high_resolution_clock::now();
kernel<<<blocks, threads>>>(d_data, N);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// WRONG: Performance will vary with refactoring
assert(duration.count() < 100); // IMPLEMENTATION-SPECIFIC
```

### ❌ WRONG: Shared Memory Assumptions
```cpp
// BAD - Shared memory state during execution
__global__ void bad_kernel(double* data, int N) {
    __shared__ double shared_data[256];
    int tid = threadIdx.x;

    shared_data[tid] = data[tid];
    __syncthreads();

    // WRONG: Assumes specific shared memory layout
    assert(shared_data[0] == data[0]); // IMPLEMENTATION-SPECIFIC
}
```

### ❌ WRONG: Modifying Kernel Signatures or Bodies
```cpp
// BAD — adds a new outputBuffer parameter to capture debug data, then rewrites
// the kernel body to accommodate it, discarding the original computation.
__global__ void pathfinder(const int* gpuWall, const int* gpuSrc,
                            int* gpuResult, int* outputBuffer,  // ← NEW PARAM ADDED
                            const int iteration, const int theHalo,
                            const int borderCols, const int cols, const int t) {
    (void)gpuWall;      // ← All original DP parameters discarded
    (void)iteration;
    (void)theHalo;
    (void)borderCols;
    (void)t;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < cols) {
        gpuResult[global_idx] = gpuSrc[global_idx];  // ← Entire DP replaced with copy
    }
    // ... debug buffer write added here ...
}
// HOST CODE: kernel call site must also be updated to pass the new parameter,
// which forces a full rewrite and commonly causes the LLM to simplify the host loop too.
```
**WHY THIS IS CATASTROPHIC:** When the kernel signature changes, the LLM must rewrite both the kernel and its call site(s). During that rewrite, it frequently simplifies or discards the core computation. The Tier 1 instrumented SOURCE_verify then produces wrong output (e.g., initial data values instead of computed DP results), poisoning the Tier 2 spec. Every generated code that computes the correct answer will then FAIL Tier 2 because it disagrees with the corrupted spec — the pipeline cannot make forward progress. This is an undetectable silent corruption: all Tier 1 assertions still PASS because the stub produces trivially valid-looking values.

**CORRECT:** Leave the kernel signature and body completely untouched. If you need to inspect a value that is computed inside the kernel, copy the relevant output buffer from device to host after the kernel completes and validate it in a host-side assertion block.

### ❌ WRONG: Comparing GPU Output to a CPU Reference or Expected Value
```cpp
// BAD — computes a CPU reference and asserts GPU output matches it
uint64_t cpu_hash = murmurhash3_cpu(key, length);
cudaMemcpy(h_out, d_out, numKeys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
{
    bool _t1_pass = (h_out[0] == cpu_hash);  // WRONG: correctness check, not structural
    printf("LASSI-VERIFY ASSERTIONS [gpu_cpu_hash_match]: %s\n", _t1_pass ? "PASS" : "FAIL");
}

// ALSO BAD — comparing host buffer to device buffer to "verify" the copy
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != expected_values[i]) { _t1_pass = false; break; }  // WRONG
    }
    printf("LASSI-VERIFY ASSERTIONS [output_matches_ref]: %s\n", _t1_pass ? "PASS" : "FAIL");
}
```
**WHY THIS IS WRONG:** Computing a CPU reference adds code that can itself be incorrect, producing FAIL on valid source code. The assertion is fragile: it fails if the CPU reference has a bug, if the data types differ, or if floating-point rounding differs between CPU and GPU. Verifying that computed output values are numerically correct is out of scope for Tier 1.

**CORRECT:** Check only that the output pointer is non-null, the count is correct, and values are finite (for floats) or within the representable range of their type (for integers). Do not check specific values.
```cpp
cudaMemcpy(h_out, d_out, numKeys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate output pointer and that kernel produced non-trivial output
{
    bool _t1_pass = (h_out != NULL && numKeys > 0);
    printf("LASSI-VERIFY ASSERTIONS [output_ptr_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### ❌ WRONG: Asserting on a Host Buffer Before the Device-to-Host Copy
```cpp
kernel<<<blocks, threads>>>(d_input, d_out, N);
cudaDeviceSynchronize();

// WRONG: h_out still has its initial value (zeros / garbage) here — the copy hasn't happened
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
{
    bool _t1_pass = (h_out[0] != 0);   // WRONG: always FAIL because h_out is still zeroed
    printf("LASSI-VERIFY ASSERTIONS [output_nonzero]: %s\n", _t1_pass ? "PASS" : "FAIL");
}
// <<< END LASSI-VERIFY ASSERTIONS <<<

cudaMemcpy(h_out, d_out, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // copy happens AFTER
```
**WHY THIS IS WRONG:** The host buffer `h_out` is only populated after `cudaMemcpy`. Asserting on it before the copy observes its initial allocation value (typically all zeros), causing the assertion to always FAIL on valid source code.

**CORRECT:** Place every host-buffer assertion **after** the corresponding `cudaMemcpy` call, not before.
```cpp
kernel<<<blocks, threads>>>(d_input, d_out, N);
cudaDeviceSynchronize();
cudaMemcpy(h_out, d_out, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // copy first

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate output pointer after transfer
{
    bool _t1_pass = (h_out != NULL && N > 0);
    printf("LASSI-VERIFY ASSERTIONS [output_valid_post_copy]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### ✅ CORRECT Alternatives for Each Case:

**Case 1 (Order):** Check final sorted property on host
```cpp
// GOOD - Check invariant property after kernel completes
cudaMemcpy(h_data, d_data, N*sizeof(double), cudaMemcpyDeviceToHost);
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Verify sorted (non-decreasing) order after kernel
{
    bool _t1_pass = true;
    for (int i = 0; i < N-1; i++) {
        if (!(h_data[i] <= h_data[i+1])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [sorted_order]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

**Case 2 (Expensive):** Validate on host
```cpp
// GOOD - Single O(N²) check on host
cudaMemcpy(h_matrix, d_matrix, N*N*sizeof(double), cudaMemcpyDeviceToHost);
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Verify matrix values are finite after transfer
{
    bool _t1_pass = true;
    for (int i = 0; i < N*N; i++) {
        if (!isfinite(h_matrix[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [matrix_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

**Case 3 (Intermediate):** Check final convergence only
```cpp
// GOOD - Check final result satisfies convergence
cudaMemcpy(h_x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
double residual = compute_residual(h_A, h_b, h_x, N);
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Verify final residual satisfies convergence tolerance
{
    bool _t1_pass = (residual < tol);
    printf("LASSI-VERIFY ASSERTIONS [convergence_residual]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Assertion Categories

## Category 1: Input Validity (7 patterns)

### 1.1 Pointer Validity
**Use when:** Functions accept pointer arguments
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate input pointers
{
    bool _t1_pass = (ptr != NULL);
    printf("LASSI-VERIFY ASSERTIONS [input_ptrs_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 1.2 Array Size Constraints
**Use when:** Functions accept array dimensions
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate array dimensions
{
    bool _t1_pass = (N > 0 && M > 0);
    printf("LASSI-VERIFY ASSERTIONS [array_dims_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 1.3 Value Range Constraints
**Use when:** Inputs have valid ranges (probabilities, angles, indices)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate input value ranges
{
    bool _t1_pass = (probability >= 0.0 && probability <= 1.0);
    printf("LASSI-VERIFY ASSERTIONS [input_range_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 1.4 Numerical Validity (Initialization)
**Use when:** Checking input data for NaN/Inf
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate numerical inputs (no NaN or Inf)
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!isfinite(array[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [input_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 1.5 Buffer Size Adequacy
**Use when:** Output buffers must be large enough
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate buffer sizes
{
    bool _t1_pass = (output_size >= required_size);
    printf("LASSI-VERIFY ASSERTIONS [buffer_size_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 1.6 Configuration Validity
**Use when:** Algorithmic parameters control correctness (convergence tolerance, max iterations, problem counts).
**Do NOT use for:** Thread block dimensions, stream counts, tile sizes, or any other hardware tuning parameter — these are Type B (see above). Changing them does not produce wrong output and must not be asserted.
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate algorithmic configuration (tolerance > 0, max_iter > 0)
{
    bool _t1_pass = (tolerance > 0.0 && tolerance < 1.0 && max_iter > 0);
    printf("LASSI-VERIFY ASSERTIONS [algo_config_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 1.7 Data Structure Integrity
**Use when:** Complex data structures are passed as input
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate structure integrity
{
    bool _t1_pass = (matrix->rows == matrix->cols);
    printf("LASSI-VERIFY ASSERTIONS [struct_integrity]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Category 2: Conservation Laws (5 patterns)

### 2.1 Mass/Energy Conservation
**Use when:** Physical quantities should be conserved
```cpp
// Declare capture variable at function scope, before computation:
double _t1_initial_total = compute_total(array, N);

// ... computation ...

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Mass/energy conservation check
{
    double _t1_final_total = compute_total(array, N);
    double _t1_denom = (fabs(_t1_initial_total) > 1e-300) ? fabs(_t1_initial_total) : 1.0;
    bool _t1_pass = (fabs(_t1_final_total - _t1_initial_total) / _t1_denom < 1e-6);
    printf("LASSI-VERIFY ASSERTIONS [mass_energy_conserved]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 2.2 Element Count Conservation
**Use when:** Number of elements should not change (transforms, permutations)
```cpp
// Declare capture variable at function scope, before computation:
int _t1_initial_count = count_elements(input, N);

// ... computation ...

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Element count conservation check
{
    int _t1_final_count = count_elements(output, M);
    bool _t1_pass = (_t1_initial_count == _t1_final_count);
    printf("LASSI-VERIFY ASSERTIONS [element_count_conserved]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 2.3 Sum Conservation
**Use when:** Total sum should remain constant
```cpp
// Declare capture variable at function scope, before computation:
double _t1_initial_sum = 0.0;
for (int i = 0; i < N; i++) _t1_initial_sum += array[i];

// ... computation ...

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Sum conservation check
{
    double _t1_final_sum = 0.0;
    for (int i = 0; i < N; i++) _t1_final_sum += array[i];
    double _t1_denom = (fabs(_t1_initial_sum) > 1e-300) ? fabs(_t1_initial_sum) : 1.0;
    bool _t1_pass = (fabs(_t1_final_sum - _t1_initial_sum) / _t1_denom < 1e-6);
    printf("LASSI-VERIFY ASSERTIONS [sum_conserved]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 2.4 Norm Conservation
**Use when:** Vector/matrix norms should be preserved (unitary transforms)
```cpp
// Declare capture variable at function scope, before computation:
double _t1_initial_norm = compute_l2_norm(vector, N);

// ... computation ...

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - L2 norm conservation check
{
    double _t1_final_norm = compute_l2_norm(vector, N);
    bool _t1_pass = (fabs(_t1_final_norm - _t1_initial_norm) < 1e-6);
    printf("LASSI-VERIFY ASSERTIONS [norm_conserved]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 2.5 Probability Distribution Conservation
**Use when:** Working with probability distributions
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Probability distribution conservation check
{
    bool _t1_pass = true;
    double _t1_sum = 0.0;
    for (int i = 0; i < N; i++) {
        if (probs[i] < 0.0) { _t1_pass = false; break; }
        _t1_sum += probs[i];
    }
    if (_t1_pass) _t1_pass = (fabs(_t1_sum - 1.0) < 1e-6);
    printf("LASSI-VERIFY ASSERTIONS [prob_distribution_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Category 3: Symmetry Properties (4 patterns)

### 3.1 Matrix Symmetry
**Use when:** Operations should preserve symmetric matrices
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Matrix symmetry check
{
    bool _t1_pass = true;
    for (int i = 0; i < N && _t1_pass; i++) {
        for (int j = i + 1; j < N; j++) {
            if (fabs(matrix[i*N + j] - matrix[j*N + i]) >= 1e-10) { _t1_pass = false; break; }
        }
    }
    printf("LASSI-VERIFY ASSERTIONS [matrix_sym]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 3.2 Permutation Symmetry
**Use when:** Result should be invariant under permutation of symmetric inputs
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Commutativity / permutation invariance check
{
    // For commutative operations like addition or multiplication
    bool _t1_pass = (result_AB == result_BA);
    printf("LASSI-VERIFY ASSERTIONS [perm_sym]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 3.3 Reflection Symmetry
**Use when:** Physical system has reflection symmetry
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Reflection symmetry check
{
    bool _t1_pass = true;
    for (int i = 0; i < N/2; i++) {
        if (fabs(field[i] - field[N-1-i]) >= tolerance) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [reflection_sym]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 3.4 Rotational Symmetry
**Use when:** System has rotational invariance
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Rotational invariance check (simplified)
{
    // For 2D field with rotational symmetry
    // (actual implementation depends on coordinate system)
    double center_value = field[N/2][N/2];
    bool _t1_pass = true; // fill with radial checks as needed
    printf("LASSI-VERIFY ASSERTIONS [rotational_sym]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Category 4: Monotonicity (4 patterns)

### 4.1 Non-Decreasing Sequence
**Use when:** Output should be sorted or monotonically increasing
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Non-decreasing (sorted) sequence check
{
    bool _t1_pass = true;
    for (int i = 0; i < N - 1; i++) {
        if (!(array[i] <= array[i+1])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [nondecreasing_seq]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 4.2 Non-Increasing Sequence
**Use when:** Values should decrease (e.g., residuals, errors)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Non-increasing sequence check
{
    bool _t1_pass = true;
    for (int i = 0; i < N - 1; i++) {
        if (!(array[i] >= array[i+1])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [nonincreasing_seq]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 4.3 Strictly Decreasing Energy/Error
**Use when:** Iterative algorithms should reduce error or energy
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Convergence monotonicity check
{
    bool _t1_pass = (current_error < previous_error * (1.0 + 1e-10));
    printf("LASSI-VERIFY ASSERTIONS [error_decreasing]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 4.4 Bounded Monotonicity
**Use when:** Values increase/decrease but within bounds
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Bounded monotonic increase check
{
    bool _t1_pass = true;
    for (int i = 0; i < N - 1; i++) {
        if (!(array[i] <= array[i+1]) || !(array[i+1] - array[i] <= max_increment)) {
            _t1_pass = false; break;
        }
    }
    printf("LASSI-VERIFY ASSERTIONS [bounded_monotone]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Category 5: Physical Constraints (6 patterns)

### 5.1 Positivity
**Use when:** Physical quantities must be positive (temperature, density, energy)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Physical positivity constraint check
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!(temperature[i] > 0.0) || !(density[i] > 0.0)) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [positivity]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 5.2 Non-Negativity
**Use when:** Quantities cannot be negative (counts, concentrations, probabilities)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Non-negativity constraint check
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!(count[i] >= 0) || !(concentration[i] >= 0.0)) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [non_negativity]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 5.3 Upper Bounds
**Use when:** Physical quantities have maximum values
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Upper bound constraint check
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!(occupancy[i] <= 1.0) || !(efficiency[i] <= 1.0)) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [upper_bounds]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 5.4 Speed of Light Constraint
**Use when:** Velocities in relativistic simulations
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Relativistic speed constraint check
{
    const double C = 299792458.0; // Speed of light in m/s
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        double speed = sqrt(vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        if (!(speed <= C)) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [speed_of_light]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 5.5 Thermodynamic Constraints
**Use when:** Working with thermodynamic quantities
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Thermodynamic constraints check
{
    bool _t1_pass = (temperature > 0.0 && entropy_change >= 0.0 && pressure > 0.0);
    printf("LASSI-VERIFY ASSERTIONS [thermodynamic]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 5.6 Charge Conservation
**Use when:** Electromagnetic simulations
```cpp
// Declare capture variable at function scope, before computation:
double _t1_initial_charge = compute_total_charge(particles, N);

// ... computation ...

// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Charge conservation check
{
    double _t1_final_charge = compute_total_charge(particles, N);
    bool _t1_pass = (fabs(_t1_final_charge - _t1_initial_charge) < 1e-10);
    printf("LASSI-VERIFY ASSERTIONS [charge_conserved]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Category 6: Numerical Stability (5 patterns)

### 6.1 NaN Detection (Output)
**Use when:** Validating output for numerical errors
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - NaN detection in output
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (isnan(result[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [output_no_nan]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 6.2 Infinity Detection
**Use when:** Checking for overflow/divergence
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Infinity detection in output
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (isinf(result[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [output_no_inf]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 6.3 Finite Value Check (Combined)
**Use when:** Ensuring all output values are finite
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Finite value check on output
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!isfinite(result[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [output_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 6.4 Condition Number Bounds
**Use when:** Matrix operations sensitive to conditioning
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Conditioning check
{
    double condition_number = estimate_condition_number(matrix, N);
    bool _t1_pass = (condition_number < 1e12);
    printf("LASSI-VERIFY ASSERTIONS [condition_number]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 6.5 Relative Error Bounds
**Use when:** Comparing computed result to known or expected value
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Relative error bound check
{
    double _t1_rel_err = fabs(computed - expected) / fabs(expected);
    bool _t1_pass = (_t1_rel_err < tolerance);
    printf("LASSI-VERIFY ASSERTIONS [rel_error_bound]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Category 7: Structural Invariants (4 patterns)

### 7.1 Sparsity Pattern Preservation
**Use when:** Sparse matrix operations should preserve zero structure
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Sparsity pattern preservation check
{
    bool _t1_pass = true;
    for (int i = 0; i < N && _t1_pass; i++) {
        for (int j = 0; j < N; j++) {
            if (original_matrix[i*N + j] == 0.0 && fabs(result_matrix[i*N + j]) >= 1e-15) {
                _t1_pass = false; break;
            }
        }
    }
    printf("LASSI-VERIFY ASSERTIONS [sparsity_preserved]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 7.2 Diagonal Dominance
**Use when:** Certain numerical methods require diagonally dominant matrices
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Diagonal dominance check
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        double diag = fabs(matrix[i*N + i]);
        double row_sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) row_sum += fabs(matrix[i*N + j]);
        }
        if (!(diag >= row_sum)) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [diag_dominant]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 7.3 Positive Definiteness
**Use when:** Matrix should remain positive definite (covariance, Gram matrices)
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Positive definiteness check (necessary condition: positive diagonal)
{
    // Check all diagonal elements are positive (necessary condition)
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!(matrix[i*N + i] > 0.0)) { _t1_pass = false; break; }
    }
    // For full check, would need eigenvalue computation (expensive)
    printf("LASSI-VERIFY ASSERTIONS [pos_definite_diag]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### 7.4 Orthogonality
**Use when:** Vectors or matrices should be orthogonal
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Orthogonality check
{
    double _t1_dot = 0.0;
    for (int i = 0; i < N; i++) {
        _t1_dot += vector1[i] * vector2[i];
    }
    bool _t1_pass = (fabs(_t1_dot) < 1e-10);
    printf("LASSI-VERIFY ASSERTIONS [orthogonality]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

---

## Selection Guidelines for GPU Code

### DO Insert Assertions For:
- ✅ **Host-side input validation** (before cudaMemcpy H→D)
- ✅ **Host-side output validation** (after cudaMemcpy D→H)
- ✅ **Kernel launch parameter validation** (grid/block dimensions)
- ✅ **Mathematical invariants** (conservation, symmetry, physical laws)
- ✅ **Properties that survive ANY parallelization or optimization**

### DO NOT Insert Assertions For:
- ❌ Thread/block/warp execution order
- ❌ Shared memory state during kernel execution
- ❌ Thread synchronization points
- ❌ Performance metrics (kernel timing, memory bandwidth)
- ❌ Implementation-specific GPU optimizations
- ❌ Intermediate kernel states (unless copied to host)

### GPU-Specific Guidelines:

**Assertion Placement Priority:**
1. **Host-side (preferred):** All invariant checks on CPU before/after GPU operations
2. **Device-side (use sparingly):** Only for critical invariants that must be checked per-thread
   - Device assertions have significant performance cost
   - May cause kernel to fail silently or hang
   - Prefer copying data to host for validation

**Memory Transfer Validation:**
```cpp
// ALWAYS validate before H→D transfer
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate input data before host-to-device transfer
{
    bool _t1_pass = (h_data != nullptr);
    if (_t1_pass) {
        for (int i = 0; i < N; i++) {
            if (!isfinite(h_data[i])) { _t1_pass = false; break; }
        }
    }
    printf("LASSI-VERIFY ASSERTIONS [htod_input_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
cudaMemcpy(d_data, h_data, N*sizeof(double), cudaMemcpyHostToDevice);

// ALWAYS validate after D→H transfer
cudaMemcpy(h_result, d_result, N*sizeof(double), cudaMemcpyDeviceToHost);
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate output data after device-to-host transfer
{
    bool _t1_pass = true;
    for (int i = 0; i < N; i++) {
        if (!isfinite(h_result[i])) { _t1_pass = false; break; }
    }
    printf("LASSI-VERIFY ASSERTIONS [dtoh_output_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

**Kernel Parameter Validation:**
```cpp
// >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
// LASSI_INSTRUMENTATION - Validate kernel launch parameters
{
    bool _t1_pass = (threads > 0 && threads <= 1024 && blocks > 0);
    printf("LASSI-VERIFY ASSERTIONS [kernel_params_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
}
// <<< END LASSI-VERIFY ASSERTIONS <<<
```

### Cost Considerations:
- **Low cost:** Pointer checks, single value checks, kernel parameter validation
- **Medium cost:** O(N) checks on host after device-to-host transfer
- **High cost:** O(N²) matrix checks on host
- **Very high cost:** Device-side assertions (avoid if possible)

**For large-scale GPU codes:**
- Prefer low-medium cost host-side assertions
- Avoid expensive checks in performance-critical paths
- Consider sampling for very large arrays (check subset of elements)
- Use compile-time flags to disable assertions in production

---

## Domain-Specific Guidance

### Computational Fluid Dynamics (CFD)
Key invariants:
- Mass conservation
- Energy conservation
- Momentum conservation
- Positivity of density, pressure, temperature
- Speed limits (subsonic/supersonic bounds)

### Linear Algebra
Key invariants:
- Matrix symmetry preservation
- Orthogonality preservation
- Norm conservation (unitary transforms)
- Rank preservation (where applicable)
- Conditioning bounds

### Molecular Dynamics
Key invariants:
- Particle count conservation
- Energy conservation (microcanonical ensemble)
- Momentum conservation
- Positivity of inter-particle distances
- Temperature bounds

### Monte Carlo Simulations
Key invariants:
- Probability sum to 1
- Non-negativity of probabilities
- Detailed balance (where applicable)
- Acceptance ratio bounds

### Iterative Solvers
Key invariants:
- Residual decrease (monotonicity)
- Solution bounds
- Convergence criteria satisfaction
- Numerical stability

---

## Example: GPU Jacobi Solver Instrumentation

**INPUT CODE (before instrumentation):**
```cpp
__global__ void jacobi_kernel(double* A, double* b, double* x, double* x_new, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) sigma += A[i*N + j] * x[j];
        }
        x_new[i] = (b[i] - sigma) / A[i*N + i];
    }
}

void jacobi_solver(double* A, double* b, double* x, int N, int max_iter, double tol) {
    double *d_A, *d_b, *d_x, *d_x_new;

    cudaMalloc(&d_A, N*N*sizeof(double));
    cudaMalloc(&d_b, N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_x_new, N*sizeof(double));

    cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int iter = 0; iter < max_iter; iter++) {
        jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x, d_x_new, N);
        cudaDeviceSynchronize();
        std::swap(d_x, d_x_new);
    }

    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_x_new);
}
```

**YOUR OUTPUT (after instrumentation):**
```cpp
__global__ void jacobi_kernel(double* A, double* b, double* x, double* x_new, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; j++) {
            if (i != j) sigma += A[i*N + j] * x[j];
        }
        x_new[i] = (b[i] - sigma) / A[i*N + i];
    }
}

void jacobi_solver(double* A, double* b, double* x, int N, int max_iter, double tol) {
    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Input pointer validity and parameter bounds
    {
        bool _t1_pass = (A != nullptr && b != nullptr && x != nullptr
                         && N > 0 && max_iter > 0 && tol > 0.0 && tol < 1.0);
        printf("LASSI-VERIFY ASSERTIONS [input_ptrs_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Matrix A finite check
    {
        bool _t1_pass = true;
        for (int i = 0; i < N*N; i++) {
            if (!isfinite(A[i])) { _t1_pass = false; break; }
        }
        printf("LASSI-VERIFY ASSERTIONS [matrix_A_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Vector b finite check
    {
        bool _t1_pass = true;
        for (int i = 0; i < N; i++) {
            if (!isfinite(b[i])) { _t1_pass = false; break; }
        }
        printf("LASSI-VERIFY ASSERTIONS [vector_b_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Initial guess x finite check
    {
        bool _t1_pass = true;
        for (int i = 0; i < N; i++) {
            if (!isfinite(x[i])) { _t1_pass = false; break; }
        }
        printf("LASSI-VERIFY ASSERTIONS [init_guess_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Diagonal dominance check (Jacobi convergence requirement)
    {
        bool _t1_pass = true;
        for (int i = 0; i < N; i++) {
            if (!(fabs(A[i*N + i]) > 1e-14)) { _t1_pass = false; break; }
        }
        printf("LASSI-VERIFY ASSERTIONS [diag_valid]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // NOTE: Matrix symmetry is NOT inserted here.
    // Jacobi iteration does not require a symmetric matrix — many valid Jacobi problems
    // use non-symmetric diagonally-dominant systems. Inserting a symmetry check would be
    // a Type B (implementation-specific) assertion for this code. Only add symmetry
    // checks if the source code itself explicitly requires or documents symmetry.

    double *d_A, *d_b, *d_x, *d_x_new;

    cudaMalloc(&d_A, N*N*sizeof(double));
    cudaMalloc(&d_b, N*sizeof(double));
    cudaMalloc(&d_x, N*sizeof(double));
    cudaMalloc(&d_x_new, N*sizeof(double));

    cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    for (int iter = 0; iter < max_iter; iter++) {
        jacobi_kernel<<<blocks, threads>>>(d_A, d_b, d_x, d_x_new, N);
        cudaDeviceSynchronize();
        std::swap(d_x, d_x_new);
    }

    cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Output x finite check (host-side after device-to-host transfer)
    {
        bool _t1_pass = true;
        for (int i = 0; i < N; i++) {
            if (!isfinite(x[i])) { _t1_pass = false; break; }
        }
        printf("LASSI-VERIFY ASSERTIONS [output_x_finite]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - Residual check (validates solution satisfies convergence criterion)
    {
        double _t1_residual_norm = 0.0;
        for (int i = 0; i < N; i++) {
            double ax_i = 0.0;
            for (int j = 0; j < N; j++) {
                ax_i += A[i*N + j] * x[j];
            }
            double r_i = b[i] - ax_i;
            _t1_residual_norm += r_i * r_i;
        }
        _t1_residual_norm = sqrt(_t1_residual_norm);
        bool _t1_pass = (_t1_residual_norm < tol * 10.0);
        printf("LASSI-VERIFY ASSERTIONS [residual_check]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
    // LASSI_INSTRUMENTATION - tier1_complete (sentinel — confirms all assertion blocks were reached)
    {
        printf("LASSI-VERIFY ASSERTIONS [tier1_complete]: PASS\n"); fflush(stdout);
    }
    // <<< END LASSI-VERIFY ASSERTIONS <<<

    cudaFree(d_A); cudaFree(d_b); cudaFree(d_x); cudaFree(d_x_new);
}
```

**Key Points Demonstrated:**
1. ✅ Assertions placed on **host side** (not in kernel)
2. ✅ Validation **before** host-to-device transfer
3. ✅ Validation **after** device-to-host transfer
4. ✅ **FAIL-only printing** — no stdout flooding on clean runs; sentinel confirms instrumentation ran
5. ✅ All assertions use `<cstdio>` for printf() and fflush(); code runs to completion — no assert(), exit(), or abort()
6. ✅ Assertions check **transformation-invariant properties** only
7. ✅ Original kernel code **unchanged**
8. ✅ Clear `// LASSI_INSTRUMENTATION` markers
8. ✅ No symmetry assumption inserted — Jacobi does not require symmetric matrices

---

## Notes on Implementation

1. **All assertions use standard library functions only:**
   - `<cstdio>` for printf() and fflush()
   - `<math.h>` for isnan(), isinf(), isfinite(), fabs(), sqrt()
   - No custom LASSI library required
   - Code runs to completion — never use assert(), exit(), or abort(). The pipeline checks all assertions after execution and provides the LLM with a complete list of failures.

2. **Assertions marked with `// LASSI_INSTRUMENTATION`:**
   - This marker tells the code generation LLM to preserve these sections
   - Enables automated verification that assertions weren't modified

3. **Rationale documentation:**
   - Each assertion block should include a brief comment explaining what invariant is being checked
   - This aids in debugging if assertions fail

4. **Tolerance values:**
   - Use appropriate tolerances for floating-point comparisons
   - Typical values: 1e-6 to 1e-10 depending on numerical precision requirements
   - Relative tolerances preferred for conservation laws

---
