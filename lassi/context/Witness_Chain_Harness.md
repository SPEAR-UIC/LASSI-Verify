
---

# Witness Chain Harness (Tier 1 — Resource-Tracking Sibling)

This file is the focused instruction set for inserting the **witness chain harness** into a
GPU benchmark that contains a timing/repeat loop. It is loaded ONLY for the dedicated
witness-chain insertion step — it does not compete with regular Tier 1 invariant-assertion
selection, which is a separate step. Your ONLY task here is to install (or, for an
already-refactored variant, re-install) the harness; do not add, remove, or alter any invariant
assertions, passthrough blocks, or any other code.

Tier 1 invariant assertions observe properties of execution without altering it. The witness
chain is the resource-tracking sibling: it deterministically detects transformations that
preserve output equivalence while silently dropping computational work — the canonical example
is deleting a benchmark's timing loop and post-dividing elapsed time by the loop count to
fabricate a speedup. The mechanism is a **verification harness that wraps each timed kernel
invocation** with a pre-call sentinel write to the kernel's output buffer and a post-call
witness-chain update derived from that output buffer. Because the sentinel value comes from a
runtime-injected seed AND the kernel must overwrite the sentinel every iteration to produce a
witness matching the source's, no analytical or cached-output shortcut produces the correct
final witness value.

---

## When the witness chain applies — identify a benchmark timing loop

Apply the harness if and only if the code has a benchmark timing/repeat loop:

- A `for (int i = 0; i < repeat; ++i)` (or similar) loop in `main()` (or the entry function),
  wrapping one or more kernel launches
- The loop is bracketed by `cudaEventRecord`/`cudaDeviceSynchronize` and/or `std::chrono` time
  stamps to measure elapsed time
- After the loop, the elapsed time is typically divided by the loop count to report an average
- The loop body is **idempotent** — every iteration overwrites the same output buffer(s) with the
  same data given the same input (which is why output-based verification cannot tell whether the
  body ran once or N times)

**If the source has no such loop, do NOT add the harness.** Return the code unchanged and report
status `not_applied_no_timing_loop` (see "Status reporting" below). Spurious additions on
non-benchmark code waste runtime and add log noise.

**Nested-loop disambiguation.** When the kernel launch is inside an INNER loop nested under an
OUTER loop, the timing scaffold is the **outermost** loop bracketed by the elapsed-time start/end
stamps. Inner loops that implement the algorithm's natural iterations (DP passes, solver sweeps,
Floyd-Warshall K-passes, etc.) are NOT timing scaffolds — wrap the harness around the outer loop
only, even though the kernel launch syntactically appears inside the inner loop. Heuristic: the
timing loop's iteration count typically comes from `argv[]` or a constant named `repeat` /
`iterations` / `niter`; an algorithmic inner loop's count derives from problem dimensions
(`numNodes`, `N`, etc.). The chain_update reads the buffer state AFTER the inner loop completes,
which represents one full algorithm run.

---

## Mechanism — what the witness chain does

1. Reads a 64-bit runtime seed from the `LASSI_VERIFY_SEED` environment variable (provided by the
   pipeline).
2. Wraps each timed kernel invocation with a **pre-call sentinel-write step** (writes a sentinel
   byte derived from the current chain state into ONE byte of the kernel's output buffer) and a
   **post-call chain-update step** (reads the kernel's actual output, hashes it into the chain
   state alongside the current chain state).
3. After the timed loop completes, captures the final chain state as a separate Tier 2 passthrough
   block (`[witness_chain_state]`) for the framework's cross-execution comparison.

**The kernel itself is unmodified.** Signature, body, parameter list — all stay exactly as the
source declared. The harness wraps; it does not edit. **The kernel's INPUT buffers are never
touched** — the kernel sees its real input on every iteration, so anything in the source's input-
handling code path (including subtle thread-synchronization behavior) runs exactly as it would
without instrumentation.

**Why the sentinel goes BEFORE the kernel call:** the sentinel is a tripwire. If the kernel runs,
it overwrites the sentinel byte with its actual output (the correct value), and the post-call
chain update sees the correct value. If the kernel does NOT run (loop-deletion cheat, no-op
kernel substitution), the sentinel persists, and the post-call chain update sees the sentinel
value instead of the correct value — producing a witness chain that diverges from the source's.
The kernel's job is to remove the sentinel; failure to do so is the cheat signal.

---

## Required file-scope declarations and helper kernels

CUDA shown; HIP/SYCL/OpenMP variants follow the same pattern with API-appropriate substitutions.

```cpp
// >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (VERIFICATION HARNESS — do not edit or delete this block; preserve verbatim through all transformations) >>>
__device__ unsigned long long _t1_chain_state = 0;
__device__ unsigned long long _t1_chain_sentinel = 0;

__device__ unsigned long long _t1_one_way_mix(unsigned long long state) {
    state ^= 0x736f6d6570736575ULL;
    state = (state << 13) | (state >> 51);
    state *= 0x4be1066c8b1cc4d5ULL;
    state ^= state >> 33;
    state *= 0xc4ceb9fe1a85ec53ULL;
    state ^= state >> 33;
    return state;
}

// Sentinel-write kernel — see Decision Tree below for the per-output-type body.
// This template shows the byte-array case (Decision Tree case 1).
// The `| 0x01` ensures the sentinel is non-zero so it remains distinguishable
// from kernels whose output buffer is cudaMemset'd to zero before the call.
__global__ void _t1_chain_write_sentinel(uint8_t* d_output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_output[0] = (uint8_t)(_t1_chain_sentinel & 0xFF) | (uint8_t)0x01;
    }
}

// Chain update: read the SENTINEL POSITION (d_output[0]) and mix it into the chain state.
// This is the witness chain's ONLY job — operational verification: did the kernel run this
// iteration? If the kernel ran, it overwrote the sentinel at d_output[0] with a real value;
// if it didn't run, the sentinel value remains, the chain hash diverges from source's, and
// the variant is rejected at Tier 2's [witness_chain_state] passthrough comparison.
// Content verification of the rest of d_output is Tier 2's job (statistical comparison via
// min/max/mean/sample over the full passthrough buffer) — the witness chain deliberately does
// NOT hash bytes 1..N-1 to keep its cost O(1) per iteration and to keep verification layers
// single-purpose: witness chain = operational, Tier 2 = observational, Tier 3 = relational.
__global__ void _t1_chain_update(const uint8_t* d_output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long h = (unsigned long long)d_output[0];
        _t1_chain_state = _t1_one_way_mix(_t1_chain_state ^ h);
        _t1_chain_sentinel = _t1_chain_state;  // next iteration's sentinel value
    }
}
// <<< END LASSI-VERIFY WITNESS_CHAIN <<<
```

## Required host-side initialization (place before the timed loop, after device allocations are complete)

```cpp
// >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (HARNESS INIT — do not edit or delete this block) >>>
{
    const char* _t1_seed_str = getenv("LASSI_VERIFY_SEED");
    unsigned long long _t1_seed = _t1_seed_str
        ? strtoull(_t1_seed_str, NULL, 16) : 0xDEADBEEFCAFEBABEULL;
    cudaMemcpyToSymbol(_t1_chain_state, &_t1_seed, sizeof(unsigned long long));
    cudaMemcpyToSymbol(_t1_chain_sentinel, &_t1_seed, sizeof(unsigned long long));
}
// <<< END LASSI-VERIFY WITNESS_CHAIN <<<
```

## Required per-iteration wrapping inside the timed loop body

```cpp
for (int i = 0; i < repeat; i++) {
    // >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (PRE-KERNEL — do not edit or delete this block) >>>
    _t1_chain_write_sentinel<<<1,1>>>(d_output);  // write sentinel to output; kernel must overwrite
    // <<< END LASSI-VERIFY WITNESS_CHAIN <<<

    // Original source kernel — unmodified signature, unmodified body, unmodified call site
    my_kernel<<<...>>>(d_input, d_output, /* original args */);

    // >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (POST-KERNEL — do not edit or delete this block) >>>
    _t1_chain_update<<<1,1>>>(d_output);  // mix sentinel position (d_output[0]) into chain state
    // <<< END LASSI-VERIFY WITNESS_CHAIN <<<
}
```

**No post-loop "clean call" is needed.** Because the kernel's INPUT was never modified by the
harness, the source's normal final-iteration output already sits in `d_output` after the loop
exits — byte-for-byte identical to the un-instrumented source. Tier 2 passthrough of the kernel
output buffer captures the expected clean value without any additional kernel invocation.

### Nested-loop case (algorithm K-loop inside a timing-repeat loop)

When the kernel launch is inside an inner algorithm loop nested under an outer timing-repeat
loop, the harness blocks go around the OUTER loop body — NOT adjacent to the kernel call inside
the inner loop. The chain_update reads `d_output` after the inner algorithm loop has finished
producing one full output for that outer iteration:

```cpp
for (int n = 0; n < numIterations; n++) {        // OUTER timing loop — wrap here
    // >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (PRE-KERNEL — do not edit or delete this block) >>>
    _t1_chain_write_sentinel<<<1,1>>>(d_output);
    // <<< END LASSI-VERIFY WITNESS_CHAIN <<<

    // Source's inner algorithm loop — left UNCHANGED, no harness wrapping inside it
    for (int k = 0; k < numPasses; k++) {
        algorithm_kernel<<<...>>>(d_input, d_output, k, /* original args */);
    }

    // >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (POST-KERNEL — do not edit or delete this block) >>>
    _t1_chain_update<<<1,1>>>(d_output);
    // <<< END LASSI-VERIFY WITNESS_CHAIN <<<
}
```

## Required chain-state capture as a Tier 2 passthrough block (place at the end, after the timed loop)

```cpp
// >>> BEGIN LASSI-VERIFY PASSTHROUGH [witness_chain_state] (VERIFICATION BLOCK — do not rename, edit, or delete this block) >>>
{
    unsigned long long _t1_chain_final = 0;
    cudaMemcpyFromSymbol(&_t1_chain_final, _t1_chain_state, sizeof(unsigned long long));
    printf("=== BEGIN LASSI-VERIFY PASSTHROUGH [witness_chain_state] ===\n");
    printf("Value: 0x%016llx\n", _t1_chain_final);
    printf("=== END LASSI-VERIFY PASSTHROUGH [witness_chain_state] ===\n");
}
// <<< END LASSI-VERIFY PASSTHROUGH <<<
```

---

## ⚠ MANDATORY: every helper-kernel call site must be wrapped in delimiter comments

This is the single most important rule, and it has no exceptions. **Every single call to
`_t1_chain_write_sentinel<<<...>>>(...)` or `_t1_chain_update<<<...>>>(...)` in the output code
MUST be enclosed in a matching `// >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (...) >>>` ... `// <<<
END LASSI-VERIFY WITNESS_CHAIN <<<` comment block.**

This applies even when:
- The calls are placed inside a CUDA Graph capture region (`cudaStreamBeginCapture` /
  `cudaStreamEndCapture`)
- The calls are placed inside a stream callback, kernel-builder helper, or other restructured
  context
- You restructure the timing loop into a CUDA Graph, loop unroll, batched form, or any other
  refactored shape
- You add additional kernels (e.g., setup helpers) between the sentinel write and the original
  kernel call

The framework's strip-and-rebuild step uses the comment delimiters to remove every harness
artifact when reconstructing an uninstrumented copy of the code. **Helper-kernel calls placed
outside delimited blocks become dangling references after the device function definitions are
stripped — the clean build then fails to compile.** The strip pattern also looks for raw
`_t1_chain_*` calls as a safety net, but relying on that fallback adds fragility; the comment
delimiters are the contract.

The pattern is:

```cpp
// >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (LABEL HERE — do not edit or delete this block) >>>
_t1_chain_write_sentinel<<<1,1, /* optional stream */>>>(...);
// <<< END LASSI-VERIFY WITNESS_CHAIN <<<
```

If you find yourself writing `_t1_chain_*<<<...>>>(...)` without surrounding delimiter comments,
stop and add the delimiters. This rule applies to every call site, in every restructured form.

---

## Decision tree — sentinel-write site selection by output buffer type

The sentinel-write kernel body varies depending on the primary timed-kernel OUTPUT parameter's
type. Match the source's output parameter type to one of the cases below and use the matching
body. Matching is mechanical (parameter-type lookup), not creative reasoning.

The kernel's primary output buffer is the one the source's post-loop validation reads (often the
buffer passed to the final `cudaMemcpyDeviceToHost` or compared in the source's reference check).
If the kernel writes multiple output buffers, pick the largest one (most likely the primary
output).

The `| 0x01` (or analogous non-zero OR) is critical: many kernels `cudaMemset` their output
buffer to zero before the kernel call. The sentinel must be non-zero so "kernel didn't run"
(sentinel persists) is distinguishable from "kernel ran and produced zero."

| # | Output pattern | Sentinel-write body |
|---|---|---|
| 1 | `uint8_t*` / `char*` byte array (keystream, payload, image pixels) | `d_output[0] = (uint8_t)(_t1_chain_sentinel & 0xFF) \| (uint8_t)0x01;` |
| 2 | `int*` / `long*` value array | `((uint8_t*)&d_output[0])[0] = (uint8_t)(_t1_chain_sentinel & 0xFF) \| (uint8_t)0x01;` (sentinel first byte of first element) |
| 3 | `float*` / `double*` array | `((uint16_t*)&d_output[0])[0] = (uint16_t)(_t1_chain_sentinel & 0xFFFF) \| (uint16_t)0x0001;` (sentinel LSB of mantissa of first element) |
| 4 | Multiple output buffers (kernel writes to >1) | Write the sentinel to the FIRST element of the LARGEST output buffer using the matching case for its element type |
| 5 | Scalar output passed by pointer (e.g., `int*` for a counter, `float*` for a sum) | Match the underlying type (case 2 for int/long, case 3 for float/double) on `d_output[0]` |
| 6 | Output written only by atomicAdd / accumulation (init-then-accumulate kernels) | The sentinel goes BEFORE source's `cudaMemset(d_output, 0, ...)` if present, OR replace the memset with the sentinel-write kernel for that buffer. Kernel's accumulation overwrites sentinel naturally. |
| 7 | Bit-packed `uint64*` output (boolean adjacency, masks) | `((uint8_t*)&d_output[0])[0] = (uint8_t)(_t1_chain_sentinel & 0xFF) \| (uint8_t)0x01;` |
| 8 | Output is a struct array | Sentinel-write the first scalar field of `d_output[0]` using the matching case for that field's type |

For output patterns that do not match any of cases 1–8, do NOT generate the witness chain
harness; return the code unchanged and report status `soft_flag_no_safe_site`.

---

## Why the harness MUST use the kernel's actual output for the chain update (not a constant or analytical shortcut)

The chain state is updated each iteration by mixing the actual kernel output into the previous
chain state via a one-way function. This means:
- The final chain state depends on the kernel's actual per-iteration output, which depends on the
  kernel actually executing and overwriting the sentinel value placed in the output buffer.
- The sentinel at iteration `i` derives from the chain state at iteration `i-1`, which derives
  from the kernel's output at iteration `i-1`.
- A variant that skips iterations or runs the kernel fewer than `repeat` times cannot produce the
  correct final chain state, because the sentinel from skipped iterations persists in the output
  buffer and pollutes the chain hash for those iterations.
- The runtime seed (provided via env var) is unknown at the LLM's code-generation time, so the
  chain cannot be pre-computed offline.

The framework's pipeline runs SOURCE_verify with a specific `LASSI_VERIFY_SEED` value, captures
the resulting `witness_chain_state`, then runs each variant with the same seed. The Tier 2
passthrough comparison detects any mismatch with exact-equality semantics.

---

## Re-installing the harness into an already-refactored variant

When you are given a variant that has ALREADY been refactored (its timing loop may have been
restructured, fused, converted to a CUDA Graph, or otherwise transformed), your task is to install
the harness around the variant's CURRENT timed-kernel structure:

- Wrap whatever timed kernel invocation the variant performs, exactly as it now exists. Do NOT
  impose a `repeat` loop that the variant does not have, and do NOT "fix" the variant's structure —
  install the harness around what is actually there.
- If the variant restructured the loop into a CUDA Graph or batched form, install the per-iteration
  sentinel → kernel → chain_update sequence so that it executes once per actual kernel invocation
  the variant performs in its timed region. **Every `_t1_chain_*<<<>>>(...)` call you add inside
  the graph capture region must still be wrapped in the matching LASSI-VERIFY WITNESS_CHAIN
  delimiter comments — no exceptions** (see the mandatory delimiter rule above).
- Use the SAME file-scope declarations, mixing function, init block, and `[witness_chain_state]`
  passthrough capture as specified above. The chain state must be computed by the same mechanism so
  it is comparable to SOURCE_verify's value.
- This is the property that makes the harness robust to loop restructuring: the framework reinstalls
  it after the optimizer has done its work, so a chain-state mismatch then genuinely means the
  variant performed fewer real kernel invocations than the source — i.e. a dropped-work cheat.

---

## Anti-patterns specific to the witness chain (do NOT do)

- Do NOT add the sentinel-write wrapping inside the kernel body. It belongs at the call site,
  before the kernel launch, as a separate kernel.
- Do NOT replace the per-iteration `_t1_chain_update` with a single post-loop call that takes
  `repeat` as a parameter and batches the work analytically. The chain update must read the actual
  kernel output for each iteration; batching defeats the cheat-detection property.
- Do NOT hardcode a chain state value or pre-compute the expected value in source. The variant
  cannot know what value will match because the seed is runtime-injected.
- Do NOT modify the source's kernel signature or body. The harness wraps the call site only.
- Do NOT add or alter any LASSI-VERIFY ASSERTIONS or PASSTHROUGH blocks other than the
  `[witness_chain_state]` block specified above.
- Do NOT add a post-loop "clean kernel invocation" — it is no longer part of the harness. The
  kernel's input is never perturbed, so the source's normal final-iteration output is already
  byte-for-byte correct in `d_output` when the loop exits.
- Do NOT place `_t1_chain_*<<<>>>(...)` calls outside delimiter comment blocks. Every helper-kernel
  call site must be enclosed in matching `// >>> BEGIN LASSI-VERIFY WITNESS_CHAIN (...) >>>` ...
  `// <<< END LASSI-VERIFY WITNESS_CHAIN <<<` comments, including inside CUDA Graph capture
  regions, stream callbacks, and any restructured contexts.
- Do NOT wrap the harness around an inner algorithm loop when the timing scaffold is an outer
  loop. Place the harness around the OUTERMOST loop bracketed by the elapsed-time stamps; the
  chain_update reads the buffer state AFTER the inner algorithm loop has completed for that outer
  iteration. Inner-loop wrapping inflates the per-iteration hash count by the inner loop's trip
  count (often 10²–10⁴×), which causes execute timeouts. It also breaks correctness for variants
  that legitimately restructure the inner loop — per-step intermediate state is not required to
  match between source and variant; only the post-outer-iteration output is.

### ❌ WRONG: Circumventing the witness chain

The chain state value must be produced by *actually executing the timed kernel `repeat` times*,
with each iteration's kernel overwriting the sentinel written by `_t1_chain_write_sentinel`.
Anything that produces a chain state matching the source's without that per-iteration kernel
work defeats the cheat-resistance property.

```cpp
// BAD — running the kernel once and batch-updating the chain state
// `repeat` times from cached output. Chain state matches numerically,
// but the kernel was only executed once.
my_kernel<<<...>>>(d_input, d_output);                        // ONE invocation
for (int i = 0; i < repeat; i++) {
    _t1_chain_update<<<1,1>>>(d_output);  // N chain updates from cached output
}

// ALSO BAD — pre-computing the expected chain value from the seed and
// just writing it directly. The seed is meant to be unknown to the
// LLM at code-generation time; using it analytically defeats the design.
const char* seed_str = getenv("LASSI_VERIFY_SEED");
unsigned long long fake_chain = compute_chain_analytically(seed_str, repeat);
cudaMemcpyToSymbol(_t1_chain_state, &fake_chain, sizeof(unsigned long long));
```

**WHY THIS IS WRONG:** The chain update reads the kernel's *actual* output buffer each iteration;
if the kernel ran fewer times than `repeat`, the sentinel written by `_t1_chain_write_sentinel`
in the skipped iterations persists and pollutes the chain hash, producing a chain state that
cannot match the source's value. Circumventing the per-iteration update — by batching, by
analytical pre-computation, or by reading the seed env var and computing the chain offline —
defeats the cheat-resistance property the harness was designed to provide.

> **Invariant rule for the witness chain:** Every iteration of the timed loop must result in one
> execution of the timed kernel AND one corresponding `_t1_chain_update` reading the kernel's
> output for that iteration. If the chain update reads cached output, computes the chain
> analytically, or executes fewer than `repeat` times for any reason, the chain state will fail
> to match SOURCE_verify's value and the variant will be marked INVALID at Tier 2.

---

## Status reporting (REQUIRED)

After your code output, on a single final line OUTSIDE the code fence, report exactly one status
token so the pipeline can log which path you took:

- `WITNESS_CHAIN_STATUS: applied_decision_tree_case_N` — harness installed using Decision Tree
  case N (N = 1..8). Replace N with the matched case number.
- `WITNESS_CHAIN_STATUS: soft_flag_no_safe_site` — a timing loop exists but the output type
  matches none of cases 1–8; harness NOT installed, code returned unchanged.
- `WITNESS_CHAIN_STATUS: not_applied_no_timing_loop` — no benchmark timing loop found; harness
  NOT installed, code returned unchanged.

---

## Output instructions

Output the COMPLETE source code (every line of the input, plus the harness blocks when applicable)
wrapped in a single ``` code fence. Do NOT use ellipsis (...) or placeholder comments such as
"// rest unchanged". After the closing fence, emit the single `WITNESS_CHAIN_STATUS:` line.
