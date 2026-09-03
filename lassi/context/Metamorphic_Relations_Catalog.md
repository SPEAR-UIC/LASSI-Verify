
---

# Metamorphic Relations Catalog for GPU Kernel Verification (Tier 3)

This catalog is the reference for LASSI-Verify Tier 3 metamorphic testing. It is loaded into the
`generate_metamorphic_contract()` LLM prompt via `prompts.get_codeknowledge("metamorphic_catalog")`.
It is the direct analog of `Invariant_Assertions_Catalog.md` (Tier 1) and `Passthrough_Catalog.md` (Tier 2).

---

## Critical Distinction: Tier 1 Assertions vs Tier 3 Metamorphic Relations

### ✅ Tier 1 (Invariant Assertions) — single-execution properties
- A property that holds or fails within ONE program execution.
- Involves only one input/output pair.
- Examples: pointer validity, output is finite, energy is conserved within a single run.

### ✅ Tier 3 (Metamorphic Relations) — multi-execution relational properties
- A property that relates TWO OR MORE executions under different (but related) inputs.
- Cannot be checked within a single run.
- Examples: output scales proportionally when input is scaled; permuting input rows produces permuted output.

**Formal grounding** (Chen et al. 1998, 2019): An MR is a necessary property R ⊆ Xⁿ × Yⁿ defined
over n≥2 inputs and their outputs. Single-execution properties (n=1) are NOT MRs — they are assertions.
LASSI-Verify uses both: Tier 1 catches single-execution violations; Tier 3 catches relational violations.
All reviewed literature confirms they detect **different fault types** and are most effective when combined.

---

## Tier 3 Architecture (How Metamorphic Testing Works Here)

### Self-Referential Testing

Each generated code is tested against **itself** under input transformation. The reference output ("Y")
is the generated code's own Tier 2 passthrough output. Tier 3 runs a transformed variant ("Z") and
checks the expected mathematical relationship.

```
Generated code + original input X   →   output Y    (already captured from Tier 2)
Generated code + transformed input  →   output Z    (new metamorphic variant run)
Check: Z satisfies expected_relation(Y)
```

SOURCE_verify is **not** the comparison baseline. SOURCE_verify variants run once in Stage 0 solely
to filter which relations are applicable (see SOURCE_verify Filter below).

### SOURCE_verify Filter (Zero-FP Requirement)

Each relation is run against SOURCE_verify in Stage 0. Relations that **fail for the source** are
marked **inactive** and silently skipped for all generated codes.

**Why:** The source is the ground truth. A relation that fails on the source is inapplicable or
requires refinement — it is not a valid requirement for generated codes. This achieves zero false
positives from inapplicable MRs.

### Separate Variant Files

All Tier 3 variants are compiled and executed as **separate binaries**. No metamorphic code is
inserted into the timed performance runs. Variant files are named `{file_name}_mt_{name}.cu`.

---

## Design Principles

### 1. Quantitative Over Qualitative

Relations must have **numerically predictable** outcomes. Directional relations ("output increases
when input increases") are insufficient — a code with a 2× scaling bug satisfies this direction
check but fails the exact quantitative check.

> "many MRs are not sensitive to mutants especially when the mutant just shifts the output from
> the real value to a different one." — Ding et al. 2016

**Strength hierarchy:** equality > quantitative-ratio > inequality/monotonicity

All relations in this catalog specify an exact expected ratio, sum, permutation, or bound.

### 2. Single-Parameter Change Per Variant

Each metamorphic variant changes exactly **one** input aspect. This isolates causality — if the
variant fails, the specific change that caused it is immediately known.

### 3. Component-Level Over System-Level

Prefer relations that apply to a **specific named buffer** at a specific observation point rather
than to entire program stdout. Component-level MRs are more discriminating and easier to diagnose.

### 4. Validation MRs from Mathematical/Physical Semantics

Tier 3 generates **validation MRs** — derived from mathematical physics and domain knowledge,
not from implementation details (Raunak & Olsen 2012, via Li et al. 2025). The LLM must reason
from what the computation *means* mathematically, not how the code happens to be structured.

> "Good MRs are strongly inspired by program semantics, NOT implementation strategy." — Segura 2016

### 5. Maximum Diversity (Different Input Parameter Types)

Select relations that touch **different input parameter types** to exercise different code paths:
- SCALING → amplitude parameters
- PERMUTATIVE → index/ordering parameters
- CONSERVATION → array decomposition/aggregation
- SYMMETRY → sign/reflection parameters
- ROTATION → angular/spatial coordinate parameters

### 6. NaN/Infinity Guard (MANDATORY)

If either the original output (Y) or the variant output (Z) contains NaN or Infinity at any
observation point, **SKIP** that relation check — do not report FAIL. Floating-point special
values are edge cases where the MR cannot distinguish correct from incorrect behavior
(GENMORPH, Ayerdi et al. 2024, Section 8.5).

### 7. Coverage Principle (How Many Relations to Propose)

**The source code IS the correctness specification.** Propose every relation that
(i) is applicable to this specific source code AND (ii) probes a fault detection
capability not already covered by another proposed relation. No more, no less.
Leaving an applicable property untested means the framework's verification claim
is weaker than it could be.

**Two rules determine the contract:**

1. **Type coverage.** For every Tier 3 relation type applicable to the source code,
   propose at least one relation of that type before considering a second relation
   of a type already covered.
2. **Within-type distinctness.** Propose a second (or third, fourth, ...) relation
   of the same type ONLY when it transforms a different input dimension AND would
   detect a fault class the first relation cannot. Two relations that catch the
   same fault class are not distinct — keep the stronger and drop the weaker.

**Reference range.** As a descriptive prior observed across many kernels, 3 to 6
diverse MRs typically reveal approximately 90% of faults. This is **a prior, not a
per-kernel target**. Some kernels yield 2 distinct relations under the rules above;
some yield 8 or more. Do NOT pad to 3 by inventing a weak third relation. Do NOT
cap at 6 by dropping an applicable, distinct relation. Propose exactly what the two
rules above produce — no fewer, no more.

---

## Formal MR Specification (3-Tuple)

Every catalog entry specifies a relation as a 3-tuple:

- **`input_relation`**: How the variant input is constructed from the original input.
  This defines Ri in: `Ri(x₁, x₂) ⇒ Ro(f(x₁), f(x₂))`.
- **`output_relation`**: What mathematical relationship must hold between Y and Z.
  This defines Ro.
- **`applicable_when`**: Preconditions on the computation that must hold for the MR to be valid.
  If these fail for SOURCE_verify, the relation is filtered inactive.

---

## Selection Framework

When generating a contract, work through these analysis questions. Apply the Coverage
Principle (§7 above) to decide how many relations belong in the contract — propose
every relation that covers an applicable type AND probes a distinct fault class,
no more and no fewer.

### Analysis Question 1: What is the core computation type?

- **Linear algebra** (matrix multiply, dense solve): SCALING → ADDITIVITY → PERMUTATIVE
- **Iterative linear solver** (stationary methods, Krylov-subspace methods): SCALING (on source term if BCs allow) → PERMUTATIVE
- **Stencil / linear CFD** (Laplace, isotropic diffusion, heat conduction with homogeneous BCs): CONSERVATION (flux balance) → SYMMETRY → ROTATION (if isotropic) → SCALING (if PDE is linear and BCs are homogeneous)
- **Coupled nonlinear iterative CFD** (Navier-Stokes, SOR/Red-Black pressure solvers, projection methods, driven-cavity flow, multi-field u-v-p coupling): CONSERVATION (incompressibility or momentum residual, if checkable from passthrough) → MONOTONICITY (residual history, if printed to stdout) — **SCALING and SYMMETRY are typically NOT applicable** (see ⚠️ Coupled Nonlinear CFD guidance in Domain-Specific section)
- **Reduction kernel** (sum, dot product, norm): SCALING → ADDITIVITY → CONSERVATION (decomposition) → PERMUTATIVE
- **Particle / N-body**: CONSERVATION (energy) → PERMUTATIVE (particle index) → SCALING
- **Categorical / equality-based kernel** (output depends only on equality/inequality of discrete attribute values, not on their magnitudes — e.g., pairwise mismatch counting, hash-based comparison, set membership, equality-driven aggregation): INVARIANCE (under attribute-position permutation, value relabeling that preserves equality classes, uniform value shift, value negation — output is determined by equality structure, not magnitudes) → PERMUTATIVE (instance / row swap, if rows are interchangeable instances)
- **Normalized distance / correlation kernel** (z-score normalized DTW, LB_Keogh, cross-correlation, standardized regression): INVARIANCE on the **co-normalized group** (raw data + precomputed mean + precomputed std, all scaled by the same α → normalized value `(x−mean)/std` unchanged → output unchanged); PERMUTATIVE (inner sum index permutation, output unchanged) — SCALING on any **single** input variable IS NOT APPLICABLE (see normalization warning below)
- **Dynamic programming / path algorithms** (shortest path, DP table fill, wavefront, graph BFS/DFS): PERMUTATIVE (row/column index permutation) → MONOTONICITY (result non-decreasing with iterations) — **SCALING IS NOT APPLICABLE** (see below)
- **Sorting / ranking / selection**: PERMUTATIVE (input permutation → permuted output) — SCALING IS NOT APPLICABLE (sort of 2×input is NOT 2× sort of input in index rank)
- **Thresholding / clamping / activation**: SCALING IS NOT APPLICABLE for any input scale that crosses a threshold; CONSERVATION may apply if the operation conserves sum under special conditions

### Analysis Question 2: Is the algorithm linear or nonlinear?

- **Linear**: BOTH SCALING and ADDITIVITY apply: `f(α×x) = α×f(x)` AND `f(x+c) = f(x)+c` (when
  linear coefficient is unity; see ADDITIVITY contraindications for non-unit coefficients)
- **Nonlinear or thresholded**: neither SCALING nor ADDITIVITY applies to the full output; check
  whether either applies to a residual metric or intermediate buffer
- **Iterative with fixed boundary conditions**: SCALING applies to the source term ONLY if BCs are
  also scaled or are homogeneous (zero); ADDITIVITY does not apply (BC constraints pin boundary values)

**⚠️ CRITICAL — Absolute convergence thresholds make SCALING unreliable for iterative solvers:**

If the solver stops at `|residual| < tol` (an *absolute* threshold — not `|r|/|r₀| < tol`), then
scaling the RHS by α does not guarantee exactly α × solution at the stopping point:
- The true converged solution scales: x* → α·x*
- But the solver terminates when absolute |r| < tol regardless of the RHS magnitude
- The number of iterations may differ between the scaled and unscaled problems, and the final
  state diverges from exactly α·x* by O(tol)

If tol is tight (≤ 1e-8) relative to the solution magnitude, the error is negligible and SCALING
may still pass. If tol is in the range 1e-6 to 1e-3, or if the stopping criterion is a dimensional
physical residual (in Pa, m/s, etc.) rather than a dimensionless normalized residual, the
deviation from the expected ratio can easily exceed the MR tolerance.

**Rule:** For iterative solvers with absolute convergence criteria, note this risk in `applicable_when`.
Expect the source filter to reject the SCALING relation for any kernel with a loose absolute threshold
or a dimensional stopping criterion. If this applies to every SCALING relation you were considering,
prioritize CONSERVATION or MONOTONICITY instead.

**⚠️ CRITICAL — MIN/MAX operations always make SCALING inapplicable to the final output:**
If the kernel uses `min()`, `max()`, `fmin()`, `fmax()`, `__min()`, `__max()`, conditional
thresholds, or any comparison-based selection, the computation is non-linear and SCALING is
**always wrong** for the output. Do NOT propose SCALING when you see these operations.

Why: `min(α×a, α×b) = α×min(a, b)` holds only when α > 0. But in a multi-iteration
dynamic programming context, the intermediate state `min(a, b)` is the INPUT to the next
iteration — so `f_N(α×x) ≠ α×f_N(x)` in general for N > 1 iteration (the min-based
path selection at each step is non-linear in the accumulated costs).

**If you see any of these patterns, eliminate SCALING immediately:**
- `min(...)`, `max(...)`, `fmin(...)`, `fmax(...)` in the kernel
- `if (a < b)` selection to write output (same as `min` semantically)
- Accumulated values that represent accumulated path costs, shortest distances, or minimum-weight sums
- The kernel accumulates values with `+` AND selects with `min/max` — this is a shortest-path/DP pattern

**⚠️ CRITICAL — Normalization (z-score / standardization) invalidates single-variable SCALING:**

If the kernel normalizes data with the pattern `(x - mean) / std` (z-score normalization), the
variables `x`, `mean`, and `std` form a **co-normalized group**. Scaling only `x` produces output
that is NOT a simple scaling of the original because the mean-subtraction breaks linearity:

```
(α·x - mean) / std   ≠   α · (x - mean) / std    [when mean ≠ 0]
```

**Pattern to recognize:** `(buffer[idx] - stats_mean[idx]) / stats_std[idx]` where `stats_mean`
and `stats_std` are precomputed arrays passed as separate kernel arguments.

**Rules for co-normalized groups:**
- Do NOT propose SCALING on `buffer` alone, `stats_mean` alone, or `stats_std` alone — the source
  filter will reject it.
- The valid relation is `type = INVARIANCE` with input transformation "scale all three
  (`buffer`, `stats_mean`, `stats_std`) by the same positive factor α" → the normalized value
  `(x − mean)/std` is unchanged → output is unchanged. (Use the INVARIANCE category for this;
  do NOT express it as SCALING with `factor = 1.0`.)
- This is a strong correctness test: any code that drops the mean subtraction or std division
  will produce wrong output under this transformation.

**If only division is present (mean = 0):** scaling `x` and `std` together by α gives
`α·x / α·std = x / std` → output unchanged. Again, express as `type = INVARIANCE` (joint α-scaling
on the co-normalized pair), not as SCALING.

**Clamped-squared deviation pattern** (common in distance kernels): `val² × (val > 0)` or
`max(0, val)²` where `val = (x - mean)/std - envelope`. Scaling the co-normalized group
(x + mean + std together) leaves `val` unchanged → output unchanged. Express as INVARIANCE.
Scaling only `x` changes `val` in a position-dependent, nonlinear way because which values are
clamped to zero also changes — do not propose it.

### Analysis Question 3: Are there physical conservation laws?

- **Explicit conservation** (mass, energy, momentum): CONSERVATION (1.1) applies directly
- **Reduction** (sum, integral): CONSERVATION (1.3) decomposition applies
- **No conservation law**: skip Category 1

### Analysis Question 4: What symmetries exist in the problem?

- **Antisymmetric function** (f(−x) = −f(x)): SYMMETRY (2.1) negation applies
- **Symmetric domain** (symmetric BCs, isotropic physics): SYMMETRY (2.2) reflection applies
- **Spatially isotropic problem** (no preferred direction): ROTATION (5.1 or 5.2) may apply
- **No symmetry**: skip Categories 2 and 5

### Analysis Question 5: If no quantitative relation applies, which structural relation does? (Universal fallback)

Quantitative relations (SCALING, ADDITIVITY, and scaling-based INVARIANCE) require the
output to be a simple function of a transformed input. Many correct kernels are nonlinear,
clamped/thresholded, min/max-based, or have pinned boundary conditions and therefore admit
**none** of them — the source filter will reject every quantitative relation you try. This
is expected and is NOT a reason to give up: producing an empty contract aborts verification
entirely, which is strictly worse than one well-chosen structural relation.

**Almost every non-degenerate deterministic kernel admits at least one STRUCTURAL relation.**
Select it by the kernel's *computational property* — this reasoning is universal and does not
depend on the kernel's scientific domain:

- **SENSITIVITY** — applies to essentially ANY deterministic map. If the kernel reads an input
  buffer and its output genuinely depends on that buffer, then perturbing that input (a single
  element, or a single bit of a packed value) MUST change at least one observed statistic or
  sample. Output that is *unchanged* under a real input perturbation means the computation
  ignored the input — a serious fault. This survives even when every quantitative relation fails.
- **MONOTONICITY** — applies to ANY iterative/convergent computation. If the kernel iterates
  toward a solution and a residual/error/objective value is observable (in a passthrough buffer
  or printed to stdout), a converging method must not let it increase from one iteration to the
  next. Use this whenever an iteration count and a convergence metric exist.
- **CONSERVATION** — applies whenever the computation preserves an aggregate/summable quantity
  (total mass, particle count, probability sum, energy). The aggregate is unchanged (or changes
  by a known amount) under the operation, regardless of per-element values.

**Rule:** If Analysis Questions 1–4 leave you with only quantitative relations that the
linear/nonlinear/threshold/normalization checks above have ruled out — or if previous-attempt
feedback reports that quantitative relations did not hold for this source — propose at least one
structural relation from this list, chosen by the properties above. A contract containing a
single well-grounded SENSITIVITY or MONOTONICITY relation is far more valuable than an empty one.
Do not pad with extra weak relations (Coverage Principle §7 still applies) — but never return
zero relations when a structural one is available.

### Fail-Fast Ordering Rule

Order selected relations **cheapest-first and most-discriminating-first** within the same cost tier:

1. CONSERVATION — cheap (1 extra compile + run; no input transformation needed)
2. SYMMETRY — cheap (1 extra compile + run; negation of input, no structural change)
3. SCALING — cheap (1 extra compile + run; **highest discriminating power**)
4. ADDITIVITY — cheap (1 extra compile + run; **complementary to SCALING** — catches additive bias bugs SCALING misses)
5. PERMUTATIVE — cheap (1 extra compile + run)
6. ROTATION — cheap (1 extra compile + run)
7. MONOTONICITY — cheap (1 extra compile + run; **weakest, last resort only**)

The fail-fast loop stops at the **first FAIL**. Put the best low-cost relations first.

---

## GPU-Specific Considerations

### Input Transformation Placement

Apply the input transformation **after** host array initialization but **before** the first
host-to-device memory copy. This ensures the GPU kernel sees the transformed data.

```cpp
// Original initialization (unchanged)
initialize_input(h_input, N);

// ↓ INSERT TRANSFORMATION HERE ↓
for (int i = 0; i < N; i++) h_input[i] *= alpha;   // SCALING example

// First H→D copy (must see transformed data)
cudaMemcpy(d_input, h_input, N * sizeof(double), cudaMemcpyHostToDevice);
```

### METAMORPHIC Output Block Format

The variant binary prints METAMORPHIC blocks to stdout after each final device-to-host transfer.
These are parsed by `check_metamorphic_relation()`.

**For SCALING, PERMUTATIVE, ROTATION** (compare transformed output to α × original output):
```
=== BEGIN LASSI-VERIFY METAMORPHIC [label] ===
Relation: scaling
Factor: 2.0
Buffer: h_output
Elements: 1024
Statistics: min=%.17g, max=%.17g, mean=%.17g
Sample_first10: %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g
=== END LASSI-VERIFY METAMORPHIC [label] ===
```

The `[label]` must **match the corresponding PASSTHROUGH block label** so the parser can pair
original and variant outputs at the same observation point.

**For CONSERVATION** (before/after sum check):
```
=== BEGIN LASSI-VERIFY METAMORPHIC [energy_before] ===
Relation: conservation
Buffer: h_energy
Elements: 1024
Sum: %.17g
=== END LASSI-VERIFY METAMORPHIC [energy_before] ===
... kernel execution ...
=== BEGIN LASSI-VERIFY METAMORPHIC [energy_after] ===
Relation: conservation
Buffer: h_energy
Elements: 1024
Sum: %.17g
=== END LASSI-VERIFY METAMORPHIC [energy_after] ===
```

**For ADDITIVITY** (compare shifted output to original + offset; same Statistics format as SCALING but with `Offset:` instead of `Factor:`):
```
=== BEGIN LASSI-VERIFY METAMORPHIC [label] ===
Relation: additivity
Offset: 1.0
Buffer: h_output
Elements: 1024
Statistics: min=%.17g, max=%.17g, mean=%.17g
Sample_first10: %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g
=== END LASSI-VERIFY METAMORPHIC [label] ===
```

The checker verifies: `Z_stat ≈ Y_stat + offset` for each statistic and sample.
The `offset` value is taken from the contract's `factor` field.

**For INVARIANCE** (compare variant output to original output; expected ratio is unity; the variant
input has been transformed by some non-trivial transformation T, but the output must be unchanged):
```
=== BEGIN LASSI-VERIFY METAMORPHIC [label] ===
Relation: <relation_name>
Invariance_under: <one-line name of the input transformation, e.g. attribute_order_reversal>
Buffer: h_output
Elements: 1024
Statistics: min=%.17g, max=%.17g, mean=%.17g
Sample_first10: %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g
=== END LASSI-VERIFY METAMORPHIC [label] ===
```

The checker verifies: `Z_stat ≈ Y_stat` for each statistic and each sample (allclose-style
tolerance, same as SCALING with factor=1.0). The `Invariance_under` line is informational —
the checker does not parse it but it documents which transformation produced the variant.
The `[label]` MUST match the corresponding PASSTHROUGH block label so the checker can pair
the variant output Z with the original output Y at the same observation point.

**For MONOTONICITY** (per-iteration residual):
```
=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_0] ===
Relation: monotonicity
Iteration: 0
Residual: %.17g
=== END LASSI-VERIFY METAMORPHIC [residual_iter_0] ===
```

### Preserve All Existing LASSI-VERIFY Blocks

Metamorphic variant files must preserve ALL existing `LASSI-VERIFY ASSERTIONS` and
`LASSI-VERIFY PASSTHROUGH` blocks without modification. The variant ONLY adds the input
transformation code and new METAMORPHIC output blocks.

### Floating-Point Tolerance

Use **allclose-style** tolerance for all metamorphic checks:

```
|Z_stat - expected_Z_stat| ≤ atol + rtol × max(|Z_stat|, |expected_Z_stat|)
where atol = |expected_Z_stat| × rtol
```

Default: `rtol = 1e-5` (slightly looser than the passthrough equivalence tolerance, to account for
accumulated FP noise in two independently-executed runs). Adjust per relation if needed (see
PERMUTATIVE note).

### METAMORPHIC Block Placement — Mandatory Rules

Incorrect block placement causes stdout floods that crash or stall the pipeline. These rules
must be followed for every variant file you produce.

**Rule 1 — NEVER place a METAMORPHIC block inside a nested inner loop.**

If the algorithm has a two-level loop structure (e.g., outer timestep loop wrapping an inner
solver or SOR loop), a block inside the inner loop emits O(timesteps × inner_iterations) output
lines — easily millions of lines for a typical iterative benchmark. Place the block AFTER the
inner loop exits, once per outer-loop iteration.

```cpp
// TWO-LEVEL STRUCTURE — correct placement
for (int t = 0; t < num_timesteps; ++t) {         // ← outer timestep loop
    for (int iter = 0; iter < max_iter; ++iter) {  // ← inner solver loop
        // ... kernel launches ...
        if (converged) break;
    }
    // ✅ CORRECT: block OUTSIDE inner loop — one block per timestep
    printf("=== BEGIN LASSI-VERIFY METAMORPHIC [residual_t_%d] ===\n", t);
    printf("Relation: monotonicity\nIteration: %d\nResidual: %.17g\n", t, final_residual);
    printf("=== END LASSI-VERIFY METAMORPHIC [residual_t_%d] ===\n", t);
}
// ❌ WRONG: placing a block inside the inner loop above would print millions of lines
```

**Rule 2 — For a single-level iteration loop, cap the number of blocks printed.**

Even when the iteration loop is the outermost loop, cap the METAMORPHIC output. Solvers with
large `max_iter` (thousands of iterations) will overflow pipeline output buffers if every
iteration is printed. A few hundred data points is sufficient to verify a monotone trend.

```cpp
// SINGLE-LEVEL LOOP — cap at 200 iterations
#define LASSI_MT_MAX_PRINT 200
for (int iter = 0; iter < max_iter; ++iter) {
    // ... kernel launches ...
    if (iter < LASSI_MT_MAX_PRINT) {
        printf("=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\n", iter);
        printf("Relation: monotonicity\nIteration: %d\nResidual: %.17g\n", iter, residual);
        printf("=== END LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\n", iter);
    }
    if (converged) break;
}
```

**Rule 3 — SCALING, SYMMETRY, CONSERVATION, PERMUTATIVE, and ADDITIVITY blocks belong
entirely outside all loops, after the final device-to-host transfer.**

These relation types compare the final output state only. There is never a reason to place
one of these blocks inside any loop. If you find yourself writing such a block inside a loop,
move it to after the loop exits.

---

## Your Instructions (for `generate_metamorphic_contract()`)

### Step 1: Analyze the Code

Identify:
- The core computation type (linear solve, stencil, reduction, particle, ...)
- Whether the algorithm is linear or nonlinear
- Physical conservation laws present
- Symmetry properties of the problem domain and boundary conditions
- Literal constants in the kernel: grid spacing `dx`, timestep `dt`, element count `N`,
  convergence threshold `tol`, domain length `L`

### Step 2: Select Relations from the Catalog (Apply the Coverage Principle)

Use the Selection Framework above. Apply the Coverage Principle from §7:
for every applicable type, propose at least one relation; within a type, propose
additional relations only when each catches a distinct fault class. Typical
contracts produce 3 to 6 relations as a result — that is an observation, not a target.

Apply BOTH approaches:

**Input-driven** (for SCALING, ADDITIVITY, PERMUTATIVE, SYMMETRY, ROTATION, INVARIANCE):
Ask "If I change this input parameter in this way, what exactly should the output be?"
For INVARIANCE the answer is "unchanged."

**Output-driven** (for CONSERVATION): Ask "What aggregation of the output should be preserved?
What input structure allows me to decompose it?"

**MANDATORY — Symbolic trace before proposing any SCALING relation:**

Before writing a SCALING relation into the contract, trace the specific input variable(s) you
plan to scale through every kernel operation symbolically and confirm the expected output ratio.

Example — incorrect reasoning (keogh failure):
- "I see squared terms → scaling subject by α should scale output by α²"  ← WRONG
- Trace: `value = (α·subject - avg) / std` → value is NOT `α·(subject-avg)/std` → ratio is
  input-position-dependent → SCALING on `subject` alone is INVALID.

Example — correct reasoning (keogh, expressed as INVARIANCE):
- Scale `subject`, `avgs`, `stds` all by α: `value = (α·s − α·avg)/(α·std) = (s − avg)/std`
  → value unchanged → output unchanged → propose this as `type = INVARIANCE` (joint α-scaling
  of the co-normalized group), NOT as SCALING with factor=1.0.

If you cannot write a symbolic trace that confirms the exact output relationship (ratio for
SCALING, shift for ADDITIVITY, equality for INVARIANCE, ...) for at least one representative
input value, do NOT propose the relation. A relation that cannot be verified symbolically will
fail the source filter and waste a contract slot.

**IMPORTANT — null-factor relations are not currently supported:** Only propose relations where
`factor` is a concrete non-null number (e.g., `2.0`, `-1.0`, `1.0`). Any relation requiring
element-wise array comparison — including reflection symmetry, general permutations, and
SYMMETRY/SCALING entries where you cannot state a single scalar `factor` — must be omitted.
These are deferred for future implementation. See the ❌ Null-Factor Anti-Pattern below.
Proposing a null-factor
relation wastes a contract slot; the pipeline will SKIP it without verification benefit.

### Step 3: Use Constants from the Kernel

For SCALING: choose α from physically meaningful kernel constants. If the kernel has a timestep `dt`,
then `α = 2.0` doubles the forcing; if it has a grid spacing `dx = 0.01`, consider factors that
preserve numerical stability. Default: α = 2.0.

For CONSERVATION decomposition: use element count N or iteration count from the kernel.

### Step 4: Return a JSON Contract

Return a single JSON block with this structure:

```json
{
  "relations": [
    {
      "name": "conservation",
      "type": "CONSERVATION",
      "description": "Total kinetic energy is conserved across the particle update",
      "factor": null,
      "input_variables": [],
      "observation_points": ["energy_before", "energy_after"],
      "expected_relationship": "|sum_after - sum_before| / |sum_before| < tolerance",
      "tolerance": 1e-8,
      "cost": "very_cheap"
    },
    {
      "name": "scaling",
      "type": "SCALING",
      "description": "Linear PDE: scaling the source term b by alpha scales the solution by alpha",
      "factor": 2.0,
      "input_variables": ["h_b"],
      "observation_points": ["final_solution"],
      "expected_relationship": "output = factor * original_output",
      "tolerance": 1e-5,
      "cost": "cheap"
    },
    {
      "name": "attribute_order_invariance",
      "type": "INVARIANCE",
      "description": "Kernel computes a sum of per-attribute equality counts; reversing the attribute order within every input row preserves the set of per-position mismatches, so the output buffer is unchanged",
      "factor": 1.0,
      "input_variables": ["h_data"],
      "observation_points": ["final_distance"],
      "expected_relationship": "output = original_output (within tolerance)",
      "tolerance": 1e-5,
      "cost": "cheap"
    }
  ]
}
```

**Field descriptions:**
- `name`: short identifier used in variant filename (`{file_name}_mt_{name}.cu`). Use lowercase with
  underscores, no spaces.
- `type`: one of `CONSERVATION`, `SYMMETRY`, `SCALING`, `ADDITIVITY`, `PERMUTATIVE`, `ROTATION`, `MONOTONICITY`, `INVARIANCE`, `SENSITIVITY`
- `description`: one sentence explaining the mathematical or physical basis of the relation. For
  INVARIANCE, name the specific input transformation being applied (e.g., "reverse the attribute
  order within every instance").
- `factor`: scaling factor (for SCALING/SYMMETRY), additive offset c (for ADDITIVITY),
  `1.0` for INVARIANCE (output ratio is unity by definition), `null` for SENSITIVITY (the
  output magnitude of change is not predicted, only its existence), or `null` otherwise
- `input_variables`: C++ variable names of host arrays/scalars to transform (from the source code)
- `observation_points`: METAMORPHIC block label(s); must match PASSTHROUGH labels when comparing
  scaled output to original. **MUST be the primary output array** (e.g., `["final_solution"]`,
  `["final_field"]`) — never use convergence scalars, error norms, residuals, or iteration
  counts as observation points for SCALING, SYMMETRY, or NEGATION relations. Convergence
  diagnostics legitimately differ between a reference implementation and an optimized one due
  to FP evaluation order changes; they cannot serve as relational checks. If the passthrough
  catalog already captures the primary output array under a specific label, use that label here
  so the METAMORPHIC block is co-located with the existing PASSTHROUGH block.
- `expected_relationship`: human-readable description of the output check
- `tolerance`: relative tolerance for the comparison (default 1e-5; use 1e-8 for conservation)
- `cost`: `"cheap"` (all current relation types require 1 extra compile + run)

---

## ❌ ANTI-PATTERNS: What NOT to Do

### ❌ Qualitative / Directional Relations

```
BAD: "output should increase when input increases"
BAD: "larger input produces larger output"
```

A code with a 2× scaling bug satisfies these but fails an exact quantitative check.
All relations must specify a numerically predictable expected value.

### ❌ System-Level Check on Entire stdout

```
BAD: "the entire stdout of the variant should be identical to the original"
```

Check specific named buffers at specific observation points.

### ❌ Multiple Input Changes in One Variant

```
BAD: "scale input by 2 AND swap rows 1 and 3 in the same variant"
```

Each variant changes exactly one input aspect.

### ❌ Implementation-Specific Relations

```
BAD: "shared memory state should be identical after 3 sync barriers"
BAD: "thread 0 should produce the same partial sum in both runs"
```

Relations must derive from mathematical/physical semantics, not GPU implementation details.

### ❌ SCALING Applied to Nonlinear or Fixed-BC Iterative Solvers

```
BAD: "f(2*b) = 2*f(b)" when Dirichlet boundary conditions in A are NOT also scaled
BAD: "f(2*x) = 2*f(x)" when the kernel applies clamping or a threshold
```

Check `applicable_when` carefully. For iterative solvers with fixed Dirichlet BCs, the solution
does not scale with the source term — the BCs dominate.

### ❌ ADDITIVITY When Output Shift ≠ Input Offset

```
BAD: ADDITIVITY on y = A*x when row sums of A ≠ 1 (shift in output mean = c × mean(row_sums(A)), not c)
BAD: ADDITIVITY on y = α*x when α ≠ 1 (shift in output = α*c, not c)
BAD: ADDITIVITY on iterative solver with fixed Dirichlet BCs (BCs pin boundary values, interior shift ≠ c)
```

ADDITIVITY requires the output to shift by exactly `c` when all inputs shift by `c`.
This holds for reductions (mean shifts by c, sum shifts by N*c but mean stays shifted by c),
and pointwise operations with unit coefficient. For a general linear operator A, the shift in
output mean is `c × mean(row_sums(A))` — if row sums ≠ 1, ADDITIVITY with expected shift = c is
wrong. The source filter will correctly catch this, but a failed source filter wastes a contract
slot. Verify the shift analytically before proposing ADDITIVITY.

### ❌ MONOTONICITY When Equality Is Available

```
BAD priority: use MONOTONICITY for a linear PDE when SCALING clearly applies
```

MONOTONICITY is the last resort. Always prefer equality/quantitative-ratio relations.

### ❌ Null-Factor SYMMETRY/SCALING Relations (Not Currently Supported)

```
BAD: { "type": "SYMMETRY", "factor": null, "name": "horizontal_reflection" }
BAD: { "type": "SCALING",  "factor": null, "name": "negation_symmetry" }
```

**What a null-factor relation is:** A SYMMETRY or SCALING relation with `"factor": null`
requires element-wise array comparison between the original output Y and the transformed
variant output Z. This capability is not currently implemented for these two relation types.

**Why to avoid it:** Such relations are **automatically SKIP'd** during the Stage 0 source
filter — the pipeline detects the null factor and marks them inactive before even attempting to
compile or run the variant. Every null-factor SYMMETRY/SCALING relation you propose wastes a
contract slot.

**What to do instead:** For SYMMETRY relations, only propose relations where `factor` is a known
non-null scalar (e.g., `factor: -1.0` for antisymmetric negation). If the symmetry you have in
mind requires no algebraic transformation of the output but merely "the output is unchanged"
under some input transformation, that is NOT a SYMMETRY relation — it is an INVARIANCE relation
(use `type = INVARIANCE`, `factor = 1.0`). If the property requires element-wise comparison that
no aggregate statistic can capture (e.g., spatial reflection equivariance Z[i] ≈ Y[N−1−i]
per-element), it is not currently supported — skip it and choose a different relation type.

**Formal note:** The current SCALING/SYMMETRY checker compares `Z_stat ≈ factor × Y_stat` for
each scalar statistic (min, max, mean). This requires an explicit numeric factor. INVARIANCE
uses the same machinery with factor=1.0 (compares `Z_stat ≈ Y_stat`). Element-wise array
comparison for non-aggregate properties (reflection, general permutation) is not currently
supported.

### ❌ SCALING or SYMMETRY on Zero-Valued Default Parameters

```
BAD: scale body force Fx by α=2.0, expect u-velocity to scale by 2.0 — when Fx = 0.0 in source config
BAD: negate external field Fy, expect output to negate — when Fy = 0.0 in source config
BAD: scale an input array by α — when the array is initialized to all-zeros before the first kernel call
```

**Why this fails:** If the parameter you are transforming is zero in the default problem configuration,
then `0 × α = 0` and `−0 = 0` — the transformed input is **identical to the original**. The output
will be unchanged (effective factor = 1.0), but the relation expects factor = α (or −1.0). The
SOURCE_verify filter correctly rejects the relation because it fails for the source. The contract slot
is consumed with zero verification benefit, and if every proposed relation has this defect, Stage 0
aborts before any generated code is verified.

**This is the most common cause of all-inactive contract slots** in kernels where the default
benchmark configuration uses zero for one or more physical forcing parameters (body forces, external
fields, gravity, source terms). Many HPC benchmarks default such parameters to zero to isolate the
algorithmic behavior of the solver.

**Mandatory check before proposing any SCALING or SYMMETRY relation:**
In Step 1 (Analyze the Code), locate the initialization or argument-setup code for **every variable
you plan to transform**. Read the actual assigned value:
- If the variable is initialized to `0`, `0.0`, or `{0}` — do **NOT** propose a multiplicative or
  sign-flip relation on it. Any factor applied to zero produces zero; the source filter will reject it.
- If the variable is a constant set by a `#define` or computed expression, evaluate whether it is
  zero in the default compilation. If you cannot determine this from the source code alone, skip
  the relation.
- If the variable has a non-zero default, proceed — but still perform the symbolic trace (Step 2)
  before committing to the relation.

**What to do instead:** If all candidate input parameters for SCALING/SYMMETRY are zero in the
default configuration, pivot to CONSERVATION or MONOTONICITY, which do not require transforming
any input parameter. If neither is applicable, it is correct to propose zero relations — Stage 0
will abort the trial, which is the right outcome: there is no valid T3 verification possible for
this trial configuration.

### ❌ Composite MRs (Single-Concern Principle — Intentionally Not Supported)

```
BAD: apply SCALING then PERMUTATIVE in the same variant
```

Each variant must apply **exactly one** relation. This is an intentional design choice, not
a temporary limitation: combining two transformations in a single variant makes it ambiguous
which transformation a FAIL diagnoses, defeating the per-relation fault-localization that
makes Tier 3 actionable. If you want to test two relations, propose them as two separate
single-concern variants in the contract.

---

## Relation Categories

---

### Category 1: CONSERVATION

**Formal basis:** Murphy et al. ADDITIVE/INCLUSIVE; Chen 1998 path decomposition (shortest path
sub-path property); GENMORPH NumericAddition template; Chen et al. 2019 (output-dependent follow-up
inputs — Concept 2). Both SYMMETRY and CONSERVATION are instances of the Zhou et al. symmetry MRP.

**Formal 3-tuple:**
- `input_relation`: source input used directly; no structural change required
- `output_relation`: |sum_after − sum_before| / |sum_before| < tol; OR sum(Y[0:k]) + sum(Y[k:N]) = sum(Y)
- `applicable_when`: computation has an identifiable conserved quantity or exhibits partial-sum
  decomposition

**Cost:** Very cheap — may use existing passthrough data without any additional variant run.

**Applicable to:**
- Physical simulations with conservation laws (mass, energy, momentum, charge)
- Reduction kernels (sum, integral, dot product)
- Finite element / finite volume codes (flux balance)
- N-body / particle dynamics

**Contraindications:**
- Explicitly dissipative schemes (energy dissipation is the design intent)
- Algorithms that deliberately modify the conserved quantity
- Non-integrable or open-boundary systems

#### 1.1 Physical Quantity Conservation (before/after)

`input_relation`: no input change
`output_relation`: |sum(h_energy_after) − sum(h_energy_before)| / |sum(h_energy_before)| < tol
`applicable_when`: kernel preserves a physical quantity (mass, energy, momentum, charge)

```cpp
// ↓ BEFORE kernel execution (after initial D→H if state starts on device) ↓
double cons_before = 0.0;
for (int i = 0; i < N; i++) cons_before += h_energy[i];
printf("=== BEGIN LASSI-VERIFY METAMORPHIC [energy_before] ===\n");
printf("Relation: conservation\nBuffer: h_energy\nElements: %d\nSum: %.17g\n",
       N, cons_before);
printf("=== END LASSI-VERIFY METAMORPHIC [energy_before] ===\n");

// ... kernel execution, final D→H transfer ...

double cons_after = 0.0;
for (int i = 0; i < N; i++) cons_after += h_energy[i];
printf("=== BEGIN LASSI-VERIFY METAMORPHIC [energy_after] ===\n");
printf("Relation: conservation\nBuffer: h_energy\nElements: %d\nSum: %.17g\n",
       N, cons_after);
printf("=== END LASSI-VERIFY METAMORPHIC [energy_after] ===\n");
```

**Check:** `|cons_after − cons_before| / |cons_before| < tolerance`

#### 1.2 Decomposition / Recurrence Conservation

`input_relation`: run with N−1 elements or one fewer iteration (NumericAddition(−1) template)
`output_relation`: f(N-1) ∘ g(step_N) = f(N) (compose partial result with one additional step)
`applicable_when`: computation is an iterated accumulation with a well-defined step function

**Based on:** GENMORPH pow recurrence MR: `k^(e−1) × k = k^e`. Generalizes to GPU reductions:
partial reduction over N−1 elements, combined with one additional element, equals full reduction.
The passthrough output provides f(N); the variant computes f(N-1).

#### 1.3 Subset Sum Decomposition

`input_relation`: no input change — use existing passthrough data
`output_relation`: sum(Y[0:k]) + sum(Y[k:N]) = sum(Y[0:N]) for some split index k
`applicable_when`: reduction kernel where the passthrough already captures per-element or per-block values

**No additional variant run required** if the passthrough captures the full output array.

---

### Category 2: SYMMETRY

**Formal basis:** Chen 1998 (reversibility / shortest-path symmetry); Murphy et al. PERMUTATIVE;
GENMORPH SequenceFlip; Zhou et al. (symmetry MRP at top of formal hierarchy, confirmed by Li et al.
2025). SYMMETRY and ROTATION (Category 5) are both instances of the symmetry MRP: "the existence
of different viewpoints from which the system appears the same."

**Formal 3-tuple:**
- `input_relation`: apply reflection, sign flip, or sequence reversal to input
- `output_relation`: output is identically reflected/negated (equivariance); OR identical (invariance)
- `applicable_when`: the computation is symmetric or antisymmetric under the transformation

**Cost:** Very cheap — may use existing passthrough data without additional variant run.

**Applicable to:**
- Antisymmetric functions: force fields, signed distances, odd polynomial evaluations
- Symmetric PDEs: isotropic diffusion, Laplace equation with symmetric BCs
- Bidirectional algorithms: shortest-path, undirected graph distances

**Contraindications:**
- Directional computations (upwinding, convection, gravity)
- Asymmetric boundary conditions
- Non-symmetric stencils

#### 2.1 Negation Symmetry (Antisymmetric Function)

`input_relation`: x₂ = −x₁ (negate all input values)
`output_relation`: Z[i] ≈ −Y[i] for all i
`applicable_when`: the kernel computes an odd (antisymmetric) function f(−x) = −f(x)

**Based on:** Chen et al. 2019: `f(−x) = −f(x)` "is still very useful as it will double the
effectiveness of concrete test cases."

```cpp
// Apply AFTER initialization, BEFORE first H→D copy
for (int i = 0; i < N; i++) h_input[i] = -h_input[i];
```

**Check:** `|Z[i] + Y[i]| ≤ atol + rtol × max(|Z[i]|, |Y[i]|)` for each element/statistic.

#### 2.2 Reflection Symmetry (Sequence Flip)

`input_relation`: x₂[i] = x₁[N−1−i] (reverse array order)
`output_relation`: Z[i] ≈ Y[N−1−i] for all i
`applicable_when`: the kernel is equivariant under index reversal (1D stencil with symmetric BCs)

```cpp
// Reverse input array (after initialization, before H→D)
for (int i = 0; i < N/2; i++) std::swap(h_input[i], h_input[N-1-i]);
```

**Check:** Compare variant output element i to original output element N−1−i.

#### 2.3 Problem Reversibility

`input_relation`: swap source and destination (or swap input and output roles)
`output_relation`: f(swap(x,y)) = f(x,y) for scalar outputs (path length, distance, energy)
`applicable_when`: algorithm is symmetric in source/destination; e.g., undirected graph, symmetric solver

**Based on:** Chen 1998: `d(y,x,G) = d(x,y,G)` for undirected shortest path.

---

### Category 3: SCALING (MULTIPLICATIVE)

**Formal basis:** Murphy et al. MULTIPLICATIVE — the canonical MT relation; GENMORPH NumericMultiplication
template; Chen et al. 2019 semi-proving framework: SCALING enables extrapolation from one verified
test case to infinitely many scaled inputs; validated across 5+ independent papers.

**Formal 3-tuple:**
- `input_relation`: x₂ = α × x₁ (all relevant input arrays/scalars multiplied by constant α)
- `output_relation`: f(x₂) = α × f(x₁)
- `applicable_when`: the kernel computes a linear function — output is proportional to input magnitude

**Cost:** Cheap — 1 additional compile + run.

**Discriminating power: HIGHEST.** SCALING is the "proving MR" from the MT literature — it enables
extrapolation from one correct test to infinitely many scaled inputs (Chen et al. 2019). Always listed
first in the fail-fast loop after zero-cost relations.

**Applicable to:**
- Linear algebra: matrix-vector multiply, linear system solve (with homogeneous BCs)
- Linear PDE solvers: source/forcing term scaling
- Reduction kernels: sum, dot product, L2-norm squared
- Any kernel computing a linear function of its input

**Contraindications:**
- Nonlinear operations (squaring, clamping, max/min pooling, sigmoid, exp/log)
- Iterative solvers with fixed Dirichlet BCs (solution does not scale with source when BCs dominate)
- Normalized operations (output is always unit length)
- Operations with positivity constraints (avoid α < 0 when density/temperature must be positive)
- Fixed convergence thresholds (stopping criterion unchanged by scaling)

**Choosing α:** Default α = 2.0. Avoid α = 0 (trivially satisfied) and α = 1 (no change). Avoid
α < 0 when physical positivity constraints exist. Consider α derived from kernel constants (grid
resolution, timestep ratio) for physical meaning.

#### 3.1 Scalar Input Scaling (General)

`input_relation`: h_input[i] *= alpha for all i; scale ALL linearly contributing input arrays
`output_relation`: Z_stat ≈ alpha × Y_stat for each statistic (min, max, mean, each sample)
`applicable_when`: kernel computes a linear map from input to output

```cpp
const double alpha = 2.0;
// AFTER initialization, BEFORE first H→D copy
for (int i = 0; i < N; i++) h_input[i] *= alpha;
// If multiple input arrays contribute linearly, scale ALL of them
```

**METAMORPHIC block** (label matches the PASSTHROUGH label for this buffer):
```cpp
printf("=== BEGIN LASSI-VERIFY METAMORPHIC [final_output] ===\n");
printf("Relation: scaling\nFactor: %.17g\n", alpha);
printf("Buffer: h_output\nElements: %d\n", N);
double mn = h_output[0], mx = h_output[0], sm = 0.0;
for (int i = 0; i < N; i++) {
    if (h_output[i] < mn) mn = h_output[i];
    if (h_output[i] > mx) mx = h_output[i];
    sm += h_output[i];
}
printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n", mn, mx, sm/N);
printf("Sample_first10:");
for (int i = 0; i < 10 && i < N; i++) printf(" %.17g", h_output[i]);
printf("\n=== END LASSI-VERIFY METAMORPHIC [final_output] ===\n");
```

**Check:** `|Z_stat − alpha × Y_stat| ≤ atol + rtol × max(|Z_stat|, |alpha × Y_stat|)` per statistic.

#### 3.2 RHS / Source Term Scaling (Linear Solvers)

`input_relation`: scale source/forcing vector b by alpha; also scale boundary condition rows if any
`output_relation`: solution vector x scales by alpha
`applicable_when`: kernel solves a linear system Ax=b with homogeneous (zero) Dirichlet BCs,
OR all boundary condition values are also scaled by alpha

**Note:** This is the most common applicable SCALING for iterative linear solvers (stationary
methods, Krylov methods). If Dirichlet BCs in the rows of A are fixed at non-zero values and
NOT also scaled, this relation does not apply — SOURCE_verify will fail it and filter it inactive.

⛔ **Do NOT use convergence scalars or residuals as observation points for SCALING/SYMMETRY.**
Use the primary output array (the converged solution field or vector). The convergence residual
at exit legitimately differs between the reference and an optimized implementation because FP
reordering changes the convergence trajectory — the solver may terminate at a slightly different
residual value even though the output field is numerically equivalent. Including the residual
as an observation point will cause the source filter to FAIL the relation incorrectly, preventing
it from being applied to any generated code.

---

### Category 4: PERMUTATIVE (INDEX PERMUTATION)

**Formal basis:** Murphy et al. PERMUTATIVE; Chen 1998 (row interchange / column interchange for
Gaussian elimination — the most direct GPU linear algebra precedent); GENMORPH PermuteParameters;
Segura 2016 (ML feature permutation); Segura 2020 (ML attribute ordering invariance).

**Formal 3-tuple:**
- `input_relation`: x₂ = permute(x₁, σ) — apply permutation σ to input array elements or matrix rows/columns
- `output_relation`: f(x₂) = permute(f(x₁), σ) (equivariance — output permutes correspondingly);
  OR f(x₂) = f(x₁) (invariance — for order-invariant computations)
- `applicable_when`: the input elements are interchangeable in meaning, or the computation is
  equivariant under the specified permutation

**Cost:** Cheap — 1 additional compile + run.

**Applicable to:**
- Matrix operations: row interchange in Ax=b; column interchange in Ax=b
- N-body / particle codes: particles are interchangeable by index
- Commutative reductions: sum, max, min (order-invariant)
- Graph algorithms: vertex relabeling should relabel output correspondingly

**Contraindications:**
- Stencil codes where array index encodes spatial position (permuting breaks geometry)
- Operations where ordering is semantically meaningful (time series, upwinding)
- Non-commutative operations

#### METAMORPHIC Output Block Format for PERMUTATIVE

The METAMORPHIC block for PERMUTATIVE includes `Swap_idx1`, `Swap_idx2`, `Sample_at_idx1`, and
`Sample_at_idx2` fields so the checker can perform **element equivariance checks**
(Z[idx1] ≈ Y[idx2] and Z[idx2] ≈ Y[idx1]):

```
=== BEGIN LASSI-VERIFY METAMORPHIC [label] ===
Relation: <relation_name>
Permutation: swap
Swap_idx1: <integer index of first swapped element>
Swap_idx2: <integer index of second swapped element, != Swap_idx1>
Buffer: <buffer_name>
Elements: <N>
Statistics: min=%.17g, max=%.17g, mean=%.17g
Sample_first10: %.17g %.17g ...  (up to 10 values, space-separated)
Sample_at_idx1: %.17g
Sample_at_idx2: %.17g
=== END LASSI-VERIFY METAMORPHIC [label] ===
```

**`Sample_at_idx1`** = output buffer value at Swap_idx1 (i.e., `lv_host_buf[swap_idx1]`).
**`Sample_at_idx2`** = output buffer value at Swap_idx2 (i.e., `lv_host_buf[swap_idx2]`).

These give the checker the Z values at the exact swapped positions. Y values (for comparison) come
from the passthrough `Sample_first10` (indices 0–9) and `Sample_last10` (indices N-10 to N-1).
**For equivariance to fire**, at least one swap index must fall in the first-10 OR last-10 range.
If both indices are in the interior (outside both captured ranges), only aggregate invariance is
checked. Do NOT use index 0 or N-1 (often fixed boundary elements).

**Recommended index strategy:**
- `idx1` from [2, 5] (first10 range — near-boundary start)
- `idx2` from [N-5, N-2] (last10 range — near-boundary end)
- For small arrays (N ≤ 20): both indices from [1, 9]

If Swap_idx1/Swap_idx2 are absent from the block, only aggregate invariance is checked (fallback).

#### 4.1 Row Interchange (Matrix Equivariance)

`input_relation`: swap rows i and j of matrix A AND corresponding entries of RHS vector b
`output_relation`: Z = Y (solution vector is identical)
`applicable_when`: rows represent interchangeable equations (mathematical problem unchanged by row permutation)

**Based on:** Chen 1998 Gaussian elimination: swap rows i and j in A and b → x is identical.

```cpp
// idx1 from first10 range, idx2 from last10 range — tests across interior
const int swap_idx1 = 3;           // first10 range, not 0 (boundary)
const int swap_idx2 = (int)N - 4;  // last10 range, not N-1 (boundary)
for (int k = 0; k < N; k++) std::swap(h_A[swap_idx1*N + k], h_A[swap_idx2*N + k]);
std::swap(h_b[swap_idx1], h_b[swap_idx2]);
// After D→H, also print in METAMORPHIC block:
//   printf("Sample_at_idx1: %.17g\n", (double)h_x[swap_idx1]);
//   printf("Sample_at_idx2: %.17g\n", (double)h_x[swap_idx2]);
```

**Check:** Aggregate invariance (min/max/mean of Z ≈ Y) + element equivariance
`Z[swap_idx1] ≈ Y[swap_idx2]` and `Z[swap_idx2] ≈ Y[swap_idx1]`.

**Note for stencil / fixed-BC iterative solvers:** Choose interior rows only (avoid boundary
rows that encode Dirichlet BCs). Swapping a boundary row with an interior row changes the
mathematical problem — SOURCE_verify will catch this and the relation will be filtered inactive.

#### 4.2 Particle / Element Index Permutation (Array Equivariance)

`input_relation`: apply permutation σ to particle index ordering in h_pos, h_vel, h_mass, etc.
`output_relation`: Z[σ(i)] ≈ Y[i] for all i (output permuted by same permutation σ)
`applicable_when`: elements are independent particles/points with no spatial-ordering semantics

```cpp
// idx1 from first10 range, idx2 from last10 range
const int swap_idx1 = 2;
const int swap_idx2 = (int)N - 3;
std::swap(h_pos[swap_idx1], h_pos[swap_idx2]);
std::swap(h_vel[swap_idx1], h_vel[swap_idx2]);
// swap all associated arrays by the same permutation
// After D→H, print in METAMORPHIC block:
//   printf("Sample_at_idx1: %.17g\n", (double)h_force[swap_idx1]);
//   printf("Sample_at_idx2: %.17g\n", (double)h_force[swap_idx2]);
```

**Check:** Aggregate invariance (min/max/mean of Z ≈ Y) + element equivariance
`Z[swap_idx1] ≈ Y[swap_idx2]` and `Z[swap_idx2] ≈ Y[swap_idx1]`.

#### 4.3 Reduction Invariance (Commutative Order)

`input_relation`: reverse input array order (or apply a fixed deterministic permutation)
`output_relation`: Z_scalar ≈ Y_scalar (same scalar result within tolerance)
`applicable_when`: the kernel computes a commutative scalar reduction (sum, max, dot product)

**Note on FP non-associativity:** GPU parallel reductions may produce slightly different results
when input order changes (tree-reduction ordering differs). Use a **generous tolerance** (1e-3 to
1e-4 for large N). Use a fixed, deterministic permutation (array reverse) rather than random
shuffling to ensure reproducibility.

---

### Category 5: ROTATION INVARIANCE

**Formal basis:** Ding et al. 2016 (MR3 — orientation invariance for sphere scatterers);
Segura 2020 (drone scene rotation finding the sunlight-not-rotated bug); Zhou et al. symmetry MRP
(confirmed via Li et al. 2025). This is the GPU physics domain-specific extension beyond
GENMORPH's template set — GENMORPH targets Java numeric/sequence types and has no spatial
coordinate geometry.

**Formal 3-tuple:**
- `input_relation`: x₂ = rotate(x₁, θ) — apply rotation to spatial domain (coordinates, BCs)
- `output_relation`: f(x₂) ≈ rotate(f(x₁), θ) — output field rotates correspondingly;
  OR f(x₂) = f(x₁) for scalar invariants (e.g., total energy)
- `applicable_when`: the physics has no preferred spatial direction (isotropic); domain and BCs
  are rotationally symmetric

**Cost:** Cheap — 1 additional compile + run. May require grid reindexing for 2D/3D arrays.

**Applicable to:**
- 2D/3D isotropic PDE solvers (isotropic diffusion, Laplace equation with circular domain)
- Radially symmetric particle potentials (Lennard-Jones, Coulomb)
- Stencils with equal x and y coefficients and periodic or symmetric BCs

**Contraindications:**
- Directional physics: convection (wind direction), gravity, electromagnetic field with fixed direction
- Non-isotropic boundary conditions (e.g., different BC values on left vs right edges)
- 1D codes (rotation is trivial)
- Codes where the grid index encodes a fixed physical direction

**The sunlight-not-rotated bug (Segura 2020):** A drone simulation with scene rotation failed
because sunlight direction was not rotated with the scene — shadows fell on the landing pad in
some orientations, causing vision system failure. This is exactly the class of bug ROTATION
INVARIANCE catches: a component that does not transform correctly under rotation.

#### 5.1 90-Degree Grid Rotation (2D Square Domain)

`input_relation`: rotate 2D input grid 90°: h_field[j*N + i] → h_field[i*N + (N-1-j)]
`output_relation`: Z[j*N + i] ≈ Y[i*N + (N-1-j)] (correspondingly rotated output)
`applicable_when`: square 2D grid, isotropic stencil, symmetric BCs in both directions

#### 5.2 180-Degree Reflection (2D, more practical)

`input_relation`: reverse both x and y indices: h_field[i*N + j] → h_field[(N-1-i)*N + (N-1-j)]
`output_relation`: Z[(N-1-i)*N + (N-1-j)] ≈ Y[i*N + j]
`applicable_when`: 2D stencil with symmetric BCs; works for non-square grids (use Nx, Ny separately)

```cpp
// 180-degree reflection: reverse entire flat array
for (int i = 0; i < (Nx*Ny)/2; i++)
    std::swap(h_field[i], h_field[Nx*Ny - 1 - i]);
```

---

### Category 6: MONOTONICITY

**Formal basis:** Segura 2016 (non-equality MRs are valid by Definition 1); Chen et al. 2019
(Concept 3 — non-equality MRs legitimate but weaker). Ding et al. 2016 confirmed: qualitative/
directional MRs are significantly weaker than quantitative relations for mutation testing.

**Formal 3-tuple:**
- `input_relation`: no input change — standard execution
- `output_relation`: residual[k] ≥ residual[k+1] − tolerance × residual[k] for all iterations k
- `applicable_when`: the algorithm is an iterative solver that converges monotonically; per-iteration
  residual is printed to stdout

**Cost:** Cheap — 1 extra run that prints per-iteration residuals; or zero extra runs if residuals
already appear in the baseline execution output.

**Discriminating power: LOWEST.** A code can converge monotonically to a completely wrong answer.
MONOTONICITY is a necessary but not sufficient condition for correctness. Use ONLY when no
quantitative equality relation applies.

**Applicable to:**
- Steepest descent / gradient descent
- Stationary iterative linear solvers with strict monotone convergence guarantees

**Contraindications:**
- Conjugate gradient (CG): energy can temporarily increase — CG is NOT monotone in residual
- Non-monotone line search methods
- Stochastic optimization: inherently non-monotone
- Any method with momentum or oscillatory convergence behavior

**⚠ Warning:** A code that converges monotonically to the wrong answer passes this check. Always
prefer SCALING, CONSERVATION, or PERMUTATIVE. Use MONOTONICITY only as the final option when no
quantitative relation is applicable to the kernel.

#### 6.1 Residual Monotone Decrease

`input_relation`: no input change
`output_relation`: residual[k] ≥ residual[k+1] (non-increasing) for all k
`applicable_when`: iterative solver with a monotone convergence guarantee (not CG)

**Before writing the METAMORPHIC blocks, identify the loop structure of the solver:**

**Case A — Single-level iteration loop** (the iteration loop is the outermost loop):
Cap the number of printed blocks to prevent stdout overflow.

```cpp
#define LASSI_MT_MAX_PRINT 200
for (int iter = 0; iter < max_iter; iter++) {
    kernel<<<blocks, threads>>>(...);
    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    double residual = compute_residual(h_A, h_b, h_x, N);
    if (iter < LASSI_MT_MAX_PRINT) {
        printf("=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\n", iter);
        printf("Relation: monotonicity\nIteration: %d\nResidual: %.17g\n", iter, residual);
        printf("=== END LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\n", iter);
    }
    if (converged) break;
}
```

**Case B — Two-level loop structure** (outer timestep/sweep loop wrapping an inner solver loop):
Place the block AFTER the inner loop, once per outer iteration. Printing inside the inner loop
would emit O(timesteps × inner_iterations) lines — potentially millions — and must never be done.

```cpp
for (int t = 0; t < num_timesteps; ++t) {
    double final_residual = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        kernel<<<blocks, threads>>>(...);
        cudaDeviceSynchronize();
        final_residual = compute_residual(...);
        if (final_residual < tol) break;
    }
    // ✅ Block is OUTSIDE the inner loop — one block per timestep
    printf("=== BEGIN LASSI-VERIFY METAMORPHIC [residual_t_%d] ===\n", t);
    printf("Relation: monotonicity\nIteration: %d\nResidual: %.17g\n", t, final_residual);
    printf("=== END LASSI-VERIFY METAMORPHIC [residual_t_%d] ===\n", t);
}
```

**Check:** `residual[k] >= residual[k+1] - tolerance * residual[k]` for all k (allow tiny
numerical noise in "near-equal" consecutive values).

---

### Category 7: ADDITIVITY (TRANSLATIONAL)

**Formal basis:** Murphy et al. ADDITIVE — the direct companion to MULTIPLICATIVE (SCALING) that
completes the linearity test. Linear operator theory: superposition principle `L(x + c) = L(x) + L(c)`.
Validated in ML testing as feature translation invariance (Segura 2016). Together with SCALING,
forms a complete two-property linearity verification:
- SCALING tests **homogeneity**: `f(α×x) = α×f(x)`
- ADDITIVITY tests **superposition**: `f(x+c) = f(x) + c` (when linear coefficient = 1)

**Why ADDITIVITY catches bugs SCALING misses:** A refactoring can preserve the multiplicative
response `f(αx) = αf(x)` while introducing an incorrect additive bias — for example, a constant
offset introduced by incorrect initialization, a spurious term added in accumulation, or a missing
subtraction in a centered computation. SCALING would pass; ADDITIVITY would fail.

**Formal 3-tuple:**
- `input_relation`: x₂ = x₁ + c (add a constant offset c to all relevant input array elements)
- `output_relation`: f(x₂) = f(x₁) + c (all output statistics and per-element values shift by c)
- `applicable_when`: the kernel computes a reduction (mean, sum) or pointwise-linear transform
  where adding c to all inputs shifts all output statistics by exactly c

**Implementation restriction:** The checker uses the contract `factor` field as the expected
shift for ALL statistics (min, max, mean) and per-element samples. This is exact only when the
linear coefficient is unity: mean(x+c) = mean(x)+c, min(x+c) = min(x)+c, max(x+c) = max(x)+c.
For kernels with non-unit coefficients (y = A*x where row sums ≠ 1), output shift ≠ c — do not
propose ADDITIVITY for such kernels.

**Cost:** Cheap — 1 additional compile + run.

**Discriminating power: HIGH (complementary to SCALING).** Tests a different fault class:
additive biases, spurious constant terms, incorrect initialization offsets.

**Applicable to:**
- Reduction kernels (sum, mean): mean(x + c) = mean(x) + c exactly
- Pointwise-linear kernels with identity coefficient: y[i] = x[i] → y[i] + c
- Linear algebra with unit row-sum property (stochastic matrices, probability kernels)
- Distance kernels where the offset propagates linearly through the computation

**Contraindications:**
- Kernels with non-unit linear coefficients (y = A*x, A's row sums ≠ 1) — shift ≠ c
- Nonlinear operations (squaring, clamping, exp, log, min/max pooling) — shift ≠ c
- Iterative solvers with fixed Dirichlet BCs — BCs pin boundary values; interior shift ≠ c
- Operations with positivity/range constraints — choose c carefully to stay in valid domain
- Normalized outputs (z-score, softmax) — normalization absorbs any constant offset

**Choosing c (the offset):** Default c = 1.0. Avoid c = 0 (trivially satisfied). Choose c that
keeps all input values within the valid physical domain. Use a value that is small relative to
input magnitude to avoid numerical instability in iterative codes.

#### 7.1 Constant Offset (Reduction / Pointwise Linear)

`input_relation`: add constant c to all elements of the relevant input array(s)
`output_relation`: Z_min ≈ Y_min + c; Z_max ≈ Y_max + c; Z_mean ≈ Y_mean + c; Z_sample[i] ≈ Y_sample[i] + c
`applicable_when`: kernel computes a reduction (mean, sum) or pointwise-linear transform where
output shifts by exactly c when inputs shift by c

```cpp
const double offset = 1.0;
// AFTER initialization, BEFORE first H→D copy:
for (int i = 0; i < N; i++) h_input[i] += offset;
// If multiple input arrays contribute linearly, offset ALL of them

// After final D→H transfer, print METAMORPHIC block:
printf("=== BEGIN LASSI-VERIFY METAMORPHIC [output_label] ===\n");
printf("Relation: additivity\nOffset: %.17g\n", offset);
printf("Buffer: h_output\nElements: %d\n", N);
double mn = h_output[0], mx = h_output[0], sm = 0.0;
for (int i = 0; i < N; i++) {
    if (h_output[i] < mn) mn = h_output[i];
    if (h_output[i] > mx) mx = h_output[i];
    sm += h_output[i];
}
printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n", mn, mx, sm/N);
printf("Sample_first10:");
for (int i = 0; i < 10 && i < N; i++) printf(" %.17g", h_output[i]);
printf("\n=== END LASSI-VERIFY METAMORPHIC [output_label] ===\n");
```

**Check:** `|Z_stat − (Y_stat + offset)| ≤ atol + rtol × max(|Z_stat|, |Y_stat + offset|)` per statistic.
The `offset` value is taken from the contract's `factor` field (same JSON field, additive interpretation for ADDITIVITY).

---

### Category 8: INVARIANCE (OUTPUT INVARIANCE UNDER INPUT TRANSFORMATION)

**Applicability guard (read first):** INVARIANCE applies when the kernel's output is genuinely
invariant under some non-trivial input transformation. Do NOT propose INVARIANCE on kernels
where the output depends on input magnitude (use SCALING), sign (use SYMMETRY), or order
(use PERMUTATIVE for equivariance under permutation). INVARIANCE is for cases where applying a
real, non-degenerate transformation T to the input leaves the output mathematically unchanged.

**Formal basis:** Zhou et al. symmetry MRP (the "different viewpoints from which the system
appears the same" formal hierarchy, confirmed by Li et al. 2025 as a top-level MRP);
Chen 1998 equivalence partitioning under input-equivalence relations; ML feature-equivalence
invariance from Segura 2016, 2020. Distinct from PERMUTATIVE (Category 4), which is the more
specific case where T is a permutation of input elements; INVARIANCE generalizes to ANY
input transformation (permutation, relabeling, uniform shift, value negation, joint scaling
of co-normalized groups) so long as the output is unchanged.

**Formal 3-tuple:**
- `input_relation`: x₂ = T(x₁), where T is a non-trivial input transformation (must actually
  change x — applying T to a zero-valued input or proposing T = identity is degenerate)
- `output_relation`: f(x₂) ≈ f(x₁) within tolerance (aggregate stats AND per-element samples)
- `applicable_when`: the kernel's algorithmic semantics ignore the dimension that T modifies
  (e.g., the kernel reads inputs only through equality comparisons, so value relabeling does
  not change the output)

**Cost:** Cheap — 1 additional compile + run.

**Discriminating power: HIGH for kernels where invariance is the natural correctness language.**
INVARIANCE catches bugs that change *what the algorithm is sensitive to* — a refactor that
silently introduces sensitivity to attribute order, value magnitude, or input ordering will
fail an INVARIANCE relation that the source code satisfied. Especially strong for categorical /
equality-based kernels where SCALING and ADDITIVITY are nonsensical.

**Applicable to:**
- Categorical / equality-based kernels: pairwise mismatch counting on discrete attributes,
  hash functions where the output set is order-independent, set membership tests, and any
  computation whose output depends only on equality/inequality of input values, not their
  magnitudes
- Co-normalized distance kernels: z-score-normalized DTW, LB_Keogh, cross-correlation,
  standardized regression (joint α-scaling of data + mean + std)
- Sort-as-set computations: bag-of-elements operations where the output is a multiset, not
  a sequence

**Contraindications:**
- Any kernel whose output depends on input magnitude (use SCALING instead)
- Any kernel whose output depends on input sign (use SYMMETRY instead)
- Any kernel whose output depends on input order in a structured way (use PERMUTATIVE for
  equivariance instead — i.e., output permutes by the same permutation as the input)
- Numerical solvers (PDE, optimization) where the output is generally sensitive to almost
  every input dimension
- Transformations that are degenerate on the default input configuration (e.g., proposing
  "scale by α" when the input array is all zeros — see ❌ Zero-Valued Default Parameters)

**Choosing the input transformation T.** The transformation must be:
1. **Non-trivial** — actually change x; identity replay (no input change) is rejected.
2. **Algorithmically meaningful** — exercise a specific part of the kernel implementation
   (different bug surface than other relations already in the contract).
3. **Source-grounded** — trace through the kernel symbolically and confirm output is
   unchanged for at least one representative input value before committing to the relation.

The contract `factor` field is **always 1.0** for INVARIANCE (output ratio is unity by
definition); the input transformation itself is described in the `description` and named
in the `Invariance_under:` line of the METAMORPHIC block.

#### 8.1 Attribute / Column-Order Invariance

`input_relation`: apply the same permutation σ to the attribute layout within every instance
  (e.g., reverse column order: `data[i*A + k] ← data[i*A + (A-1-k)]` for every row i)
`output_relation`: Z ≈ Y (output buffer identical within tolerance)
`applicable_when`: the kernel reduces over attribute positions in a way that is independent
  of the order (e.g., counting mismatches, summing pairwise products with no positional
  weighting); the attribute-order is the inner reduction dimension

**Example (categorical attribute kernel where output is a sum over per-attribute equality
checks):**
```cpp
// Reverse the attribute order within every input row — apply AFTER initialization,
// BEFORE the first H→D copy.
for (int i = 0; i < NUM_ROWS; i++) {
    for (int k = 0; k < ATTRIBUTES_PER_ROW / 2; k++) {
        std::swap(h_data[i * ATTRIBUTES_PER_ROW + k],
                  h_data[i * ATTRIBUTES_PER_ROW + (ATTRIBUTES_PER_ROW - 1 - k)]);
    }
}
```

After the final D→H transfer, emit the METAMORPHIC block at the same `[label]` as the
original PASSTHROUGH block, with `Relation: <relation_name>`,
`Invariance_under: attribute_order_reversal`, and the standard
`Statistics: ... / Sample_first10: ...` lines.

#### 8.2 Value-Relabeling Invariance (Categorical Alphabet Bijection)

`input_relation`: apply a fixed bijection π to the categorical alphabet across the entire
  dataset (e.g., map every value v → (v + k) mod ALPHABET_SIZE for some fixed k ≠ 0)
`output_relation`: Z ≈ Y
`applicable_when`: the kernel uses input values only through equality comparisons (==, !=,
  XOR for char-equality testing) — the specific value labels do not enter the computation,
  only their equivalence classes

**Example (categorical attribute kernel):**
```cpp
// Add a constant shift to every value, then take mod ALPHABET_SIZE — preserves equality
// structure (a == b  iff  ((a + k) % A) == ((b + k) % A)). For char-data with no explicit
// alphabet size, simple XOR with a fixed constant works for the same reason.
const int relabel_shift = 7;
for (int i = 0; i < NUM_ROWS * ATTRIBUTES_PER_ROW; i++) {
    h_data[i] = (char)(((int)h_data[i] + relabel_shift) % ALPHABET_SIZE);
}
```

#### 8.3 Uniform Value Shift Invariance

`input_relation`: add a constant c to every input value (different c from the relabeling case;
  here c need not preserve membership in the original alphabet — it just shifts all values
  by the same amount)
`output_relation`: Z ≈ Y
`applicable_when`: the kernel computes pairwise differences or equality predicates where
  uniform shift cancels (e.g., `data[i] - data[j]` becomes `(data[i] + c) - (data[j] + c)`;
  `data[i] == data[j]` is unchanged under shared additive shift)

**Example:**
```cpp
const int uniform_shift = 5;
for (int i = 0; i < NUM_ROWS * ATTRIBUTES_PER_ROW; i++) h_data[i] = h_data[i] + uniform_shift;
```

#### 8.4 Negation Invariance (Equality-Class-Preserving Sign Flip)

`input_relation`: negate or bitwise-NOT every input value (must preserve the equality
  structure — e.g., bitwise-NOT on integers, sign-flip on signed values where the
  comparison is on equality not ordering)
`output_relation`: Z ≈ Y
`applicable_when`: the kernel uses input values only through equality comparisons AND the
  negation is a bijection on the input domain (so it preserves equality classes); distinct
  from SYMMETRY (Category 2.1) where negation produces a NEGATED output, not an unchanged one

#### 8.5 Co-Normalized Group Invariance (Joint α-Scaling)

`input_relation`: multiply every member of a co-normalized group (data, precomputed mean,
  precomputed std) by the same positive factor α
`output_relation`: Z ≈ Y
`applicable_when`: the kernel computes `(x − mean) / std` (z-score normalization) somewhere
  in its body; the normalized value is unchanged under joint α-scaling, so the output is
  unchanged

For the co-normalized group case, use `type = INVARIANCE` with `factor = 1.0`. Do NOT
express it as SCALING with factor=1.0 — the validator will drop SCALING+factor=1.0 as a
degenerate relation.

---

### Category 9: SENSITIVITY (OUTPUT MUST CHANGE UNDER INPUT PERTURBATION)

**Applicability guard (read first):** SENSITIVITY is the explicit logical complement of
INVARIANCE (Category 8). It applies when the kernel's output is genuinely sensitive to a
specific input dimension — flipping a bit or changing a byte in that dimension MUST produce
a different output. Do NOT propose SENSITIVITY on dimensions where the kernel is legitimately
invariant (use INVARIANCE), where the response is linear and predictable (use SCALING or
ADDITIVITY), or where the response follows a specific symmetry (use SYMMETRY). SENSITIVITY
is for cases where the *existence* of a response can be asserted but the magnitude or shape
of the response cannot be predicted with a closed-form formula.

**Formal basis:** The "robustness MRP" companion to invariance in the symmetry-MRP
hierarchy (Zhou et al.; Li et al. 2025): if invariance asserts "the system looks the same
from this viewpoint," sensitivity asserts "the system MUST look different from this
viewpoint." Shannon's cryptographic principles of confusion and diffusion (1949): a cipher's
output must be sensitive to every bit of key and plaintext. The avalanche property (Feistel
1973): a single-bit change in input should produce ~50% bit-difference in output. In
metamorphic testing literature, SENSITIVITY is the negation of equivalence-partitioning
relations — robustness testing in the sense of Segura et al. 2020.

**Formal 3-tuple:**
- `input_relation`: x₂ = T(x₁), where T is a non-trivial input perturbation (single-bit
  flip, byte change, or element value change) on a specific dimension of the input.
- `output_relation`: f(x₂) ≠ f(x₁) — output must differ beyond floating-point noise
  tolerance. The MAGNITUDE of the difference is not predicted; only its existence.
- `applicable_when`: the kernel's algorithmic semantics genuinely depend on the dimension
  T modifies. The SOURCE_verify filter empirically confirms this: relation is active iff
  source's perturbed output actually differs from source's un-perturbed output.

**Cost:** Cheap — 1 additional compile + run (same as INVARIANCE).

**Discriminating power: HIGH for nonlinear / non-relational kernels where SCALING and
ADDITIVITY do not apply.** Specifically catches:
- Variants stub-implemented as memcpy from a precomputed buffer or from a constant defined
  in the source (e.g., reading the source's expected-answer constant directly).
- Variants that short-circuit per-element computation by reading source-side constants.
- Variants that selectively ignore the input dimension named in `input_variables`.

**Companion to INVARIANCE:** INVARIANCE catches variants that GAIN false sensitivity to
inputs that should not matter; SENSITIVITY catches variants that LOSE true sensitivity to
inputs that SHOULD matter. Together they fence off the "what does the variant actually
attend to" question from both sides.

**Applicable to:**
- Cryptographic kernels: ciphers (chacha20, AES), MACs, KDFs — output must respond to every
  bit of key, nonce, plaintext.
- Hash functions designed with avalanche: murmurhash3, FNV, SipHash, BLAKE — one bit of
  input must change a substantial fraction of output bits.
- Search / lookup kernels: changing the query value should produce a different result.
- Sort kernels (non-stable): permuting input order changes output order.
- Any nonlinear data-transforming kernel whose output is genuinely data-dependent.

**Contraindications:**
- Kernels with genuine invariance to the chosen perturbation (use INVARIANCE).
- Kernels with predictable linear response (use SCALING or ADDITIVITY — these are stronger
  forms of sensitivity that also predict the magnitude of response).
- Kernels with predictable symmetric response (use SYMMETRY).
- Aggregating / reduction kernels where the perturbation gets averaged below the noise
  floor (a 1-byte change in a 16 MB sum may not move any statistical digit). For these,
  choose a larger perturbation or a different relation type.
- **A single-element perturbation on a shared output structure that many independent
  operations write into, where the update is NOT invertible.** This covers atomic writes to
  a shared table, scatter operations into overlapping output regions, hash insertions with
  collisions, or any design where a large number of updates act on the same output space —
  but only when the update operation itself is destructive. **Recognize this with a single,
  domain-agnostic self-check question: is the shared update operation invertible — does an
  earlier change survive as a recoverable difference in the final state no matter what lands
  afterward?** Operations like XOR-fold or modular addition ARE invertible: perturbing a
  value before any updates apply changes the final state by a fixed, provable delta
  regardless of how many other updates subsequently touch the same location (e.g. for XOR,
  `final = initial ⊕ all_updates`, so perturbing `initial` shifts `final` by exactly that
  same perturbation, always). Operations like plain overwrite (`a[i] = x`), keep-the-max, or
  keep-the-last-write are NOT invertible: a later write to the same location genuinely and
  permanently erases the earlier one, with zero recoverable trace.
  - **If the update is invertible:** single-element SENSITIVITY is still a valid choice — the
    perturbation's effect is real and provable, it may simply be too localized for
    Statistics/Sample_first10 to happen to land on in a large buffer. This is a MEASUREMENT
    problem, not an algorithm problem, and the framework's exhaustive checksum comparison
    (integer/byte buffers only) is built to catch exactly this — do not avoid the relation on
    this basis.
  - **If the update is NOT invertible:** single-element SENSITIVITY on that shared buffer is
    genuinely unfalsifiable — the effect can be legitimately erased with nothing left to
    detect, by any comparison tool. Perturb a large enough FRACTION of the input that the
    aggregate statistics must shift, or choose a relation type that does not require one
    point's effect to individually survive to the final readout.
  - **This contraindication is about the perturbation, not the kernel as a whole.** A kernel
    with a lossy shared-write update can still be validly tested with a SENSITIVITY relation
    that perturbs something broader than one element of the shared buffer (e.g. a scalar
    parameter that shifts the whole computation) — that is not a single-element perturbation
    on the shared structure and this contraindication does not apply to it.
- Stochastic / non-deterministic kernels whose output varies even without perturbation.

**Choosing the input perturbation T.** The perturbation must be:
1. **Non-trivial** — actually modify x; identity (no change) is rejected at source filter.
2. **Algorithmically meaningful for the chosen dimension** — perturb an input the kernel
   reads; do not perturb dead inputs.
3. **Source-grounded** — SOURCE_verify must empirically demonstrate that f(T(x)) ≠ f(x).
   This is the source filter; relations where source happens to be insensitive get filtered
   inactive.

**Multiple perturbations per SENSITIVITY relation (applied automatically).** The framework
probes each SENSITIVITY relation with MORE THAN ONE perturbation of the chosen input
dimension — a MINIMAL bump (the smallest change that could produce a response) AND a LARGE
generic OUTLIER (one element pushed far outside the observed min–max of the sampled input).
Both are applied IDENTICALLY to the source and to every generated variant, so the source
filter's guarantee transfers to each variant test. The outlier guarantees a response from
rank / threshold / argmax / comparison kernels where a minimal bump may not cross a decision
boundary. **Verdict rule:** at the source filter, only perturbations the SOURCE responds to
are kept ("source-active"); a variant then PASSES the relation only if it responds to EVERY
source-active perturbation (a variant that ignores any of them is failing that input
dimension). This stays SINGLE-CONCERN — every perturbation probes the SAME one property
("output is sensitive to input dimension D"); the perturbations are just multiple stimuli for
that property, not multiple properties. You therefore do NOT need to enumerate perturbations
in the contract; propose the SENSITIVITY relation naming the sensitive input dimension and the
framework supplies the perturbation set. Express any perturbation you do describe GENERICALLY
in terms of the input's own observed range — never a value specific to one kernel.

**The outlier perturbation must stay within the kernel's actual valid domain.** "Far outside
the observed range" means far outside the range actually present in the sampled input array —
NOT the full numeric range of the storage type. If the code elsewhere reveals a hard domain
constraint on an input (a fixed-size lookup/histogram table indexed directly by the value, or
an existing invariant assertion checking the value's range), the outlier must stay within that
domain. Pushing a value outside it produces undefined behavior (e.g. an out-of-bounds table
index) rather than a legitimate large-but-valid stimulus, and the framework's Stage-0 filter
will reject such a perturbation and ask for a corrected one. If the chosen input dimension
instead controls a loop bound, buffer size, or repeat/iteration count rather than a per-element
value, an outlier pushed far outside the observed range can blow up runtime rather than
producing an invalid value — bound it to a value that still completes within the harness's
execution timeout, not just one that is numerically far from the observed range.

**Large output arrays and locally-confined perturbations.** A perturbation to a single
element only ever affects a small, local region of a kernel's output — for a huge output array
that region can be invisible to generic aggregate/sample-based comparison. When authoring the
actual perturbation code (a separate, later step — not part of contract proposal), the
framework's instructions offer an OPTIONAL local-response probe for kernels whose output
preserves a bounded positional correspondence to input (elementwise maps, stencils/windowed
kernels, nearest-neighbor/interpolation). It does not apply to reductions, global aggregates,
sorts, or scatter/gather kernels with data-dependent output indices — nothing to propose here,
just be aware the eventual perturbation code may or may not include it depending on the
kernel's structure.

The contract `factor` field is **always null** for SENSITIVITY (the output magnitude of
change is not specified, only its existence). The input perturbation is named in the
`description` and the `Perturbation_under:` line of the METAMORPHIC block.

#### 9.1 Single-Element Input Perturbation Sensitivity

`input_relation`: flip one bit of one byte of the chosen input array (e.g., `h_input[0] ^= 0x01`)
`output_relation`: Z ≠ Y — output differs from un-perturbed output in at least one
  aggregate stat or sample position, beyond noise tolerance
`applicable_when`: the kernel's output is genuinely data-dependent on the perturbed input

**Example (cipher kernel where output is bit-mixed from the key):**
```cpp
// Flip LSB of first byte of the cipher key — apply AFTER initialization,
// BEFORE the first H→D copy.
if (key_len > 0) h_key[0] ^= 0x01;
```

After the final D→H transfer, emit the METAMORPHIC block at the same `[label]` as the
original PASSTHROUGH block, with `Relation: <relation_name>`,
`Perturbation_under: input_byte_bit_flip`, and the standard `Statistics: ... /
Sample_first10: ...` lines.

**Check (per perturbation):** Compare the variant's METAMORPHIC stats/samples to source's
PASSTHROUGH stats/samples (source's un-perturbed output, captured during Tier 2 baseline).
This single-perturbation check PASSes if AT LEAST ONE statistic OR sample position differs
beyond noise tolerance (`|var − orig| > atol + rtol × max(|var|, |orig|)`), and FAILs if all
stats and samples are within tolerance — the variant did not respond to that perturbation.
The relation's overall verdict then aggregates over the source-active perturbation set as
described above (the variant must respond to every source-active perturbation). The minimal
bit-flip above is the "minimal" perturbation; the framework additionally applies a generic
large-outlier perturbation of the same input dimension.

---

## Domain-Specific Guidance

### Computational Fluid Dynamics — Linear / Stencil Codes
(Laplace equation, isotropic diffusion, heat conduction, Poisson with homogeneous BCs, linear wave equation)

Priority order:
1. **CONSERVATION** (1.1): flux/energy conservation across a stencil sweep
2. **SYMMETRY** (2.2): reflecting initial condition produces reflected output (symmetric BCs required)
3. **ROTATION** (5.1 or 5.2): for 2D isotropic problems with symmetric BCs
4. **SCALING** (3.1): ONLY if the PDE is linear and BCs are homogeneous

Contraindication: convection-dominated flows are nonlinear — SCALING rarely applies. Asymmetric
BCs invalidate SYMMETRY and ROTATION.

### Computational Fluid Dynamics — Coupled Nonlinear Iterative Solvers
(Navier-Stokes, incompressible flow, SOR / Red-Black pressure solvers, projection-method solvers,
driven-cavity flow, multi-field coupling between velocity and pressure fields)

**⚠️ WARNING: Most standard relation families do NOT apply to this class.** Evaluate each
contraindication before proposing any relation:

- **Nonlinear convection term** (u·∇u): any kernel with a nonlinear advection term makes SCALING
  invalid for the velocity fields. Scaling the driving boundary condition does NOT scale the velocity
  by the same factor because the nonlinear term responds quadratically.
- **Non-homogeneous Dirichlet BCs** (fixed lid velocity, fixed wall velocity, fixed temperature):
  SCALING the interior source/forcing term fails because the boundary values are not scaled alongside.
  The boundary-dominated part of the solution does not scale.
- **Zero-valued physical forcing in the default benchmark configuration**: many incompressible flow
  benchmarks set gravity, body forces, or external fields to zero to isolate solver behavior. See
  ❌ SCALING on Zero-Valued Default Parameters — any SCALING or SYMMETRY relation on a zero-valued
  parameter is trivially degenerate and will be rejected by the source filter.
- **Absolute convergence threshold** (`|r| < tol`): even for the linear pressure Poisson sub-problem,
  an absolute stopping criterion means the scaled and unscaled problems may terminate at different
  iteration counts, producing non-proportional final states. See ❌ Absolute convergence thresholds.
- **Permutative on grid indices**: stencil codes encode spatial position in the array index — swapping
  two grid cells changes the geometry and breaks physical semantics.

Priority order for this class:
1. **CONSERVATION** (1.1): Verify a physical invariant at the converged state. For incompressible
   flow, the velocity divergence should sum to approximately zero across the domain. For energy-based
   formulations, check that a conserved quantity is unchanged across a full timestep or solver sweep.
   This is applicable ONLY if the passthrough captures the output fields at sufficient resolution to
   compute the invariant, or if the source already prints a residual/conservation diagnostic.
2. **MONOTONICITY** (6.1): If the solver prints per-iteration convergence residuals to stdout in the
   baseline execution, MONOTONICITY is the most reliably applicable relation for this class. It is
   weak (a code can converge monotonically to a wrong answer), but it is better than an aborted trial.
3. **SCALING** (3.2 joint-scaling): ONLY in the rare case where (a) the flow is effectively linear
   (Stokes limit, Re → 0), (b) ALL driving terms — including all non-homogeneous BCs and the full
   initial condition — are scaled simultaneously by the same α, AND (c) the convergence criterion is
   relative (`|r|/|r₀| < tol`). If any of these three conditions is absent, do not propose SCALING.

**If none of the above is applicable** (no conservation law checkable from passthrough statistics,
no residual history in stdout, nonlinear coupling, non-homogeneous BCs, zero body forces), propose
zero relations. This causes Stage 0 to abort — which is the correct outcome. It is better to abort
than to propose relations that will all fail the source filter, consuming pipeline time with no
verification benefit.

### Linear Algebra (Dense Solve, Matrix Multiply)
Priority order:
1. **SCALING** (3.1): f(alpha×x) = alpha×f(x) — strongest relation, often directly applicable
2. **PERMUTATIVE** (4.1): row interchange should produce identical solution
3. **SYMMETRY** (2.1): negation symmetry if the operation is antisymmetric

### Iterative Linear Solvers (stationary methods, Krylov-subspace methods)
Priority order:
1. **PERMUTATIVE** (4.1): row interchange of coefficient matrix and RHS → identical solution
2. **SCALING** (3.2): scale source term b by alpha → scale solution by alpha (homogeneous BCs only)
3. **MONOTONICITY** (6.1): ONLY if no quantitative relation applies; note CG is NOT monotone

**Stencil-solver caution:** A known weak-spot of Tier 2 passthrough is the pointer-swap bug
pattern, where the double-buffered swap of input/output pointers is dropped or reordered,
silently breaking Dirichlet boundary conditions in the interior of the grid. Boundary elements
remain unchanged so the standard `Sample_first10` capture (indices 0–9, often on a boundary row)
gives the impression that the output matches. Tier 3 PERMUTATIVE relations using **interior**
row indices (e.g., row_i = N/4, row_j = N/2) catch this class of bug — pick interior indices
when proposing row-interchange relations for stencil solvers.

### Particle / N-Body Simulation
Priority order:
1. **CONSERVATION** (1.1): total energy (kinetic + potential) conservation
2. **PERMUTATIVE** (4.2): particle index permutation → permuted force/energy arrays
3. **SCALING** (3.1): if interactions are linear in mass/charge, scaling inputs scales forces

### Reduction Kernels (Sum, Dot Product, Norm)
Priority order:
1. **SCALING** (3.1): scaling input by alpha scales scalar output by alpha (or alpha² for squared norms)
2. **CONSERVATION** (1.3): partial sum decomposition (no extra run if passthrough covers full array)
3. **PERMUTATIVE** (4.3): shuffling input produces same scalar result (generous FP tolerance)

### Categorical / Equality-Based Kernels (discrete-attribute distance, hash, comparison, set membership)
Priority order:
1. **INVARIANCE** (Category 8): output depends only on equality structure, not on values themselves.
   At least one of: attribute-order invariance (8.1), value-relabeling invariance (8.2),
   uniform-value-shift invariance (8.3), or equality-class-preserving negation invariance (8.4).
   Select transformations that exercise different parts of the kernel (e.g., one that probes
   the attribute-stride memory layout, one that probes the equality predicate).
2. **PERMUTATIVE** (4.1 or 4.2): if rows / instances are interchangeable (the kernel does not
   treat position as semantically meaningful), instance-row swap should leave aggregate output
   statistics invariant.
3. **SCALING / ADDITIVITY / SYMMETRY**: NOT APPLICABLE — the output depends on equality classes,
   not on magnitudes, sums, or signs. Do not propose these for this kernel class.

**Caution for kernels that mix categorical inputs with numeric reductions** (e.g., a categorical
classifier whose output is a weighted sum where the weights are numeric): INVARIANCE on the
categorical inputs is valid; SCALING/ADDITIVITY may also apply to the numeric weights as a
separate relation. Treat as two independent input dimensions, one relation per dimension.

### Dynamic Programming / Shortest Path / Wavefront (e.g., grid-based shortest path, BFS distance, edit distance, wavefront accumulation)

**⚠️ SCALING is NEVER applicable to this class.** The accumulated result (minimum path cost,
shortest distance, edit distance) is computed via `min()` or `max()` at every step, making the
output non-linear in the input. Scaling the input grid by 2.0 does NOT multiply the output by 2.0
when `min()` is applied — it would only hold trivially if no computation occurred (identity function).
If SCALING holds perfectly (Z_stat ≈ 2.0 × Y_stat) for this class, it is strong evidence that
the passthrough is capturing pre-computation data (the initial grid values) rather than the
computed result. **Treat a perfectly-satisfied SCALING relation for a DP/path algorithm as a
red flag indicating a passthrough placement error, not a correct verification result.**

Priority order:
1. **PERMUTATIVE** (4.1): permuting input rows/columns (and corresponding grid values) should produce
   the permuted output. For a 2D wavefront DP over a ROWS×COLS grid, swapping two columns of the
   input grid and verifying the same two columns of the output are swapped is an equivariance relation.
2. **MONOTONICITY** (6.1): for shortest-path/DP with non-negative costs, the result accumulated over
   more iterations is non-decreasing (each additional row adds at least 0 to the minimum path cost).
   Use MONOTONICITY as a fallback only — it is the weakest available relation.
3. **CONSERVATION**: rarely applicable to path algorithms unless the algorithm has an explicit
   global invariant (e.g., total cost is bounded). Skip unless you can identify a concrete law.

**Do NOT propose SCALING or SYMMETRY** for DP/path algorithms — they do not hold for the
final computed output, and if they appear to hold, the passthrough is measuring the wrong state.

---

## Example: Linear Iterative Stencil Solver Metamorphic Contract

**Context:** A stationary iterative solver for `A x = b`. Matrix A is diagonally dominant
with Dirichlet boundary conditions fixed in boundary rows; interior rows are interchangeable.
The source term `b` drives the solution.

**Analysis:**
- Computation type: linear iterative solver
- Linear? Yes (the kernel solves a linear system)
- Conservation law? No — this solver class does not conserve a physical quantity
- SCALING (source term)? Marginal — boundary-condition rows have fixed values that are NOT
  scaled with b, so the boundary part of the solution does not scale; SOURCE_verify will
  likely filter this inactive; include speculatively.
- PERMUTATIVE (row swap)? Yes — swapping two interior rows i and j (and corresponding b
  entries) should produce the identical solution vector.
- SYMMETRY (negation)? Yes — if BCs and b are all negated, the solution should negate:
  f(−b) = −f(b).

**Generated contract:**
```json
{
  "relations": [
    {
      "name": "row_swap",
      "type": "PERMUTATIVE",
      "description": "Swapping two interior rows of coefficient matrix and RHS produces identical solution",
      "factor": null,
      "input_variables": ["h_A", "h_b"],
      "observation_points": ["final_solution"],
      "expected_relationship": "output = original_output",
      "tolerance": 1e-5,
      "cost": "cheap"
    },
    {
      "name": "negation",
      "type": "SYMMETRY",
      "description": "Negating source term b (and BCs) should negate the solution for a linear system",
      "factor": -1.0,
      "input_variables": ["h_b"],
      "observation_points": ["final_solution"],
      "expected_relationship": "output = -1.0 * original_output",
      "tolerance": 1e-5,
      "cost": "cheap"
    }
  ]
}
```

**Note on SCALING for fixed-BC iterative solvers:** Scaling only h_b while leaving boundary
rows of A unchanged changes the mathematical problem — the boundary values constrain the
solution independently. The SOURCE_verify filter will catch this and mark SCALING inactive.
The contract can include it speculatively; the filter will exclude it if inapplicable.

---

## Implementation Notes

1. **METAMORPHIC block label must match PASSTHROUGH block label** (for SCALING, PERMUTATIVE,
   SYMMETRY, ROTATION). This allows `check_metamorphic_relation()` to pair original output Y with
   variant output Z at the same observation point. Conservation and Monotonicity use their own
   paired before/after or per-iteration labels.

2. **NaN/Infinity guard is mandatory.** Before comparing any statistic: `if (!isfinite(value))
   → SKIP this block, do not FAIL.`

3. **allclose-style tolerance** (same as Tier 2):
   ```
   |Z_stat - expected_Z_stat| ≤ atol + rtol × max(|Z_stat|, |expected_Z_stat|)
   atol = |expected_Z_stat| × rtol;  default rtol = 1e-5
   ```

4. **FP non-associativity for PERMUTATIVE reductions:** GPU parallel reductions produce slightly
   different results when element order changes. Use tolerance 1e-3 to 1e-4 for large N, or use
   a fixed deterministic permutation (array reverse) for reproducibility.

5. **SOURCE_verify filter** marks relations active or inactive before any generated code is tested.
   A relation that fails SOURCE_verify is marked inactive and silently skipped — it does not indicate
   a source code bug. It means the relation is inapplicable to this computation.

6. **One variant per relation.** Do not combine two transformations in a single variant file.

7. **MT false positive caveat:** MT cannot guarantee absence of bugs — both source and follow-up
   executions may be wrong in the same way. Multiple diverse MRs reduce but do not eliminate this
   risk. Tier 3 PASS is evidence of correctness, not a proof.

---
