
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
positives from inapplicable MRs. This mechanizes the "manual MR validation" step identified as
future work by Li et al. 2025 (survey of 81 MR generation papers, Directions 2–3).

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

### 7. 3–6 Diverse MRs: The Gold Standard

Three to six diverse MRs reveal approximately 90% of faults detectable by a full oracle
(Liu et al., confirmed by Segura 2016, Chen et al. 2019, and independent replication studies).
Target **2–4 relations** per generated code.

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

When generating a contract, work through these analysis questions. Stop at 2–4 relations — do not
add more simply because they appear in the catalog.

### Analysis Question 1: What is the core computation type?

- **Linear algebra** (matrix multiply, dense solve): SCALING → PERMUTATIVE
- **Iterative solver** (Jacobi, CG, GMRES): SCALING (on source term if BCs allow) → PERMUTATIVE
- **Stencil / CFD**: CONSERVATION (flux balance) → SYMMETRY → ROTATION (if isotropic)
- **Reduction kernel** (sum, dot product, norm): SCALING → CONSERVATION (decomposition) → PERMUTATIVE
- **Particle / N-body**: CONSERVATION (energy) → PERMUTATIVE (particle index) → SCALING
- **Normalized distance / correlation kernel** (z-score normalized DTW, LB_Keogh, cross-correlation, standardized regression): SCALING on the **co-normalized group** (raw data + precomputed mean + precomputed std, all by the same α) → output UNCHANGED (factor = 1.0); PERMUTATIVE (inner sum index permutation) → output UNCHANGED (factor = 1.0) — SCALING on any **single** input variable IS NOT APPLICABLE (see normalization warning below)
- **Dynamic programming / path algorithms** (shortest path, DP table fill, wavefront, graph BFS/DFS): PERMUTATIVE (row/column index permutation) → MONOTONICITY (result non-decreasing with iterations) — **SCALING IS NOT APPLICABLE** (see below)
- **Sorting / ranking / selection**: PERMUTATIVE (input permutation → permuted output) — SCALING IS NOT APPLICABLE (sort of 2×input is NOT 2× sort of input in index rank)
- **Thresholding / clamping / activation**: SCALING IS NOT APPLICABLE for any input scale that crosses a threshold; CONSERVATION may apply if the operation conserves sum under special conditions

### Analysis Question 2: Is the algorithm linear or nonlinear?

- **Linear**: SCALING is directly applicable: `f(α×x) = α×f(x)`
- **Nonlinear or thresholded**: SCALING does not apply to the full output; check whether it applies
  to a residual metric or intermediate buffer
- **Iterative with fixed boundary conditions**: SCALING applies to the source term ONLY if BCs are
  also scaled or are homogeneous (zero)

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
- Do NOT propose SCALING on `buffer` alone, `stats_mean` alone, or `stats_std` alone.
- The valid SCALING is: scale **all three** (`buffer`, `stats_mean`, `stats_std`) by the same
  positive factor α → the normalized value is unchanged → output is unchanged (factor = 1.0).
- This `factor = 1.0` MR is a strong correctness test: any code that drops the mean subtraction
  or std division will produce wrong output under this transformation.

**If only division is present (mean = 0):** scaling `x` and `std` together by α gives `α·x / α·std = x / std` → output unchanged (factor = 1.0). Scale both.

**Clamped-squared deviation pattern** (common in distance kernels): `val² × (val > 0)` or
`max(0, val)²` where `val = (x - mean)/std - envelope`. Scaling the co-normalized group
(x + mean + std together) leaves `val` unchanged → output unchanged. Scaling only `x` changes
`val` in a position-dependent, nonlinear way because which values are clamped to zero also changes.

### Analysis Question 3: Are there physical conservation laws?

- **Explicit conservation** (mass, energy, momentum): CONSERVATION (1.1) applies directly
- **Reduction** (sum, integral): CONSERVATION (1.3) decomposition applies
- **No conservation law**: skip Category 1

### Analysis Question 4: What symmetries exist in the problem?

- **Antisymmetric function** (f(−x) = −f(x)): SYMMETRY (2.1) negation applies
- **Symmetric domain** (symmetric BCs, isotropic physics): SYMMETRY (2.2) reflection applies
- **Spatially isotropic problem** (no preferred direction): ROTATION (5.1 or 5.2) may apply
- **No symmetry**: skip Categories 2 and 5

### Fail-Fast Ordering Rule

Order selected relations **cheapest-first and most-discriminating-first** within the same cost tier:

1. CONSERVATION — cheap (1 extra compile + run; no input transformation needed)
2. SYMMETRY — cheap (1 extra compile + run; negation of input, no structural change)
3. SCALING — cheap (1 extra compile + run; **highest discriminating power**)
4. PERMUTATIVE — cheap (1 extra compile + run)
5. ROTATION — cheap (1 extra compile + run)
6. MONOTONICITY — cheap (1 extra compile + run; **weakest, last resort only**)

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

Default: `rtol = 1e-5` (slightly looser than Tier 2's 1e-6, to account for accumulated FP noise
in two independently-computed runs). Adjust per relation if needed (see PERMUTATIVE note).

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

### Step 2: Select 2–4 Relations from the Catalog

Use the Selection Framework above. Apply BOTH approaches:

**Input-driven** (for SCALING, PERMUTATIVE, SYMMETRY, ROTATION): Ask "If I change this input
parameter in this way, what exactly should change in the output?"

**Output-driven** (for CONSERVATION): Ask "What aggregation of the output should be preserved?
What input structure allows me to decompose it?"

**MANDATORY — Symbolic trace before proposing any SCALING relation:**

Before writing a SCALING relation into the contract, trace the specific input variable(s) you
plan to scale through every kernel operation symbolically and confirm the expected output ratio.

Example — incorrect reasoning (keogh failure):
- "I see squared terms → scaling subject by α should scale output by α²"  ← WRONG
- Trace: `value = (α·subject - avg) / std` → value is NOT `α·(subject-avg)/std` → ratio is
  input-position-dependent → SCALING on `subject` alone is INVALID.

Example — correct reasoning (keogh):
- Scale `subject`, `avgs`, `stds` all by α: `value = (α·s - α·avg)/(α·std) = (s-avg)/std`
  → value unchanged → output unchanged → factor = 1.0 ✓

If you cannot write a symbolic trace that confirms the exact output ratio for at least one
representative input value, do NOT propose the relation. A relation that cannot be verified
symbolically will fail the source filter and waste a contract slot.

**IMPORTANT — null-factor relations are not supported in Phase 1/2:** Only propose relations where
`factor` is a concrete non-null number (e.g., `2.0`, `-1.0`, `1.0`). Any relation requiring
element-wise array comparison — including reflection symmetry, general permutations, and
SYMMETRY/SCALING entries where you cannot state a single scalar `factor` — must be omitted.
These are Phase 3 future work. See the ❌ Null-Factor Anti-Pattern below. Proposing a null-factor
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
    }
  ]
}
```

**Field descriptions:**
- `name`: short identifier used in variant filename (`{file_name}_mt_{name}.cu`). Use lowercase with
  underscores, no spaces.
- `type`: one of `CONSERVATION`, `SYMMETRY`, `SCALING`, `PERMUTATIVE`, `ROTATION`, `MONOTONICITY`
- `description`: one sentence explaining the mathematical or physical basis of the relation
- `factor`: scaling factor (for SCALING/SYMMETRY), or `null`
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

### ❌ MONOTONICITY When Equality Is Available

```
BAD priority: use MONOTONICITY for a linear PDE when SCALING clearly applies
```

MONOTONICITY is the last resort. Always prefer equality/quantitative-ratio relations.

### ❌ Null-Factor Relations (Phase 3 Only — Not Yet Supported)

```
BAD (Phase 1/2): { "type": "SYMMETRY", "factor": null, "name": "horizontal_reflection" }
BAD (Phase 1/2): { "type": "SCALING",  "factor": null, "name": "negation_symmetry" }
```

**What a null-factor relation is:** A relation with `"factor": null` requires element-wise array
comparison between the original output Y and the transformed variant output Z. This is a Phase 3
capability that is not yet implemented in the LASSI-Verify pipeline.

**Why to avoid it:** Relations with `factor: null` are **automatically SKIP'd** during the Stage 0
source filter — the pipeline detects the null factor and marks them inactive before even attempting
to compile or run the variant. Every null-factor relation you propose wastes a contract slot and
produces zero verification value for any trial.

**What to do instead:** For SYMMETRY relations, only propose relations where `factor` is a known
non-null scalar (e.g., `factor: -1.0` for antisymmetric negation, `factor: 1.0` for element-wise
equivalence under permutation with a paired PERMUTATIVE check). If the symmetry you have in mind
requires inspecting individual array elements rather than comparing aggregate statistics (min/max/mean)
scaled by a constant, that is a null-factor relation — skip it and choose a SCALING or CONSERVATION
relation instead.

**Formal note:** The current checker compares `Z_stat ≈ factor × Y_stat` for each scalar statistic
(min, max, mean). This requires an explicit numeric factor. Element-wise comparison (needed for
reflection symmetry, general permutations, etc.) is deferred to Phase 3.

### ❌ Composite MRs in Phase 1

```
BAD (Phase 1): apply SCALING then PERMUTATIVE in the same variant
```

Composite MRs (more effective per Qiu et al. 2019) are tagged as Phase 3 future work.

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

**Note:** This is the most common applicable SCALING for iterative solvers (Jacobi, CG, etc.).
If Dirichlet BCs in the rows of A are fixed at non-zero values and NOT also scaled, this relation
does not apply — SOURCE_verify will fail it and filter it inactive.

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

**Note for Jacobi:** Choose interior rows only (avoid boundary rows that encode Dirichlet BCs).
Swapping a boundary row with an interior row changes the problem — SOURCE_verify will catch this.

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
- Simple Jacobi iteration with strict convergence

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

```cpp
for (int iter = 0; iter < max_iter; iter++) {
    kernel<<<blocks, threads>>>(...);
    cudaDeviceSynchronize();
    cudaMemcpy(h_x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    double residual = compute_residual(h_A, h_b, h_x, N);
    printf("=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\n", iter);
    printf("Relation: monotonicity\nIteration: %d\nResidual: %.17g\n",
           iter, residual);
    printf("=== END LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\n", iter);
}
```

**Check:** `residual[k] >= residual[k+1] - tolerance * residual[k]` for all k (allow tiny
numerical noise in "near-equal" consecutive values).

---

## Domain-Specific Guidance

### Computational Fluid Dynamics / Stencil Codes
Priority order:
1. **CONSERVATION** (1.1): flux/energy conservation across a stencil sweep
2. **SYMMETRY** (2.2): reflecting initial condition produces reflected output (symmetric BCs required)
3. **ROTATION** (5.1 or 5.2): for 2D isotropic problems with symmetric BCs
4. **SCALING** (3.1): ONLY if the PDE is linear and BCs are homogeneous

Contraindication: convection-dominated flows are nonlinear — SCALING rarely applies. Asymmetric
BCs invalidate SYMMETRY and ROTATION.

### Linear Algebra (Dense Solve, Matrix Multiply)
Priority order:
1. **SCALING** (3.1): f(alpha×x) = alpha×f(x) — strongest relation, often directly applicable
2. **PERMUTATIVE** (4.1): row interchange should produce identical solution
3. **SYMMETRY** (2.1): negation symmetry if the operation is antisymmetric

### Iterative Solvers (Jacobi, Gauss-Seidel)
Priority order:
1. **PERMUTATIVE** (4.1): row interchange of coefficient matrix and RHS → identical solution
2. **SCALING** (3.2): scale source term b by alpha → scale solution by alpha (homogeneous BCs only)
3. **MONOTONICITY** (6.1): ONLY if no quantitative relation applies; note CG is NOT monotone

**Jacobi-specific caution from MEMORY.md:** The pointer-swap bug (jacobi March 11 run) broke
Dirichlet BCs in the interior but was missed by Tier 2 passthrough because boundary elements
(indices 0–9) were unchanged. Row interchange using interior rows (row_i = N/4, row_j = N/2)
would have been more discriminating than boundary-adjacent checks.

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

### Dynamic Programming / Shortest Path / Wavefront (e.g., pathfinder, BFS distance, edit distance)

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
   the permuted output. For pathfinder (ROWS×COLS grid), swapping two columns of the input grid and
   the same two columns of the output is an equivariance relation.
2. **MONOTONICITY** (6.1): for shortest-path/DP with non-negative costs, the result accumulated over
   more iterations is non-decreasing (each additional row adds at least 0 to the minimum path cost).
   Use MONOTONICITY as a fallback only — it is the weakest available relation.
3. **CONSERVATION**: rarely applicable to path algorithms unless the algorithm has an explicit
   global invariant (e.g., total cost is bounded). Skip unless you can identify a concrete law.

**Do NOT propose SCALING or SYMMETRY** for DP/path algorithms — they do not hold for the
final computed output, and if they appear to hold, the passthrough is measuring the wrong state.

---

## Example: Jacobi Solver Metamorphic Contract

**Context:** Jacobi iteration for Ax = b. Matrix A is diagonally dominant with Dirichlet BCs
fixed in boundary rows. Interior rows are interchangeable. Source term b drives the solution.

**Analysis:**
- Computation type: linear iterative solver
- Linear? Yes (Jacobi solves a linear system)
- Conservation law? No — Jacobi does not conserve a physical quantity
- SCALING (source term)? Marginal — boundary condition rows have fixed ±1 values not scaled with b.
  SOURCE_verify will likely filter this inactive; include conditionally.
- PERMUTATIVE (row swap)? Yes — swapping two interior rows i and j (and corresponding b entries)
  should produce the identical solution vector.
- SYMMETRY (negation)? Yes — if BCs and b are all negated, solution should negate: f(−b) = −f(b).

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

**Note on SCALING for Jacobi:** Scaling only h_b while leaving boundary rows of A unchanged
changes the mathematical problem — the boundary values constrain the solution independently.
The SOURCE_verify filter will catch this and mark SCALING inactive. The contract can include it
speculatively; the filter will exclude it if inapplicable.

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

7. **Composite MRs (Phase 3 future work):** Chaining two relations (e.g., SCALING then PERMUTATIVE)
   is more effective (Qiu et al. 2019: composite MRs catch faults neither component catches alone).
   Not in Phase 1.

8. **MT false positive caveat:** MT cannot guarantee absence of bugs — both source and follow-up
   executions may be wrong in the same way (Segura 2020; documented in the jacobi false-positive
   case in MEMORY.md). Multiple diverse MRs reduce but do not eliminate this risk. Tier 3 PASS
   is evidence of correctness, not a proof.

9. **Concurrent execution (Phase 3 future):** CUDA kernels are concurrent programs. Thread-
   interleaving MRs require an extended MR definition (Sun et al. 2023 via Li et al. 2025).
   Out of scope for Phase 1.

10. **Scientific simulation domain precedents:** Luu et al. (ocean software MT), Raunak & Olsen
    (simulation model MT), Lidbury et al. (OpenCL compiler fuzzing — 50+ bugs in OpenCL compilers
    on NVIDIA hardware via MT). LASSI-Verify Tier 3 is consistent with established GPU-MT practice.

---
