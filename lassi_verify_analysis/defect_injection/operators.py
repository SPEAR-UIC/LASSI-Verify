"""
The 5 generic-rule program-defect operators (catalog entries 1, 2, 4, 5, 6 -- FP-REASSOCIATION
and REDUCTION-ORDER-CHANGE follow in this same module once their source patterns are confirmed).

Every patch function here does a strict match-then-replace against a specific, pre-confirmed
excerpt of a kernel's canonical-base source (confirmed by direct read this session -- see
planning/LASSI_Verify_Phase2_Bridge.md's Step 4 injection-implementation dated entry for the
full evidence). Each function raises PatternNotFoundError rather than silently no-op'ing or
guessing at an alternate location if the expected text has moved -- "refuse to guess on
mismatch" applies to defect injection exactly as it does everywhere else in this project.

Several patterns confirmed this session DIVERGE from the original applicability matrix's
assumptions (built against pristine HeCBench source or an earlier code_version, not always the
actual campaign-winning canonical base used here) -- each function's docstring notes the
correction where one was needed.
"""
from __future__ import annotations

import os
import re


class PatternNotFoundError(RuntimeError):
    pass


class NotApplicableError(RuntimeError):
    """Raised when re-verification against the real canonical base shows the operator's
    pattern genuinely isn't present -- a corrected applicability finding, not a bug."""


def _require(pattern: str, code: str, kernel: str, operator: str, flags=0) -> re.Match:
    m = re.search(pattern, code, flags)
    if not m:
        raise PatternNotFoundError(
            f"{operator} on {kernel}: expected pattern not found -- "
            f"re-verify against the current canonical base before trusting this operator "
            f"(pattern={pattern!r})"
        )
    return m


def _require_literal(needle: str, code: str, kernel: str, operator: str) -> None:
    if needle not in code:
        raise PatternNotFoundError(
            f"{operator} on {kernel}: expected literal text not found:\n{needle}"
        )


# ---------------------------------------------------------------------------
# 1. SYNC-REMOVAL
# ---------------------------------------------------------------------------
# Applicable (confirmed 2026-08-23 against the actual canonical-base source): bsearch,
# lid-driven-cavity, pathfinder, threadfence (4/17, NOT 5 -- see correction below).
#
# CORRECTION to the catalog: all-pairs-distance's canonical base (v4_Trial1) has NO literal
# __syncthreads()/__threadfence() anywhere -- its k2/k3 kernels use cub::BlockReduce, whose
# internal sync calls are inside the library template, not textual in this source file. The
# applicability matrix's "all-pairs-distance: YES" cell does not hold for this specific
# campaign-winning variant. Dropped to 4 applicable cells.

def sync_removal_bsearch(code: str) -> tuple[str, str]:
    """bsearch's kernel_BS4: single-writer broadcast hazard. Write `k = 1ULL << (nbits-1);`
    (thread lid==0), __syncthreads(), broadcast read `size_t p = k;` by all threads."""
    kernel = "bsearch"
    m = _require(r"(\n\s*__syncthreads\(\);\n)(\s*size_t p = k;)", code, kernel, "SYNC-REMOVAL")
    patched = code[: m.start(1)] + "\n" + code[m.end(1):]
    diff = f"removed: {m.group(1).strip()!r} (was immediately before {m.group(2).strip()!r})"
    return patched, diff


def sync_removal_lid_driven_cavity(code: str) -> tuple[str, str]:
    """lid-driven-cavity's sum_pressure: warp-sums write, __syncthreads(), warp-0 cross-warp
    read of warp_sums[]. (3 structurally identical sibling reductions exist in this file;
    this patches the first, sum_pressure.)"""
    kernel = "lid-driven-cavity"
    m = _require(
        r"(if \(lane == 0\) warp_sums\[warp\] = val;\n\s*)(__syncthreads\(\);)",
        code, kernel, "SYNC-REMOVAL",
    )
    patched = code[: m.start(2)] + code[m.end(2):]
    return patched, f"removed: {m.group(2)!r} after warp_sums[warp] write in sum_pressure"


def sync_removal_pathfinder(code: str) -> tuple[str, str]:
    """pathfinder's `pathfinder` kernel: write prev[tx]=gpuSrc[xidx], __syncthreads(), then
    genuinely cross-thread reads prev[W]/prev[tx]/prev[E]. Strongest write-then-read hazard
    of the batch (real neighbor access, not just single-writer broadcast)."""
    kernel = "pathfinder"
    m = _require(
        r"(prev\[tx\] = gpuSrc\[xidx\];\n\s*\}\n\s*)(__syncthreads\(\);)",
        code, kernel, "SYNC-REMOVAL",
    )
    patched = code[: m.start(2)] + code[m.end(2):]
    return patched, f"removed: {m.group(2)!r} after prev[tx]=gpuSrc[xidx] write"


def sync_removal_threadfence(code: str) -> tuple[str, str]:
    """threadfence's `sum` kernel: no literal __threadfence() in this LLM-refactored variant
    (it replaced the original in-place multi-pass reduction with a ping-pong buffer scheme,
    per an in-file comment) -- CORRECTION: falls back to the kernel's own __syncthreads()
    gating warpSums[warpId] write -> warp-0 cross-warp read, per the catalog's "or
    __threadfence()" wording (either qualifies as this operator's target)."""
    kernel = "threadfence"
    m = _require(
        r"(warpSums\[warpId\] = threadSum;\n\s*\}\n\s*)(__syncthreads\(\);)",
        code, kernel, "SYNC-REMOVAL",
    )
    patched = code[: m.start(2)] + code[m.end(2):]
    return patched, f"removed: {m.group(2)!r} after warpSums[warpId] write (no literal __threadfence() present in this variant; used its __syncthreads() instead, per catalog's either/or wording)"


SYNC_REMOVAL = {
    "bsearch": sync_removal_bsearch,
    "lid-driven-cavity": sync_removal_lid_driven_cavity,
    "pathfinder": sync_removal_pathfinder,
    "threadfence": sync_removal_threadfence,
}
SYNC_REMOVAL_NOT_APPLICABLE = {
    "all-pairs-distance": (
        "no literal __syncthreads()/__threadfence() in the canonical base (v4_Trial1) -- "
        "its cub::BlockReduce usage hides synchronization inside the library template. "
        "Confirmed by direct source read 2026-08-23; corrects the original applicability matrix."
    ),
}


# ---------------------------------------------------------------------------
# 2. ATOMICITY-WEAKENING
# ---------------------------------------------------------------------------
# Applicable (confirmed): all-pairs-distance, jaccard, randomaccess (3/17, NOT 4 -- see
# correction below).
#
# CORRECTION to the catalog: threadfence's canonical base (v5_Trial1) has ZERO atomic
# operations anywhere in the file (confirmed by exhaustive grep -- no `atomicAdd`/`atomicXor`
# at all; the only kernels present are the witness-chain sentinel functions and `sum`, a
# warp-shuffle + ping-pong-buffer reduction with no atomics). The catalog's assumed
# "countMismatches" kernel with `atomicAdd(&blockSum, local)` does not exist in this variant
# -- likely conflated with a different threadfence variant or the pristine HeCBench source.
# Dropped to 3 applicable cells; confirmed 2026-08-23 (caught by this operator's own
# refuse-to-guess pattern check raising PatternNotFoundError rather than silently no-op'ing).

def atomicity_weakening_all_pairs_distance(code: str) -> tuple[str, str]:
    """k1: atomicAdd(distance + INSTANCES*gx + gy, count) -- every thread in block (gx,gy)
    hits the same slot across loop iterations."""
    kernel = "all-pairs-distance"
    needle = "atomicAdd(distance + INSTANCES*gx + gy, count);"
    _require_literal(needle, code, kernel, "ATOMICITY-WEAKENING")
    replacement = "distance[INSTANCES*gx + gy] += count;"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r}"


def atomicity_weakening_jaccard(code: str) -> tuple[str, str]:
    """jaccard_is: atomicAdd(&weight_i[j], ref_val) -- multiple i-loop iterations (and
    thread combinations, for symmetric graphs) can map to the same j."""
    kernel = "jaccard"
    needle = "atomicAdd(&weight_i[j],ref_val);"
    if needle not in code:
        # tolerate the LLM's own whitespace variance around the same call
        m = _require(r"atomicAdd\(&weight_i\[j\],\s*ref_val\);", code, kernel, "ATOMICITY-WEAKENING")
        needle = m.group(0)
    replacement = "weight_i[j] += ref_val;"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r}"


def atomicity_weakening_randomaccess(code: str) -> tuple[str, str]:
    """update kernel: atomicXor((unsigned long long*)&Table[state & mask], ...) -- an LFSR
    state stream deliberately designed to collide across threads on Table[] slots."""
    kernel = "randomaccess"
    needle = "atomicXor((unsigned long long*)&Table[state & mask], (unsigned long long)state);"
    _require_literal(needle, code, kernel, "ATOMICITY-WEAKENING")
    replacement = "Table[state & mask] ^= state;"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r}"


ATOMICITY_WEAKENING = {
    "all-pairs-distance": atomicity_weakening_all_pairs_distance,
    "jaccard": atomicity_weakening_jaccard,
    "randomaccess": atomicity_weakening_randomaccess,
}
ATOMICITY_WEAKENING_NOT_APPLICABLE = {
    "threadfence": (
        "canonical base (v5_Trial1) has zero atomic operations anywhere in the file -- "
        "confirmed by exhaustive grep 2026-08-23. The catalog's assumed countMismatches "
        "kernel does not exist in this variant."
    ),
}


# ---------------------------------------------------------------------------
# 6. BOUNDARY-GUARD-CORRUPTION
# ---------------------------------------------------------------------------
# Applicable (confirmed): colorwheel, entropy, extrema (2D kernel), murmurhash3, pathfinder,
# randomaccess, threadfence, matrix-rotate (8/17, as the catalog states -- but THREE of the
# 8 need a corrected mechanism, not the literal `if (idx < N)` shape the catalog assumed):
#   - murmurhash3, randomaccess, threadfence: bound is a grid-stride FOR-LOOP condition
#     (`for (...; i < N; ...)`), not a standalone `if` -- widen the loop condition instead.
#   - matrix-rotate: the catalog's assumed `if (layer < n/2)` pattern is from the PRISTINE
#     HeCBench source, not present in the actual campaign-winning variant, which uses an
#     out-of-place 2D kernel with `if (row < n && col < n)` guards instead. Patches that
#     shape for this canonical base.
#   - extrema: the 2D kernel's guard lives in an #include'd header
#     (source_codes/HeCBench/extrema/kernels.h), not the Trial .cu file itself. The Trial
#     file #includes it by ABSOLUTE path -- patch a scratch copy of the header and rewrite
#     the #include line to point at the scratch copy, so the shared header (used by every
#     other extrema experiment) is never mutated in place.
#   - randomaccess: the `update`/`validationReplay` kernels' `if (begin >= end) return;`
#     guards are NOT used here -- their actual array index (`state & mask`) is pre-masked
#     independent of begin/end, so widening that guard would be a harmless no-op, not a
#     genuine OOB. `initTable`'s grid-stride loop (`i < TableSize`, unmasked `Table[i]`
#     write) is the real target.

def boundary_guard_colorwheel(code: str) -> tuple[str, str]:
    kernel = "colorwheel"
    needle = "if (y < size && x < size) {"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "if (y <= size && x <= size) {"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r}"


def boundary_guard_entropy(code: str) -> tuple[str, str]:
    """Baseline `entropy` kernel (not the _opt variants): early-return form."""
    kernel = "entropy"
    needle = "if (x >= width || y >= height) return;"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "if (x > width || y > height) return;"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r}"


def boundary_guard_murmurhash3(code: str) -> tuple[str, str]:
    """No standalone if-guard -- MurmurHash3_x64_128_kernel's grid-stride loop condition
    `i < numKeys` is the functional bound. Widen the loop condition."""
    kernel = "murmurhash3"
    needle = "i < numKeys;"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "i <= numKeys;"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r} (grid-stride loop bound, no standalone if-guard present)"


def boundary_guard_pathfinder(code: str) -> tuple[str, str]:
    """No entry idx<N guard -- nearest data-touching guard is IN_RANGE(xidx, 0, cols-1)
    protecting the gpuSrc[xidx] read. Widen by admitting xidx==cols."""
    kernel = "pathfinder"
    needle = "if(IN_RANGE(xidx, 0, cols-1))"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "if(IN_RANGE(xidx, 0, cols))"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r} (widens the read-bound macro by one; no literal if(idx<N){{}} form exists in this kernel)"


def boundary_guard_randomaccess(code: str) -> tuple[str, str]:
    """initTable's grid-stride loop `i < TableSize` bounds the unmasked write Table[i] --
    the real OOB target (update/validationReplay's begin>=end guards would be a no-op since
    their actual index is pre-masked by `state & mask`, independent of begin/end)."""
    kernel = "randomaccess"
    needle = "for (; i < TableSize; i += stride) {\n    Table[i] = i;"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "for (; i <= TableSize; i += stride) {\n    Table[i] = i;"
    return code.replace(needle, replacement, 1), f"widened initTable's loop bound i<TableSize -> i<=TableSize (update/validationReplay's guards excluded -- pre-masked index, widening there is a no-op)"


def boundary_guard_threadfence(code: str) -> tuple[str, str]:
    """sum kernel's grid-stride loop `idx < (unsigned int)N` bounds the unmasked read
    array[idx]. Widen the loop condition."""
    kernel = "threadfence"
    needle = "idx < (unsigned int)N;"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "idx <= (unsigned int)N;"
    return code.replace(needle, replacement, 1), f"{needle!r} -> {replacement!r} (grid-stride loop bound, no standalone if-guard present)"


def boundary_guard_matrix_rotate(code: str) -> tuple[str, str]:
    """CORRECTION: the catalog's assumed `if (layer < n/2)` is from pristine HeCBench source,
    not this campaign-winning variant, which is an out-of-place 2D rotation with
    `if (row < n && col < n)` guards in BOTH rotate_matrix_parallel and
    rotate_matrix_parallel_outofplace. Patches whichever is actually present (both share the
    identical guard text; replace_all is safe since both, if both present, are equally
    genuine OOB targets -- only the one actually launched in this Trial matters for
    detection, but patching both keeps the file internally consistent)."""
    kernel = "matrix-rotate"
    needle = "if (row < n && col < n) {"
    _require_literal(needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "if (row <= n && col <= n) {"
    n = code.count(needle)
    patched = code.replace(needle, replacement)
    return patched, (
        f"{needle!r} -> {replacement!r} ({n} occurrence(s); catalog's assumed `layer<n/2` "
        f"pattern is from pristine HeCBench source, not present in this variant -- corrected "
        f"2026-08-23)"
    )


# ---------------------------------------------------------------------------
# 4. FP-REASSOCIATION
# ---------------------------------------------------------------------------
# Applicable (confirmed): entropy, threadfence, atomicCost (3/17, NOT 4 -- see correction).
#
# CORRECTION to the catalog: jaccard's only genuine `T sum = 0; sum += expr;`-shaped
# accumulation (jaccard_is_opt's local_sum, a two-pointer sorted-merge CSR intersection walk)
# has a data-DEPENDENT trip count -- only iterations where ref_col==cur_col execute the +=,
# so a naive midpoint index split does not evenly (or even validly) bisect the actual
# accumulation, and forcing one risks a hand-authored, kernel-specific patch rather than a
# clean generic-rule transform. Its other candidate (parallel_prefix_sum) is a cooperative
# warp-shuffle Hillis-Steele scan -- reordering it changes cross-lane dependencies, not just
# accumulation grouping, which is REDUCTION-ORDER-CHANGE's concern, not this operator's.
# Dropped to 3 applicable cells; confirmed 2026-08-23.

def fp_reassociation_entropy(code: str) -> tuple[str, str]:
    """Baseline `entropy` kernel (not the _opt variants): 16-iteration compile-time-bounded
    Shannon-entropy accumulation `ent -= p*log2f(p)`. Split into two 8-iteration partial
    sums combined at the end (reassociation preserves -= as += of a negated term)."""
    kernel = "entropy"
    needle = (
        "  float ent = 0.0f;\n"
        "  if (total >= 1) {\n"
        "    #pragma unroll\n"
        "    for(int k = 0; k < 16; k++) {\n"
        "      if (count[k] > 0) {\n"
        "        float p = __fdividef((float)count[k], (float)total);\n"
        "        ent -= p * log2f(p);\n"
        "      }\n"
        "    }\n"
        "  }\n"
    )
    _require_literal(needle, code, kernel, "FP-REASSOCIATION")
    replacement = (
        "  float ent = 0.0f;\n"
        "  if (total >= 1) {\n"
        "    float ent_lo = 0.0f;\n"
        "    #pragma unroll\n"
        "    for(int k = 0; k < 8; k++) {\n"
        "      if (count[k] > 0) {\n"
        "        float p = __fdividef((float)count[k], (float)total);\n"
        "        ent_lo -= p * log2f(p);\n"
        "      }\n"
        "    }\n"
        "    float ent_hi = 0.0f;\n"
        "    #pragma unroll\n"
        "    for(int k = 8; k < 16; k++) {\n"
        "      if (count[k] > 0) {\n"
        "        float p = __fdividef((float)count[k], (float)total);\n"
        "        ent_hi -= p * log2f(p);\n"
        "      }\n"
        "    }\n"
        "    ent = ent_lo + ent_hi;\n"
        "  }\n"
    )
    return code.replace(needle, replacement, 1), "split entropy's 16-iteration accumulation into two 8-iteration partial sums (k<8, k>=8) combined at the end"


def fp_reassociation_threadfence(code: str) -> tuple[str, str]:
    """`sum` kernel's per-thread strided accumulation `threadSum += array[idx]`. Trip count
    is runtime (depends on N and totalThreads) so a midpoint split isn't precomputable
    up front -- reassociate via an even/odd-iteration pairwise split instead (classic
    pairwise-summation shape, no precomputed count needed)."""
    kernel = "threadfence"
    needle = (
        "  float threadSum = 0.0f;\n\n"
        "  for (; idx < (unsigned int)N; idx += totalThreads) {\n"
        "    threadSum += array[idx];\n"
        "  }\n"
    )
    _require_literal(needle, code, kernel, "FP-REASSOCIATION")
    replacement = (
        "  float threadSum = 0.0f;\n"
        "  float threadSum_a = 0.0f, threadSum_b = 0.0f;\n"
        "  bool _fpr_toggle = false;\n\n"
        "  for (; idx < (unsigned int)N; idx += totalThreads) {\n"
        "    if (_fpr_toggle) threadSum_b += array[idx]; else threadSum_a += array[idx];\n"
        "    _fpr_toggle = !_fpr_toggle;\n"
        "  }\n"
        "  threadSum = threadSum_a + threadSum_b;\n"
    )
    return code.replace(needle, replacement, 1), "split threadSum's strided accumulation into even/odd-iteration partial sums (pairwise-style), combined at the end"


def fp_reassociation_atomiccost(code: str) -> tuple[str, str]:
    """woAtomicOnGlobalMem<T>'s countable range accumulation `acc += (i&1)`. begin/end are a
    known runtime range -- split at the midpoint into two partial sums added back to the
    pre-existing residual `acc`. Integer T instantiations are exactly associative (expect
    NEUTRAL there); float/double instantiations may show rounding drift (expect
    BEHAVIOR_CHANGING) -- a real demonstration of the same operator being type-dependent."""
    kernel = "atomicCost"
    needle = (
        "  T acc = result[tid];\n"
        "  for (unsigned int i = begin; i < end; ++i) {\n"
        "    acc += static_cast<T>(i & 1U);\n"
        "  }\n"
        "  result[tid] = acc;\n"
    )
    _require_literal(needle, code, kernel, "FP-REASSOCIATION")
    replacement = (
        "  T acc = result[tid];\n"
        "  unsigned int _fpr_mid = begin + (end - begin) / 2;\n"
        "  T acc_lo = static_cast<T>(0);\n"
        "  for (unsigned int i = begin; i < _fpr_mid; ++i) {\n"
        "    acc_lo += static_cast<T>(i & 1U);\n"
        "  }\n"
        "  T acc_hi = static_cast<T>(0);\n"
        "  for (unsigned int i = _fpr_mid; i < end; ++i) {\n"
        "    acc_hi += static_cast<T>(i & 1U);\n"
        "  }\n"
        "  acc += acc_lo + acc_hi;\n"
        "  result[tid] = acc;\n"
    )
    return code.replace(needle, replacement, 1), "split woAtomicOnGlobalMem<T>'s [begin,end) accumulation at the midpoint into two partial sums added back to the residual acc"


FP_REASSOCIATION = {
    "entropy": fp_reassociation_entropy,
    "threadfence": fp_reassociation_threadfence,
    "atomicCost": fp_reassociation_atomiccost,
}
FP_REASSOCIATION_NOT_APPLICABLE = {
    "jaccard": (
        "its only genuine sum+=expr accumulation (jaccard_is_opt's local_sum) has a "
        "data-dependent trip count (two-pointer CSR sorted-merge, only intersection "
        "iterations execute the +=) -- no clean generic-rule midpoint split exists without "
        "risking a hand-authored, kernel-specific patch. Confirmed 2026-08-23."
    ),
}


# ---------------------------------------------------------------------------
# 5. REDUCTION-ORDER-CHANGE
# ---------------------------------------------------------------------------
# CORRECTION to the catalog: 0/17 applicable in the real corpus, not 1/17. lid-driven-cavity's
# canonical base (v10_Trial1) does NOT contain the assumed shared-memory tree-reduction
# pattern (`while(s!=0){ if(tid<s) sdata[tid]=op(...) } s>>=1;`) in any of sum_pressure/
# calc_residual/calculate_u/calculate_v -- all 4 use a two-level warp-shuffle reduction
# (warp_reduce_sum/warp_reduce_max + a `lane < numWarps` block-level merge guard) instead.
# The closest structural analog (flipping `lane < numWarps` to `lane >= numWarps`) changes a
# genuinely different code shape than "which half of threads wins a shared-memory tree step"
# and was judged, per the "coverage = every applicable cell filled, a thin true count is a
# correct answer, not a gap" convention already established for this operator, not worth
# forcing into a supposedly generic-rule operator. No REDUCTION-ORDER-CHANGE instance is
# implemented this session; this operator contributes 0 cells to the corpus.
REDUCTION_ORDER_CHANGE = {}
REDUCTION_ORDER_CHANGE_NOT_APPLICABLE = {
    "lid-driven-cavity": (
        "canonical base (v10_Trial1) uses warp-shuffle reduction throughout (warp_reduce_sum/"
        "warp_reduce_max), not the shared-memory tree pattern the catalog assumed -- confirmed "
        "absent by direct read of all 4 named kernels, 2026-08-23. Corpus-wide applicable "
        "count for this operator is 0/17, not 1/17."
    ),
}


_EXTREMA_KERNELS_H = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source_codes", "HeCBench", "extrema", "kernels.h",
)


def boundary_guard_extrema(code: str, scratch_header_path: str) -> tuple[str, str, dict]:
    """The 2D kernel's guard lives in kernels.h, #include'd by absolute path -- never mutate
    the shared header in place (used by every other extrema experiment). Instead: write a
    patched copy to `scratch_header_path` and rewrite the Trial file's #include line to
    point there. Returns (patched_trial_code, diff, {scratch_header_path: patched_header_text}).
    """
    kernel = "extrema"
    with open(_EXTREMA_KERNELS_H) as f:
        header = f.read()
    needle = "if ( ( tx < in_y ) && ( ty < in_x ) ) {"
    _require_literal(needle, header, kernel, "BOUNDARY-GUARD-CORRUPTION")
    replacement = "if ( ( tx <= in_y ) && ( ty <= in_x ) ) {"
    patched_header = header.replace(needle, replacement, 1)

    include_needle = f'#include "{_EXTREMA_KERNELS_H}"'
    _require_literal(include_needle, code, kernel, "BOUNDARY-GUARD-CORRUPTION")
    patched_code = code.replace(include_needle, f'#include "{scratch_header_path}"', 1)

    diff = (
        f"kernels.h: {needle!r} -> {replacement!r} (relextrema_2D guard, patched in a scratch "
        f"copy at {scratch_header_path}, NOT the shared source_codes header); "
        f"Trial file's #include rewritten to point at the scratch copy"
    )
    return patched_code, diff, {scratch_header_path: patched_header}


BOUNDARY_GUARD_CORRUPTION = {
    "colorwheel": boundary_guard_colorwheel,
    "entropy": boundary_guard_entropy,
    "murmurhash3": boundary_guard_murmurhash3,
    "pathfinder": boundary_guard_pathfinder,
    "randomaccess": boundary_guard_randomaccess,
    "threadfence": boundary_guard_threadfence,
    "matrix-rotate": boundary_guard_matrix_rotate,
    # "extrema" handled separately via boundary_guard_extrema() -- different signature
    # (needs a scratch_header_path) since it's the one two-file instance in this operator.
}
