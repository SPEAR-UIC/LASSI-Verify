"""
CHEAT-LOOP-DELETION: collapse the outer HeCBench repeat/timing loop to one iteration
(catalog entry 3; hand-authored per the no-generic-syntactic-rule convention -- idempotence is
a semantic property, not mechanically checkable without risking a false-idempotent
classification, so each kernel's applicability is individually confirmed by direct trace of
what each iteration reads, not inferred from a `repeat`-looking variable's mere presence).

Mechanism (uniform across all 13, per the operator's own "delete OR collapse to one iteration"
definition): rewrite the loop bound expression to a literal `1`, rather than deleting/unwrapping
the loop body -- avoids brace-matching risk and is explicitly sanctioned by the catalog wording.
Each kernel's loop-variable/bound-variable names and idempotence argument were confirmed by
direct source read this session (2026-08-23) -- see planning/LASSI_Verify_Phase2_Bridge.md's
Step 4 injection-implementation dated entry for the full per-kernel evidence this module reuses
verbatim rather than re-deriving.

floydwarshall carries an internal caveat (already in the catalog): only the OUTER
`numIterations` loop is a valid target -- the INNER `numTiles` loop is the real sequential
Floyd-Warshall algorithm (tile i depends on tile i-1) and must never be touched.
"""
from __future__ import annotations

import re

from .operators import PatternNotFoundError


def _collapse_loop_bound(code: str, kernel: str, pattern: str, expected_count: int) -> tuple[str, str]:
    """`pattern` must have exactly one capture group around the bound expression to replace
    with a literal `1`. Fails loudly if the match count doesn't equal `expected_count` --
    a loop appearing more or fewer times than confirmed this session means the file has
    drifted and this operator must not be applied blindly."""
    matches = list(re.finditer(pattern, code))
    if len(matches) != expected_count:
        raise PatternNotFoundError(
            f"CHEAT-LOOP-DELETION on {kernel}: expected {expected_count} loop header(s) "
            f"matching {pattern!r}, found {len(matches)} -- re-verify before trusting this "
            f"operator on the current canonical base"
        )

    out = []
    last = 0
    replaced_bounds = []
    for m in matches:
        out.append(code[last:m.start(1)])
        replaced_bounds.append(m.group(1))
        out.append("1")
        last = m.end(1)
    out.append(code[last:])
    patched = "".join(out)
    diff = (
        f"collapsed {len(matches)} loop bound(s) {replaced_bounds} -> 1 "
        f"(pattern={pattern!r})"
    )
    return patched, diff


# Each entry: (regex with one capture group around the bound expression, expected occurrence
# count, idempotence argument confirmed this session).
_KERNEL_LOOPS: dict[str, dict] = {
    "all-pairs-distance": dict(
        pattern=r"for\s*\(\s*int\s+n\s*=\s*0\s*;\s*n\s*<\s*(iterations)\s*;\s*n\+\+\s*\)",
        expected_count=3,
        idempotence="k1/k2/k3 each recompute distance_device fully from the fixed, never-"
                    "mutated data_char_device input every iteration; k1's in-loop memset "
                    "only zeroes the output before recompute.",
    ),
    "bsearch": dict(
        pattern=r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(repeat)\s*;\s*i\+\+\s*\)",
        expected_count=4,
        idempotence="bs/bs2/bs3/bs4 each fully overwrite d_r from const, never-mutated "
                    "d_a/d_z every call; no pre-loop one-time init to preserve.",
    ),
    "chacha20": dict(
        pattern=r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(repeat)\s*;\s*i\+\+\s*\)",
        expected_count=1,
        idempotence="loop body replays a captured CUDA graph (cudaGraphLaunch, not a raw "
                    "kernel<<<>>>); d_result is fully overwritten by test_keystreams on every "
                    "replay from fixed d_raw_key/d_raw_nonce -- collapsing the bound to 1 "
                    "still yields exactly one graph replay, no special-casing needed.",
    ),
    "colorwheel": dict(
        pattern=r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(repeat)\s*;\s*i\+\+\s*\)",
        expected_count=1,
        idempotence="color kernel computes each pixel purely from fixed scalar params, full "
                    "overwrite of d_pix every call; one-time cudaMemset(d_pix) sits well "
                    "before the loop and is unaffected by collapsing it.",
    ),
    "extrema": dict(
        pattern=r"for\s*\(\s*int\s+n\s*=\s*0\s*;\s*n\s*<\s*(repeat)\s*;\s*n\+\+\s*\)",
        expected_count=2,
        idempotence="both the 1D and 2D test loops fully recompute d_result from d_x, "
                    "which is copied once before the loop and never mutated inside it.",
    ),
    "floydwarshall": dict(
        pattern=r"for\s*\(\s*unsigned\s+int\s+n\s*=\s*0\s*;\s*n\s*<\s*(numIterations)\s*;\s*n\+\+\s*\)",
        expected_count=1,
        idempotence="OUTER loop only -- re-uploads pathDistanceMatrix/pathMatrix from the "
                    "unchanging host arrays at the top of every outer iteration before "
                    "recomputing; the INNER numTiles loop (tile i depends on tile i-1) is "
                    "the real sequential algorithm and is never touched by this pattern.",
    ),
    "layout": dict(
        pattern=r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(iterations)\s*;\s*i\+\+\s*\)",
        expected_count=2,
        idempotence="both the AoS and SoA loops fully overwrite outputBuffer from "
                    "inputBuffer, which is not written inside either loop body.",
    ),
    "murmurhash3": dict(
        pattern=r"for\s*\(\s*uint32_t\s+n\s*=\s*0\s*;\s*n\s*<\s*(repeat)\s*;\s*n\+\+\s*\)",
        expected_count=1,
        idempotence="dev_out is fully overwritten from fixed dev_keys/dev_length/key_length "
                    "every iteration.",
    ),
    "randomaccess": dict(
        pattern=r"for\s*\(\s*int\s+irep\s*=\s*0\s*;\s*irep\s*<\s*(repeat)\s*;\s*irep\+\+\s*\)",
        expected_count=1,
        idempotence="canonical-init idempotence: initTable() re-resets d_Table to its "
                    "canonical starting state INSIDE the loop, immediately before update() "
                    "runs -- each iteration discards the prior iteration's mutated table.",
    ),
    "segment-reduce": dict(
        pattern=r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(repeat)\s*;\s*i\+\+\s*\)",
        expected_count=1,
        idempotence="d_keys_out/d_out are fully recomputed each iteration from the same "
                    "fixed transform-iterator inputs; the only cross-iteration state "
                    "(actual_segments, captured on i==repeat-1) is a printf-only report "
                    "value, harmless when repeat collapses to 1 (i==repeat-1 still holds).",
    ),
    "threadfence": dict(
        pattern=r"for\s*\(\s*int\s+n\s*=\s*0\s*;\s*n\s*<\s*(repeat)\s*;\s*n\+\+\s*\)",
        expected_count=1,
        idempotence="d_array is fixed and never mutated inside the loop; d_result/d_partial "
                    "are fully recomputed each outer iteration via the inner ping-pong "
                    "reduction. Note: this variant's `count` kernel parameter is dead code "
                    "((void)count;) -- the classic threadfence last-block atomic pattern is "
                    "not actually exercised by this refactor.",
    ),
    "jaccard": dict(
        pattern=r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*(iteration)\s*;\s*i\+\+\s*\)",
        expected_count=1,
        idempotence="explicit-reset idempotence: d_weight_j/d_weight_i/d_work/d_weight_s are "
                    "all re-filled or fully overwritten at the start of every iteration "
                    "before being read; graph-structure inputs are copied once before the "
                    "loop and never mutated inside it.",
    ),
}

APPLICABLE_KERNELS = list(_KERNEL_LOOPS.keys())
NOT_APPLICABLE = {
    "atomicCost": "accumulates across repeat iterations with no reset -- genuinely non-idempotent.",
    "pathfinder": "its only loop is the real sequential dynamic-programming algorithm itself -- no separate timing wrapper exists.",
    "lid-driven-cavity": "its only loop is a genuine physical time-stepping loop -- no separate timing wrapper exists.",
    "matrix-rotate": "composing rotations is not idempotent; deleting the loop changes the answer whenever repeat mod 4 != 0 (the validated workload deliberately uses repeat=101).",
    "entropy": "the kernel IS idempotent (this part of the original applicability check was "
               "correct), but the framework's own validated workload for this kernel already "
               "runs repeat=1 (confirmed 2026-08-27 from the real campaign's own monitor log: "
               "'Execute command: [...,\"10000\",\"1024\",\"1\"]') -- collapsing a loop bound "
               "from 1 to 1 is a no-op. Applicability must check the validated repeat count, not "
               "just idempotence; this was missed in the original design-proper applicability "
               "pass and produced a spurious 'undetected everywhere' result before being caught.",
}


def apply_cheat_loop_deletion(kernel: str, code: str) -> tuple[str, str]:
    if kernel not in _KERNEL_LOOPS:
        raise ValueError(
            f"{kernel} is not an applicable CHEAT-LOOP-DELETION cell"
            + (f" ({NOT_APPLICABLE[kernel]})" if kernel in NOT_APPLICABLE else "")
        )
    spec = _KERNEL_LOOPS[kernel]
    patched, diff = _collapse_loop_bound(code, kernel, spec["pattern"], spec["expected_count"])
    full_diff = f"{diff}\nidempotence argument: {spec['idempotence']}"
    return patched, full_diff
