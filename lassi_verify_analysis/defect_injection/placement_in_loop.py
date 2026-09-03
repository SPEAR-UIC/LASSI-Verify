"""
PLACEMENT-IN-LOOP: the sole instrumentation-defect operator (catalog entry 7).

Mechanism (corrected 2026-08-23, see planning/LASSI_Verify_Phase2_Bridge.md's design-proper
entry): install the PRIMARY witness device-function bodies (`_t1_chain_write_sentinel`/
`_t1_chain_update` writing/checking a fixed magic sentinel) where the live pipeline would
normally select the FALLBACK (snapshot-based) bodies on a kernel already characterized as
in-place-with-no-full-reinit. Detector is the non-interference gate (compare_passthrough_outputs
between a T1+T2-only reference and the full witness-instrumented build) -- NOT Tiers 1-3.

Applicable cells: matrix-rotate, atomicCost, pathfinder (floydwarshall recorded
inapplicable-as-is, see the injection-implementation session's dated bridge entry -- compounding
with a reinit-removal would break the one-canonical-base-one-operator measurement design).

Genuine finding from this session, empirically verified (not assumed from prior documentation):
ALL THREE canonical-base files, exactly as archived, ALREADY have the PRIMARY design installed --
`_enforce_canonical_witness_device_fns()` is therefore a no-op on all three (diff is empty).
This contradicts the injection hand-off doc's stated fact that matrix-rotate's canonical base
had the fallback design active -- that claim was based on the trial's SOURCE_verify file (which
does use fallback, confirmed) without checking whether the fallback decision propagated into the
specific winning variant. It did not, for this variant. Live-tested (this session) results:
  - matrix-rotate (v3_Trial1, out-of-place/double-buffered per the bridge doc's own prior
    finding): gate verdict PASS -- NEUTRAL. This variant doesn't read back the sentinel-targeted
    buffer, so forcing primary (already present) doesn't corrupt it. Does NOT reproduce the
    "confirmed fatal" characterization -- that finding applies to matrix-rotate's SOURCE_verify
    structure or a different variant, not to this specific campaign winner.
  - atomicCost (v4_Trial1): gate verdict FAIL -- detected. Matches "confirmed plausible."
  - pathfinder (v6_Trial1): gate verdict FAIL -- detected. Matches "confirmed present."
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import lassi.verify as verify  # noqa: E402

APPLICABLE_KERNELS = ["matrix-rotate", "atomicCost", "pathfinder"]
INAPPLICABLE_AS_IS = {
    "floydwarshall": (
        "in-place across its inner numPasses loop but protected in practice by the outer "
        "loop's full host reinit; compounding with a reinit-removal would be a two-operator "
        "instance, out of scope for a single-operator cell (2026-08-23 decision)."
    ),
}


def force_primary_design(code: str) -> tuple[str, int, bool]:
    """Force the primary witness device-function bodies onto `code`.

    Returns (patched_code, n_functions_replaced, was_noop). was_noop=True means the file
    already had the primary design installed (n_functions_replaced still reflects the
    (idempotent) replacement count from _enforce_canonical_witness_device_fns, but the text
    is unchanged) -- this was found to be the case for all 3 applicable kernels' canonical
    bases this session, so callers should not assume a nonzero diff.
    """
    patched, n = verify._enforce_canonical_witness_device_fns(code)
    return patched, n, (patched == code)


def build_instance(kernel: str) -> dict:
    """Build one PLACEMENT-IN-LOOP instance's pre-gate materials for `kernel`.

    Does not compile/run -- that's the sweep harness's job (reuses the same T1+T2-only-vs-
    full-witness comparison as the non-interference gate itself).
    """
    if kernel not in APPLICABLE_KERNELS:
        raise ValueError(f"{kernel} is not an applicable PLACEMENT-IN-LOOP cell")
    from .kernel_bases import base_main_path

    with open(base_main_path(kernel)) as f:
        original = f.read()
    patched, n_replaced, was_noop = force_primary_design(original)
    t1t2_only_reference = verify._strip_witness_chain_blocks(patched)
    return {
        "kernel": kernel,
        "injected_code": patched,
        "t1t2_only_reference": t1t2_only_reference,
        "n_functions_replaced": n_replaced,
        "was_noop": was_noop,
        "diff": "" if was_noop else "(primary witness device-function bodies installed)",
    }
