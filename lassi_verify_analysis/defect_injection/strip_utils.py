"""
Tier-stripping helpers for the Step 4 sweep, on top of lassi.verify's own strippers.

Empirically confirmed 2026-08-23 (see planning/LASSI_Verify_Phase2_Bridge.md's Step 4
injection-implementation dated entry): `lassi.verify._strip_t2_blocks()` matches the generic
`BEGIN/END LASSI-VERIFY PASSTHROUGH` delimiter with no label filter, and the witness-chain
harness's `witness_chain_state` print uses that same delimiter convention (it IS an ordinary
PASSTHROUGH block, just with a special label). So calling `_strip_t2_blocks()` on an
already-witness-instrumented file also deletes the witness_chain_state block -- collapsing the
"minus-T2" and "minus-witness" ablations, which are supposed to be independent leave-one-out
configurations. In production this never arises (the T1+T2-only reference is captured BEFORE
witness-chain insertion, never derived by stripping an already-instrumented file), but our
sweep needs both ablations from one archived, fully-instrumented file.

`strip_t2_blocks_except_witness()` is the fix: same strip, with the witness_chain_state block
carved out and preserved. Verified directly against matrix-rotate's canonical base: original 4
PASSTHROUGH blocks (2x witness_chain_state, 2x final_rotated_matrix) -> after this function, the
2 witness_chain_state blocks survive, the 2 ordinary blocks are gone, and the witness sentinel
device-function calls (`_t1_chain_write_sentinel`/`_t1_chain_update`) remain wired in.
"""
from __future__ import annotations

import re

_PASSTHROUGH_BLOCK_RE = re.compile(
    r"// >>> BEGIN LASSI-VERIFY PASSTHROUGH.*?// <<< END LASSI-VERIFY PASSTHROUGH <<<\n?",
    re.DOTALL,
)

WITNESS_STATE_LABEL = "[witness_chain_state]"


def strip_t2_blocks_except_witness(code: str) -> str:
    """Strip ordinary Tier-2 PASSTHROUGH blocks; preserve the witness_chain_state block(s)
    so a 'minus-T2' configuration doesn't also silently disable witness-chain detection."""

    def _repl(m: re.Match) -> str:
        block = m.group(0)
        return block if WITNESS_STATE_LABEL in block else ""

    return _PASSTHROUGH_BLOCK_RE.sub(_repl, code)
