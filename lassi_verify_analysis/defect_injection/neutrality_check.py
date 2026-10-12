"""
Neutrality check: is an injected defect actually behavior-changing on the tested workload, or
is it silently a no-op (e.g. FP reassociation that doesn't move the result at this scale, sync
removal benign under the launch config actually used)? Per the design doc (bridge doc's
2026-08-23 entries), this uses the SAME uninstrumented-full-state-diff playbook already used on
the pathfinder/atomicCost decisive tests -- deliberately independent of Tier 2's own checksum/
allclose comparator, to avoid circularity (that comparator is exactly what's under test).

Mechanism: rather than hand-patch each kernel with a bespoke raw dump (the pathfinder/atomicCost
precedent), this generalizes it. Every LASSI-VERIFY PASSTHROUGH block in this project follows one
fixed generated template (confirmed 2026-08-23 across matrix-rotate/pathfinder/randomaccess/
threadfence source): a setup segment computing `lv_N`/`lv_min`/`lv_max`/`lv_sum` over a host
array via a `for (... lv_i = 0; lv_i < <count>; lv_i++)` loop, immediately followed by the
printf-based report. The setup segment already correctly identifies the buffer/count/type for
that label -- circularity-free plumbing, not a comparison mechanism -- so this module keeps the
setup segment verbatim and replaces only the printf reporting with a raw binary `fwrite`. This
is a *block transform*, not the project's block *strip*: `strip_verify_blocks()` still runs
first to remove everything else (T1/T2/T3/witness), and this transform is applied to the label(s)
under test before that strip would otherwise delete them entirely.
"""
from __future__ import annotations

import hashlib
import os
import re

import lassi.verify as verify

from . import kernel_bases
from lassi_verify_analysis import oracle_harness as oh

FULL_BLOCK_RE = re.compile(
    r"// >>> BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\][^\n]*\n(.*?)"
    r"// <<< END LASSI-VERIFY PASSTHROUGH <<<\n?",
    re.DOTALL,
)
SETUP_SPLIT_RE = re.compile(r'printf\(\s*"\\n?=== BEGIN LASSI-VERIFY PASSTHROUGH')
BUF_TYPE_RE = re.compile(r"(\w+)\s+lv_min\s*=\s*(\w+)\s*\[\s*0\s*\]")
COUNT_RE = re.compile(r"for\s*\([^;]*lv_i\s*=\s*0\s*;\s*lv_i\s*<\s*(.+?)\s*;\s*lv_i\+\+\s*\)")

_C_TYPE_TO_STRUCT_ELEM = {
    "float": "float", "double": "double", "int": "int", "long": "long",
    "unsigned": "unsigned int", "u64Int": "unsigned long long",
    "s64Int": "long long", "bool": "int",
}


class DumpSpecError(RuntimeError):
    pass


def extract_dump_spec(code: str, label: str) -> dict:
    """Extract (setup_code, c_type, array_var, count_expr) for one PASSTHROUGH label from the
    ORIGINAL (unstripped) source. Raises DumpSpecError if the fixed template isn't matched --
    refuse to guess rather than silently emit a wrong dump."""
    for m in FULL_BLOCK_RE.finditer(code):
        if m.group(1) != label:
            continue
        body = m.group(2)
        parts = SETUP_SPLIT_RE.split(body, maxsplit=1)
        if len(parts) != 2:
            raise DumpSpecError(f"label {label!r}: printf report marker not found in block")
        setup = parts[0]
        buf_m = BUF_TYPE_RE.search(setup)
        cnt_m = COUNT_RE.search(setup)
        if not buf_m or not cnt_m:
            raise DumpSpecError(
                f"label {label!r}: setup doesn't match the fixed lv_min/lv_N template "
                f"(buf_m={bool(buf_m)}, cnt_m={bool(cnt_m)}) -- do not guess, inspect this "
                f"kernel's PASSTHROUGH block by hand before adding it to the neutrality check"
            )
        return {
            "setup": setup,
            "c_type": buf_m.group(1),
            "array_var": buf_m.group(2),
            "count_expr": cnt_m.group(1),
            "block_span": m.span(),
        }
    raise DumpSpecError(f"label {label!r} not found as a PASSTHROUGH block in this source")


def _dump_block_text(spec: dict, dump_path: str) -> str:
    """`spec['setup']` already opens with the original block's own leading `{` (it's the
    text from the block's opening brace up to the printf marker) -- do NOT add another one,
    or the block ends up with two opens and only one close (a real bug caught by the pilot
    run: bsearch/lid-driven-cavity/pathfinder/threadfence all failed to compile with
    "expected a }" until this was fixed). Just close what setup opened."""
    elem_type = _C_TYPE_TO_STRUCT_ELEM.get(spec["c_type"], spec["c_type"])
    return (
        f"{spec['setup']}"
        f'  FILE* _neutrality_fh = fopen("{dump_path}", "wb");\n'
        f"  fwrite({spec['array_var']}, sizeof({elem_type}), (size_t)({spec['count_expr']}), "
        "_neutrality_fh);\n"
        "  fclose(_neutrality_fh);\n"
        "}\n"
    )


def inject_raw_dumps(code: str, labels: list[str], dump_dir: str, tag: str) -> tuple[str, dict]:
    """Replace each named PASSTHROUGH block's printf reporting with a raw fwrite dump, in a
    single pass over `code` (so span offsets from separate extract_dump_spec calls don't go
    stale after the first replacement). Returns (patched_code, {label: dump_path})."""
    os.makedirs(dump_dir, exist_ok=True)
    dump_paths = {label: os.path.join(dump_dir, f"{tag}_{label}.bin") for label in labels}

    specs = {label: extract_dump_spec(code, label) for label in labels}

    def _repl(m: re.Match) -> str:
        label = m.group(1)
        if label not in specs:
            return m.group(0)
        return _dump_block_text(specs[label], dump_paths[label])

    patched = FULL_BLOCK_RE.sub(_repl, code)
    return patched, dump_paths


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_dumps(base_path: str, injected_path: str, comparison_mode: str,
                   atol: float = 0.0, rtol: float = 0.0, c_type: str = "float") -> dict:
    if comparison_mode == "exact":
        base_hash, inj_hash = _md5(base_path), _md5(injected_path)
        return {"mode": "exact", "match": base_hash == inj_hash,
                "base_md5": base_hash, "injected_md5": inj_hash}
    import numpy as np
    dtype_map = {"float": np.float32, "double": np.float64, "int": np.int32,
                 "u64Int": np.uint64, "s64Int": np.int64, "long": np.int64}
    dt = dtype_map.get(c_type, np.float32)
    a = np.fromfile(base_path, dtype=dt)
    b = np.fromfile(injected_path, dtype=dt)
    if a.shape != b.shape:
        return {"mode": "tolerant", "match": False, "reason": "shape mismatch",
                "base_shape": a.shape, "injected_shape": b.shape}
    match = bool(np.allclose(a.astype(np.float64), b.astype(np.float64), atol=atol, rtol=rtol))
    return {"mode": "tolerant", "match": match, "atol": atol, "rtol": rtol,
            "max_abs_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if a.size else None}


def run_neutrality_check(
    kernel: str,
    base_code: str,
    injected_code: str,
    workdir: str,
    n_repeat: int = 5,
) -> dict:
    """Runs the decisive-test playbook: strip both to uninstrumented, add raw dumps for every
    non-witness PASSTHROUGH label, compile+run n_repeat times each, compare per-label per the
    kernel's buffer_classification.json (default 'exact' for labels/kernels with none on file,
    e.g. matrix-rotate). Returns a decisive_test_record dict for the provenance schema."""
    os.makedirs(workdir, exist_ok=True)
    labels = [lbl for lbl in {m.group(1) for m in FULL_BLOCK_RE.finditer(base_code)}
              if lbl != "witness_chain_state"]

    classification = _load_classification(kernel)

    # Design doc's step 1+3 ("strip everything, then add a fresh generic dump") is realized
    # here as transform-then-strip-the-rest: inject dumps into the UNSTRIPPED code first (so
    # the block-match regex still finds the delimiters), then strip everything else.
    base_dumped, base_paths = inject_raw_dumps(base_code, labels, workdir, "base")
    injected_dumped, injected_paths = inject_raw_dumps(injected_code, labels, workdir, "injected")
    # Now strip all OTHER LASSI-VERIFY blocks (T1/T2-other-labels/T3/witness) but keep our
    # inserted dump blocks -- they are plain C, not LASSI-VERIFY-delimited, so the standard
    # strippers pass over them untouched automatically once the delimiter comments are gone.
    base_final = _strip_non_dump_verify_blocks(base_dumped)
    injected_final = _strip_non_dump_verify_blocks(injected_dumped)

    p_base = os.path.join(workdir, "base_dumped.cu")
    p_inj = os.path.join(workdir, "injected_dumped.cu")
    with open(p_base, "w") as f:
        f.write(base_final)
    with open(p_inj, "w") as f:
        f.write(injected_final)

    exe_base = os.path.join(workdir, "base_dumped")
    exe_inj = os.path.join(workdir, "injected_dumped")
    ok_b, err_b = oh.compile_cu(p_base, exe_base, [])
    ok_i, err_i = oh.compile_cu(p_inj, exe_inj, [])
    if not (ok_b and ok_i):
        return {"method": "uninstrumented_full_state_diff", "n_repeat": n_repeat,
                "verdict": "UNRESOLVED", "error": f"compile failure: base={err_b[:300]} "
                                                    f"injected={err_i[:300]}"}

    run_args = oh.KERNEL_CONFIGS[kernel].run_args if kernel in oh.KERNEL_CONFIGS else []
    per_label_results = {}
    any_change = False
    for i in range(n_repeat):
        ok1, out1, e1 = oh.run_exe(exe_base, run_args)
        ok2, out2, e2 = oh.run_exe(exe_inj, run_args)
        if not (ok1 and ok2):
            if ok1 and not ok2:
                # Base runs fine, injected build crashes -- unambiguous evidence of a real
                # behavior change (a hard crash IS a difference), not an inconclusive result.
                # Matches the same crash-classification sweep_harness.py uses. Confirmed real
                # case: murmurhash3's BOUNDARY-GUARD-CORRUPTION walks off an allocated buffer
                # into unmapped memory -- "illegal memory access" on every run.
                return {"method": "uninstrumented_full_state_diff", "n_repeat": n_repeat,
                        "verdict": "BEHAVIOR_CHANGING",
                        "note": "injected build crashes on run while base succeeds -- crash "
                                "itself is the evidence of behavior change",
                        "error": f"run failure on rep {i}: {e2[:200]}"}
            return {"method": "uninstrumented_full_state_diff", "n_repeat": n_repeat,
                    "verdict": "UNRESOLVED", "error": f"run failure on rep {i}: {e1[:200]} {e2[:200]}"}
        for label in labels:
            # Some blocks (e.g. bsearch's bs/bs2/bs3 outputs) are wrapped in #ifdef DEBUG and
            # preprocessed out of a standard build -- their dump file is never written. Not an
            # error: skip that label rather than crash, and don't count it in NEUTRAL/
            # BEHAVIOR_CHANGING (nothing to compare, this block simply isn't in this binary).
            if not (os.path.isfile(base_paths[label]) and os.path.isfile(injected_paths[label])):
                per_label_results.setdefault(label, []).append(
                    {"mode": "n/a", "match": None, "reason": "dump file not produced "
                     "(block likely preprocessed out, e.g. #ifdef DEBUG)"})
                continue
            mode_info = classification.get(label, {"comparison_mode": "exact"})
            mode = mode_info.get("comparison_mode", "exact")
            tol = mode_info.get("tolerance") or {}
            c_type = mode_info.get("declared_type") or _guess_type(base_code, label)
            res = compare_dumps(base_paths[label], injected_paths[label], mode,
                                 atol=tol.get("atol", 0.0), rtol=tol.get("rtol", 0.0),
                                 c_type=c_type)
            per_label_results.setdefault(label, []).append(res)
            if not res["match"]:
                any_change = True

    verdict = "BEHAVIOR_CHANGING" if any_change else "NEUTRAL"
    return {
        "method": "uninstrumented_full_state_diff",
        "n_repeat": n_repeat,
        "per_buffer_comparison_mode_used": {lbl: classification.get(lbl, {}).get("comparison_mode", "exact")
                                             for lbl in labels},
        "per_label_results": per_label_results,
        "verdict": verdict,
    }


def _guess_type(code: str, label: str) -> str:
    try:
        return extract_dump_spec(code, label)["c_type"]
    except DumpSpecError:
        return "float"


def _strip_non_dump_verify_blocks(code: str) -> str:
    """After inject_raw_dumps() has already turned the labels-of-interest's PASSTHROUGH blocks
    into plain C (no LASSI-VERIFY delimiters left), run the normal full strip to remove
    everything else (T1, other/witness PASSTHROUGH blocks, WITNESS_CHAIN launches)."""
    return verify.strip_verify_blocks(code)


def _load_classification(kernel: str) -> dict:
    path = kernel_bases.buffer_classification_path(kernel)
    if path is None or not os.path.isfile(path):
        return {}
    import json
    with open(path) as f:
        data = json.load(f)
    return data.get("buffers", {})
