"""
Step 4 orchestration: runs every confirmed-applicable program-defect cell (operator x kernel)
through neutrality-check + the 5-configuration sweep, populates provenance JSON per operator,
and writes the two detection-rate tables. PLACEMENT-IN-LOOP (the instrumentation-defect
operator) is recorded separately from real results already obtained this session (see
placement_in_loop.py's module docstring) -- it is NOT re-run through this script, since its
detector is the non-interference gate, not Tiers 1-3, and its population must never be pooled
with the program-defect operators below.

Usage:
    python -m lassi_verify_analysis.defect_injection.run_step4 [--pilot] [--operator NAME]
                                                                 [--kernel NAME]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lassi_verify_analysis.defect_injection import operators as ops
from lassi_verify_analysis.defect_injection import cheat_loop_deletion as cld
from lassi_verify_analysis.defect_injection import sweep_harness as sh
from lassi_verify_analysis.defect_injection import neutrality_check as nc
from lassi_verify_analysis.defect_injection import provenance as prov
from lassi_verify_analysis.defect_injection.kernel_bases import (
    KERNEL_BASES, base_main_path, base_t3_paths,
)

WORKDIR_ROOT = "/tmp/step4_sweep"

OPERATOR_META = {
    "sync-removal": dict(
        family="program-defect", detector="tiers_1_3", mechanism="generic_rule",
        observed_failure_class_generalized="deleted synchronization gating a write-then-read hazard",
        evidence_pointer="planning/LASSI_Verify_Phase2_Bridge.md 2026-08-23 design-proper entry, catalog entry 1",
        mechanism_description="delete a __syncthreads()/__threadfence() call gating a shared-memory or shared-accumulator write-then-read hazard",
        applicability_rule="a __syncthreads()/__threadfence() call textually precedes a shared-memory write read by a different code path",
        patch_map=ops.SYNC_REMOVAL,
    ),
    "atomicity-weakening": dict(
        family="program-defect", detector="tiers_1_3", mechanism="generic_rule",
        observed_failure_class_generalized="atomic intrinsic replaced with non-atomic equivalent on a multi-thread-addressable buffer",
        evidence_pointer="planning/LASSI_Verify_Phase2_Bridge.md 2026-08-23 design-proper entry, catalog entry 2",
        mechanism_description="replace atomicAdd/atomicXor with a non-atomic op= on a data-dependent (non-bijective) index",
        applicability_rule="an atomic call's index expression is confirmed data-dependent, not a bijective per-thread id",
        patch_map=ops.ATOMICITY_WEAKENING,
    ),
    "fp-reassociation": dict(
        family="program-defect", detector="tiers_1_3", mechanism="generic_rule",
        observed_failure_class_generalized="floating-point accumulation reordered beyond the buffer's measured stability",
        evidence_pointer="planning/LASSI_Verify_Phase2_Bridge.md 2026-08-23 design-proper entry, catalog entry 4",
        mechanism_description="split a serial FP accumulation loop into two partial sums combined at the end",
        applicability_rule="a genuine T sum=0; sum+=expr; (or -=) accumulation loop feeding an output buffer, with a staticaly analyzable trip range",
        patch_map=ops.FP_REASSOCIATION,
    ),
    "boundary-guard-corruption": dict(
        family="program-defect", detector="tiers_1_3", mechanism="generic_rule",
        observed_failure_class_generalized="runtime-parametric bounds guard widened by one, causing an OOB read/write",
        evidence_pointer="planning/LASSI_Verify_Phase2_Bridge.md 2026-08-23 design-proper entry, catalog entry 6",
        mechanism_description="widen a runtime-parametric entry guard (if-guard or grid-stride loop bound) by one",
        applicability_rule="a runtime/CLI-derived bound N gates an array access, and is not a compile-time #define",
        patch_map=ops.BOUNDARY_GUARD_CORRUPTION,
    ),
    "cheat-loop-deletion": dict(
        family="program-defect", detector="tiers_1_3", mechanism="hand_authored",
        observed_failure_class_generalized="idempotent outer repeat/timing loop collapsed to one iteration",
        evidence_pointer="planning/LASSI_Verify_Phase2_Bridge.md 2026-08-23 design-proper entry, catalog entry 3; generalizes project_chacha20_role's documented cheat",
        mechanism_description="collapse the outer HeCBench repeat/timing loop's bound to 1",
        applicability_rule="hand-traced idempotence: the loop body does not read back its own prior iteration's output as input",
        patch_map=None,  # handled specially via cheat_loop_deletion module
    ),
}


def load(path: str) -> str:
    with open(path) as f:
        return f.read()


def load_t3(kernel: str) -> dict[str, str]:
    return {os.path.basename(p): load(p) for p in base_t3_paths(kernel)}


def build_cells(operator_filter=None, kernel_filter=None):
    cells = []
    for op_id, meta in OPERATOR_META.items():
        if operator_filter and op_id != operator_filter:
            continue
        if op_id == "cheat-loop-deletion":
            kernels = cld.APPLICABLE_KERNELS
        elif op_id == "boundary-guard-corruption":
            # extrema is handled specially in run_cell() (its guard lives in an #include'd
            # header, not the Trial file) -- not a key in ops.BOUNDARY_GUARD_CORRUPTION, so
            # it must be added explicitly here or the cell is silently dropped.
            kernels = list(meta["patch_map"].keys()) + ["extrema"]
        else:
            kernels = list(meta["patch_map"].keys())
        for kernel in kernels:
            if kernel_filter and kernel != kernel_filter:
                continue
            cells.append((op_id, kernel))
    return cells


def apply_operator(op_id: str, kernel: str, code: str) -> tuple[str, str]:
    if op_id == "cheat-loop-deletion":
        return cld.apply_cheat_loop_deletion(kernel, code)
    if kernel == "extrema" and op_id == "boundary-guard-corruption":
        raise RuntimeError("extrema needs the special 2-file boundary_guard_extrema() path")
    return OPERATOR_META[op_id]["patch_map"][kernel](code)


def run_cell(op_id: str, kernel: str, n_repeat: int = 3) -> dict:
    base_main = load(base_main_path(kernel))
    base_t3 = load_t3(kernel)
    extra_files = {}

    if kernel == "extrema" and op_id == "boundary-guard-corruption":
        scratch_header = os.path.join(WORKDIR_ROOT, f"{op_id}_{kernel}", "kernels_patched.h")
        os.makedirs(os.path.dirname(scratch_header), exist_ok=True)
        injected_main, diff, extra_files = ops.boundary_guard_extrema(base_main, scratch_header)
        for path, content in extra_files.items():
            with open(path, "w") as f:
                f.write(content)
        injected_t3 = dict(base_t3)  # extrema's T3 companions don't reference the 2D kernel's guard
    else:
        injected_main, diff = apply_operator(op_id, kernel, base_main)
        injected_t3 = {}
        for fn, code in base_t3.items():
            try:
                patched, _ = apply_operator(op_id, kernel, code)
                injected_t3[fn] = patched
            except ops.PatternNotFoundError:
                injected_t3[fn] = code  # pattern not present in this companion; leave as base

    workdir = os.path.join(WORKDIR_ROOT, f"{op_id}_{kernel}")
    os.makedirs(workdir, exist_ok=True)

    neutrality_record = {}
    try:
        neutrality_record = nc.run_neutrality_check(
            kernel, base_main, injected_main, os.path.join(workdir, "neutrality"), n_repeat=n_repeat
        )
    except Exception as e:
        neutrality_record = {"verdict": "UNRESOLVED", "error": f"{type(e).__name__}: {e}"}

    sweep_results = {}
    try:
        sweep_results = sh.run_instance(
            kernel, base_main, injected_main, base_t3, injected_t3,
            os.path.join(workdir, "sweep"),
        )
    except Exception as e:
        sweep_results = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}

    b = KERNEL_BASES[kernel]
    return {
        "kernel": kernel,
        "base_id": {"experiment_dir": b["exp_dir"], "trial": b["trial"], "code_version": b["code_version"]},
        "diff": diff,
        "neutrality_verdict": neutrality_record.get("verdict", "UNRESOLVED"),
        "decisive_test_record": neutrality_record,
        "in_place_kernel_metadata": None,
        "tier_configuration_results": {
            cfg: sweep_results.get(cfg, {}).get("verdict") if isinstance(sweep_results, dict) and cfg in sweep_results else None
            for cfg in sh.CONFIGS
        },
        "raw_sweep_results": sweep_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--operator", default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--n-repeat", type=int, default=3)
    args = ap.parse_args()

    cells = build_cells(args.operator, args.kernel)
    if args.pilot:
        cells = cells[:4]

    print(f"Running {len(cells)} cell(s)...", flush=True)
    by_operator: dict[str, list[dict]] = {}
    for i, (op_id, kernel) in enumerate(cells):
        t0 = time.time()
        print(f"[{i+1}/{len(cells)}] {op_id} / {kernel} ...", flush=True)
        try:
            instance = run_cell(op_id, kernel, n_repeat=args.n_repeat)
        except Exception as e:
            instance = {"kernel": kernel, "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()}
        by_operator.setdefault(op_id, []).append(instance)
        print(f"    done in {time.time()-t0:.1f}s: neutrality={instance.get('neutrality_verdict')} "
              f"tier_results={instance.get('tier_configuration_results')}", flush=True)

    for op_id, instances in by_operator.items():
        meta = OPERATOR_META[op_id]
        record = prov.new_operator_record(
            operator_id=op_id, family=meta["family"], detector=meta["detector"],
            observed_failure_class_generalized=meta["observed_failure_class_generalized"],
            evidence_pointer=meta["evidence_pointer"], mechanism=meta["mechanism"],
            mechanism_description=meta["mechanism_description"],
            applicability_rule=meta["applicability_rule"],
        )
        record["per_instance"] = instances
        path = prov.save_operator_record(record)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
