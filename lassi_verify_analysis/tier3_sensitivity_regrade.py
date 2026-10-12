"""Tier-3 SENSITIVITY regrade harness — validates the entropy Trial-3 `single_pixel_sensitivity`
false-reject fix (Tier-1 assertion-regression guardrail + local-response probe) WITHOUT a full
live LLM pipeline re-run.

Two independent checks, mirroring the two pieces of the fix:

  Piece 1 (assertion-regression guardrail): recompiles/re-runs the ALREADY on-disk
  `single_pixel_sensitivity__p_outlier` Stage-0 binaries from
  KERNEL_TEST_ENTROPY_POST-JOBii_1__07-18-2026_t21_41_31 (source + v3..v8) against the plain
  unperturbed `_verify_Trial3` baseline, and confirms `lassi.verify._check_assertion_regression`
  flags `input_value_range_valid` as regressed for every one of them — i.e. Stage 0 would now
  reject this exact outlier value and retry, instead of freezing it as the canonical stimulus.
  This replays real, previously-captured framework artifacts; no LLM calls, no re-instrumentation.

  Piece 2 (local-response probe parsing/folding): since no live kernel has produced the new
  `Sample_at_target_before`/`Sample_at_target_after` fields yet (that requires a fresh LLM-authored
  run), this exercises `check_metamorphic_relation`'s SENSITIVITY branch directly against
  hand-built synthetic METAMORPHIC/PASSTHROUGH stdout fixtures: (a) a "today" fixture with no
  target-window fields, confirming backward-compatible FAIL is unchanged, and (b) a fixture where
  global stats/first10 match (the invisible-locally-perturbed-huge-array scenario) but the target
  window differs, confirming the new fields alone flip the verdict to PASS.

Usage:
    python -m lassi_verify_analysis.tier3_sensitivity_regrade \
        [--exp-dir generated_codes/experiment/KERNEL_TEST_ENTROPY_POST-JOBii_1__07-18-2026_t21_41_31]
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess

from lassi.verify import (
    _check_assertion_regression,
    check_metamorphic_relation,
    _extract_mt_transform_block,
    _extract_mt_target_index_block,
    _enforce_canonical_transform,
    _enforce_canonical_target_index,
)

DEFAULT_EXP_DIR = (
    "generated_codes/experiment/KERNEL_TEST_ENTROPY_POST-JOBii_1__07-18-2026_t21_41_31"
)
RUN_ARGS = ["10000", "1024", "1"]  # width height repeat, as recorded for this trial
TIMEOUT_S = 60


def _run(binary_path):
    exe_dir = os.path.dirname(os.path.abspath(binary_path))
    exe_base = os.path.basename(binary_path)
    result = subprocess.run(
        ["./" + exe_base] + RUN_ARGS, cwd=exe_dir,
        capture_output=True, text=True, timeout=TIMEOUT_S,
    )
    return result.stdout


def validate_piece1(exp_dir):
    print("\n=== Piece 1: Tier-1 assertion-regression guardrail (real on-disk artifacts) ===")
    baseline_glob = glob.glob(os.path.join(exp_dir, "*_verify_Trial3"))
    baseline_glob = [p for p in baseline_glob if os.access(p, os.X_OK)]
    if not baseline_glob:
        print(f"  SKIP: no *_verify_Trial3 baseline binary found in {exp_dir}")
        return
    baseline_stdout = _run(baseline_glob[0])

    variant_pattern = os.path.join(
        exp_dir, "*_mt_single_pixel_sensitivity__p_outlier_*"
    )
    candidates = [
        p for p in sorted(glob.glob(variant_pattern))
        if os.access(p, os.X_OK) and not p.endswith(".cu")
    ]
    if not candidates:
        print(f"  SKIP: no p_outlier binaries found matching {variant_pattern}")
        return

    all_flagged = True
    for path in candidates:
        label = os.path.basename(path).split("p_outlier_")[-1]
        stdout = _run(path)
        regressed = _check_assertion_regression(baseline_stdout, stdout)
        status = "FLAGGED" if regressed else "NOT FLAGGED"
        if not regressed:
            all_flagged = False
        print(f"  [{label:10s}] {status:12s} regressed_labels={regressed}")

    print(f"\n  Result: {'PASS' if all_flagged else 'FAIL'} — "
          f"{'every' if all_flagged else 'NOT every'} p_outlier build was flagged as a "
          f"domain-invalid stimulus.")


# --- Piece 2: synthetic fixtures -------------------------------------------------------------

_PASS_BLOCK_TEMPLATE = """
=== BEGIN LASSI-VERIFY PASSTHROUGH [final_entropy_output] ===
Buffer: d_output
Type: float
Elements: 10240000
Statistics: min=2.1464657783508301, max=3.9238560199737549, mean=3.4919920967611251
Sample_first10: 2.503258228302002 2.9182956218719482 3.3735573291778564 3.3735573291778564 3.1898980140686035 3.2729055881500244 3.0062389373779297 2.6892464160919189 2.8225798606872559 2.8729057312011719
=== END LASSI-VERIFY PASSTHROUGH [final_entropy_output] ===
"""

_META_BLOCK_NO_PROBE = """
=== BEGIN LASSI-VERIFY METAMORPHIC [final_entropy_output] ===
Relation: single_pixel_sensitivity__p_outlier
Perturbation_under: single_pixel_sensitivity__p_outlier
Buffer: d_output
Elements: 10240000
Statistics: min=2.1464657783508301, max=3.9238560199737549, mean=3.4919920967611251
Sample_first10: 2.503258228302002 2.9182956218719482 3.3735573291778564 3.3735573291778564 3.1898980140686035 3.2729055881500244 3.0062389373779297 2.6892464160919189 2.8225798606872559 2.8729057312011719
=== END LASSI-VERIFY METAMORPHIC [final_entropy_output] ===
"""

_META_BLOCK_WITH_PROBE = _META_BLOCK_NO_PROBE.replace(
    "=== END LASSI-VERIFY METAMORPHIC [final_entropy_output] ===",
    "Sample_at_target_before: 3.19 3.27 3.01 2.69 2.82\n"
    "Sample_at_target_after: 3.19 3.27 3.55 2.69 2.82\n"
    "=== END LASSI-VERIFY METAMORPHIC [final_entropy_output] ==="
)


def validate_piece2_synthetic():
    print("\n=== Piece 2: local-response probe parsing/folding (synthetic fixtures) ===")
    relation = {
        "name": "single_pixel_sensitivity__p_outlier",
        "type": "SENSITIVITY",
        "tolerance": 1e-5,
    }

    verdict_no_probe, diag_no_probe = check_metamorphic_relation(
        _META_BLOCK_NO_PROBE, _PASS_BLOCK_TEMPLATE, relation, 1e-5
    )
    print(f"  (a) No probe fields (today's behavior): verdict={verdict_no_probe} "
          f"(expect FAIL — globally-invisible local perturbation, unchanged from today)")

    verdict_with_probe, diag_with_probe = check_metamorphic_relation(
        _META_BLOCK_WITH_PROBE, _PASS_BLOCK_TEMPLATE, relation, 1e-5
    )
    print(f"  (b) With Sample_at_target_before/after (one differing value): "
          f"verdict={verdict_with_probe} (expect PASS — local response now visible)")
    if diag_with_probe.get("summary"):
        print(f"      {diag_with_probe['summary']}")

    ok = (verdict_no_probe == "FAIL" and verdict_with_probe == "PASS")
    print(f"\n  Result: {'PASS' if ok else 'FAIL'} — probe fields correctly flip the verdict "
          f"without affecting probe-absent (backward-compatible) scoring.")


# --- Piece 2b: multi-block canonical capture/enforcement -------------------------------------
# Regression test for the bug found live in KERNEL_EXP-SC26_ENTROPY_1__07-19-2026_t11_43_40
# Trial 1: the local-response probe's MT_TARGET_INDEX (early, index-only) and MT_TRANSFORM
# (later, perturbation-write-only) blocks share one key. The original single-block-per-key
# capture/enforcement (`_extract_mt_transform_block` returns on first match;
# `_enforce_canonical_transform` treats any other same-key MT_TRANSFORM occurrence as a
# duplicate to strip) silently dropped the perturbation-write block whenever the probe's two
# sites both used the MT_TRANSFORM delimiter. Fixed by promoting MT_TARGET_INDEX to its own
# top-level delimiter with its own extract/enforce pair, so each delimiter type keeps exactly
# one occurrence per key again. This reproduces that exact two-block structure and confirms both
# blocks now survive capture and enforcement, in the correct relative order.

_MULTIBLOCK_KEY = "single_pixel_perturbation_sensitivity__p_small::p_small"

_MULTIBLOCK_SOURCE = f"""
  input = init();

// >>> BEGIN LASSI-VERIFY MT_TARGET_INDEX [{_MULTIBLOCK_KEY}] >>>
    mt_target_idx = 5;
// <<< END LASSI-VERIFY MT_TARGET_INDEX <<<

  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
  run_kernel();
  cudaMemcpy(output_before, d_output, bytes, cudaMemcpyDeviceToHost);

// >>> BEGIN LASSI-VERIFY MT_TRANSFORM [{_MULTIBLOCK_KEY}] >>>
    input[mt_target_idx] = (input[mt_target_idx] + 1) & 0x0F;
// <<< END LASSI-VERIFY MT_TRANSFORM <<<

  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
  run_kernel();
  cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
"""

_MULTIBLOCK_VARIANT = f"""
  input = init();

// >>> BEGIN LASSI-VERIFY MT_TARGET_INDEX [{_MULTIBLOCK_KEY}] >>>
    mt_target_idx = 0;   // LLM-authored placeholder — must be overwritten by enforcement
// <<< END LASSI-VERIFY MT_TARGET_INDEX <<<

  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
  run_kernel_variant();
  cudaMemcpy(output_before, d_output, bytes, cudaMemcpyDeviceToHost);

// >>> BEGIN LASSI-VERIFY MT_TRANSFORM [{_MULTIBLOCK_KEY}] >>>
    // LLM-authored placeholder — must be overwritten by enforcement
// <<< END LASSI-VERIFY MT_TRANSFORM <<<

  cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
  run_kernel_variant();
  cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);
"""


def validate_piece2_multiblock_capture():
    print("\n=== Piece 2b: MT_TARGET_INDEX / MT_TRANSFORM multi-block capture+enforcement "
          "(regression test for the Trial-1 bug) ===")
    ti = _extract_mt_target_index_block(_MULTIBLOCK_SOURCE, _MULTIBLOCK_KEY)
    tf = _extract_mt_transform_block(_MULTIBLOCK_SOURCE, _MULTIBLOCK_KEY)
    capture_ok = (
        ti is not None and tf is not None
        and "mt_target_idx = 5" in ti and "input[mt_target_idx]" not in ti
        and "input[mt_target_idx] = (input[mt_target_idx] + 1)" in tf
        and "mt_target_idx = 5" not in tf
    )
    print(f"  Capture: {'PASS' if capture_ok else 'FAIL'} — index and transform captured "
          f"independently, no cross-contamination.")

    enforced, ti_path = _enforce_canonical_target_index(_MULTIBLOCK_VARIANT, _MULTIBLOCK_KEY, ti)
    enforced, tf_path = _enforce_canonical_transform(enforced, _MULTIBLOCK_KEY, tf)
    enforce_ok = (
        enforced.count("BEGIN LASSI-VERIFY MT_TARGET_INDEX") == 1
        and enforced.count("BEGIN LASSI-VERIFY MT_TRANSFORM") == 1
        and "mt_target_idx = 5" in enforced
        and "input[mt_target_idx] = (input[mt_target_idx] + 1)" in enforced
        and enforced.index("MT_TARGET_INDEX [") < enforced.index("MT_TRANSFORM [")
    )
    print(f"  Enforcement ({ti_path}/{tf_path}): {'PASS' if enforce_ok else 'FAIL'} — both "
          f"blocks survive with canonical content, correct order, no duplicate/stripped block.")

    ok = capture_ok and enforce_ok
    print(f"\n  Result: {'PASS' if ok else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", default=DEFAULT_EXP_DIR)
    args = parser.parse_args()

    validate_piece1(args.exp_dir)
    validate_piece2_synthetic()
    validate_piece2_multiblock_capture()


if __name__ == "__main__":
    main()
