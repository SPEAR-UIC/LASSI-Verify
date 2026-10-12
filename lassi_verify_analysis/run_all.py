"""
CLI runner for LASSI-Verify analysis.

Usage:
    python -m lassi_verify_analysis.run_all --data-dir ./data

    # Optionally override the output location (default: <data-dir>/analysis_results/):
    python -m lassi_verify_analysis.run_all --data-dir ./data --output-dir ./results

Loads the two pre-aggregated CSV files from --data-dir and runs all
implemented metrics, writing outputs (tables, figures) to --output-dir.

Additional metric modules will be added here as they are implemented.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .data_loader import get_kernels, load_data
from . import basellm_comparison
from . import judge_baseline
from . import pass_at_k
from . import verification_analysis
from . import verification_evidence


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lassi_verify_analysis",
        description="Compute metrics from LASSI-Verify experimental data.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Directory containing pre-aggregated CSVs: "
            "lassi-verify_pipeline_data.csv (required) and "
            "lassi-verify_attempt_log.csv (optional)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory where tables and figures will be written. "
            "Defaults to <data-dir>/analysis_results/."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.data_dir, "analysis_results")

    _configure_logging(args.verbose)
    log = logging.getLogger(__name__)

    # --- Load data ---
    log.info("Loading data from: %s", args.data_dir)
    try:
        df_trials, df_attempts = load_data(args.data_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    kernels = get_kernels(df_trials)
    log.info(
        "Loaded %d trials across %d kernel(s): %s",
        len(df_trials), len(kernels), kernels,
    )

    # --- Metric 1: verified-pass@k ---
    log.info("Computing Metric 1: verified-pass@k ...")
    m1_results = pass_at_k.compute(df_trials, df_attempts)
    pass_at_k.report(m1_results, output_dir=output_dir)
    log.info("Metric 1 complete.")

    # --- Verification Effectiveness Analysis (sub-metrics A–E) ---
    log.info("Computing Verification Effectiveness Analysis (sub-metrics A–E) ...")
    va_results = verification_analysis.compute(df_trials, df_attempts)
    verification_analysis.report(va_results, output_dir=output_dir)
    log.info("Verification analysis complete.")

    # --- Verification Evidence Summary ---
    log.info("Generating Verification Evidence Summary ...")
    ve_results = verification_evidence.compute(df_trials, df_attempts)
    verification_evidence.report(ve_results, output_dir=output_dir)
    log.info("Verification evidence complete.")

    # --- Judge-baseline vs. framework agreement (per-attempt) ---
    # Reads judge_discrepancies arrays from each per-trial verification_log JSON
    # sidecar and exports a flat per-attempt CSV. Uses data_dir directly because
    # sidecars are JSON, not CSV (data_loader handles only the two CSV tables).
    log.info("Loading per-attempt judge discrepancies from JSON sidecars ...")
    df_disc = judge_baseline.load_discrepancies(args.data_dir)
    if not df_disc.empty:
        judge_baseline.report(df_disc, output_dir=output_dir)
    log.info("Judge-baseline export complete.")

    # --- basellm vs LASSI runtime comparison (apples-to-apples) ---
    log.info("Computing basellm vs LASSI runtime comparison ...")
    bc_results = basellm_comparison.compute(df_trials)
    basellm_comparison.report(bc_results, output_dir=output_dir)
    log.info("basellm comparison complete.")

    log.info("All metrics complete. Outputs written to: %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
