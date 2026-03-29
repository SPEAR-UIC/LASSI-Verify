"""
CLI runner for LASSI-Verify analysis.

Usage:
    python -m lassi_verify_analysis.run_all \\
        --data-dir ./data \\
        --output-dir ./results

Loads the two pre-aggregated CSV files from --data-dir and runs all
implemented metrics, writing outputs (tables, figures) to --output-dir.

Additional metric modules will be added here as they are implemented.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .data_loader import get_kernels, load_data
from . import pass_at_k


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
        required=True,
        help="Directory where tables and figures will be written.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

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
    pass_at_k.report(m1_results, output_dir=args.output_dir)
    log.info("Metric 1 complete.")

    # --- Future metrics will be added here ---
    # log.info("Computing Metric 5: judge accuracy ...")
    # from . import judge_accuracy
    # m5_results = judge_accuracy.compute(df_trials, df_attempts)
    # judge_accuracy.report(m5_results, output_dir=args.output_dir)

    log.info("All metrics complete. Outputs written to: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
