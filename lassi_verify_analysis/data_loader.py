"""
Data loader for LASSI-Verify analysis.

Loads and normalizes the two CSV files produced by the pipeline:
  - lassi-verify_pipeline_data.csv  (Table 1: one row per kernel × trial)
  - lassi-verify_attempt_log.csv    (Table 2: one row per code attempt)

Key design decisions:
- Expects pre-aggregated flat CSVs (user concatenates per-trial CSVs before analysis).
- Derives `kernel_name` from `app_name` by stripping the '-cuda_main...' suffix.
  e.g., "jacobi-cuda_main_ee_refactorcode_CUDA_exp..." → "jacobi"
  e.g., "lid-driven-cavity-cuda_main..." → "lid-driven-cavity"
- All downstream grouping uses `kernel_name` (not `app_name`).
- Missing optional columns trigger a warning, not a crash (graceful degradation).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    BOOL_COLUMNS_ATTEMPTS,
    FLOAT_COLUMNS_ATTEMPTS,
    FLOAT_COLUMNS_TRIALS,
    INT_COLUMNS_ATTEMPTS,
    INT_COLUMNS_TRIALS,
    LIST_COLUMNS_ATTEMPTS,
    LIST_COLUMNS_TRIALS,
    VERDICT_COLUMNS_ATTEMPTS,
    VERDICT_COLUMNS_TRIALS,
)

log = logging.getLogger(__name__)

# Required columns for Metric 1 (loader warns if any are absent).
_REQUIRED_TRIALS_M1 = [
    "app_name",
    "trial_number",
    "lassiee_valid",
    "lassiee_version",
    "lassiee_clean_runtime_diff",
    "lassiee_clean_status",
    "lassiee_runtime_diff",
    "base_v1_valid",
]
_OPTIONAL_TRIALS_M1 = ["base_llm_verified_valid"]  # absent → Variant 4 skipped
_REQUIRED_ATTEMPTS_M1 = ["app_name", "trial_number", "is_best_code", "judge_verdict"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize Table 1 and Table 2 from *data_dir*.

    Args:
        data_dir: Directory containing ``lassi-verify_pipeline_data.csv``
                  and optionally ``lassi-verify_attempt_log.csv``.

    Returns:
        ``(df_trials, df_attempts)`` — normalized DataFrames.
        ``df_attempts`` is an empty DataFrame with the expected columns if
        the attempt log file is missing.

    Warns (via logging) on missing optional columns or missing Table 2 file.
    """
    # Support both bare filenames and the result_-prefixed names the pipeline writes.
    def _resolve(name: str) -> str:
        bare = os.path.join(data_dir, name)
        prefixed = os.path.join(data_dir, "result_" + name)
        if os.path.isfile(bare):
            return bare
        if os.path.isfile(prefixed):
            return prefixed
        return bare  # let the caller raise FileNotFoundError on the bare path

    trials_path = _resolve("lassi-verify_pipeline_data.csv")
    attempts_path = _resolve("lassi-verify_attempt_log.csv")

    if not os.path.isfile(trials_path):
        raise FileNotFoundError(f"Table 1 not found: {trials_path}")

    # index_col=False prevents pandas from auto-detecting extra data fields as
    # index columns when the data has more columns than the header (e.g. when
    # new columns were added to data rows before the header was updated).
    # Extra trailing data fields are silently dropped; a ParserWarning is emitted
    # but suppressed here — the user is responsible for keeping header in sync.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.ParserWarning)
        df_trials = pd.read_csv(trials_path, dtype=str, index_col=False)
    log.info("Loaded Table 1: %d rows from %s", len(df_trials), trials_path)

    if os.path.isfile(attempts_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.ParserWarning)
            df_attempts = pd.read_csv(attempts_path, dtype=str, index_col=False)
        log.info("Loaded Table 2: %d rows from %s", len(df_attempts), attempts_path)
    else:
        log.warning(
            "Table 2 not found at %s. Variant 2 (judge-pass@k LASSI) will be skipped.",
            attempts_path,
        )
        df_attempts = pd.DataFrame()

    df_trials = _normalize_trials(df_trials)
    if not df_attempts.empty:
        df_attempts = _normalize_attempts(df_attempts)

    _check_required_columns(df_trials, _REQUIRED_TRIALS_M1, "Table 1 (Metric 1 required)")
    for col in _OPTIONAL_TRIALS_M1:
        if col not in df_trials.columns:
            log.warning(
                "Optional column '%s' missing from Table 1. Variant 4 (verified-pass zero-shot) will be skipped.",
                col,
            )
    if not df_attempts.empty:
        _check_required_columns(df_attempts, _REQUIRED_ATTEMPTS_M1, "Table 2 (Metric 1 Variant 2)")

    return df_trials, df_attempts


def get_kernels(df_trials: pd.DataFrame) -> list[str]:
    """Return sorted unique ``kernel_name`` values from Table 1.

    ``kernel_name`` is the grouping key for all metrics — NOT ``app_name``.
    """
    if "kernel_name" not in df_trials.columns:
        raise KeyError("kernel_name column not found. Was load_data() called?")
    return sorted(df_trials["kernel_name"].dropna().unique().tolist())


def get_trials_for_kernel(df_trials: pd.DataFrame, kernel: str) -> pd.DataFrame:
    """Return all Table 1 rows for *kernel* (matched on ``kernel_name``)."""
    return df_trials[df_trials["kernel_name"] == kernel].copy()


# ---------------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------------

def _derive_kernel_name(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``kernel_name`` column derived from ``app_name``.

    Strips everything from ``-cuda_main`` onward (case-sensitive).
    Falls back to the full ``app_name`` if the pattern is not found.

    Examples:
        "jacobi-cuda_main_ee_refactorcode_CUDA_exp..."  →  "jacobi"
        "lid-driven-cavity-cuda_main..."               →  "lid-driven-cavity"
        "jacobi"                                        →  "jacobi"   (no suffix → unchanged)
    """
    if "app_name" not in df.columns:
        log.warning("app_name column not found; kernel_name will not be derived.")
        return df
    df = df.copy()
    df["kernel_name"] = df["app_name"].str.replace(
        r"-cuda_main.*", "", regex=True
    ).str.strip()
    return df


def _normalize_verdict_strings(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Strip whitespace and uppercase the listed verdict/status columns (in-place copy)."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            # Pandas reads empty/NaN cells as the string "NAN" after astype(str).
            # Map these back to pd.NA so downstream checks work correctly.
            df[col] = df[col].replace({"NAN": pd.NA, "": pd.NA})
    return df


def _parse_list_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Parse comma-separated string columns into Python lists of stripped strings.

    Empty strings and NaN → empty list ``[]``.
    """
    for col in columns:
        if col not in df.columns:
            continue
        def _parse(val: object) -> list[str]:
            if pd.isna(val) or str(val).strip() in ("", "nan", "NAN"):
                return []
            return [s.strip() for s in str(val).split(",") if s.strip()]
        df[col] = df[col].apply(_parse)
    return df


def _parse_bool_column(series: pd.Series, nullable: bool = False) -> pd.Series:
    """Convert a string/mixed boolean column to Python bool (or None if nullable).

    Handles: True/False/true/false/1/0/"NA"/"nan"/"NAN"/NaN.
    - nullable=False  → NA-like values → False
    - nullable=True   → NA-like values → None
    """
    TRUE_STRINGS = {"true", "1", "yes"}
    FALSE_STRINGS = {"false", "0", "no"}
    NA_STRINGS = {"na", "nan", "none", "", "nat"}

    def _convert(val: object) -> object:
        if pd.isna(val):
            return None if nullable else False
        s = str(val).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        if s in NA_STRINGS:
            return None if nullable else False
        log.debug("Unexpected boolean value: %r — treating as False/None", val)
        return None if nullable else False

    return series.apply(_convert)


def _cast_float_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast listed columns to float (coerce errors to NaN)."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _cast_int_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast listed columns to pandas nullable Int64 (coerce errors to pd.NA)."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _cast_lassiee_version(df: pd.DataFrame) -> pd.DataFrame:
    """Cast ``lassiee_version`` to int; NaN/empty → -1 (sentinel: no best LASSI code)."""
    if "lassiee_version" not in df.columns:
        return df
    df["lassiee_version"] = (
        pd.to_numeric(df["lassiee_version"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    return df


def _normalize_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalization steps to Table 1 (df_trials)."""
    df = df.copy()
    df = _derive_kernel_name(df)
    df = _normalize_verdict_strings(df, VERDICT_COLUMNS_TRIALS)
    df = _parse_list_columns(df, LIST_COLUMNS_TRIALS)
    df = _cast_float_columns(df, FLOAT_COLUMNS_TRIALS)
    df = _cast_int_columns(df, INT_COLUMNS_TRIALS)
    df = _cast_lassiee_version(df)

    # base_llm_verified_valid: tri-state — True / False / None (if verification failed)
    if "base_llm_verified_valid" in df.columns:
        df["base_llm_verified_valid"] = _parse_bool_column(
            df["base_llm_verified_valid"], nullable=True
        )

    return df


def _normalize_attempts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalization steps to Table 2 (df_attempts)."""
    df = df.copy()
    df = _derive_kernel_name(df)
    df = _normalize_verdict_strings(df, VERDICT_COLUMNS_ATTEMPTS)
    df = _parse_list_columns(df, LIST_COLUMNS_ATTEMPTS)
    df = _cast_float_columns(df, FLOAT_COLUMNS_ATTEMPTS)
    df = _cast_int_columns(df, INT_COLUMNS_ATTEMPTS)

    for col in BOOL_COLUMNS_ATTEMPTS:
        if col in df.columns:
            df[col] = _parse_bool_column(df[col], nullable=False)

    return df


def _check_required_columns(
    df: pd.DataFrame, required: list[str], context: str
) -> None:
    """Warn (not raise) for each required column that is missing."""
    missing = [c for c in required if c not in df.columns]
    for col in missing:
        log.warning("Required column '%s' missing from %s.", col, context)
