"""
basellm vs LASSI runtime comparison (uninstrumented vs uninstrumented).

The LASSI-Verify pipeline records runtime measurements under several
instrumentation regimes. The only fair fairness-comparison for the paper is
uninstrumented vs uninstrumented — comparing instrumented basellm against
instrumented LASSI mixes a calculated quantity ("speedup with framework
overhead included") that does not represent the code that would actually ship.

Relevant fields:

  zeroshot_runtime_s          uninstrumented   (basellm code as the vanilla LLM
                                                generated it from original source)
  lassiee_clean_runtime_s     uninstrumented   (best LASSI code with all
                                                LASSI-VERIFY blocks stripped,
                                                recompiled + re-executed)
  basellm_zs_verify_runtime_s instrumented     (same basellm code re-instrumented
                                                with the combined T1+T2 template —
                                                kept as raw diagnostic data, NOT
                                                used for a calculated ratio here)
  lassiee_runtime_s           instrumented     (best LASSI code as the iteration
                                                loop measured it — kept as raw
                                                diagnostic data, NOT used here)

This module computes ONE comparison:

  Uninstr/uninstr ("what you would ship"):
      zeroshot_runtime_s  ↔  lassiee_clean_runtime_s

Per-trial: raw pair + ratio. Per-kernel: median + IQR of the ratio across
trials where both members of the pair are available.

A speedup ratio > 1.0 means LASSI is faster than basellm.
"""

from __future__ import annotations

import logging
import os
from statistics import median

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Per-trial CSV column order (written to lassi_basellm_runtime_comparison.csv).
# Instrumented runtimes are retained as raw diagnostic data — included for
# transparency but never combined into a calculated ratio here.
COMPARISON_COLUMNS = [
    "kernel_name",
    "trial_number",
    "lassiee_valid",
    "basellm_zs_verify_status",
    "lassiee_clean_status",
    "zeroshot_runtime_s",
    "lassiee_clean_runtime_s",
    "speedup_uninstr",              # zeroshot_runtime_s / lassiee_clean_runtime_s
    "basellm_zs_verify_runtime_s",  # raw diagnostic only
    "lassiee_runtime_s",            # raw diagnostic only
]


def compute(df_trials: pd.DataFrame) -> dict:
    """Build per-trial and per-kernel apples-to-apples comparison tables.

    Args:
        df_trials: Table 1 DataFrame from data_loader.load_data.

    Returns:
        Dict with keys:
          - "per_trial": DataFrame, one row per trial, columns per
            :data:`COMPARISON_COLUMNS`. Trials missing either side of the
            comparison have NaN in the speedup_uninstr column.
          - "per_kernel": dict {kernel_name: {"uninstr": {...stats}}} where
            each stats dict has count, median, q25, q75, min, max. Kernels
            with no usable trials are omitted.
          - "overall": same shape as a per_kernel entry, aggregated across kernels.
    """
    if df_trials.empty:
        log.warning("df_trials is empty — no comparison to compute.")
        return {"per_trial": pd.DataFrame(columns=COMPARISON_COLUMNS),
                "per_kernel": {}, "overall": {}}

    rows = []
    for _, r in df_trials.iterrows():
        zs_u   = _f(r.get("zeroshot_runtime_s"))
        lass_c = _f(r.get("lassiee_clean_runtime_s"))
        bvf    = _f(r.get("basellm_zs_verify_runtime_s"))   # raw diagnostic
        lass_i = _f(r.get("lassiee_runtime_s"))             # raw diagnostic

        speedup_u = (zs_u / lass_c) if (zs_u and lass_c and lass_c > 0) else np.nan

        rows.append({
            "kernel_name":                 r.get("kernel_name"),
            "trial_number":                r.get("trial_number"),
            "lassiee_valid":               r.get("lassiee_valid"),
            "basellm_zs_verify_status":    r.get("basellm_zs_verify_status"),
            "lassiee_clean_status":        r.get("lassiee_clean_status"),
            "zeroshot_runtime_s":          zs_u,
            "lassiee_clean_runtime_s":     lass_c,
            "speedup_uninstr":             speedup_u,
            "basellm_zs_verify_runtime_s": bvf,
            "lassiee_runtime_s":           lass_i,
        })

    per_trial = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)

    per_kernel: dict = {}
    for k in sorted(per_trial["kernel_name"].dropna().unique().tolist()):
        sub = per_trial[per_trial["kernel_name"] == k]
        per_kernel[k] = {
            "uninstr": _stats(sub["speedup_uninstr"]),
        }

    overall = {
        "uninstr": _stats(per_trial["speedup_uninstr"]),
    }

    return {"per_trial": per_trial, "per_kernel": per_kernel, "overall": overall}


def report(results: dict, output_dir: str,
           filename: str = "lassi_basellm_runtime_comparison.csv") -> str:
    """Write per-trial CSV and print the per-kernel summary table.

    Returns the absolute path of the CSV file (empty string if nothing written).
    """
    per_trial = results.get("per_trial")
    if per_trial is None or per_trial.empty:
        log.warning("No comparison rows to write.")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(output_dir, filename))
    per_trial.to_csv(out_path, index=False, float_format="%.4f")
    log.info("Wrote %d per-trial comparison rows to %s", len(per_trial), out_path)

    print("\n=== basellm vs LASSI runtime comparison (uninstr/uninstr) ===")
    print(f"Wrote: {out_path} ({len(per_trial)} rows)")
    print(f"\nA speedup ratio > 1.0 means LASSI is faster than basellm.\n")

    for scope_label, stats_dict in [("OVERALL", results.get("overall", {}))] + \
                                    [(k, v) for k, v in results.get("per_kernel", {}).items()]:
        print(f"  {scope_label}:")
        s = stats_dict.get("uninstr", {})
        n = s.get("count", 0)
        label = "uninstr/uninstr (what you would ship)"
        if n == 0:
            print(f"    {label}: n=0 (no usable trials)")
        else:
            med = s.get("median")
            q25, q75 = s.get("q25"), s.get("q75")
            mn, mx = s.get("min"), s.get("max")
            print(f"    {label}: n={n}  median={med:.2f}x  "
                  f"IQR=[{q25:.2f}x, {q75:.2f}x]  range=[{mn:.2f}x, {mx:.2f}x]")
        print()

    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v) -> float | None:
    """Coerce to float; return None on NaN, None, or non-numeric."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if not np.isnan(f) else None


def _stats(series: pd.Series) -> dict:
    """Drop NaN and return {count, median, q25, q75, min, max}."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"count": 0}
    return {
        "count":  n,
        "median": float(s.median()),
        "q25":    float(s.quantile(0.25)),
        "q75":    float(s.quantile(0.75)),
        "min":    float(s.min()),
        "max":    float(s.max()),
    }
