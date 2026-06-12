"""
basellm vs LASSI runtime comparison (apples-to-apples).

The LASSI-Verify pipeline records four runtime measurements per trial, each
under a different instrumentation regime (see planning/LASSI_Verify_Analysis_v2_plan.md
§3.1). The naming is asymmetric because each field was motivated by a different
concern:

  zeroshot_runtime_s        uninstrumented   (basellm code as the vanilla LLM
                                              generated it from original source)
  basellm_verify_runtime_s  instrumented     (same basellm code, re-instrumented
                                              with the SOURCE_verify T1+T2 template,
                                              then recompiled + re-executed)
  lassiee_runtime_s         instrumented     (best LASSI code as the iteration
                                              loop measured it; LASSI codes carry
                                              T1+T2 instrumentation natively)
  lassiee_clean_runtime_s   uninstrumented   (best LASSI code with all LASSI-VERIFY
                                              blocks stripped, recompiled + re-executed)

This module computes the TWO genuine apples-to-apples comparisons (NOT the
mixed uninstr/instr pairing that biases the result in LASSI's favor):

  Uninstr/uninstr ("what you would ship"):
      zeroshot_runtime_s  ↔  lassiee_clean_runtime_s
  Instr/instr ("same instrumentation footprint"):
      basellm_verify_runtime_s  ↔  lassiee_runtime_s

Per-trial: raw pair + ratio for each comparison. Per-kernel: median + IQR of
each ratio across trials where both members of the pair are available.

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
COMPARISON_COLUMNS = [
    "kernel_name",
    "trial_number",
    "lassiee_valid",
    "basellm_verify_status",
    "lassiee_clean_status",
    "zeroshot_runtime_s",
    "lassiee_clean_runtime_s",
    "speedup_uninstr",          # zeroshot_runtime_s / lassiee_clean_runtime_s
    "basellm_verify_runtime_s",
    "lassiee_runtime_s",
    "speedup_instr",            # basellm_verify_runtime_s / lassiee_runtime_s
]


def compute(df_trials: pd.DataFrame) -> dict:
    """Build per-trial and per-kernel apples-to-apples comparison tables.

    Args:
        df_trials: Table 1 DataFrame from data_loader.load_data.

    Returns:
        Dict with keys:
          - "per_trial": DataFrame, one row per trial, columns per
            :data:`COMPARISON_COLUMNS`. Trials missing either side of a
            comparison have NaN in the corresponding ratio.
          - "per_kernel": dict {kernel_name: {"uninstr": {...stats},
            "instr": {...stats}}} where each stats dict has count, median,
            q25, q75, min, max. Kernels with no usable trials are omitted.
          - "overall": same shape as per_kernel entry, aggregated across kernels.
    """
    if df_trials.empty:
        log.warning("df_trials is empty — no comparison to compute.")
        return {"per_trial": pd.DataFrame(columns=COMPARISON_COLUMNS),
                "per_kernel": {}, "overall": {}}

    rows = []
    for _, r in df_trials.iterrows():
        zs_u   = _f(r.get("zeroshot_runtime_s"))
        lass_c = _f(r.get("lassiee_clean_runtime_s"))
        bvf    = _f(r.get("basellm_verify_runtime_s"))
        lass_i = _f(r.get("lassiee_runtime_s"))

        speedup_u = (zs_u / lass_c) if (zs_u and lass_c and lass_c > 0) else np.nan
        speedup_i = (bvf / lass_i) if (bvf and lass_i and lass_i > 0) else np.nan

        rows.append({
            "kernel_name":              r.get("kernel_name"),
            "trial_number":             r.get("trial_number"),
            "lassiee_valid":            r.get("lassiee_valid"),
            "basellm_verify_status":    r.get("basellm_verify_status"),
            "lassiee_clean_status":     r.get("lassiee_clean_status"),
            "zeroshot_runtime_s":       zs_u,
            "lassiee_clean_runtime_s":  lass_c,
            "speedup_uninstr":          speedup_u,
            "basellm_verify_runtime_s": bvf,
            "lassiee_runtime_s":        lass_i,
            "speedup_instr":            speedup_i,
        })

    per_trial = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)

    per_kernel: dict = {}
    for k in sorted(per_trial["kernel_name"].dropna().unique().tolist()):
        sub = per_trial[per_trial["kernel_name"] == k]
        per_kernel[k] = {
            "uninstr": _stats(sub["speedup_uninstr"]),
            "instr":   _stats(sub["speedup_instr"]),
        }

    overall = {
        "uninstr": _stats(per_trial["speedup_uninstr"]),
        "instr":   _stats(per_trial["speedup_instr"]),
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

    print("\n=== basellm vs LASSI runtime comparison (apples-to-apples) ===")
    print(f"Wrote: {out_path} ({len(per_trial)} rows)")
    print(f"\nA speedup ratio > 1.0 means LASSI is faster than basellm. The two")
    print(f"comparisons should agree closely; large divergence indicates that")
    print(f"instrumentation overhead is distorting the headline result.\n")

    for scope_label, stats_dict in [("OVERALL", results.get("overall", {}))] + \
                                    [(k, v) for k, v in results.get("per_kernel", {}).items()]:
        print(f"  {scope_label}:")
        for tag, label in [("uninstr", "uninstr/uninstr (what you would ship)"),
                           ("instr",   "instr/instr   (same instr footprint) ")]:
            s = stats_dict.get(tag, {})
            n = s.get("count", 0)
            if n == 0:
                print(f"    {label}: n=0 (no usable trials)")
                continue
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
