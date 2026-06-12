"""
Metric 1: verified-pass@k and comparison variants.

Implements the Chen et al. (2021) combinatorial pass@k estimator for four
variants that together support RQ1 (three-tier vs. judge-only correctness):

  V1. verified-pass@k (LASSI)   — headline metric
  V2. judge-pass@k (LASSI)      — judge-only verdict on same best codes
  V3. judge-pass@k (zero-shot)  — judge verdict on vanilla LLM output
  V4. verified-pass@k (zero-shot) — three-tier verdict on vanilla LLM output

Formula:  pass@k = E[1 - C(n-c, k) / C(n, k)]
where n = trials per kernel, c = trials that "pass."

Bootstrap (B=5000 resamples) provides 95% CIs.
Monte Carlo is NOT used here — the combinatorial formula gives exact point
estimates; Monte Carlo is only needed for Metric 2 (runtime-reduction@k).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import BOOTSTRAP_B, K_VALUES, RANDOM_SEED
from .data_loader import get_kernels, get_trials_for_kernel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core combinatorial formula
# ---------------------------------------------------------------------------

def combinatorial_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k = 1 - C(n-c, k) / C(n, k).

    Uses math.comb (Python 3.8+). All edge cases are handled:
    - c == 0 → 0.0 (no passes, no chance of success)
    - c >= n → 1.0 (all trials pass)
    - n - c < k → 1.0 (C(n-c, k) == 0; every selection of k contains a pass)
    - k > n → float(c > 0) (degenerate; shouldn't occur with n=30, k≤5)
    - k == 0 → 0.0 (selecting 0 candidates always fails by convention)

    Args:
        n: Total number of trials.
        c: Number of trials that passed.
        k: Number of candidates to draw.

    Returns:
        Estimated probability that at least one of k draws is a passing trial.
    """
    if k <= 0:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    if k > n:
        return float(c > 0)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_pass_at_k(
    pass_vector: np.ndarray,
    k: int,
    B: int = BOOTSTRAP_B,
    seed: int = RANDOM_SEED,
) -> dict:
    """Compute pass@k point estimate and bootstrap 95% CI.

    Args:
        pass_vector: Boolean array of length n (one element per trial;
                     True = trial passes the pass criterion for this variant).
        k:           Number of candidates to draw.
        B:           Number of bootstrap resamples.
        seed:        Random seed for reproducibility.

    Returns:
        Dict with keys: "point", "ci_low", "ci_high", "n", "c".
        "point" is the exact combinatorial estimate (no bootstrapping applied
        to the point estimate itself — bootstrap is used only for the CI).
    """
    rng = np.random.RandomState(seed)
    n = len(pass_vector)
    c = int(np.sum(pass_vector))
    point = combinatorial_pass_at_k(n, c, k)

    boot_estimates: list[float] = []
    for _ in range(B):
        resampled = rng.choice(pass_vector, size=n, replace=True)
        c_boot = int(np.sum(resampled))
        boot_estimates.append(combinatorial_pass_at_k(n, c_boot, k))

    ci_low = float(np.percentile(boot_estimates, 2.5))
    ci_high = float(np.percentile(boot_estimates, 97.5))

    return {"point": point, "ci_low": ci_low, "ci_high": ci_high, "n": n, "c": c}


# ---------------------------------------------------------------------------
# Pass criteria: four variants
# ---------------------------------------------------------------------------

def compute_verified_lassi_pass_vector(df_kernel: pd.DataFrame) -> np.ndarray:
    """Variant 1: verified-pass@k (LASSI) — the headline metric.

    A trial passes iff ALL of:
    1. lassiee_valid == "VALID"  (compiled + executed + all three tiers passed)
    2. lassiee_version >= 3 (not a SOURCE_verify fallback; 3+ = LASSI-generated codes)

    Runtime improvement is intentionally excluded: pass@k measures functional
    correctness only.  Performance (runtime-reduction@k) is a separate metric.

    Returns:
        Boolean numpy array of shape (n_trials,).
    """
    valid = df_kernel["lassiee_valid"].eq("VALID")
    version_ok = df_kernel["lassiee_version"].ge(3)
    return (valid & version_ok).values


def compute_judge_lassi_pass_vector(
    df_kernel: pd.DataFrame,
    best_code_merged: pd.DataFrame,
    kernel: str,
) -> np.ndarray:
    """Variant 2: judge-pass@k (LASSI) — judge baseline verdict on same best codes.

    Uses Table 2 is_best_code==True rows (pre-joined with Table 1 runtime cols).
    A trial passes iff:
    1. There is an is_best_code==True row in Table 2 for this trial
    2. That row's judge_baseline_verdict == "VALID"

    Under the judge-as-baseline design, judge_baseline_verdict is the raw LLM
    judge output captured BEFORE Tier 1/2/3 ever run; it is never modified by
    tier results. V2 thus measures what a vanilla LLM-only practitioner (no
    three-tier framework) would have approved on the same best-code set, in
    direct contrast to V1 (verified-pass@k LASSI, which uses the framework
    verdict). V1 and V2 are independent measurements on identical codes.

    Runtime improvement is intentionally excluded: pass@k measures functional
    correctness only.  Performance is a separate metric.

    Args:
        df_kernel: Table 1 rows for this kernel.
        best_code_merged: Pre-joined Table 2 best-code rows with runtime cols
                          from Table 1. Must contain kernel_name column.
        kernel: Kernel name string (used to filter best_code_merged).

    Returns:
        Boolean numpy array of shape (n_trials,). If best_code_merged is empty,
        returns all-False array with a warning.
    """
    if best_code_merged.empty:
        log.warning(
            "Kernel '%s': Table 2 is empty — Variant 2 returns all-False.", kernel
        )
        return np.zeros(len(df_kernel), dtype=bool)

    bc_kernel = best_code_merged[best_code_merged["kernel_name"] == kernel]
    if bc_kernel.empty:
        log.warning(
            "Kernel '%s': no is_best_code==True rows in Table 2 — Variant 2 returns all-False.",
            kernel,
        )
        return np.zeros(len(df_kernel), dtype=bool)

    # Build (app_name, trial_number) → pass mapping from Table 2 best-code rows.
    # Using the compound key handles both scenarios:
    #   - Dev (per-trial directories concatenated): app_name differs per trial
    #     (includes timestamp), trial_number is always 1.
    #   - Production (single aggregated CSV): app_name is consistent per kernel,
    #     trial_number increments 1–30.
    # The compound key is unique in both cases.
    pass_map: dict[tuple, bool] = {}
    for _, row in bc_kernel.iterrows():
        key = (row.get("app_name"), row.get("trial_number"))
        judge = row.get("judge_baseline_verdict")
        pass_map[key] = str(judge).strip().upper() == "VALID"

    # Map back to the trial order in df_kernel using the compound key
    compound_keys = list(zip(df_kernel["app_name"], df_kernel["trial_number"]))
    result = pd.Series(
        [pass_map.get(k, False) for k in compound_keys],
        index=df_kernel.index,
        dtype=bool,
    )

    # Warn for trials with no matching best-code row
    missing = [k for k in compound_keys if k not in pass_map]
    if missing:
        log.warning(
            "Kernel '%s': %d trial(s) have no is_best_code==True row in Table 2 — "
            "treated as fail for Variant 2.",
            kernel, len(missing),
        )

    return result.values


def compute_judge_zeroshot_pass_vector(
    df_kernel: pd.DataFrame,
    df_attempts: pd.DataFrame,
) -> np.ndarray:
    """Variant 3: judge-pass@k (zero-shot).

    A trial passes iff the LLM judge said VALID for the zero-shot code,
    using judge_baseline_verdict from the attempt log
    (generation_type == 'zeroshot' rows).

    Under the judge-as-baseline design (post-June 2026), judge_baseline_verdict
    is the raw LLM judge output for the zero-shot code and is recorded
    INDEPENDENTLY of the empirical verify (no override). V3 (judge baseline) and
    V4 (empirical/framework verdict) are parallel measurements on the same ZS
    code — comparing them directly answers "how often would the judge alone have
    approved what the three-tier framework also approves?". Reading the
    attempt log is preferred over Table 1's base_v1_valid because base_v1_valid
    holds the framework verdict (= empirical when verify_zeroshot=True, else
    the judge baseline), which is V4, not V3.

    Trials where the zero-shot code failed to compile or execute have no
    zeroshot attempt row (the judge was never called) and default to False —
    the correct interpretation for both V3 and V4.

    Falls back to base_v1_valid == "VALID" if df_attempts is empty or contains
    no zeroshot rows (preserves backward compatibility with runs that pre-date
    the attempt log).

    Returns:
        Boolean numpy array of shape (n_trials,).
    """
    # Prefer judge_baseline_verdict from the attempt log.
    # generation_type is uppercased by the data_loader normalization step.
    if not df_attempts.empty and "generation_type" in df_attempts.columns:
        zs_rows = df_attempts[df_attempts["generation_type"] == "ZEROSHOT"]
        if not zs_rows.empty and "judge_baseline_verdict" in zs_rows.columns:
            # Build (app_name, trial_number) → raw judge verdict mapping
            judge_map: dict[tuple, bool] = {}
            for _, row in zs_rows.iterrows():
                key = (row.get("app_name"), row.get("trial_number"))
                judge_map[key] = str(row.get("judge_baseline_verdict", "")).strip().upper() == "VALID"

            compound_keys = list(zip(df_kernel["app_name"], df_kernel["trial_number"]))
            result = pd.Series(
                [judge_map.get(k, False) for k in compound_keys],
                index=df_kernel.index,
                dtype=bool,
            )

            missing = [k for k in compound_keys if k not in judge_map]
            if missing:
                log.debug(
                    "Variant 3: %d trial(s) have no zeroshot attempt row "
                    "(ZS failed to compile/execute — judge not called, treated as not-pass).",
                    len(missing),
                )
            return result.values

    # Fallback: use base_v1_valid from Table 1 for runs that pre-date the attempt
    # log. NOTE: base_v1_valid holds the framework verdict (= empirical when
    # verify_zeroshot=True, else the judge baseline); for runs where empirical
    # was run, this fallback measures V4 (framework), not V3 (judge baseline)
    # — they may differ when empirical and judge disagree.
    log.debug(
        "Variant 3: falling back to base_v1_valid (attempt log unavailable or "
        "missing zeroshot rows — note that on verify_zeroshot=True runs this "
        "fallback measures the framework verdict, not the judge baseline)."
    )
    return df_kernel["base_v1_valid"].eq("VALID").values


def compute_verified_zeroshot_pass_vector(
    df_kernel: pd.DataFrame,
) -> Optional[np.ndarray]:
    """Variant 4: verified-pass@k (zero-shot).

    A trial passes iff zeroshot_empirical_verdict == "PASS" — meaning the
    zero-shot code compiled, executed, and passed all three verification tiers
    when the pipeline ran empirical verification on it.

    SKIP and FAIL both count as not-pass:
    - SKIP: the zero-shot code failed to compile or execute (code is broken)
    - FAIL: one or more tiers detected a correctness violation

    Returns None if the column is missing (caller will skip this variant).

    Returns:
        Boolean numpy array of shape (n_trials,), or None if column absent.
    """
    if "zeroshot_empirical_verdict" not in df_kernel.columns:
        return None
    return df_kernel["zeroshot_empirical_verdict"].eq("PASS").values


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(df_trials: pd.DataFrame, df_attempts: pd.DataFrame) -> dict:
    """Compute all four pass@k variants for all kernels.

    Args:
        df_trials:  Normalized Table 1 DataFrame (from data_loader.load_data).
        df_attempts: Normalized Table 2 DataFrame (may be empty if unavailable).

    Returns:
        Nested dict structure:
        {
          "per_kernel": {
            "<kernel_name>": {
              "n": int,
              "c_<variant>": int,
              "<variant>_pass_at_k": {
                k: {"point": float, "ci_low": float, "ci_high": float, "n": int, "c": int}
              }
            }
          },
          "summary": {
            "<variant>_pass_at_k": {
              k: {"mean": float, "std": float, "per_kernel_points": [float, ...]}
            }
          },
          "skipped_variants": [str, ...]  # variants that could not be computed
        }
    """
    kernels = get_kernels(df_trials)
    results: dict = {"per_kernel": {}, "summary": {}, "skipped_variants": []}

    # Pre-compute Variant 2 join: best-code rows from Table 2 + runtime from Table 1
    if not df_attempts.empty and "is_best_code" in df_attempts.columns:
        best_code_rows = df_attempts[df_attempts["is_best_code"] == True].copy()
        runtime_cols = [
            "app_name", "trial_number", "kernel_name",
            "lassiee_clean_runtime_diff", "lassiee_clean_status", "lassiee_runtime_diff",
        ]
        t1_runtime = df_trials[[c for c in runtime_cols if c in df_trials.columns]]
        best_code_merged = best_code_rows.merge(
            t1_runtime,
            on=[c for c in ["app_name", "trial_number", "kernel_name"] if c in t1_runtime.columns],
            how="left",
            suffixes=("", "_t1"),
        )
    else:
        log.warning(
            "Table 2 unavailable or missing 'is_best_code' column. "
            "Variant 2 (judge-pass LASSI) will be skipped."
        )
        best_code_merged = pd.DataFrame()
        if "judge_lassi" not in results["skipped_variants"]:
            results["skipped_variants"].append("judge_lassi")

    # Check whether Variant 4 is computable
    v4_available = "zeroshot_empirical_verdict" in df_trials.columns
    if not v4_available:
        log.warning(
            "Column 'zeroshot_empirical_verdict' missing from Table 1. "
            "Variant 4 (verified-pass zero-shot) will be skipped."
        )
        if "verified_zeroshot" not in results["skipped_variants"]:
            results["skipped_variants"].append("verified_zeroshot")

    for kernel in kernels:
        df_k = get_trials_for_kernel(df_trials, kernel)
        n = len(df_k)
        if n == 0:
            log.warning("Kernel '%s' has 0 trials — skipping.", kernel)
            continue
        if n < max(K_VALUES):
            log.warning(
                "Kernel '%s' has only %d trials (< k_max=%d). "
                "Results for k > n are degenerate.",
                kernel, n, max(K_VALUES),
            )

        # Compute pass vectors for all four variants
        v1_pass = compute_verified_lassi_pass_vector(df_k)
        v2_pass = (
            compute_judge_lassi_pass_vector(df_k, best_code_merged, kernel)
            if not best_code_merged.empty
            else None
        )
        v3_pass = compute_judge_zeroshot_pass_vector(df_k, df_attempts)
        v4_pass = compute_verified_zeroshot_pass_vector(df_k) if v4_available else None

        kernel_results: dict = {"n": n}

        variants = [
            ("verified_lassi", v1_pass),
            ("judge_lassi", v2_pass),
            ("judge_zeroshot", v3_pass),
            ("verified_zeroshot", v4_pass),
        ]

        for name, pass_vec in variants:
            if pass_vec is None:
                # variant was skipped
                continue
            kernel_results[f"c_{name}"] = int(np.sum(pass_vec))
            kernel_results[f"{name}_pass_at_k"] = {}
            for k in K_VALUES:
                kernel_results[f"{name}_pass_at_k"][k] = bootstrap_pass_at_k(
                    pass_vec, k, B=BOOTSTRAP_B, seed=RANDOM_SEED
                )

        results["per_kernel"][kernel] = kernel_results
        log.info(
            "Kernel '%s': n=%d, c(V1)=%d, c(V2)=%s, c(V3)=%d, c(V4)=%s",
            kernel, n,
            int(np.sum(v1_pass)),
            int(np.sum(v2_pass)) if v2_pass is not None else "N/A",
            int(np.sum(v3_pass)),
            int(np.sum(v4_pass)) if v4_pass is not None else "N/A",
        )

    # Summary: unweighted mean (and std) across kernels for each variant × k
    all_variants = ["verified_lassi", "judge_lassi", "judge_zeroshot", "verified_zeroshot"]
    active_kernels = [k for k in kernels if k in results["per_kernel"]]

    for name in all_variants:
        results["summary"][f"{name}_pass_at_k"] = {}
        for k in K_VALUES:
            points = [
                results["per_kernel"][ker][f"{name}_pass_at_k"][k]["point"]
                for ker in active_kernels
                if f"{name}_pass_at_k" in results["per_kernel"].get(ker, {})
            ]
            if points:
                results["summary"][f"{name}_pass_at_k"][k] = {
                    "mean": float(np.mean(points)),
                    "std": float(np.std(points)),
                    "per_kernel_points": points,
                }
            else:
                results["summary"][f"{name}_pass_at_k"][k] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "per_kernel_points": [],
                }

    return results


# ---------------------------------------------------------------------------
# Reporting: tables + figure
# ---------------------------------------------------------------------------

def report(results: dict, output_dir: str) -> None:
    """Write tables and figure for Metric 1 to *output_dir*.

    Outputs:
    - pass_at_k_lassi.csv       — per-kernel table for LASSI variants (V1, V2)
    - pass_at_k_zeroshot.csv    — per-kernel table for zero-shot variants (V3, V4)
    - pass_at_k_summary.png     — grouped bar chart (all four variants)

    Cell format: "0.83 (0.67–0.93)".
    Average row at bottom of each table uses unweighted mean across kernels.
    Error bars on figure: cross-kernel std (not bootstrap CI — per LASSI-EE convention).
    """
    os.makedirs(output_dir, exist_ok=True)

    _write_lassi_table(results, output_dir)
    _write_zeroshot_table(results, output_dir)
    _write_summary_figure(results, output_dir)


def _cell(stats: dict) -> str:
    """Format a result dict as 'point (ci_low–ci_high)'."""
    p = stats["point"]
    lo = stats["ci_low"]
    hi = stats["ci_high"]
    return f"{p:.2f} ({lo:.2f}\u2013{hi:.2f})"


def _write_lassi_table(results: dict, output_dir: str) -> None:
    """Write pass_at_k_lassi.csv — LASSI variants (V1, V2) per kernel."""
    kernels = sorted(results["per_kernel"].keys())
    rows = []

    for kernel in kernels:
        kr = results["per_kernel"][kernel]
        row = {"Kernel": kernel}
        for k in K_VALUES:
            v1 = kr.get("verified_lassi_pass_at_k", {}).get(k)
            v2 = kr.get("judge_lassi_pass_at_k", {}).get(k)
            row[f"Verified k={k}"] = _cell(v1) if v1 else "N/A"
            row[f"Judge(LASSI) k={k}"] = _cell(v2) if v2 else "N/A"
        rows.append(row)

    # Average row
    avg_row = {"Kernel": "Average"}
    for k in K_VALUES:
        for name, label in [("verified_lassi", "Verified"), ("judge_lassi", "Judge(LASSI)")]:
            s = results["summary"].get(f"{name}_pass_at_k", {}).get(k, {})
            mean_val = s.get("mean", float("nan"))
            std_val = s.get("std", float("nan"))
            avg_row[f"{label} k={k}"] = (
                f"{mean_val:.2f} (\u00b1{std_val:.2f})" if not math.isnan(mean_val) else "N/A"
            )
    rows.append(avg_row)

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "pass_at_k_lassi.csv")
    df_out.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    print(f"\n=== pass@k — LASSI variants ===")
    print(df_out.to_string(index=False))


def _write_zeroshot_table(results: dict, output_dir: str) -> None:
    """Write pass_at_k_zeroshot.csv — zero-shot variants (V3, V4) per kernel."""
    kernels = sorted(results["per_kernel"].keys())
    rows = []

    for kernel in kernels:
        kr = results["per_kernel"][kernel]
        row = {"Kernel": kernel}
        for k in K_VALUES:
            v3 = kr.get("judge_zeroshot_pass_at_k", {}).get(k)
            v4 = kr.get("verified_zeroshot_pass_at_k", {}).get(k)
            row[f"Judge(ZS) k={k}"] = _cell(v3) if v3 else "N/A"
            row[f"Verified(ZS) k={k}"] = _cell(v4) if v4 else "N/A"
        rows.append(row)

    # Average row
    avg_row = {"Kernel": "Average"}
    for k in K_VALUES:
        for name, label in [("judge_zeroshot", "Judge(ZS)"), ("verified_zeroshot", "Verified(ZS)")]:
            s = results["summary"].get(f"{name}_pass_at_k", {}).get(k, {})
            mean_val = s.get("mean", float("nan"))
            std_val = s.get("std", float("nan"))
            avg_row[f"{label} k={k}"] = (
                f"{mean_val:.2f} (\u00b1{std_val:.2f})" if not math.isnan(mean_val) else "N/A"
            )
    rows.append(avg_row)

    df_out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "pass_at_k_zeroshot.csv")
    df_out.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    print(f"\n=== pass@k — Zero-shot variants ===")
    print(df_out.to_string(index=False))


def _write_summary_figure(results: dict, output_dir: str) -> None:
    """Write pass_at_k_summary.png/pdf — grouped bar chart, three variants.

    V2 (Judge LASSI) is excluded from the figure to keep the visual focused on
    the cross-system comparison (LASSI framework vs. vanilla baseline). V1
    (verified-pass@k LASSI) and V2 (judge-pass@k LASSI) are independent
    measurements on the same best-code set under the judge-as-baseline design,
    and are reported side-by-side in the per-kernel CSVs; the figure highlights
    V1 vs. V3/V4 (the across-system contrast) because that's the headline result.
    """
    variants = [
        ("verified_lassi",    "Verified (LASSI)",    "#1f4e79"),  # dark blue
        ("verified_zeroshot", "Verified (Vanilla)", "#85c1e9"),  # light blue (same family as LASSI)
        ("judge_zeroshot",    "LLM Judge (Vanilla)", "#e07b39"),  # orange
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    n_k = len(K_VALUES)
    n_variants = len(variants)
    bar_width = 0.22
    group_gap = 0.08
    group_width = n_variants * bar_width + group_gap
    x_positions = np.arange(n_k) * group_width

    for i, (name, label, color) in enumerate(variants):
        summary_key = f"{name}_pass_at_k"
        means = []
        stds = []
        for k in K_VALUES:
            s = results["summary"].get(summary_key, {}).get(k, {})
            means.append(s.get("mean", float("nan")))
            stds.append(s.get("std", float("nan")))

        if all(math.isnan(m) for m in means):
            continue  # variant not computed (column was missing)

        x_offset = x_positions + i * bar_width - (n_variants - 1) * bar_width / 2
        ax.bar(
            x_offset, means, bar_width,
            yerr=stds,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "#555555"},
        )

    ax.set_xlabel("k (number of candidates)", fontsize=12)
    ax.set_ylabel("pass@k", fontsize=12)
    # pad=50 leaves room between the title and the legend row that sits just above the axes
    ax.set_title("verified-pass@k vs. judge-pass@k", fontsize=13, pad=25)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"k={k}" for k in K_VALUES], fontsize=11)
    ax.set_ylim(0.0, 1.05)
    # Horizontal legend row between the title and the bars (anchored to axes top, centered)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(variants),
        fontsize=9,
        frameon=False,
        borderaxespad=0.5,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Outside the axes, same row as the x-axis label, right-justified.
    # Uses axes-fraction coordinates; bbox_inches='tight' on save captures it.
    ax.text(
        1.0, -0.10,
        "Error bars: cross-kernel std",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color="#666666",
    )

    fig.tight_layout()
    out_path = os.path.join(output_dir, "pass_at_k_summary.png")
    # bbox_inches='tight' ensures the legend above the axes boundary is not clipped
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Wrote %s", out_path)
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")
