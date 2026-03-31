"""
Verification Effectiveness Analysis.

Reads Table 2 (lassi-verify_attempt_log.csv) and Table 1 for sub-metric D.
Together these five sub-metrics make the empirical case for the three-tier
adaptive verification framework as a mechanism, not just a filter.

Sub-metrics
-----------
A. Correction Yield
   "When the framework detects a bug, is the feedback specific enough to fix it?"
   For each self-correction attempt triggered by a tier failure, did the next
   attempt resolve that specific tier?  Success requires strict PASS (SKIP means
   the code failed earlier and never reached the tier — not a fix).

B. Attempts to Resolution
   "How expensive is it to go from detection to correction?"
   Distribution of total attempts per code version, broken down by outcome
   (VALID, budget-exhausted, other-INVALID).

C. Budget Exhaustion Rate
   "How often does the framework detect a problem it can't help the LLM fix?"
   Fraction of code versions that exhaust their per-code self-correction budget,
   broken down by which budget fired (tier1, tier2, tier3, global).

D. SC Budget Breakdown by Trigger Type
   "Does the pipeline spend its effort on verification-driven corrections or
   infrastructure failures?"
   If most corrections are tier-driven rather than compile/execute errors,
   the verification is doing work that matters.  Reads aggregate counts from
   Table 1 (compile_selfcorrections, tier2_selfcorrections, etc.).

E. Tier Detection vs. Judge Comparison
   "How often would the judge alone have missed bugs that the tiers caught?"
   Every tier-FAIL event is extracted with its diagnostic label/relation,
   whether the judge agreed, and whether the next attempt fixed it.
   The judge-disagreement rate is the headline number for the paper's
   central argument.
"""

from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import get_kernels

log = logging.getLogger(__name__)

# generation_type → which tier column it was correcting.
# Values are stored UPPERCASED by _normalize_verdict_strings in data_loader.
_TIER_SC_MAP = {
    "SELF_CORRECTION_TIER1": "tier1_verdict",
    "SELF_CORRECTION_TIER2": "tier2_verdict",
    "SELF_CORRECTION_TIER3": "tier3_verdict",
}
_INFRA_SC_MAP = {
    "SELF_CORRECTION_COMPILE": "compile",
    "SELF_CORRECTION_EXECUTE": "execute",
}
# Rows to exclude from trajectory analysis (not part of the LASSI loop)
_EXCLUDE_GEN_TYPES = {"LASSI_FRESH", "ZEROSHOT"}

# Substrings that flag a budget-exhaustion override
_EXHAUSTION_SUBSTRINGS = {"exhausted", "tier1_exhausted"}


def _is_exhausted(override: object) -> bool:
    if pd.isna(override) or override is None:
        return False
    s = str(override).strip().lower()
    return s != "none" and any(sub in s for sub in _EXHAUSTION_SUBSTRINGS)


def _version_groups(df: pd.DataFrame):
    """Iterate over unique code versions (app_name, trial_number, code_block_index)."""
    return df.groupby(["app_name", "trial_number", "code_block_index"], sort=False)


# ---------------------------------------------------------------------------
# Sub-metric A: Correction Yield
# ---------------------------------------------------------------------------

def _compute_correction_yield(df: pd.DataFrame) -> dict:
    """
    For every tier-triggered self-correction attempt, did the corrected code
    pass ALL three tiers?

    Success definition: tier1_verdict == PASS AND tier2_verdict == PASS AND
    tier3_verdict == PASS.  No SKIP, no FAIL anywhere.  The triggering tier
    determines which bucket the attempt is counted in, but the outcome is
    binary: fully passing or not.

    Regression sub-count (within failures): cases where the triggering tier
    was fixed (PASS) but a different tier failed or SKIPped.  Distinguishes
    "feedback was unhelpful" from "feedback was right but fix was incomplete."

    Infrastructure corrections (compile, execute) use a simpler rule: success
    means compile_status == SUCCESS (and execute_status == SUCCESS for execute
    corrections).  These measure whether the LLM can fix a syntax/runtime
    error, not whether the verification feedback loop is effective.

    Returns:
      {
        "overall":    { "tier1": {attempts, successes, rate},
                        "tier2": ..., "tier3": ...,
                        "compile": ..., "execute": ...,
                        "regressions": int },
        "per_kernel": { kernel: same structure }
      }
    """
    sc_rows = df[~df["generation_type"].isin(_EXCLUDE_GEN_TYPES)].copy()

    def _events_for_version(grp: pd.DataFrame) -> list[dict]:
        records = grp.sort_values("attempt_number").to_dict("records")
        events = []
        for row in records:
            gen = row.get("generation_type", "")
            if gen in _TIER_SC_MAP:
                tier_col = _TIER_SC_MAP[gen]
                # gen is uppercased by normalization; strip prefix and lowercase
                tier_key = gen.replace("SELF_CORRECTION_", "").lower()
                # Success requires ALL THREE tiers PASS — no SKIP, no FAIL anywhere.
                # A correction that fixes the triggering tier but breaks another is
                # still a failure; there is no "partial success" category.
                t1 = str(row.get("tier1_verdict", "")).strip().upper()
                t2 = str(row.get("tier2_verdict", "")).strip().upper()
                t3 = str(row.get("tier3_verdict", "")).strip().upper()
                success = (t1 == "PASS" and t2 == "PASS" and t3 == "PASS")
                # Regression: triggering tier was fixed (PASS) but overall still
                # failed because a different tier failed or SKIPped.
                # Diagnostic sub-count within failures — not a success modifier.
                triggering_verdict = str(row.get(tier_col, "")).strip().upper()
                regression = (not success) and (triggering_verdict == "PASS")
                events.append({"kind": tier_key, "success": success, "regression": regression})

            elif gen in _INFRA_SC_MAP:
                infra_key = _INFRA_SC_MAP[gen]
                if infra_key == "compile":
                    success = str(row.get("compile_status", "")).strip().upper() == "SUCCESS"
                else:
                    success = (
                        str(row.get("compile_status", "")).strip().upper() == "SUCCESS"
                        and str(row.get("execute_status", "")).strip().upper() == "SUCCESS"
                    )
                events.append({"kind": infra_key, "success": success, "regression": False})
        return events

    all_events: list[dict] = []
    kernel_events: dict[str, list[dict]] = {}

    for (app_name, trial, cbi), grp in _version_groups(sc_rows):
        kernel = (
            df.loc[df["app_name"] == app_name, "kernel_name"].iloc[0]
            if "kernel_name" in df.columns else app_name
        )
        evts = _events_for_version(grp)
        all_events.extend(evts)
        kernel_events.setdefault(kernel, []).extend(evts)

    def _summarise(events: list[dict]) -> dict:
        out = {}
        for cat in ("tier1", "tier2", "tier3", "compile", "execute"):
            cat_evts = [e for e in events if e["kind"] == cat]
            n = len(cat_evts)
            s = sum(e["success"] for e in cat_evts)
            out[cat] = {"attempts": n, "successes": s,
                        "rate": (s / n) if n > 0 else None}
        out["regressions"] = sum(e["regression"] for e in events)
        return out

    return {
        "overall":    _summarise(all_events),
        "per_kernel": {k: _summarise(v) for k, v in kernel_events.items()},
    }


# ---------------------------------------------------------------------------
# Sub-metric B: Attempts to Resolution
# ---------------------------------------------------------------------------

def _compute_attempts_to_resolution(df: pd.DataFrame) -> dict:
    """
    Total attempts per code version grouped by final outcome.

    Outcomes:
      "valid"     — final_verdict == VALID
      "exhausted" — override_applied indicates budget exhaustion
      "invalid"   — INVALID but not exhausted (e.g. judge-only override)
    """
    # Exclude zeroshot single-pass rows
    df_main = df[df["generation_type"] != "ZEROSHOT"].copy()

    max_attempts = (
        df_main.groupby(["app_name", "trial_number", "code_block_index"])["attempt_number"]
        .max()
        .reset_index(name="total_attempts")
    )

    finals = df_main[df_main["is_final_attempt"] == True].merge(
        max_attempts, on=["app_name", "trial_number", "code_block_index"], how="left"
    )

    def _classify(row) -> str:
        fv = str(row.get("final_verdict", "")).strip().upper()
        if fv == "VALID":
            return "valid"
        if _is_exhausted(row.get("override_applied")):
            return "exhausted"
        return "invalid"

    finals["outcome"] = finals.apply(_classify, axis=1)

    def _stats(df_sub: pd.DataFrame) -> dict:
        out = {}
        for outcome in ("valid", "exhausted", "invalid"):
            counts = df_sub.loc[df_sub["outcome"] == outcome, "total_attempts"]
            n = len(counts)
            out[outcome] = {
                "n_versions": n,
                "mean":   float(counts.mean())   if n > 0 else None,
                "median": float(counts.median()) if n > 0 else None,
                "max":    int(counts.max())      if n > 0 else None,
                "raw":    counts.tolist(),
            }
        return out

    per_kernel = {}
    if "kernel_name" in finals.columns:
        for kernel, grp in finals.groupby("kernel_name"):
            per_kernel[kernel] = _stats(grp)

    return {"overall": _stats(finals), "per_kernel": per_kernel}


# ---------------------------------------------------------------------------
# Sub-metric C: Budget Exhaustion Rate
# ---------------------------------------------------------------------------

def _compute_exhaustion_rate(df: pd.DataFrame) -> dict:
    """Fraction of code versions that exhaust their SC budget, by budget type."""
    finals = df[
        (df["is_final_attempt"] == True) & (df["generation_type"] != "zeroshot")
    ].copy()

    def _budget_type(override: object) -> str:
        if pd.isna(override) or str(override).strip().lower() in ("", "none", "nan"):
            return "not_exhausted"
        s = str(override).strip().lower()
        if "tier1" in s:     return "tier1_budget"
        if "tier2" in s:     return "tier2_budget"
        if "tier3" in s:     return "tier3_budget"
        if "global" in s or "sc_budget" in s: return "global_budget"
        if "exhausted" in s: return "other_exhausted"
        return "not_exhausted"

    finals["budget_type"] = finals["override_applied"].apply(_budget_type)

    def _stats(df_sub: pd.DataFrame) -> dict:
        total = len(df_sub)
        ex = df_sub[df_sub["budget_type"] != "not_exhausted"]
        n_ex = len(ex)
        return {
            "total_versions":    total,
            "exhausted_versions": n_ex,
            "exhaustion_rate":   (n_ex / total) if total > 0 else None,
            "breakdown":         ex["budget_type"].value_counts().to_dict(),
        }

    per_kernel = {}
    if "kernel_name" in finals.columns:
        for kernel, grp in finals.groupby("kernel_name"):
            per_kernel[kernel] = _stats(grp)

    return {"overall": _stats(finals), "per_kernel": per_kernel}


# ---------------------------------------------------------------------------
# Sub-metric D: SC Budget Breakdown by Trigger Type (Table 1)
# ---------------------------------------------------------------------------

def _compute_budget_breakdown(df_trials: pd.DataFrame) -> dict:
    """Fraction of total self-corrections by trigger category."""
    sc_cols = {
        "compile":  "compile_selfcorrections",
        "execute":  "execute_selfcorrections",
        "tier1":    "tier1_selfcorrections",
        "tier2":    "tier2_selfcorrections",
        "tier3":    "tier3_selfcorrections",
        "total":    "total_selfcorrections",
    }
    missing = [v for v in sc_cols.values() if v not in df_trials.columns]
    if missing:
        log.warning("Budget breakdown: missing Table 1 columns %s — skipping.", missing)
        return {}

    def _stats(df_sub: pd.DataFrame) -> dict:
        counts = {k: int(df_sub[v].sum()) for k, v in sc_cols.items()}
        total = counts["total"]
        fracs = (
            {k: counts[k] / total for k in ("compile", "execute", "tier1", "tier2", "tier3")}
            if total > 0 else {}
        )
        return {"counts": counts, "fractions": fracs}

    per_kernel = {}
    if "kernel_name" in df_trials.columns:
        for kernel, grp in df_trials.groupby("kernel_name"):
            per_kernel[kernel] = _stats(grp)

    return {"overall": _stats(df_trials), "per_kernel": per_kernel}


# ---------------------------------------------------------------------------
# Sub-metric E: Tier Detection vs. Judge Comparison
# ---------------------------------------------------------------------------

def _compute_tier_detection_cases(df: pd.DataFrame) -> pd.DataFrame:
    """One row per tier-FAIL event with judge agreement and fix outcome."""
    if df.empty:
        return pd.DataFrame()

    df_main = df[df["generation_type"] != "ZEROSHOT"].copy()
    rows = []

    for (app_name, trial, cbi), grp in _version_groups(df_main):
        kernel = (
            grp["kernel_name"].iloc[0] if "kernel_name" in grp.columns else app_name
        )
        records = grp.sort_values("attempt_number").reset_index(drop=True).to_dict("records")

        for i, row in enumerate(records):
            for tier, verdict_col, label_col in [
                ("tier1", "tier1_verdict", "tier1_failed_labels"),
                ("tier2", "tier2_verdict", "tier2_failed_labels"),
                ("tier3", "tier3_verdict", "tier3_failed_relation"),
            ]:
                if str(row.get(verdict_col, "")).strip().upper() != "FAIL":
                    continue

                is_final = row.get("is_final_attempt") == True
                judge_raw = row.get("judge_verdict") if is_final else None
                judge_str = (
                    str(judge_raw).strip().upper()
                    if judge_raw and not pd.isna(judge_raw)
                    else None
                )
                if judge_str in (None, "NONE", "NAN", ""):
                    judge_str = None

                # judge_agreed_with_fail: judge also said INVALID
                judge_agreed = (judge_str == "INVALID") if judge_str else None

                # Did the next attempt fix this tier?
                next_fixed = None
                if i + 1 < len(records):
                    nv = str(records[i + 1].get(verdict_col, "")).strip().upper()
                    next_fixed = (nv == "PASS")

                rows.append({
                    "kernel_name":               kernel,
                    "trial_number":              trial,
                    "code_block_index":          cbi,
                    "attempt_number":            row.get("attempt_number"),
                    "failed_tier":               tier,
                    "failed_labels_or_relation": row.get(label_col, ""),
                    "self_correction_trigger":   row.get("self_correction_trigger", ""),
                    "is_final_attempt":          is_final,
                    "judge_verdict":             judge_str,
                    "judge_agreed_with_fail":    judge_agreed,
                    "next_attempt_fixed":        next_fixed,
                })

    return pd.DataFrame(rows)


def _summarise_detection(cases: pd.DataFrame) -> dict:
    if cases.empty:
        return {}

    total = len(cases)
    by_tier = cases.groupby("failed_tier").size().to_dict()

    # Judge comparison — only final attempts where judge actually ran
    final_judged = cases[
        (cases["is_final_attempt"] == True)
        & cases["judge_verdict"].notna()
        & ~cases["judge_verdict"].isin(["NONE", "NAN", ""])
    ]
    n_judged = len(final_judged)
    n_agreed = int(final_judged["judge_agreed_with_fail"].sum()) if n_judged > 0 else 0
    n_disagreed = n_judged - n_agreed  # judge said VALID while tier caught a bug

    # Fix rate
    has_next = cases[cases["next_attempt_fixed"].notna()]
    n_has_next = len(has_next)
    n_fixed = int(has_next["next_attempt_fixed"].sum()) if n_has_next > 0 else 0

    per_kernel = {}
    for kernel, grp in cases.groupby("kernel_name"):
        kfj = grp[
            (grp["is_final_attempt"] == True)
            & grp["judge_verdict"].notna()
            & ~grp["judge_verdict"].isin(["NONE", "NAN", ""])
        ]
        kn = len(kfj)
        ka = int(kfj["judge_agreed_with_fail"].sum()) if kn > 0 else 0
        per_kernel[kernel] = {
            "total_tier_failures":   len(grp),
            "by_tier":               grp.groupby("failed_tier").size().to_dict(),
            "judged_final_failures": kn,
            "judge_agreed":          ka,
            "judge_disagreed":       kn - ka,
            "judge_disagreement_rate": ((kn - ka) / kn) if kn > 0 else None,
        }

    return {
        "total_tier_failures":       total,
        "by_tier":                   by_tier,
        "judged_final_failures":     n_judged,
        "judge_agreed_with_tier":    n_agreed,
        "judge_disagreed_with_tier": n_disagreed,
        "judge_disagreement_rate":   (n_disagreed / n_judged) if n_judged > 0 else None,
        "correction_attempts_with_next": n_has_next,
        "next_attempt_fixed_count":  n_fixed,
        "correction_fix_rate":       (n_fixed / n_has_next) if n_has_next > 0 else None,
        "by_kernel":                 per_kernel,
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def compute(df_trials: pd.DataFrame, df_attempts: pd.DataFrame) -> dict:
    """Run all sub-metrics. Returns a nested results dict.

    Returns {} if df_attempts is empty (Table 2 is required).
    """
    if df_attempts.empty:
        log.warning(
            "Table 2 (attempt log) is empty — verification_analysis requires "
            "per-attempt data. Skipping."
        )
        return {}

    log.info("Correction Yield (A) ...")
    vcy = _compute_correction_yield(df_attempts)

    log.info("Attempts to Resolution (B) ...")
    attempts = _compute_attempts_to_resolution(df_attempts)

    log.info("Budget Exhaustion Rate (C) ...")
    exhaustion = _compute_exhaustion_rate(df_attempts)

    log.info("SC Budget Breakdown (D) ...")
    budget = _compute_budget_breakdown(df_trials)

    log.info("Tier Detection Case Studies (E) ...")
    cases = _compute_tier_detection_cases(df_attempts)
    detection_summary = _summarise_detection(cases)

    return {
        "correction_yield":    vcy,
        "attempts":            attempts,
        "exhaustion":          exhaustion,
        "budget_breakdown":    budget,
        "detection_cases":     cases,
        "detection_summary":   detection_summary,
    }


def report(results: dict, output_dir: str) -> None:
    """Write CSVs and summary figure; print tables to stdout."""
    if not results:
        log.warning("verification_analysis: no results to report.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- A: Correction Yield ---
    vcy = results["correction_yield"]
    vcy_rows = []
    for scope, data in [("OVERALL", vcy["overall"])] + list(vcy["per_kernel"].items()):
        for cat in ("tier1", "tier2", "tier3", "compile", "execute"):
            entry = data.get(cat, {})
            n = entry.get("attempts", 0)
            if n == 0:
                continue
            vcy_rows.append({
                "scope":     scope,
                "trigger":   cat,
                "attempts":  n,
                "successes": entry["successes"],
                "yield":     f"{entry['rate']:.1%}" if entry["rate"] is not None else "N/A",
            })
    vcy_df = pd.DataFrame(vcy_rows)
    _write(vcy_df, output_dir, "verification_analysis_yield.csv")
    print("\n=== A. Correction Yield ===")
    print(vcy_df.to_string(index=False))
    print(f"  Regressions (fixed trigger tier, broke another): "
          f"{vcy['overall'].get('regressions', 0)}")

    # --- B: Attempts to Resolution ---
    att = results["attempts"]
    att_rows = []
    for scope, data in [("OVERALL", att["overall"])] + list(att.get("per_kernel", {}).items()):
        for outcome in ("valid", "exhausted", "invalid"):
            entry = data.get(outcome, {})
            n = entry.get("n_versions", 0)
            if n == 0:
                continue
            att_rows.append({
                "scope":   scope,
                "outcome": outcome,
                "n":       n,
                "mean":    f"{entry['mean']:.1f}" if entry.get("mean") else "N/A",
                "median":  f"{entry['median']:.1f}" if entry.get("median") else "N/A",
                "max":     entry.get("max", "N/A"),
            })
    att_df = pd.DataFrame(att_rows)
    _write(att_df, output_dir, "verification_analysis_attempts.csv")
    print("\n=== B. Attempts to Resolution ===")
    print(att_df.to_string(index=False))

    # --- C: Exhaustion Rate ---
    exh = results["exhaustion"]
    exh_rows = []
    for scope, data in [("OVERALL", exh["overall"])] + list(exh.get("per_kernel", {}).items()):
        rate = data.get("exhaustion_rate")
        exh_rows.append({
            "scope":              scope,
            "total_versions":     data["total_versions"],
            "exhausted_versions": data["exhausted_versions"],
            "exhaustion_rate":    f"{rate:.1%}" if rate is not None else "N/A",
            "breakdown":          str(data.get("breakdown", {})),
        })
    exh_df = pd.DataFrame(exh_rows)
    _write(exh_df, output_dir, "verification_analysis_exhaustion.csv")
    print("\n=== C. Budget Exhaustion Rate ===")
    print(exh_df.to_string(index=False))

    # --- D: Budget Breakdown ---
    bud = results.get("budget_breakdown", {})
    bud_rows = []
    for scope, data in [("OVERALL", bud.get("overall", {}))] + list(bud.get("per_kernel", {}).items()):
        if not data:
            continue
        c = data.get("counts", {})
        f = data.get("fractions", {})
        bud_rows.append({
            "scope":     scope,
            "total_sc":  c.get("total", 0),
            "compile":   f"{f['compile']:.0%} ({c.get('compile',0)})" if f else "N/A",
            "execute":   f"{f['execute']:.0%} ({c.get('execute',0)})" if f else "N/A",
            "tier1":     f"{f['tier1']:.0%} ({c.get('tier1',0)})"   if f else "N/A",
            "tier2":     f"{f['tier2']:.0%} ({c.get('tier2',0)})"   if f else "N/A",
            "tier3":     f"{f['tier3']:.0%} ({c.get('tier3',0)})"   if f else "N/A",
        })
    bud_df = pd.DataFrame(bud_rows)
    _write(bud_df, output_dir, "verification_analysis_budget.csv")
    print("\n=== D. SC Budget Breakdown by Trigger Type ===")
    print(bud_df.to_string(index=False))

    # --- E: Detection Cases + Summary ---
    cases = results.get("detection_cases", pd.DataFrame())
    if not cases.empty:
        _write(cases, output_dir, "verification_analysis_cases.csv")

    ds = results.get("detection_summary", {})
    print("\n=== E. Tier Detection vs. Judge ===")
    print(f"  Total tier-FAIL events:          {ds.get('total_tier_failures', 0)}")
    print(f"  By tier: {ds.get('by_tier', {})}")
    print(f"  Final attempts with judge data:  {ds.get('judged_final_failures', 0)}")
    print(f"  Judge agreed (said INVALID):     {ds.get('judge_agreed_with_tier', 0)}")
    judged = ds.get("judged_final_failures", 0)
    disagreed = ds.get("judge_disagreed_with_tier", 0)
    dr = ds.get("judge_disagreement_rate")
    print(f"  Judge missed bug (said VALID):   {disagreed}/{judged}  "
          f"({f'{dr:.1%}' if dr is not None else 'N/A'})")
    fix_rate = ds.get("correction_fix_rate")
    print(f"  Next attempt fixed the tier:     {ds.get('next_attempt_fixed_count', 0)}/"
          f"{ds.get('correction_attempts_with_next', 0)}  "
          f"({f'{fix_rate:.1%}' if fix_rate is not None else 'N/A'})")
    if ds.get("by_kernel"):
        for kernel, kd in ds["by_kernel"].items():
            dr_k = kd.get("judge_disagreement_rate")
            print(f"    {kernel}: {kd['total_tier_failures']} failures, "
                  f"judge missed {kd['judge_disagreed']}/{kd['judged_final_failures']} "
                  f"({f'{dr_k:.1%}' if dr_k is not None else 'N/A'})")

    _plot_summary(results, output_dir)


def _write(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    log.info("Wrote %s", path)


def _plot_summary(results: dict, output_dir: str) -> None:
    """Two-panel figure: Correction Yield per tier (A) | SC budget breakdown (D)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Panel A: Correction Yield per tier
    ax = axes[0]
    vcy_ov = results["correction_yield"]["overall"]
    tier_keys   = ["tier1", "tier2", "tier3"]
    tier_labels = ["Tier 1", "Tier 2", "Tier 3"]
    colors_a    = ["#4C9BE8", "#F0A500", "#E85C4C"]
    rates  = [vcy_ov[t]["rate"]     if vcy_ov[t]["rate"] is not None else 0.0 for t in tier_keys]
    counts = [vcy_ov[t]["attempts"] for t in tier_keys]
    bars = ax.bar(tier_labels, rates, color=colors_a, alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Correction Success Rate")
    ax.set_title("A. Correction Yield\nper Triggering Tier")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, n, r in zip(bars, counts, rates):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{r:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)

    # Panel D: SC Budget Breakdown stacked bar
    ax2 = axes[1]
    bud_ov = results.get("budget_breakdown", {}).get("overall", {})
    fracs  = bud_ov.get("fractions", {})
    counts_bud = bud_ov.get("counts", {})
    if fracs:
        categories  = ["compile", "execute", "tier1", "tier2", "tier3"]
        cat_labels  = ["Compile", "Execute", "Tier 1", "Tier 2", "Tier 3"]
        colors_d    = ["#888888", "#AAAAAA", "#4C9BE8", "#F0A500", "#E85C4C"]
        bottom = 0.0
        for cat, label, color in zip(categories, cat_labels, colors_d):
            frac = fracs.get(cat, 0.0)
            if frac > 0:
                ax2.bar(0, frac, bottom=bottom, color=color, alpha=0.85,
                        label=f"{label} ({counts_bud.get(cat, 0)})", width=0.5)
                ax2.text(0, bottom + frac / 2, f"{frac:.0%}",
                         ha="center", va="center", fontsize=9,
                         color="white", fontweight="bold")
                bottom += frac
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_xticks([])
        ax2.set_ylabel("Fraction of Total Self-Corrections")
        ax2.set_title(f"D. SC Budget Breakdown\n(total={counts_bud.get('total', 0)})")
        ax2.legend(loc="upper right", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No budget data available",
                 ha="center", va="center", transform=ax2.transAxes, color="gray")
        ax2.set_title("D. SC Budget Breakdown")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"verification_analysis_summary.{ext}")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Wrote %s", path)
    plt.close(fig)
    print(f"\nFigure saved to "
          f"{os.path.join(output_dir, 'verification_analysis_summary.png')}")
