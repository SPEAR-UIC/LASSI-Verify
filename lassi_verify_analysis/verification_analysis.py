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

    log.info("Confusion Matrix (Judge vs. Tiers) ...")
    confusion_matrix = _compute_confusion_matrix(df_attempts)

    return {
        "correction_yield":    vcy,
        "attempts":            attempts,
        "exhaustion":          exhaustion,
        "budget_breakdown":    budget,
        "detection_cases":     cases,
        "detection_summary":   detection_summary,
        "confusion_matrix":    confusion_matrix,
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

    # --- Bug class breakdown CSV (for tier complementarity) ---
    cases = results.get("detection_cases", pd.DataFrame())
    if not cases.empty:
        bc_rows = []
        for tier in ("tier1", "tier2", "tier3"):
            tier_cases = cases[cases["failed_tier"] == tier]
            class_counts: dict[str, int] = {}
            for val in tier_cases["failed_labels_or_relation"]:
                for cls in _extract_bug_classes(val):
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            total = sum(class_counts.values())
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                bc_rows.append({
                    "tier": tier,
                    "bug_class": cls,
                    "count": count,
                    "fraction": f"{count/total:.1%}" if total > 0 else "N/A",
                })
        if bc_rows:
            _write(pd.DataFrame(bc_rows), output_dir, "verification_analysis_bug_classes.csv")

    # --- Confusion matrix print + CSV ---
    cm = results.get("confusion_matrix", {})
    if cm:
        tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
        total = cm["total"]
        print("\n=== Judge vs. Three-Tier Confusion Matrix ===")
        print(f"  (tiers = ground truth, judge = classifier, n={total} final LASSI attempts)")
        print(f"  TP (tiers PASS + judge VALID):    {tp}  "
              f"({tp/total:.0%})" if total else "  TP: 0")
        print(f"  FN (tiers PASS + judge INVALID):  {fn}  "
              f"({fn/total:.0%})" if total else "  FN: 0")
        print(f"  FP (tiers FAIL + judge VALID):    {fp}  "
              f"({fp/total:.0%}) ← bugs judge missed" if total else "  FP: 0")
        print(f"  TN (tiers FAIL + judge INVALID):  {tn}  "
              f"({tn/total:.0%})" if total else "  TN: 0")
        _write(
            pd.DataFrame([{"cell": "TP", "count": tp},
                          {"cell": "FN", "count": fn},
                          {"cell": "FP", "count": fp},
                          {"cell": "TN", "count": tn},
                          {"cell": "total", "count": total}]),
            output_dir, "verification_analysis_confusion_matrix.csv"
        )

    _plot_summary(results, output_dir)


def _write(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    log.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Confusion matrix computation
# ---------------------------------------------------------------------------

def _compute_confusion_matrix(df: pd.DataFrame) -> dict:
    """2×2 confusion matrix: tiers (ground truth) vs. LLM judge (classifier).

    Ground truth:
      PASS = T1==PASS AND T2==PASS AND T3==PASS (all three tiers agree code is valid)
      FAIL = any tier failed

    Classifier = judge_verdict (VALID / INVALID).

    Only final LASSI attempts where the judge actually ran are included.

    Matrix cells (tiers = rows, judge = columns):
      TP: tiers PASS + judge VALID   (both agree: valid)
      FN: tiers PASS + judge INVALID (judge too strict — false rejection)
      FP: tiers FAIL + judge VALID   (judge missed the bug — false acceptance)
      TN: tiers FAIL + judge INVALID (both agree: invalid)

    For the paper argument, FP is the critical number: bugs that the three-tier
    framework caught but the judge would have let through.
    """
    if df.empty:
        return {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total": 0}

    finals = df[
        (df["is_final_attempt"] == True) &
        (~df["generation_type"].isin(_EXCLUDE_GEN_TYPES))
    ].copy()

    def _tiers_pass(row) -> bool:
        t1 = str(row.get("tier1_verdict", "")).strip().upper()
        t2 = str(row.get("tier2_verdict", "")).strip().upper()
        t3 = str(row.get("tier3_verdict", "")).strip().upper()
        return t1 == "PASS" and t2 == "PASS" and t3 == "PASS"

    def _judge_valid(v) -> object:
        if pd.isna(v) or v is None:
            return None
        s = str(v).strip().upper()
        if s in ("", "NONE", "NAN"):
            return None
        return s == "VALID"

    finals["tiers_pass"] = finals.apply(_tiers_pass, axis=1)
    finals["judge_valid"] = finals["judge_verdict"].apply(_judge_valid)

    # Only include rows where judge produced an actual verdict
    has_judge = finals[finals["judge_valid"].notna()].copy()
    n_total = len(has_judge)

    if n_total == 0:
        return {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "total": 0}

    tp = int(((has_judge["tiers_pass"] == True)  & (has_judge["judge_valid"] == True)).sum())
    fn = int(((has_judge["tiers_pass"] == True)  & (has_judge["judge_valid"] == False)).sum())
    fp = int(((has_judge["tiers_pass"] == False) & (has_judge["judge_valid"] == True)).sum())
    tn = int(((has_judge["tiers_pass"] == False) & (has_judge["judge_valid"] == False)).sum())

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": n_total}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _extract_bug_classes(val: object) -> list[str]:
    """Return a list of bug class names from a failed_labels_or_relation value.

    Handles Python lists (from LIST_COLUMNS_ATTEMPTS parsing), comma-separated
    strings, and single strings.  Empty / null values → ["(unknown)"].
    """
    if isinstance(val, list):
        cleaned = [s.strip() for s in val if str(s).strip()]
        return cleaned if cleaned else ["(unknown)"]
    if pd.isna(val) or val is None:
        return ["(unknown)"]
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "[]", ""):
        return ["(unknown)"]
    # handle stringified lists like "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].replace("'", "").replace('"', "")
        parts = [x.strip() for x in inner.split(",") if x.strip()]
        return parts if parts else ["(unknown)"]
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return parts if parts else ["(unknown)"]


_TIER_PALETTES = {
    # T1: blue family (dark → light)
    "tier1": ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"],
    # T2: orange family
    "tier2": ["#7f2704", "#d94801", "#f16913", "#fd8d3c", "#fdae6b", "#fdd0a2", "#fee6ce"],
    # T3: red/pink family
    "tier3": ["#67000d", "#a50f15", "#cb181d", "#ef3b2c", "#fb6a4a", "#fc9272", "#fcbba1"],
}


def _plot_tier_complementarity(cases: pd.DataFrame, ax: plt.Axes) -> None:
    """Horizontal stacked bar: FAIL events per tier, broken down by bug class.

    One row per tier (T1 at top, T3 at bottom).  Within each row the bar is
    stacked by bug class label, using shades within that tier's color family.
    A bug-class breakdown CSV is written separately (see report()).
    """
    tier_order  = ["tier1", "tier2", "tier3"]
    tier_labels = ["Tier 1", "Tier 2", "Tier 3"]
    y_positions = [2, 1, 0]  # tier1 at top

    legend_handles = []

    for tier, y_pos, row_label in zip(tier_order, y_positions, tier_labels):
        tier_cases = cases[cases["failed_tier"] == tier] if not cases.empty else pd.DataFrame()
        palette = _TIER_PALETTES.get(tier, ["#888888"])

        if tier_cases.empty:
            ax.barh(y_pos, 0, left=0, height=0.55, color=palette[0], alpha=0.3)
            ax.text(0.5, y_pos, "0 events", va="center", fontsize=8, color="#777777")
            continue

        # Count occurrences of each bug class across all FAIL events for this tier.
        # A single event can trigger multiple bug classes (e.g. two assertion labels
        # both FAIL in the same attempt), so we expand lists.
        class_counts: dict[str, int] = {}
        for val in tier_cases["failed_labels_or_relation"]:
            for cls in _extract_bug_classes(val):
                class_counts[cls] = class_counts.get(cls, 0) + 1

        # Sort by frequency descending so dominant class is leftmost
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        total_events = sum(c for _, c in sorted_classes)

        left = 0
        for j, (cls, count) in enumerate(sorted_classes):
            color = palette[j % len(palette)]
            ax.barh(y_pos, count, left=left, height=0.55,
                    color=color, edgecolor="white", linewidth=0.6)
            # White count label inside segment if wide enough
            seg_frac = count / total_events if total_events > 0 else 0
            if seg_frac >= 0.08:
                ax.text(left + count / 2, y_pos, str(count),
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
            # Truncated class name below bar (only first class per tier for clarity)
            if j == 0:
                short = cls[:22] + "…" if len(cls) > 22 else cls
                ax.text(left + count / 2, y_pos - 0.32, short,
                        ha="center", va="top", fontsize=6.5, color="#333333",
                        style="italic")
            left += count

        # Total count label at far right
        ax.text(left + 0.4, y_pos, f"n={total_events}", va="center",
                fontsize=8.5, color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(tier_labels, fontsize=10)
    ax.set_xlabel("Number of FAIL Events (all attempts)", fontsize=9)
    ax.set_title("Tier Complementarity: Bug Classes Detected by Each Tier", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.65, 2.65)


def _plot_confusion_matrix(cm: dict, ax: plt.Axes) -> None:
    """2×2 heatmap: tiers (rows) vs. judge (columns).

    TP/TN cells are green (both agree); FP cell is red (judge missed bug);
    FN cell is yellow (judge too strict).
    """
    tp = cm.get("tp", 0)
    fn = cm.get("fn", 0)
    fp = cm.get("fp", 0)
    tn = cm.get("tn", 0)
    total = cm.get("total", tp + fn + fp + tn)

    # Cell layout: [[TP, FN], [FP, TN]]
    # Rows: Tiers PASS (top), Tiers FAIL (bottom)
    # Cols: Judge VALID (left), Judge INVALID (right)
    cell_values   = [[tp,   fn  ], [fp,   tn  ]]
    cell_colors   = [["#2ca02c", "#d6b24a"], ["#d62728", "#2ca02c"]]
    cell_alphas   = [[0.75, 0.65          ], [0.85,      0.75     ]]

    for r in range(2):
        for c in range(2):
            val   = cell_values[r][c]
            color = cell_colors[r][c]
            alpha = cell_alphas[r][c]
            pct   = f"({val/total:.0%})" if total > 0 else ""
            ax.add_patch(
                plt.Rectangle((c, 1 - r), 1, 1,
                              facecolor=color, alpha=alpha, edgecolor="white", linewidth=2)
            )
            ax.text(c + 0.5, 1.5 - r, f"{val}\n{pct}",
                    ha="center", va="center", fontsize=13, fontweight="bold", color="white")

    # Axis labels
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Judge: VALID", "Judge: INVALID"], fontsize=9)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Tiers: FAIL", "Tiers: PASS"], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_title("Judge vs. Three-Tier Verdict\n(final LASSI attempts)", fontsize=10, pad=18)

    # Diagonal annotations
    ax.text(0.5, 1.5, "TP", ha="center", va="bottom",
            fontsize=7.5, color="white", alpha=0.8)
    ax.text(1.5, 1.5, "FN", ha="center", va="bottom",
            fontsize=7.5, color="white", alpha=0.8)
    ax.text(0.5, 0.5, "FP\n(missed bug)", ha="center", va="bottom",
            fontsize=7.5, color="white", alpha=0.8)
    ax.text(1.5, 0.5, "TN", ha="center", va="bottom",
            fontsize=7.5, color="white", alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)


def _plot_correction_yield(vcy_ov: dict, ax: plt.Axes) -> None:
    """Bar chart: correction yield per triggering tier."""
    tier_keys   = ["tier1", "tier2", "tier3"]
    tier_labels = ["Tier 1", "Tier 2", "Tier 3"]
    colors      = ["#4C9BE8", "#F0A500", "#E85C4C"]

    rates  = [vcy_ov[t]["rate"]     if vcy_ov[t]["rate"] is not None else 0.0 for t in tier_keys]
    counts = [vcy_ov[t]["attempts"] for t in tier_keys]

    bars = ax.bar(tier_labels, rates, color=colors, alpha=0.85)
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Correction Success Rate", fontsize=9)
    ax.set_title("Correction Yield\nper Triggering Tier", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for bar, n, r in zip(bars, counts, rates):
        if n > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.04,
                    f"{r:.0%}\n(n={n})", ha="center", va="bottom", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_summary(results: dict, output_dir: str) -> None:
    """Three-panel figure: tier complementarity (top) | confusion matrix + correction yield (bottom)."""
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.55, wspace=0.38)
    ax_comp = fig.add_subplot(gs[0, :])   # full-width top row
    ax_cm   = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_yld  = fig.add_subplot(gs[1, 1])   # bottom-right

    cases = results.get("detection_cases", pd.DataFrame())
    _plot_tier_complementarity(cases, ax_comp)

    cm = results.get("confusion_matrix", {})
    _plot_confusion_matrix(cm, ax_cm)

    vcy_ov = results["correction_yield"]["overall"]
    _plot_correction_yield(vcy_ov, ax_yld)

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"verification_analysis_summary.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Wrote %s", path)
    plt.close(fig)
    print(f"\nFigure saved to "
          f"{os.path.join(output_dir, 'verification_analysis_summary.png')}")
