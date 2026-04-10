"""
Verification Evidence Summary.

Extracts concrete verification evidence from the best code produced by each
trial.  For each verified-valid best code (lassiee_valid == "VALID"), this
module records what was specifically checked and what the evidence looked like
across all three tiers and the LLM judge.

The goal is to give a reader inspectable evidence of verification quality —
not just pass/fail counts — supporting the paper's argument that empirical
three-tier verification provides meaningful formal-methods-analogous coverage
even where full formal verification is infeasible for this code domain.

Tier mapping to formal methods concepts:
  Tier 1 (Invariant Assertions)   → Owicki-Gries auxiliary variable checks
  Tier 2 (Passthrough Equivalence) → Observational equivalence / forward simulation
  Tier 3 (Metamorphic Testing)     → Relational Hoare Logic / property-based testing

Outputs:
  verification_evidence_summary.csv  — one row per verified-valid trial
  verification_evidence_summary.txt  — aggregated per-kernel narrative summary
"""

from __future__ import annotations

import logging
import os
import statistics
from collections import Counter

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_labels(val: object) -> list[str]:
    """Parse a comma-separated label string (or list) into a clean list."""
    if isinstance(val, list):
        return [str(s).strip() for s in val if str(s).strip()]
    if pd.isna(val) or val is None:
        return []
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "[]", ""):
        return []
    # Handle stringified Python lists: "['a', 'b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].replace("'", "").replace('"', "")
        return [x.strip() for x in inner.split(",") if x.strip()]
    return [x.strip() for x in s.split(",") if x.strip()]


def _range_str(values: list, fmt: str = ".0f") -> str:
    """'min–max (median N)' or just 'N' if all equal."""
    if not values:
        return "N/A"
    lo, hi = min(values), max(values)
    med = statistics.median(values)
    if lo == hi:
        return f"{lo:{fmt}}"
    return f"{lo:{fmt}}–{hi:{fmt}} (median {med:{fmt}})"


def _error_range_str(values: list[float]) -> str:
    """Format relative error range, calling out exact-zero as bitwise-identical."""
    if not values:
        return "N/A"
    lo, hi = min(values), max(values)
    if hi == 0.0:
        return "0.0 (bitwise-identical output across all trials)"
    if lo == hi:
        return f"{lo:.2e}"
    if lo == 0.0:
        return f"0.0 – {hi:.2e}"
    return f"{lo:.2e} – {hi:.2e}"


def _label_sample(labels: list[str], n: int = 4) -> str:
    """First n labels with '...' suffix if list is longer."""
    if not labels:
        return "(none)"
    return ", ".join(labels[:n]) + (", ..." if len(labels) > n else "")


def _safe_float(val: object) -> float | None:
    try:
        f = float(val)
        return None if f != f else f  # drop NaN
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def compute(df_trials: pd.DataFrame, df_attempts: pd.DataFrame) -> dict:
    """Extract verification evidence for best codes across all trials.

    Args:
        df_trials:   Table 1 (pipeline_data). Provides lassiee_valid,
                     runtime diff, and kernel_name.
        df_attempts: Table 2 (attempt_log). Provides best-code rows with
                     all tier verdict, label, and diagnostic columns.

    Returns:
        {
          "per_trial":  [list of per-trial evidence dicts],
          "per_kernel": { kernel_name: aggregated evidence dict },
        }
    """
    if df_attempts.empty:
        log.warning("Table 2 is empty — verification evidence requires per-attempt data.")
        return {}

    if "is_best_code" not in df_attempts.columns:
        log.warning("'is_best_code' column missing from Table 2 — skipping.")
        return {}

    # One row per trial: the final attempt of the best LASSI-generated code.
    # is_best_code == True  → this code won the pipeline's performance comparison
    # is_final_attempt == True → take the verdict row, not any earlier retry attempt
    # generation_type != ZEROSHOT → exclude the vanilla LLM baseline row
    best = df_attempts[
        (df_attempts["is_best_code"] == True) &
        (df_attempts["is_final_attempt"] == True) &
        (df_attempts["generation_type"] != "ZEROSHOT")
    ].copy()

    if best.empty:
        log.warning("No is_best_code==True rows found in Table 2.")
        return {}

    # Join Table 1 fields (lassiee_valid, runtime, kernel_name) onto best-code rows.
    # Key: app_name + trial_number (both tables always have these).
    t1_cols = [c for c in [
        "app_name", "trial_number", "kernel_name",
        "lassiee_valid", "lassiee_runtime_diff", "lassiee_clean_runtime_diff",
    ] if c in df_trials.columns]

    join_keys = [c for c in ["app_name", "trial_number"] if c in df_trials.columns]
    joined = best.merge(
        df_trials[t1_cols],
        on=join_keys,
        how="left",
        suffixes=("", "_t1"),
    )

    # Resolve kernel_name: prefer the joined column, fall back to app_name
    if "kernel_name" not in joined.columns:
        joined["kernel_name"] = joined.get("kernel_name_t1", joined.get("app_name", "unknown"))

    # Filter to verified-valid best codes only
    if "lassiee_valid" in joined.columns:
        joined = joined[
            joined["lassiee_valid"].astype(str).str.strip().str.upper().eq("VALID")
        ].copy()

    if joined.empty:
        log.warning("No verified-valid best codes found after filtering lassiee_valid==VALID.")
        return {}

    # Build per-trial evidence dicts
    per_trial: list[dict] = []

    for _, row in joined.iterrows():
        t1_labels = _parse_labels(row.get("tier1_instrumented_labels"))
        t2_labels = _parse_labels(row.get("tier2_instrumented_labels"))
        t3_rels   = _parse_labels(row.get("tier3_tested_relation_names"))

        t2_err = _safe_float(row.get("tier2_max_relative_error"))

        # Prefer clean (uninstrumented) runtime diff; fall back to instrumented
        rt_diff = _safe_float(
            row.get("lassiee_clean_runtime_diff") or row.get("lassiee_runtime_diff")
        )

        per_trial.append({
            # Identification
            "kernel_name":          row.get("kernel_name", ""),
            "trial_number":         row.get("trial_number"),
            "code_block_index":     row.get("code_block_index"),
            "runtime_diff_pct":     f"{rt_diff:+.1f}%" if rt_diff is not None else "N/A",
            # Tier 1
            "t1_verdict":           str(row.get("tier1_verdict", "N/A")).strip().upper(),
            "t1_assertions_total":  row.get("tier1_assertions_total", "N/A"),
            "t1_assertions_passed": row.get("tier1_assertions_passed", "N/A"),
            "t1_label_count":       len(t1_labels),
            "t1_labels_sample":     _label_sample(t1_labels),
            "t1_labels_full":       "; ".join(t1_labels),
            # Tier 2
            "t2_verdict":           str(row.get("tier2_verdict", "N/A")).strip().upper(),
            "t2_blocks_total":      row.get("tier2_blocks_total", "N/A"),
            "t2_blocks_matched":    row.get("tier2_blocks_matched", "N/A"),
            # Store raw float for aggregation; formatted string for CSV display
            "_t2_err_raw":          t2_err,
            "t2_max_relative_error": f"{t2_err:.2e}" if t2_err is not None else "N/A",
            "t2_observation_points": "; ".join(t2_labels),
            # Tier 3
            "t3_verdict":           str(row.get("tier3_verdict", "N/A")).strip().upper(),
            "t3_relations_tested":  row.get("tier3_relations_tested", "N/A"),
            "t3_relations_passed":  row.get("tier3_relations_passed", "N/A"),
            "t3_relation_names":    "; ".join(t3_rels),
            # Judge
            "judge_verdict":        str(row.get("judge_verdict", "N/A")).strip().upper(),
        })

    # Aggregate per kernel
    per_kernel: dict[str, dict] = {}
    kernels = sorted({r["kernel_name"] for r in per_trial})

    for kernel in kernels:
        rows = [r for r in per_trial if r["kernel_name"] == kernel]
        n = len(rows)

        # T1: label frequency + assertion count range
        t1_label_freq: Counter = Counter()
        t1_assertion_counts: list[int] = []
        for r in rows:
            labels = [lbl for lbl in r["t1_labels_full"].split(";") if lbl.strip()]
            t1_label_freq.update(lbl.strip() for lbl in labels)
            try:
                t1_assertion_counts.append(int(r["t1_assertions_total"]))
            except (TypeError, ValueError):
                pass

        # T2: observation point frequency + error range
        t2_obs_freq: Counter = Counter()
        t2_errors: list[float] = []
        for r in rows:
            obs = [o.strip() for o in r["t2_observation_points"].split(";") if o.strip()]
            t2_obs_freq.update(obs)
            if r["_t2_err_raw"] is not None:
                t2_errors.append(r["_t2_err_raw"])

        # T3: relation name frequency + count range
        t3_rel_freq: Counter = Counter()
        t3_rel_counts: list[int] = []
        for r in rows:
            rels = [rel.strip() for rel in r["t3_relation_names"].split(";") if rel.strip()]
            t3_rel_freq.update(rels)
            try:
                t3_rel_counts.append(int(r["t3_relations_tested"]))
            except (TypeError, ValueError):
                pass

        per_kernel[kernel] = {
            "n_valid_trials":      n,
            "t1_label_freq":       t1_label_freq,
            "t1_assertion_counts": t1_assertion_counts,
            "t2_obs_freq":         t2_obs_freq,
            "t2_errors":           t2_errors,
            "t3_rel_freq":         t3_rel_freq,
            "t3_rel_counts":       t3_rel_counts,
            "judge_valid_count":   sum(
                1 for r in rows if r.get("judge_verdict") == "VALID"
            ),
        }

    log.info(
        "Verification evidence: %d verified-valid best codes across %d kernel(s).",
        len(per_trial), len(per_kernel),
    )
    return {"per_trial": per_trial, "per_kernel": per_kernel}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results: dict, output_dir: str) -> None:
    """Write CSV and text summary; print narrative to stdout."""
    if not results:
        log.warning("verification_evidence: no results to report.")
        return

    os.makedirs(output_dir, exist_ok=True)
    per_trial  = results.get("per_trial", [])
    per_kernel = results.get("per_kernel", {})

    # --- CSV: one row per trial (drop internal raw float column) ---
    if per_trial:
        csv_cols = [k for k in per_trial[0] if not k.startswith("_")]
        df_out = pd.DataFrame(per_trial)[csv_cols]
        csv_path = os.path.join(output_dir, "verification_evidence_summary.csv")
        df_out.to_csv(csv_path, index=False)
        log.info("Wrote %s", csv_path)

    # --- Formatted per-kernel narrative ---
    lines: list[str] = []

    for kernel, agg in sorted(per_kernel.items()):
        n  = agg["n_valid_trials"]
        lines.append(f"\n{'='*60}")
        lines.append(f"  {kernel}  ({n} verified-valid best codes)")
        lines.append(f"{'='*60}")

        # Tier 1
        t1_freq   = agg["t1_label_freq"]
        t1_counts = agg["t1_assertion_counts"]
        t1_pass   = sum(
            1 for r in per_trial
            if r["kernel_name"] == kernel and r["t1_verdict"] == "PASS"
        )
        lines.append("\nTier 1 — Invariant Assertion Checking")
        lines.append(f"  Formal analogue : Owicki-Gries auxiliary variable checks")
        if t1_counts:
            lines.append(f"  Assertions/code : {_range_str(t1_counts, '.0f')}")
        if t1_freq:
            # Top labels with per-trial frequency
            top_labels = t1_freq.most_common(8)
            label_strs = [f"{lbl} ({cnt}/{n})" for lbl, cnt in top_labels]
            suffix = f"  [+{len(t1_freq) - 8} more across trials]" if len(t1_freq) > 8 else ""
            lines.append(f"  Labels (top by frequency):")
            # Wrap long label list at ~80 chars
            line_buf, indent = "    ", "    "
            for s in label_strs:
                if len(line_buf) + len(s) + 2 > 78:
                    lines.append(line_buf.rstrip(", "))
                    line_buf = indent
                line_buf += s + ", "
            if line_buf.strip(", "):
                lines.append(line_buf.rstrip(", ") + suffix)
        lines.append(f"  Result          : {t1_pass}/{n} PASS")

        # Tier 2
        t2_obs   = agg["t2_obs_freq"]
        t2_errs  = agg["t2_errors"]
        t2_pass  = sum(
            1 for r in per_trial
            if r["kernel_name"] == kernel and r["t2_verdict"] == "PASS"
        )
        lines.append("\nTier 2 — Passthrough Equivalence")
        lines.append(f"  Formal analogue : Observational equivalence / forward simulation")
        if t2_obs:
            obs_strs = [f"{obs} ({cnt}/{n})" for obs, cnt in t2_obs.most_common()]
            lines.append(f"  Observation pts : {', '.join(obs_strs)}")
        lines.append(f"  Max rel. error  : {_error_range_str(t2_errs)}")
        lines.append(f"  Result          : {t2_pass}/{n} PASS")

        # Tier 3
        t3_rel    = agg["t3_rel_freq"]
        t3_counts = agg["t3_rel_counts"]
        t3_pass   = sum(
            1 for r in per_trial
            if r["kernel_name"] == kernel and r["t3_verdict"] == "PASS"
        )
        lines.append("\nTier 3 — Metamorphic Testing")
        lines.append(f"  Formal analogue : Relational Hoare Logic / property-based testing")
        if t3_counts:
            lines.append(f"  Relations/trial : {_range_str(t3_counts, '.0f')}")
        if t3_rel:
            lines.append(f"  Relations tested (by frequency):")
            # Align counts column: pad relation name to consistent width
            max_rel_len = max(len(rel) for rel, _ in t3_rel.most_common())
            for rel, cnt in t3_rel.most_common():
                lines.append(f"    {rel:<{max_rel_len}}  ({cnt}/{n})")
        lines.append(f"  Result          : {t3_pass}/{n} PASS")

        # Judge agreement
        jv = agg["judge_valid_count"]
        discrepancies = n - jv
        disc_note = (
            "judge agreed on all"
            if discrepancies == 0
            else f"{discrepancies} discrepanc{'y' if discrepancies == 1 else 'ies'}: "
                 f"judge said INVALID on a tier-verified code (false negative)"
        )
        lines.append(f"\nLLM Judge        : {jv}/{n} VALID  ({disc_note})")
        lines.append("")

    summary = "\n".join(lines)
    print(summary)

    txt_path = os.path.join(output_dir, "verification_evidence_summary.txt")
    with open(txt_path, "w") as fh:
        fh.write(summary)
    log.info("Wrote %s", txt_path)
    print(f"Verification evidence summary saved to {txt_path}")
