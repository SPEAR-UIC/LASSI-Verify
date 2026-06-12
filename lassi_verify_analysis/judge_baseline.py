"""
Judge-baseline vs. framework comparison.

Exports the per-attempt (judge_baseline_verdict, tier verdicts, final_verdict)
log that the pipeline writes into each trial's verification_log JSON sidecar
as a flat CSV. Downstream consumers (paper figures, narrative tables) can
read this single artifact instead of walking every sidecar.

The judge-as-baseline design (planning/fix_judge_ordering_plan.md) records
one discrepancy entry per attempt that reached the three-tier gate — i.e.,
every attempt whose generated code compiled, executed, and got an LLM-judge
verdict before tier analysis ran. Compile/execute failures don't reach the
judge and therefore don't appear here.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re

import pandas as pd

log = logging.getLogger(__name__)

# Per-attempt discrepancy columns, in the order written to CSV.
DISCREPANCY_COLUMNS = [
    "kernel_name",
    "app_name",
    "trial_number",
    "code_version",
    "zero_shot",
    "judge_baseline_verdict",
    "tier1_verdict",
    "tier2_verdict",
    "tier3_verdict",
    "final_verdict",
]

# Filename pattern for verification-log JSON sidecars; trial number is captured.
_SIDECAR_PATTERN = re.compile(
    r"result_verification_log_(?P<app_name>.+?)_Trial(?P<trial>\d+)\.json$"
)

# Strip "-cuda_main..." suffix from app_name to recover kernel_name.
# Matches the rule in data_loader._derive_kernel_name; duplicated here to keep
# this module's dependency surface tight (no private import).
_APP_TO_KERNEL_RE = re.compile(r"-cuda_main.*$")


def load_discrepancies(data_dir: str) -> pd.DataFrame:
    """Discover and load all judge-discrepancy entries under *data_dir*.

    Walks the directory tree for files matching
    ``result_verification_log_<app_name>_Trial<N>.json``, reads the
    ``judge_discrepancies`` list from each, and flattens to one row per attempt.

    Args:
        data_dir: Path containing per-trial sidecars (directly or in subdirs).

    Returns:
        A DataFrame with columns from :data:`DISCREPANCY_COLUMNS`. Empty (with
        the expected columns set) if no sidecars are found or none contain
        discrepancy entries.

    Warns (via logging) on sidecars whose ``judge_discrepancies`` field is
    absent (e.g., pre-judge-as-baseline runs).
    """
    sidecars = sorted(glob.glob(os.path.join(data_dir, "**", "result_verification_log_*.json"),
                                recursive=True))
    if not sidecars:
        log.warning("No verification_log_*.json sidecars found under %s.", data_dir)
        return pd.DataFrame(columns=DISCREPANCY_COLUMNS)

    rows: list[dict] = []
    missing_field = 0
    legacy_schema = 0
    for path in sidecars:
        name = os.path.basename(path)
        match = _SIDECAR_PATTERN.search(name)
        if not match:
            log.debug("Skipping sidecar with unexpected name: %s", name)
            continue
        app_name = match.group("app_name")
        trial = int(match.group("trial"))
        kernel_name = _APP_TO_KERNEL_RE.sub("", app_name)

        try:
            with open(path, "r") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Failed to read %s: %s", path, exc)
            continue

        discs = doc.get("judge_discrepancies")
        if discs is None:
            missing_field += 1
            continue

        for entry in discs:
            # Legacy entries from before the judge-as-baseline rework used a
            # `type` field ("false_positive"/"false_negative") and the old
            # `judge_verdict` key. Skip those rather than mix schemas.
            if "type" in entry or "judge_verdict" in entry:
                legacy_schema += 1
                continue
            rows.append({
                "kernel_name":             kernel_name,
                "app_name":                app_name,
                "trial_number":            trial,
                "code_version":            entry.get("code_version"),
                "zero_shot":               entry.get("zero_shot"),
                "judge_baseline_verdict":  entry.get("judge_baseline_verdict"),
                "tier1_verdict":           entry.get("tier1_verdict"),
                "tier2_verdict":           entry.get("tier2_verdict"),
                "tier3_verdict":           entry.get("tier3_verdict"),
                "final_verdict":           entry.get("final_verdict"),
            })

    if missing_field:
        log.warning("%d sidecar(s) had no `judge_discrepancies` field (likely "
                    "pre-judge-as-baseline schema) — skipped.", missing_field)
    if legacy_schema:
        log.warning("%d entry/entries used the pre-judge-as-baseline schema "
                    "(had `type` or `judge_verdict` key) — skipped.", legacy_schema)

    df = pd.DataFrame(rows, columns=DISCREPANCY_COLUMNS)
    log.info("Loaded %d discrepancy entries from %d sidecar(s) under %s.",
             len(df), len(sidecars), data_dir)
    return df


def summarize(df_disc: pd.DataFrame) -> dict:
    """Aggregate FP/FN/TP/TN counts from the discrepancy DataFrame.

    Computes counts both overall and per kernel. Cell definitions match the
    pipeline-side recomputation in pipeline_v4.py (judge-as-baseline design):
      - TP: judge VALID and framework VALID  (judge agrees with framework, both pass)
      - FN: judge INVALID and framework VALID  (judge too strict; framework recovers)
      - FP: judge VALID and framework INVALID  (judge missed a real bug)
      - TN: judge INVALID and framework INVALID  (judge agrees with framework, both fail)

    Returns:
        Dict shaped::

            {
              "overall": {"tp": n, "fn": n, "fp": n, "tn": n, "total": n,
                          "agreement_rate": float, "judge_recall": float|None},
              "per_kernel": {kernel_name: {...same keys...}, ...},
            }
    """
    def _cells(sub: pd.DataFrame) -> dict:
        tp = int(((sub["judge_baseline_verdict"] == "VALID") & (sub["final_verdict"] == "VALID")).sum())
        fn = int(((sub["judge_baseline_verdict"] == "INVALID") & (sub["final_verdict"] == "VALID")).sum())
        fp = int(((sub["judge_baseline_verdict"] == "VALID") & (sub["final_verdict"] == "INVALID")).sum())
        tn = int(((sub["judge_baseline_verdict"] == "INVALID") & (sub["final_verdict"] == "INVALID")).sum())
        total = tp + fn + fp + tn
        agreement = (tp + tn) / total if total else None
        recall = tp / (tp + fn) if (tp + fn) else None  # judge VALID rate among truly-VALID
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "total": total,
                "agreement_rate": agreement, "judge_recall": recall}

    overall = _cells(df_disc)
    per_kernel = {k: _cells(df_disc[df_disc["kernel_name"] == k])
                  for k in sorted(df_disc["kernel_name"].dropna().unique().tolist())}
    return {"overall": overall, "per_kernel": per_kernel}


def report(df_disc: pd.DataFrame, output_dir: str,
           filename: str = "judge_baseline_vs_framework.csv") -> str:
    """Write the per-attempt discrepancy CSV and print a short summary table.

    Args:
        df_disc: DataFrame from :func:`load_discrepancies`.
        output_dir: Destination directory (created if missing).
        filename: CSV filename. Default ``"judge_baseline_vs_framework.csv"``.

    Returns:
        Absolute path to the written CSV (empty string if df_disc is empty).
    """
    if df_disc.empty:
        log.warning("Discrepancy DataFrame is empty — nothing to write.")
        return ""

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.abspath(os.path.join(output_dir, filename))
    df_disc.to_csv(out_path, index=False)
    log.info("Wrote %d discrepancy rows to %s", len(df_disc), out_path)

    summary = summarize(df_disc)
    print("\n=== Judge-baseline vs. framework agreement (per-attempt) ===")
    print(f"Wrote: {out_path} ({len(df_disc)} rows)")
    ov = summary["overall"]
    print(f"\nOverall (n={ov['total']}):")
    print(f"  TP (judge VALID + framework VALID):     {ov['tp']}")
    print(f"  TN (judge INVALID + framework INVALID): {ov['tn']}")
    print(f"  FN (judge INVALID + framework VALID):   {ov['fn']}  "
          f"(judge too strict; framework recovers)")
    print(f"  FP (judge VALID + framework INVALID):   {ov['fp']}  "
          f"(judge missed a real bug)")
    if ov['agreement_rate'] is not None:
        print(f"  Agreement rate: {ov['agreement_rate']:.1%}")
    if ov['judge_recall'] is not None:
        print(f"  Judge recall (judge VALID among framework-VALID): {ov['judge_recall']:.1%}")

    if summary["per_kernel"]:
        print(f"\nPer kernel:")
        for k, c in summary["per_kernel"].items():
            agree = f"{c['agreement_rate']:.1%}" if c['agreement_rate'] is not None else "N/A"
            print(f"  {k:>30}: n={c['total']:>3}  TP={c['tp']:>3} TN={c['tn']:>3} "
                  f"FN={c['fn']:>3} FP={c['fp']:>3}  agreement={agree}")

    return out_path
