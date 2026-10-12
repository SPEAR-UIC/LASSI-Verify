"""
Oracle-anchored confusion matrices (plan §7, T-A / T-B).

Joins the independent oracle verdicts (`oracle_results.csv`, from oracle_harness.py) to the
framework's own tier verdicts and the LLM judge's verdicts (`*_attempt_log.csv`), then builds
the two headline tables:

  T-A  framework verdict  × oracle   — the soundness result (false-accept ≈ 0?)
  T-B  judge verdict      × oracle   — judge false-accept / false-reject, externally adjudicated

Join keys (confirmed, plan §6): the variant filename `_v{N}_TrialM.cu` `N` equals the attempt
log's `code_block_index`; the framework/judge verdict is taken from the `is_final_attempt`,
non-zeroshot row of that (trial, code_block_index) group.

  framework_VALID ≝ tier2==PASS ∧ tier1∈{PASS,SKIP} ∧ tier3∈{PASS,SKIP}
                    (witness catches are folded into tier2 as FAIL w/ `witness_chain_state`).
  We deliberately do NOT use `final_verdict`/`override_applied` (folds in judge/override).

Only variants the oracle could actually run (verdict PASS/FAIL) enter the matrix; UNTESTABLE*
rows (strip/compile/run failures — mostly framework-INVALID witness-tampered variants) are
reported as a separate partition, per the plan.

Usage:
    python -m lassi_verify_analysis.oracle_confusion --exp-dir DIR1 [DIR2 ...]
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

TESTABLE = {"PASS", "FAIL"}


def _framework_verdict(row) -> str:
    t1 = str(row.get("tier1_verdict", "")).strip().upper()
    t2 = str(row.get("tier2_verdict", "")).strip().upper()
    t3 = str(row.get("tier3_verdict", "")).strip().upper()
    if t2 == "PASS" and t1 in ("PASS", "SKIP") and t3 in ("PASS", "SKIP"):
        return "VALID"
    return "INVALID"


def _load_one(exp_dir: str, zeroshot: bool = False) -> pd.DataFrame | None:
    """Return per-variant joined df for one experiment dir, or None if inputs missing.

    zeroshot=False (default): joins `oracle_results.csv` (self-corrected LASSI variants) to
    the non-zeroshot, final-attempt attempt-log rows — the headline 274-variant matrix.
    zeroshot=True: joins `oracle_results_zeroshot.csv` (raw `_v{N}-baseLLM` one-shot
    baselines, from `oracle_harness --generation zeroshot`) to the ZEROSHOT attempt-log rows.
    The two populations live in separate CSVs and never mix."""
    orc_name = "oracle_results_zeroshot.csv" if zeroshot else "oracle_results.csv"
    orc_p = os.path.join(exp_dir, orc_name)
    al_g = glob.glob(os.path.join(exp_dir, "*attempt_log.csv"))
    if not os.path.exists(orc_p) or not al_g:
        return None
    orc = pd.read_csv(orc_p, index_col=False)

    al = pd.read_csv(al_g[0], index_col=False)
    is_zs = al["generation_type"].astype(str).str.lower() == "zeroshot"
    al = al[(is_zs if zeroshot else ~is_zs) & (al["is_final_attempt"] == True)].copy()
    al["framework"] = al.apply(_framework_verdict, axis=1)
    al["judge"] = al["judge_baseline_verdict"].astype(str).str.strip().str.upper()
    fw = al[["trial_number", "code_block_index", "framework", "judge"]].rename(
        columns={"code_block_index": "code_version"})

    m = orc.merge(fw, on=["trial_number", "code_version"], how="left")
    return m


def _confusion(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """2x(2) crosstab of pred_col (framework/judge) vs oracle, testable rows only."""
    sub = df[df["oracle_verdict"].isin(TESTABLE) & df[pred_col].isin(["VALID", "INVALID"])]
    if sub.empty:
        return pd.DataFrame()
    ct = pd.crosstab(sub[pred_col], sub["oracle_verdict"])
    for c in ("PASS", "FAIL"):
        if c not in ct.columns:
            ct[c] = 0
    return ct[["PASS", "FAIL"]]


def _summary_line(ct: pd.DataFrame, label: str) -> list[str]:
    """Human-readable cells: false-accept (VALID+FAIL) and over-strict (INVALID+PASS)."""
    def g(r, c):
        try:
            return int(ct.loc[r, c])
        except (KeyError, TypeError):
            return 0
    fa, ta = g("VALID", "FAIL"), g("VALID", "PASS")
    fr, tr = g("INVALID", "PASS"), g("INVALID", "FAIL")
    n = fa + ta + fr + tr
    out = [f"  {label}: n={n}",
           f"    VALID : PASS(true accept)={ta}  FAIL(FALSE ACCEPT)={fa}",
           f"    INVALID: PASS(over-strict)={fr}  FAIL(true reject)={tr}"]
    return out


def run(exp_dirs: list[str], zeroshot: bool = False) -> None:
    if zeroshot:
        print("=" * 60)
        print("ZERO-SHOT population (raw _v{N}-baseLLM one-shot baselines)")
        print("=" * 60)
    frames = []
    for d in exp_dirs:
        m = _load_one(d, zeroshot=zeroshot)
        if m is None:
            miss = "oracle_results_zeroshot.csv" if zeroshot else "oracle_results.csv"
            print(f"[skip] no {miss}/attempt_log in {d}")
            continue
        kernel = m["app_name"].iloc[0] if "app_name" in m and len(m) else os.path.basename(d)
        frames.append(m)

        # per-kernel partition report
        vc = m["oracle_verdict"].value_counts().to_dict()
        untest = {k: v for k, v in vc.items() if k not in TESTABLE}
        print(f"\n===== {kernel} =====")
        print(f"  oracle tally: {vc}")
        if untest:
            # correlate UNTESTABLE with framework verdict
            u = m[~m["oracle_verdict"].isin(TESTABLE)]
            fwc = u["framework"].value_counts(dropna=False).to_dict()
            print(f"  UNTESTABLE partition ({sum(untest.values())}): framework verdict of those = {fwc}")
        ctf = _confusion(m, "framework")
        if not ctf.empty:
            print("  framework × oracle:")
            for ln in _summary_line(ctf, "T-A"):
                print(ln)

    if not frames:
        print("no data")
        return

    allm = pd.concat(frames, ignore_index=True)
    print("\n" + "=" * 60)
    print("AGGREGATE (all kernels)")
    print("=" * 60)
    ctf = _confusion(allm, "framework")
    ctj = _confusion(allm, "judge")
    print("\n[T-A] framework × oracle")
    print(ctf if not ctf.empty else "  (empty)")
    for ln in _summary_line(ctf, "T-A"):
        print(ln)
    print("\n[T-B] judge × oracle")
    print(ctj if not ctj.empty else "  (empty)")
    for ln in _summary_line(ctj, "T-B"):
        print(ln)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Oracle-anchored confusion matrices (T-A/T-B)")
    ap.add_argument("--exp-dir", nargs="+", required=True)
    ap.add_argument("--zeroshot", action="store_true",
                    help="grade the raw zero-shot baseline population (oracle_results_zeroshot.csv "
                         "× zeroshot attempt-log rows) instead of the self-corrected LASSI variants")
    a = ap.parse_args(argv)
    run(a.exp_dir, zeroshot=a.zeroshot)


if __name__ == "__main__":
    main()
