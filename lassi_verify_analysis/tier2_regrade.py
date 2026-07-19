"""Tier-2 re-grade harness (Refinement #8 impact measurement).

Recomputes each on-disk instrumented variant's Tier-2 verdict under BOTH the
historical strictly-positional comparison and the order-insensitive bijection
match, so we can quantify the false-reject (completeness) impact of Refinement #8
WITHOUT a full pipeline re-run (no LLM calls, no self-correction loops).

Method (faithful to the live framework):
  * SOURCE spec  = the run's saved instrumented SOURCE_verify  `*_verify_Trial{N}.cu`.
  * variants     = the run's saved instrumented `*_v{V}_Trial{N}.cu`.
  * Both are recompiled and executed with the SAME recorded witness-chain seed
    (`witness_chain_seed` from the trial's verification-log JSON) and the SAME
    framework workload (`KernelOracleConfig.run_args`), then compared with the
    framework's OWN `lassi.verify.compare_passthrough_outputs()` — the single
    source of truth — once with order_insensitive=False (the pre-refinement
    baseline) and once with order_insensitive=True (post-refinement).

We never modify a HeCBench kernel and never re-instrument: we replay exactly the
artifacts the pipeline already produced. The oracle verdict (independent ground
truth) is joined from the existing `oracle_results.csv`; the framework's recorded
Tier-2 verdict is joined from the verification-log JSON as a reproduction check
(positional re-run should reproduce it).

Usage:
    python -m lassi_verify_analysis.tier2_regrade --kernel extrema \
        --exp-dir generated_codes/experiment/KERNEL_TEST_EXTREMA_WC_3__* \
        [--out tier2_regrade.csv]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile

import pandas as pd

from lassi.verify import compare_passthrough_outputs
from lassi_verify_analysis.oracle_harness import (
    KERNEL_CONFIGS,
    RUN_TIMEOUT_S,
    compile_cu,
)


def _recover_seed(exp_dir: str, trial: int) -> str | None:
    """Return the witness-chain seed hex recorded for this trial, or None."""
    matches = glob.glob(os.path.join(exp_dir, f"result_verification_log_*Trial{trial}.json"))
    if not matches:
        return None
    try:
        return json.load(open(matches[0])).get("witness_chain_seed")
    except (json.JSONDecodeError, OSError):
        return None


def _recorded_tier2(exp_dir: str, trial: int) -> dict[int, str]:
    """Map code_version -> recorded Tier-2 verdict from the trial's verification log.
    Used only as a reproduction cross-check (our positional re-run should match)."""
    out: dict[int, str] = {}
    matches = glob.glob(os.path.join(exp_dir, f"result_verification_log_*Trial{trial}.json"))
    if not matches:
        return out
    try:
        d = json.load(open(matches[0]))
    except (json.JSONDecodeError, OSError):
        return out
    for row in d.get("tier2_results", []) or []:
        cv = row.get("code_version")
        if cv is not None:
            out[int(cv)] = str(row.get("verdict", "")).strip().upper()
    return out


def _find_source(exp_dir: str, trial: int) -> str | None:
    """Return the instrumented SOURCE_verify .cu for this trial (excludes the
    zero-shot `basellm_zs_verify` source)."""
    cands = [
        p for p in glob.glob(os.path.join(exp_dir, f"*_verify_Trial{trial}.cu"))
        if "basellm" not in os.path.basename(p).lower()
    ]
    return cands[0] if cands else None


def _find_variants(exp_dir: str, trial: int) -> list[tuple[int, str]]:
    """Return sorted (version, path) for `*_v{V}_Trial{trial}.cu` (excludes
    `-baseLLM` zero-shot variants)."""
    out: list[tuple[int, str]] = []
    for p in glob.glob(os.path.join(exp_dir, f"*_v*_Trial{trial}.cu")):
        name = os.path.basename(p)
        if "baseLLM" in name:
            continue
        # version = the integer between the last "_v" and "_Trial"
        try:
            vtag = name.rsplit("_Trial", 1)[0].rsplit("_v", 1)[1]
            version = int(vtag)
        except (IndexError, ValueError):
            continue
        out.append((version, p))
    return sorted(out)


def _run_seeded(exe: str, args: list[str], seed: str | None) -> tuple[bool, str, str]:
    """Run the instrumented binary with LASSI_VERIFY_SEED set (so witness-chain
    hashes are reproducible), returning (ok, stdout, stderr)."""
    env = dict(os.environ)
    if seed:
        env["LASSI_VERIFY_SEED"] = seed
    try:
        p = subprocess.run([exe, *args], capture_output=True, text=True,
                           timeout=RUN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return False, "", "run_timeout"
    return (p.returncode == 0), p.stdout, p.stderr


def _load_oracle(exp_dir: str) -> dict[tuple[int, int], str]:
    """Map (trial, code_version) -> oracle_verdict from oracle_results.csv."""
    path = os.path.join(exp_dir, "oracle_results.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        (int(r.trial_number), int(r.code_version)): str(r.oracle_verdict)
        for r in df.itertuples()
        if pd.notna(r.trial_number) and pd.notna(r.code_version)
    }


def regrade_dir(kernel: str, exp_dir: str, trials: range = range(1, 4),
                compile_extra: list[str] | None = None) -> list[dict]:
    """Re-grade every instrumented variant in exp_dir. Returns a list of row dicts.

    compile_extra overrides the kernel's oracle compile flags. Pass [] to drop
    oracle-only flags (e.g. pathfinder's -DBENCH_PRINT wall dump) that are not part of
    the framework's own Tier-2 instrumentation build."""
    cfg = KERNEL_CONFIGS[kernel]
    cextra = cfg.compile_extra if compile_extra is None else compile_extra
    oracle = _load_oracle(exp_dir)
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as wd:
        for trial in trials:
            src = _find_source(exp_dir, trial)
            if src is None:
                continue
            seed = _recover_seed(exp_dir, trial)
            recorded = _recorded_tier2(exp_dir, trial)

            # Compile + run the instrumented source once per trial (cache its stdout).
            sexe = os.path.join(wd, f"src_t{trial}.exe")
            sok, serr = compile_cu(src, sexe, cextra)
            if not sok:
                rows.append({"app_name": kernel, "exp_dir": os.path.basename(exp_dir),
                             "trial": trial, "version": None, "status": f"source_compile_fail:{serr[:80]}"})
                continue
            rok, source_stdout, rerr = _run_seeded(sexe, cfg.run_args, seed)
            if not rok:
                rows.append({"app_name": kernel, "exp_dir": os.path.basename(exp_dir),
                             "trial": trial, "version": None, "status": f"source_run_fail:{rerr[:80]}"})
                continue

            for version, vpath in _find_variants(exp_dir, trial):
                vexe = os.path.join(wd, f"v_t{trial}_v{version}.exe")
                cok, cerr = compile_cu(vpath, vexe, cextra)
                if not cok:
                    rows.append({"app_name": kernel, "exp_dir": os.path.basename(exp_dir),
                                 "trial": trial, "version": version, "status": "compile_fail"})
                    continue
                vok, variant_stdout, verr = _run_seeded(vexe, cfg.run_args, seed)
                if not vok:
                    rows.append({"app_name": kernel, "exp_dir": os.path.basename(exp_dir),
                                 "trial": trial, "version": version, "status": "run_fail"})
                    continue

                t2_pos, _ = compare_passthrough_outputs(
                    source_stdout, variant_stdout,
                    n_safety_override=None, order_insensitive=False)
                t2_ord, _ = compare_passthrough_outputs(
                    source_stdout, variant_stdout,
                    n_safety_override=None, order_insensitive=True)

                oracle_v = oracle.get((trial, version), "UNKNOWN")
                rows.append({
                    "app_name": kernel,
                    "exp_dir": os.path.basename(exp_dir),
                    "trial": trial,
                    "version": version,
                    "status": "ok",
                    "t2_positional": t2_pos,       # pre-Refinement-#8 baseline
                    "t2_orderins": t2_ord,         # post-Refinement-#8
                    "rescued": (t2_pos == "FAIL" and t2_ord == "PASS"),
                    "oracle_verdict": oracle_v,
                    "recorded_t2": recorded.get(version, ""),  # reproduction cross-check
                })
    return rows


def _summarize(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy() if "status" in df else pd.DataFrame()
    print(f"\n==== Tier-2 re-grade summary  (n_rows={len(df)}, testable={len(ok)}) ====")
    if ok.empty:
        nonok = df[df["status"] != "ok"] if "status" in df else df
        for _, r in nonok.iterrows():
            print(f"  [skip] trial={r.get('trial')} v={r.get('version')}: {r.get('status')}")
        return

    rescued = ok[ok["rescued"]]
    print(f"  positional PASS: {(ok['t2_positional']=='PASS').sum():3d}   "
          f"order-insensitive PASS: {(ok['t2_orderins']=='PASS').sum():3d}")
    print(f"  RESCUED (positional FAIL -> order-insensitive PASS): {len(rescued)}")

    # Soundness guard: a rescue is only legitimate if the oracle agrees the variant
    # is correct. A rescue on an oracle-FAIL variant would be a NEW false-accept.
    bad = rescued[rescued["oracle_verdict"] == "FAIL"]
    print(f"    of which oracle-PASS (legit false-reject recovery): "
          f"{(rescued['oracle_verdict']=='PASS').sum()}")
    print(f"    of which oracle-FAIL (**NEW FALSE ACCEPT — must be 0**): {len(bad)}")
    if len(bad):
        for _, r in bad.iterrows():
            print(f"      !! trial={r.trial} v={r.version} oracle=FAIL rescued to PASS")

    # Reproduction cross-check: our positional re-run should match the framework's
    # recorded Tier-2 verdict (validates the harness faithfully replays the pipeline).
    have_rec = ok[ok["recorded_t2"].isin(["PASS", "FAIL"])]
    if len(have_rec):
        agree = (have_rec["t2_positional"] == have_rec["recorded_t2"]).sum()
        print(f"  reproduction check: positional re-run matches recorded Tier-2 "
              f"{agree}/{len(have_rec)}")
        mism = have_rec[have_rec["t2_positional"] != have_rec["recorded_t2"]]
        for _, r in mism.iterrows():
            print(f"      mismatch trial={r.trial} v={r.version}: "
                  f"rerun={r.t2_positional} recorded={r.recorded_t2}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier-2 order-insensitive re-grade (Refinement #8)")
    ap.add_argument("--kernel", required=True, choices=sorted(KERNEL_CONFIGS.keys()))
    ap.add_argument("--exp-dir", nargs="+", required=True,
                    help="one or more experiment folders (globs allowed)")
    ap.add_argument("--out", default=None, help="CSV path to write per-variant rows")
    args = ap.parse_args(argv)

    exp_dirs: list[str] = []
    for pat in args.exp_dir:
        exp_dirs.extend(sorted(glob.glob(pat)) or [pat])

    all_rows: list[dict] = []
    for d in exp_dirs:
        if not os.path.isdir(d):
            print(f"  [warn] not a directory: {d}")
            continue
        print(f"\n--- re-grading {args.kernel} in {os.path.basename(d)} ---")
        rows = regrade_dir(args.kernel, d)
        all_rows.extend(rows)

    _summarize(all_rows)

    if args.out and all_rows:
        pd.DataFrame(all_rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
