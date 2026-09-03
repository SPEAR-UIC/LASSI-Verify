"""No-regression check for the buffer_classification Tier-2 wiring fix
(planning/LASSI_Verify_Phase2_Bridge.md Step 1 wiring, 2026-08-27 session).

Recompiles+reruns every archived instrumented SOURCE_verify and variant `.cu` file for
every kernel that has a `{kernel}_buffer_classification.json` sidecar, from the real SC26
campaign directory recorded in that file's own `_meta.exp_dir` -- no LLM calls, no new
experiment/pipeline run, same "standalone compile/run against already-archived code"
pattern lassi_verify_analysis/tier2_regrade.py already uses. For every (trial, version)
pair, computes the Tier-2 verdict twice: buffer_classification=None (pre-fix behavior) vs.
loaded (post-fix behavior). A regression is any case where the old verdict was PASS and the
new verdict is FAIL -- this is the actual gate for whether the fix is safe to land, not a
formality: every currently-classified buffer across all 16 kernels is "exact"
(classification_summary.csv), including several float/double/`real` buffers where a
legitimately-reassociated (but correct) generated variant could plausibly diverge from
source by a sub-ULP amount that the old type-derived tolerance forgave but bit-exact
grading would not.

Usage:
    python -m lassi_verify_analysis.buffer_classification_regrade [--kernel KERNEL ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile

from lassi.verify import compare_passthrough_outputs, load_buffer_classification
from lassi_verify_analysis.oracle_harness import KERNEL_CONFIGS, compile_cu

CLASSIFICATION_DIR = os.path.join(os.path.dirname(__file__), "buffer_classification")
EXPERIMENT_ROOT = "generated_codes/experiment"
RUN_TIMEOUT_S = 600


def _find_exp_dir(exp_dir_name: str) -> str | None:
    matches = glob.glob(os.path.join(EXPERIMENT_ROOT, "**", exp_dir_name), recursive=True)
    return matches[0] if matches else None


def _recover_seed(exp_dir: str, trial: int) -> str | None:
    matches = glob.glob(os.path.join(exp_dir, f"result_verification_log_*Trial{trial}.json"))
    if not matches:
        return None
    try:
        return json.load(open(matches[0])).get("witness_chain_seed")
    except (OSError, json.JSONDecodeError):
        return None


def _find_source(exp_dir: str, trial: int) -> str | None:
    cands = [
        p for p in glob.glob(os.path.join(exp_dir, f"*_verify_Trial{trial}.cu"))
        if "basellm" not in os.path.basename(p).lower()
    ]
    return cands[0] if cands else None


def _find_variants(exp_dir: str, trial: int) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for p in glob.glob(os.path.join(exp_dir, f"*_v*_Trial{trial}.cu")):
        name = os.path.basename(p)
        if "baseLLM" in name:
            continue
        try:
            vtag = name.rsplit("_Trial", 1)[0].rsplit("_v", 1)[1]
            version = int(vtag)
        except (IndexError, ValueError):
            continue
        out.append((version, p))
    return sorted(out)


def _run_seeded(exe: str, args: list[str], seed: str | None) -> tuple[bool, str, str]:
    env = dict(os.environ)
    if seed:
        env["LASSI_VERIFY_SEED"] = seed
    try:
        p = subprocess.run([exe, *args], capture_output=True, text=True,
                           timeout=RUN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return False, "", "run_timeout"
    return (p.returncode == 0), p.stdout, p.stderr


def check_kernel(kernel: str, trials: tuple[int, ...] = (1, 2, 3)) -> list[dict]:
    """Re-grade every archived (trial, version) pair for one kernel. Returns row dicts."""
    class_file = os.path.join(CLASSIFICATION_DIR, f"{kernel}_buffer_classification.json")
    meta = json.load(open(class_file))["_meta"]
    exp_dir = _find_exp_dir(meta["exp_dir"])
    if exp_dir is None:
        return [{"kernel": kernel, "trial": None, "version": None,
                  "status": f"exp_dir_not_found:{meta['exp_dir']}"}]

    classification = load_buffer_classification(kernel)
    cfg = KERNEL_CONFIGS.get(kernel)
    if cfg is None:
        return [{"kernel": kernel, "trial": None, "version": None,
                  "status": "no_kernel_oracle_config"}]

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as wd:
        for trial in trials:
            src = _find_source(exp_dir, trial)
            if src is None:
                continue
            seed = _recover_seed(exp_dir, trial)

            sexe = os.path.join(wd, f"src_t{trial}")
            sok, serr = compile_cu(src, sexe, [])
            if not sok:
                rows.append({"kernel": kernel, "trial": trial, "version": None,
                             "status": f"source_compile_fail:{serr[:120]}"})
                continue
            srun_ok, source_stdout, srerr = _run_seeded(sexe, cfg.run_args, seed)
            if not srun_ok:
                rows.append({"kernel": kernel, "trial": trial, "version": None,
                             "status": f"source_run_fail:{srerr[:120]}"})
                continue

            for version, vpath in _find_variants(exp_dir, trial):
                vexe = os.path.join(wd, f"v_t{trial}_v{version}")
                cok, cerr = compile_cu(vpath, vexe, [])
                if not cok:
                    rows.append({"kernel": kernel, "trial": trial, "version": version,
                                 "status": f"variant_compile_fail:{cerr[:120]}"})
                    continue
                vrun_ok, variant_stdout, verr = _run_seeded(vexe, cfg.run_args, seed)
                if not vrun_ok:
                    rows.append({"kernel": kernel, "trial": trial, "version": version,
                                 "status": f"variant_run_fail:{verr[:120]}"})
                    continue

                old_verdict, _ = compare_passthrough_outputs(
                    source_stdout, variant_stdout, n_safety_override=None)
                new_verdict, new_diag = compare_passthrough_outputs(
                    source_stdout, variant_stdout, n_safety_override=None,
                    buffer_classification=classification)

                regression = (old_verdict == "PASS" and new_verdict == "FAIL")
                rows.append({
                    "kernel": kernel, "trial": trial, "version": version, "status": "ok",
                    "old_verdict": old_verdict, "new_verdict": new_verdict,
                    "regression": regression,
                    "new_failing_labels": sorted({
                        f.get("label", "").split("#")[0] for f in new_diag.get("failures", [])
                    }) if regression else [],
                })
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", nargs="+", default=None,
                     help="restrict to these kernels (default: every classified kernel)")
    args = ap.parse_args(argv)

    all_files = sorted(glob.glob(os.path.join(CLASSIFICATION_DIR, "*_buffer_classification.json")))
    kernels = args.kernel or [
        os.path.basename(p)[: -len("_buffer_classification.json")] for p in all_files
    ]

    all_rows: list[dict] = []
    for kernel in kernels:
        print(f"\n--- re-grading {kernel} ---")
        rows = check_kernel(kernel)
        all_rows.extend(rows)
        for r in rows:
            if r.get("status") != "ok":
                print(f"  [skip] trial={r.get('trial')} v={r.get('version')}: {r.get('status')}")
            else:
                tag = "REGRESSION" if r["regression"] else "ok"
                print(f"  trial={r['trial']} v={r['version']}: "
                      f"{r['old_verdict']} -> {r['new_verdict']}  [{tag}]"
                      + (f"  labels={r['new_failing_labels']}" if r["regression"] else ""))

    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    regressions = [r for r in ok_rows if r["regression"]]
    print(f"\n==== SUMMARY: {len(ok_rows)} variant comparisons graded, "
          f"{len(regressions)} regression(s) (old=PASS -> new=FAIL) ====")
    for r in regressions:
        print(f"  !! {r['kernel']} trial={r['trial']} v={r['version']} "
              f"labels={r['new_failing_labels']}")

    out_path = os.path.join(CLASSIFICATION_DIR, "no_regression_check_results.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
