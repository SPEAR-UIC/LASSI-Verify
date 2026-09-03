"""Tier-2 buffer stability classification (Phase 2 Bridge, Step 1 checkpoint).

Builds the per-buffer stability evidence Step 1 of
`planning/LASSI_Verify_Phase2_Bridge.md` calls for: for each PASSTHROUGH label a kernel's
SOURCE_verify binary emits, is that label's Statistics/Sample content bit-stable across
repeated seeded re-executions ("exact" candidate), or does it wander ("tolerant"
candidate, evidence-backed by the observed divergence)?

Deliberately independent of the live pipeline (no changes to pipeline_v4.py / verify.py's
comparison logic): this script does its own dedicated re-execution runs, reusing the same
primitives the existing regrade/oracle scripts already use.

Two phases:
  Phase A (--spot-check): one seeded run per kernel, timed. Produces
      buffer_classification/timing_spotcheck.csv
  Phase B (--classify):   dedicated stability sub-runs up to a target n per kernel
      (default n derived from --target-p, or pass --n explicitly per kernel via
      --n-overrides). Produces one sidecar JSON per kernel plus a combined
      buffer_classification/classification_summary.csv

Source resolution is driven by lassi_verify_analysis/oracle_confusion_matrix_cumulative.csv
(the authoritative record of which archived experiment directory backs each kernel's final
SC26 numbers), not a hand-maintained table, so it stays correct if that CSV is regenerated.

Usage:
    python -m lassi_verify_analysis.buffer_stability_classification --spot-check
    python -m lassi_verify_analysis.buffer_stability_classification --classify \
        --target-p 0.02 --time-cap-s 600
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import tempfile
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lassi.verify import compare_passthrough_outputs, _compute_fp_tolerance  # noqa: E402
from lassi_verify_analysis.oracle_harness import KERNEL_CONFIGS, compile_cu  # noqa: E402
from lassi_verify_analysis.tier2_regrade import _find_source, _find_variants, _recover_seed, _run_seeded  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.join(REPO_ROOT, "generated_codes", "experiment")
CONFUSION_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "oracle_confusion_matrix_cumulative.csv")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buffer_classification")

# jacobi excluded per standing project convention; matrix-rotate held back for Step 3.
EXCLUDED_KERNELS = {"jacobi", "matrix-rotate"}

WITNESS_CHAIN_LABEL = "witness_chain_state"

_BLOCK_PATTERN = re.compile(
    r'=== BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\] ===\n(.*?)\n=== END LASSI-VERIFY PASSTHROUGH \[[^\]]+\] ===',
    re.DOTALL
)
_STATS_PATTERN = re.compile(r'Statistics:\s*min=([^,\s]+),\s*max=([^,\s]+),\s*mean=([^,\s]+)')


def _kernel_population() -> list[str]:
    df = pd.read_csv(CONFUSION_CSV)
    kernels = sorted(k for k in df["kernel"].dropna().unique() if k not in EXCLUDED_KERNELS)
    return kernels


def _candidate_exp_dirs(kernel: str) -> list[str]:
    """Resolve on-disk experiment directory candidates for a kernel from the confusion
    matrix CSV's exp_dir column. Handles the floydwarshall-style combined "A+B" entry."""
    df = pd.read_csv(CONFUSION_CSV)
    rows = df[df["kernel"] == kernel]
    dirs: list[str] = []
    for raw in rows["exp_dir"].dropna().unique():
        for part in str(raw).split("+"):
            part = part.strip()
            matches = sorted(glob.glob(os.path.join(EXPERIMENT_DIR, part + "*")))
            dirs.extend(matches)
    seen = set()
    out = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def resolve_kernel_source(kernel: str) -> dict:
    """Find a compilable, runnable SOURCE_verify .cu for this kernel. Tries each
    candidate exp_dir's trials 1/2/3 in order, preferring a trial that actually
    COMPLETED (produced at least one real generated variant on disk) over one that merely
    left a source file behind. A trial can have a source file but zero variants if the
    live pipeline aborted it before Stage 0 finished (e.g. extrema's Trial 1, which the
    determinism gate correctly rejected in the original campaign for reasons unrelated to
    the variants that actually shipped) — picking such a trial would classify a discarded,
    possibly-buggy artifact as if it were representative. Falls back to any trial with a
    source file only if none completed (would mean no usable campaign data exists at all
    for this kernel; surfaced as a loud error below rather than silently misattributed).
    Returns a dict with exp_dir, trial, source_path, seed, cfg — or raises RuntimeError."""
    cfg = KERNEL_CONFIGS[kernel]
    fallback = None
    for exp_dir in _candidate_exp_dirs(kernel):
        for trial in (1, 2, 3):
            src = _find_source(exp_dir, trial)
            if src is None:
                continue
            candidate = {
                "exp_dir": exp_dir,
                "trial": trial,
                "source_path": src,
                "seed": _recover_seed(exp_dir, trial),
                "cfg": cfg,
            }
            if _find_variants(exp_dir, trial):
                return candidate
            if fallback is None:
                fallback = candidate
    if fallback is not None:
        print(f"  [warn] {kernel}: no trial with completed variants found — falling back "
              f"to trial {fallback['trial']} (source only, no variants) in "
              f"{os.path.basename(fallback['exp_dir'])}. This trial may have been aborted "
              f"before Stage 0 finished; treat its classification as low-confidence.")
        return fallback
    raise RuntimeError(f"no usable SOURCE_verify source found for kernel={kernel!r} "
                        f"(candidates: {_candidate_exp_dirs(kernel)})")


def _compile(source_info: dict, workdir: str) -> str:
    exe = os.path.join(workdir, "source.exe")
    ok, err = compile_cu(source_info["source_path"], exe, source_info["cfg"].compile_extra)
    if not ok:
        raise RuntimeError(f"compile failed for {source_info['source_path']}: {err[:300]}")
    return exe


_T1_FAIL_PATTERN = re.compile(r'LASSI-VERIFY ASSERTIONS \[([^\]]+)\]:\s*FAIL')


def _run_once(exe: str, source_info: dict) -> tuple[str, float]:
    """Run once, seeded. Raises if the process failed OR if any T1 assertion in its own
    stdout reports FAIL — most notably `device_allocations_valid`, which silently passes
    with all-zero/garbage buffers on a broken CUDA runtime (nvidia-smi can look healthy
    while cudaMalloc/cudaGetDeviceCount return "unknown error" — see project memory
    reference_nvidia_driver_recovery.md, recurred 2026-07-27 and 2026-08-10). A script that
    only inspects PASSTHROUGH value fields would not otherwise notice this class of failure."""
    cfg = source_info["cfg"]
    t0 = time.monotonic()
    ok, stdout, stderr = _run_seeded(exe, cfg.run_args, source_info["seed"])
    elapsed = time.monotonic() - t0
    if not ok:
        raise RuntimeError(f"run failed: {stderr[:300]}")
    t1_fails = _T1_FAIL_PATTERN.findall(stdout)
    if t1_fails:
        raise RuntimeError(f"T1 assertion FAIL in SOURCE_verify's own stdout "
                            f"({', '.join(t1_fails)}) - treating output as untrustworthy "
                            f"rather than a genuine buffer result")
    return stdout, elapsed


def _extract_labels(stdout: str) -> dict:
    """label -> list of block bodies (multi-occurrence labels keep every occurrence)."""
    result: dict = {}
    for m in _BLOCK_PATTERN.finditer(stdout):
        result.setdefault(m.group(1), []).append(m.group(2))
    return result


def _is_numeric_block(body: str) -> bool:
    """True iff the block has a Statistics: line (eligible for the stability loop)."""
    return _STATS_PATTERN.search(body) is not None


# ---------------------------------------------------------------------------
# Phase A: timing spot-check
# ---------------------------------------------------------------------------

def spot_check(kernels: list[str]) -> list[dict]:
    rows = []
    for kernel in kernels:
        print(f"\n--- spot-check: {kernel} ---")
        try:
            info = resolve_kernel_source(kernel)
            with tempfile.TemporaryDirectory() as wd:
                exe = _compile(info, wd)
                stdout, elapsed = _run_once(exe, info)
            labels = _extract_labels(stdout)
            n_numeric = sum(1 for bodies in labels.values() for b in bodies if _is_numeric_block(b))
            print(f"    exp_dir={os.path.basename(info['exp_dir'])} trial={info['trial']} "
                  f"seconds={elapsed:.2f} labels={len(labels)} numeric_blocks={n_numeric}")
            rows.append({
                "kernel": kernel, "exp_dir": os.path.basename(info["exp_dir"]),
                "trial": info["trial"], "seconds": round(elapsed, 3),
                "n_labels": len(labels), "n_numeric_blocks": n_numeric,
                "status": "ok",
            })
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")
            rows.append({"kernel": kernel, "exp_dir": "", "trial": "", "seconds": None,
                         "n_labels": None, "n_numeric_blocks": None,
                         "status": f"error: {e}"[:200]})
    return rows


# ---------------------------------------------------------------------------
# Phase B: dedicated stability sub-runs
# ---------------------------------------------------------------------------

def classify_kernel(kernel: str, target_n: int, time_cap_s: float | None) -> dict:
    """Run up to target_n seeded re-executions (or until time_cap_s elapses), tracking
    per-label divergence against run 1. Returns {label: record} per Section 9.3, plus a
    '_meta' key with achieved depth / timing / any early-stop reason."""
    info = resolve_kernel_source(kernel)
    with tempfile.TemporaryDirectory() as wd:
        exe = _compile(info, wd)
        run1_stdout, run1_elapsed = _run_once(exe, info)
        run1_labels = _extract_labels(run1_stdout)

        # Per-label state, keyed by label (multi-occurrence labels are compared as a
        # whole label via compare_passthrough_outputs, matching Tier 2's own semantics).
        state = {
            label: {
                "n_checked": 1,  # run 1 is itself one observation of "no divergence yet"
                "diverged": False,
                "max_abs_err": None,
                "max_rel_err": None,
                "is_numeric": any(_is_numeric_block(b) for b in bodies),
            }
            for label, bodies in run1_labels.items()
        }
        # Structural (non-numeric) labels, including witness_chain_state, are classified
        # exact-by-construction and excluded from the divergence loop entirely.
        pending = {l for l, s in state.items() if s["is_numeric"]}

        t_start = time.monotonic()
        run_count = 1
        stop_reason = "target_n_reached"
        while pending and run_count < target_n:
            if time_cap_s is not None and (time.monotonic() - t_start) > time_cap_s:
                stop_reason = "time_cap"
                break
            run_i_stdout, _ = _run_once(exe, info)
            run_count += 1
            _, diag = compare_passthrough_outputs(run1_stdout, run_i_stdout,
                                                   n_safety_override=None)
            failures_by_label: dict = {}
            for f in diag.get("failures", []):
                base_label = f["label"].split("#", 1)[0]
                failures_by_label.setdefault(base_label, []).append(f)

            newly_settled = set()
            for label in pending:
                s = state[label]
                s["n_checked"] += 1
                fs = failures_by_label.get(label, [])
                if fs:
                    s["diverged"] = True
                    abs_errs = [f.get("abs_err") for f in fs if isinstance(f.get("abs_err"), (int, float))]
                    rel_errs = [f.get("rel_err") for f in fs if isinstance(f.get("rel_err"), (int, float))]
                    if abs_errs:
                        s["max_abs_err"] = max([e for e in abs_errs if math.isfinite(e)], default=s["max_abs_err"])
                    if rel_errs:
                        s["max_rel_err"] = max([e for e in rel_errs if math.isfinite(e)], default=s["max_rel_err"])
                    newly_settled.add(label)  # instability is conclusive — stop checking this label
            pending -= newly_settled
        else:
            if not pending:
                stop_reason = "all_labels_settled"

        elapsed_total = time.monotonic() - t_start + run1_elapsed

    records = {}
    for label, s in state.items():
        first_body = run1_labels[label][0]
        if not s["is_numeric"]:
            records[label] = {
                "comparison_mode": "exact",
                "evidence": {"method": "structural_exact", "n_comparisons": None,
                             "observed_stable": None, "max_observed_delta": None},
                "declared_type": None,
                "declared_max_magnitude": None,
                "tolerance": None,
                "last_updated": None,
            }
            continue
        # Captured once from run 1's own block — a weak prior for calibration's scale
        # basis (Part C), never authoritative for the mode decision itself.
        type_meta = _compute_fp_tolerance(first_body)["metadata"]
        records[label] = {
            "comparison_mode": "tolerant" if s["diverged"] else "exact",
            "evidence": {
                "method": "dedicated_stability_run",
                "n_comparisons": s["n_checked"],
                "observed_stable": not s["diverged"],
                "max_observed_delta": s["max_abs_err"],
                "max_observed_relative_delta": s["max_rel_err"],
            },
            "declared_type": type_meta["type"],
            "declared_max_magnitude": type_meta["max_magnitude"],
            "tolerance": None,  # populated only for a "tolerant" record — see calibrate_tolerance()
            "last_updated": None,
        }

    records["_meta"] = {
        "kernel": kernel,
        "exp_dir": os.path.basename(info["exp_dir"]),
        "trial": info["trial"],
        "target_n": target_n,
        "runs_executed": run_count,
        "stop_reason": stop_reason,
        "elapsed_seconds": round(elapsed_total, 2),
    }
    return records


# ---------------------------------------------------------------------------
# Calibration: turn a "tolerant" mode verdict into a usable numeric tolerance
# ---------------------------------------------------------------------------

CALIBRATION_SAFETY_FACTOR = 2.0  # matches oracle_harness.py's build_source_baseline()
                                  # source-self-variance margin (`spread * 2.0`) — same
                                  # idea (measured wander x2 safety margin), same factor,
                                  # for consistency across the codebase's two independent
                                  # "measure the source, don't assume the type" mechanisms.


def calibrate_tolerance(kernel: str, label: str, n_calibration_runs: int = 10) -> dict:
    """Turn an already-settled "tolerant" classification into a usable (atol, rtol) by
    sampling the label's divergence more thoroughly than mode-detection needs.

    Deliberately separate from classify_kernel()'s early-stop loop: mode-detection only
    needs ONE divergence to conclude "tolerant" (the asymmetric rule), but calibrating a
    defensible tolerance benefits from more samples of HOW LARGE the divergence gets, not
    just that it occurs. Runs n_calibration_runs MORE seeded re-executions against a fresh
    run 1, without early-stopping, and takes the max observed absolute error across all of
    them as the basis for atol (x CALIBRATION_SAFETY_FACTOR margin, no rtol term — we have
    a directly measured absolute bound, not a reliable relative-error sample to build one
    from). Writes the result into the existing {kernel}_buffer_classification.json's
    tolerance block for this label; does not touch comparison_mode or evidence.

    This is a deliberate, explicit, reviewable step — not run automatically the moment a
    buffer settles tolerant — so a human looks at the calibrated number before it's used
    to license anything, consistent with how every other number in this workstream (n
    policy, target divergence rate) went through a review step rather than being
    auto-computed and trusted.

    UNVALIDATED against a real tolerant case as of 2026-08-11 (the only classified-tolerant
    buffer found this session, extrema, turned out to be a tooling artifact — see
    project_extrema_passthrough_capture_bug.md — not a genuine instability). Built now per
    Matthew's direction as needed infrastructure, not because a live case demanded it.
    """
    out_json = os.path.join(OUT_DIR, f"{kernel}_buffer_classification.json")
    with open(out_json) as fh:
        doc = json.load(fh)
    record = doc["buffers"].get(label)
    if record is None:
        raise ValueError(f"no classification record for {kernel}/{label} in {out_json}")
    if record["comparison_mode"] != "tolerant":
        raise ValueError(f"{kernel}/{label} is classified {record['comparison_mode']!r}, "
                          f"not 'tolerant' — nothing to calibrate")

    info = resolve_kernel_source(kernel)
    with tempfile.TemporaryDirectory() as wd:
        exe = _compile(info, wd)
        run1_stdout, _ = _run_once(exe, info)
        max_abs_err = record["evidence"].get("max_observed_delta")  # seed with what we already have
        for _ in range(n_calibration_runs):
            run_i_stdout, _ = _run_once(exe, info)
            _, diag = compare_passthrough_outputs(run1_stdout, run_i_stdout, n_safety_override=None)
            for f in diag.get("failures", []):
                if f["label"].split("#", 1)[0] != label:
                    continue
                abs_err = f.get("abs_err")
                if isinstance(abs_err, (int, float)) and math.isfinite(abs_err):
                    max_abs_err = abs_err if max_abs_err is None else max(max_abs_err, abs_err)

    if max_abs_err is None:
        raise RuntimeError(f"{kernel}/{label}: no finite divergence observed across "
                            f"{n_calibration_runs} calibration runs — cannot calibrate "
                            f"(the buffer may have settled 'tolerant' on a non-numeric "
                            f"failure type, e.g. block_missing; recheck its evidence)")

    record["tolerance"] = {
        "atol": CALIBRATION_SAFETY_FACTOR * max_abs_err,
        "rtol": 0.0,
        "source": "measured",
    }
    record["last_updated"] = None  # Date.now()-equivalent unavailable here; stamp on write-back by caller if needed
    doc["buffers"][label] = record
    with open(out_json, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"calibrated {kernel}/{label}: atol={record['tolerance']['atol']!r} "
          f"(max_abs_err={max_abs_err!r} x {CALIBRATION_SAFETY_FACTOR}), wrote {out_json}")
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _n_for_target_p(p: float, confidence: float = 0.95) -> int:
    return max(1, math.ceil(math.log(1 - confidence) / math.log(1 - p)))


def probe_kernel(kernel: str, n_runs: int, time_cap_s: float | None = None) -> dict:
    """Exploratory: run n_runs seeded re-executions WITHOUT early-stopping on first
    divergence, so we can empirically estimate how often a given buffer actually diverges
    rather than just settling "tolerant" at the first observed mismatch. Compares every run
    against run 1 (fixed baseline, matching how classify_kernel/Tier 2 treat the spec) and
    also against the immediately preceding run (to see whether divergence is a one-off jump
    or a sticky/bursty pattern across consecutive launches)."""
    info = resolve_kernel_source(kernel)
    with tempfile.TemporaryDirectory() as wd:
        exe = _compile(info, wd)
        stdouts = []
        t_start = time.monotonic()
        for i in range(n_runs):
            if time_cap_s is not None and (time.monotonic() - t_start) > time_cap_s:
                print(f"  [time cap hit after {i} runs]")
                break
            out, _ = _run_once(exe, info)
            stdouts.append(out)
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  ...{i + 1}/{n_runs} runs ({time.monotonic() - t_start:.0f}s elapsed)")

    labels0 = _extract_labels(stdouts[0])
    numeric_labels = [l for l, bodies in labels0.items() if any(_is_numeric_block(b) for b in bodies)]
    results = {l: {"vs_run1": [], "vs_prev": []} for l in numeric_labels}

    for i in range(1, len(stdouts)):
        _, diag_vs1 = compare_passthrough_outputs(stdouts[0], stdouts[i], n_safety_override=None)
        diverged_vs1 = {f["label"].split("#", 1)[0] for f in diag_vs1.get("failures", [])}
        _, diag_prev = compare_passthrough_outputs(stdouts[i - 1], stdouts[i], n_safety_override=None)
        diverged_prev = {f["label"].split("#", 1)[0] for f in diag_prev.get("failures", [])}
        for l in numeric_labels:
            results[l]["vs_run1"].append(l in diverged_vs1)
            results[l]["vs_prev"].append(l in diverged_prev)

    n_comparisons = len(stdouts) - 1
    print(f"\n=== probe: {kernel}, {len(stdouts)} runs, {n_comparisons} comparisons ===")
    for l, r in results.items():
        k_vs1 = sum(r["vs_run1"])
        k_prev = sum(r["vs_prev"])
        p_hat = k_vs1 / n_comparisons if n_comparisons else float("nan")
        print(f"  {l}: diverged {k_vs1}/{n_comparisons} vs fixed run1 "
              f"(p_hat={p_hat:.3f}), {k_prev}/{n_comparisons} vs previous run "
              f"[pattern: {''.join('X' if d else '.' for d in r['vs_run1'])}]")
    return {"kernel": kernel, "n_runs": len(stdouts), "results": results}


def aggregate_summary() -> None:
    """Rebuild classification_summary.csv from every *_buffer_classification.json sidecar
    currently on disk. Each --classify invocation only knows about the kernels it just ran
    and overwrites the combined CSV, so running Phase B in more than one batch requires this
    to recover the full picture without re-running earlier kernels."""
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "*_buffer_classification.json"))):
        kernel = os.path.basename(path)[: -len("_buffer_classification.json")]
        with open(path) as fh:
            doc = json.load(fh)
        for label, rec in doc.get("buffers", {}).items():
            ev = rec["evidence"]
            rows.append({
                "kernel": kernel, "label": label,
                "comparison_mode": rec["comparison_mode"],
                "n_comparisons": ev["n_comparisons"],
                "observed_stable": ev["observed_stable"],
                "max_observed_delta": ev["max_observed_delta"],
                "declared_type": rec["declared_type"],
                "status": "ok",
            })
    out_csv = os.path.join(OUT_DIR, "classification_summary.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} rows from {len(glob.glob(os.path.join(OUT_DIR, '*_buffer_classification.json')))} kernel files)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spot-check", action="store_true", help="Phase A: one timed run per kernel")
    ap.add_argument("--classify", action="store_true", help="Phase B: dedicated stability sub-runs")
    ap.add_argument("--aggregate", action="store_true",
                     help="rebuild classification_summary.csv from all on-disk sidecar JSONs")
    ap.add_argument("--probe", action="store_true",
                     help="exploratory: N runs with no early-stop, report empirical divergence rate per label")
    ap.add_argument("--calibrate", action="store_true",
                     help="calibrate a numeric tolerance for an already-settled 'tolerant' label "
                          "(requires --kernel and --label)")
    ap.add_argument("--label", default=None, help="label to calibrate (with --calibrate)")
    ap.add_argument("--kernel", action="append", default=None,
                     help="restrict to specific kernel(s); default = full 16-kernel population")
    ap.add_argument("--target-p", type=float, default=0.02,
                     help="divergence rate to catch at 95%% confidence (default 0.02 -> n~149)")
    ap.add_argument("--n", type=int, default=None, help="override target n directly (skips --target-p)")
    ap.add_argument("--time-cap-s", type=float, default=None,
                     help="per-kernel wall-clock cap for Phase B; achieved n is recorded honestly if hit")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    kernels = args.kernel if args.kernel else _kernel_population()

    if args.spot_check:
        rows = spot_check(kernels)
        out_csv = os.path.join(OUT_DIR, "timing_spotcheck.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")

    if args.classify:
        target_n = args.n if args.n is not None else _n_for_target_p(args.target_p)
        print(f"\nPhase B target n = {target_n}"
              f"{' (explicit --n)' if args.n is not None else f' (from target_p={args.target_p})'}")
        summary_rows = []
        for kernel in kernels:
            print(f"\n--- classify: {kernel} (target_n={target_n}) ---")
            try:
                records = classify_kernel(kernel, target_n, args.time_cap_s)
            except Exception as e:  # noqa: BLE001
                print(f"    FAILED: {e}")
                summary_rows.append({"kernel": kernel, "label": "", "comparison_mode": "",
                                      "n_comparisons": None, "observed_stable": None,
                                      "max_observed_delta": None, "declared_type": "",
                                      "status": f"error: {e}"[:200]})
                continue
            meta = records.pop("_meta")
            print(f"    runs={meta['runs_executed']} stop={meta['stop_reason']} "
                  f"elapsed={meta['elapsed_seconds']}s")
            out_json = os.path.join(OUT_DIR, f"{kernel}_buffer_classification.json")
            with open(out_json, "w") as fh:
                json.dump({"_meta": meta, "buffers": records}, fh, indent=2)
            print(f"    wrote {out_json}")
            for label, rec in records.items():
                ev = rec["evidence"]
                tolerant_flag = "  <-- TOLERANT" if rec["comparison_mode"] == "tolerant" else ""
                print(f"      {label}: {rec['comparison_mode']} "
                      f"(n={ev['n_comparisons']}, max_delta={ev['max_observed_delta']}){tolerant_flag}")
                summary_rows.append({
                    "kernel": kernel, "label": label,
                    "comparison_mode": rec["comparison_mode"],
                    "n_comparisons": ev["n_comparisons"],
                    "observed_stable": ev["observed_stable"],
                    "max_observed_delta": ev["max_observed_delta"],
                    "declared_type": rec["declared_type"],
                    "status": "ok",
                })
        out_csv = os.path.join(OUT_DIR, "classification_summary.csv")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"\nwrote {out_csv}")

    if args.aggregate:
        aggregate_summary()

    if args.probe:
        target_n = args.n if args.n is not None else _n_for_target_p(args.target_p)
        for kernel in kernels:
            probe_kernel(kernel, target_n, args.time_cap_s)

    if args.calibrate:
        if not args.kernel or len(args.kernel) != 1 or not args.label:
            raise SystemExit("--calibrate requires exactly one --kernel and a --label")
        n_calibration_runs = args.n if args.n is not None else 10
        calibrate_tolerance(args.kernel[0], args.label, n_calibration_runs)


if __name__ == "__main__":
    main()
