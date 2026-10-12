"""Post-pipeline regrade harness for the 81 in-flight T1+T3 catches from the SC26 campaign.

planning/LASSI_Verify_Phase2_Bridge.md, Section 2 Step 2; output schema Section 9.1.

What this recovers that a naive "read the archived _v{N}_Trial{M}.cu file" approach cannot:
pipeline_v4.py writes that file unconditionally right after compile+execute succeed, BEFORE
any tier check runs. When a tier then fails and self-correction retries, the same file gets
OVERWRITTEN by the next successful attempt for that code_version slot. So for every catch that
was repaired within budget (is_final_attempt == False) -- which is all 19 T1 catches and 55 of
62 T3 catches in this campaign -- the file on disk today is the fixed code, not the code that
actually triggered the catch. Only budget-exhausted catches (is_final_attempt == True, no later
attempt reuses that slot) have trustworthy archived files.

The failing code is still recoverable: every self-correction retry prompt embeds the exact
failing `code_block` verbatim between "[START OF CODE TO FIX]" / "[/END OF CODE TO FIX]"
markers in the trial's monitor log (pipeline_v4.py:1579/1721/1775/1827, one call site per
failure type). This module extracts it from there, matching each occurrence to its
attempt_log.csv row by ordinal position within the trial (validated 1:1 against chacha20's
three trials: 12/14/12 "needs-fix" rows exactly match 12/14/12 marker pairs).

Known scope limitation (documented, not silently dropped): `other_tier_verdict` for T1
catches is computed from Tier 2 only. Tier 3 evaluation would require a fresh per-relation
LLM call to author METAMORPHIC output blocks for this specific recovered code
(code_insert_metamorphic_variant() is "called once per relation per code being tested" --
verify.py:2575) -- the canonical *transform content* re-inserts deterministically
(_enforce_canonical_transform), but the block's placement/output-capture instrumentation does
not exist for code that never reached Tier 3 in the original run. Out of scope for a
"no new LLM calls" step; recorded as other_tier_t3_note="NOT_COMPUTED" rather than guessed.

No new LLM calls anywhere in this module. Compile/execute/compare only.

Usage:
    python -m lassi_verify_analysis.inflight_catch_regrade --out inflight_catch_regrade.csv
    python -m lassi_verify_analysis.inflight_catch_regrade --kernel chacha20 -v
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import logging
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lassi.verify import compare_passthrough_outputs, _strip_t1_blocks, strip_verify_blocks  # noqa: E402
from lassi_verify_analysis.oracle_harness import (  # noqa: E402
    KERNEL_CONFIGS, compile_cu, run_exe, build_source_baseline, oracle_one, RUN_TIMEOUT_S,
)

log = logging.getLogger("inflight_catch_regrade")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_ROOT = os.path.join(REPO_ROOT, "generated_codes", "experiment")


def _resolve_exp_dir(exp_dir_name: str) -> str | None:
    """Find exp_dir_name under EXP_ROOT regardless of subfolder reorganization.

    Experiment folders get moved around for housekeeping (e.g. archived into
    subfolders) independent of this analysis code, so a fixed direct-child path
    is fragile. Search recursively and cache misses aren't worth it at this scale
    (dozens of directories, called once per kernel per run)."""
    direct = os.path.join(EXP_ROOT, exp_dir_name)
    if os.path.isdir(direct):
        return direct
    for root, dirs, _ in os.walk(EXP_ROOT):
        if exp_dir_name in dirs:
            return os.path.join(root, exp_dir_name)
    return None

# Canonical SC26 experiment directory per kernel -- the folder(s) whose data feeds the paper's
# 16-kernel n=350 population (from oracle_confusion_matrix_cumulative.csv, axis=framework,
# generation=lassi). jacobi and matrix-rotate excluded per standing project convention.
KERNEL_EXP_DIRS: dict[str, list[str]] = {
    "entropy": ["KERNEL_EXP-SC26_ENTROPY_2__07-19-2026_t15_29_00"],
    "extrema": ["KERNEL_EXP-SC26_EXTREMA_1__07-19-2026_t21_47_11"],
    "all-pairs-distance": ["KERNEL_EXP-SC26_ALLPAIRS_1__07-20-2026_t00_11_40"],
    "atomicCost": ["KERNEL_EXP-SC26_ATOMIC_1__07-20-2026_t08_52_15"],
    "threadfence": ["KERNEL_EXP-SC26_THREADF_1__07-20-2026_t14_07_07"],
    "layout": ["KERNEL_EXP-SC26_LAYOUT_1__07-20-2026_t18_40_16"],
    "bsearch": ["KERNEL_EXP-SC26_BSEARCH_1__07-21-2026_t07_47_59"],
    "randomaccess": ["KERNEL_EXP-SC26_RANDA_1__07-21-2026_t21_03_54"],
    "colorwheel": ["KERNEL_EXP-SC26_COLORW_1__07-22-2026_t13_50_27"],
    "chacha20": ["KERNEL_EXP-SC26_CHACHA20_1__07-22-2026_t17_22_45"],
    "floydwarshall": [
        "KERNEL_EXP-SC26_FLOYDW_1__07-23-2026_t00_31_21",
        "KERNEL_EXP-SC26_FLOYDW_2__07-23-2026_t07_42_47",
    ],
    "murmurhash3": ["KERNEL_EXP-SC26_MURMURH_1__07-23-2026_t19_45_45"],
    "jaccard": ["KERNEL_EXP-SC26_JACCARD_1__07-24-2026_t10_49_40"],
    "lid-driven-cavity": ["KERNEL_EXP-SC26_LIDDRIVENC_1__07-24-2026_t16_03_41"],
    "segment-reduce": ["KERNEL_EXP-SC26_SEGMENTR_1__07-25-2026_t19_56_25"],
    "pathfinder": ["KERNEL_EXP-SC26_PATHFINDER_1__07-26-2026_t05_34_18"],
}

# Oracle-quality taxonomy (planning/LASSI_Verify_Paper_Facts.md L48-58 /
# per_kernel_oracle_decisions.md), matched 1:1 against the bridge doc's Section 2 Step 2 table.
ORACLE_GRADE: dict[str, str] = {
    "entropy": "gold", "all-pairs-distance": "gold", "threadfence": "gold",
    "layout": "gold", "bsearch": "gold", "floydwarshall": "gold",
    "chacha20": "gold", "extrema": "gold", "segment-reduce": "gold", "murmurhash3": "gold",
    "atomicCost": "reliable",
    "randomaccess": "weak", "jaccard": "weak",
    "colorwheel": "broken",
    "pathfinder": "absent", "lid-driven-cavity": "absent",
}

INTERPRETATION_GRID = {
    ("PASS", "FAIL"): "unique_catch",
    ("CATCH", "FAIL"): "redundant_catch",
    ("PASS", "PASS"): "isolated_overstrictness",
    ("CATCH", "PASS"): "shared_overstrictness",
}

_TS_RE = re.compile(r"^\[\d\d:\d\d:\d\d\] ")
_START_MARKER = "[START OF CODE TO FIX]"
_END_MARKER = "[/END OF CODE TO FIX]"


# ---------------------------------------------------------------------------
# Monitor-log failing-code recovery
# ---------------------------------------------------------------------------

def _strip_ts(line: str) -> str:
    return _TS_RE.sub("", line, count=1)


def _load_stripped_lines(monitor_log_path: str) -> list[str]:
    with open(monitor_log_path, "r", errors="replace") as f:
        return [_strip_ts(line.rstrip("\n")) for line in f]


def _find_fix_blocks(lines: list[str]) -> list[str]:
    """Recover every [START OF CODE TO FIX]...[/END OF CODE TO FIX] code block, in order."""
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith(_START_MARKER):
            first = lines[i][len(_START_MARKER):]
            if first.startswith(" "):
                first = first[1:]
            body = [first]
            j = i + 1
            while j < n and not lines[j].startswith(_END_MARKER):
                body.append(lines[j])
                j += 1
            blocks.append("\n".join(body).rstrip("\n"))
            i = j + 1
        else:
            i += 1
    return blocks


def _needs_fix_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that triggered a self-correction retry (and therefore have a matching
    [START OF CODE TO FIX] block): non-final LASSI attempts that failed something.

    tier1_verdict=="SKIP" (sentinel not found -- e.g. the LLM stripped the assertion
    blocks) is included alongside "FAIL": check_tier1_assertions() docs this as being
    internally treated as FAIL and retried by pipeline_v4.py, even though the literal
    string recorded in the attempt log is "SKIP", not "FAIL". Missing this case caused a
    3-row marker-count mismatch (lid-driven-cavity, segment-reduce) that the safety check
    in recover_failing_code() correctly caught rather than silently misaligning."""
    is_lassi = df["generation_type"].astype(str).str.strip().str.lower() != "zeroshot"
    return (
        is_lassi
        & (df["is_final_attempt"] == False)  # noqa: E712
        & (
            (df["tier1_verdict"].isin(["FAIL", "SKIP"]))
            | (df["tier2_verdict"] == "FAIL")
            | (df["tier3_verdict"] == "FAIL")
            | (df["execute_status"] == "execute_error")
            | (df["compile_status"] != "success")
        )
    )


def _monitor_log_path(exp_dir: str, trial: int) -> str | None:
    matches = [
        p for p in glob.glob(os.path.join(exp_dir, f"*_monitor_log_*Trial{trial}.txt"))
    ]
    return matches[0] if matches else None


def recover_failing_code(exp_dir: str, attempt_df: pd.DataFrame, trial: int,
                          code_version: int, attempt_number: int) -> tuple[str | None, str]:
    """Recover the exact code that triggered a non-final (repaired) catch.

    Returns (code_text_or_None, status). status is "ok" or a specific failure reason.
    """
    trial_df = attempt_df[attempt_df["trial_number"] == trial]
    mask = _needs_fix_mask(trial_df)
    ordered = trial_df[mask].reset_index(drop=True)
    pos = ordered.index[
        (ordered["code_block_index"] == code_version) & (ordered["attempt_number"] == attempt_number)
    ]
    if len(pos) != 1:
        return None, f"ambiguous_row_match:{len(pos)}_candidates"
    pos = int(pos[0])

    log_path = _monitor_log_path(exp_dir, trial)
    if log_path is None:
        return None, "no_monitor_log"
    lines = _load_stripped_lines(log_path)
    blocks = _find_fix_blocks(lines)
    if len(blocks) != len(ordered):
        return None, f"marker_count_mismatch:{len(blocks)}_blocks_vs_{len(ordered)}_needs_fix_rows"

    code = blocks[pos]
    if not code.strip() or "LASSI-VERIFY" not in code:
        return None, "extracted_code_sanity_check_failed"
    return code, "ok"


# ---------------------------------------------------------------------------
# Archived-file lookup (trustworthy only for budget-exhausted, is_final_attempt=True catches)
# ---------------------------------------------------------------------------

def _find_archived_variant(exp_dir: str, code_version: int, trial: int) -> str | None:
    cands = [
        p for p in glob.glob(os.path.join(exp_dir, f"*_v{code_version}_Trial{trial}.cu"))
        if "baseLLM" not in os.path.basename(p)
    ]
    return cands[0] if cands else None


def _find_source_cu(exp_dir: str, trial: int) -> str | None:
    cands = [
        p for p in glob.glob(os.path.join(exp_dir, f"*_verify_Trial{trial}.cu"))
        if "basellm" not in os.path.basename(p).lower()
    ]
    return cands[0] if cands else None


def _recover_seed(exp_dir: str, trial: int) -> str | None:
    matches = glob.glob(os.path.join(exp_dir, f"result_verification_log_*Trial{trial}.json"))
    if not matches:
        return None
    try:
        return json.load(open(matches[0])).get("witness_chain_seed")
    except (json.JSONDecodeError, OSError):
        return None


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


# ---------------------------------------------------------------------------
# Catch identification
# ---------------------------------------------------------------------------

def _clean_str(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def _load_attempt_log(exp_dir: str) -> pd.DataFrame | None:
    g = glob.glob(os.path.join(exp_dir, "*attempt_log.csv"))
    if not g:
        return None
    return pd.read_csv(g[0], index_col=False)


def find_catches(kernel: str, exp_dir: str) -> list[dict]:
    """Return one dict per in-flight T1 or T3 catch in this experiment dir."""
    df = _load_attempt_log(exp_dir)
    if df is None:
        return []
    is_lassi = df["generation_type"].astype(str).str.strip().str.lower() != "zeroshot"

    catches = []
    t1 = df[is_lassi & (df["tier1_verdict"] == "FAIL")]
    for _, r in t1.iterrows():
        catches.append({
            "kernel": kernel, "exp_dir": exp_dir, "trial": int(r["trial_number"]),
            "code_version": int(r["code_block_index"]), "attempt_number": int(r["attempt_number"]),
            "is_final_attempt": bool(r["is_final_attempt"]),
            "catching_tier": "T1", "catching_observer": _clean_str(r.get("tier1_failed_labels", "")),
        })
    t3 = df[is_lassi & (df["tier3_verdict"] == "FAIL")]
    for _, r in t3.iterrows():
        catches.append({
            "kernel": kernel, "exp_dir": exp_dir, "trial": int(r["trial_number"]),
            "code_version": int(r["code_block_index"]), "attempt_number": int(r["attempt_number"]),
            "is_final_attempt": bool(r["is_final_attempt"]),
            "catching_tier": "T3", "catching_observer": _clean_str(r.get("tier3_failed_relation", "")),
        })
    return catches


# ---------------------------------------------------------------------------
# Per-catch regrade
# ---------------------------------------------------------------------------

def _oracle_grade_one_code(code_text: str, kernel: str, workdir: str, base) -> dict:
    """Strip -> build -> run -> parse against the kernel's oracle config, on raw code text.
    `base` is a pre-built SourceBaseline for this kernel (cached once per kernel by the caller
    -- rebuilding it per-catch would recompile+rerun the pristine source up to 81 times, and
    up to 5x that for tolerance-mode kernels' self-variance runs)."""
    cfg = KERNEL_CONFIGS[kernel]
    if not base.ok:
        return {"oracle_verdict": "UNGRADABLE", "ungradable_reason": f"source_baseline_fail:{base.note}"}
    cu_path = os.path.join(workdir, "catch_variant.cu")
    with open(cu_path, "w") as fh:
        fh.write(code_text)
    row = oracle_one(cu_path, cfg, base, workdir)
    verdict = row["oracle_verdict"]
    if verdict in ("PASS", "FAIL"):
        return {"oracle_verdict": verdict, "ungradable_reason": ""}
    return {"oracle_verdict": "UNGRADABLE", "ungradable_reason": f"{verdict}:{row.get('oracle_detail','')}"}


def regrade_t1_catch(catch: dict, attempt_df: pd.DataFrame, workdir: str, source_base) -> dict:
    """T1 catch: recover code, compute other_tier_verdict via Tier-2 regrade, oracle-grade."""
    kernel, exp_dir = catch["kernel"], catch["exp_dir"]
    trial, code_version = catch["trial"], catch["code_version"]
    out = dict(catch)
    out["repaired_within_budget"] = not catch["is_final_attempt"]
    out["other_tier_verdict"] = ""
    out["other_tier_observer"] = ""
    out["other_tier_t3_note"] = "NOT_COMPUTED: requires new per-relation LLM call (code_insert_metamorphic_variant), out of scope for this step"
    out["oracle_verdict"] = ""
    out["ungradable_reason"] = ""
    out["source_file"] = ""

    if catch["is_final_attempt"]:
        code = None
        vpath = _find_archived_variant(exp_dir, code_version, trial)
        if vpath is None:
            out["ungradable_reason"] = "archived_file_not_found"
            out["oracle_verdict"] = "UNGRADABLE"
            out["interpretation_cell"] = "ungradable"
            return out
        code = open(vpath, "r", errors="replace").read()
        out["source_file"] = vpath
    else:
        code, status = recover_failing_code(exp_dir, attempt_df, trial, code_version, catch["attempt_number"])
        out["source_file"] = f"log_extracted:trial{trial}:cv{code_version}:a{catch['attempt_number']}"
        if code is None:
            out["ungradable_reason"] = f"code_recovery_failed:{status}"
            out["oracle_verdict"] = "UNGRADABLE"
            out["interpretation_cell"] = "ungradable"
            return out

    stripped = _strip_t1_blocks(code)

    src_path = _find_source_cu(exp_dir, trial)
    seed = _recover_seed(exp_dir, trial)
    if src_path is None:
        out["other_tier_verdict"] = "N/A"
        out["ungradable_reason"] = "no_source_verify_file_for_t2_compare"
    else:
        cfg = KERNEL_CONFIGS[kernel]
        sexe = os.path.join(workdir, "src.exe")
        sok, serr = compile_cu(src_path, sexe, cfg.compile_extra)
        vexe = os.path.join(workdir, "variant.exe")
        vcu = os.path.join(workdir, "variant_stripped.cu")
        with open(vcu, "w") as fh:
            fh.write(stripped)
        vok, verr = compile_cu(vcu, vexe, cfg.compile_extra)
        if not sok or not vok:
            out["other_tier_verdict"] = "N/A"
            out["ungradable_reason"] = f"t2_compile_fail: source_ok={sok} variant_ok={vok}"
        else:
            srun_ok, s_stdout, _ = _run_seeded(sexe, cfg.run_args, seed)
            vrun_ok, v_stdout, _ = _run_seeded(vexe, cfg.run_args, seed)
            if not srun_ok or not vrun_ok:
                out["other_tier_verdict"] = "N/A"
                out["ungradable_reason"] = f"t2_run_fail: source_ok={srun_ok} variant_ok={vrun_ok}"
            else:
                t2_verdict, t2_diag = compare_passthrough_outputs(s_stdout, v_stdout, n_safety_override=None)
                out["other_tier_verdict"] = "PASS" if t2_verdict == "PASS" else "CATCH"
                if t2_verdict != "PASS":
                    failing = t2_diag.get("failures", []) if isinstance(t2_diag, dict) else []
                    labels = sorted({f.get("label", "") for f in failing if isinstance(f, dict)})
                    out["other_tier_observer"] = ",".join(labels)

    oc = _oracle_grade_one_code(strip_verify_blocks(code), kernel, workdir, source_base)
    out["oracle_verdict"] = oc["oracle_verdict"]
    if oc["ungradable_reason"]:
        out["ungradable_reason"] = (out["ungradable_reason"] + ";" if out["ungradable_reason"] else "") + oc["ungradable_reason"]

    other = out["other_tier_verdict"] if out["other_tier_verdict"] in ("PASS", "CATCH") else None
    oracle = out["oracle_verdict"] if out["oracle_verdict"] in ("PASS", "FAIL") else None
    out["interpretation_cell"] = INTERPRETATION_GRID.get((other, oracle), "ungradable")
    return out


def regrade_t3_catch(catch: dict, attempt_df: pd.DataFrame, workdir: str, source_base) -> dict:
    """T3 catch: T1+T2 already passed by construction; only need the oracle verdict."""
    kernel, exp_dir = catch["kernel"], catch["exp_dir"]
    trial, code_version = catch["trial"], catch["code_version"]
    out = dict(catch)
    out["repaired_within_budget"] = not catch["is_final_attempt"]
    out["other_tier_verdict"] = "PASS"  # architecturally always PASS: T1/T2 ran and passed, strict ordering
    out["other_tier_observer"] = ""
    out["other_tier_t3_note"] = ""
    out["oracle_verdict"] = ""
    out["ungradable_reason"] = ""
    out["source_file"] = ""

    if catch["is_final_attempt"]:
        vpath = _find_archived_variant(exp_dir, code_version, trial)
        if vpath is None:
            out["ungradable_reason"] = "archived_file_not_found"
            out["oracle_verdict"] = "UNGRADABLE"
            out["interpretation_cell"] = "ungradable"
            return out
        code = open(vpath, "r", errors="replace").read()
        out["source_file"] = vpath
    else:
        code, status = recover_failing_code(exp_dir, attempt_df, trial, code_version, catch["attempt_number"])
        out["source_file"] = f"log_extracted:trial{trial}:cv{code_version}:a{catch['attempt_number']}"
        if code is None:
            out["ungradable_reason"] = f"code_recovery_failed:{status}"
            out["oracle_verdict"] = "UNGRADABLE"
            out["interpretation_cell"] = "ungradable"
            return out

    oc = _oracle_grade_one_code(strip_verify_blocks(code), kernel, workdir, source_base)
    out["oracle_verdict"] = oc["oracle_verdict"]
    out["ungradable_reason"] = oc["ungradable_reason"]

    oracle = out["oracle_verdict"] if out["oracle_verdict"] in ("PASS", "FAIL") else None
    out["interpretation_cell"] = INTERPRETATION_GRID.get(("PASS", oracle), "ungradable")
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

SCHEMA_COLS = [
    "kernel", "trial", "code_version", "catching_tier", "catching_observer",
    "repaired_within_budget", "other_tier_verdict", "other_tier_observer", "other_tier_t3_note",
    "oracle_verdict", "oracle_grade", "ungradable_reason", "interpretation_cell",
    "reproduction_check", "source_file", "regrade_run_timestamp",
]


def run_all(kernels: list[str] | None = None, out_path: str | None = None) -> pd.DataFrame:
    kernels = kernels or sorted(KERNEL_EXP_DIRS.keys())
    rows = []
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    for kernel in kernels:
        for exp_dir_name in KERNEL_EXP_DIRS[kernel]:
            exp_dir = _resolve_exp_dir(exp_dir_name)
            if exp_dir is None:
                log.warning("missing exp dir: %s", exp_dir_name)
                continue
            attempt_df = _load_attempt_log(exp_dir)
            if attempt_df is None:
                log.warning("no attempt_log.csv in %s", exp_dir)
                continue
            catches = find_catches(kernel, exp_dir)
            log.info("%s (%s): %d catches", kernel, exp_dir_name, len(catches))
            if not catches:
                continue
            with tempfile.TemporaryDirectory(prefix="inflight_") as workdir:
                source_base = build_source_baseline(KERNEL_CONFIGS[kernel], workdir)
                if not source_base.ok:
                    log.warning("  SOURCE baseline failed for %s: %s", kernel, source_base.note)
                for catch in catches:
                    try:
                        if catch["catching_tier"] == "T1":
                            row = regrade_t1_catch(catch, attempt_df, workdir, source_base)
                        else:
                            row = regrade_t3_catch(catch, attempt_df, workdir, source_base)
                    except Exception as e:  # noqa: BLE001
                        row = dict(catch)
                        row.update(
                            repaired_within_budget=not catch["is_final_attempt"],
                            other_tier_verdict="", other_tier_observer="", other_tier_t3_note="",
                            oracle_verdict="UNGRADABLE", ungradable_reason=f"exception:{e}",
                            interpretation_cell="ungradable", source_file="",
                        )
                    row["code_version"] = row.pop("code_version")
                    row["trial"] = row.pop("trial")
                    row["oracle_grade"] = ORACLE_GRADE.get(kernel, "")
                    row["reproduction_check"] = ""
                    row["regrade_run_timestamp"] = ts
                    rows.append(row)
                    log.info("  trial%s v%-3s [%s/%s] other=%s oracle=%s -> %s",
                             catch["trial"], catch["code_version"], catch["catching_tier"],
                             catch["catching_observer"][:30], row.get("other_tier_verdict"),
                             row.get("oracle_verdict"), row.get("interpretation_cell"))

    df = pd.DataFrame(rows)
    if not df.empty:
        for c in SCHEMA_COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[SCHEMA_COLS]
    if out_path and not df.empty:
        df.to_csv(out_path, index=False)
        log.info("wrote %s (%d rows)", out_path, len(df))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", nargs="+", default=None, choices=sorted(KERNEL_EXP_DIRS.keys()))
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(message)s")
    run_all(a.kernel, a.out)


if __name__ == "__main__":
    main()
