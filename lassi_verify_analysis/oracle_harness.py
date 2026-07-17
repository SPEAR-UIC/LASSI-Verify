"""
Oracle Harness — "verify the verifier" independent ground-truth collector.

Phase 1 of `planning/verify_the_verifier_analysis_plan.md`.

For every generated variant on disk, this harness reconstructs an INDEPENDENT
correctness verdict (the "oracle") to compare against the framework's own tier verdict.
It is deliberately kept OUT of the production pipeline: a production LASSI-Verify only
needs the final winner, not analysis of every variant. Keeping the oracle in a separate
tool also makes the methodological claim cleaner ("an independent post-hoc oracle") than
having the pipeline score itself.

Two oracle classes (see plan §3):
  Class-1  built-in author check (PASS/FAIL, error count, ...) — independent of both the
           LLM and the source-as-spec premise. The strong claim.
  Class-2  full-output diff of the stripped, deployable variant vs SOURCE at a fixed
           deterministic workload — independent of the instrumentation (unlike Tier 2,
           which compares instrumented passthrough points a cheat can game), but not of
           the source-as-spec premise.

Tolerance policy (plan §4): byte-exact fingerprint by default; a SOURCE-self-variance-
derived tolerance only for genuinely non-deterministic (FP-reduction) outputs. We never
reuse the pipeline's own T2/T3 tolerance (that would make framework-VALID => oracle-PASS
nearly tautological).

Independence note: the harness does its OWN nvcc compile + subprocess run. It reuses only
`lassi.verify.strip_verify_blocks()` — the exact strip logic whose fidelity we separately
validate (see `validate_strip_roundtrip`) against the pipeline's own `_LASSI_RESULT_` files.

Usage:
    python -m lassi_verify_analysis.oracle_harness --kernel pathfinder \
        --exp-dir generated_codes/experiment/KERNEL_TEST_PATHFINDER_WC_1__07-11-2026_t08_19_05
    python -m lassi_verify_analysis.oracle_harness --kernel jacobi \
        --exp-dir generated_codes/experiment/KERNEL_TEST_JACOBI_WC_2__07-08-2026_t12_56_57

Writes `oracle_results.csv` into the experiment directory (or --out).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

# Reuse the pipeline's canonical strip (and nothing else from the exec path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lassi.verify import strip_verify_blocks  # noqa: E402

log = logging.getLogger("oracle_harness")

# ---------------------------------------------------------------------------
# Compile / run configuration
# ---------------------------------------------------------------------------

NVCC = "nvcc"
NVCC_FLAGS = ["-std=c++14", "-arch=sm_89", "-O3"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HECBENCH = os.path.join(REPO_ROOT, "source_codes", "HeCBench")

COMPILE_TIMEOUT_S = 300
RUN_TIMEOUT_S = 600


# ---------------------------------------------------------------------------
# Per-kernel test configuration
#
# This is TEST configuration (how we externally grade outputs), categorically
# different from the kernel-specific *framework bias* that feedback_no_kernel_specific_bias
# forbids (that rule governs the framework's prompts/relations/feedback, not the oracle).
# ---------------------------------------------------------------------------

@dataclass
class KernelOracleConfig:
    app_name: str
    source_rel: str                 # pristine source, relative to HECBENCH
    run_args: list[str]             # argv for the deployable
    compile_extra: list[str] = field(default_factory=list)
    # Which axis is the primary oracle verdict: "class1" (built-in author check — strong
    # where the check compares against a reference) or "class2" (source-output diff).
    oracle_primary: str = "class2"
    # Class-1 built-in check.
    #   class1_mode "token": class1_regex captures a PASS/FAIL token compared to class1_pass_value.
    #   class1_mode "fail_if_present": FAIL iff class1_fail_pattern appears (e.g. bsearch
    #     prints "incorrect result" only on failure; silence == PASS).
    class1_mode: str = "token"
    class1_regex: str | None = None
    class1_pass_value: str | None = None
    class1_fail_pattern: str | None = None
    # Class-2 comparison mode: "fingerprint" (byte-exact) | "tolerance" (numeric signal) | "none".
    class2_mode: str = "fingerprint"
    # If set, fingerprint only the LAST N lines of stdout (via `tail`), so a kernel that dumps
    # a huge deterministic input array before its result (pathfinder's -DBENCH_PRINT wall dump,
    # ~5GB at the validated workload) can be run at the validated workload without capturing it.
    # The discriminating output (data+result) is the last 2 lines. No kernel modification.
    capture_tail_lines: int | None = None
    # For tolerance mode: regex with one float capture group = the comparison signal
    # (last match is used).
    class2_signal_regex: str | None = None
    # Lines to drop before fingerprinting (nondeterministic timing / perf noise).
    stdout_drop_patterns: list[str] = field(default_factory=list)
    # How many times to run SOURCE to measure self-variance (tolerance mode only).
    source_variance_runs: int = 5


KERNEL_CONFIGS: dict[str, KernelOracleConfig] = {
    # pathfinder: no printed correctness check (the "SUCCESS" is `return EXIT_SUCCESS`).
    # Class-2 only, via -DBENCH_PRINT which dumps the DP `data`/`result` arrays.
    # Uses a REDUCED but fixed deterministic workload (M_SEED is fixed) so the full
    # BENCH_PRINT stdout stays small; source and every variant use the same workload, so
    # this is a valid equivalence oracle (not the pipeline's timing workload). See open
    # items in the plan for optionally scaling this up with result-only fingerprinting.
    "pathfinder": KernelOracleConfig(
        app_name="pathfinder",
        source_rel="pathfinder/pathfinder-cuda_main.cu",
        run_args=["100000", "10000", "2500"],   # the framework's validated workload (app_params)
        compile_extra=["-DBENCH_PRINT"],
        class2_mode="fingerprint",
        # -DBENCH_PRINT dumps a ~5GB wall array first; the discriminating DP output (data +
        # result) is the last 2 lines. Capture only those via tail (no kernel edit).
        capture_tail_lines=2,
    ),
    # jacobi: Class-1 PASS/FAIL is a *convergence* check (weak — a wrong-but-converging
    # variant can still print PASS), so we ALSO run a Class-2 tolerance diff on the final
    # printed error. FP atomic reduction => output wanders run-to-run => source-self-
    # variance tolerance (plan §4).
    "jacobi": KernelOracleConfig(
        app_name="jacobi",
        source_rel="jacobi/jacobi-cuda_main.cu",
        run_args=[],
        class1_regex=r"^(PASS|FAIL)\s*$",
        class1_pass_value="PASS",
        class2_mode="tolerance",
        class2_signal_regex=r"Error after iteration\s+\d+\s*=\s*([0-9.eE+\-]+)",
        stdout_drop_patterns=[
            r"Average execution time per iteration",
            r"Total elapsed time",
        ],
        source_variance_runs=5,
    ),
    # --- Class-1-primary kernels: the built-in author check IS the strong oracle. A stealth
    #     cheat produces wrong output => the check FAILs, so these need no source diff. ---
    "threadfence": KernelOracleConfig(
        app_name="threadfence", source_rel="threadfence/threadfence-cuda_main.cu",
        run_args=["100", "250000000"], oracle_primary="class1", class2_mode="none",
        class1_regex=r"(PASS|FAIL)$", class1_pass_value="PASS",
    ),
    # keogh's check does its own FP tolerance (fabsf(lb-lb_h) > 1e-2) => strong, not weak.
    "keogh": KernelOracleConfig(
        app_name="keogh", source_rel="keogh/keogh-cuda_main.cu",
        run_args=["256", "22500000", "100"], oracle_primary="class1", class2_mode="none",
        class1_regex=r"(PASS|FAIL)$", class1_pass_value="PASS",
    ),
    # bsearch: verify() prints "incorrect result" only on failure; silence == PASS.
    # (T3-transformed runs legitimately print it, but the oracle runs untransformed input.)
    "bsearch": KernelOracleConfig(
        app_name="bsearch", source_rel="bsearch/bsearch-cuda_main.cu",
        run_args=["8388608", "100"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"incorrect result",
    ),
    "segment-reduce": KernelOracleConfig(
        # Integer segmented reduction: built-in check compares each segment sum against an
        # int reference (exact arithmetic, reliable oracle). It prints an error line only on
        # failure => fail_if_present, like bsearch.
        app_name="segment-reduce", source_rel="segment-reduce/segment-reduce-cuda_main.cu",
        run_args=["16384", "100"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"does not agree with the reference",
    ),
    "extrema": KernelOracleConfig(
        # Relative-extrema detection (1D + 2D). Output is a bool array; the built-in check
        # compares GPU bools against a CPU reference (cpu_relextrema_*) element-by-element with
        # EXACT equality. Inputs are integer-valued (rand() % length) so even the float/double
        # instantiations compare exactly — no FP-tolerance fragility. CPU and GPU use the SAME
        # input array, so the check is self-consistent regardless of rand() seed (reliable
        # oracle). It prints "1D test: FAILED" / "2D test: FAILED" only on failure =>
        # fail_if_present, like bsearch/segment-reduce. The single arg is `repeat` (timing loop
        # count) and does not affect correctness; 750 matches the framework-validated workload.
        app_name="extrema", source_rel="extrema/extrema-cuda_main.cu",
        run_args=["750"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"test: FAILED",
    ),
    # colorwheel: EXCLUDED from the oracle analysis (2026-07-15). Its built-in check is an EXACT
    # byte `memcmp` of the host reference image vs the device image, but both are computed from
    # FP trig (host `computeColor` vs GPU `color` kernel). Host and device rounding differ, so the
    # PRISTINE UNMODIFIED SOURCE fails its own check (max_error=1 on sm_89). Unreliable exact-FP
    # oracle, same class as keogh/threadfence — we do NOT grade variants against a check the source
    # cannot pass, and we do NOT modify the kernel. Config kept for provenance; do not aggregate.
    # A source-device (not host-device) diff with a small pixel tolerance is a possible future
    # revisit (see planning/per_kernel_oracle_decisions.md).
    "colorwheel": KernelOracleConfig(
        app_name="colorwheel", source_rel="colorwheel/colorwheel-cuda_main.cu",
        run_args=["10000", "256", "500000"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present",
        class1_fail_pattern=r"Maximum error between host and device",
    ),
    "chacha20": KernelOracleConfig(
        app_name="chacha20", source_rel="chacha20/chacha20-cuda_main.cu",
        run_args=["300000"], oracle_primary="class1", class2_mode="none",
        class1_regex=r"(PASS|FAIL)$", class1_pass_value="PASS",
    ),
    "floydwarshall": KernelOracleConfig(
        app_name="floydwarshall", source_rel="floydwarshall/floydwarshall-cuda_main.cu",
        run_args=["2048", "100", "16"], oracle_primary="class1", class2_mode="none",
        class1_regex=r"(PASS|FAIL)$", class1_pass_value="PASS",
    ),
    "randomaccess": KernelOracleConfig(
        app_name="randomaccess", source_rel="randomaccess/randomaccess-cuda_main.cu",
        run_args=["10"], oracle_primary="class1", class2_mode="none",
        class1_regex=r"Found .*\((PASS|FAIL)\)", class1_pass_value="PASS",
    ),
    "murmurhash3": KernelOracleConfig(
        app_name="murmurhash3", source_rel="murmurhash3/murmurhash3-cuda_main.cu",
        run_args=["750000", "500"], oracle_primary="class1", class2_mode="none",
        class1_regex=r"(SUCCESS|FAIL)$", class1_pass_value="SUCCESS",
    ),
    # lid-driven-cavity: NO built-in correctness check (Navier-Stokes). FP kernel emitting a
    # residual scalar => WEAK Class-2 (like jacobi). Preliminary reading via `res =`; proper
    # full-field oracle is Phase 1b. Flagged weak in reporting.
    "lid-driven-cavity": KernelOracleConfig(
        app_name="lid-driven-cavity", source_rel="lid-driven-cavity/lid-driven-cavity-cuda_main.cu",
        run_args=[], oracle_primary="class2", class2_mode="tolerance",
        class2_signal_regex=r"res\s*=\s*([0-9.eE+\-]+)",
        stdout_drop_patterns=[r"execution time", r"elapsed"],
        source_variance_runs=5,
    ),
    # all-pairs-distance: built-in check computes a CPU reference and prints PASS/FAIL for each of
    # THREE sub-checks (three distance formulations). Integer-valued data => exact comparison,
    # reliable oracle. Multiple PASS/FAIL lines => fail_if_present (FAIL iff ANY sub-check fails);
    # token mode would only read the last line. app_params=["10000"].
    "all-pairs-distance": KernelOracleConfig(
        app_name="all-pairs-distance",
        source_rel="all-pairs-distance/all-pairs-distance-cuda_main.cu",
        run_args=["10000"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"FAIL",
    ),
    # layout: AoS/SoA data-layout kernel. Built-in check builds an int reference and prints
    # PASS/FAIL for the AoS result and the SoA result (two checks). Integer => exact, reliable.
    # Two PASS/FAIL lines => fail_if_present. app_params=["2500"].
    "layout": KernelOracleConfig(
        app_name="layout", source_rel="layout/layout-cuda_main.cu",
        run_args=["2500"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"FAIL",
    ),
    # entropy: local 5x5-window Shannon entropy. Built-in check runs a CPU reference() over the
    # same input and prints a single PASS/FAIL. app_params=["10000","1024","1"]. fail_if_present
    # for uniformity (single PASS/FAIL line either way).
    "entropy": KernelOracleConfig(
        app_name="entropy", source_rel="entropy/entropy-cuda_main.cu",
        run_args=["10000", "1024", "1"], oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"FAIL",
    ),
    # jaccard: EXCLUDED-by-default from the aggregate oracle (added 2026-07-16). Its known-answer
    # PASS/FAIL check compares weight_j[0] against HARD-CODED constants (0.306667 / 0.230000) that
    # are only correct for the fixed 4x4 `#ifdef DEBUG` matrix — NOT the random benchmark workload
    # ["1024","512","1000"] the framework validated variants at. So the reliable check exists only
    # under -DDEBUG at a workload the framework never used (weaker independent check; see
    # per_kernel_oracle_decisions.md). Config kept for provenance / an explicit caveated run; do
    # NOT fold into the headline aggregate without stating the workload-mismatch caveat.
    "jaccard": KernelOracleConfig(
        app_name="jaccard", source_rel="jaccard/jaccard-cuda_main.cu",
        run_args=["1024", "512", "1000"], compile_extra=["-DDEBUG"],
        oracle_primary="class1", class2_mode="none",
        class1_mode="fail_if_present", class1_fail_pattern=r"FAIL",
    ),
}


# ---------------------------------------------------------------------------
# Build / run primitives (independent of the pipeline exec path)
# ---------------------------------------------------------------------------

def compile_cu(cu_path: str, out_path: str, extra: list[str]) -> tuple[bool, str]:
    """nvcc compile. Returns (ok, error_log)."""
    cmd = [NVCC, *NVCC_FLAGS, *extra, cu_path, "-o", out_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "compile_timeout"
    if p.returncode != 0:
        return False, (p.stderr or p.stdout)[-2000:]
    return True, ""


def run_exe(exe_path: str, args: list[str], tail_lines: int | None = None) -> tuple[bool, str, str]:
    """Run the binary. Returns (ok, stdout, stderr).

    If tail_lines is set, pipe stdout through `tail -n K` so a huge deterministic pre-amble
    (e.g. pathfinder's wall dump) is streamed+discarded by tail instead of captured — only the
    last K lines (the discriminating output) reach us. `set -o pipefail` propagates the exe's
    exit code, not tail's."""
    try:
        if tail_lines:
            cmd = f'set -o pipefail; "{exe_path}" {" ".join(args)} 2>/dev/null | tail -n {tail_lines}'
            p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                               timeout=RUN_TIMEOUT_S)
        else:
            p = subprocess.run([exe_path, *args], capture_output=True, text=True,
                               timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "", "run_timeout"
    return (p.returncode == 0), p.stdout, p.stderr


def _filter_stdout(stdout: str, drop_patterns: list[str]) -> str:
    """Drop nondeterministic lines (timing/perf) before fingerprinting."""
    if not drop_patterns:
        return stdout
    keep = []
    compiled = [re.compile(pp) for pp in drop_patterns]
    for line in stdout.splitlines():
        if any(c.search(line) for c in compiled):
            continue
        keep.append(line)
    return "\n".join(keep)


def fingerprint(stdout: str, drop_patterns: list[str]) -> str:
    filtered = _filter_stdout(stdout, drop_patterns)
    return hashlib.sha256(filtered.encode("utf-8", "replace")).hexdigest()[:16]


def parse_class1(stdout: str, cfg: KernelOracleConfig) -> str | None:
    """Return 'PASS'/'FAIL'/None from the built-in check."""
    if cfg.class1_mode == "fail_if_present":
        if not cfg.class1_fail_pattern:
            return None
        return "FAIL" if re.search(cfg.class1_fail_pattern, stdout) else "PASS"
    # token mode
    if not cfg.class1_regex:
        return None
    m = None
    for line in stdout.splitlines():
        mm = re.match(cfg.class1_regex, line.strip())
        if mm:
            m = mm  # last match wins
    if not m:
        return None
    return "PASS" if m.group(1) == cfg.class1_pass_value else "FAIL"


def parse_signal(stdout: str, cfg: KernelOracleConfig) -> float | None:
    """Return the last numeric Class-2 signal value, or None."""
    s = _last_signal_str(stdout, cfg)
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _last_signal_str(stdout: str, cfg: KernelOracleConfig) -> str | None:
    """Return the raw printed string of the last Class-2 signal match (for precision)."""
    if not cfg.class2_signal_regex:
        return None
    vals = re.findall(cfg.class2_signal_regex, stdout)
    return vals[-1] if vals else None


def printed_quantization(num_str: str) -> float | None:
    """Value of 1 unit in the last printed significant digit of `num_str`.

    A source scalar printed to finite precision (e.g. cout's default 6 sig figs,
    "1.38643e-05") cannot be resolved finer than this step. The oracle tolerance must be
    floored here: two values that differ by less than the benchmark prints are
    indistinguishable, not "wrong". E.g. "1.38643e-05" -> 1e-10; "0.0268275" -> 1e-7."""
    try:
        s = num_str.strip().lower()
        mant, exp = (s.split("e") + ["0"])[:2]
        exp = int(exp)
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return 10.0 ** (exp - dec)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Variant discovery
# ---------------------------------------------------------------------------

_VARIANT_RE = re.compile(r"_v(\d+)_Trial(\d+)\.cu$")
# Zero-shot baseline files: `_v{N}-baseLLM_Trial{M}.cu`. The `-baseLLM` infix keeps them out
# of _VARIANT_RE (numeric-only), so the two populations never mix in one CSV.
_ZS_VARIANT_RE = re.compile(r"_v(\d+)-baseLLM_Trial(\d+)\.cu$")


def discover_variants(exp_dir: str) -> list[tuple[int, int, str]]:
    """Return sorted (trial, version, path) for LASSI variant files `_v{N}_Trial{M}.cu`.

    Excludes the `_v{N}-baseLLM_Trial{M}.cu` zero-shot baselines (numeric-only regex);
    those are collected separately via discover_zeroshot_variants (T-D iii)."""
    out = []
    for name in os.listdir(exp_dir):
        m = _VARIANT_RE.search(name)
        if m:
            out.append((int(m.group(2)), int(m.group(1)), os.path.join(exp_dir, name)))
    return sorted(out)


def discover_zeroshot_variants(exp_dir: str) -> list[tuple[int, int, str]]:
    """Return sorted (trial, version, path) for zero-shot baseline files
    `_v{N}-baseLLM_Trial{M}.cu` — the raw one-shot LLM generations (no self-correction).

    These are the FAIR-baseline population for the oracle-anchored zero-shot 3-way (plan
    T-D iii): raw generations contain genuine errors, unlike the framework's self-corrected
    winners, so they actually exercise a verifier's discrimination. Each is a full program
    (it still includes the kernel's built-in check), so the same Class-1/Class-2 oracle
    config applies unchanged."""
    out = []
    for name in os.listdir(exp_dir):
        m = _ZS_VARIANT_RE.search(name)
        if m:
            out.append((int(m.group(2)), int(m.group(1)), os.path.join(exp_dir, name)))
    return sorted(out)


# ---------------------------------------------------------------------------
# Source baseline (fingerprint + self-variance tolerance)
# ---------------------------------------------------------------------------

@dataclass
class SourceBaseline:
    fingerprint: str | None
    class1: str | None
    signal: float | None
    signal_runs: list[float]
    tolerance: float | None          # source-self-variance-derived (abs)
    ok: bool
    note: str = ""


def build_source_baseline(cfg: KernelOracleConfig, workdir: str) -> SourceBaseline:
    src = os.path.join(HECBENCH, cfg.source_rel)
    exe = os.path.join(workdir, "source.exe")
    ok, err = compile_cu(src, exe, cfg.compile_extra)
    if not ok:
        return SourceBaseline(None, None, None, [], None, False, f"source_compile_fail: {err[:120]}")

    runs = max(1, cfg.source_variance_runs if cfg.class2_mode == "tolerance" else 1)
    stdouts, signals = [], []
    for _ in range(runs):
        rok, so, se = run_exe(exe, cfg.run_args, tail_lines=cfg.capture_tail_lines)
        if not rok:
            return SourceBaseline(None, None, None, [], None, False, f"source_run_fail: {se[:120]}")
        stdouts.append(so)
        s = parse_signal(so, cfg)
        if s is not None:
            signals.append(s)

    fp = fingerprint(stdouts[0], cfg.stdout_drop_patterns) if cfg.class2_mode == "fingerprint" else None
    c1 = parse_class1(stdouts[-1], cfg)

    tol = None
    if cfg.class2_mode == "tolerance" and signals:
        spread = max(signals) - min(signals)
        # Tolerance = MAX of two independent floors (plan §4):
        #   (a) source's own run-to-run wander (self-variance), ×2 safety margin;
        #   (b) the printed-precision quantization step, ×2 (a scalar printed to finite
        #       precision cannot be resolved finer than 1 unit in its last digit, so
        #       sub-quantization deltas are rounding noise, not errors).
        # We never reuse the pipeline's own T2/T3 tolerance (that would be circular).
        # The benign/wrong delta gap is ~1e4x here, so the exact ×2 factor is not load-bearing.
        q = printed_quantization(_last_signal_str(stdouts[0], cfg) or "")
        spread_floor = spread * 2.0
        quant_floor = (q * 2.0) if q else 0.0
        tol = max(spread_floor, quant_floor)
        if tol == 0.0:                       # ultimate fallback (no spread, no precision info)
            tol = abs(signals[0]) * 1e-9
    return SourceBaseline(fp, c1, (signals[-1] if signals else None), signals, tol, True)


# ---------------------------------------------------------------------------
# Per-variant oracle verdict
# ---------------------------------------------------------------------------

def oracle_one(cu_path: str, cfg: KernelOracleConfig, base: SourceBaseline,
               workdir: str) -> dict:
    """Strip -> build -> run -> parse. Returns one oracle_results row dict."""
    _method = {"class1": "builtin_check", "class2": (
        "fingerprint_exact" if cfg.class2_mode == "fingerprint" else "fingerprint_tol")}[cfg.oracle_primary]
    row = {
        "app_name": cfg.app_name,
        "oracle_class": cfg.oracle_primary,
        "oracle_method": _method,
        "run_args": " ".join(cfg.run_args),
        "source_fingerprint": base.fingerprint or "",
        "variant_fingerprint": "",
        "class1_verdict": "",
        "class2_verdict": "",
        "oracle_verdict": "",
        "strip_status": "",
        "oracle_detail": "",
        "source_signal": ("" if base.signal is None else repr(base.signal)),
        "variant_signal": "",
        "tolerance": ("" if base.tolerance is None else repr(base.tolerance)),
    }

    # 1. strip
    try:
        raw = open(cu_path, "r", errors="replace").read()
        stripped = strip_verify_blocks(raw)
    except Exception as e:  # noqa: BLE001
        row.update(strip_status="failed", oracle_verdict="UNTESTABLE",
                   oracle_detail=f"strip_error: {e}")
        return row
    row["strip_status"] = "ok"

    # 2. build
    scu = os.path.join(workdir, "variant.cu")
    with open(scu, "w") as fh:
        fh.write(stripped)
    exe = os.path.join(workdir, "variant.exe")
    cok, cerr = compile_cu(scu, exe, cfg.compile_extra)
    if not cok:
        row.update(oracle_verdict="UNTESTABLE_COMPILE", oracle_detail=cerr[:200])
        return row

    # 3. run
    rok, so, se = run_exe(exe, cfg.run_args, tail_lines=cfg.capture_tail_lines)
    if not rok:
        row.update(oracle_verdict="UNTESTABLE_RUN", oracle_detail=(se or "nonzero_exit")[:200])
        return row

    # 4a. Class-1 (recorded whenever a check exists)
    c1 = parse_class1(so, cfg)
    if c1 is not None:
        row["class1_verdict"] = c1

    # 4b. Class-2 (only when it is the primary or configured)
    c2 = None
    if cfg.class2_mode == "fingerprint":
        vfp = fingerprint(so, cfg.stdout_drop_patterns)
        row["variant_fingerprint"] = vfp
        c2 = "PASS" if (base.fingerprint and vfp == base.fingerprint) else "FAIL"
        row["oracle_detail"] = "fingerprint match" if c2 == "PASS" else "fingerprint mismatch"
    elif cfg.class2_mode == "tolerance":
        sig = parse_signal(so, cfg)
        row["variant_signal"] = ("" if sig is None else repr(sig))
        if sig is None or base.signal is None or base.tolerance is None:
            c2 = "UNTESTABLE"
            row["oracle_detail"] = "no comparable signal"
        else:
            delta = abs(sig - base.signal)
            c2 = "PASS" if delta <= base.tolerance else "FAIL"
            row["oracle_detail"] = f"|delta|={delta:.3e} tol={base.tolerance:.3e}"
    if c2 is not None:
        row["class2_verdict"] = c2

    # Primary verdict per config: class1 (strong built-in check) or class2 (source diff).
    if cfg.oracle_primary == "class1":
        row["oracle_verdict"] = c1 if c1 is not None else "UNTESTABLE"
        if c1 is None:
            row["oracle_detail"] = "no class1 check output"
    else:
        row["oracle_verdict"] = c2 if c2 is not None else "UNTESTABLE"
    return row


# ---------------------------------------------------------------------------
# Strip-fidelity validation
# ---------------------------------------------------------------------------

def validate_strip_roundtrip(exp_dir: str) -> str:
    """Compare our strip of a winner's `_v{N}_TrialM.cu` against the pipeline's own
    `_LASSI_RESULT_TrialM.cu` (its stripped deployable). If they match, our strip ==
    the pipeline's strip. Returns a short human-readable report."""
    results = []
    for name in os.listdir(exp_dir):
        m = re.search(r"_LASSI_RESULT_Trial(\d+)\.cu$", name)
        if not m or "_instrumented_" in name:
            continue
        trial = m.group(1)
        pipeline_clean = open(os.path.join(exp_dir, name), "r", errors="replace").read()
        # Real instrumentation markers (ignore provenance-header comments that merely mention
        # "LASSI-VERIFY"): the BEGIN/END delimiters, tier symbols, and instrumentation prints.
        def instr_markers(s: str) -> int:
            return (len(re.findall(r">>> BEGIN LASSI-VERIFY", s))
                    + s.count("_t1_chain")
                    + len(re.findall(r'printf\("LASSI-VERIFY', s)))
        results.append(
            f"Trial {trial}: pipeline LASSI_RESULT instr_markers={instr_markers(pipeline_clean)}")
    return "; ".join(results) if results else "no _LASSI_RESULT_ files found"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_kernel(kernel: str, exp_dir: str, out_path: str, limit: int | None,
               generation: str = "lassi") -> None:
    cfg = KERNEL_CONFIGS[kernel]
    variants = (discover_zeroshot_variants(exp_dir) if generation == "zeroshot"
                else discover_variants(exp_dir))
    if limit:
        variants = variants[:limit]
    log.info("Discovered %d %s variant file(s) in %s", len(variants), generation, exp_dir)

    with tempfile.TemporaryDirectory(prefix="oracle_") as workdir:
        log.info("Building SOURCE baseline (self-variance runs=%d)...",
                 cfg.source_variance_runs if cfg.class2_mode == "tolerance" else 1)
        base = build_source_baseline(cfg, workdir)
        if not base.ok:
            log.error("SOURCE baseline failed: %s", base.note)
            return
        log.info("SOURCE ok | fingerprint=%s class1=%s signal=%s tol=%s runs=%s",
                 base.fingerprint, base.class1, base.signal, base.tolerance, base.signal_runs)

        rows = []
        for trial, version, path in variants:
            r = oracle_one(path, cfg, base, workdir)
            r["trial_number"] = trial
            r["code_version"] = version
            r["generation_type"] = generation
            rows.append(r)
            log.info("  T%s v%-3s -> %-18s %s", trial, version,
                     r["oracle_verdict"], r["oracle_detail"][:60])

    cols = ["app_name", "trial_number", "code_version", "generation_type", "oracle_class",
            "oracle_method", "oracle_verdict", "class1_verdict", "class2_verdict",
            "oracle_detail", "strip_status", "run_args", "source_fingerprint",
            "variant_fingerprint", "source_signal", "variant_signal", "tolerance"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    log.info("Wrote %s (%d rows)", out_path, len(rows))

    # Strip-fidelity note
    log.info("Strip round-trip: %s", validate_strip_roundtrip(exp_dir))

    # Quick readout
    from collections import Counter
    vc = Counter(r["oracle_verdict"] for r in rows)
    log.info("Verdict tally: %s", dict(vc))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Oracle harness (verify-the-verifier Phase 1)")
    ap.add_argument("--kernel", required=True, choices=sorted(KERNEL_CONFIGS))
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <exp-dir>/oracle_results[_zeroshot].csv)")
    ap.add_argument("--generation", choices=["lassi", "zeroshot"], default="lassi",
                    help="which variant population to oracle: 'lassi' (self-corrected pipeline "
                         "variants, default) or 'zeroshot' (raw _v{N}-baseLLM one-shot baselines)")
    ap.add_argument("--limit", type=int, default=None, help="limit variants (smoke test)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    default_name = "oracle_results_zeroshot.csv" if a.generation == "zeroshot" else "oracle_results.csv"
    out = a.out or os.path.join(a.exp_dir, default_name)
    run_kernel(a.kernel, a.exp_dir, out, a.limit, generation=a.generation)


if __name__ == "__main__":
    main()
