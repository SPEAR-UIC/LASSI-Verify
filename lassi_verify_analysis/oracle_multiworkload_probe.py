"""
Multi-workload probe — resolve the witness-catch interpretation (plan Direction-C).

For variants the witness chain flagged but whose output is correct at the validated workload
(the INVALID+PASS cell), decide between:
  (A) over-strictness — variant is semantically equivalent; witness rejected a legit
      restructuring (its intermediate states differ). => matches source at EVERY workload.
  (B) extra-sensitivity — variant is subtly non-equivalent (correct at the validated size by
      reordering/luck, like pathfinder v9); witness caught a real divergence single-workload
      output checking misses. => DIVERGES from source at some other workload.

Method: strip+build each variant, run at several workloads, use the kernel's own reliable
built-in check (exact) at each. Source is the control (must PASS at all). No kernel edits.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lassi_verify_analysis.oracle_harness import (  # noqa: E402
    KERNEL_CONFIGS, HECBENCH, REPO_ROOT, compile_cu, run_exe, parse_class1, strip_verify_blocks,
)
import glob  # noqa: E402

# The 8 pure-witness catches (kernel -> list of (trial, version)), plus their probe workloads.
TARGETS = {
    "threadfence": {
        "exp": "KERNEL_TEST4_THREADF_2*",
        "variants": [(3, 3), (3, 5), (3, 6), (3, 7), (3, 8)],
        # args = [repeat, N]; correctness (h_sum == -N) is independent of repeat, so fix repeat=10.
        "workloads": [["10", "250000000"], ["10", "100000000"], ["10", "123456789"],
                      ["10", "50000000"], ["10", "1000003"]],
    },
    "randomaccess": {
        "exp": "KERNEL_TEST_RAND_WC_1*",
        "variants": [(1, 3), (1, 12)],
        "workloads": [["10"], ["8"], ["9"], ["11"], ["12"]],
    },
    "murmurhash3": {
        "exp": "KERNEL_TEST_MURMUR_WC_1*",
        "variants": [(2, 10)],
        "workloads": [["750000", "500"], ["500000", "500"], ["1000000", "500"],
                      ["333331", "500"], ["123457", "777"]],
    },
}


def _verdict(exe, args, cfg):
    ok, so, se = run_exe(exe, args, tail_lines=cfg.capture_tail_lines)
    if not ok:
        return "RUNFAIL"
    v = parse_class1(so, cfg)
    return v or "NOCHECK"


def main():
    for kernel, spec in TARGETS.items():
        cfg = KERNEL_CONFIGS[kernel]
        expdir = glob.glob(os.path.join(REPO_ROOT, "generated_codes", "experiment", spec["exp"]))[0]
        wls = spec["workloads"]
        print(f"\n===== {kernel} =====")
        print("workloads:", [" ".join(w) for w in wls])
        with tempfile.TemporaryDirectory(prefix="mwp_") as wd:
            # source control
            src = os.path.join(HECBENCH, cfg.source_rel)
            sexe = os.path.join(wd, "src.exe")
            ok, err = compile_cu(src, sexe, cfg.compile_extra)
            src_v = [_verdict(sexe, w, cfg) for w in wls] if ok else ["COMPILEFAIL"] * len(wls)
            print(f"  SOURCE       : {src_v}")
            # each variant
            for (trial, ver) in spec["variants"]:
                f = glob.glob(os.path.join(expdir, f"*_v{ver}_Trial{trial}.cu"))
                if not f:
                    print(f"  T{trial} v{ver}: FILE NOT FOUND")
                    continue
                stripped = strip_verify_blocks(open(f[0], errors="replace").read())
                vcu = os.path.join(wd, "v.cu")
                open(vcu, "w").write(stripped)
                vexe = os.path.join(wd, "v.exe")
                ok, err = compile_cu(vcu, vexe, cfg.compile_extra)
                if not ok:
                    print(f"  T{trial} v{ver}: COMPILEFAIL {err[:80]}")
                    continue
                vv = [_verdict(vexe, w, cfg) for w in wls]
                # classify vs source
                divergent = any(sv == "PASS" and vv_i != "PASS" for sv, vv_i in zip(src_v, vv))
                verdict = "B: NON-EQUIVALENT (diverges)" if divergent else "A: equivalent (all match source)"
                print(f"  T{trial} v{ver}: {vv}  -> {verdict}")


if __name__ == "__main__":
    main()
