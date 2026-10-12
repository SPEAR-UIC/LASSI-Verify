"""
Sweep harness: drives one injected-defect instance through the 5 tier configurations
(full-stack, minus-T1, minus-T2, minus-T3, minus-witness), compiles+runs each, and populates
the provenance schema's tier_configuration_results.

Per-tier detection mechanics (confirmed empirically this session, see
planning/LASSI_Verify_Phase2_Bridge.md's Step 4 injection-implementation dated entry):
  - T1 is self-contained: `LASSI-VERIFY ASSERTIONS [label]: PASS/FAIL` lines from a single
    compile+run of a file, graded like lassi.verify.check_tier1_assertions().
  - T2 is base-vs-injected: lassi.verify.compare_passthrough_outputs() on two separate stdout
    captures (base main file = spec, injected main file = generated).
  - Witness-chain detection IS T2's comparison, scoped to the witness_chain_state label
    specifically (exact-match requirement) -- same comparator call, different label.
  - T3 compares the INJECTED companion's METAMORPHIC block (perturbed input) against the
    SAME INJECTED main file's own unperturbed PASSTHROUGH block for that observation point --
    an internal-consistency check, not base-vs-injected. Needs the relation's full definition
    (name/type/factor/observation_points/tolerance), loaded from the trial's own
    result_verification_log_*.json (tier3_contract.relations) -- the exact production
    definitions, not re-derived.

Real, confirmed architectural coupling (not a bug): minus-T2 strips the main file's ordinary
PASSTHROUGH blocks, which ALSO removes T3's own unperturbed-reference block for any
observation point that isn't the witness_chain_state label -- so T3 has nothing to compare
against in that configuration. Per 2026-08-23 decision, this is recorded as T3 verdict
"not_available" in minus-T2, not forced into "undetected".

Also confirmed this session (correcting an assumption in the design doc): T3 companion files
DO carry their own embedded T1 ASSERTIONS blocks (contrary to "they don't embed T1 blocks in
the first place") -- minus-T1 strips T1 from companions too, for a clean ablation.
"""
from __future__ import annotations

import json
import os
import re

import lassi.verify as verify

from . import kernel_bases
from .strip_utils import strip_t2_blocks_except_witness
from lassi_verify_analysis import oracle_harness as oh

T1_LINE_RE = re.compile(r"LASSI-VERIFY ASSERTIONS \[([^\]]+)\]:\s*(PASS|FAIL)")
PASSTHROUGH_LABEL_RE = re.compile(r"BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\]")

CONFIGS = ["full_stack", "minus_t1", "minus_t2", "minus_t3", "minus_witness"]


def verification_log_path(kernel: str, trial: int | None = None) -> str:
    b = kernel_bases.KERNEL_BASES[kernel]
    trial = trial or b["trial"]
    stem = re.sub(r"_v\d+_Trial\d+\.cu$", "", b["main_file"])
    fname = f"result_verification_log_{stem}_Trial{trial}.json"
    return os.path.join(kernel_bases.REPO_ROOT, b["exp_dir"], fname)


def load_active_relations(kernel: str) -> dict[str, dict]:
    path = verification_log_path(kernel)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    relations = (data.get("tier3_contract") or {}).get("relations") or []
    return {r["name"]: r for r in relations if "name" in r}


def match_companion_to_relation(companion_filename: str, relations: dict[str, dict]) -> dict | None:
    for name, rel in relations.items():
        if f"_mt_{name}_" in companion_filename or companion_filename.endswith(f"_mt_{name}.cu"):
            return rel
    # fall back to substring containment (some companions carry a perturbation suffix like
    # "__p_outlier" after the relation name)
    for name, rel in relations.items():
        if name in companion_filename:
            return rel
    return None


def parse_t1_verdict(stdout: str) -> str:
    matches = T1_LINE_RE.findall(stdout)
    if not matches:
        return "SKIP"
    if any(v == "FAIL" for _, v in matches):
        return "FAIL"
    return "PASS"


def build_tier_configs(
    main_code: str, t3_codes: dict[str, str]
) -> dict[str, tuple[str, dict[str, str]]]:
    """Returns {config_name: (main_code, {companion_filename: companion_code})}."""
    return {
        "full_stack": (main_code, dict(t3_codes)),
        "minus_t1": (
            verify._strip_t1_blocks(main_code),
            {fn: verify._strip_t1_blocks(c) for fn, c in t3_codes.items()},
        ),
        "minus_t2": (strip_t2_blocks_except_witness(main_code), dict(t3_codes)),
        "minus_t3": (main_code, {}),
        "minus_witness": (verify._strip_witness_chain_blocks(main_code), dict(t3_codes)),
    }


def _compile_run(code: str, workdir: str, name: str, run_args: list[str]):
    path = os.path.join(workdir, f"{name}.cu")
    with open(path, "w") as f:
        f.write(code)
    exe = os.path.join(workdir, name)
    ok, err = oh.compile_cu(path, exe, [])
    if not ok:
        return False, "", f"COMPILE FAIL: {err[:500]}"
    ok, out, err = oh.run_exe(exe, run_args)
    if not ok:
        return False, out, f"RUN FAIL: {err[:500]}"
    return True, out, None


def run_instance(
    kernel: str,
    base_main_code: str,
    injected_main_code: str,
    base_t3_codes: dict[str, str],
    injected_t3_codes: dict[str, str],
    workdir: str,
) -> dict:
    """Runs all 5 tier configurations for one injected-defect instance. Returns the
    tier_configuration_results dict (+ per-config raw diagnostics for debugging)."""
    os.makedirs(workdir, exist_ok=True)
    run_args = oh.KERNEL_CONFIGS[kernel].run_args if kernel in oh.KERNEL_CONFIGS else []
    relations = load_active_relations(kernel)
    # Same classification-aware T2 grading the live pipeline now uses (planning/
    # LASSI_Verify_Phase2_Bridge.md Step 1 / buffer_classification wiring) -- without this
    # the sweep would keep exercising the pre-fix generic-tolerance behavior even after the
    # production code path changed.
    buffer_classification = verify.load_buffer_classification(kernel)

    base_configs = build_tier_configs(base_main_code, base_t3_codes)
    inj_configs = build_tier_configs(injected_main_code, injected_t3_codes)

    results = {}
    for config in CONFIGS:
        cfg_dir = os.path.join(workdir, config)
        os.makedirs(cfg_dir, exist_ok=True)
        base_main, base_t3 = base_configs[config]
        inj_main, inj_t3 = inj_configs[config]

        ok_b, out_b, err_b = _compile_run(base_main, cfg_dir, "base_main", run_args)
        ok_i, out_i, err_i = _compile_run(inj_main, cfg_dir, "inj_main", run_args)
        if not (ok_b and ok_i):
            if ok_b and not ok_i:
                # Base compiles+runs fine, injected build crashes/fails to run -- this is
                # NOT ambiguous tooling failure, it's an unambiguous, maximally-severe form
                # of detection (a real OOB access hard-crashing rather than corrupting a
                # value silently). Confirmed real case: murmurhash3's BOUNDARY-GUARD-
                # CORRUPTION widens a grid-stride loop bound by one, walking off the end of
                # an allocated device buffer into unmapped memory -> "illegal memory access"
                # in every configuration (a program-level crash bypasses tier stripping
                # entirely, so it fires identically regardless of which blocks survived).
                results[config] = {
                    "verdict": "detected", "t1": "not_available", "t2": "not_available",
                    "witness": "not_available", "t3": {}, "catching_observer": ["crash"],
                    "reason": err_i,
                }
            else:
                results[config] = {
                    "verdict": "UNRESOLVED", "reason": err_b or err_i, "catching_observer": None,
                }
            continue

        catching = []

        # T1: self-contained, from the INJECTED build only.
        t1_verdict = "not_available" if config == "minus_t1" else parse_t1_verdict(out_i)
        if t1_verdict == "FAIL":
            catching.append("tier1")

        # T2 + witness: one comparator call (whenever ANY PASSTHROUGH label survived this
        # config's strip), then split into two INDEPENDENT verdicts by which label(s)
        # actually mismatched -- never skip the call just because ordinary buffers are gone,
        # or minus-T2 would silently lose witness detection too (the whole point of
        # strip_t2_blocks_except_witness).
        t2_labels_present = set(PASSTHROUGH_LABEL_RE.findall(inj_main))
        ordinary_labels = t2_labels_present - {"witness_chain_state"}
        witness_present = "witness_chain_state" in t2_labels_present

        failing_labels: set[str] = set()
        if t2_labels_present:
            cmp_verdict, cmp_diag = verify.compare_passthrough_outputs(
                out_b, out_i, tolerance=1e-6, buffer_classification=buffer_classification)
            if cmp_verdict == "FAIL":
                failing_labels = {f["label"].split("#")[0] for f in cmp_diag.get("failures", [])}

        t2_verdict = "not_available" if not ordinary_labels else (
            "FAIL" if failing_labels & ordinary_labels else "PASS"
        )
        witness_verdict = "not_available" if not witness_present else (
            "FAIL" if "witness_chain_state" in failing_labels else "PASS"
        )
        if t2_verdict == "FAIL":
            catching.append("tier2")
        if witness_verdict == "FAIL":
            catching.append("witness")

        # T3: internal-consistency check per active companion, using the INJECTED companion's
        # own perturbed output vs the INJECTED main file's own unperturbed reference.
        t3_verdicts = {}
        if config == "minus_t3":
            t3_verdicts = {"*": "not_evaluated"}
        else:
            for fname, comp_code in inj_t3.items():
                rel = match_companion_to_relation(fname, relations)
                if rel is None:
                    t3_verdicts[fname] = "not_available"
                    continue
                obs_points = rel.get("observation_points", [])
                if config == "minus_t2" and not any(op in t2_labels_present for op in obs_points):
                    t3_verdicts[fname] = "not_available"
                    continue
                ok_c, out_c, err_c = _compile_run(
                    comp_code, cfg_dir, f"t3_{os.path.splitext(fname)[0][-40:]}", run_args
                )
                if not ok_c:
                    t3_verdicts[fname] = f"UNRESOLVED: {err_c}"
                    continue
                verdict, diag = verify.check_metamorphic_relation(out_c, out_i, rel)
                # A FAIL composed entirely of "block missing" stats is a labeling/plumbing
                # failure (the relation's declared observation_points don't match either
                # file's actual printed label -- a real archived-file inconsistency, found
                # 2026-08-25 across 5 kernels: bsearch, chacha20, entropy, extrema,
                # segment-reduce), NOT a genuine semantic catch. Recording it as FAIL would
                # silently manufacture a false "T3 detected this" result. Reclassify as
                # not_available so it's honestly excluded rather than counted either way.
                if verdict == "FAIL":
                    failure_stats = {f.get("stat") for f in diag.get("failures", [])}
                    if failure_stats and failure_stats <= {"metamorphic_block_missing", "passthrough_block_missing"}:
                        verdict = "not_available"
                t3_verdicts[fname] = verdict
        if any(v == "FAIL" for v in t3_verdicts.values()):
            catching.append("tier3")

        overall = "detected" if catching else "undetected"

        results[config] = {
            "verdict": overall,
            "t1": t1_verdict,
            "t2": t2_verdict,
            "witness": witness_verdict,
            "t3": t3_verdicts,
            "catching_observer": catching or None,
        }

    return results
