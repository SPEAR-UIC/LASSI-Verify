"""
Provenance schema for the Step 4 injected-defect corpus, finalized in
planning/LASSI_Verify_Phase2_Bridge.md's 2026-08-23 design-proper entry (writing agent's
proposed field set, adopted verbatim). One JSON file per operator; each holds the operator-level
metadata plus a list of per-instance records.
"""
from __future__ import annotations

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def new_operator_record(
    operator_id: str,
    family: str,               # "program-defect" | "instrumentation-defect"
    detector: str,             # "tiers_1_3" | "noninterference_gate_and_repair"
    observed_failure_class_generalized: str,
    evidence_pointer: str,
    mechanism: str,            # "generic_rule" | "hand_authored"
    mechanism_description: str,
    applicability_rule: str,
) -> dict:
    return {
        "operator_id": operator_id,
        "family": family,
        "detector": detector,
        "observed_failure_class_generalized": observed_failure_class_generalized,
        "evidence_pointer": evidence_pointer,
        "mechanism": mechanism,
        "mechanism_description": mechanism_description,
        "applicability_rule": applicability_rule,
        "per_instance": [],
    }


def new_instance_record(
    kernel: str,
    base_id: dict,             # {experiment_dir, trial, code_version}
    diff: str,
    neutrality_verdict: str = "UNRESOLVED",   # NEUTRAL | BEHAVIOR_CHANGING | UNRESOLVED
    decisive_test_record: dict | None = None,
    in_place_kernel_metadata: dict | None = None,
) -> dict:
    return {
        "kernel": kernel,
        "base_id": base_id,
        "diff": diff,
        "neutrality_verdict": neutrality_verdict,
        "decisive_test_record": decisive_test_record or {},
        "in_place_kernel_metadata": in_place_kernel_metadata,
        "tier_configuration_results": {
            "full_stack": None,
            "minus_t1": None,
            "minus_t2": None,
            "minus_t3": None,
            "minus_witness": None,
            "catching_observer": None,
        },
    }


def save_operator_record(record: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{record['operator_id']}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def load_operator_record(operator_id: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{operator_id}.json")
    with open(path) as f:
        return json.load(f)
