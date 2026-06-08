"""
Global constants for the LASSI-Verify analysis package.

All metric modules import from here to ensure consistent settings.
"""

# --- pass@k configuration ---
K_VALUES: list[int] = [1, 3, 5]
BOOTSTRAP_B: int = 5000       # bootstrap resamples for CI (all pass@k metrics)
MONTE_CARLO_SAMPLES: int = 500  # Monte Carlo samples for Metric 2 (runtime-reduction@k) only
CI_LEVEL: float = 0.95
RANDOM_SEED: int = 42

# --- Verdict vocabulary (Table 1: lassiee_valid) ---
# Trials with these verdicts do NOT count as a pass for any metric.
NON_PASS_TRIAL_VERDICTS: set[str] = {"INVALID", "SOURCE", "N/A"}

# --- Tier verdict vocabulary (Table 2: tier1/2/3_verdict) ---
TIER_PASS_VERDICTS: set[str] = {"PASS", "SKIP"}  # tier ran and code passed, or not applicable
TIER_FAIL_VERDICTS: set[str] = {"FAIL"}           # tier ran and code failed
TIER_NOT_RUN: set[str] = {"SKIP", "NA"}           # tier was not meaningfully tested

# --- Judge verdict vocabulary ---
JUDGE_VALID: str = "VALID"
JUDGE_INVALID: str = "INVALID"
# judge_baseline_verdict values that mean "no usable verdict was produced"
JUDGE_NO_VERDICT: set = {"NONE", None}

# Canonical attempt-log column name for the (raw, unmodified) LLM-judge verdict.
# Renamed from "judge_verdict" (pre-judge-as-baseline schema). The data loader maps
# the legacy name onto this canonical name when reading older CSVs.
JUDGE_BASELINE_COLUMN: str = "judge_baseline_verdict"
LEGACY_JUDGE_COLUMN: str = "judge_verdict"

# --- Verdict columns to normalize (strip whitespace + uppercase) ---
# Applied in data_loader to both df_trials and df_attempts where columns exist.
VERDICT_COLUMNS_TRIALS: list[str] = [
    "lassiee_valid",
    "base_v1_valid",
    "base_llm_tier1_verdict",
    "base_llm_tier2_verdict",
    "base_llm_tier3_verdict",
    "lassiee_clean_status",
    "zeroshot_compile_status",
    "zeroshot_execute_status",
    "zeroshot_tier1_verdict",
    "zeroshot_tier2_verdict",
    "zeroshot_tier3_verdict",
    "zeroshot_empirical_verdict",
]

VERDICT_COLUMNS_ATTEMPTS: list[str] = [
    "compile_status",
    "execute_status",
    "tier1_verdict",
    "tier2_verdict",
    "tier3_verdict",
    "judge_baseline_verdict",   # canonical (post-rename); legacy "judge_verdict" is mapped on load
    "final_verdict",
    "override_applied",
    "generation_type",
]

# --- Comma-separated list columns to parse into Python lists ---
LIST_COLUMNS_TRIALS: list[str] = [
    "tier3_active_relations",
    "tier3_inactive_relations",
]

LIST_COLUMNS_ATTEMPTS: list[str] = [
    "tier1_failed_labels",
    "tier2_failed_labels",
    "tier1_instrumented_labels",
    "tier2_instrumented_labels",
    "tier3_tested_relation_names",
]

# --- Boolean columns ---
BOOL_COLUMNS_ATTEMPTS: list[str] = [
    "is_final_attempt",
    "is_best_code",
    "tier1_sentinel_present",
]
# base_llm_verified_valid is tri-state (True/False/None) — handled separately.

# --- Float columns (Table 1) ---
FLOAT_COLUMNS_TRIALS: list[str] = [
    "source_v0_runtime_s",
    "source_verify_runtime_s",
    "lassiee_runtime_s",
    "lassiee_runtime_diff",
    "lassiee_clean_runtime_s",
    "lassiee_clean_runtime_diff",
    "lassiee_energy_J",
    "lassiee_energy_diff",
    "lassiee_clean_energy_J",
    "lassiee_clean_energy_diff",
    "source_v0_energy_J",
    "source_v0_avg_power_W",
    "pipeline_runtime_min",
    "tier2_max_relative_error",
    "zeroshot_runtime_s",
]

# --- Float columns (Table 2) ---
FLOAT_COLUMNS_ATTEMPTS: list[str] = [
    "runtime_s",
    "energy_J",
    "ave_power_W",
    "llm_temperature",
    "tier2_max_relative_error",
]

# --- Nullable int columns (Table 1) ---
INT_COLUMNS_TRIALS: list[str] = [
    "trial_number",
    "total_code_versions",
    "total_code_attempts",
    "total_selfcorrections",
    "tier1_selfcorrections",
    "tier2_selfcorrections",
    "tier3_selfcorrections",
    "compile_selfcorrections",
    "execute_selfcorrections",
    "global_counter_at_end",
    "tier2_pass_count",
    "tier2_fail_total",
    "tier3_pass_count",
    "tier3_fail_total",
    "judge_false_positives",
    "judge_false_negatives",
]

# --- Nullable int columns (Table 2) ---
INT_COLUMNS_ATTEMPTS: list[str] = [
    "trial_number",
    "code_block_index",
    "attempt_number",
    "tier1_assertions_total",
    "tier1_assertions_passed",
    "tier1_assertions_failed",
    "tier2_blocks_total",
    "tier2_blocks_matched",
    "tier2_blocks_failed",
    "tier3_relations_tested",
    "tier3_relations_passed",
    "tier3_relations_failed",
]
