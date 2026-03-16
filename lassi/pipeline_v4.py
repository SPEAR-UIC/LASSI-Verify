"""
LASSI-Verify pipeline v4
Here, we first embrace the capability of the base LLM
by running the initial refactor as a zero-shot, no context attempt.
Then, subsequent iterations, we bring in self-prompting and background context
to see if we can improve the energy efficiency outcome further.

Then, we split up the base LLM (without any help) and the base LLM + self-correction
9/3/2015:
For focus, we will not do the base LLM + self-correction. Just base LLM then LASSI for a pass@k metric.
"""

import os
import csv
import json
import re
import time
import gc
from datetime import datetime
from zoneinfo import ZoneInfo
import lassi.code_manager
import lassi.llm_methods
import lassi.token_analysis
import lassi.llm_config
import lassi.get_power
import lassi.pipeline_methods
import lassi.verify
import lassi.monitor
from lassi.get_power import report_gpu_memory, query_gpu_state
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary
from lassi.gpu_detector import detect_gpu_vendors



def _build_attempt_record(app_name, trial_number, code_block_index, attempt_number,
                          is_final_attempt, generation_type, sc_trigger,
                          llm_temperature, compile_status, execute_status,
                          runtime_s, energy_J, ave_power_W,
                          tier1_verdict, tier1_instrumented_labels, tier1_diagnostic,
                          tier2_verdict, tier2_instrumented_labels, tier2_diagnostic,
                          tier3_verdict, tier3_diagnostic,
                          judge_verdict, final_verdict, override_applied,
                          is_best_code=False):
    """Build one Table 2 (per-attempt) row dict for the attempt log CSV.
    Called at each self-correction overwrite (intermediate attempt) and after the
    judge + override block (final attempt). Buffered in attempt_log_buffer; written
    at trial end once is_best_code can be set on the winning row."""
    t1d = tier1_diagnostic or {}
    t2d = tier2_diagnostic or {}
    t3d = tier3_diagnostic or {}
    failures_t2 = t2d.get("failures", [])
    # Tier 2 failed block labels: exclude block_missing and parse_error entries
    # (those are structural, not label-specific failures)
    t2_failed_labels = sorted(set(
        f.get("label", "") for f in failures_t2
        if f.get("stat") not in ("block_missing", "parse_error") and f.get("label", "")
    ))
    # Count distinct failed blocks (not individual stat-level failures within a block).
    # failures_t2 has one entry per failing stat; a single block can generate many stat failures.
    t2_failed_block_count = len(set(
        f.get("label", "") for f in failures_t2 if f.get("label", "")
    ))
    return {
        "app_name": app_name,
        "trial_number": trial_number,
        "code_block_index": code_block_index,
        "attempt_number": attempt_number,
        "is_final_attempt": is_final_attempt,
        "attempt_id": f"{app_name}_T{str(trial_number).zfill(2)}_cb{code_block_index}_a{attempt_number}",
        "generation_type": generation_type,
        "self_correction_trigger": sc_trigger,
        "llm_temperature": round(llm_temperature, 1),
        "compile_status": compile_status,
        "execute_status": execute_status,
        "runtime_s": runtime_s,
        "energy_J": energy_J,
        "ave_power_W": ave_power_W,
        # --- Tier 1 ---
        "tier1_verdict": tier1_verdict,
        "tier1_instrumented_labels": tier1_instrumented_labels,
        "tier1_assertions_total": t1d.get("pass_count", 0) + t1d.get("fail_count", 0),
        "tier1_assertions_passed": t1d.get("pass_count", 0),
        "tier1_assertions_failed": t1d.get("fail_count", 0),
        # sentinel_present: True if FAIL-only format (sentinel found, no explicit PASS lines),
        # or if any assertion lines appeared at all (legacy format).
        "tier1_sentinel_present": bool(t1d.get("passing_labels") or t1d.get("failing_labels") or
                                       (tier1_verdict == "PASS" and t1d.get("pass_count", 0) == 0)),
        "tier1_failed_labels": ",".join(t1d.get("failing_labels", [])),
        # --- Tier 2 ---
        "tier2_verdict": tier2_verdict,
        "tier2_instrumented_labels": tier2_instrumented_labels,
        "tier2_blocks_total": t2d.get("blocks_compared", 0),
        "tier2_blocks_matched": max(0, t2d.get("blocks_compared", 0) - t2_failed_block_count),
        "tier2_blocks_failed": t2_failed_block_count,
        "tier2_failed_labels": ",".join(t2_failed_labels),
        "tier2_max_relative_error": t2d.get("max_relative_error"),
        # --- Tier 3 ---
        "tier3_verdict": tier3_verdict,
        "tier3_tested_relation_names": ",".join(t3d.get("tested_relations", [])),
        "tier3_relations_tested": len(t3d.get("tested_relations", [])),
        "tier3_relations_passed": t3d.get("relations_passed", 0),
        "tier3_relations_failed": t3d.get("relations_failed", 0),
        "tier3_failed_relation": t3d.get("relation_name") or None,
        "tier3_failed_type": t3d.get("relation_type") or None,
        # --- Judge + combined verdict (final attempt only; None for intermediate) ---
        "judge_verdict": judge_verdict,
        "final_verdict": final_verdict,
        "override_applied": override_applied,
        "is_best_code": is_best_code,
    }


# Column order for the per-attempt CSV (Table 2 in the SC26 data schema).
# Must match the keys returned by _build_attempt_record().
ATTEMPT_LOG_FIELDNAMES = [
    "app_name", "trial_number", "code_block_index", "attempt_number",
    "is_final_attempt", "attempt_id", "generation_type", "self_correction_trigger",
    "llm_temperature", "compile_status", "execute_status",
    "runtime_s", "energy_J", "ave_power_W",
    # Tier 1
    "tier1_verdict", "tier1_instrumented_labels",
    "tier1_assertions_total", "tier1_assertions_passed", "tier1_assertions_failed",
    "tier1_sentinel_present", "tier1_failed_labels",
    # Tier 2
    "tier2_verdict", "tier2_instrumented_labels",
    "tier2_blocks_total", "tier2_blocks_matched", "tier2_blocks_failed",
    "tier2_failed_labels", "tier2_max_relative_error",
    # Tier 3
    "tier3_verdict", "tier3_tested_relation_names",
    "tier3_relations_tested", "tier3_relations_passed", "tier3_relations_failed",
    "tier3_failed_relation", "tier3_failed_type",
    # Judge + combined verdict
    "judge_verdict", "final_verdict", "override_applied", "is_best_code",
]


def auto_code_llmgeneration_pipeline(prompts,
                                     this_llm_systemcode_prompt,
                                     this_llm_prompt,  # The refactoring instruction prompt
                                     llm_source, 
                                     model,
                                     model_coder,
                                     model_judge,
                                     num_ctx,
                                     num_ctx_coder,
                                     # arrayormatrix_size_range,
                                     # thread_count_range,
                                     use_context,
                                     use_context_summaries_only,
                                     few_shot_learning,
                                     run_code_analysis,
                                     input_context,
                                     input_code_file_subdirectory,
                                     input_code,  # <-- THE SOURCE CODE
                                     lines_of_code,
                                     # target_code,
                                     target_lang,
                                     refactored_code_extension,
                                     source_lang,
                                     code_compiler,
                                     code_compiler_kwds,
                                     build_script,
                                     make_script,
                                     code_build_script,
                                     directory,
                                     file_name,
                                     source_file_path,
                                     display_response,
                                     test_with_error,
                                     generate_test_case,
                                     yes_execute,
                                     exe_input_parameters,
                                     meta_data_file,
                                     power_measure,
                                     # test_code_first DEPRECATED: Stage 0 (source code test) is now always required
                                     #   to collect baseline metrics for both SOURCE_orig and SOURCE_verify comparisons.
                                     yes_power_feedback,
                                     dt_string,
                                     baseline_oneshot_goal,
                                     use_self_prompting,
                                     vary_llm_temp,
                                     experiment,
                                     verifyprompt,
                                     trial_number,
                                     experiment_data_directory,
                                     experiment_data_file_object,
                                     experiment_data,
                                     app_name,
                                     run_tier2=False,        # Enable Tier 2 empirical equivalence checking
                                     tier2_tolerance=1e-6,   # Relative tolerance for passthrough comparison
                                     run_llm_judge=True,     # Enable LLM-as-a-judge functional equivalence check (pre-Tier-1/2 stop-gap; disable for pure three-tier mode)
                                     run_tier3=False,        # Enable Tier 3 metamorphic testing
                                     tier3_tolerance=1e-5,   # Default relative tolerance for metamorphic relation checks (slightly looser than Tier 2 to account for two independently-accumulated FP runs)
                                     attempt_log_file_path=None  # Path to per-attempt CSV (Table 2 ★); None = skip writing
                                    ):

    # -----------------
    # ----- SETUP -----
    # -----------------
    now = datetime.now(ZoneInfo("America/Chicago"))
    dt_string_data = now.strftime("%m-%d-%Y_t%H_%M_%S")
    experiment_data["trial_number"] = trial_number
    experiment_data["timestamp"] = dt_string_data
    experiment_data["experiment"] = experiment
    experiment_data["app_name"] = file_name
    experiment_data["code_line_count"] = lines_of_code
    experiment_data["llm"] = model
    experiment_data["llm_coder"] = model_coder
    experiment_data["llm_judge"] = model_judge
    
    actual_energy = 0
    actual_ave_power = 0
    idle_energy_joules = 0
    idle_ave_power = 0
    total_energy_joules_refactor = 0
    ave_power_refactor = 0
    duration_refactor = 0
    sourcecode_total_energy_joules = 0
    sourcecode_ave_power = 0
    duration = 0
    self_correction_max = 10
    self_correction_power_counter_max = 10
    self_correction_verify_counter = 0    # Global total Tier 2 + Tier 3 failures across all generated codes; pipeline-level safety net (Tier 1 excluded — it has its own per-code budget)
    self_correction_verify_counter_max = 15  # Global limit (increased from 10 to support multiple codes × per-code budget)
    self_correction_verify_counter_code = 0   # Per-code Tier 2 self-correction attempts; resets on fresh code generation
    self_correction_verify_counter_code_max = 5  # Per-code limit: if exhausted, mark INVALID and generate a fresh code
    tier2_exhausted = False   # Set True when per-code Tier 2 budget is exhausted; forces INVALID after judge block
    # --- Tier 3: Metamorphic Testing state variables ---
    tier3_contract = {}             # Metamorphic contract from Stage 0 LLM call
    tier3_active_relations = []     # Relations that passed the SOURCE_verify filter
    self_correction_tier3_counter_code = 0    # Per-code Tier 3 retry count (separate from Tier 2 counter)
    self_correction_tier3_counter_code_max = 3  # Tier 3 gets fewer retries than Tier 2 (structural math violations are harder to fix)
    tier3_exhausted = False         # Set True when per-code Tier 3 budget exhausted; forces INVALID
    tier3_verdict = "SKIP"          # Most recent Tier 3 verdict for the current code
    tier3_diagnostic = {}           # Most recent Tier 3 first-failure diagnostic
    tier3_inactive_relations = []   # Relations filtered out at Stage 0 source filter (complement of tier3_active_relations)
    # --- Tier 1: Invariant Assertion Check state variables ---
    self_correction_tier1_counter_code = 0    # Per-code Tier 1 retry count; resets on fresh code
    self_correction_tier1_counter_code_max = 3  # Per-code limit (matches Tier 3; assertion failures are structural)
    tier1_verdict = "SKIP"          # Most recent Tier 1 verdict for the current code
    tier1_diagnostic = {}           # Most recent Tier 1 assertion diagnostic
    exe_output_source = ""
    source_orig_stdout = ""    # stdout from SOURCE_orig execution (Stage 0)
    source_verify_stdout = ""  # stdout from SOURCE_verify execution (Stage 1)
    passthrough_spec_output = ""   # Tier 2 source-as-specification reference (= source_verify_stdout)
    passthrough_outputs = []       # per-code-block stdout list (indexed like code_blocks)
    energy_efficiency_refactor_success = False
    energy_efficiency_iteration = 0
    code_version = 0
    refactoring_plan = "None"
    tracking_best_version = 0  # to account for the source code at index 0
    get_system_info_flag = True  # Used in the selfprompting context builder
    valid_code_flag = "INVALID"  # default value, or "VALID"
    force_save_baseline_llm = False
    change_strategy = False
    code_gen_temp = 0.2  # default LLM temperature
    code_gen_temp_max = 1.0
    energy_history = []
    power_history = []
    runtime_history = []
    valid_code_history = []
    code_blocks = []  # Collection of all codes generated
    bad_refactor_plans = []  # Collection of refactoring plan texts that did not result in 'good' code.
    iteration_log = {}  # initiate the iteration log for each code version
    verification_log = {}  # iniiate a dict log for code verifications
    # iteration_log[version] = {"self_correction": 0, "energy_refactor": 0, "LLM_temperature": 0}
    # --- Table 2 per-attempt log state variables ---
    attempt_number = 0              # Attempt counter within the current code_block_index; resets on fresh generation
    attempt_log_buffer = []         # Accumulates Table 2 rows during the trial; written to CSV at trial end
    compile_sc_count = 0            # Self-corrections triggered by compile errors
    execute_sc_count = 0            # Self-corrections triggered by execute errors
    tier1_sc_count = 0              # Self-corrections triggered by Tier 1 assertion failures
    tier2_sc_count = 0              # Self-corrections triggered by Tier 2 equivalence failures
    tier3_sc_count = 0              # Self-corrections triggered by Tier 3 metamorphic failures
    current_sc_trigger = None       # Trigger string (e.g. "tier1_fail:output_finite") carried into next iteration's attempt record
    current_generation_type = "lassi_fresh"   # Generation type for the current attempt (set at top of each while iteration)
    zeroshot_compile_status = "skipped"       # Compile status for zero-shot code; updated when zero-shot runs
    zeroshot_execute_status = "skipped"       # Execute status for zero-shot code
    best_refactored_code_dict = {}
    best_refactored_code_judge = ""
    best_refactoring_plan = ""
    code_comparison_response = ""
    input_content_llmjudge_dict = {"source_code": "",
                                   "source_stdout": "",
                                   "source_metrics": "",
                                   "refactored_code_lang": "",
                                   "refactored_code": "",
                                   "refactored_stdout": "",
                                   "refactored_metrics": "",
                                  }  # Input values to be used by the LLM-as-a-judge phase

    # Add source code to the first index of code_blocks
    # LASSI-V update: append this after the invariant assertions are inserted
    # code_blocks.append(input_code)
    input_content_llmjudge_dict["source_code"] = input_code

    input_content_llmjudge_dict["refactored_code_lang"] = target_lang
        
    lassi.monitor.clear_log()   # reset monitor so saved log contains only this run
    print("\n----- WELCOME TO PIPELINE v4! CODE -----\n")

    vendors = detect_gpu_vendors()
    if not vendors:
        print("❌ PIPELINE ERROR: No GPUs detected. Stopping LASSI.")
        # Stop pipeline here
    else:
        detected_vendor = vendors[0]['vendor']
        print(f"🔍 Detected GPU vendor: {detected_vendor.upper()}")

    report_gpu_memory()
        
    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END) 
    
    # --------------------------------
    # ----- STAGE 0: TEST SOURCE -----
    # --------------------------------
    # Test compile and execute the source code to validate compiler and execution environment.
    # Collect baseline energy measurements.
    """Initial stage to run the SOURCE CODE and gather metrics."""
    (idle_energy_joules, idle_ave_power, input_content_llmjudge_dict, sourcecode_total_energy_joules, sourcecode_ave_power, duration,
     energy_history, power_history, runtime_history, valid_code_history,
     source_orig_stdout) = lassi.pipeline_methods.test_source_code(app_name, experiment_data_directory, code_version, trial_number, source_file_path, build_script, make_script, directory, source_lang, target_lang,
                                                                                            exe_input_parameters, code_compiler, code_compiler_kwds,
                                                                                            yes_execute, power_measure, input_content_llmjudge_dict,
                                                                                            energy_history, power_history, runtime_history, valid_code_history
                                                                                           )
    # Save the output details for this as the "best" code version for metadata record
    # This is necessary to capture now if no better refactored code is generated.
    # Snapshot with dict() so subsequent runs don't overwrite the stored values.
    best_refactored_code_dict = dict(input_content_llmjudge_dict)

    experiment_data["source_v0_energy_J"] = sourcecode_total_energy_joules
    experiment_data["source_v0_avg_power_W"] = sourcecode_ave_power
    experiment_data["source_v0_runtime_s"] = duration

    # Append original (uninstrumented) source code as code_blocks[0] = SOURCE_orig.
    # Used by the vanilla LLM judge and as the non-instrumented comparison baseline.
    code_blocks.append(input_code)
    valid_code_history[-1] = "SOURCE_orig"  # Override "SOURCE" label from test_source_code

    print(print("\n----- END OF STAGE 0: TEST SOURCE -----\n"))
    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)


    
    # ------------------------------------------
    # ----- STAGE 1: VANILLA LLM REFACTOR -----
    # ------------------------------------------
    """
    STAGE 1: ZERO-SHOT REFACTOR - STANDALONE BASE LLM CAPABILITY
    + Implement a straightforward and strong user prompt only.
    + No context and no self-correction feedback. Observe capability of unassisted base LLM.
    + NOTE: This code generation is implemented within the LASSI-EE iteration loop below.
    """
    # Start LASSI-EE Pipeline Timer
    start_time = time.time()

    # print("\n\n----- " + color.BOLD + color.BLUE + "STARTING STAGE 1: STANDALONE BASE LLM REFACTORING " + color.END + " -----\n")

    stage = "stage1"
    zero_shot = True
    generate_context_flag = False
    content = ""
    content_summary_only = ""
    compile_result = ""
    execute_result = ""
    error_message = ""
    error_source = ""  # "compiled" or "executed"
    code_block = ""
    self_correction_counter = 0
    self_correction_power_counter = 0
    fresh_generation_count = 0              # Table 1 ★: count of distinct fresh LASSI code generations started (includes codes that never executed)

    # Pull in prompts used for base (vanilla) LLM
    # Same prompting strategy for STAGE 1 and STAGE 2
    original_context_content_prompt, refactor_instructions, system_prompt = lassi.pipeline_methods.selfprompting_context_builder(
        generate_context_flag,
        get_system_info_flag,
        source_lang=source_lang,
        llm_source=llm_source,
        model=model,
        num_ctx=num_ctx,
        target_lang=target_lang,
        display_response=display_response,
        input_code=input_code,
        total_energy_joules=sourcecode_total_energy_joules,
        ave_power=sourcecode_ave_power,
        verifyprompt=verifyprompt,
        power_measure=power_measure,
    )

    # Locally hosted open-source LLMs
    # Unload model first if we want a different coding model
    if (model_coder != model and llm_source == "ollama"):
        # CHANGE MODEL FOR CODE GENERATION
        print(" (... changing LLM for code gen)")
        lassi.llm_config.unload_ollama_model(model)
        time.sleep(10)  # Pause for 10 seconds
        report_gpu_memory()
    
    # Clean up Python-level objects
    gc.collect()
    time.sleep(5)

    # Generate Code Analysis of source code for verification
    # Before we begin the LASSI-EE iterations
    if run_code_analysis:
        code_analysis = lassi.pipeline_methods.generate_code_analysis(
                            source_lang=source_lang,
                            input_code=input_code,
                            llm_source=llm_source,
                            model=model,
                            num_ctx=num_ctx,
                            system_prompt=system_prompt,
                            context_prompt=original_context_content_prompt,
                            display_response=display_response,
                            verifyprompt=verifyprompt
                        )
    else:
        code_analysis = "None"
    verification_log["sourcecode_analysis"] = code_analysis
    # Tier 2 verification log fields (initialized here; populated during the main loop)
    verification_log["run_llm_judge"] = run_llm_judge
    verification_log["tier2_enabled"] = run_tier2
    verification_log["tier2_tolerance"] = tier2_tolerance
    verification_log["tier2_spec_output"] = ""          # populated after Stage 1 execution
    verification_log["tier2_results"] = []              # per-iteration list of {code_version, verdict, diagnostic}
    verification_log["verified_pass_count"] = 0         # count of iterations where Tier 2 PASS
    verification_log["judge_discrepancies"] = []        # cases where LLM judge and three-tier verdict disagree
    # judge_discrepancy entry schema: {code_version, zero_shot, type ("false_positive"|"false_negative"), judge_verdict, tier2_verdict, final_verdict}
    # Tier 3 verification log fields (initialized here; populated after Stage 0 contract generation)
    verification_log["tier3_enabled"] = run_tier3
    verification_log["tier3_tolerance"] = tier3_tolerance
    verification_log["tier3_contract"] = {}          # populated after Stage 0 contract generation
    verification_log["tier3_active_relations"] = []  # populated after Stage 0 source variant filter
    verification_log["tier3_results"] = []           # per-iteration list of {code_version, verdict, results}
    verification_log["tier3_pass_count"] = 0


    # -----------------------------------------------------------
    # ----- STAGE 1: INSERT VERIFY INSTRUMENTATION -----
    # -----------------------------------------------------------
    # With source code that can compile and execute on the given system and device,
    #  we next prompt LLM inference to insert Invariant Assertians into the source code
    #  with protective commenting to ensure subsequent LLM-driven code transformations do not
    #  change the assertians. (Next TODO is to insert passthrough data dump blocks.)
    # Save original source before instrumentation so the vanilla LLM zero-shot stage
    #  receives the unaltered source code (not the assertion-instrumented version).
    # original_input_code is now accessible as code_blocks[0] (SOURCE_orig).
    original_input_code = input_code

    input_code = lassi.verify.code_insert_asserts(
        input_content_llmjudge_dict["source_code"],
        llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=True
    )

    input_content_llmjudge_dict["source_code"] = input_code

    # --- Tier 2: Insert passthrough data dump blocks into the already-asserted code ---
    # code_insert_passthrough() is called on the Tier-1-instrumented code so both assertion
    # blocks and passthrough blocks appear in SOURCE_verify and are carried through all
    # subsequent LLM-generated transformations unaltered.
    # The passthrough prompt explicitly preserves existing LASSI-VERIFY ASSERTIONS blocks.
    if run_tier2:
        passthrough_code = lassi.verify.code_insert_passthrough(
            input_code,  # already has Tier 1 assertions
            llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=True
        )
        if passthrough_code != "":
            input_code = passthrough_code
            input_content_llmjudge_dict["source_code"] = input_code
            print(f"\n----- LASSI-Verify Tier 2: passthrough blocks inserted into SOURCE_verify -----")
        else:
            print(f"\n----- LASSI-Verify Tier 2: passthrough insertion returned empty — Tier 2 will be SKIPPED -----")
            run_tier2 = False  # Disable Tier 2 for this run if insertion failed

    # Write the instrumented source to disk so test_source_code compiles and runs it —
    # not the original .cu file. This gives true SOURCE_verify metrics that include
    # any overhead from the inserted invariant assertion checks (and passthrough blocks if Tier 2 enabled).
    # Saved to the experiment folder (not alongside the original source) — it is an experiment artifact,
    # not a permanent modification of the benchmark source. After this point, the pipeline always uses
    # code_blocks[1] (in-memory) and never reads this file again.
    _, source_ext = os.path.splitext(source_file_path)
    verify_source_file_path = os.path.join(experiment_data_directory, file_name + "_verify" + source_ext)
    with open(verify_source_file_path, 'w') as _vf:
        _vf.write(input_code)
    print(f"\n----- LASSI-Verify: instrumented source written to {verify_source_file_path} -----")

    """Run the instrumented source code to capture SOURCE_verify baseline metrics.
    Compiles and executes verify_source_file_path (the assertion-instrumented .cu file),
    so history[1] reflects the true overhead of the invariant assertion instrumentation."""
    (idle_energy_joules, idle_ave_power, input_content_llmjudge_dict, sourcecode_total_energy_joules, sourcecode_ave_power, duration,
     energy_history, power_history, runtime_history, valid_code_history,
     source_verify_stdout) = lassi.pipeline_methods.test_source_code(app_name, experiment_data_directory, code_version, trial_number, verify_source_file_path, build_script, make_script, directory, source_lang, target_lang,
                                                                                            exe_input_parameters, code_compiler, code_compiler_kwds,
                                                                                            yes_execute, power_measure, input_content_llmjudge_dict,
                                                                                            energy_history, power_history, runtime_history, valid_code_history
                                                                                           )
    valid_code_history[-1] = "SOURCE_verify"  # Override "SOURCE" label from test_source_code
    experiment_data["source_verify_runtime_s"] = runtime_history[1]   # Wall-clock time of instrumented source (Table 1 ★)

    # Tier 2 source-as-specification reference: stdout from SOURCE_verify execution.
    # Per the source-as-specification premise, SOURCE_verify's observable behavior is the
    # sole correctness criterion. Any LASSI-generated transformation must preserve this behavior.
    passthrough_spec_output = source_verify_stdout
    verification_log["tier2_spec_output"] = passthrough_spec_output

    # Initialize per-code-block stdout list (indexed like code_blocks):
    #   [0] = SOURCE_orig stdout (no passthrough instrumentation)
    #   [1] = SOURCE_verify stdout (Tier 2 reference)
    passthrough_outputs = [source_orig_stdout, source_verify_stdout]

    # --- Tier 3: Generate metamorphic contract and filter active relations (Stage 0) ---
    # run_metamorphic_variants is called here with SOURCE_verify to determine which
    # relations are applicable to this computation. Relations that fail for SOURCE_verify
    # are marked inactive and silently skipped for all generated codes in the main loop.
    # This achieves zero false positives from inapplicable MRs (mechanizes the manual
    # MR validation step identified as future work in Li et al. 2025, Directions 2-3).
    if run_tier3 and run_tier2:  # Tier 3 requires Tier 2 passthrough baseline
        print(f"\n----- LASSI-Verify Tier 3: generating metamorphic contract -----")
        tier3_contract = lassi.verify.generate_metamorphic_contract(
            input_code,  # SOURCE_verify (Tier 1 + Tier 2 instrumented)
            llm_source, model, num_ctx, system_prompt, display_response,
            verifyprompt=True
        )
        if tier3_contract and tier3_contract.get("relations"):
            print(f"\n----- LASSI-Verify Tier 3: running SOURCE_verify metamorphic variants (filter) -----")
            _, filter_results, _ = lassi.verify.run_metamorphic_variants(
                input_code,                 # SOURCE_verify code
                tier3_contract["relations"],
                tier3_contract,
                passthrough_spec_output,    # SOURCE_verify's own Tier 2 passthrough output (Tier 3 baseline)
                llm_source, model, num_ctx, system_prompt,
                display_response,
                directory, file_name, code_compiler, code_compiler_kwds,
                build_script, make_script, code_build_script,
                exe_input_parameters,
                experiment_data_directory=experiment_data_directory,
                verifyprompt=True
            )
            # Mark relations active/inactive based on source filter results.
            # "PASS" → relation is applicable; "FAIL" or "SKIP" → filter inactive.
            tier3_active_relations = [
                r for r in tier3_contract["relations"]
                if r.get("source_status") == "PASS"
            ]
            _active_names = {r["name"] for r in tier3_active_relations}
            tier3_inactive_relations = [
                r["name"] for r in tier3_contract["relations"]
                if r["name"] not in _active_names
            ]
            n_total  = len(tier3_contract["relations"])
            n_active = len(tier3_active_relations)
            print(f"\n----- LASSI-Verify Tier 3: {n_active}/{n_total} relations active after source filter -----")
            if n_active == 0:
                print(f"      (All relations failed or skipped for SOURCE_verify — Tier 3 will be SKIP for all generated codes)")
        else:
            print(f"\n----- LASSI-Verify Tier 3: contract generation failed — Tier 3 disabled -----")
            run_tier3 = False  # Disable Tier 3 for this run
        verification_log["tier3_contract"] = tier3_contract
        verification_log["tier3_active_relations"] = [r["name"] for r in tier3_active_relations]
    elif run_tier3 and not run_tier2:
        print(f"\n----- LASSI-Verify Tier 3: Tier 2 is disabled — Tier 3 requires Tier 2 passthrough baseline; Tier 3 disabled -----")
        run_tier3 = False

    # Append instrumented source code as code_blocks[1] = SOURCE_verify.
    # LASSI pipeline refactors from this baseline; the LASSI judge compares against it.
    code_blocks.append(input_code)
    
    
    
    # --------------------------------------------------------
    # INITIATE LASSI-EE LLM CODE GENERATION ITERATIVE PIPELINE
    while (
        (compile_result != "" or stage == "stage1" or stage == "stage2")
        and self_correction_counter <= self_correction_max
        and self_correction_power_counter <= self_correction_power_counter_max
        and self_correction_verify_counter <= self_correction_verify_counter_max
    ):
        # Save compile_result at the start of this iteration before compile/execute can modify it.
        # Used later to detect whether we're entering a Tier 2 self-correction retry vs. a fresh code.
        compile_result_before_generation = compile_result

        # --- Table 2 ★: determine generation type and attempt number for the attempt record ---
        # generation_type and attempt_number track what kind of code generation this iteration performs.
        # Only updated for LASSI codes (not zero_shot); zero-shot is logged separately in Table 1.
        if not zero_shot:
            _tier_retry_map = {
                "tier1_failure": "self_correction_tier1",
                "tier2_failure": "self_correction_tier2",
                "tier3_failure": "self_correction_tier3",
            }
            if compile_result_before_generation in ("", "energy_efficiency"):
                # Fresh LASSI code generation — new code_block_index, reset attempt counter
                current_generation_type = "lassi_fresh"
                attempt_number = 1
                fresh_generation_count += 1     # Table 1 ★: count all fresh code starts regardless of outcome
            elif compile_result_before_generation in _tier_retry_map:
                current_generation_type = _tier_retry_map[compile_result_before_generation]
                attempt_number += 1
            elif compile_result_before_generation == "execute_error":
                current_generation_type = "self_correction_execute"
                attempt_number += 1
            else:
                # Any other non-empty, non-tier compile_result is a compile error retry
                current_generation_type = "self_correction_compile"
                attempt_number += 1

        # After the initial zero-shot and iteration attempts until no improvement, then
        # Update the prompt strategy with context and self-prompting responses.
        if change_strategy and not zero_shot and compile_result == "energy_efficiency":
            print("\n\n" + color.BOLD + "CHANGE PROMPT STRATEGY! -----" + color.END)
            print("\n\n----- " + color.BOLD + color.BLUE + "STARTING LASSI: ITERATIVE SELF-PROMPTING WITH CONTEXT AND SELF-CORRECTION" + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            generate_context_flag = True
            # Reset tracking to SOURCE_verify (code_blocks[1]) as the LASSI baseline.
            # LASSI refactors from the assertion-instrumented source and must beat its metrics.
            tracking_best_version = 1
            original_context_knowledge, context_prompt, system_prompt = lassi.pipeline_methods.selfprompting_context_builder(
                generate_context_flag,
                get_system_info_flag,
                source_lang=source_lang,
                input_context=input_context,
                llm_source=llm_source,
                model=model,
                num_ctx=num_ctx,
                target_lang=target_lang,
                display_response=display_response,
                use_context_summaries_only=use_context_summaries_only,
                input_code=code_blocks[tracking_best_version],
                original_code=original_input_code,  # Pass uninstrumented source for code description step
                total_energy_joules=energy_history[tracking_best_version] if power_measure else runtime_history[tracking_best_version],
                ave_power=power_history[tracking_best_version] if power_measure else 0.0,
                verifyprompt=verifyprompt,
                power_measure=power_measure,
            )
            change_strategy = False

        # Initial zero-shot refactor attempt    
        if zero_shot and compile_result == "" and stage == "stage1":
            # Set code_version to the index where baseLLM will be stored in code_blocks
            # (code_blocks already has [SOURCE_orig, SOURCE_verify], so len=2 = next index).
            # Using len(code_blocks) instead of code_version+1 keeps code_version equal to
            # the last code_blocks index after each successful append — no base offset.
            # get_power.py's existing local +1 for non-zero-shot executions still correctly
            # compensates: zero-shot executes with code_version=2 (energy file "v2" = baseLLM),
            # LASSI first executes with code_version=2 (+1 → "v3" = code_blocks[3]), etc.
            code_version = len(code_blocks)  # = 2: index where baseLLM will be appended
            print("\n\n----- " + color.BOLD + color.BLUE + "STARTING STAGE 1: BASE LLM REFACTORING WITHOUT SELF-CORRECTION " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            
            # BUILD REFACTING INSTRUCTION PROMPT
            instructions_prompt = refactor_instructions
            # Append the ORIGINAL (uninstrumented) source code -- vanilla LLM should not see assertion instrumentation
            instructions_prompt = instructions_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + original_input_code + "\n[/END OF CODE TO REFACTOR]"
            # Save the original context + energy efficiency context for re-use during energy efficiency correction
            standalone_basellm_prompt_tokens = lassi.token_analysis.token_count(instructions_prompt)

            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE ZERO-SHOT INSTRUCTIONS TASK PROMPT " + color.END + " (prompt token count, excluding code): " + str(standalone_basellm_prompt_tokens) + "\n")  #: " + instructions_prompt + "\n")  # + instructions_prompt + "\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "initial zero-shot attempt: prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, round(code_gen_temp,1), system_prompt, instructions_prompt, display_response, verifyprompt)  

        elif zero_shot and compile_result == "" and stage == "stage2":
            """
            STAGE X: ZERO-SHOT REFACTOR - BASE LLM CAPABILITY
            ** 09-06-2025 UPDATE: We will not run with BASELINE + self-correction (it's too redundant, not enough to spend time on at this phase in the research).
            + Implement a straightforward and strong user prompt only.
            + Enable self-correction feedback, iterate until generate "good code" that
              compiles, executes, passes LLM-as-a-Judge
            + Stopping condition through 0.2 increments of LLM temperature parameter until = 1.
            """
            print("\n\n----- " + color.BOLD + color.BLUE + "STARTING STAGE 2: BASE LLM REFACTORING WITH SELF-CORRECTION " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            
            # CALL THE LLM to generate the Stage 2 code refactoring using the source code
            # Prompting was built before WHILE loop initiates
            # report_gpu_memory()
            # Unload model first if we want a different coding model
            if (model_coder != model and llm_source == "ollama"):
                # CHANGE MODEL FOR CODE GENERATION
                print(" (... changing LLM for code gen)")
                lassi.llm_config.unload_ollama_model(model)
                time.sleep(10)  # Pause for 10 seconds
                report_gpu_memory()
        
            if llm_source == "ollama":
                time.sleep(10)
            report_gpu_memory()

            # BUILD REFACTING INSTRUCTION PROMPT
            instructions_prompt = refactor_instructions
            # instructions_prompt = instructions_prompt + "\n\n" + "[INSTRUCTIONS TO REFACTOR CODE TO IMPROVE ENERGY EFFICIENCY]\n" + refactor_instructions
            # Append the background knowledge and/or LLM-generated summaries
            # instructions_prompt = instructions_prompt + "\n\n" + original_context_content_prompt
            instructions_prompt_core_iterations = instructions_prompt  # Save off for next iteration
            # Append the ORIGINAL (uninstrumented) source code -- vanilla LLM should not see assertion instrumentation
            instructions_prompt = instructions_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + original_input_code + "\n[/END OF CODE TO REFACTOR]"
            # instructions_prompt = instructions_prompt + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(sourcecode_total_energy_joules, 2)) + " Joules at an average power of " + str(round(sourcecode_ave_power, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                 
            # Save the original context + energy efficiency context for re-use during energy efficiency correction
            original_ee_context_content_prompt = instructions_prompt_core_iterations
            original_ee_prompt_tokens = lassi.token_analysis.token_count(original_ee_context_content_prompt)

            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE ZERO-SHOT INSTRUCTIONS TASK PROMPT " + color.END + " (prompt token count, excluding code): " + str(original_ee_prompt_tokens) + "\n")  # + instructions_prompt + "\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "initial zero-shot attempt: prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, instructions_prompt, display_response, verifyprompt)
        
        elif compile_result == "tier1_failure" and not zero_shot:
            """
            Tier 1 (invariant assertion) verification failure self-correction.
            The generated code compiled and executed successfully but one or more invariant
            assertion blocks printed FAIL to stdout. All failing assertion labels are reported
            so the LLM has full visibility into every violated invariant.
            """
            print("\n\n----- " + color.BOLD + color.RED + "LASSI-VERIFY Tier 1 SELF-CORRECTION: " + str(self_correction_verify_counter) + color.END + " -----")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)

            tier1_diag_summary = tier1_diagnostic.get("summary", "(no diagnostic available)")
            failing_labels = tier1_diagnostic.get("failing_labels", [])
            failing_labels_str = ", ".join(f"[{l}]" for l in failing_labels)

            # Extract the full assertion block source for each failing label so the LLM knows
            # exactly what condition needs to evaluate to true — not just the label name.
            _block_re = re.compile(
                r'(// >>> BEGIN LASSI-VERIFY ASSERTIONS.*?// <<< END LASSI-VERIFY ASSERTIONS <<<)',
                re.DOTALL
            )
            failing_block_texts = []
            for _label in failing_labels:
                for _bm in _block_re.finditer(code_blocks[1]):  # SOURCE_verify is authoritative — unmodified by LLM
                    if f'[{_label}]' in _bm.group(1):
                        failing_block_texts.append(_bm.group(1).strip())
                        break
            failing_blocks_section = (
                "\n\n=== FAILING ASSERTION BLOCK(S) — source of the condition that printed FAIL ===\n"
                + "\n\n".join(failing_block_texts) +
                "\n=== END FAILING ASSERTION BLOCK(S) ===\n"
            ) if failing_block_texts else ""

            error_correction_prompt_intro = (
                "\n\nThe above code compiled and executed successfully but FAILED LASSI-Verify "
                "Tier 1 invariant assertion checking. One or more assertion blocks printed FAIL "
                "to stdout, indicating a mathematical correctness violation.\n\n"
                "=== TIER 1 DIAGNOSTIC ===\n"
                + tier1_diag_summary +
                "\n\nFailing assertion label(s): " + (failing_labels_str if failing_labels_str else "(see diagnostic above)") +
                failing_blocks_section +
                "\n=== END TIER 1 DIAGNOSTIC ===\n\n"
                "Fix the generated code so that the condition(s) shown in the failing assertion "
                "block(s) above evaluate to true. "
                "Do NOT remove, comment out, or modify any LASSI-VERIFY ASSERTIONS or "
                "LASSI-VERIFY PASSTHROUGH blocks — the fix must be in the surrounding "
                "computation, not in the assertions themselves. "
                "The issue is a mathematical correctness error — the computation violates a "
                "transformation-invariant property that any correct implementation must satisfy."
            )

            error_correction_prompt = prompts.get_code_error_correction_prompt(source_lang + "_error")
            error_prompt = error_correction_prompt + "\n\n" + "[START OF CODE TO FIX] " + code_block + "\n\n" + "[/END OF CODE TO FIX] " + "\n\n" + error_correction_prompt_intro
            error_prompt_tokens = lassi.token_analysis.token_count(error_prompt)
            if verifyprompt:
                print("\n\n" + color.BOLD + "REVIEW PROMPT " + color.END + "(Tier 1 assertion self-correction, token count: " + str(error_prompt_tokens) + "): " + error_prompt + "\n\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to fix code following Tier 1 assertion failure." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, 0.2, this_llm_systemcode_prompt, error_prompt, display_response, verifyprompt)

        elif compile_result == "tier2_failure" and not zero_shot:
            """
            Tier 2 (empirical equivalence) verification failure self-correction.
            The generated code compiled and executed successfully but its passthrough output
            did not match the SOURCE_verify source-as-specification reference.
            This branch uses self_correction_verify_counter (shared with future Tier 3 failures).
            """
            print("\n\n----- " + color.BOLD + color.RED + "LASSI-VERIFY Tier 2 SELF-CORRECTION: " + str(self_correction_verify_counter) + color.END + " -----")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)

            # Build the Tier 2 verification error prompt:
            #   - the generated code that failed
            #   - the diagnostic summary (which observation points diverged and by how much)
            #   - raw passthrough output from SOURCE_verify (source-as-specification reference)
            #   - raw passthrough output from the generated code
            tier2_diag_summary = tier2_diagnostic.get("summary", "(no diagnostic available)")
            # Truncate raw outputs to avoid exceeding context window
            spec_output_excerpt  = passthrough_spec_output[:4000]  if len(passthrough_spec_output)  > 4000  else passthrough_spec_output
            gen_output_excerpt   = passthrough_outputs[-1][:4000]  if passthrough_outputs and len(passthrough_outputs[-1]) > 4000 else (passthrough_outputs[-1] if passthrough_outputs else "")

            error_correction_prompt_intro = (
                "\n\nThe above code compiled and executed successfully but FAILED LASSI-Verify "
                "Tier 2 empirical equivalence checking. Its passthrough observation point output "
                "does not match the SOURCE_verify source-as-specification reference.\n\n"
                "=== TIER 2 DIAGNOSTIC ===\n"
                + tier2_diag_summary +
                "\n\n=== SOURCE_verify PASSTHROUGH OUTPUT (source-as-specification reference) ===\n"
                + spec_output_excerpt +
                "\n\n=== GENERATED CODE PASSTHROUGH OUTPUT ===\n"
                + gen_output_excerpt +
                "\n\n=== END TIER 2 DIAGNOSTIC ===\n\n"
                "Fix the generated code so that its passthrough observation point output matches "
                "the source-as-specification reference within floating-point tolerance. "
                "Do NOT remove or modify any LASSI-VERIFY ASSERTIONS or LASSI-VERIFY PASSTHROUGH blocks. "
                "The issue likely involves incorrect computation, wrong algorithm, or wrong data transformation."
            )

            error_correction_prompt = prompts.get_code_error_correction_prompt(source_lang + "_error")
            error_prompt = error_correction_prompt + "\n\n" + "[START OF CODE TO FIX] " + code_block + "\n\n" + "[/END OF CODE TO FIX] " + "\n\n" + error_correction_prompt_intro
            error_prompt_tokens = lassi.token_analysis.token_count(error_prompt)
            if verifyprompt:
                print("\n\n" + color.BOLD + "REVIEW PROMPT " + color.END + "(Tier 2 verification self-correction, token count: " + str(error_prompt_tokens) + "): " + error_prompt + "\n\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to fix code following Tier 2 verification failure." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, 0.2, this_llm_systemcode_prompt, error_prompt, display_response, verifyprompt)

        elif compile_result == "tier3_failure" and not zero_shot:
            """
            Tier 3 (metamorphic testing) self-correction.
            The generated code passed Tier 2 but failed a metamorphic relation check.
            Provide the failing relation's diagnostic as self-correction context.
            This branch uses self_correction_verify_counter (shared with Tier 2 failures).
            """
            print("\n\n----- " + color.BOLD + color.RED + "LASSI-VERIFY Tier 3 SELF-CORRECTION: " + str(self_correction_verify_counter) + color.END + " -----")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)

            tier3_diag_summary  = tier3_diagnostic.get("summary", "(no diagnostic available)")
            tier3_relation_name = tier3_diagnostic.get("relation_name", "(unknown)")

            error_correction_prompt_intro = (
                "\n\nThe above code passed Tier 2 empirical equivalence checking but FAILED "
                "LASSI-Verify Tier 3 metamorphic testing.\n\n"
                "The metamorphic relation that failed: " + tier3_relation_name + "\n\n"
                "=== TIER 3 DIAGNOSTIC ===\n"
                + tier3_diag_summary +
                "\n=== END TIER 3 DIAGNOSTIC ===\n\n"
                "Fix the generated code so that it satisfies this mathematical property. "
                "Do NOT remove or modify any LASSI-VERIFY ASSERTIONS or LASSI-VERIFY PASSTHROUGH blocks. "
                "The issue is a structural property of the computation, not a single output value difference."
            )

            error_correction_prompt = prompts.get_code_error_correction_prompt(source_lang + "_error")
            error_prompt = error_correction_prompt + "\n\n" + "[START OF CODE TO FIX] " + code_block + "\n\n" + "[/END OF CODE TO FIX] " + "\n\n" + error_correction_prompt_intro
            error_prompt_tokens = lassi.token_analysis.token_count(error_prompt)
            if verifyprompt:
                print("\n\n" + color.BOLD + "REVIEW PROMPT " + color.END + "(Tier 3 metamorphic self-correction, token count: " + str(error_prompt_tokens) + "): " + error_prompt + "\n\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to fix code following Tier 3 metamorphic test failure." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, 0.2, this_llm_systemcode_prompt, error_prompt, display_response, verifyprompt)

        elif compile_result != "" and compile_result != "energy_efficiency" and compile_result != "tier1_failure" and compile_result != "tier2_failure" and compile_result != "tier3_failure" and not zero_shot:
            """
            This condition is for handling CODE ERRORS during compilation or execution.
            """
            print("\n\n----- " + color.BOLD + color.BLUE + "HANDLING CODE ERRORS " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
           
            # CALL THE LLM WITH CODE ERROR CORRECTION - COMPILE TIME OR RUN TIME
            print("\n\n----- " + color.BOLD + color.RED + "SELF-CORRECTION: " + str(self_correction_counter) + color.END + " -----")
            if compile_result == "execute_error":
                self_correction_counter = self_correction_counter + 1
                execute_sc_count += 1                       # Table 1 ★ per-type SC counter
                current_sc_trigger = "execute_error"        # Table 2 ★ trigger for next attempt record
                iteration_log["v" + str(code_version)]["self_correction_count"] += 1
                error_correction_prompt_intro = "\n -- The above CODE TO FIX was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and produced the following execution error: "
                error_message = execute_result
            else:  # Must be a compiler error
                self_correction_counter = self_correction_counter + 1
                compile_sc_count += 1                       # Table 1 ★ per-type SC counter
                current_sc_trigger = "compile_error"        # Table 2 ★ trigger for next attempt record
                iteration_log["v" + str(code_version)]["self_correction_count"] += 1
                # NOTE: Maybe do not provide the original code in the other language when trying to fix code in the target language.
                error_correction_prompt_intro = "\n\n -- The above CODE TO FIX was compiled with " + code_compiler + " " + code_compiler_kwds + " and produced the following compile error: "
                error_message = compile_result
            
            # error_correction_prompt_outro = " Eliminate this specific error by re-factoring the above code with an appropriate fix while also using the following background context: " + original_context_content_prompt
            # error_prompt = code_block + error_correction_prompt_intro + error_message + error_correction_prompt_outro
            # error_correction_prompt_outro = " Eliminate this specific error by re-factoring the above code with an appropriate fix."
            error_correction_prompt = prompts.get_code_error_correction_prompt(source_lang + "_error")
            # TODO: maybe not include context for the code errors? original_context_content_prompt
            # Guard: code_block should always be set by the generation branch before this point.
            # If empty (LLM returned no code fences AND no fallback pattern matched), fall back to
            # code_blocks[-1] so the prompt has something useful for the LLM to work with.
            if code_block == "":
                print(color.BOLD + color.RED + "\n[WARNING] code_block is empty at error-correction prompt — LLM generation may have returned no code. Falling back to code_blocks[-1]." + color.END)
                code_block = code_blocks[-1] if code_blocks else ""
            error_prompt = error_correction_prompt + "\n\n" + "[START OF CODE TO FIX] " + code_block + "\n\n" + "[/END OF CODE TO FIX] " + "\n\n" + error_correction_prompt_intro + "\n\n" + error_message
            error_prompt_tokens = lassi.token_analysis.token_count(error_prompt)
            print("\n\n----- " + color.BOLD + color.RED + "ERROR SELF-CORRECTION" + color.END + " -----")
            if verifyprompt:
                print("\n\n" + color.BOLD + "REVIEW PROMPT " + color.END + "(error correction, token count: " + str(error_prompt_tokens) + "): " + error_prompt + "\n\n")
            # error_prompt = re.sub('\n', ' ', error_prompt)
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code following an error." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, 0.2, this_llm_systemcode_prompt, error_prompt, display_response, verifyprompt)

        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == True and zero_shot: 
            """
            This condition is available only if we want to REPEAT an iteration for the zero_shot attempt.
            Re-attempt until energy efficiency improvement is not acheived (then change prompting strategy).
            """
            print("\n\n----- " + color.BOLD + color.BLUE + "ITERATION FOR STAGE 2: BASE LLM REFACTORING WITH SELF-CORRECTION " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
           
            # CALL THE LLM to generate the another base LLM code refactoring using the prior refactored code
            # Using the same prompting as in the compile_result == "start" condition.
            if llm_source != "argo":
                time.sleep(10)
            # report_gpu_memory()

            # Append the updated refactored code to the prompt
            instructions_prompt = instructions_prompt_core_iterations + "\n\n[START OF CODE TO REFACTOR]\n" + code_blocks[tracking_best_version] + "\n[/END OF CODE TO REFACTOR]"
            # instructions_prompt = instructions_prompt + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(total_energy_joules_refactor, 2)) + " Joules at an average power of " + str(round(ave_power_refactor, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                 
            # Save the original context + energy efficiency context for re-use during energy efficiency correction
            original_ee_context_content_prompt = instructions_prompt_core_iterations
            original_ee_prompt_tokens = lassi.token_analysis.token_count(original_ee_context_content_prompt)
            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE ZERO-SHOT INSTRUCTIONS TASK PROMPT " + color.END + " (prompt token count, excluding code): " + str(original_ee_prompt_tokens) + "\n")  # + instructions_prompt + "\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "iteratate base LLM zero-shot attempt: prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, round(code_gen_temp,1), system_prompt, instructions_prompt, display_response, verifyprompt)

        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == False and not zero_shot:  #baseline_oneshot_goal:
            
            print("\n\n----- " + color.BOLD + color.BLUE + "EE Self-correction, Increment LLM temp " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            
            """
            This condition only runs after zero_shot first attempt is completed.
            Self-correction feedback for improving energy efficiency following no prior improvement.
            This condition does not generate a new refactoring plan, but re-uses the last one.
            This condition increments the LLM temperature by 0.2. If this reaches 1.0, then terminate LASSI pipeline.
            """
            print("\n\n----- " + color.BOLD + color.PURPLE + "SELF-CORRECTION FOR ENERGY EFFICIENCY: " + str(self_correction_power_counter) + color.END + " -----")
            if vary_llm_temp and self_correction_power_counter > 2:
                code_gen_temp += 0.2
                if code_gen_temp >= code_gen_temp_max:
                    # Set condition to end LASSI pipeline
                    self_correction_power_counter = self_correction_power_counter_max + 1
                    print("\n\n----- END LASSI PIPELINE BECAUSE REACHED MAX LLM TEMPERATURE -----")
                    break
            print("LLM Temperature: ", round(code_gen_temp, 1))
                
            # report_gpu_memory()

            # BUILD REFACTORING PLAN
            # On the first try following the base LLM zero shot attempts,
            # otherwise, new refactoring plans will only be created when
            # a new "best" code version is generated by the pipeline.
            if self_correction_power_counter == 0:
                # Generate a new refactoring plan following a new "best" version of the refactored code.
                # When tracking_best_version <= 1 (SOURCE_orig or SOURCE_verify), pass the uninstrumented
                # original source so the LLM focuses on the algorithm, not the assertion instrumentation.
                plan_input_code = original_input_code if tracking_best_version <= 1 else code_blocks[tracking_best_version]
                print(f"  Using refactored code version to write refactoring plan: {tracking_best_version} (uninstrumented original: {tracking_best_version <= 1})")
                refactoring_plan, code_to_refactor = lassi.pipeline_methods.generate_refactoring_plan(
                    source_lang=source_lang,
                    input_code=plan_input_code,
                    total_energy_joules=energy_history[tracking_best_version] if power_measure else runtime_history[tracking_best_version],
                    ave_power=power_history[tracking_best_version] if power_measure else 0.0,
                    llm_source=llm_source,
                    model=model,
                    num_ctx=num_ctx,
                    system_prompt=system_prompt,
                    context_prompt=context_prompt,
                    prior_refactoring_plans=bad_refactor_plans,
                    display_response=display_response,
                    few_shot_learning=few_shot_learning,
                    verifyprompt=verifyprompt,
                    power_measure=power_measure,
                )
                # When tracking_best_version <= 1, the plan was written using original_input_code
                # (uninstrumented) to help the LLM focus on the algorithm. But the actual code
                # to refactor must be code_blocks[1] (SOURCE_verify, with assertion instrumentation),
                # not the original. For tracking_best_version >= 2, plan_input_code ==
                # code_blocks[tracking_best_version] so code_to_refactor is already correct.
                if tracking_best_version <= 1:
                    code_to_refactor = "\n\n[START OF CODE TO REFACTOR]\n" + code_blocks[1] + "\n[/END OF CODE TO REFACTOR]"
                    if power_measure:
                        code_to_refactor += "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(energy_history[1], 2)) + " Joules at an average power of " + str(round(power_history[1], 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                    else:
                        code_to_refactor += "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and had a wall-clock execution time of " + str(round(runtime_history[1], 4)) + " seconds.\n"
            else:
                # Build a list of tuples (index, code_chunk) where the status is "VALID"
                ## valid_prior_codes = [(i, code) for i, (code, status) in enumerate(zip(code_blocks, valid_code_history)) if (status == "VALID" or status == "SOURCE" or status == "basellm_VALID")]
                # Refactor the latest "VALID" code from the list
                ## code_index, code_to_refactor = valid_prior_codes[-1]
                ## code_to_refactor = code_to_refactor + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(energy_history[code_index], 2)) + " Joules at an average power of " + str(round(power_history[code_index], 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                ## print(f"... Prior code index selected to refactor: {code_index}")

                # CHANGE IDEA: Only work on the last 'best' code version, instead of the last 'valid' code
                # Refactor the latest "BEST" code from the list (best = tracking code)
                code_to_refactor = code_blocks[tracking_best_version]
                if power_measure:
                    code_to_refactor = code_to_refactor + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(energy_history[tracking_best_version], 2)) + " Joules at an average power of " + str(round(power_history[tracking_best_version], 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                else:
                    code_to_refactor = code_to_refactor + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and had a wall-clock execution time of " + str(round(runtime_history[tracking_best_version], 4)) + " seconds.\n"
                print(f"... Prior code index selected to refactor: {tracking_best_version}")
                
            # Context:
            # if only the first pass, then refactoring_plan_original will be for the source code,
            # otherwise, it will include the latest refactoring plan for the best code
            iter_suffix = "_refactor_iteration" if power_measure else "_refactor_iteration_runtime"
            refactoring_plan_guide_iteration = prompts.get_code_ee_refactoring_prompt(source_lang + iter_suffix)
            # Append few-shot learning examples
            if few_shot_learning:
                refactoring_plan_guide_iteration = refactoring_plan_guide_iteration + "\n\n" + prompts.get_codeknowledge(source_lang + "_c_few_shot_examples")
            
            # LASSI-VERIFY preservation notice: injected immediately before the code so the LLM
            # cannot miss it. Redundant with the prompt_dictionary warning; belt-and-suspenders.
            _lv_notice = (
                "\n\n[LASSI-VERIFY PRESERVATION REQUIREMENT] The CODE TO REFACTOR below contains "
                "LASSI-VERIFY ASSERTIONS and LASSI-VERIFY PASSTHROUGH blocks delimited by "
                "'// >>> BEGIN LASSI-VERIFY' and '// <<< END LASSI-VERIFY <<<'. "
                "You MUST copy ALL of these blocks into your output exactly as written. "
                "Do NOT remove, modify, restructure, or move them. "
                "Removing these blocks will cause correctness verification to fail."
            )
            refactoring_plan_guide_iteration = refactoring_plan_guide_iteration + "\n\n" + refactoring_plan + _lv_notice + "\n\n" + code_to_refactor  # "\n\n" + context_prompt +
            refactoring_plan_iteration_tokens = lassi.token_analysis.token_count(refactoring_plan_guide_iteration)

            print("\n\n" + color.BOLD + color.GREEN + "----- REFACTORING PLAN WITH INSTRUCTIONS PROMPT " + color.END + "(energy efficiency *iteration* with code, token count: " + str(refactoring_plan_iteration_tokens) + "): ")  # + refactoring_plan_iteration + "\n\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code for energy efficiency -- improvement iteration." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, round(code_gen_temp,1), system_prompt, refactoring_plan_guide_iteration, display_response, verifyprompt)        

        
        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == True and not zero_shot:  #baseline_oneshot_goal:

            print("\n\n----- " + color.BOLD + color.BLUE + "EE Self-correction, New Refactor Plan " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            
            """
            This condition only runs after zero_shot first attempt is completed.
            Self-correction feedback for improving energy efficiency following prior improvement.
            The condition will generate a new refactoring plan for the latest best code.
            """
            print("\n\n----- " + color.BOLD + color.PURPLE + "SELF-IMPROVEMENT FOR ENERGY EFFICIENCY: " + str(energy_efficiency_iteration) + color.END + " -----\n")
            print(f"  Using *just improved* refactored code version: {tracking_best_version} (uninstrumented original: {tracking_best_version <= 1})")

            #report_gpu_memory(llm_source)

            # Generate a new refactoring plan following a new "best" version of the refactored code.
            # For the experiment, LASSI always starts from SOURCE_verify (index 1) to keep the
            # vanilla LLM vs. LASSI comparison clean — even if vanilla LLM found a better version.
            # When tracking_best_version <= 1 (SOURCE_orig or SOURCE_verify), pass the uninstrumented
            # original source so the LLM focuses on the algorithm, not the assertion instrumentation.
            # Metrics still come from track_history[tracking_best_version] (SOURCE_verify baseline).
            plan_input_code = original_input_code if tracking_best_version <= 1 else code_blocks[tracking_best_version]
            refactoring_plan, code_to_refactor = lassi.pipeline_methods.generate_refactoring_plan(
                source_lang=source_lang,
                input_code=plan_input_code,
                total_energy_joules=energy_history[tracking_best_version] if power_measure else runtime_history[tracking_best_version],
                ave_power=power_history[tracking_best_version] if power_measure else 0.0,
                llm_source=llm_source,
                model=model,
                num_ctx=num_ctx,
                system_prompt=system_prompt,
                context_prompt=context_prompt,
                prior_refactoring_plans=bad_refactor_plans,
                display_response=display_response,
                few_shot_learning=False,
                verifyprompt=verifyprompt,
                power_measure=power_measure,
            )
            # Same split as in the iteration block: plan used original_input_code for clean algorithm
            # analysis, but the actual code to refactor must be code_blocks[1] (SOURCE_verify with
            # assertions). For tracking_best_version >= 2, both plan and code_to_refactor already
            # use code_blocks[tracking_best_version] consistently.
            if tracking_best_version <= 1:
                code_to_refactor = "\n\n[START OF CODE TO REFACTOR]\n" + code_blocks[1] + "\n[/END OF CODE TO REFACTOR]"
                if power_measure:
                    code_to_refactor += "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(energy_history[1], 2)) + " Joules at an average power of " + str(round(power_history[1], 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                else:
                    code_to_refactor += "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and had a wall-clock execution time of " + str(round(runtime_history[1], 4)) + " seconds.\n"

            refactor_suffix = "_refactor" if power_measure else "_refactor_runtime"
            refactoring_plan_guide = prompts.get_code_ee_refactoring_prompt(source_lang + refactor_suffix)
            # Append few-shot learning examples
            if few_shot_learning:
                refactoring_plan_guide = refactoring_plan_guide + "\n\n" + prompts.get_codeknowledge(source_lang + "_c_few_shot_examples")
            
            # LASSI-VERIFY preservation notice: injected immediately before the code so the LLM
            # cannot miss it. Redundant with the prompt_dictionary warning; belt-and-suspenders.
            _lv_notice = (
                "\n\n[LASSI-VERIFY PRESERVATION REQUIREMENT] The CODE TO REFACTOR below contains "
                "LASSI-VERIFY ASSERTIONS and LASSI-VERIFY PASSTHROUGH blocks delimited by "
                "'// >>> BEGIN LASSI-VERIFY' and '// <<< END LASSI-VERIFY <<<'. "
                "You MUST copy ALL of these blocks into your output exactly as written. "
                "Do NOT remove, modify, restructure, or move them. "
                "Removing these blocks will cause correctness verification to fail."
            )
            refactoring_plan_guide = refactoring_plan_guide + "\n\n" + refactoring_plan + _lv_notice + "\n\n" + code_to_refactor  # "\n\n" + context_prompt +
            refactoring_plan_tokens = lassi.token_analysis.token_count(refactoring_plan_guide)
           
            print("\n\n" + color.BOLD + color.GREEN + "----- REFACTORING PLAN WITH INSTRUCTIONS PROMPT " + color.END + "(energy efficiency improvement with code, token count: " + str(refactoring_plan_tokens) + "): ")  # \n\n" + refactoring_plan + "\n\n")
            # refactoring_plan_iteration = re.sub('\n', '', refactoring_plan)
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code for energy efficiency -- with a new refactoring plan." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, refactoring_plan_guide, display_response, verifyprompt)
        else:
            print("\n\n ????? NO CONDITION MET DURING ITERATION ?????")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
        
        # -------------------------------------
        # ----- GENERATED CODE PROCESSING -----
        if code_block.startswith('c++') or code_block.startswith('cpp'):
            code_block = code_block[3:]
        if code_block.startswith('c'):
            code_block = code_block[1:]    
    
        print("\n\n----- " + color.BOLD + color.BLUE + "POST-PROCESSING of GENERATED CODE" + color.END + " -----\n")
        print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
        
        # SAVE THE GENERATIVE CODE -- before compiling
        # If make script, then save edited code as same filename and extension
        if make_script:
            # We need to overwrite the same code file,
            # instead of creating a new file.
            print("SAVING THE CODE - WHERE ARE WE: ", os.getcwd())
            file_name_save = input_code_file_subdirectory + file_name
            file_path = os.getcwd() + "/" + file_name_save + "." + refactored_code_extension
            file_versioning_path = experiment_data_directory  + "/" + file_name_save + "." + refactored_code_extension
            os.makedirs(os.path.dirname(file_versioning_path), exist_ok=True)
            with open(file_versioning_path, 'w') as file:
                file.write(code_block)
            file.close()
        else:
            file_name_save = file_name + "_" + dt_string
            # file_path = directory + "/" + file_name_save + "." + refactored_code_extension
            file_path = experiment_data_directory + "/" + file_name_save + "_Trial" + str(trial_number) + "." + refactored_code_extension
        print("saving to file_path: ", file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(code_block)
        file.close()
    
        # Compile the generated code, compile_result == "" if no error
        if build_script:
            compile_result = lassi.code_manager.build_script_code("generated_codes", code_build_script)
            # execute_file = "/home/cc/lassi/LASSI_dev/generated_codes/altis/build/bin/level1/bfs"
            # execute_file = "/home/cc/lassi/LASSI_dev/generated_codes/altis/build/bin/level2/raytracing"
            execute_file = "/home/cc/lassi/LASSI_dev/generated_codes/altis/build/bin/level1/" + file_name_save
        elif make_script:
            # run make command
            print("\n** RUN MAKE COMMAND TO COMPILE **")
            compile_result = lassi.code_manager.make_script_code(directory, code_compiler_kwds, source_lang)
            # print("compile_result: ", compile_result)
            if app_name == "miniMDock":
                execute_file = "bin/autodock_gpu_64wi"  # HARD CODED FOR MINIMDOCK
                exe_input_parameters = "-lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 100".split()
            elif app_name == "XSBench":
                execute_file = "XSBench"  # HARD CODED FOR XSBENCH
                exe_input_parameters = "-m event -l 1700000000".split()
        else:
            compile_result, execute_file = lassi.code_manager.compile_code(file_path, target_lang, code_compiler, code_compiler_kwds)

        print("\nExecute file: " + str(execute_file))

        # If the Baseline LLM in Stage 1 fails to compile, then move on to LASSI
        if zero_shot and stage == "stage1" and compile_result != "":
            # Force the switch to LASSI-EE pipeline.
            # Still need to record empty values for history
            zeroshot_compile_status = "compile_error"   # capture before compile_result is overwritten
            zeroshot_execute_status = "skipped"
            stage = ""
            compile_result = "energy_efficiency"
            change_strategy = True
            zero_shot = False  # Stop base LLM attempt after one try.
            force_save_baseline_llm = True  # We need to save Null data to maintain history tracking
        elif not zero_shot and compile_result != "" and compile_result != "energy_efficiency":
            # LASSI code compile error — emit intermediate Table 2 row before the retry.
            # code_block_index uses len(code_blocks) (intended next index; code never appended).
            _cb_idx = len(code_blocks)
            attempt_log_buffer.append(_build_attempt_record(
                app_name=file_name, trial_number=trial_number,
                code_block_index=_cb_idx, attempt_number=attempt_number,
                is_final_attempt=False, generation_type=current_generation_type,
                sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                compile_status="compile_error", execute_status="skipped",
                runtime_s=None, energy_J=None, ave_power_W=None,
                tier1_verdict="SKIP", tier1_instrumented_labels="", tier1_diagnostic={},
                tier2_verdict="SKIP", tier2_instrumented_labels="", tier2_diagnostic={},
                tier3_verdict="SKIP", tier3_diagnostic={},
                judge_verdict=None, final_verdict=None, override_applied=None,
            ))
            current_sc_trigger = "compile_error"  # Table 2 ★ carry into next iteration's record

        if yes_execute and compile_result == "" and self_correction_counter <= self_correction_max:
            code_source = "generated"  # target
            # Unload any previously loaded LLMs in the Ollama server 
            # (clear out the memory)
            if llm_source == "ollama":
                lassi.llm_config.unload_ollama_model(model_coder)
                time.sleep(10)
                report_gpu_memory()
                
            exe_file_path = execute_file  # directory + "/" + execute_file
            execute_result, exe_output_generated, total_energy_joules_refactor, ave_power_refactor, std_power_std, duration_refactor, metrics_results_string = lassi.code_manager.execute_code_selectvendor(experiment_data_directory, code_version, zero_shot, trial_number, directory, exe_file_path, exe_input_parameters, power_measure, code_source, target_lang, build_script, make_script, gpu_vendor="auto", wait=5)
            if execute_result != "0":
                compile_result = "execute_error"  # just a flag
                # If the Baseline LLM in Stage 1 fails to execute, then move on to LASSI
                if zero_shot and stage == "stage1":
                    # Force the switch to LASSI-EE pipeline.
                    # Still need to record empty values for history
                    zeroshot_compile_status = "success"    # compiled OK; execution failed
                    zeroshot_execute_status = "execute_error"
                    stage = ""
                    compile_result = "energy_efficiency"
                    change_strategy = True
                    zero_shot = False  # Stop base LLM attempt after one try.
                    force_save_baseline_llm = True  # We need to save Null data to maintain history tracking
                elif not zero_shot:
                    # LASSI code execute error — emit intermediate Table 2 row before the retry.
                    # code_block_index uses len(code_blocks) (intended next index; code never appended).
                    _cb_idx = len(code_blocks)
                    attempt_log_buffer.append(_build_attempt_record(
                        app_name=file_name, trial_number=trial_number,
                        code_block_index=_cb_idx, attempt_number=attempt_number,
                        is_final_attempt=False, generation_type=current_generation_type,
                        sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                        compile_status="success", execute_status="execute_error",
                        runtime_s=None, energy_J=None, ave_power_W=None,
                        tier1_verdict="SKIP", tier1_instrumented_labels="", tier1_diagnostic={},
                        tier2_verdict="SKIP", tier2_instrumented_labels="", tier2_diagnostic={},
                        tier3_verdict="SKIP", tier3_diagnostic={},
                        judge_verdict=None, final_verdict=None, override_applied=None,
                    ))
                    current_sc_trigger = "execute_error"  # Table 2 ★ carry into next iteration's record
            else:
                # We have a keeper! Save the generated code as a separate versioned copy
                # Increment code version to get ready for the next iteration.
                if not zero_shot:
                    code_version = code_version + 1
                # Initiate a new iteration log for this code version
                iteration_log["v" + str(code_version)] = {"self_correction_count": 0, "energy_refactor_count": 0, "LLM_temperature": round(code_gen_temp,1)}
                print("--> SAVING code version: " + str(code_version))
                if not zero_shot:
                    this_code_version = code_version
                else:
                    this_code_version = str(code_version) + "-baseLLM"
                # file_path = directory + "/" + file_name_save + "_v" + str(this_code_version) + "." + refactored_code_extension
                file_path = experiment_data_directory + "/" + file_name_save + "_v" + str(this_code_version) + "_Trial" + str(trial_number) + "." + refactored_code_extension
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as file:
                    file.write(code_block)
                file.close()
                code_blocks.append(code_block)  # save this code in our code collection list
                input_content_llmjudge_dict["refactored_code"] = code_block
                input_content_llmjudge_dict["refactored_stdout"] = exe_output_generated
                input_content_llmjudge_dict["refactored_metrics"] = metrics_results_string

        elif yes_execute and compile_result == "" and self_correction_counter > self_correction_max and not zero_shot:
            # Execute budget exhausted — code compiled but execute was skipped.
            # Emit final attempt record so cb_idx has a is_final_attempt=True row in Table 2.
            # This closes the last open code_block_index that ran out of SC retries before executing.
            _cb_idx = len(code_blocks)  # code never appended (execute never ran)
            attempt_log_buffer.append(_build_attempt_record(
                app_name=file_name, trial_number=trial_number,
                code_block_index=_cb_idx, attempt_number=attempt_number,
                is_final_attempt=True, generation_type=current_generation_type,
                sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                compile_status="success", execute_status="skipped",
                runtime_s=None, energy_J=None, ave_power_W=None,
                tier1_verdict="SKIP", tier1_instrumented_labels="", tier1_diagnostic={},
                tier2_verdict="SKIP", tier2_instrumented_labels="", tier2_diagnostic={},
                tier3_verdict="SKIP", tier3_diagnostic={},
                judge_verdict=None, final_verdict="INVALID", override_applied="exhausted_sc_budget",
            ))

        # If the Baseline LLM generated code that failed to compile or execute,
        #  then we still need to save off zero'd data to maintain our history tracking used by LASSI
        if force_save_baseline_llm:
            # code_version = code_version + 1
            # Initiate a new iteration log for this code version
            iteration_log["v" + str(code_version)] = {"self_correction_count": 0, "energy_refactor_count": 0, "LLM_temperature": round(code_gen_temp,1)}
            print("--> Force SAVING failed baseline LLM code version: " + str(code_version))
            this_code_version = str(code_version) + "-baseLLM"
            file_path = experiment_data_directory + "/" + file_name_save + "_v" + str(this_code_version) + "_Trial" + str(trial_number) + "." + refactored_code_extension
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write(code_block)
            file.close()
            code_blocks.append(code_block)  # save this code in our code collection list
            energy_history.append(0)
            power_history.append(0)
            runtime_history.append(0)
            valid_code_history.append("basellm_INVALID")
            experiment_data["base_v1_energy_J"] = "NA"
            experiment_data["base_v1_avg_power_W"] = "NA"
            experiment_data["base_v1_runtime_s"] = "NA"
            experiment_data["base_v1_valid"] = "INVALID"
            experiment_data["base_v1_energy_diff"] = "NA"
            experiment_data["base_v1_avg_power_diff"] = "NA"
            experiment_data["base_v1_runtime_diff"] = "NA"
            experiment_data["zeroshot_compile_status"] = zeroshot_compile_status
            experiment_data["zeroshot_execute_status"] = zeroshot_execute_status
            experiment_data["zeroshot_runtime_s"] = "NA"
            # Reset flag because this should be a one-time action.
            force_save_baseline_llm = False
                    
            
        # Begin POWER METRICS feedback loop
        if yes_power_feedback and yes_execute and compile_result == "" and execute_result == "0" and self_correction_power_counter <= self_correction_power_counter_max:

            print("\n\n----- " + color.BOLD + color.BLUE + "POWER METRICS Feedback Loop " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            
            if not zero_shot and compile_result_before_generation in ("", "energy_efficiency"):  # baseline_oneshot_goal:
                # This will initiate the subsequent iterations to introduce context and self-prompting.
                # Only increment for fresh code generations — tier1/tier2/tier3/execute SC retries must not
                # count against the EE refactor budget (otherwise budget exhausts early on difficult apps).
                self_correction_power_counter = self_correction_power_counter + 1
                iteration_log["v" + str(code_version)]["energy_refactor_count"] = self_correction_power_counter
            # else:
            # Do not enable feedback iteration in baseline mode
            #self_correction_power_counter = self_correction_power_counter_max + 1
            
            # --- TIER 1: Invariant Assertion Check (fail-fast before Tier 2) ---
            # Parses stdout for LASSI-VERIFY ASSERTIONS [label]: FAIL lines.
            # All assertions are checked before deciding (code runs to completion), giving the
            # LLM full visibility into every violated invariant — not just the first one.
            # Applies only to LASSI-generated codes (zero_shot has no assertion blocks).
            # Reset per-code counter on fresh code; preserve across tier1_failure retries.
            if compile_result_before_generation != "tier1_failure":
                self_correction_tier1_counter_code = 0
            tier1_verdict = "SKIP"
            tier1_diagnostic = {}
            if not zero_shot:
                tier1_verdict, tier1_diagnostic = lassi.verify.check_tier1_assertions(exe_output_generated)
                if tier1_verdict == "PASS":
                    if tier1_diagnostic['pass_count'] == 0:
                        # Sentinel fired but no assertion blocks in stdout — LLM likely stripped them.
                        print(f"\n----- LASSI-Verify Tier 1: PASS (sentinel only — 0 assertion blocks found; LASSI-VERIFY blocks may have been stripped by LLM) -----")
                    else:
                        print(f"\n----- LASSI-Verify Tier 1: PASS ({tier1_diagnostic['pass_count']} assertion(s)) -----")
                elif tier1_verdict == "SKIP":
                    print(f"\n----- LASSI-Verify Tier 1: SKIP (no assertion blocks found in stdout) -----")
                elif tier1_verdict == "FAIL":
                    print(f"\n----- LASSI-Verify Tier 1: FAIL — {tier1_diagnostic['fail_count']} assertion(s) violated -----")
                    print(tier1_diagnostic["summary"])
                    # Note: Tier 1 does NOT increment self_correction_verify_counter (the global Tier 2+3 safety net).
                    # Tier 1 has its own per-code budget (self_correction_tier1_counter_code_max) which bounds it.
                    # Including Tier 1 in the global counter causes early termination on hard kernels with
                    # many Tier 1 retries before Tier 2 is ever reached.
                    self_correction_tier1_counter_code += 1
                    tier1_sc_count += 1                         # Table 1 ★ per-type SC counter
                    iteration_log["v" + str(code_version)]["self_correction_count"] += 1
                    if self_correction_tier1_counter_code <= self_correction_tier1_counter_code_max:
                        # Table 2 ★ intermediate attempt row — log before pop overwrites this code_block_index
                        _cb_idx = len(code_blocks) - 1
                        _t1_instr = lassi.verify.extract_tier1_labels(code_blocks[-1])
                        attempt_log_buffer.append(_build_attempt_record(
                            app_name=file_name, trial_number=trial_number,
                            code_block_index=_cb_idx, attempt_number=attempt_number,
                            is_final_attempt=False, generation_type=current_generation_type,
                            sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                            compile_status="success", execute_status="success",
                            runtime_s=duration_refactor, energy_J=total_energy_joules_refactor,
                            ave_power_W=ave_power_refactor,
                            tier1_verdict=tier1_verdict, tier1_instrumented_labels=_t1_instr,
                            tier1_diagnostic=tier1_diagnostic,
                            tier2_verdict="SKIP", tier2_instrumented_labels="", tier2_diagnostic={},
                            tier3_verdict="SKIP", tier3_diagnostic={},
                            judge_verdict=None, final_verdict=None, override_applied=None,
                        ))
                        current_sc_trigger = ("tier1_fail:" + tier1_diagnostic["failing_labels"][0]
                                              if tier1_diagnostic.get("failing_labels") else "tier1_fail:unknown")
                        code_version -= 1  # roll back: pop removes the code so code_version must stay in sync
                        code_blocks.pop()
                        compile_result = "tier1_failure"
                        print(f"\n----- LASSI-Verify Tier 1: self-correction "
                              f"{self_correction_tier1_counter_code}/{self_correction_tier1_counter_code_max} "
                              f"(global: {self_correction_verify_counter}) -----")
                        continue  # Skip Tier 2, Tier 3, judge; return to top of while loop
                    else:
                        print(f"\n----- LASSI-Verify Tier 1: per-code budget exhausted "
                              f"({self_correction_tier1_counter_code_max} attempts), marking INVALID -----")
                        # Force INVALID immediately; skip Tier 2, Tier 3, and judge to avoid unnecessary overhead.
                        # (Mirrors the non-exhausted Tier 1 path which also uses continue to skip subsequent tiers.)
                        # Table 2 ★ final attempt row — Tier 1 exhaustion is the final verdict; judge never runs
                        _cb_idx = len(code_blocks) - 1
                        _t1_instr = lassi.verify.extract_tier1_labels(code_blocks[-1])
                        attempt_log_buffer.append(_build_attempt_record(
                            app_name=file_name, trial_number=trial_number,
                            code_block_index=_cb_idx, attempt_number=attempt_number,
                            is_final_attempt=True, generation_type=current_generation_type,
                            sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                            compile_status="success", execute_status="success",
                            runtime_s=duration_refactor, energy_J=total_energy_joules_refactor,
                            ave_power_W=ave_power_refactor,
                            tier1_verdict=tier1_verdict, tier1_instrumented_labels=_t1_instr,
                            tier1_diagnostic=tier1_diagnostic,
                            tier2_verdict="SKIP", tier2_instrumented_labels="", tier2_diagnostic={},
                            tier3_verdict="SKIP", tier3_diagnostic={},
                            judge_verdict=None, final_verdict="INVALID", override_applied="tier1_exhausted",
                        ))
                        current_sc_trigger = None  # exhausted → next iteration is fresh
                        valid_code_flag = "INVALID"
                        compile_result = "energy_efficiency"  # drive fresh generation next iteration
                        continue

            # --- TIER 2: Empirical Equivalence Check (fail-fast before judge) ---
            # Applies only to LASSI-generated codes (not zero_shot). Zero-shot codes refactor from
            # original_input_code (no passthrough blocks), so compare_passthrough_outputs() would
            # return SKIP — which is the correct behavior for the zero-shot path.
            # Reset per-code Tier 2 self-correction counter when processing a fresh code.
            # A Tier 2 retry enters here with compile_result_before_generation == "tier2_failure";
            # anything else (initial, win, INVALID/no-improvement) means a fresh code.
            if compile_result_before_generation not in ("tier2_failure", "tier1_failure"):
                self_correction_verify_counter_code = 0
                tier2_exhausted = False  # defensive reset: ensure no stale exhausted state from prior iteration
            tier2_verdict = "SKIP"
            tier2_diagnostic = {}
            if run_tier2 and not zero_shot and passthrough_spec_output != "":
                tier2_verdict, tier2_diagnostic = lassi.verify.compare_passthrough_outputs(
                    spec_stdout=passthrough_spec_output,
                    generated_stdout=exe_output_generated,
                    tolerance=tier2_tolerance,
                    llm_source=llm_source, model=model, num_ctx=num_ctx,
                    system_prompt=system_prompt, verifyprompt=verifyprompt
                )
                print(f"\n----- LASSI-Verify Tier 2: {tier2_verdict} -----")
                if tier2_diagnostic.get("failures"):
                    print(tier2_diagnostic.get("summary", ""))
                # Record Tier 2 result in verification log
                if "tier2_results" in verification_log:
                    verification_log["tier2_results"].append({
                        "code_version": code_version,
                        "verdict": tier2_verdict,
                        "diagnostic": tier2_diagnostic,
                    })

            passthrough_outputs.append(exe_output_generated)  # record stdout for every generated attempt

            if tier2_verdict.startswith("FAIL"):
                # Tier 2 failed: increment both global and per-code self-correction counters.
                self_correction_verify_counter += 1
                self_correction_verify_counter_code += 1
                tier2_sc_count += 1                             # Table 1 ★ per-type SC counter
                iteration_log["v" + str(code_version)]["self_correction_count"] += 1

                if self_correction_verify_counter_code < self_correction_verify_counter_code_max:
                    # Still within per-code budget: pop code, self-correct with Tier 2 feedback.
                    # code_blocks is popped to keep it in sync with runtime_history (not updated on
                    # Tier 2 failure). Without this pop, tracking_best_version = len(code_blocks)-1
                    # after a later win would be out of bounds for runtime_history (IndexError).
                    # Table 2 ★ intermediate attempt row — log before pop overwrites this code_block_index
                    _cb_idx = len(code_blocks) - 1
                    _t1_instr = lassi.verify.extract_tier1_labels(code_blocks[-1])
                    _t2_instr = lassi.verify.extract_tier2_labels(code_blocks[-1])
                    attempt_log_buffer.append(_build_attempt_record(
                        app_name=file_name, trial_number=trial_number,
                        code_block_index=_cb_idx, attempt_number=attempt_number,
                        is_final_attempt=False, generation_type=current_generation_type,
                        sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                        compile_status="success", execute_status="success",
                        runtime_s=duration_refactor, energy_J=total_energy_joules_refactor,
                        ave_power_W=ave_power_refactor,
                        tier1_verdict=tier1_verdict, tier1_instrumented_labels=_t1_instr,
                        tier1_diagnostic=tier1_diagnostic,
                        tier2_verdict=tier2_verdict, tier2_instrumented_labels=_t2_instr,
                        tier2_diagnostic=tier2_diagnostic,
                        tier3_verdict="SKIP", tier3_diagnostic={},
                        judge_verdict=None, final_verdict=None, override_applied=None,
                    ))
                    _t2_fail_label = (tier2_diagnostic.get("failures", [{}])[0].get("label", "unknown")
                                      if tier2_diagnostic.get("failures") else "unknown")
                    current_sc_trigger = f"tier2_fail:{_t2_fail_label}"
                    code_version -= 1  # roll back: pop removes the code so code_version must stay in sync
                    code_blocks.pop()
                    compile_result = "tier2_failure"
                    print(f"\n----- LASSI-Verify Tier 2: FAIL — self-correction {self_correction_verify_counter_code}/{self_correction_verify_counter_code_max} (global total: {self_correction_verify_counter}) -----")
                    continue  # Skip judge and metrics; return to top of while loop
                else:
                    # Per-code Tier 2 budget exhausted: this code's approach is fundamentally wrong.
                    # Keep code in code_blocks (it ran — it's a real artifact for research records).
                    # Mark INVALID so it doesn't update tracking_best_version. Set compile_result to
                    # "energy_efficiency" so next iteration generates a fresh code (not Tier 2 retry).
                    # Tier 2 self-correction context is cleared automatically: compile_result != "tier2_failure"
                    # on the next iteration triggers reset of self_correction_verify_counter_code.
                    print(f"\n----- LASSI-Verify Tier 2: FAIL — per-code budget exhausted ({self_correction_verify_counter_code_max} attempts), marking INVALID, fresh generation next -----")
                    # Note: tier2_sc_count was already incremented unconditionally above (line 1371).
                    # Do NOT increment again here — that would double-count the exhausted event.
                    tier2_exhausted = True
                    compile_result = "energy_efficiency"  # drive fresh code generation (not Tier 2 retry)
            # Tier 2 PASS or SKIP: record in verification log and proceed to judge
            if run_tier2 and not zero_shot and "verified_pass_count" in verification_log and tier2_verdict.startswith("PASS"):
                verification_log["verified_pass_count"] += 1

            # --- TIER 3: Metamorphic Testing (after Tier 2 PASS/SKIP, before judge) ---
            # Applies only to LASSI-generated codes with active relations.
            # Zero-shot codes skip entirely (no Tier 2 passthrough baseline exists for them).
            # Reset per-code Tier 3 counter on fresh code (mirrors Tier 2 counter reset logic).
            if compile_result_before_generation != "tier3_failure":
                self_correction_tier3_counter_code = 0
            tier3_verdict = "SKIP"
            tier3_diagnostic = {}
            tier3_results_this_iter = []          # Table 2 ★ always defined; populated if Tier 3 runs
            if run_tier3 and not zero_shot and tier3_active_relations and not tier2_verdict.startswith("FAIL"):
                # Tier 3 baseline: the generated code's own Tier 2 passthrough stdout.
                # passthrough_outputs[-1] is the stdout just appended above (this iteration's run).
                generated_passthrough_stdout = passthrough_outputs[-1] if passthrough_outputs else ""
                tier3_verdict, tier3_results_this_iter, tier3_diagnostic = lassi.verify.run_metamorphic_variants(
                    code_blocks[-1],             # generated code source
                    tier3_active_relations,
                    tier3_contract,
                    generated_passthrough_stdout,  # generated code's own output as Tier 3 baseline (Y)
                    llm_source, model, num_ctx, system_prompt,
                    display_response,
                    directory, file_name, code_compiler, code_compiler_kwds,
                    build_script, make_script, code_build_script,
                    exe_input_parameters,
                    experiment_data_directory=experiment_data_directory,
                    code_version=code_version,
                    verifyprompt=verifyprompt
                )
                # Enrich tier3_diagnostic with per-relation aggregates for Table 2 ★
                _t3_names = [r["name"] for r in tier3_results_this_iter]
                _t3_passed = sum(1 for r in tier3_results_this_iter if r["verdict"] == "PASS")
                _t3_failed = sum(1 for r in tier3_results_this_iter if r["verdict"] == "FAIL")
                _t3_fail_item = next((r for r in tier3_results_this_iter if r["verdict"] == "FAIL"), None)
                tier3_diagnostic["tested_relations"] = _t3_names
                tier3_diagnostic["relations_passed"] = _t3_passed
                tier3_diagnostic["relations_failed"] = _t3_failed
                if _t3_fail_item:
                    _fail_rel_obj = next((rel for rel in tier3_active_relations if rel["name"] == _t3_fail_item["name"]), {})
                    tier3_diagnostic.setdefault("relation_name", _t3_fail_item["name"])
                    tier3_diagnostic["relation_type"] = _fail_rel_obj.get("type", None)
                print(f"\n----- LASSI-Verify Tier 3: {tier3_verdict} -----")
                if tier3_diagnostic.get("failures"):
                    print(tier3_diagnostic.get("summary", ""))
                if "tier3_results" in verification_log:
                    verification_log["tier3_results"].append({
                        "code_version": code_version,
                        "verdict": tier3_verdict,
                        "results": tier3_results_this_iter,
                    })

            if tier3_verdict.startswith("FAIL"):
                # Increment shared global counter and per-code Tier 3 counter.
                self_correction_verify_counter += 1
                self_correction_tier3_counter_code += 1
                iteration_log["v" + str(code_version)]["self_correction_count"] += 1
                if self_correction_tier3_counter_code <= self_correction_tier3_counter_code_max:
                    # Still within per-code budget: pop code_blocks to keep in sync with history,
                    # then self-correct with Tier 3 diagnostic feedback.
                    tier3_sc_count += 1                             # Table 1 ★ per-type SC counter
                    # Table 2 ★ intermediate attempt row — log before pop overwrites this code_block_index
                    _cb_idx = len(code_blocks) - 1
                    _t1_instr = lassi.verify.extract_tier1_labels(code_blocks[-1])
                    _t2_instr = lassi.verify.extract_tier2_labels(code_blocks[-1])
                    attempt_log_buffer.append(_build_attempt_record(
                        app_name=file_name, trial_number=trial_number,
                        code_block_index=_cb_idx, attempt_number=attempt_number,
                        is_final_attempt=False, generation_type=current_generation_type,
                        sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                        compile_status="success", execute_status="success",
                        runtime_s=duration_refactor, energy_J=total_energy_joules_refactor,
                        ave_power_W=ave_power_refactor,
                        tier1_verdict=tier1_verdict, tier1_instrumented_labels=_t1_instr,
                        tier1_diagnostic=tier1_diagnostic,
                        tier2_verdict=tier2_verdict, tier2_instrumented_labels=_t2_instr,
                        tier2_diagnostic=tier2_diagnostic,
                        tier3_verdict=tier3_verdict, tier3_diagnostic=tier3_diagnostic,
                        judge_verdict=None, final_verdict=None, override_applied=None,
                    ))
                    current_sc_trigger = f"tier3_fail:{tier3_diagnostic.get('relation_name', 'unknown')}"
                    code_version -= 1  # roll back: pop removes the code so code_version must stay in sync
                    code_blocks.pop()
                    compile_result = "tier3_failure"
                    print(f"\n----- LASSI-Verify Tier 3: FAIL — self-correction "
                          f"{self_correction_tier3_counter_code}/{self_correction_tier3_counter_code_max} "
                          f"(global: {self_correction_verify_counter}) -----")
                    continue  # Skip judge and metrics; return to top of while loop
                else:
                    # Per-code Tier 3 budget exhausted: flag for Override 3 (below) to force INVALID.
                    print(f"\n----- LASSI-Verify Tier 3: FAIL — per-code budget exhausted "
                          f"({self_correction_tier3_counter_code_max} attempts), marking INVALID -----")
                    tier3_sc_count += 1             # Table 1 ★ exhaustion counts as a SC event (the last retry)
                    tier3_exhausted = True
                    compile_result = "energy_efficiency"  # drive fresh generation next iteration

            if run_tier3 and not zero_shot and "tier3_pass_count" in verification_log and tier3_verdict == "PASS":
                verification_log["tier3_pass_count"] += 1

            # Call LLM-as-a-judge to review SOURCE CODE and REFACTORED CODE to evaluate if likely valid.
            # Judge always runs for zero_shot (vanilla LLM has no Tier 1/2 instrumentation — the judge
            # is its only correctness gate). For LASSI codes, judge is controlled by run_llm_judge:
            #   run_llm_judge=True:  judge runs; three-tier verdict can override it (see below)
            #   run_llm_judge=False: pure three-tier mode for LASSI codes; judge skipped; vanilla still judged
            # For the zero-shot (vanilla LLM) pass, judge against the original uninstrumented source code.
            # For the LASSI pipeline (not zero_shot), judge against the assertion-instrumented source code.
            # code_blocks[0]=SOURCE_orig, code_blocks[1]=SOURCE_verify (both always pre-populated before while loop).
            if run_llm_judge or zero_shot:
                if zero_shot:
                    judge_dict = dict(input_content_llmjudge_dict)  # shallow copy to avoid mutating the shared dict
                    judge_dict["source_code"] = code_blocks[0]      # compare vanilla LLM against original (uninstrumented) source
                    judge_dict["source_stdout"] = source_orig_stdout  # use uninstrumented stdout; source_verify_stdout has passthrough dump noise
                else:
                    judge_dict = input_content_llmjudge_dict         # compare LASSI against instrumented source
                valid_code_flag, judge_response_text, code_analysis = lassi.llm_methods.llm_judge(target_lang, llm_source, model_judge, num_ctx, system_prompt, judge_dict, power_measure, run_code_analysis, verification_log["sourcecode_analysis"], display_response, verifyprompt)
            else:
                # Pure three-tier mode (LASSI only, never zero_shot): Tier 1 + Tier 2 determine validity.
                valid_code_flag = "VALID"
                judge_response_text = "LLM judge disabled (run_llm_judge=False)"
                code_analysis = "None"
                print("\n----- LLM judge disabled: LASSI code accepted by Tier 1 + Tier 2 deterministic gates -----")

            # Capture the raw judge verdict immediately after the judge call, before zero-energy guard
            # or three-tier overrides can modify valid_code_flag. Used for discrepancy logging below.
            judge_raw_verdict = valid_code_flag
            # Normalize: the judge prompt only allows "VALID" or "INVALID". Any other response
            # (e.g. "NONE", malformed output) is treated as INVALID.
            if valid_code_flag != "VALID":
                if valid_code_flag != "INVALID":
                    print(f"\n----- LLM judge returned unexpected verdict '{valid_code_flag}' — normalizing to INVALID -----")
                valid_code_flag = "INVALID"

            # Handle scenario where we might have a bad power measurement
            # (only applies in EE mode; in runtime mode energy is always 0)
            if power_measure and total_energy_joules_refactor == 0:
                valid_code_flag = "INVALID"

            # --- THREE-TIER PRIORITY OVERRIDES ---
            # Three-tier verdict takes priority over the LLM judge for LASSI codes (not zero_shot).
            # Discrepancies are logged to verification_log["judge_discrepancies"] for research analysis:
            # they reveal cases where the LLM judge was wrong in either direction.
            # Capture exhausted flags BEFORE overrides reset them (needed for override_applied in Table 2 ★)
            _tier2_was_exhausted = tier2_exhausted
            _tier3_was_exhausted = tier3_exhausted
            _override2_fired = False

            # Override 1 (false positive): Tier 2 budget exhausted → INVALID even if judge said VALID.
            # Tier 2 already ran and failed self_correction_verify_counter_code_max times — code is wrong.
            if tier2_exhausted and not zero_shot:
                if judge_raw_verdict == "VALID":
                    verification_log["judge_discrepancies"].append({
                        "code_version": code_version, "zero_shot": zero_shot,
                        "type": "false_positive",
                        "judge_verdict": judge_raw_verdict, "tier2_verdict": "FAIL_exhausted",
                        "final_verdict": "INVALID",
                    })
                    print(f"\n----- Three-tier override: judge=VALID overridden to INVALID (Tier 2 exhausted — judge false positive) -----")
                valid_code_flag = "INVALID"
                judge_response_text = f"Tier 2 self-correction budget exhausted ({self_correction_verify_counter_code_max} attempts): code marked INVALID; fresh generation next."
                code_analysis = "None"
                tier2_exhausted = False  # reset for next iteration

            # Override 2 (false negative): Tier 1 PASS/SKIP + Tier 2 PASS + Tier 3 PASS/SKIP → VALID even if judge said INVALID.
            # All active tiers must be satisfied: Tier 2 PASS alone is insufficient if Tier 1 or Tier 3 is active.
            # tier1_verdict == "SKIP" is acceptable (Tier 1 inactive/disabled or no assertions found in output).
            # tier3_verdict == "SKIP" is acceptable (Tier 3 inactive/disabled/no active relations).
            # Tier 1 exhausted is handled inline (continue before judge); Tier 3 exhausted by Override 3.
            # Only applies to LASSI codes (zero_shot has no instrumentation → SKIP, no override).
            if (tier1_verdict in ("PASS", "SKIP") and tier2_verdict == "PASS"
                    and tier3_verdict in ("PASS", "SKIP")
                    and valid_code_flag != "VALID" and not zero_shot):
                _t1_label = (f"Tier 1 {tier1_verdict}" if not zero_shot else "Tier 1 inactive")
                _t3_label = (f"Tier 3 {tier3_verdict}" if (run_tier3 and tier3_active_relations)
                             else "Tier 3 inactive")
                verification_log["judge_discrepancies"].append({
                    "code_version": code_version, "zero_shot": zero_shot,
                    "type": "false_negative",
                    "judge_verdict": judge_raw_verdict, "tier1_verdict": tier1_verdict,
                    "tier2_verdict": tier2_verdict, "tier3_verdict": tier3_verdict,
                    "final_verdict": "VALID",
                })
                print(f"\n----- Three-tier override: judge={judge_raw_verdict} overridden to VALID "
                      f"({_t1_label} + Tier 2 PASS + {_t3_label} — judge false negative) -----")
                valid_code_flag = "VALID"
                _override2_fired = True

            # Override 3 (false positive): Tier 3 budget exhausted → INVALID regardless of judge or Tier 2.
            # Tier 3 is the final authority: a code that cannot satisfy a metamorphic relation after
            # self_correction_tier3_counter_code_max attempts has a structural mathematical flaw.
            # This override runs AFTER Override 2 so it has the last word.
            if tier3_exhausted and not zero_shot:
                if valid_code_flag == "VALID":
                    verification_log["judge_discrepancies"].append({
                        "code_version": code_version, "zero_shot": zero_shot,
                        "type": "false_positive",
                        "judge_verdict": judge_raw_verdict, "tier3_verdict": "FAIL_exhausted",
                        "final_verdict": "INVALID",
                    })
                    print(f"\n----- Three-tier override: verdict=VALID overridden to INVALID (Tier 3 exhausted — false positive) -----")
                valid_code_flag = "INVALID"
                judge_response_text = (f"Tier 3 metamorphic testing budget exhausted "
                                       f"({self_correction_tier3_counter_code_max} attempts): code marked INVALID.")
                code_analysis = "None"
                tier3_exhausted = False  # reset for next iteration

            # Table 2 ★ final attempt row — emitted after all overrides have settled valid_code_flag.
            # Only for LASSI codes (zero_shot has no attempt log rows); compile/execute errors never reach here.
            if not zero_shot:
                _override_applied = (
                    "override3_tier3_exhausted" if _tier3_was_exhausted else
                    "override2_tiers_pass"      if _override2_fired     else
                    "override1_tier2_exhausted" if _tier2_was_exhausted else
                    "none"
                )
                _cb_idx = len(code_blocks) - 1
                _t1_instr = lassi.verify.extract_tier1_labels(code_blocks[-1])
                _t2_instr = lassi.verify.extract_tier2_labels(code_blocks[-1])
                attempt_log_buffer.append(_build_attempt_record(
                    app_name=file_name, trial_number=trial_number,
                    code_block_index=_cb_idx, attempt_number=attempt_number,
                    is_final_attempt=True, generation_type=current_generation_type,
                    sc_trigger=current_sc_trigger, llm_temperature=code_gen_temp,
                    compile_status="success", execute_status="success",
                    runtime_s=duration_refactor, energy_J=total_energy_joules_refactor,
                    ave_power_W=ave_power_refactor,
                    tier1_verdict=tier1_verdict, tier1_instrumented_labels=_t1_instr,
                    tier1_diagnostic=tier1_diagnostic,
                    tier2_verdict=tier2_verdict, tier2_instrumented_labels=_t2_instr,
                    tier2_diagnostic=tier2_diagnostic,
                    tier3_verdict=tier3_verdict, tier3_diagnostic=tier3_diagnostic,
                    judge_verdict=judge_raw_verdict, final_verdict=valid_code_flag,
                    override_applied=_override_applied,
                ))
                current_sc_trigger = None  # reset; next fresh generation starts clean

            # valid_code_flag can be: (1) "VALID", (2) "INVALID", or (3) "NONE"
            if not zero_shot:
                valid_code_history.append(valid_code_flag)
            else:
                valid_code_history.append("basellm_" + stage + "_" + valid_code_flag)
            
            # Substract out idle metrics: we do this during the power data collection
            total_energy_joules_lessidle = total_energy_joules_refactor - idle_energy_joules
            energy_history.append(round(total_energy_joules_refactor, 2))
            ave_power_lessidle = ave_power_refactor - idle_ave_power
            power_history.append(round(float(ave_power_refactor),2))
            runtime_history.append(round(float(duration_refactor),4))
            print("\nLASSI Iteration Report: Tracking Metrics Data")
            print("----- Energy consumed (J) -----")
            print("  Source code: " + str(round(sourcecode_total_energy_joules, 2)))
            print("  Refactored code: " + str(round(total_energy_joules_refactor, 2)))
            print("----- Average power (W) -----")
            print("  Source code: " + str(round(sourcecode_ave_power, 2)))
            print("  Refactored code: " + str(round(ave_power_refactor, 2)))
            print("----- Runtime (s) -----")
            print("  Source code: " + str(round(duration, 2)))
            print("  Refactored code: " + str(round(duration_refactor, 2)))
            print("----- Tracking metrics -----")
            print("  Energy (J) history: ", end=' ')
            print(energy_history)
            print("  Ave Power (W) history: ", end=' ')
            print(power_history)
            print("  Runtime (s) history: ", end=' ')
            print(runtime_history)
            print("  Valid code history: ", end=' ')
            print(valid_code_history)
            print("  Iteration log: ", end= ' ')
            print(iteration_log)
            # print("  LLM Judge Code Analysis: ")
            # print(code_analysis)

            if zero_shot:
                if stage == "stage1":
                    experiment_data["base_v1_energy_J"] = total_energy_joules_refactor
                    experiment_data["base_v1_avg_power_W"] = ave_power_refactor
                    experiment_data["base_v1_runtime_s"] = duration_refactor   # retained for backward compat
                    experiment_data["zeroshot_runtime_s"] = duration_refactor  # schema-aligned name (Table 1 ★)
                    experiment_data["zeroshot_compile_status"] = "success"     # reached here → compiled and executed OK
                    experiment_data["zeroshot_execute_status"] = "success"
                    experiment_data["base_v1_valid"] = valid_code_flag
                    experiment_data["base_v1_energy_diff"] = round(((total_energy_joules_refactor - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2) if power_measure and sourcecode_total_energy_joules != 0 else "NA"
                    experiment_data["base_v1_avg_power_diff"] = round(((ave_power_refactor - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2) if power_measure and sourcecode_ave_power != 0 else "NA"
                    experiment_data["base_v1_runtime_diff"] = round(((duration_refactor - duration) / duration) * 100 ,2) if duration != 0 else "NA"
                    # experiment_data["base_v1_optimization_methods"] = "None"
                elif stage == "stage2":
                    experiment_data["base_v2_energy_J"] = total_energy_joules_refactor
                    experiment_data["base_v2_avg_power_W"] = ave_power_refactor
                    experiment_data["base_v2_runtime_s"] = duration_refactor
                    experiment_data["base_v2_valid"] = valid_code_flag
                    experiment_data["base_v2_selfcorrections"] = iteration_log["v2"]["self_correction_count"]
                    experiment_data["base_v2_energy_diff"] = round(((total_energy_joules_refactor - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2) if power_measure and sourcecode_total_energy_joules != 0 else "NA"
                    experiment_data["base_v2_avg_power_diff"] = round(((ave_power_refactor - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2) if power_measure and sourcecode_ave_power != 0 else "NA"
                    experiment_data["base_v3_runtime_diff"] = round(((duration_refactor - duration) / duration) * 100 ,2) if duration != 0 else "NA"
                    experiment_data["base_v2_optimization_methods"] = "None"
            # Select optimization metric: energy (EE mode) or runtime (runtime/translation mode)
            track_history = energy_history if power_measure else runtime_history
            current_metric = total_energy_joules_refactor if power_measure else duration_refactor

            if (
                current_metric < track_history[tracking_best_version]
                and valid_code_flag == "VALID"
            ):
                # WE HAVE A WINNER!
                # sourcecode_total_energy_joules:  # or ave_power_refactor < sourcecode_ave_power:
                # Save the Refactoring Plan for this code version
                best_refactoring_plan = refactoring_plan
                # Save the LLM-as-a-Judge response for this code version
                best_refactored_code_judge = "LLM-as-a-Judge Response: \n\n" + judge_response_text + "\n\nCode Analysis: \n\n" + code_analysis
                compile_result = "energy_efficiency"
                if zero_shot:
                    if stage == "stage1":
                        # compile_result = ""  # Clear compile result so no iteration
                        # stage = "stage2"  # Switch to Stage 2 after one try.
                        # Force the switch to LASSI-EE pipeline.
                        stage = ""
                        change_strategy = True
                        zero_shot = False  # Stop base LLM attempt after one try.
                    elif stage == "stage2":
                        # NOTE: WE ARE NOT GOING TO DO STAGE 2 FOR THE FIRST LASSI-EE RESEARCH (mtd: 09-06-2025)
                        change_strategy = True
                        zero_shot = False  # Stop base LLM attempt after one try.

                """ Let's only care about the total energy consumption and not average power. """
                print("\n\n***** Generated refactored code is better than source or previous code! *****")
                print("... based on refactoring from code at index: " + str(tracking_best_version) + "\n\n")
                energy_efficiency_iteration += 1
                energy_efficiency_refactor_success = True
                # Save the output details for this "best" code version for metadata record.
                # Snapshot with dict() — input_content_llmjudge_dict is mutated by every
                # subsequent code run, so a reference assignment would always reflect the last run.
                best_refactored_code_dict = dict(input_content_llmjudge_dict)
                # Reset the LLM temperature parameter for subsquent code generation
                code_gen_temp = 0.2
                # I THINK WE HAVE THE FOLLOWING NOW WITH THE BETTER VERSION TRACKING:
                # TODO: self-review for multiple attempts to further improve energy efficiency
                # We need to set a flag that we had success at least one time.
                # Then, if follow-up is not a success, then STOP and return the latest success.
                # We need to track the refactor VERSION, so may need to have a list
                # May want a list for each code vesion as well. Whew. We are going to be tracking a lot here.
                if len(track_history) > 1:
                    if len(code_blocks) > 0:
                        # If we have saved a "best" code refactor,
                        # then compare the latest to the best
                        previous_metric = track_history[tracking_best_version]
                        current_metric_val = track_history[-1]
                    else:
                        # If we haven't saved a "best" code refactor,
                        # then just compare to the source code
                        previous_metric = track_history[0]
                        current_metric_val = track_history[-1]
                    previous_power = power_history[tracking_best_version] if len(power_history) > 1 else 0
                    current_power = power_history[-1] if len(power_history) > 1 else 0

                    # Track the "best" code generation
                    # If current_metric_val is < previous_metric,
                    # then update tracking version to this current code generation.
                    # Otherwise, do not change the tracking version (keep it the same)
                    if current_metric_val < previous_metric:
                        # Use len(code_blocks)-1: robust against code_version/code_blocks offset
                        # (code_blocks[0]=SOURCE_orig, [1]=SOURCE_verify, [2+]=generated codes).
                        tracking_best_version = len(code_blocks) - 1

                    print("\n\n----- " + color.BOLD + color.BLUE + "COMPARING CURRENT vs PREV VALUES " + color.END + " -----\n")
                    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)

                    # Calculate the fractional improvement
                    if previous_metric != 0:
                        metric_improvement = (previous_metric - current_metric_val) / previous_metric
                    else:
                        metric_improvement = current_metric_val
                    if power_measure and previous_power != 0:
                        power_improvement = (previous_power - current_power) / previous_power
                    else:
                        power_improvement = metric_improvement  # fall back to primary metric when not measuring power

                    # Check if an improvement is < 1% AND that measurement is still better than SOURCE_verify (index 1)
                    metric_stagnant = metric_improvement < 0.011 and current_metric_val < track_history[1]
                    power_stagnant = power_measure and power_improvement < 0.011 and current_power < power_history[0]

                    # Stopping condition:
                    # Stop if at least one metric is stagnant, while the other is not improving by more than 1%
                    # Do NOT stop if both are worsening (negative improvement values)
                    # Adjustment (03.20.2025): only allow stagnation stop on energy consuption, increase min ee iteration
                    if (
                        energy_efficiency_iteration > 4
                        and
                        (metric_stagnant)  # or power_stagnant)
                        and
                        (metric_improvement > 0 or power_improvement > 0)
                    ):
                        print("\n\n-----> LASSI IS FINISHED! <-----\n")
                        print(" metric_stagnant = " + str(metric_stagnant))
                        print(" power_stagnant = " + str(power_stagnant))
                        print(" metric_improvement = " + str(round(metric_improvement, 2)))
                        print(" power_improvement = " + str(round(power_improvement, 2)))
                        print("\n\n")
                        break
            else:
                # TODO: Change label: invalid or not more energy efficient
                print("\n\n----- " + color.BOLD + color.BLUE + "WE HAVE GENERATED CODE THAT IS INVALID OR NOT MORE ENERGY EFFICIENT" + color.END + " -----\n")
                print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
                
                compile_result = "energy_efficiency"
                if zero_shot:
                    if stage == "stage1":
                        # compile_result = ""  # Clear compile result so no iteration
                        # stage = "stage2" # Switch to Stage 2 after one try.
                        # Force the switch to LASSI-EE pipeline.
                        stage = ""
                        change_strategy = True
                        zero_shot = False  # Stop base LLM attempt after one try.
                    elif stage == "stage2":
                        # NOTE: WE ARE NOT GOING TO DO STAGE 2 FOR THE FIRST LASSI-EE RESEARCH (mtd: 09-06-2025)
                        change_strategy = True
                        zero_shot = False  # Stop base LLM attempt if improvement not acheived after iteration.
                else:
                    # Save this refactoring plan for the memory of "bad" plans
                    bad_refactor_plans.append(refactoring_plan)
                energy_efficiency_refactor_success = False
                                
                if (valid_code_flag == "VALID"):
                    print("\n\n***** Generated refactored code is NOT better than source or previous code... *****")
                    if not zero_shot:
                        print("\n" + color.GREEN + color.BOLD + " ----- OPTIMIZATION REVIEW: Let's provide feedback on the metrics. -----" + color.END + "\n\n")
                elif valid_code_flag != "VALID":
                    print("\n\n" + color.RED + color.BOLD + " ----- GENERATED REFACTORED CODE: May be INVALID or not determined. ----- " + color.END + "\n\n")
                    # We're done!
                    # break
            '''
            What's the strategy? The LLM was already provided the original code power.
            So, if the generated code power is < than original code power, then exit this loop and finish.
            Else, prompt the LLM again with the original code power, its generated code power, and a refactor again,
            restating the goal that it is should try to reduce the power consumption on the system.
            ... Source code energy variable: sourcecode_total_energy_joules
            ... Source code average power: sourcecode_ave_power
            ... Generated code energy varaible: total_energy_joules
            ... Generated code average power: total_energy_joules
            + We also need to include details of the system in the context.
            + We also need to include better context for coding best practices.
            + Continue this self-feedback loop incrementing self_correction_power_counter until maxing at 7 tries.
            '''

    # END OF LASSI-EE PIPELIE WHILE LOOP
    # ----------------------------------

    end_time = time.time() - start_time
    
    print("\n\n----- " + color.BOLD + color.GREEN + "LASSI-EE CODE GENERATION COMPLETE" + color.END + " -----\n")

    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
    
    end_time_minutes = round(end_time/60, 2)
    print("CODE GENERATION TIME: " + str(end_time_minutes) + " minutes.")
    if valid_code_flag != "VALID":
        print(f"  Stopped LASSI early due to invalid code on the last generation version number = {code_version - 1}.")
    print(" Total Code Error Self-correction iteration count: " + str(self_correction_counter))
    # if not baseline_oneshot_goal:
    #     print(" Self-improvement ee refactor count: " + str(self_correction_power_counter))
    print(" Total Energy Efficiency Self-improvement iteration count: " + str(len(energy_history) - 2))
    print(" Energy efficiency improvements: " + str(energy_efficiency_iteration))
    # code_blocks layout: [0]=SOURCE_orig, [1]=SOURCE_verify, [2]=vanilla LLM Stage 1,
    # [3+]=LASSI-generated codes. tracking_best_version indexes into this layout.
    if tracking_best_version == 0:
        tracking_best_version_label = "SOURCE_orig"
    elif tracking_best_version == 1:
        tracking_best_version_label = "SOURCE_verify"
    elif tracking_best_version == 2:
        tracking_best_version_label = "STANDALONE BASE LLM"
    else:
        tracking_best_version_label = "LASSI v" + str(tracking_best_version)
    print(" Best code, version: " + tracking_best_version_label + "\n" )

    # code_blocks layout: [0]=SOURCE_orig, [1]=SOURCE_verify, [2]=vanilla LLM, [3+]=LASSI.
    # If no generated code beat SOURCE_verify, check whether any had lower ave power (EE mode).
    smallest_index, smallest_ave_power_value = None, None  # Initialize before conditionals
    if tracking_best_version_label in ("SOURCE_orig", "SOURCE_verify"):
        source_ave_power_value = power_history[0]
        # Filter pairs (index, value) where:
        # - value is less than source_ave_power_value
        # - index is not 0 or 1 (exclude both source versions)
        # - the corresponding valid_code_history value is "VALID"
        filtered = [
            (i, v)
            for i, v in enumerate(power_history)
            if v < source_ave_power_value and i > 1 and valid_code_history[i] == "VALID"
        ]
        if filtered:
            smallest_index, smallest_ave_power_value = min(filtered, key=lambda x: x[1])

        if smallest_ave_power_value is not None:
            print(" Best ave power in refactored code, version: " + str(smallest_index) + "\n" )
    elif "BASE LLM" in tracking_best_version_label:
        source_ave_power_value = min(power_history[0], power_history[1], power_history[2])
        # Filter pairs (index, value) where:
        # - value is less than source_ave_power_value
        # - index is not 0, 1, or 2 (exclude both source versions and vanilla LLM)
        # - the corresponding valid_code_history value is "VALID"
        filtered = [
            (i, v)
            for i, v in enumerate(power_history)
            if v < source_ave_power_value and i > 2 and valid_code_history[i] == "VALID"
        ]
        if filtered:
            smallest_index, smallest_ave_power_value = min(filtered, key=lambda x: x[1])

        if smallest_ave_power_value is not None:
            print(" Best ave power over Base LLM in refactored code, version: " + str(smallest_index) + "\n" )


    # ============================================================
    # POST-LOOP: Best Clean Run
    # Strip Tier 1+2 instrumentation from the best LASSI-generated code, recompile,
    # and execute to capture uninstrumented performance metrics for fair comparison.
    # If compile/execute fails: report gracefully and fall back to instrumented metrics.
    # This run does NOT affect rankings, history, or Tier verification; it is purely
    # for reporting a clean uninstrumented runtime/energy baseline for the paper.
    # ============================================================
    lassiee_clean_status = "skipped"
    lassiee_clean_runtime_s = "NA"
    lassiee_clean_energy_J = "NA"
    lassiee_clean_runtime_diff = "NA"
    lassiee_clean_energy_diff = "NA"
    best_clean_code = None  # uninstrumented code for final LLM analysis

    if tracking_best_version > 1:
        best_code_raw = code_blocks[tracking_best_version]
        best_clean_code = lassi.verify.strip_verify_blocks(best_code_raw)

        best_clean_path = os.path.join(experiment_data_directory, file_name_save + "_best_clean.cu")
        with open(best_clean_path, 'w') as _bcf:
            _bcf.write(best_clean_code)

        print("\n----- [Best Clean Run] Compiling uninstrumented best code ... -----")
        compile_err_clean, best_clean_exe = lassi.code_manager.compile_code(
            best_clean_path, target_lang, code_compiler, code_compiler_kwds
        )

        if compile_err_clean:
            print(f"[Best Clean Run] COMPILE ERROR — using instrumented metrics for final report.\n{str(compile_err_clean)[:300]}")
            lassiee_clean_status = "compile_error"
        else:
            print("[Best Clean Run] Executing ...")
            try:
                _clean_result, _clean_stdout, _clean_energy_J, _clean_avg_power, _clean_std_power, _clean_duration, _clean_metrics = lassi.code_manager.execute_code_selectvendor(
                    experiment_data_directory,
                    tracking_best_version,  # code_version label
                    False,                  # zero_shot=False (it's a LASSI code)
                    trial_number,
                    directory,
                    best_clean_exe,
                    exe_input_parameters,
                    power_measure,
                    "best_clean",           # code_source label
                    target_lang,
                    build_script,
                    make_script,
                    gpu_vendor="auto",
                    wait=3                  # shorter settle time — no power thread for non-power runs
                )
                if _clean_result != "0":
                    print("[Best Clean Run] EXECUTE ERROR — using instrumented metrics for final report.")
                    lassiee_clean_status = "execute_error"
                else:
                    lassiee_clean_status = "success"
                    lassiee_clean_runtime_s = round(_clean_duration, 4)
                    if power_measure and _clean_energy_J > 0:
                        lassiee_clean_energy_J = round(_clean_energy_J, 4)
                        if sourcecode_total_energy_joules and sourcecode_total_energy_joules > 0:
                            lassiee_clean_energy_diff = round(100.0 * (_clean_energy_J - sourcecode_total_energy_joules) / sourcecode_total_energy_joules, 2)
                    # Baseline: runtime_history[0] = SOURCE_orig (uninstrumented) — the fair comparison.
                    # duration at this point is SOURCE_verify (overwritten at Stage 0b line ~371).
                    _src_rt_baseline = runtime_history[0] if runtime_history and runtime_history[0] > 0 else None
                    if _src_rt_baseline:
                        lassiee_clean_runtime_diff = round(100.0 * (_clean_duration - _src_rt_baseline) / _src_rt_baseline, 2)
                    print(f"[Best Clean Run] Runtime: {lassiee_clean_runtime_s:.4f}s  ({lassiee_clean_runtime_diff}% vs source uninstrumented)")
                    if power_measure:
                        print(f"[Best Clean Run] Energy: {lassiee_clean_energy_J} J  ({lassiee_clean_energy_diff}% vs source)")
            except Exception as _clean_ex:
                print(f"[Best Clean Run] EXECUTE EXCEPTION — using instrumented metrics for final report.\n{_clean_ex}")
                lassiee_clean_status = "execute_error"

    # Prompt the LLM to compare the codes between SOURCE and the "BEST" refactored version.
    # Only run if a generated code exists (tracking_best_version > 1; indices 0-1 are source versions).
    if (tracking_best_version > 1):

        # TODO: Complete organizing this into lassi.llm_methods.llm_code_compare()
        
        print("\n\n----- " + color.BOLD + color.BLUE + "FINAL CODE COMPARISON: Source Code vs. Best Refactored Code" + color.END + " -----\n")
        print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        if detected_vendor.upper() == "NVIDIA":
            compare_code = prompts.get_code_comparison("CUDA_cpp")
        elif detected_vendor.upper() == "AMD":
            compare_code = prompts.get_code_comparison("HIP_cpp")
        compare_code = compare_code + "\n\n" + "[BEGIN SOURCE CODE AND RESULTS]\n\n" + original_input_code  # uninstrumented source (no Tier 1/2 blocks)
        compare_code = compare_code + "\n\nOUTPUT: \n\n" + input_content_llmjudge_dict["source_stdout"]
        compare_code = compare_code + "\n\nCode execution duration: " + str(round(duration, 2)) + " s"
        compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(sourcecode_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(sourcecode_ave_power,2)) + " W"
        compare_code = compare_code + "\n\n[/END SOURCE CODE AND RESULTS]"
        
        # Use uninstrumented best_clean_code if available; fall back to instrumented version
        _compare_best_code = best_clean_code if best_clean_code is not None else code_blocks[tracking_best_version]
        # Use clean runtime/energy for comparison if the clean run succeeded
        _compare_runtime = lassiee_clean_runtime_s if lassiee_clean_status == "success" and lassiee_clean_runtime_s != "NA" else runtime_history[tracking_best_version]
        _compare_energy = lassiee_clean_energy_J if lassiee_clean_status == "success" and lassiee_clean_energy_J != "NA" else energy_history[tracking_best_version]
        _compare_note = " (uninstrumented clean run)" if lassiee_clean_status == "success" else " (instrumented — clean run unavailable)"

        compare_code = compare_code + "\n\n" + "[BEGIN REFACTORED CODE AND RESULTS]\n\n" + _compare_best_code
        compare_code = compare_code + "\n\nOUTPUT: \n\n" + best_refactored_code_dict["refactored_stdout"]
        compare_code = compare_code + "\n\nCode execution duration: " + str(round(_compare_runtime, 2)) + " s" + _compare_note
        compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(_compare_energy if isinstance(_compare_energy, (int, float)) else energy_history[tracking_best_version],2)) + " J" + "\nAverage Power: " + str(round(power_history[tracking_best_version],2)) + " W"
        compare_code = compare_code + "\n\n[/END REFACTORED CODE AND RESULTS]"

        compare_code_prompt_tokens = lassi.token_analysis.token_count(compare_code)

        if verifyprompt:
            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE CODE COMPARISON PROMPT " + color.END + " (prompt token count: " + str(compare_code_prompt_tokens) + "): \n\n" + compare_code + "\n")

        # Send prompt to LLM endpoint
        print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "LLM prompt to compare codes." + color.END + "\n\n")
        code_comparison_response = lassi.llm_methods.llm_call("code comparison", llm_source, model, num_ctx, system_prompt, compare_code, display_response, verifyprompt)
        

    # MEASURE SIMILARITY BETWEEN GENERATED TRANSLATION CODE AND ORIGINAL TARGET CODE
    # similarity_tokens = lassi.token_analysis.token_similarity(target_code, code_block, "tokenize")
    # print(f"Token similarity (tokenize): {similarity_tokens:.2f}")

    # similarity_tokens_tiktoken = lassi.token_analysis.token_similarity(target_code, code_block, "tiktoken")
    # print(f"Token similarity (tiktoken): {similarity_tokens_tiktoken:.2f}")
    
    # similarity_lines = lassi.token_analysis.compare_lines_with_reordering(target_code, code_block)
    # print(f"Line-based similarity with reordering: {similarity_lines:.2f}")

    print("\n\nEnergy consumed history: ")
    print(energy_history)
    print("\nAve power history: ")
    print(power_history)
    print("\nRuntime history: ")
    print(runtime_history)
    print("\nValid code history: ")
    print(valid_code_history)
    
    meta_data_file += "\n\n"
    meta_data_file += "----- PIPELINE METRICS -----"
    meta_data_file += "\n"
    meta_data_file += "GPU Device Vendor: " + str(detected_vendor)
    meta_data_file += "\n"
    meta_data_file += "Experiment date-time: " + str(dt_string)
    meta_data_file += "\n"
    meta_data_file += "Time (sec): " + str(round(end_time, 2)) + " | " + str(round(end_time/60, 2)) + " (min)"
    meta_data_file += "\n"
    meta_data_file += "Experiment trial: " + str(trial_number)
    meta_data_file += "\n"
    # meta_data_file += f"Token similarity (tokenize): {similarity_tokens:.2f}"
    # meta_data_file += "\n"
    # meta_data_file += f"Token similarity (tiktoken): {similarity_tokens_tiktoken:.2f}"
    # meta_data_file += "\n"
    # meta_data_file += f"Line-based similarity with reordering: {similarity_lines:.2f}"
    # meta_data_file += "\n"
    meta_data_file += "Code Error Self-correction iteration count: " + str(self_correction_counter)
    meta_data_file += "\n"
    # ee_iterations = number of times a LASSI-generated code became the new best (metric improved).
    # 0 means LASSI never improved on the source. Distinct from fresh_generation_count (total attempts).
    ee_iterations = energy_efficiency_iteration
    meta_data_file += "Energy Efficiency Self-improvement iteration count: " + str(ee_iterations)  # str(self_correction_power_counter)
    
    if yes_power_feedback:
        
        print("\n\n----- " + color.BOLD + color.BLUE + "SAVING FINAL RESULTS" + color.END + " -----\n")
        print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
        
        # Save "best" index: if no generated code beat the source/vanilla baseline on the
        # primary metric but one had lower ave power, use that as the "best" for logging.
        if (tracking_best_version_label in ("SOURCE_orig", "SOURCE_verify", "STANDALONE BASE LLM")) and smallest_index is not None:
            best_index = smallest_index
        else:
            best_index = tracking_best_version

        # TODO: Instead of N/A we want to record the least 'energy_history' of the best_index > 1
        # Then, record the corresponding values for this same index
        experiment_data["lassiee_energy_J"] = energy_history[best_index] if best_index > 0  else "N/A"
        experiment_data["lassiee_avg_power_W"] = power_history[best_index] if best_index > 0  else "N/A"
        experiment_data["lassiee_runtime_s"] = runtime_history[best_index] if best_index > 0  else "N/A"
        _vh = valid_code_history[best_index] if best_index > 0 else "N/A"
        if _vh in ("SOURCE_orig", "SOURCE_verify"):
            _vh = "SOURCE"  # LASSI failed to improve; best is source — treated as failed generation for pass@k metric
        experiment_data["lassiee_valid"] = _vh
        experiment_data["lassiee_version"] = str(best_index)
        # iteration_log only has entries for generated codes (v2+); SOURCE_verify (v1) is never added.
        # Guard must be > 1, not > 0, to avoid KeyError when best_index = 1 (SOURCE_verify stayed best).
        experiment_data["lassiee_llm_temp"] = iteration_log["v" + str(best_index)]["LLM_temperature"] if best_index > 1  else "N/A"
        experiment_data["lassiee_selfcorrections"] = iteration_log["v" + str(best_index)]["self_correction_count"] if best_index > 1  else "N/A"
        experiment_data["lassiee_eeiterations"] = iteration_log["v" + str(best_index)]["energy_refactor_count"] if best_index > 1  else "N/A"
        # best_index > 1 guard: when best is SOURCE_verify (index 1), diffs would reflect instrumentation
        # overhead (SOURCE_verify vs SOURCE_orig), not an optimization result — suppress as N/A.
        experiment_data["lassiee_energy_diff"] = round(((energy_history[best_index] - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2) if best_index > 1 and power_measure and sourcecode_total_energy_joules != 0 else "N/A"
        experiment_data["lassiee_avg_power_diff"] = round(((power_history[best_index] - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2) if best_index > 1 and power_measure and sourcecode_ave_power != 0 else "N/A"
        # Baseline: runtime_history[0] = SOURCE_orig (uninstrumented). duration at this point
        # is SOURCE_verify runtime (overwritten at Stage 0b); using it would give an inflated diff.
        experiment_data["lassiee_runtime_diff"] = round(((runtime_history[best_index] - runtime_history[0]) / runtime_history[0]) * 100 ,2) if best_index > 1 and len(runtime_history) > 0 and runtime_history[0] != 0 else "N/A"
        match_code_comparison_response = re.search(r"\[OPTIMIZATION LIST\](.*?)\[/OPTIMIZATION LIST\]", code_comparison_response, re.DOTALL)
        if match_code_comparison_response:
            optimization_list_text = match_code_comparison_response.group(1).strip()
        else:
            if tracking_best_version_label == "SOURCE":
                optimization_list_text = "SOURCE"
            else:
                optimization_list_text = "None"
        experiment_data["lassiee_optimization_methods"] = str(optimization_list_text)
        # Table 1 ★ per-type self-correction counts (recompute total to fix historic undercounting)
        experiment_data["compile_selfcorrections"] = compile_sc_count
        experiment_data["execute_selfcorrections"] = execute_sc_count
        experiment_data["tier1_selfcorrections"] = tier1_sc_count
        experiment_data["tier2_selfcorrections"] = tier2_sc_count
        experiment_data["tier3_selfcorrections"] = tier3_sc_count
        experiment_data["total_selfcorrections"] = (compile_sc_count + execute_sc_count +
                                                     tier1_sc_count + tier2_sc_count + tier3_sc_count)
        experiment_data["global_counter_at_end"] = self_correction_verify_counter
        experiment_data["total_code_versions"] = fresh_generation_count  # all fresh LASSI code generations started, regardless of outcome
        experiment_data["total_code_attempts"] = len(attempt_log_buffer)
        experiment_data["tier3_inactive_relations"] = ",".join(tier3_inactive_relations)
        experiment_data["total_eeiterations"] = ee_iterations
        experiment_data["pipeline_runtime_min"] = round(end_time/60, 2)

    # --- LASSI-VERIFY SUMMARY: populate experiment_data fields from verification_log ---
    # Computed outside the yes_power_feedback block so they are always recorded.
    tier2_results = verification_log.get("tier2_results", [])
    discrepancies = verification_log.get("judge_discrepancies", [])
    experiment_data["tier2_pass_count"] = verification_log.get("verified_pass_count", 0)
    # tier2_fail_total / tier3_fail_total: recompute from attempt_log_buffer to capture all retry failures
    # (verification_log["tier2_results"] only stores final per-version outcomes, missing intermediate retries)
    experiment_data["tier2_fail_total"] = sum(1 for r in attempt_log_buffer if str(r.get("tier2_verdict", "")).startswith("FAIL"))
    experiment_data["judge_false_positives"] = sum(1 for d in discrepancies if d.get("type") == "false_positive")
    experiment_data["judge_false_negatives"] = sum(1 for d in discrepancies if d.get("type") == "false_negative")
    tier3_results_vlog = verification_log.get("tier3_results", [])
    experiment_data["tier3_pass_count"] = verification_log.get("tier3_pass_count", 0)
    experiment_data["tier3_fail_total"] = sum(1 for r in attempt_log_buffer if str(r.get("tier3_verdict", "")).startswith("FAIL"))
    experiment_data["tier3_active_relations"] = ",".join(verification_log.get("tier3_active_relations", []))

    # --- Table 2 ★ attempt log: mark best code row, then write to CSV ---
    if yes_power_feedback and attempt_log_buffer:
        # best_index is set inside the yes_power_feedback block above
        for row in attempt_log_buffer:
            if row["code_block_index"] == best_index and row["is_final_attempt"]:
                row["is_best_code"] = True
                break  # exactly one row should receive this flag
        if attempt_log_file_path:
            with open(attempt_log_file_path, "a", newline="") as _alf:
                import csv as _csv
                _writer = _csv.DictWriter(_alf, fieldnames=ATTEMPT_LOG_FIELDNAMES)
                _writer.writerows(attempt_log_buffer)
            print(f"\nAttempt log: {len(attempt_log_buffer)} rows written to {attempt_log_file_path}")

    # --- BEST CLEAN RUN: uninstrumented performance of best LASSI code ---
    # lassiee_clean_* variables set in the post-loop block above.
    experiment_data["lassiee_clean_status"] = lassiee_clean_status
    experiment_data["lassiee_clean_runtime_s"] = lassiee_clean_runtime_s
    experiment_data["lassiee_clean_runtime_diff"] = lassiee_clean_runtime_diff
    experiment_data["lassiee_clean_energy_J"] = lassiee_clean_energy_J
    experiment_data["lassiee_clean_energy_diff"] = lassiee_clean_energy_diff

    # --- POST-PIPELINE: Three-Tier Verification of Base LLM (Zero-Shot) Code ---
    # Apply the same three-tier verification framework to the base LLM code to determine
    # its trustworthiness independent of the LLM judge. No self-correction: this step
    # measures the code "as-is" to provide an honest, empirically grounded verdict.
    # If any tier fails, base_llm_verified_valid = False (may override the prior judge verdict).
    # Gated on run_tier2 and a non-empty passthrough spec — if Tier 2 never ran, we have no
    # SOURCE_verify reference output to compare against.
    base_llm_tier1_verdict = "NA"
    base_llm_tier2_verdict = "NA"
    base_llm_tier3_verdict = "NA"
    base_llm_verified_valid = "NA"
    base_llm_verification = {
        "skipped": True,
        "reason": "base LLM code not available or Tier 2 not enabled",
        "tier1_verdict": "NA", "tier2_verdict": "NA", "tier3_verdict": "NA", "verified_valid": "NA",
    }

    if len(code_blocks) > 2 and run_tier2 and passthrough_spec_output != "":
        print("\n\n=== POST-PIPELINE: Three-Tier Verification of Base LLM Code ===")
        print("Instrumenting zero-shot code with same Tier 1+2 observation points as SOURCE_verify ...")
        try:
            # Step 1: Template-guided instrumentation — replicate SOURCE_verify's LASSI-VERIFY blocks
            # into the base LLM code at equivalent locations (single LLM call, Tier 1 + Tier 2).
            base_llm_instrumented = lassi.verify.code_insert_instrumentation_from_template(
                code_blocks[2],     # base LLM code (generated from SOURCE_orig, no instrumentation)
                input_code,         # SOURCE_verify (Tier 1 + Tier 2 instrumented — the template)
                llm_source, model, num_ctx, system_prompt,
                display_response, verifyprompt
            )

            if not base_llm_instrumented:
                print("[Base LLM Verification] Instrumentation LLM returned empty code — skipping.")
                base_llm_verification["reason"] = "instrumentation_failed"
            else:
                # Write instrumented code to disk
                base_llm_verify_path = os.path.join(
                    experiment_data_directory, file_name_save + "_base_llm_verify.cu"
                )
                with open(base_llm_verify_path, 'w') as _blf:
                    _blf.write(base_llm_instrumented)

                # Step 2: Compile instrumented base LLM code
                print("[Base LLM Verification] Compiling instrumented base LLM code ...")
                compile_err_bl, base_llm_exe = lassi.code_manager.compile_code(
                    base_llm_verify_path, target_lang, code_compiler, code_compiler_kwds
                )

                if compile_err_bl:
                    print(f"[Base LLM Verification] COMPILE ERROR:\n{str(compile_err_bl)[:300]}")
                    base_llm_verification = {
                        "skipped": False, "reason": "compile_error",
                        "tier1_verdict": "NA", "tier2_verdict": "NA",
                        "tier3_verdict": "NA", "verified_valid": False,
                    }
                else:
                    # Step 3: Execute (power_measure=False — correctness run only, no energy overhead)
                    print("[Base LLM Verification] Executing ...")
                    _bl_result, _bl_stdout, _bl_energy_J, _bl_avg_power, _bl_std_power, \
                        _bl_duration, _bl_metrics = lassi.code_manager.execute_code_selectvendor(
                        experiment_data_directory,
                        2,                    # code_version = base LLM index
                        True,                 # zero_shot=True
                        trial_number,
                        directory,
                        base_llm_exe,
                        exe_input_parameters,
                        False,                # power_measure=False (correctness run only)
                        "base_llm_verify",    # code_source label
                        target_lang,
                        build_script,
                        make_script,
                        gpu_vendor="auto",
                        wait=3
                    )

                    if _bl_result != "0":
                        print("[Base LLM Verification] EXECUTE ERROR")
                        base_llm_verification = {
                            "skipped": False, "reason": "execute_error",
                            "tier1_verdict": "NA", "tier2_verdict": "NA",
                            "tier3_verdict": "NA", "verified_valid": False,
                        }
                    else:
                        # Step 4: Tier 1 — invariant assertion check
                        _bl_tier1_verdict, _bl_tier1_diag = lassi.verify.check_tier1_assertions(_bl_stdout)
                        print(f"[Base LLM Verification] Tier 1: {_bl_tier1_verdict}")
                        if _bl_tier1_verdict == "FAIL":
                            print(_bl_tier1_diag.get("summary", ""))
                        base_llm_tier1_verdict = _bl_tier1_verdict

                        # Step 5: Tier 2 — empirical equivalence vs SOURCE_verify reference
                        _bl_tier2_verdict, _bl_tier2_diag = lassi.verify.compare_passthrough_outputs(
                            spec_stdout=passthrough_spec_output,
                            generated_stdout=_bl_stdout,
                            tolerance=tier2_tolerance,
                            llm_source=llm_source, model=model, num_ctx=num_ctx,
                            system_prompt=system_prompt, verifyprompt=verifyprompt
                        )
                        print(f"[Base LLM Verification] Tier 2: {_bl_tier2_verdict}")
                        if _bl_tier2_diag.get("failures"):
                            print(_bl_tier2_diag.get("summary", ""))
                        base_llm_tier2_verdict = _bl_tier2_verdict

                        # Step 6: Tier 3 — metamorphic testing (if active and Tier 2 not failed)
                        _bl_tier3_verdict = "SKIP"
                        _bl_tier3_results = []
                        if run_tier3 and tier3_active_relations and not _bl_tier2_verdict.startswith("FAIL"):
                            _bl_tier3_verdict, _bl_tier3_results, _bl_tier3_diag = \
                                lassi.verify.run_metamorphic_variants(
                                base_llm_instrumented,
                                tier3_active_relations,
                                tier3_contract,
                                _bl_stdout,        # base LLM's own output as Tier 3 baseline (self-referential)
                                llm_source, model, num_ctx, system_prompt,
                                display_response,
                                directory, file_name, code_compiler, code_compiler_kwds,
                                build_script, make_script, code_build_script,
                                exe_input_parameters,
                                experiment_data_directory=experiment_data_directory,
                                code_version="base_llm",
                                verifyprompt=verifyprompt
                            )
                            print(f"[Base LLM Verification] Tier 3: {_bl_tier3_verdict}")
                        base_llm_tier3_verdict = _bl_tier3_verdict

                        # Step 7: Final verdict — any FAIL in any tier overrides prior judge decision
                        _bl_all_pass = (
                            _bl_tier1_verdict == "PASS" and
                            not _bl_tier2_verdict.startswith("FAIL") and
                            not _bl_tier3_verdict.startswith("FAIL")
                        )
                        base_llm_verified_valid = _bl_all_pass
                        base_llm_verification = {
                            "skipped": False,
                            "tier1_verdict": _bl_tier1_verdict,
                            "tier1_diagnostic": _bl_tier1_diag,
                            "tier2_verdict": _bl_tier2_verdict,
                            "tier2_diagnostic": _bl_tier2_diag,
                            "tier3_verdict": _bl_tier3_verdict,
                            "tier3_results": _bl_tier3_results,
                            "verified_valid": _bl_all_pass,
                        }
                        print(f"\n[Base LLM Verification] FINAL VERDICT: "
                              f"{'VALID' if _bl_all_pass else 'INVALID'} "
                              f"(T1:{_bl_tier1_verdict}, T2:{_bl_tier2_verdict}, T3:{_bl_tier3_verdict})")

        except Exception as _bl_ex:
            print(f"[Base LLM Verification] EXCEPTION: {_bl_ex}")
            base_llm_verification = {
                "skipped": False, "reason": f"exception: {str(_bl_ex)}",
                "tier1_verdict": "NA", "tier2_verdict": "NA", "tier3_verdict": "NA",
                "verified_valid": False,
            }

    verification_log["base_llm_verification"] = base_llm_verification
    experiment_data["base_llm_tier1_verdict"] = base_llm_tier1_verdict
    experiment_data["base_llm_tier2_verdict"] = base_llm_tier2_verdict
    experiment_data["base_llm_tier3_verdict"] = base_llm_tier3_verdict
    experiment_data["base_llm_verified_valid"] = base_llm_verified_valid

    # --- LASSI-VERIFY JSON SIDECAR: save full verification_log for detailed inspection ---
    # tier2_spec_output (large stdout) and sourcecode_analysis (large text) are excluded here;
    # they are captured in the metadata .txt file. All other fields are saved verbatim.
    vlog_save = {k: v for k, v in verification_log.items() if k not in ("tier2_spec_output", "sourcecode_analysis")}
    vlog_filename = os.path.join(experiment_data_directory, "result_verification_log_" + file_name_save + "_Trial" + str(trial_number) + ".json")
    with open(vlog_filename, 'w') as vlog_file:
        json.dump(vlog_save, vlog_file, indent=2, default=str)
    print(f"\nVerification log saved: {vlog_filename}")

    if yes_power_feedback:
        meta_data_file += "\n\n\n----- ENERGY EFFICIENCY METRICS -----\n"
        meta_data_file += "Best refactored code version: " + str(best_index)
        meta_data_file += "\nLLM Judge Validation Result: " + str(valid_code_history[best_index])
        meta_data_file += "\n"
        meta_data_file += "Total Energy (J): "
        meta_data_file += "  Idle: " + str(round(idle_energy_joules, 2))
        meta_data_file += "  Source code: " + str(round(sourcecode_total_energy_joules, 2))
        meta_data_file += "  Base LLM code: " + (str(round(energy_history[2], 2)) if len(energy_history) > 2 else "NA")
        meta_data_file += "  Best Refactored code: " + str(round(energy_history[best_index], 2))
        meta_data_file += "\n"
        meta_data_file += "Average Power (W): "
        meta_data_file += "  Idle: " + str(round(idle_ave_power, 2))
        meta_data_file += "  Source code: " + str(round(sourcecode_ave_power, 2))
        meta_data_file += "  Base LLM code: " + (str(round(power_history[2], 2)) if len(power_history) > 2 else "NA")
        meta_data_file += "  Best Refactored code: " + str(round(power_history[best_index], 2))
        meta_data_file += "\n"
        # Runtime: use uninstrumented values wherever available.
        # - Source code: runtime_history[0] = SOURCE_orig (uninstrumented; no Tier 1/2 overhead)
        # - Base LLM code: runtime_history[2] = zero-shot output (generated from SOURCE_orig, no LASSI-VERIFY blocks)
        # - Best refactored code: prefer lassiee_clean_runtime_s (strip-and-rerun) if available;
        #   fall back to instrumented runtime_history[best_index] with a note.
        # Idle runtime is omitted — not meaningful for runtime-performance mode.
        _base_llm_rt = str(round(runtime_history[2], 4)) if len(runtime_history) > 2 else "NA"
        _best_rt_val = lassiee_clean_runtime_s if lassiee_clean_runtime_s != "NA" else runtime_history[best_index]
        _best_rt_note = " (uninstrumented)" if lassiee_clean_runtime_s != "NA" else " (instrumented — clean run unavailable)"
        meta_data_file += "Runtime (s): "
        meta_data_file += "  Source code (uninstrumented): " + str(round(runtime_history[0], 4))
        meta_data_file += "  Base LLM code: " + _base_llm_rt
        meta_data_file += "  Best Refactored code: " + str(round(_best_rt_val, 4)) + _best_rt_note
    if yes_execute:
        meta_data_file += "\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n"
        meta_data_file += best_refactored_code_dict["source_stdout"]
        meta_data_file += "\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n"
        meta_data_file += best_refactored_code_dict["refactored_stdout"]
    if yes_power_feedback:
        meta_data_file += "\n\n\n----- FINAL CODE COMPARISON SOURCE vs BEST -----\n"
        meta_data_file += code_comparison_response
        meta_data_file += "\n\n\n----- REFACTORING PLAN -----\n"
        meta_data_file += best_refactoring_plan
        meta_data_file += "\n\n\n----- LLM-as-a-JUDGE RESPONSE -----\n"
        meta_data_file += best_refactored_code_judge
        meta_data_file += "\n\n\n----- TRACKED METRICS -----\n"
        meta_data_file += "Energy (J) history: " + str(energy_history)
        meta_data_file += "\n"
        meta_data_file += "Ave Power (W) history: " + str(power_history)
        meta_data_file += "\n"
        meta_data_file += "Runtime (s) history: " + str(runtime_history)
        meta_data_file += "\n"
        meta_data_file += "Valid code history: " + str(valid_code_history)
        meta_data_file += "\n"
        meta_data_file += "Iteration log: " + str(iteration_log)
            
    # Write meta data for this experiment
    if make_script:
        # filename = directory + "/" + "metadata" + "_" + experiment + ".txt"
        filename = os.path.join(experiment_data_directory, "result_metadata" + "_" + experiment.replace("/", "_") + "_Trial" + str(trial_number) + ".txt")
    else:
        # filename = directory + "/" + "metadata" + "_" + file_name_save + ".txt"
        filename = os.path.join(experiment_data_directory, "result_metadata" + "_" + file_name_save + "_Trial" + str(trial_number) + ".txt")
    with open(filename, 'w') as file:
        file.write(meta_data_file)
    file.close()
    print("\nMetadata file: " + str(filename))

    # Append collected data to experiment spreadsheet
    print("\nSave experimental data to file.")
    fieldnames = list(experiment_data.keys())
    with open(experiment_data_file_object, "a", newline="") as exp_file:
        writer = csv.DictWriter(exp_file, fieldnames=fieldnames)
        writer.writerow(experiment_data)
    exp_file.close()

    # --- MONITOR LOG: auto-save to experiment folder alongside other output files ---
    monitor_log_path = os.path.join(experiment_data_directory,
                                    "result_monitor_log_" + file_name_save + "_Trial" + str(trial_number) + ".txt")
    lassi.monitor.save_log(monitor_log_path)

    return True