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
from lassi.get_power import report_gpu_memory, query_gpu_state
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary
from lassi.gpu_detector import detect_gpu_vendors



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
                                     run_llm_judge=True      # Enable LLM-as-a-judge functional equivalence check (pre-Tier-1/2 stop-gap; disable for pure three-tier mode)
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
    self_correction_verify_counter = 0    # Global total Tier 2 failures across all generated codes; pipeline-level safety net
    self_correction_verify_counter_max = 15  # Global limit (increased from 10 to support multiple codes × per-code budget)
    self_correction_verify_counter_code = 0   # Per-code Tier 2 self-correction attempts; resets on fresh code generation
    self_correction_verify_counter_code_max = 5  # Per-code limit: if exhausted, mark INVALID and generate a fresh code
    tier2_exhausted = False   # Set True when per-code Tier 2 budget is exhausted; forces INVALID after judge block
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
    source_base, source_ext = os.path.splitext(source_file_path)
    verify_source_file_path = source_base + "_verify" + source_ext
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

    # Tier 2 source-as-specification reference: stdout from SOURCE_verify execution.
    # Per the source-as-specification premise, SOURCE_verify's observable behavior is the
    # sole correctness criterion. Any LASSI-generated transformation must preserve this behavior.
    passthrough_spec_output = source_verify_stdout
    verification_log["tier2_spec_output"] = passthrough_spec_output

    # Initialize per-code-block stdout list (indexed like code_blocks):
    #   [0] = SOURCE_orig stdout (no passthrough instrumentation)
    #   [1] = SOURCE_verify stdout (Tier 2 reference)
    passthrough_outputs = [source_orig_stdout, source_verify_stdout]

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
            # Increment code version for Stage 1
            code_version = code_version + 1
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

        elif compile_result != "" and compile_result != "energy_efficiency" and compile_result != "tier2_failure" and not zero_shot:
            """
            This condition is for handling CODE ERRORS during compilation or execution.
            """
            print("\n\n----- " + color.BOLD + color.BLUE + "HANDLING CODE ERRORS " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
           
            # CALL THE LLM WITH CODE ERROR CORRECTION - COMPILE TIME OR RUN TIME
            print("\n\n----- " + color.BOLD + color.RED + "SELF-CORRECTION: " + str(self_correction_counter) + color.END + " -----")
            if compile_result == "execute_error":
                self_correction_counter = self_correction_counter + 1
                iteration_log["v" + str(code_version)]["self_correction_count"] += 1
                error_correction_prompt_intro = "\n -- The above CODE TO FIX was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and produced the following execution error: "
                error_message = execute_result
            else:  # Must be a compiler error
                self_correction_counter = self_correction_counter + 1
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
            print("LLM Temperature: ", code_gen_temp)
                
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
            
            refactoring_plan_guide_iteration = refactoring_plan_guide_iteration + "\n\n" + refactoring_plan + "\n\n" + code_to_refactor  # "\n\n" + context_prompt +
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
            
            refactoring_plan_guide = refactoring_plan_guide + "\n\n" + refactoring_plan + "\n\n" + code_to_refactor  # "\n\n" + context_prompt +
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
            stage = ""
            compile_result = "energy_efficiency"
            change_strategy = True
            zero_shot = False  # Stop base LLM attempt after one try.
            force_save_baseline_llm = True  # We need to save Null data to maintain history tracking
        
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
                    stage = ""
                    compile_result = "energy_efficiency"
                    change_strategy = True
                    zero_shot = False  # Stop base LLM attempt after one try.
                    force_save_baseline_llm = True  # We need to save Null data to maintain history tracking
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
            # Reset flag because this should be a one-time action.
            force_save_baseline_llm = False
                    
            
        # Begin POWER METRICS feedback loop
        if yes_power_feedback and yes_execute and compile_result == "" and execute_result == "0" and self_correction_power_counter <= self_correction_power_counter_max:

            print("\n\n----- " + color.BOLD + color.BLUE + "POWER METRICS Feedback Loop " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " | code_blocks len: " + str(len(code_blocks)) + " (next idx: " + str(len(code_blocks)) + ") ***" + color.END)
            
            if not zero_shot:  # baseline_oneshot_goal:
                # This will initiate the subsequent iterations to introduce context and self-prompting
                self_correction_power_counter = self_correction_power_counter + 1
                iteration_log["v" + str(code_version)]["energy_refactor_count"] = self_correction_power_counter
            # else:
            # Do not enable feedback iteration in baseline mode
            #self_correction_power_counter = self_correction_power_counter_max + 1
            
            # --- TIER 2: Empirical Equivalence Check (fail-fast before judge) ---
            # Applies only to LASSI-generated codes (not zero_shot). Zero-shot codes refactor from
            # original_input_code (no passthrough blocks), so compare_passthrough_outputs() would
            # return SKIP — which is the correct behavior for the zero-shot path.
            # Reset per-code Tier 2 self-correction counter when processing a fresh code.
            # A Tier 2 retry enters here with compile_result_before_generation == "tier2_failure";
            # anything else (initial, win, INVALID/no-improvement) means a fresh code.
            if compile_result_before_generation != "tier2_failure":
                self_correction_verify_counter_code = 0
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
                iteration_log["v" + str(code_version)]["self_correction_count"] += 1

                if self_correction_verify_counter_code < self_correction_verify_counter_code_max:
                    # Still within per-code budget: pop code, self-correct with Tier 2 feedback.
                    # code_blocks is popped to keep it in sync with runtime_history (not updated on
                    # Tier 2 failure). Without this pop, tracking_best_version = len(code_blocks)-1
                    # after a later win would be out of bounds for runtime_history (IndexError).
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
                    tier2_exhausted = True
                    compile_result = "energy_efficiency"  # drive fresh code generation (not Tier 2 retry)
            # Tier 2 PASS or SKIP: record in verification log and proceed to judge
            if run_tier2 and not zero_shot and "verified_pass_count" in verification_log and tier2_verdict.startswith("PASS"):
                verification_log["verified_pass_count"] += 1

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

            # Handle scenario where we might have a bad power measurement
            # (only applies in EE mode; in runtime mode energy is always 0)
            if power_measure and total_energy_joules_refactor == 0:
                valid_code_flag = "INVALID"

            # --- THREE-TIER PRIORITY OVERRIDES ---
            # Three-tier verdict takes priority over the LLM judge for LASSI codes (not zero_shot).
            # Discrepancies are logged to verification_log["judge_discrepancies"] for research analysis:
            # they reveal cases where the LLM judge was wrong in either direction.

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

            # Override 2 (false negative): Tier 2 PASS → VALID even if judge said INVALID.
            # Empirical equivalence (Tier 2) is a stronger correctness signal than LLM opinion.
            # Only applies to LASSI codes (zero_shot has no Tier 2 instrumentation → SKIP, no override).
            if tier2_verdict == "PASS" and valid_code_flag != "VALID" and not zero_shot:
                verification_log["judge_discrepancies"].append({
                    "code_version": code_version, "zero_shot": zero_shot,
                    "type": "false_negative",
                    "judge_verdict": judge_raw_verdict, "tier2_verdict": tier2_verdict,
                    "final_verdict": "VALID",
                })
                print(f"\n----- Three-tier override: judge={judge_raw_verdict} overridden to VALID (Tier 2 PASS — judge false negative) -----")
                valid_code_flag = "VALID"

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
            runtime_history.append(round(float(duration_refactor),2))
            print("\nPower and Energy Feedback data: ")
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
                    experiment_data["base_v1_runtime_s"] = duration_refactor
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
        compare_code = compare_code + "\n\n" + "[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_code
        compare_code = compare_code + "\n\nOUTPUT: \n\n" + input_content_llmjudge_dict["source_stdout"]
        compare_code = compare_code + "\n\nCode execution duration: " + str(round(duration, 2)) + " s"
        compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(sourcecode_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(sourcecode_ave_power,2)) + " W"
        compare_code = compare_code + "\n\n[/END SOURCE CODE AND RESULTS]"
        
        compare_code = compare_code + "\n\n" + "[BEGIN REFACTORED CODE AND RESULTS]\n\n" + code_blocks[tracking_best_version]
        compare_code = compare_code + "\n\nOUTPUT: \n\n" + best_refactored_code_dict["refactored_stdout"]
        compare_code = compare_code + "\n\nCode execution duration: " + str(round(runtime_history[tracking_best_version], 2)) + " s"
        compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(energy_history[tracking_best_version],2)) + " J" + "\nAverage Power: " + str(round(power_history[tracking_best_version],2)) + " W"
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
    # less 2 on the data collection length
    # because the first two values are the source code and the base LLM result.
    ee_iterations = len(energy_history) - 2
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
        experiment_data["lassiee_valid"] = valid_code_history[best_index] if best_index > 0  else "N/A"
        experiment_data["lassiee_version"] = str(best_index)
        experiment_data["lassiee_llm_temp"] = iteration_log["v" + str(best_index)]["LLM_temperature"] if best_index > 0  else "N/A"
        experiment_data["lassiee_selfcorrections"] = iteration_log["v" + str(best_index)]["self_correction_count"] if best_index > 0  else "N/A"
        experiment_data["lassiee_eeiterations"] = iteration_log["v" + str(best_index)]["energy_refactor_count"] if best_index > 0  else "N/A"
        experiment_data["lassiee_energy_diff"] = round(((energy_history[best_index] - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2) if best_index > 0 and power_measure and sourcecode_total_energy_joules != 0 else "N/A"
        experiment_data["lassiee_avg_power_diff"] = round(((power_history[best_index] - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2) if best_index > 0 and power_measure and sourcecode_ave_power != 0 else "N/A"
        experiment_data["lassiee_runtime_diff"] = round(((runtime_history[best_index] - duration) / duration) * 100 ,2) if best_index > 0 and duration != 0 else "N/A"
        match_code_comparison_response = re.search(r"\[OPTIMIZATION LIST\](.*?)\[/OPTIMIZATION LIST\]", code_comparison_response, re.DOTALL)
        if match_code_comparison_response:
            optimization_list_text = match_code_comparison_response.group(1).strip()
        else:
            if tracking_best_version_label == "SOURCE":
                optimization_list_text = "SOURCE"
            else:
                optimization_list_text = "None"
        experiment_data["lassiee_optimization_methods"] = str(optimization_list_text)
        experiment_data["total_selfcorrections"] = self_correction_counter
        experiment_data["total_eeiterations"] = ee_iterations
        experiment_data["pipeline_runtime_min"] = round(end_time/60, 2)

    # --- LASSI-VERIFY SUMMARY: populate experiment_data fields from verification_log ---
    # Computed outside the yes_power_feedback block so they are always recorded.
    tier2_results = verification_log.get("tier2_results", [])
    discrepancies = verification_log.get("judge_discrepancies", [])
    experiment_data["tier2_pass_count"] = verification_log.get("verified_pass_count", 0)
    experiment_data["tier2_fail_total"] = sum(1 for r in tier2_results if r.get("verdict", "").startswith("FAIL"))
    experiment_data["judge_false_positives"] = sum(1 for d in discrepancies if d.get("type") == "false_positive")
    experiment_data["judge_false_negatives"] = sum(1 for d in discrepancies if d.get("type") == "false_negative")

    # --- LASSI-VERIFY JSON SIDECAR: save full verification_log for detailed inspection ---
    # tier2_spec_output (large stdout) and sourcecode_analysis (large text) are excluded here;
    # they are captured in the metadata .txt file. All other fields are saved verbatim.
    vlog_save = {k: v for k, v in verification_log.items() if k not in ("tier2_spec_output", "sourcecode_analysis")}
    vlog_filename = os.path.join(experiment_data_directory, "verification_log_" + file_name_save + "_Trial" + str(trial_number) + ".json")
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
        meta_data_file += "  Base LLM code: " + str(round(energy_history[1], 2))
        meta_data_file += "  Best Refactored code: " + str(round(energy_history[best_index], 2))
        meta_data_file += "\n"
        meta_data_file += "Average Power (W): "
        meta_data_file += "  Idle: " + str(round(idle_ave_power, 2))
        meta_data_file += "  Source code: " + str(round(sourcecode_ave_power, 2))
        meta_data_file += "  Base LLM code: " + str(round(power_history[1], 2))
        meta_data_file += "  Best Refactored code: " + str(round(power_history[best_index], 2))
        meta_data_file += "\n"
        meta_data_file += "Runtime (s): "
        meta_data_file += "  Idle: " + str(round(duration, 2))
        meta_data_file += "  Source code: " + str(round(duration, 2))
        meta_data_file += "  Base LLM code: " + str(round(runtime_history[1], 2))
        meta_data_file += "  Best Refactored code: " + str(round(runtime_history[best_index], 2))
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
        filename = os.path.join(experiment_data_directory, "metadata" + "_" + experiment.replace("/", "_") + "_Trial" + str(trial_number) + ".txt")
    else:
        # filename = directory + "/" + "metadata" + "_" + file_name_save + ".txt"
        filename = os.path.join(experiment_data_directory, "metadata" + "_" + file_name_save + "_Trial" + str(trial_number) + ".txt")
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
    
    return True