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
                                     test_code_first,
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
                                     app_name
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
    exe_output_source = ""
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
        
    print("\n----- WELCOME TO PIPELINE v3! CODE -----\n")

    vendors = detect_gpu_vendors()
    if not vendors:
        print("❌ PIPELINE ERROR: No GPUs detected. Stopping LASSI.")
        # Stop pipeline here
    else:
        detected_vendor = vendors[0]['vendor']
        print(f"🔍 Detected GPU vendor: {detected_vendor.upper()}")

    report_gpu_memory()
        
    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END) 
    
    # --------------------------------
    # ----- STAGE 0: TEST SOURCE -----
    # --------------------------------
    # Test compile and execute the source code to validate compiler and execution environment.
    # Collect baseline energy measurements.
    if test_code_first:
        """Initial stage to run the SOURCE CODE and gather metrics."""
        (idle_energy_joules, idle_ave_power, input_content_llmjudge_dict, sourcecode_total_energy_joules, sourcecode_ave_power, duration,
         energy_history, power_history, runtime_history, valid_code_history) = lassi.pipeline_methods.test_source_code(app_name, experiment_data_directory, code_version, trial_number, source_file_path, build_script, make_script, directory, source_lang, target_lang,
                                                                                                exe_input_parameters, code_compiler, code_compiler_kwds,
                                                                                                yes_execute, power_measure, input_content_llmjudge_dict,
                                                                                                energy_history, power_history, runtime_history, valid_code_history,
                                                                                                second_test=False
                                                                                               )
        # Save the output details for this as the "best" code version for metadata record
        # This is necessary to capture now if no better refactored code is generated:
        best_refactored_code_dict = input_content_llmjudge_dict

        experiment_data["source_v0_energy_J"] = sourcecode_total_energy_joules
        experiment_data["source_v0_avg_power_W"] = sourcecode_ave_power
        experiment_data["source_v0_runtime_s"] = duration

        print(print("\n----- END OF STAGE 0: TEST SOURCE -----\n"))
        print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)


    
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

    # Pull in prompts used for base LLM
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
        verifyprompt=verifyprompt
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
    
    
    # -----------------------------------------------------------
    # ----- STAGE 1: INSERT VERIFY INSTRUMENTATION -----
    # -----------------------------------------------------------
    # With source code that can compile and execute on the given system and device,
    #  we next prompt LLM inference to insert Invariant Assertians into the source code
    #  with protective commenting to ensure subsequent LLM-driven code transformations do not
    #  change the assertians. (Next TODO is to insert passthrough data dump blocks.)
    # Let's overwright the source code file in input_content_llmjudge_dict["source_code"]
    #  -- if we need the original source later, then we can create a new dictionary key
    
    input_code = lassi.verify.code_insert_asserts(
        input_content_llmjudge_dict["source_code"],
        llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=True
    )

    input_content_llmjudge_dict["source_code"] = input_code

    if test_code_first:
        """Run the source code test AGAIN after instrumenting the invariant assertions."""
        compile_result, execute_result = lassi.pipeline_methods.test_source_code(app_name, experiment_data_directory,
                                                                                                                        code_version, trial_number, source_file_path, build_script,
                                                                                                                        make_script, directory, source_lang, target_lang,
                                                                                                                        exe_input_parameters, code_compiler, code_compiler_kwds,
                                                                                                                        yes_execute, power_measure, input_content_llmjudge_dict,
                                                                                                                        energy_history, power_history, runtime_history, valid_code_history,
                                                                                                                        second_test=True
                                                                                                                    )

    # Add source code instrumented with verification to the first index of code_blocks
    code_blocks.append(input_code)
    
    # --------------------------------------------------------
    # INITIATE LASSI-EE LLM CODE GENERATION ITERATIVE PIPELINE
    while (
        (compile_result != "" or stage == "stage1" or stage == "stage2")
        and self_correction_counter <= self_correction_max
        and self_correction_power_counter <= self_correction_power_counter_max
    ):
        # After the initial zero-shot and iteration attempts until no improvement, then
        # Update the prompt strategy with context and self-prompting responses.
        if change_strategy and not zero_shot and compile_result == "energy_efficiency":             
            print("\n\n" + color.BOLD + "CHANGE PROMPT STRATEGY! -----" + color.END)
            print("\n\n----- " + color.BOLD + color.BLUE + "STARTING LASSI: ITERATIVE SELF-PROMPTING WITH CONTEXT AND SELF-CORRECTION" + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            generate_context_flag = True
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
                total_energy_joules=energy_history[tracking_best_version],
                ave_power=power_history[tracking_best_version],
                verifyprompt=verifyprompt
            )
            change_strategy = False

        # Initial zero-shot refactor attempt    
        if zero_shot and compile_result == "" and stage == "stage1":
            # Increment code version for Stage 1
            code_version = code_version + 1
            print("\n\n----- " + color.BOLD + color.BLUE + "STARTING STAGE 1: BASE LLM REFACTORING WITHOUT SELF-CORRECTION " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            # BUILD REFACTING INSTRUCTION PROMPT
            instructions_prompt = refactor_instructions
            # Append the source code to the prompt
            instructions_prompt = instructions_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]"                
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
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
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
            # Append the source code to the prompt
            instructions_prompt = instructions_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]"
            # instructions_prompt = instructions_prompt + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(sourcecode_total_energy_joules, 2)) + " Joules at an average power of " + str(round(sourcecode_ave_power, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                 
            # Save the original context + energy efficiency context for re-use during energy efficiency correction
            original_ee_context_content_prompt = instructions_prompt_core_iterations
            original_ee_prompt_tokens = lassi.token_analysis.token_count(original_ee_context_content_prompt)

            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE ZERO-SHOT INSTRUCTIONS TASK PROMPT " + color.END + " (prompt token count, excluding code): " + str(original_ee_prompt_tokens) + "\n")  # + instructions_prompt + "\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "initial zero-shot attempt: prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, instructions_prompt, display_response, verifyprompt)
        
        elif compile_result != "" and compile_result != "energy_efficiency" and not zero_shot:
            """
            This condition is for handling CODE ERRORS during compilation or execution.
            """
            print("\n\n----- " + color.BOLD + color.BLUE + "HANDLING CODE ERRORS " + color.END + " -----\n")
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
           
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
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
           
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
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
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
                # Generate a new refactoring plan following a new "best" version of the refactored code
                print(f"  Using refactored code version to write refactoring plan: {tracking_best_version}")  # The -1 because we are not needing the index number here.
                refactoring_plan, code_to_refactor = lassi.pipeline_methods.generate_refactoring_plan(
                    source_lang=source_lang,
                    input_code=code_blocks[tracking_best_version],
                    total_energy_joules=energy_history[tracking_best_version],
                    ave_power=power_history[tracking_best_version],
                    llm_source=llm_source,
                    model=model,
                    num_ctx=num_ctx,
                    system_prompt=system_prompt,
                    context_prompt=context_prompt,
                    prior_refactoring_plans=bad_refactor_plans,
                    display_response=display_response,
                    few_shot_learning=few_shot_learning,
                    verifyprompt=verifyprompt
                )
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
                code_to_refactor = code_to_refactor + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(energy_history[tracking_best_version], 2)) + " Joules at an average power of " + str(round(power_history[tracking_best_version], 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
                print(f"... Prior code index selected to refactor: {tracking_best_version}")
                
            # Context:
            # if only the first pass, then refactoring_plan_original will be for the source code,
            # otherwise, it will include the latest refactoring plan for the best code
            refactoring_plan_guide_iteration = prompts.get_code_ee_refactoring_prompt(source_lang + "_refactor_iteration")
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
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            """
            This condition only runs after zero_shot first attempt is completed.
            Self-correction feedback for improving energy efficiency following prior improvement.
            The condition will generate a new refactoring plan for the latest best code.
            """
            print("\n\n----- " + color.BOLD + color.PURPLE + "SELF-IMPROVEMENT FOR ENERGY EFFICIENCY: " + str(energy_efficiency_iteration) + color.END + " -----\n")
            print(f"  Using *just improved* refactored code version: {tracking_best_version}")  # The -1 because we are not needing the index number here.
            
            #report_gpu_memory(llm_source)

            # Generate a new refactoring plan following a new "best" version of the refactored code
            refactoring_plan, code_to_refactor = lassi.pipeline_methods.generate_refactoring_plan(
                source_lang=source_lang,
                input_code=code_blocks[tracking_best_version],
                total_energy_joules=energy_history[tracking_best_version],
                ave_power=power_history[tracking_best_version],
                llm_source=llm_source,
                model=model,
                num_ctx=num_ctx,
                system_prompt=system_prompt,
                context_prompt=context_prompt,
                prior_refactoring_plans=bad_refactor_plans,
                display_response=display_response,
                few_shot_learning=False,
                verifyprompt=verifyprompt
            )

            refactoring_plan_guide = prompts.get_code_ee_refactoring_prompt(source_lang + "_refactor")
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
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        # -------------------------------------
        # ----- GENERATED CODE PROCESSING -----
        if code_block.startswith('c++') or code_block.startswith('cpp'):
            code_block = code_block[3:]
        if code_block.startswith('c'):
            code_block = code_block[1:]    
    
        print("\n\n----- " + color.BOLD + color.BLUE + "POST-PROCESSING of GENERATED CODE" + color.END + " -----\n")
        print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
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
            print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            if not zero_shot:  # baseline_oneshot_goal:
                # This will initiate the subsequent iterations to introduce context and self-prompting
                self_correction_power_counter = self_correction_power_counter + 1
                iteration_log["v" + str(code_version)]["energy_refactor_count"] = self_correction_power_counter
            # else:
            # Do not enable feedback iteration in baseline mode
            #self_correction_power_counter = self_correction_power_counter_max + 1
            
            # Call LLM-as-a-judge to review SOURCE CODE and REFACTORED CODE to evaluate if likely valid
            valid_code_flag, judge_response_text, code_analysis = lassi.llm_methods.llm_judge(target_lang, llm_source, model_judge, num_ctx, system_prompt, input_content_llmjudge_dict, power_measure, run_code_analysis, verification_log["sourcecode_analysis"], display_response, verifyprompt)
            
            # Handle scenario where we might have a bad power measurement
            if total_energy_joules_refactor == 0:
                valid_code_flag = "INVALID"
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
                    experiment_data["base_v1_energy_diff"] = round(((total_energy_joules_refactor - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2)
                    experiment_data["base_v1_avg_power_diff"] = round(((ave_power_refactor - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2)
                    experiment_data["base_v1_runtime_diff"] = round(((duration_refactor - duration) / duration) * 100 ,2)
                    # experiment_data["base_v1_optimization_methods"] = "None"
                elif stage == "stage2":
                    experiment_data["base_v2_energy_J"] = total_energy_joules_refactor
                    experiment_data["base_v2_avg_power_W"] = ave_power_refactor
                    experiment_data["base_v2_runtime_s"] = duration_refactor
                    experiment_data["base_v2_valid"] = valid_code_flag
                    experiment_data["base_v2_selfcorrections"] = iteration_log["v2"]["self_correction_count"]
                    experiment_data["base_v2_energy_diff"] = round(((total_energy_joules_refactor - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2)
                    experiment_data["base_v2_avg_power_diff"] = round(((ave_power_refactor - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2)
                    experiment_data["base_v3_runtime_diff"] = round(((duration_refactor - duration) / duration) * 100 ,2)
                    experiment_data["base_v2_optimization_methods"] = "None"
            if (
                total_energy_joules_refactor < energy_history[tracking_best_version]
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
                print("\n\n***** Generated refactored code is more energy efficient than source or previous code! *****")
                print("... based on refactoring from code at index: " + str(tracking_best_version) + "\n\n")
                energy_efficiency_iteration += 1
                energy_efficiency_refactor_success = True
                # Save the output details for this "best" code version for metadata record:
                best_refactored_code_dict = input_content_llmjudge_dict
                # Reset the LLM temperature parameter for subsquent code generation
                code_gen_temp = 0.2
                # I THINK WE HAVE THE FOLLOWING NOW WITH THE BETTER VERSION TRACKING:
                # TODO: self-review for multiple attempts to further improve energy efficiency
                # We need to set a flag that we had success at least one time.
                # Then, if follow-up is not a success, then STOP and return the latest success.
                # We need to track the refactor VERSION, so may need to have a list
                # May want a list for each code vesion as well. Whew. We are going to be tracking a lot here.
                if len(energy_history) > 1 and len(power_history) > 1:
                    if len(code_blocks) > 0:
                        # If we have saved a "best" code refactor,
                        # then compare the latest to the best
                        previous_energy = energy_history[tracking_best_version]
                        current_energy = energy_history[-1]
                        previous_power = power_history[tracking_best_version]
                        current_power = power_history[-1]
                    else:
                        # If we haven't saved a "best" code refactor,
                        # then just compare to the source code
                        previous_energy = energy_history[0]
                        current_energy = energy_history[-1]
                        previous_power = power_history[0]
                        current_power = power_history[-1]
                    
                    # Track the "best" code generation
                    # If current_energy is < previous_energy, 
                    # then update tracking version to this current code generation.
                    # Otherwise, do not change the tracking version (keep it the same)
                    if current_energy < previous_energy:
                        tracking_best_version = code_version  # IGNORE this comment: code_version has already been incremented, so we need to step one back
                    
                    print("\n\n----- " + color.BOLD + color.BLUE + "COMPARING CURRENT vs PREV ENERGY VALUES " + color.END + " -----\n")
                    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
                    
                    # Calculate the fractional improvement
                    if previous_energy != 0:
                        energy_improvement = (previous_energy - current_energy) / previous_energy
                    else:
                        energy_improvement = current_energy
                    if previous_power != 0:
                        power_improvement = (previous_power - current_power) / previous_power
                    else:
                        power_improvement = current_power

                    # Check if an improvement is < 1% AND that measurement is still better than the source code
                    energy_stagnant = energy_improvement < 0.011 and current_energy < energy_history[0]
                    power_stagnant = power_improvement < 0.011 and current_power < power_history[0]

                    # Stopping condition:
                    # Stop if at least one metric is stagnant, while the other is not improving by more than 1%
                    # Do NOT stop if both are worsening (negative improvement values)    
                    # Adjustment (03.20.2025): only allow stagnation stop on energy consuption, increase min ee iteration
                    if (
                        energy_efficiency_iteration > 4 
                        and
                        (energy_stagnant)  # or power_stagnant)
                        and
                        (energy_improvement > 0 or power_improvement > 0)
                    ):
                        print("\n\n-----> LASSI for Energy Efficiency IS FINISHED! <-----\n")
                        print(" energy_stagnant = " + str(energy_stagnant))
                        print(" power_stagnant = " + str(power_stagnant))
                        print(" energy_improvement = " + str(round(energy_improvement, 2)))
                        print(" power_improvement = " + str(round(power_improvement, 2)))
                        print("\n\n")
                        break
            else:
                # TODO: Change label: invalid or not more energy efficient
                print("\n\n----- " + color.BOLD + color.BLUE + "WE HAVE GENERATED CODE THAT IS INVALID OR NOT MORE ENERGY EFFICIENT" + color.END + " -----\n")
                print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
                
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
                    print("\n\n***** Generated refactored code is NOT more energy efficient than source or previous code... *****")
                    if not zero_shot:
                        print("\n" + color.GREEN + color.BOLD + " ----- ENERGY EFFICIENCY REVIEW: Let's provide feedback on the energy consumption. -----" + color.END + "\n\n")
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

    print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
    
    end_time_minutes = round(end_time/60, 2)
    print("CODE GENERATION TIME: " + str(end_time_minutes) + " minutes.")
    if valid_code_flag != "VALID":
        print(f"  Stopped LASSI early due to invalid code on the last generation version number = {code_version - 1}.")
    print(" Total Code Error Self-correction iteration count: " + str(self_correction_counter))
    # if not baseline_oneshot_goal:
    #     print(" Self-improvement ee refactor count: " + str(self_correction_power_counter))
    print(" Total Energy Efficiency Self-improvement iteration count: " + str(len(energy_history) - 2))
    print(" Energy efficiency improvements: " + str(energy_efficiency_iteration))
    if tracking_best_version == 0:
        tracking_best_version_label = "SOURCE"
    elif tracking_best_version == 1:
        tracking_best_version_label = "STANDALONE BASE LLM"
    elif tracking_best_version == 2:
        tracking_best_version_label = "SELF-CORRECTED BASE LLM"
    else:
        tracking_best_version_label = "v" + str(tracking_best_version)
    print(" Best code, version: " + tracking_best_version_label + "\n" )

    # If no code refactor v1+ resulted in lower energy consumption,
    # then look to see if any one of the LASSI v3+ had lower ave power.
    if tracking_best_version_label == "SOURCE":
        source_ave_power_value = power_history[0]
        # Filter pairs (index, value) where:
        # - value is less than source_ave_power_value
        # - index is not 1
        # - the corresponding valid_code_history value is "VALID"
        filtered = [
            (i, v)
            for i, v in enumerate(power_history)
            if v < source_ave_power_value and i != 1 and valid_code_history[i] == "VALID"
        ]
        if filtered:
            smallest_index, smallest_ave_power_value = min(filtered, key=lambda x: x[1])
        else:
            smallest_index, smallest_ave_power_value = None, None
        
        if smallest_ave_power_value is not None:
            print(" Best ave power in refactored code, version: " + str(smallest_index-1) + "\n" )
    elif "BASE LLM" in tracking_best_version_label:
        source_ave_power_value = min(power_history[1], power_history[2])
        # Filter pairs (index, value) where:
        # - value is less than source_ave_power_value
        # - index is not 0
        # - the corresponding valid_code_history value is "VALID"
        filtered = [
            (i, v)
            for i, v in enumerate(power_history)
            if v < source_ave_power_value and i != 0 and valid_code_history[i] == "VALID"
        ]
        if filtered:
            smallest_index, smallest_ave_power_value = min(filtered, key=lambda x: x[1])
        else:
            smallest_index, smallest_ave_power_value = None, None
        
        if smallest_ave_power_value is not None:
            print(" Best ave power over Base LLM in refactored code, version: " + str(smallest_index-1) + "\n" )
        
        
    # Prompt the LLM to compare the codes between SOURCE and the "BEST" refactored version
    if (tracking_best_version != 0):

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
        print(color.BOLD + color.DARKORANGE + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        # Save "best" index
        if (tracking_best_version_label == "SOURCE" or tracking_best_version_label == "BASE LLM") and smallest_index is not None:
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
        experiment_data["lassiee_energy_diff"] = round(((energy_history[best_index] - sourcecode_total_energy_joules) / sourcecode_total_energy_joules) * 100 ,2) if best_index > 0  else "N/A"
        experiment_data["lassiee_avg_power_diff"] = round(((power_history[best_index] - sourcecode_ave_power) / sourcecode_ave_power) * 100 ,2) if best_index > 0  else "N/A"
        experiment_data["lassiee_runtime_diff"] = round(((runtime_history[best_index] - duration) / duration) * 100 ,2) if best_index > 0  else "N/A"
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