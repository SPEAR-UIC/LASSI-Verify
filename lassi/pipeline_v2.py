"""
LASSI - EE pipeline v2
Here, we first embrase the capability of the base LLM
by running the initial refactor as a zero-shot, no context attempt.
Then, subsequent iterations, we bring in self-prompting and background context
to see if we can improve the energy efficiency outcome further.
"""

import os
import re
import time
import gc
from datetime import datetime
import lassi.code_manager
import lassi.llm_methods
import lassi.token_analysis
import lassi.llm_config
import lassi.get_power
import lassi.pipeline_methods
from lassi.get_power import report_gpu_memory, query_gpu_state
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary




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
                                     input_context,
                                     input_code_file_subdirectory,
                                     input_code,  # THE SOURCE CODE
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
                                     verifyprompt
                                    ):
    
    # Test compile and execute the source code to validate compiler and execution environment
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
    tracking_best_version = 0  # to account for the source code at index 0
    get_system_info_flag = True  # Used in the selfprompting context builder
    valid_code_flag = "INVALID"  # default value, or "VALID"
    change_strategy = False
    code_gen_temp = 0.2  # default LLM temperature
    code_gen_temp_max = 1.0
    energy_history = []
    power_history = []
    runtime_history = []
    valid_code_history = []
    code_blocks = []  # Collection of all codes generated
    best_refactored_code_dict = {}
    input_content_llmjudge_dict = {"source_code": "",
                                  "source_stdout": "",
                                  "source_metrics": "",
                                  "refactored_code": "",
                                  "refactored_stdout": "",
                                  "refactored_metrics": "",
                                  }  # Input values to be used by the LLM-as-a-judge phase

    # Add source code to the first index of code_blocks
    code_blocks.append(input_code)
    input_content_llmjudge_dict["source_code"] = input_code
    
    report_gpu_memory()
    
    print("\n----- WELCOME TO PIPELINE v2! CODE -----\n")

    # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END) 
    
    """ STAGE 0: TEST SOURCE CODE """

    if test_code_first:
        """Initial stage to run the SOURCE CODE and gather metrics."""
        (idle_energy_joules, idle_ave_power, input_content_llmjudge_dict, sourcecode_total_energy_joules, sourcecode_ave_power, duration,
         energy_history, power_history, runtime_history, valid_code_history) = lassi.pipeline_methods.test_source_code(source_file_path, build_script, make_script, directory, source_lang, target_lang,
                                                                                                exe_input_parameters, code_compiler, code_compiler_kwds,
                                                                                                yes_execute, power_measure, input_content_llmjudge_dict,
                                                                                                energy_history, power_history, runtime_history, valid_code_history
                                                                                               )
        # Save the output details for this as the "best" code version for metadata record
        # This is necessary for if no better refactored code is generated:
        best_refactored_code_dict = input_content_llmjudge_dict
        
    # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
    
    """
    STAGE 1: ZERO-SHOT REFACTOR - BASE LLM CAPABILITY
    + Implement a straightforward and strong user prompt only. 
    """
    
    compile_result = "start"
    zero_shot = True
    generate_context_flag = False
    content = ""
    content_summary_only = ""
    execute_result = ""
    error_message = ""
    error_source = ""  # "compiled" or "executed"
    code_block = ""
    self_correction_counter = 0
    self_correction_power_counter = 0
   
    print("\n\n----- " + color.BOLD + color.BLUE + "STARTING FIRST STAGE: ZERO-SHOT BASE LLM REFACTORING" + color.END + " -----\n")
    
    # Pull in prompts used for base LLM
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
    
    # report_gpu_memory()
    # Unload model first if we want a different coding model
    if (model_coder != model and llm_source != "argo"):
        # CHANGE MODEL FOR CODE GENERATION
        print(" (... changing LLM for code gen)")
        lassi.llm_config.unload_ollama_model(model)
        time.sleep(10)  # Pause for 10 seconds
        report_gpu_memory()
    
    # Clean up Python-level objects
    gc.collect()
    time.sleep(5)
    
    start_time = time.time()
    
    # INITIATE LLM CODE GENERATION ITERATIVE PIPELINE
    while (
        compile_result != ""
        and self_correction_counter <= self_correction_max
        and self_correction_power_counter <= self_correction_power_counter_max
    ):
        # After the initial zero-shot and iteration attempts until no improvement, then
        # Update the prompt strategy with context and self-prompting responses.
        if change_strategy and not zero_shot and compile_result == "energy_efficiency":
                       
            print("\n\nCHANGE PROMPT STRATEGY! -----")
            print("\n\n----- " + color.BOLD + color.BLUE + "STARTING LASSI: ITERATIVE SELF-PROMPTING WITH CONTEXT AND SELF-CORRECTION" + color.END + " -----\n")
            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
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
        if compile_result == "start":
            
            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            # CALL THE LLM to generate the first code refactoring using the source code
            # Prompting was built before WHILE loop initiates
            if llm_source != "argo":
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

            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE ZERO-SHOT INSTRUCTIONS TASK PROMPT " + color.END + " (prompt without code, token count: " + str(original_ee_prompt_tokens) + "): \n")  # + instructions_prompt + "\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "initial zero-shot attempt: prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, instructions_prompt, display_response, verifyprompt)
        
        elif compile_result != "energy_efficiency":
            
            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            """
            This condition is for handing CODE ERRORS during compilation or runtime.
            """
            # CALL THE LLM WITH CODE ERROR CORRECTION - COMPILE TIME OR RUN TIME
            print("\n\n----- " + color.BOLD + color.RED + "SELF-CORRECTION: " + str(self_correction_counter) + color.END + " -----")
            if compile_result == "execute_error":
                self_correction_counter = self_correction_counter + 1
                error_correction_prompt_intro = "\n -- The above CODE TO FIX was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and produced the following execution error: "
                error_message = execute_result
            else:  # Must be a compiler error
                self_correction_counter = self_correction_counter + 1
                # NOTE: Maybe do not provide the original code in the other language when trying to fix code in the target language.
                error_correction_prompt_intro = "\n\n -- The above CODE TO FIX was compiled with " + code_compiler + " " + code_compiler_kwds + " and produced the following compile error: "
                error_message = compile_result
            
            # error_correction_prompt_outro = " Eliminate this specific error by re-factoring the above code with an appropriate fix while also using the following background context: " + original_context_content_prompt
            # error_prompt = code_block + error_correction_prompt_intro + error_message + error_correction_prompt_outro
            # error_correction_prompt_outro = " Eliminate this specific error by re-factoring the above code with an appropriate fix."
            error_correction_prompt = prompts.get_code_error_correction_prompt("CUDA_error")
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

            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            """
            This condition is available only if we want to REPEAT an iteration for the zero_shot attempt.
            Re-attempt until energy efficiency improvement is not acheived (then change prompting strategy).
            """
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
            print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE ZERO-SHOT INSTRUCTIONS TASK PROMPT " + color.END + " (prompt without code, token count: " + str(original_ee_prompt_tokens) + "): \n")  # + instructions_prompt + "\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "iteratate base LLM zero-shot attempt: prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, instructions_prompt, display_response, verifyprompt)

        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == False and not zero_shot:  #baseline_oneshot_goal:
            
            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
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
                
            # report_gpu_memory()

            # BUILD REFACTORING PLAN
            # On the first try following the base LLM zero shot attempts,
            # otherwise, new refactoring plans will only be created when
            # a new "best" code version is generated by the pipeline.
            if self_correction_power_counter == 0:
                # Generate a new refactoring plan following a new "best" version of the refactored code
                print(f"  Using refactored code version: {tracking_best_version}")  # The -1 because we are not needing the index number here.
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
                    display_response=display_response,
                    few_shot_learning=False
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
            refactoring_plan_guide_iteration = prompts.get_code_ee_refactoring_prompt("CUDA_refactor_iteration")
            # Append few-shot learning examples
            if few_shot_learning:
                refactoring_plan_guide_iteration = refactoring_plan_guide_iteration + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
            
            refactoring_plan_guide_iteration = refactoring_plan_guide_iteration + "\n\n" + refactoring_plan + "\n\n" + code_to_refactor  # "\n\n" + context_prompt +
            refactoring_plan_iteration_tokens = lassi.token_analysis.token_count(refactoring_plan_guide_iteration)

            print("\n\n" + color.BOLD + color.GREEN + "----- REFACTORING PLAN WITH INSTRUCTIONS PROMPT " + color.END + "(energy efficiency *iteration* with code, token count: " + str(refactoring_plan_iteration_tokens) + "): ")  # + refactoring_plan_iteration + "\n\n")
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code for energy efficiency -- improvement iteration." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, refactoring_plan_guide_iteration, display_response, verifyprompt)        

        
        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == True and not zero_shot:  #baseline_oneshot_goal:

            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
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
                display_response=display_response,
                few_shot_learning=False,
                verifyprompt=verifyprompt
            )

            refactoring_plan_guide = prompts.get_code_ee_refactoring_prompt("CUDA_refactor")
            # Append few-shot learning examples
            if few_shot_learning:
                refactoring_plan_guide = refactoring_plan_guide + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
            
            refactoring_plan_guide = refactoring_plan_guide + "\n\n" + refactoring_plan + "\n\n" + code_to_refactor  # "\n\n" + context_prompt +
            refactoring_plan_tokens = lassi.token_analysis.token_count(refactoring_plan_guide)
           
            print("\n\n" + color.BOLD + color.GREEN + "----- REFACTORING PLAN WITH INSTRUCTIONS PROMPT " + color.END + "(energy efficiency improvement with code, token count: " + str(refactoring_plan_tokens) + "): ")  # \n\n" + refactoring_plan + "\n\n")
            # refactoring_plan_iteration = re.sub('\n', '', refactoring_plan)
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code for energy efficiency -- new refactoring plan." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, system_prompt, refactoring_plan_guide, display_response, verifyprompt)
        else:
            print("\n\n ????? NO CONDITION MET DURING ITERATION ?????")
            print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        if code_block.startswith('c++') or code_block.startswith('cpp'):
            code_block = code_block[3:]
        if code_block.startswith('c'):
            code_block = code_block[1:]    
    
        # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        # SAVE THE GENERATIVE CODE -- before compiling
        # If make script, then save edited code as same filename and extension
        if make_script:
            # We need to overwrite the same code file,
            # instead of creating a new file.
            print("SAVING THE CODE - WHERE ARE WE: ", os.getcwd())
            file_name_save = input_code_file_subdirectory + file_name
            file_path = os.getcwd() + "/" + file_name_save + "." + refactored_code_extension
        else:
            file_name_save = file_name + "_" + dt_string
            file_path = directory + "/" + file_name_save + "." + refactored_code_extension
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
            compile_result = lassi.code_manager.make_script_code(directory, code_compiler_kwds)
            # print("compile_result: ", compile_result)
            execute_file = "bin/autodock_gpu_64wi"  # HARD CODED FOR MINIMDOCK
            exe_input_parameters = "-lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 100".split()
        else:
            compile_result, execute_file = lassi.code_manager.compile_code(file_path, target_lang, code_compiler, code_compiler_kwds)

        print("\nExecute file: " + str(execute_file))

        if yes_execute and compile_result == "" and self_correction_counter <= self_correction_max:
            code_source = "generated"  # target
            # Unload any previously loaded LLMs in the Ollama server 
            # (clear out the memory)
            if llm_source == "ollama":
                lassi.llm_config.unload_ollama_model(model_coder)
                time.sleep(10)
                report_gpu_memory()
                
            exe_file_path = execute_file  # directory + "/" + execute_file
            execute_result, exe_output_generated, total_energy_joules_refactor, ave_power_refactor, std_power_std, duration_refactor, metrics_results_string = lassi.code_manager.execute_code(directory, exe_file_path, exe_input_parameters, power_measure, code_source, target_lang, build_script, make_script, wait=5)
            if execute_result != "0":
                compile_result = "execute_error"  # just a flag
            else:
                # We have a keeper! Save the generated code as a separate versioned copy
                code_version = code_version + 1  # Increment code version to get ready for the next iteration.
                print("--> SAVING code version: " + str(code_version))
                if not zero_shot:
                    this_code_version = code_version
                else:
                    this_code_version = str(code_version) + "-baseLLM"
                file_path = directory + "/" + file_name_save + "_v" + str(this_code_version) + "." + refactored_code_extension
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as file:
                    file.write(code_block)
                file.close()
                code_blocks.append(code_block)  # save this code in our code collection list
                input_content_llmjudge_dict["refactored_code"] = code_block
                input_content_llmjudge_dict["refactored_stdout"] = exe_output_generated
                input_content_llmjudge_dict["refactored_metrics"] = metrics_results_string
                

        # Begin POWER METRICS feedback loop
        if yes_power_feedback and yes_execute and compile_result == "" and execute_result == "0" and self_correction_power_counter <= self_correction_power_counter_max:

            # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
            
            if not zero_shot:  # baseline_oneshot_goal:
                # This will initiate the subsequent iterations to introduce context and self-prompting
                self_correction_power_counter = self_correction_power_counter + 1
            # else:
            # Do not enable feedback iteration in baseline mode
            #self_correction_power_counter = self_correction_power_counter_max + 1
            
            # Call LLM-as-a-judge to review SOURCE CODE and REFACTORED CODE to evaluate if likely valid
            valid_code_flag = lassi.llm_methods.llm_judge(llm_source, model_judge, num_ctx, system_prompt, input_content_llmjudge_dict, power_measure, display_response, verifyprompt)
            
            # valid_code_flag can be: (1) "VALID", (2) "INVALID", or (3) "NONE"
            if not zero_shot:
                valid_code_history.append(valid_code_flag)
            else:
                valid_code_history.append("basellm_" + valid_code_flag)
            
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
                                    
            
            if (
                total_energy_joules_refactor < energy_history[tracking_best_version]
                and valid_code_flag == "VALID"
            ):  
                # sourcecode_total_energy_joules:  # or ave_power_refactor < sourcecode_ave_power:
                if zero_shot:
                    change_strategy = True
                    zero_shot = False  # Stop base LLM attempt after one try.
                """ Let's only care about the total energy consumption and not average power. """
                print("\n\n***** Generated refactored code is more energy efficient than source or previous code! *****")
                print("... based on refactoring from code at index: " + str(tracking_best_version) + "\n\n")
                energy_efficiency_iteration += 1
                compile_result = "energy_efficiency"
                energy_efficiency_refactor_success = True
                # Save the output details for this "best" code version for metadata record:
                best_refactored_code_dict = input_content_llmjudge_dict
                # Reset the LLM temperature parameter for subsquent code generation
                code_gen_temp = 0.2
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
                    
                    # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
                    
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
                
                # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
                
                compile_result = "energy_efficiency"
                if zero_shot:
                    change_strategy = True
                    zero_shot = False  # Stop base LLM attempt if improvement not acheived after iteration.
                energy_efficiency_refactor_success = False
                if (valid_code_flag == "VALID"):
                    print("\n\n***** Generated refactored code is NOT more energy efficient than source or previous code... *****")
                    print("\n" + color.GREEN + color.BOLD + " ----- ENERGY EFFICIENCY REVIEW: Let's provide feedback on the energy consumption. -----" + color.END + "\n\n")
                elif valid_code_flag != "VALID":
                    print("\n\n" + color.RED + color.BOLD + " ----- GENERATED REFACTORED CODE: May not be valid or not determined. ----- " + color.END + "\n\n")
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
                
    # end of while pipeline loop

    end_time = time.time() - start_time
    
    print("\n\n----- " + color.BOLD + color.GREEN + "CODE GENERATION COMPLETE" + color.END + " -----\n")

    # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
    
    end_time_minutes = round(end_time/60, 2)
    print("CODE GENERATION TIME: " + str(end_time_minutes) + " minutes.")
    if valid_code_flag != "VALID":
        print(f"  Stopped LASSI early due to invalid code on the last generation version number = {code_version - 1}.")
    print(" Total Code Error Self-correction iteration count: " + str(self_correction_counter))
    # if not baseline_oneshot_goal:
    #     print(" Self-improvement ee refactor count: " + str(self_correction_power_counter))
    print(" Total Energy Efficiency Self-improvement iteration count: " + str(len(energy_history) - 2))
    print(" Energy efficiency improvements: " + str(energy_efficiency_iteration))
    if tracking_best_version < 0:
        tracking_best_version_label = "SOURCE"
    elif tracking_best_version == 0:
        tracking_best_version_label = "BASE LLM"
    else:
        tracking_best_version_label = str(tracking_best_version)
    print(" Best code, version: " + tracking_best_version_label + "\n" )

    # If no code refactor v0 or v1+ resulted in lower energy consumption,
    # then look to see if any one of the LASSI v1+ had lower ave power.
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
    elif tracking_best_version_label == "BASE LLM":
        source_ave_power_value = power_history[1]
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
    if (energy_efficiency_refactor_success or baseline_oneshot_goal):

        print("\n\n----- " + color.BOLD + color.BLUE + "FINAL CODE COMPARISON: Source Code vs. Best Refactored Code" + color.END + " -----\n")

        # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        compare_code = prompts.get_code_comparison("cuda_cpp")
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
    meta_data_file += "Experiment date-time: " + str(dt_string)
    meta_data_file += "\n"
    meta_data_file += "Time (sec): " + str(round(end_time, 2)) + " | " + str(round(end_time/60, 2)) + " (min)"
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
    meta_data_file += "Energy Efficiency Self-improvement iteration count: " + str(len(energy_history) - 2)  # str(self_correction_power_counter)
    
    if yes_power_feedback:
        
        # print(color.BOLD + color.RED + "\n*** DEBUG: code_version: " + str(code_version) + " | tracking_best_version: " + str(tracking_best_version) + " ***"  + color.END)
        
        # Save "best" index
        if (tracking_best_version_label == "SOURCE" or tracking_best_version_label == "BASE LLM") and smallest_index is not None:
            best_index = smallest_index
        else:
            best_index = tracking_best_version
        
        meta_data_file += "\n\n\n----- ENERGY EFFICIENCY METRICS -----\n"
        meta_data_file += "Best refactored code version: " + str(best_index)  # leave off +1 because we don't want the array index
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
    if energy_efficiency_refactor_success:
        meta_data_file += "\n\n\n----- CODE COMPARISON -----\n"
        meta_data_file += code_comparison_response

    # Write meta data for this experiment
    if make_script:
        filename = directory + "/" + "metadata" + "_" + experiment + ".txt"
    else:
        filename = directory + "/" + "metadata" + "_" + file_name_save + ".txt"
    with open(filename, 'w') as file:
        file.write(meta_data_file)
    file.close()

    print("\nMetadata file: " + str(filename))

    return True