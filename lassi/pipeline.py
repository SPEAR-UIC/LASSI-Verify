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
from lassi.get_power import report_gpu_memory
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary

prompts = PromptDictionary()

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
                                     input_code,
                                     # target_code,
                                     target_lang,
                                     refactored_code_extension,
                                     source_lang,
                                     code_compiler,
                                     code_compiler_kwds,
                                     build_script,
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
                                     vary_llm_temp
                                    ):
    
    # Test compile and execute the source code to validate compiler and execution environment
    actual_energy = 0
    actual_ave_power = 0
    idle_energy_joules = 0
    idle_ave_power = 0
    total_energy_joules_refactor = 0
    ave_power_refactor = 0
    duration_refactor = 0
    self_correction_max = 10
    self_correction_power_counter_max = 10
    exe_output_source = ""
    energy_efficiency_refactor_success = False
    energy_efficiency_iteration = 0
    code_version = 0
    tracking_best_version = -1  # to account for the source code at index 0
    valid_code_flag = "INVALID"  # default value, or "VALID"
    code_gen_temp = 0.2  # default LLM temperature
    code_gen_temp_max = 1.0
    energy_history = []
    power_history = []
    runtime_history = []
    valid_code_history = []
    code_blocks = []  # Collection of all codes generated
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
    
    report_gpu_memory(llm_source)
    
    if test_code_first:
        """Initial stage to run the SOURCE CODE and gather metrics."""
        code_source = "source"  # generated
        print("\n\n----- " + color.BLUE + "SOURCE CODE: Test compile and execute" + color.END + " -----")
        print("\nTest compile the original source code: " + str(source_file_path))
        if build_script:
            compile_result = lassi.code_manager.build_script_code("source_codes", code_build_script)
            execute_file = "/home/cc/lassi/LASSI_dev/source_codes/altis/build/bin/level1/" + file_name
        else:
            compile_result, execute_file = lassi.code_manager.compile_code(source_file_path, source_lang, code_compiler, code_compiler_kwds)
        if compile_result != "":
            print("\n\nFailed compile result: " + str(compile_result))
            return False
        if yes_execute:
            print("\nTest executing the compiled original source code: " + str(execute_file))
            exe_file_path = execute_file  # directory + "/" + execute_file
            execute_result, exe_output_source, total_energy_joules, ave_power, duration, metrics_results_string = lassi.code_manager.execute_code(exe_file_path, exe_input_parameters, power_measure, code_source, target_lang, build_script, wait=1)
            input_content_llmjudge_dict["source_stdout"] = exe_output_source
            input_content_llmjudge_dict["source_metrics"] = metrics_results_string
            # Save source code energy metrics:
            sourcecode_total_energy_joules = total_energy_joules
            sourcecode_ave_power = ave_power
            if execute_result != "0":
                return False
            print("\n\nMeasure system idle power...")
            idle_energy_joules, idle_ave_power = lassi.get_power.idlepower(duration)
            actual_energy = sourcecode_total_energy_joules - idle_energy_joules
            energy_history.append(round(sourcecode_total_energy_joules, 2))
            actual_ave_power = sourcecode_ave_power - idle_ave_power
            power_history.append(round(float(sourcecode_ave_power), 2))
            runtime_history.append(round(float(duration),2))
            valid_code_history.append("SOURCE")
    
    print("\n\n")

    compile_result = "start"
    content = ""
    content_summary_only = ""
    execute_result = ""
    error_message = ""
    error_source = ""  # "compiled" or "executed"
    code_block = ""
    self_correction_counter = 0
    self_correction_power_counter = 0

    if use_context:
        general_system_prompt = prompts.get_system_prompt("general_system_ee")
    else:
        general_system_prompt = prompts.get_system_prompt("general_system_nocontext_ee")
        this_llm_systemcode_prompt = general_system_prompt

    if use_self_prompting:
        print("\n----- " + color.BOLD + color.BLUE + "STARTING LLM CONTEXT BUILDING WITH SELF-PROMPTING" + color.END + " -----\n")
    else:
        print("\n----- " + color.BOLD + color.BLUE + "STARTING LLM CONTEXT BUILDING WITHOUT SELF-PROMPTING" + color.END + " -----\n")
    
    report_gpu_memory(llm_source)
    
    # Augmented prompting to handle processing of provided knowledge context
    if use_context:
        # Inject provided knowledge context--must be provided from top-level LASSI framework script.
        content = "" + input_context + "\n\n"  # [BEGIN CONTEXT KNOWLEDGE]  [/END CONTEXT KNOWLEDGE]

        original_context_knowledge = content

        # Request LLM summarization of provided context knowledge
        if not baseline_oneshot_goal:
            # Note, the input_context object includes all context requested by the user in the LASSI framework.
            if use_self_prompting:
                print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to summarize context knowledge." + color.END + "\n")
                llm_code_knowledge_summary_response = lassi.llm_methods.code_knowledge_llm(llm_source, model, num_ctx, general_system_prompt, input_context, target_lang, display_response, verifyprompt=True)
                if use_context_summaries_only:
                    # Save only LLM summary as context
                    # NOTE: this resets the content to not include the original context
                    content = "[YOUR SUMMARY OF BACKGROUND KNOWLEDGE] " + llm_code_knowledge_summary_response + "\n\n"
                else:
                    # Keep the original context in the content to be included in the LLM prompting
                    content = content + "[YOUR SUMMARY OF BACKGROUND KNOWLEDGE] " + llm_code_knowledge_summary_response + "\n\n"  # [/END CONTEXT KNOWLEDGE SUMMARY]

        report_gpu_memory(llm_source)

        # Requet LLM summary description of the provided source code
        if (not baseline_oneshot_goal and use_self_prompting):
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to describe the source code." + color.END + "\n")
            llm_code_description_response = lassi.llm_methods.code_description_llm(llm_source, model, num_ctx, general_system_prompt, input_context, input_code, display_response, verifyprompt=True)
            content = content + "[YOUR ANALYSIS OF THE CODE TO REFACTOR] " + llm_code_description_response + "\n\n"
        
    else:
        # Not using context
        content = "Think carefully before developing the code following these instructions."

    # Save the original context prompt for re-use during error correction.
    original_context_content_prompt = content
    # This token count does not include the source code
    original_prompt_tokens = lassi.token_analysis.token_count(original_context_content_prompt)

    # BUILD INSTRUCTION PROMPT
    instructions_prompt = ""
    instructions_prompt = instructions_prompt + "\n\n" + "[INSTRUCTIONS TO REFACTOR CODE TO IMPROVE ENERGY EFFICIENCY]\n" + this_llm_prompt
    # Append the background knowledge and/or LLM-generated summaries
    instructions_prompt = instructions_prompt + "\n\n" + original_context_content_prompt
    instructions_prompt_core_iterations = instructions_prompt
    # Append the source code to the prompt
    instructions_prompt = instructions_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]"
    instructions_prompt = instructions_prompt + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(sourcecode_total_energy_joules, 2)) + " Joules at an average power of " + str(round(sourcecode_ave_power, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
     
    # Save the original context + energy efficiency context for re-use during energy efficiency correction
    original_ee_context_content_prompt = instructions_prompt
    original_ee_prompt_tokens = lassi.token_analysis.token_count(original_ee_context_content_prompt)

    report_gpu_memory(llm_source)
    
    print("\n\n----- " + color.BOLD + color.GREEN + "REVIEW THE INSTRUCTIONS TASK PROMPT " + color.END + " (prompt with code, token count: " + str(original_ee_prompt_tokens) + "): \n")  # + instructions_prompt + "\n")
    
    # Call LLM to generate a refactoring plan
    if (not baseline_oneshot_goal and use_self_prompting):
        # Prompt structure: GOAL - RETURN FORMAT - WARNINGS - CONTEXT
        refactoring_plan = prompts.get_code_ee_refactoring_prompt("CUDA_refactor")
        # Context:
        refactoring_plan = refactoring_plan + "\n\n" + original_context_content_prompt
        # Generate the refactoring plan
        refactoring_plan = refactoring_plan + "\n\n" + "[YOUR REFACTORING PLAN]\n" + lassi.llm_methods.llm_call("refactoring plan", llm_source, model, num_ctx, general_system_prompt, instructions_prompt, display_response, verifyprompt=True)
        # Append few-shot learning examples
        if few_shot_learning:
            refactoring_plan = refactoring_plan + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
        refactoring_plan_original = refactoring_plan  # Save for self-correction iterations.
        # Append code to refactor
        refactoring_plan = refactoring_plan_original + "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]" 
        
        refactoring_plan_tokens = lassi.token_analysis.token_count(refactoring_plan)
    
        print("\n\n----- " + color.BOLD + color.GREEN + "REVIEW THE REFACTORING PLAN PROMPT " + color.END + " (prompt with code, token count: " + str(refactoring_plan_tokens) + "): \n")  # + refactoring_plan + "\n")

    else:
        # Just use fixed prompting instructions for baseline LLM capability
        refactoring_plan = instructions_prompt
        refactoring_plan_original = instructions_prompt_core_iterations  # Save for self-correction iterations.
   
    print("\n\n----- " + color.BOLD + color.BLUE + "STARTING CODE GENERATION PIPELINE" + color.END + " -----\n")
    
    report_gpu_memory(llm_source)
    
    # Unload model first if we want a different coding model
    if (model_coder != model and llm_source != "argo"):
        # CHANGE MODEL FOR CODE GENERATION
        print(" (... changing LLM for code gen)")
        lassi.llm_config.unload_ollama_model(model)
        time.sleep(10)  # Pause for 10 seconds
    
    # Clean up Python-level objects
    gc.collect()
    time.sleep(5)
    
    report_gpu_memory(llm_source)
    
    start_time = time.time()
    
    # INITIATE LLM CODE GENERATION ITERATIVE PIPELINE
    while compile_result != "" and self_correction_counter <= self_correction_max and self_correction_power_counter <= self_correction_power_counter_max:
                
        if compile_result == "start":
            # CALL THE LLM to generate the first code refactoring using the source code        
            if llm_source != "argo":
                time.sleep(10)
            report_gpu_memory(llm_source)
            
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt the LLM to generate code." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, this_llm_systemcode_prompt, refactoring_plan, display_response, verifyprompt=True)
        
        elif compile_result != "energy_efficiency":
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
            print("\n\n" + color.BOLD + "REVIEW PROMPT " + color.END + "(error correction, token count: " + str(error_prompt_tokens) + "): " + error_prompt + "\n\n")
            # error_prompt = re.sub('\n', ' ', error_prompt)
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code following an error." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, 0.2, this_llm_systemcode_prompt, error_prompt, display_response, verifyprompt=True)

        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == False and not baseline_oneshot_goal:
            """Self-correction feedback for improving energy efficiency following no prior improvement.
            This condition does not generate a new refactoring plan, but re-uses the last one.
            This considtion increments the LLM temperature by 0.2. If this reaches 1.0, then terminate LASSI pipeline."""
            print("\n\n----- " + color.BOLD + color.PURPLE + "SELF-CORRECTION FOR ENERGY EFFICIENCY: " + str(self_correction_power_counter) + color.END + " -----")
            print(f"  Using refactored code version: {tracking_best_version}")
            if vary_llm_temp:
                code_gen_temp += 0.2
                if code_gen_temp >= code_gen_temp_max:
                    # Set condition to end LASSI pipeline
                    self_correction_power_counter = self_correction_power_counter_max + 1
                    print("\n\n----- END LASSI PIPELINE BECAUSE REACHED MAX LLM TEMPERATURE -----")
                    break
                
            report_gpu_memory(llm_source)
                                   
            if use_self_prompting:
                refactoring_plan_iteration = prompts.get_code_ee_refactoring_prompt("CUDA_refactor_iteration")
            else:
                refactoring_plan_iteration = prompts.get_code_ee_refactoring_prompt("CUDA_refactor_iteration_noplan")
            # Context:
            # if only the first pass, then refactoring_plan_original will be for the source code,
            # otherwise, it will include the latest refactoring plan for the best code
            refactoring_plan_iteration = refactoring_plan_iteration + "\n\n" + refactoring_plan_original
            # Append few-shot learning examples
            if few_shot_learning:
                refactoring_plan_iteration = refactoring_plan_iteration + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
            # Append code to refactor            
            refactoring_plan_iteration = refactoring_plan_iteration + "\n\n" + "[START OF CODE TO REFACTOR]\n" + code_blocks[tracking_best_version+1] + "\n[/END OF CODE TO REFACTOR]"
            refactoring_plan_iteration = refactoring_plan_iteration + "\n\n" + "The above code was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and consumed " + str(round(total_energy_joules_refactor, 2)) + " joules of energy with an average power of " + str(round(ave_power_refactor, 2)) + " Watts."

            refactoring_plan_iteration_tokens = lassi.token_analysis.token_count(refactoring_plan_iteration)
                        
            # ee_correction_prompt_intro = "\n -- The above code was executed after a successul compile with " + code_compiler + " " + code_compiler_kwds + " and consumed " + str(round(total_energy_joules_refactor, 2)) + " joules of energy with an average power of " + str(round(ave_power_refactor, 2)) + " Watts."
            # ee_correction_prompt_outro = " Please re-factor the above code with optimization techniques to reduce total energy consumption during execution. REMINDER: Be sure to maintain all function signatures, class structures, and kernel logic in your refactored code while improving energy efficiency. Only provide the final code in your response."
            # ee_prompt = original_ee_context_content_prompt + "\n\n" + "[START OF CODE TO REFACTOR] " + code_block + "  [/END OF CODE TO REFACTOR] " + "\n\n" + ee_correction_prompt_intro + "\n\n" + ee_correction_prompt_outro
            # ee_prompt_tokens = lassi.token_analysis.token_count(ee_prompt)
            # print("\n\n----- " + color.BOLD + color.RED + "ENERGY EFFICIENCY SELF-CORRECTION" + color.END + " -----")
            print("\n\n" + color.BOLD + "REVIEW PROMPT " + color.END + "(energy efficiency correction, token count: " + str(refactoring_plan_iteration_tokens) + "): ")  # + refactoring_plan_iteration + "\n\n")
            # refactoring_plan_iteration = re.sub('\n', '', refactoring_plan_iteration)
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code for energy efficiency -- correction iteration." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, this_llm_systemcode_prompt, refactoring_plan_iteration, display_response, verifyprompt=True)        

        elif compile_result == "energy_efficiency" and energy_efficiency_refactor_success == True and not baseline_oneshot_goal:
            """Self-correction feedback for improving energy efficiency following prior improvement.
            The condition will generate a new refactoring plan for the latest best code."""
            print("\n\n----- " + color.BOLD + color.PURPLE + "SELF-IMPROVEMENT FOR ENERGY EFFICIENCY: " + str(energy_efficiency_iteration) + color.END + " -----\n")
            print(f"  Using refactored code version: {tracking_best_version}")
            report_gpu_memory(llm_source)

            # Append the original background knowledge and core instructions
            instructions_prompt = instructions_prompt_core_iterations
            # Append the latest (i.e., "best") refactored code -- as the new source code -- to the prompt
            instructions_prompt = instructions_prompt + "\n\n[START OF CODE TO REFACTOR]\n" + code_blocks[tracking_best_version+1] + "\n[/END OF CODE TO REFACTOR]"
            instructions_prompt = instructions_prompt + "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(total_energy_joules_refactor, 2)) + " Joules at an average power of " + str(round(ave_power_refactor, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
            # instructions_prompt = instructions_prompt + "\n\n" + "[INSTRUCTIONS TO REFACTOR CODE TO IMPROVE ENERGY EFFICIENCY]\n" + this_llm_prompt
            
            # Save the original context + energy efficiency context for re-use during energy efficiency correction
            original_ee_context_content_prompt = instructions_prompt
            original_ee_prompt_tokens = lassi.token_analysis.token_count(original_ee_context_content_prompt)

            # Call LLM to generate a NEW refactoring plan
            # Prompt structure: GOAL - RETURN FORMAT - WARNINGS - CONTEXT
            if use_self_prompting:
                print("\n... " + color.GREEN + "llm_call to generate a new refactoring plan " + color.END + "...\n")
                refactoring_plan = prompts.get_code_ee_refactoring_prompt("CUDA_refactor_iteration")
            else:
                refactoring_plan = ""
            # Context:
            # Reset context: We don't want the Instructions to Refactor to appear in our code generation prompt later
            refactoring_plan = refactoring_plan + "\n\n" + original_context_content_prompt
            if use_self_prompting:
                # Generate the NEW refactoring plan
                refactoring_plan = refactoring_plan + "\n\n" + "[YOUR REFACTORING PLAN]\n" + lassi.llm_methods.llm_call("refactoring plan", llm_source, model, num_ctx, general_system_prompt, instructions_prompt, display_response, verifyprompt=True)
            else:
                refactoring_plan = instructions_prompt_core_iterations
            # Append few-shot learning examples
            if few_shot_learning:
                refactoring_plan = refactoring_plan + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
            
            refactoring_plan_original = refactoring_plan  # Save for self-correction iterations.
            # Provide the code version that is tracking as the "best"
            refactoring_plan = refactoring_plan_original + "\n\n[START OF CODE TO REFACTOR]\n" + code_blocks[tracking_best_version+1] + "\n[/END OF CODE TO REFACTOR]"
            
            refactoring_plan_tokens = lassi.token_analysis.token_count(refactoring_plan)
            
            print("\n\n" + color.BOLD + color.GREEN + "----- REVIEW PROMPT " + color.END + "(energy efficiency improvement, token count: " + str(refactoring_plan_tokens) + "): ")  # \n\n" + refactoring_plan + "\n\n")
            # refactoring_plan_iteration = re.sub('\n', '', refactoring_plan)
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to generate code for energy efficiency -- improvement iteration." + color.END + "\n\n")
            code_response, code_block = lassi.llm_methods.code_generator_llm(llm_source, model_coder, num_ctx_coder, code_gen_temp, this_llm_systemcode_prompt, refactoring_plan, display_response, verifyprompt=True)
        
        if code_block.startswith('c++') or code_block.startswith('cpp'):
            code_block = code_block[3:]
        if code_block.startswith('c'):
            code_block = code_block[1:]    
    
        # SAVE THE GENERATIVE CODE -- before compiling
        file_name_save = file_name + "_" + dt_string
        file_path = directory + "/" + file_name_save + "." + refactored_code_extension
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

            report_gpu_memory(llm_source)
            
            exe_file_path = execute_file  # directory + "/" + execute_file
            execute_result, exe_output_generated, total_energy_joules_refactor, ave_power_refactor, duration_refactor, metrics_results_string = lassi.code_manager.execute_code(exe_file_path, exe_input_parameters, power_measure, code_source, target_lang, build_script, wait=5)
            if execute_result != "0":
                compile_result = "execute_error"  # just a flag
            else:
                # Save the generated code as a separate versioned copy
                file_path = directory + "/" + file_name_save + "_v" + str(code_version) + "." + refactored_code_extension
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as file:
                    file.write(code_block)
                file.close()
                code_blocks.append(code_block)  # save this code in our code collection list
                input_content_llmjudge_dict["refactored_code"] = code_block
                input_content_llmjudge_dict["refactored_stdout"] = exe_output_generated
                input_content_llmjudge_dict["refactored_metrics"] = metrics_results_string
                code_version = code_version + 1
                

        # Begin POWER METRICS feedback loop
        if yes_power_feedback and yes_execute and compile_result == "" and execute_result == "0" and self_correction_power_counter <= self_correction_power_counter_max:
            if not baseline_oneshot_goal:
                self_correction_power_counter = self_correction_power_counter + 1
            else:
                # Do not enable feedback iteration in baseline mode
                self_correction_power_counter = self_correction_power_counter_max + 1
            
            # Call LLM-as-a-judge to review SOURCE CODE and REFACTORED CODE to evaluate if likely valid
            valid_code_flag = lassi.llm_methods.llm_judge(llm_source, model_judge, num_ctx, general_system_prompt, input_content_llmjudge_dict, display_response, verifyprompt=True)
            
            # valid_code_flag can be: (1) "VALID", (2) "INVALID", or (3) "NONE"
            valid_code_history.append(valid_code_flag)
            
            # Substract out idle metrics
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
                total_energy_joules_refactor < energy_history[tracking_best_version+1]
                and valid_code_flag == "VALID"
            ):  # sourcecode_total_energy_joules:  # or ave_power_refactor < sourcecode_ave_power:
                """ Let's only care about the total energy consumption and not average power. """
                print("\n\n***** Generated refactored code is more energy efficient than source or previous code! *****")
                print("... following refactor iteration: " + str(energy_efficiency_iteration) + "\n\n")
                energy_efficiency_iteration += 1
                compile_result = "energy_efficiency"
                energy_efficiency_refactor_success = True
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
                        previous_energy = energy_history[tracking_best_version+1]
                        current_energy = energy_history[-1]
                        previous_power = power_history[tracking_best_version+1]
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
                        tracking_best_version = code_version - 1  # code_version has already been incremented, so we need to step one back
                    
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
                if (not baseline_oneshot_goal and valid_code_flag == "VALID"):
                    print("\n\n" + color.GREEN + color.BOLD + " ----- ENERGY EFFICIENCY REVIEW: Let's provide feedback on the energy consumption. -----" + color.END + "\n\n")
                    compile_result = "energy_efficiency"
                elif valid_code_flag != "VALID":
                    print("\n\n" + color.RED + color.BOLD + " ----- GENERATED REFACTORED CODE: May not be valid or not determined. ----- " + color.END + "\n\n")
                    # We're done!
                    break
                energy_efficiency_refactor_success = False
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
    
    print("\n\n-----" + color.BOLD + color.GREEN + "CODE GENERATION COMPLETE" + color.END + "-----\n")
    end_time_minutes = round(end_time/60, 2)
    print("CODE GENERATION TIME: " + str(end_time_minutes) + " minutes.")
    if valid_code_flag != "VALID":
        print(f"  Stopped LASSI early due to invalid code on the last generation version number = {code_version - 1}.")
    print(" Self-correction count: " + str(self_correction_counter))
    if not baseline_oneshot_goal:
        print(" Self-improvement ee refactor count: " + str(self_correction_power_counter))
    print(" Energy efficiency improvement count: " + str(energy_efficiency_iteration))
    if tracking_best_version < 0:
        tracking_best_version = "SOURCE"
    print(" Best refactored code, version: " + str( (tracking_best_version) ) + "\n" )

    # Prompt the LLM to compare the codes
    if (energy_efficiency_refactor_success or baseline_oneshot_goal):

        print("\n\n----- " + color.BOLD + color.BLUE + "BEGIN CODE COMPARISON" + color.END + " -----\n")
        
        compare_code = prompts.get_code_comparison("cuda_cpp")
        compare_code = compare_code + "\n\n" + "[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_code
        compare_code = compare_code + "\n\nOUTPUT: \n\n" + exe_output_source
        compare_code = compare_code + "\n\nCode execution duration: " + str(round(duration,2)) + " s"
        compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(sourcecode_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(sourcecode_ave_power,2)) + " W"
        compare_code = compare_code + "\n\n[/END SOURCE CODE AND RESULTS]"
        
        compare_code = compare_code + "\n\n" + "[BEGIN REFACTORED CODE AND RESULTS]\n\n" + code_block
        compare_code = compare_code + "\n\nOUTPUT: \n\n" + exe_output_generated
        compare_code = compare_code + "\n\nCode execution duration: " + str(round(duration_refactor,2)) + " s"
        compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(total_energy_joules_refactor,2)) + " J" + "\nAverage Power: " + str(round(ave_power_refactor,2)) + " W"
        compare_code = compare_code + "\n\n[/END REFACTORED CODE AND RESULTS]"

        compare_code_prompt_tokens = lassi.token_analysis.token_count(compare_code)

        print("\n----- " + color.BOLD + color.GREEN + "REVIEW THE CODE COMPARISON PROMPT " + color.END + " (prompt token count: " + str(compare_code_prompt_tokens) + "): \n\n" + compare_code + "\n")

        # Send prompt to LLM endpoint
        print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "LLM prompt to compare codes." + color.END + "\n\n")
        code_comparison_response = lassi.llm_methods.llm_call("code comparison", llm_source, model, num_ctx, general_system_prompt, compare_code, display_response, verifyprompt=False)
        

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
    meta_data_file += "----- METRICS -----"
    meta_data_file += "\n"
    meta_data_file += "Experiment date-time: " + str(dt_string)
    meta_data_file += "\n"
    meta_data_file += "Time (sec): " + str(round(end_time, 2))
    meta_data_file += "\n"
    # meta_data_file += f"Token similarity (tokenize): {similarity_tokens:.2f}"
    # meta_data_file += "\n"
    # meta_data_file += f"Token similarity (tiktoken): {similarity_tokens_tiktoken:.2f}"
    # meta_data_file += "\n"
    # meta_data_file += f"Line-based similarity with reordering: {similarity_lines:.2f}"
    # meta_data_file += "\n"
    meta_data_file += "Self-correction iteration count: " + str(self_correction_counter)
    if not baseline_oneshot_goal:
        meta_data_file += "Self-improvement energy efficiency refactor count" + str(self_correction_power_counter)
    if yes_execute:
        meta_data_file += "\n\n\n----- SOURCE: EXECUTION OUTPUT -----\n"
        meta_data_file += exe_output_source
        meta_data_file += "\n\n\n----- GENERATED: EXECUTION OUTPUT -----\n"
        meta_data_file += exe_output_generated
    if yes_power_feedback:
        meta_data_file += "\n\n\n----- ENERGY EFFICIENCY METRICS -----\n"
        meta_data_file += "Total Energy (J): "
        meta_data_file += "  Idle: " + str(round(idle_energy_joules, 2))
        meta_data_file += "  Source code: " + str(round(sourcecode_total_energy_joules, 2))
        meta_data_file += "  Refactored code: " + str(round(total_energy_joules_refactor, 2))
        meta_data_file += "\n"
        meta_data_file += "Average Power (W): "
        meta_data_file += "  Idle: " + str(round(idle_ave_power, 2))
        meta_data_file += "  Source code: " + str(round(sourcecode_ave_power, 2))
        meta_data_file += "  Refactored code: " + str(round(ave_power_refactor, 2))
        meta_data_file += "\n"
        meta_data_file += "Runtime (s): "
        meta_data_file += "  Idle: " + str(round(duration, 2))
        meta_data_file += "  Source code: " + str(round(duration, 2))
        meta_data_file += "  Refactored code: " + str(round(duration_refactor, 2))
    if energy_efficiency_refactor_success:
        meta_data_file += "\n\n\n----- CODE COMPARISON -----\n"
        meta_data_file += code_comparison_response

    # Write meta data for this experiment
    filename = directory + "/" + "metadata" + "_" + file_name_save + ".txt"
    with open(filename, 'w') as file:
        file.write(meta_data_file)
    file.close()

    print("\nMetadata file: " + str(filename))

    return True