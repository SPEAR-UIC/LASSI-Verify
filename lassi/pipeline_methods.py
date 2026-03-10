import subprocess
import lassi.code_manager
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary
from lassi.gpu_detector import detect_gpu_vendors

prompts = PromptDictionary()


def run_command(command_list):
    try:
        return subprocess.check_output(command_list, universal_newlines=True)
    except Exception as e:
        return f"Error running {' '.join(command_list)}: {e}"


def get_gpu_system_details():
    """
    Get comprehensive GPU system details for both NVIDIA and AMD GPUs:
    + Extract device details to provide the LLM with comprehensive context about
        the target system regardless of GPU vendor, enabling it to generate optimized
        code for the specific hardware.
    
    + For NVIDIA systems:
        CUDA compiler version
        Full nvidia-smi details
        Power/clock/P-state information
        PCIe details
        System info
    
    + For AMD systems:
        HIP compiler version
        ROCm system information
        rocm-smi GPU overview
        Detailed power, memory, clock, temperature data
        System info
    """
    details = []
    
    # Detect available GPU vendors
    vendors = detect_gpu_vendors()
    if not vendors:
        details.append("=== No GPUs Detected ===")
        details.append("No NVIDIA or AMD GPUs found on this system")
        return "\n".join(details)
    
    detected_vendor = vendors[0]['vendor']
    details.append(f"=== Detected GPU Vendor: {detected_vendor.upper()} ===")
    
    if detected_vendor == "nvidia":
        # NVIDIA-specific details
        details.append("=== CUDA Compiler Version (nvcc --version) ===")
        details.append(run_command(["nvcc", "--version"]))
        
        # Full GPU details from nvidia-smi -q
        details.append("=== NVIDIA SMI Full Details (nvidia-smi -q) ===")
        details.append(run_command(["nvidia-smi", "-q"]))

        # Granular power and clock metrics
        details.append("=== Power and Clock Metrics ===")
        details.append(run_command([
            "nvidia-smi",
            "--query-gpu=power.draw,power.limit,clocks.sm,clocks.mem,clocks.gr",
            "--format=csv"
        ]))

        # GPU Performance State (P-State)
        details.append("=== GPU Performance State (P-State) ===")
        details.append(run_command([
            "nvidia-smi",
            "--query-gpu=pstate",
            "--format=csv"
        ]))

        # PCIe details
        details.append("=== PCIe Details ===")
        details.append(run_command([
            "nvidia-smi",
            "--query-gpu=pci.bus_id,pci.device_id,pci.domain,pci.bus",
            "--format=csv"
        ]))
    
    elif detected_vendor == "amd":
        # AMD-specific details
        details.append("=== HIP Compiler Version (hipcc --version) ===")
        details.append(run_command(["hipcc", "--version"]))
        
        # ROCm version and system info
        details.append("=== ROCm System Information (rocminfo) ===")
        details.append(run_command(["rocminfo"]))
        
        # AMD GPU details from rocm-smi
        details.append("=== AMD GPU Overview (rocm-smi) ===")
        details.append(run_command(["rocm-smi"]))
        
        # Detailed power information
        details.append("=== AMD GPU Power Details (rocm-smi --showpower) ===")
        details.append(run_command(["rocm-smi", "--showpower"]))
        
        # Memory information
        details.append("=== AMD GPU Memory Details (rocm-smi --showmeminfo vram) ===")
        details.append(run_command(["rocm-smi", "--showmeminfo", "vram"]))
        
        # Clock information
        details.append("=== AMD GPU Clock Details (rocm-smi --showclocks) ===")
        details.append(run_command(["rocm-smi", "--showclocks"]))
        
        # Temperature information
        details.append("=== AMD GPU Temperature Details (rocm-smi --showtemp) ===")
        details.append(run_command(["rocm-smi", "--showtemp"]))
    
    # Common system-level details for both vendors
    details.append("=== CPU Information (lscpu) ===")
    details.append(run_command(["lscpu"]))
    
    details.append("=== Memory Information (free -h) ===")
    details.append(run_command(["free", "-h"]))
    
    # Kernel and OS information
    details.append("=== System Information (uname -a) ===")
    details.append(run_command(["uname", "-a"]))
    
    details.append("=== OS Release Information ===")
    details.append(run_command(["cat", "/etc/os-release"]))

    # Combine all details into a single text string
    return "\n".join(details)


def test_source_code(app_name, experiment_data_directory,
                     code_version,
                     trial_number,
                     source_file_path,
                     build_script,
                     make_script,
                     directory,
                     source_lang,
                     target_lang,
                     exe_input_parameters,
                     code_compiler,
                     code_compiler_kwds,
                     yes_execute,
                     power_measure,
                     input_content_llmjudge_dict,
                     energy_history,
                     power_history,
                     runtime_history,
                     valid_code_history
                    ):
    """
    Compile, execute, and measure power profile of source code.
    """
    code_source = "source"  # generated
    print("\n\n----- " + color.BLUE + "SOURCE CODE: Test compile and execute" + color.END + " -----")
    print("\nTest compile the original source code: " + str(source_file_path))
    if build_script:
        compile_result = lassi.code_manager.build_script_code("source_codes", code_build_script)
        execute_file = "/home/cc/lassi/LASSI_dev/source_codes/altis/build/bin/level1/" + file_nameexe_input_parameters
    elif make_script:
        compile_result = lassi.code_manager.make_script_code(directory, code_compiler_kwds, source_lang)
        if compile_result == "":
            if source_lang == "CUDA":
                if app_name == "miniMDock":
                    execute_file = "bin/autodock_gpu_64wi"  # HARD CODED FOR MINIMDOCK
                    exe_input_parameters = "-lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 250".split()
                elif app_name == "XSBench":
                    execute_file = "XSBench"  # HARD CODED FOR XSBENCH
                    exe_input_parameters = "-m event -l 1700000000".split()
            elif source_lang == "HIP":
                if app_name == "miniMDock":
                    execute_file = "bin/autodock_hip_gpu_64wi"  # HARD CODED FOR MINIMDOCK
                    exe_input_parameters = "-lfile ./input/7cpa/7cpa_ligand.pdbqt -nrun 250".split()
                elif app_name == "XSBench":
                    execute_file = "XSBench"  # HARD CODED FOR XSBENCH
                    exe_input_parameters = "-m event -l 1700000000".split()
    else:
        compile_result, execute_file = lassi.code_manager.compile_code(source_file_path, source_lang, code_compiler, code_compiler_kwds)
    
    if compile_result != "":
        print("\n\nFailed compile result: " + str(compile_result))
        return False
    if yes_execute:
        print("\nTest executing the compiled original source code: " + str(execute_file))
        exe_file_path = execute_file  # directory + "/" + execute_file
        zero_shot = False
        execute_result, exe_output_source, total_energy_joules, ave_power, std_power, duration, metrics_results_string = lassi.code_manager.execute_code_selectvendor(experiment_data_directory, code_version, zero_shot, trial_number, directory, exe_file_path, exe_input_parameters, power_measure, code_source, target_lang, build_script, make_script, gpu_vendor="auto", wait=1)
        input_content_llmjudge_dict["source_stdout"] = exe_output_source
        input_content_llmjudge_dict["source_metrics"] = metrics_results_string
        # Save source code energy metrics:
        sourcecode_total_energy_joules = total_energy_joules
        sourcecode_ave_power = ave_power
        if execute_result != "0":
            return False
        idle_energy_joules = 0
        idle_ave_power = 0
        idle_std_power = 0
        if power_measure:
            print("\n\nMeasure system idle power...")
            idle_energy_joules, idle_ave_power, idle_std_power = lassi.get_power.idlepower_selectvendor(experiment_data_directory, trial_number, duration, gpu_vendor="auto")
        actual_energy = sourcecode_total_energy_joules - idle_energy_joules
        energy_history.append(round(sourcecode_total_energy_joules, 2))
        actual_ave_power = sourcecode_ave_power - idle_ave_power
        power_history.append(round(float(sourcecode_ave_power), 2))
        runtime_history.append(round(float(duration),2))
        valid_code_history.append("SOURCE")

    return (idle_energy_joules, idle_ave_power, input_content_llmjudge_dict, sourcecode_total_energy_joules, sourcecode_ave_power,
            duration, energy_history, power_history, runtime_history, valid_code_history)


def selfprompting_context_builder(generate_context_flag,
                                  get_system_info_flag,
                                  source_lang=None,
                                  this_llm_prompt=None,
                                  input_context=None,
                                  llm_source=None,
                                  model=None,
                                  num_ctx=None,
                                  target_lang=None,
                                  display_response=None,
                                  use_context_summaries_only=None,
                                  input_code=None,
                                  original_code=None,  # Uninstrumented source for code description step; defaults to input_code
                                  total_energy_joules=None,
                                  ave_power=None,
                                  verifyprompt=False,
                                  power_measure=True):   # power_measure=False selects runtime optimization prompts
    """
    Update the system and user content prompts to include
    background context and self-prompted context.
    """
    content = ""
    original_context_knowledge = ""
   
    # Augmented prompting to handle processing of provided knowledge context
    if generate_context_flag:
        # Fetch system prompt when using context
        if source_lang == "CUDA":
            sys_key = "general_system_cuda_ee" if power_measure else "general_system_CUDA_runtime"
            general_system_prompt = prompts.get_system_prompt(sys_key)
        elif source_lang == "HIP":
            sys_key = "general_system_hip_ee" if power_measure else "general_system_HIP_runtime"
            general_system_prompt = prompts.get_system_prompt(sys_key)
        # Inject provided knowledge context--must be provided from top-level LASSI framework script.
        content = "" + input_context + "\n\n"  # [BEGIN CONTEXT KNOWLEDGE]  [/END CONTEXT KNOWLEDGE]

        original_context_knowledge = content

        # Request LLM summarization of provided context knowledge
        # Note, the input_context object includes all context requested by the user in the LASSI framework.
        print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to summarize context knowledge." + color.END + "\n")
        llm_code_knowledge_summary_response = lassi.llm_methods.code_knowledge_llm(llm_source, model, num_ctx, general_system_prompt, input_context, target_lang, display_response, verifyprompt, power_measure=power_measure)
        
        if use_context_summaries_only:
            # Save only LLM summary as context
            # NOTE: this resets the content to not include the original context
            content = "[YOUR SUMMARY OF BACKGROUND KNOWLEDGE]\n" + llm_code_knowledge_summary_response + "\n\n"
        else:
            # Keep the original context in the content to be included in the LLM prompting
            content = content + "[YOUR SUMMARY OF BACKGROUND KNOWLEDGE]\n" + llm_code_knowledge_summary_response + "\n\n"  # [/END CONTEXT KNOWLEDGE SUMMARY]

        # Capture system info, have LLM summarize details
        if get_system_info_flag:
            print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to describe the system." + color.END + "\n")
            gpu_system_details = get_gpu_system_details()
            # CALL LLM HERE
            llm_system_description_response = lassi.llm_methods.system_description_llm(llm_source, target_lang, model, num_ctx, general_system_prompt, gpu_system_details, display_response, verifyprompt)
            content = content + "[YOUR DESCRIPTION OF THE CPU/GPU SYSTEM]\n" + llm_system_description_response + "\n\n"

            
        # Requet LLM summary description of the provided SOURCE code
        # Use original_code (uninstrumented) if provided so the LLM describes the clean source
        # without being distracted by invariant assertion instrumentation comments.
        code_for_description = original_code if original_code is not None else input_code
        print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to describe the source code." + color.END + "\n")
        llm_code_description_response = lassi.llm_methods.code_description_llm(llm_source, model, num_ctx, general_system_prompt, input_context, code_for_description, display_response, verifyprompt)
        content = content + "[YOUR ANALYSIS OF THE SOURCE CODE TO REFACTOR]\n" + llm_code_description_response + "\n\n"
        # refactor_instructions = prompts.get_code_ee_refactoring_prompt(str(source_lang + "_refactor"))  # Includes prompt to generate a refactoring plan

    else:
        # For initial zero-shot attempt
        # Fetch system prompt when not using context
        if source_lang == "CUDA":
            sys_key = "general_system_nocontext_cuda_ee" if power_measure else "general_system_nocontext_CUDA_runtime"
            general_system_prompt = prompts.get_system_prompt(sys_key)
        elif source_lang == "HIP":
            sys_key = "general_system_nocontext_hip_ee" if power_measure else "general_system_nocontext_CUDA_runtime"
            general_system_prompt = prompts.get_system_prompt(sys_key)
        content = ""  # "Think carefully before developing the code following these instructions."
        baseline_key = source_lang + "_baseline_nocontext" if power_measure else source_lang + "_baseline_nocontext_runtime"
        content = content + "\n\n" + prompts.get_code_ee_refactoring_prompt(baseline_key)
        
    
    # Save the original context prompt for re-use during error correction.
    # original_context_content_prompt = content
    # This token count does not include the source code
    # original_prompt_tokens = lassi.token_analysis.token_count(original_context_content_prompt)
    
    return original_context_knowledge, content, general_system_prompt


def generate_refactoring_plan(source_lang,
                              input_code,
                              total_energy_joules,
                              ave_power,
                              llm_source,
                              model,
                              num_ctx,
                              system_prompt,
                              context_prompt,
                              prior_refactoring_plans,
                              display_response,
                              few_shot_learning=False,
                              verifyprompt=False,
                              power_measure=True):   # power_measure=False selects runtime optimization prompts
                                                     # NOTE: when power_measure=False, total_energy_joules holds the
                                                     # runtime duration value (passed from runtime_history at call sites)

    # Prompt structure: GOAL - RETURN FORMAT - WARNINGS - CONTEXT
    plan_key = source_lang + "_refactorplan" if power_measure else source_lang + "_refactorplan_runtime"
    refactoring_plan_instructions = prompts.get_code_ee_refactoring_prompt(plan_key)
    
    # Prepend the Context:
    refactoring_plan_instructions = refactoring_plan_instructions + "\n\n" + context_prompt
    
    # Append previous failed plans (if available)
    if len(prior_refactoring_plans) > 0:
        if power_measure:
            failed_plan_context = "For reference to your refactoring plan, the following plans were previously used that did not result in a successful outcome of code that was functionally equivalent to the source code and reduced energy consumption during runtime. Based on these previous plans, try to not repeat specific ideas that may have been unsuccessful."
        else:
            failed_plan_context = "For reference to your refactoring plan, the following plans were previously used that did not result in a successful outcome of code that was functionally equivalent to the source code and reduced wall-clock execution time. Based on these previous plans, try to not repeat specific ideas that may have been unsuccessful."
        refactoring_plan_instructions = refactoring_plan_instructions + "\n\n" + failed_plan_context
        refactoring_plan_instructions = refactoring_plan_instructions + "\n" + " | PREVIOUS FAILED PLAN: ".join(prior_refactoring_plans)
    
    # Append code to be refactored for generating the plan
    code_to_refactor = "\n\n[START OF CODE TO REFACTOR]\n" + input_code + "\n[/END OF CODE TO REFACTOR]"
    if power_measure:
        metric_context = "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and consumed a total energy of " + str(round(total_energy_joules, 2)) + " Joules at an average power of " + str(round(ave_power, 2)) + " Watts, which includes the idle system energy consumption and average power.\n"
    else:
        # NOTE: total_energy_joules holds the runtime duration value when power_measure=False
        # (runtime_history[tracking_best_version] is passed in that argument slot at call sites)
        metric_context = "\n\nFor reference, the CODE TO REFACTOR provided above was executed on this system and had a wall-clock execution time of " + str(round(total_energy_joules, 4)) + " seconds.\n"
    code_to_refactor = code_to_refactor + metric_context
    refactoring_plan_instructions = refactoring_plan_instructions + code_to_refactor
    
    # Generate the refactoring plan
    refactoring_plan = "\n\n" + "[YOUR REFACTORING PLAN]\n" + lassi.llm_methods.llm_call("refactoring plan", llm_source, model, num_ctx, system_prompt, refactoring_plan_instructions, display_response, verifyprompt)
    
    # Append few-shot learning examples
    if few_shot_learning:
        refactoring_plan = refactoring_plan + "\n\n" + prompts.get_codeknowledge("cuda_c_few_shot_examples")
    # refactoring_plan_original = refactoring_plan  # Save for self-correction iterations.
    
    # Append code to refactor to the new refactoring plan
    # NOTE: this is not returned, just for token counting
    refactoring_plan_full = refactoring_plan + code_to_refactor
    refactoring_plan_tokens = lassi.token_analysis.token_count(refactoring_plan_full)
    print("\n\n----- " + color.GREEN + "THE REFACTORING PLAN PROMPT " + color.END + " (prompt with code, token count: " + str(refactoring_plan_tokens) + "): \n")  # + refactoring_plan + "\n")

    return refactoring_plan, code_to_refactor


def generate_code_analysis(source_lang, input_code,
                              llm_source,
                              model,
                              num_ctx,
                              system_prompt,
                              context_prompt,
                              display_response,
                              verifyprompt=False
                             ):
    code_analysis_instructions = """
    You are a formal methods and parallel programming expert and you specialize in identifying key characteristics of a given CUDA code. The characteristics can be something like invariants (a property of the program state that is always true), properties, patterns, unit tests, etc. The goal is to use these characteristics to determine if a refactored version of the code provide the same general functionality. Your task is to:
    1. Identify underlying assumptions of the code
    2. Identify invariants
    3. Identify design patterns
    4. Identify memory access patterns
    5. Identify memory allocation/deallocation
    6. Other key characteristics of the code
    Please output your findings in a json file format. Wrap the json output in a ```json code fence.
    """

    code_to_analyze = "\n\n[START OF CODE TO ANALYZE]\n" + input_code + "\n[/END OF CODE TO ANALYZE]"
    code_analysis_instructions = code_analysis_instructions + code_to_analyze
    code_analysis = lassi.llm_methods.llm_call("code analysis", llm_source, model, num_ctx, system_prompt, code_analysis_instructions, display_response, verifyprompt)

    return code_analysis
    