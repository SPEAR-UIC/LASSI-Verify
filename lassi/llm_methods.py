import re
from lassi.llm_config import call_llm
from lassi.format_colors import color


def llm_code_compare(prompts, input_content_llmjudge_dict, source_duration, source_total_energy_joules, source_ave_power, refactored_duration, refactored_total_energy_joules, refactored_ave_power, llm_source, model, num_ctx, system_prompt, display_response, verifyprompt):
    print("\n\n----- " + color.BOLD + color.BLUE + "FINAL CODE COMPARISON: Source Code vs. Best Refactored Code" + color.END + " -----\n")
            
    compare_code = prompts.get_code_comparison(input_content_llmjudge_dict["refactored_code_lang"] + "_cpp")
    compare_code = compare_code + "\n\n" + "[BEGIN SOURCE CODE AND RESULTS]\n\n" + input_content_llmjudge_dict["source_code"]
    compare_code = compare_code + "\n\nOUTPUT: \n\n" + input_content_llmjudge_dict["source_stdout"]
    compare_code = compare_code + "\n\nCode execution duration: " + str(round(source_duration, 2)) + " s"
    compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(source_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(source_ave_power,2)) + " W"
    compare_code = compare_code + "\n\n[/END SOURCE CODE AND RESULTS]"
            
    compare_code = compare_code + "\n\n" + "[BEGIN REFACTORED CODE AND RESULTS]\n\n" + input_content_llmjudge_dict["refactored_code"]
    compare_code = compare_code + "\n\nOUTPUT: \n\n" + input_content_llmjudge_dict["refactored_stdout"]
    compare_code = compare_code + "\n\nCode execution duration: " + str(round(refactored_duration, 2)) + " s"
    compare_code = compare_code + "\nTotal Energy Consumed: " + str(round(refactored_total_energy_joules,2)) + " J" + "\nAverage Power: " + str(round(refactored_ave_power,2)) + " W"
    compare_code = compare_code + "\n\n[/END REFACTORED CODE AND RESULTS]"
    
    # Send prompt to LLM endpoint
    print("\n" + color.GREEN + "--> ... " + color.UNDERLINE + "LLM prompt to compare codes." + color.END + "\n\n")
    code_comparison_response = llm_call("code comparison", llm_source, model, num_ctx, system_prompt, compare_code, display_response, verifyprompt)

    return code_comparison_response


def llm_call(intent, llm_source, model, num_ctx, system_prompt, content_prompt, display_response, verifyprompt=False):

    if intent == "refactoring plan":
        print("\n\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to write a refactoring plan." + color.END + "\n")
    if intent == "code analysis":
        print("\n\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to perform code analysis." + color.END + "\n")

    response_text = call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("< BEGIN: LLM RESPONSE · " + intent + " >")
    print(response_text)
    print("< END: LLM RESPONSE >")

    return response_text


def llm_judge(target_lang, llm_source, model, num_ctx, system_prompt, input_content_dict, power_measure, run_code_analysis, sourcecode_analysis, display_response, verifyprompt=False):

    intent = "code validation judge"

    print("\n\n" + color.GREEN + "--> ... " + color.UNDERLINE + "prompt LLM to judge the validity of the refactored code." + color.END + "\n")
    
    # Use same system prompt.
    # Build user prompt:
    # Fixed instructions first
    # "SOURCE CODE"
    # <insert source code>
    # "STANDARD OUTPUT
    # <insert runtime output>
    # "METRICS"
    # <insert measurements>
    # "REFACTORED CODE"
    # <insert refactored code>
    # "STANDARD OUTPUT
    # <insert runtime output>
    # "METRICS"
    # <insert measurements>
    content_prompt = f"""
    Please compare the difference between the following two codes written in {target_lang} C++.\n
    First, I will provide you with the SOURCE CODE followed by its STANDARD OUTPUT and METRICS measured during code runtime.\n
    Next, I will provide you with a REFACTORED CODE followed by its STANDARD OUTPUT and METRICS measured during code runtime.\n
    After you have reviewed both codes and their output, the REFACTORED CODE is expected to produce equivalent functionality and reasonably similar standard output in its layout and content. However, the REFACTORED CODE may not meet these expectations.\n
    Your task is to strictly evaluate whether the REFACTORED CODE (1) preserves all intended functionality of the SOURCE CODE and (2) produces printed nearly equivalent standard output, including its displayed layout and content, compared to the SOURCE CODE.\n
    Your careful evaluation should focus on the functional equivalence of the two codes and nearly exact printed standard output of the two codes. Minor differences in numerical values are acceptable in the printed standard output only if they are consistent with the REFACTORED CODE improving energy efficiency and these specific values are reasonable and consistent with the SOURCE CODE's intended purpose. Formatting differences are acceptable as long as key details are included that are seen in the standard output of the SOURCE CODE.\n
    Please explain your reasoning, and then conclude your response with VALID if the REFACTORED CODE is functionally equivalent, or INVALID if it is not.
    """

    content_prompt += f"\n\n# [SOURCE CODE]\n {input_content_dict['source_code']}"
    content_prompt += f"\n\n## [STANDARD OUTPUT]\n {input_content_dict['source_stdout']}"
    content_prompt += f"\n\n## [METRICS]\n {input_content_dict['source_metrics']}"
    content_prompt += f"\n\n# [REFACTORED CODE]\n {input_content_dict['refactored_code']}"
    content_prompt += f"\n\n## [STANDARD OUTPUT]\n {input_content_dict['refactored_stdout']}"
    content_prompt += f"\n\n## [METRICS]\n {input_content_dict['refactored_metrics']}"

    if power_measure:
        if input_content_dict['refactored_metrics'] == "":
        # No energy metrics measured. This refactored code should be flagged INVALID.
            return "INVALID"
    
    response_text = call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("< BEGIN: LLM RESPONSE · " + intent + " >")
    print(response_text)
    print("< END: LLM RESPONSE >")

    # look for "VALID" or "INVALID" in the LLM response
    # only return this value
    # NOTE: check "INVALID" first, because "VALID" is a substring
    if "INVALID" in response_text:
        response_validation = "INVALID"
    elif "VALID" in response_text:
        response_validation = "VALID"
    else:
        response_validation = "NONE"

    # If valid, then run test case agent:
    # 1) prompt with source code 

    # APPLY CODE ANALYSIS TO JUDGE REVIEW
    code_analysis = "None"
    if run_code_analysis:
        if response_validation == "VALID":
            print("\n\nLLM JUDGE: START CODE ANALYSIS ---- ")
            
            code_analysis_instructions = f"""
                Your are a {target_lang} code evaluator. Your task is to determine if the provided {target_lang} code satisfies a given list of characteristics.
                The characteristics will be given in a json format and will be given between the tags: "<rubric>" and "</rubric>".
                The {target_lang} code you are to evaluate will be given between "<code>" and "</code>".
                Please give a simple "PASS" or "FAIL" evaluation and also list which criteria pass and which fail (without additional details).
                Please put your final, overall evaluation between the tags: "<result>" and "</result>" and put the list in a Markdown table format (using | and - characters). Do not use HTML tags in your response.
                """
            code_rubric = "\n\n<rubric>\n" + sourcecode_analysis + "\n</rubric>"
            code_to_analyze = "\n\n<code>\n" + input_content_dict['refactored_code'] + "\n</code>"
            code_analysis_instructions = code_analysis_instructions + code_rubric + code_to_analyze
            code_analysis = llm_call("code analysis", llm_source, model, num_ctx, system_prompt, code_analysis_instructions, display_response, verifyprompt)
            # print("\n\nCODE ANALYSIS RESULT: " + str(code_analysis))
    
            # Change the validation response if code analysis fails
            code_analysis_result = re.search(r'<result>(.*?)</result>', code_analysis, re.DOTALL)
            code_analysis_result = code_analysis_result.group(1).strip().lower()  # Normalize: remove whitespace and make lowercase
            if code_analysis_result == "fail":
                print("-- CHANGE JUDGE VALIDATION RESULT! --")
                response_validation = "INVALID"

    return response_validation, response_text, code_analysis

    
def code_knowledge_llm(llm_source, model, num_ctx, system_prompt, input_context, target_lang, display_response, verifyprompt=False, power_measure=True):

    if power_measure:
        content_prompt = """
    Read and Analyze: First, read the following background knowledge.
    Then, summarize its key relevant points that we should apply to refactoring code for improving energy efficiency.
    """
    else:
        content_prompt = """
    Read and Analyze: First, read the following background knowledge.
    Then, summarize its key relevant points that we should apply to refactoring code for reducing wall-clock execution time.
    """

    if target_lang == "OMP":
        content_prompt = content_prompt + """ 
        Specifically, describe how to implement the omp pragma directive 'target teams' for distributing for loops on a GPU.
        """
        
    # content_prompt = content_prompt + """ 
    # Only respond with your summary details and avoid repeating the provided source code: 
    # """

    content_prompt = content_prompt + "\n\n" + input_context

    response_text = call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("< BEGIN: LLM RESPONSE · context knowledge summary >")
    print(response_text)
    print("< END: LLM RESPONSE >")

    return response_text


def code_description_llm(llm_source, model, num_ctx, system_prompt, input_context, input_code, display_response, verifyprompt=False):

    # content_prompt = """
    # Please provide a detailed summary describing the functionality of the following source code.  
    # Outline the function of the source code step-by-step to include all details so that 
    # you can later refactor this code to include all its features and functions. 
    # Only respond with your summary description and avoid repeating the provided source code: \n\n
    # [BEGIN SOURCE CODE] \n\n
    # """

    content_prompt = """
    Read and Analyze: First, read the following background knowledge.
    Then, carefully read the entire code provided below.
    Describe its complete functionality and summarize the functional blocks: list out all major classes, functions, and data structures, and briefly describe what they do.
    Only respond with your description and analysis. Avoid responding with any code changes or repeating the source code. 
    """
    
    content_prompt = content_prompt + "\n\n" + input_context
    content_prompt = content_prompt + "\n\n[START OF CODE TO DESCRIBE AND ANALYZE]\n" + input_code + "\n\n"  # [/END SOURCE CODE]"

    response_text = call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("< BEGIN: LLM RESPONSE · code description >")
    print(response_text)
    print("< END: LLM RESPONSE >")

    return response_text


def system_description_llm(llm_source, target_lang, model, num_ctx, system_prompt, gpu_system_details, display_response, verifyprompt=False):

    # content_prompt = """
    # Please provide a detailed summary describing the functionality of the following source code.  
    # Outline the function of the source code step-by-step to include all details so that 
    # you can later refactor this code to include all its features and functions. 
    # Only respond with your summary description and avoid repeating the provided source code: \n\n
    # [BEGIN SOURCE CODE] \n\n
    # """

    content_prompt = f"""
    Below are the detailed system metrics of the GPU and related hardware used for executing {target_lang} C++ code.
    Your task is to analyze this information and provide a complete, clear description of the system that highlights
    any details relevant to assessing and improving energy efficiency. 
    In your description, focus on key aspects such as power consumption, clock speeds, utilization, thermal metrics, PCIe configuration,
    and any CPU/memory details provided. 
    Your description should included specific system details important as background context for the task of identifying
    inefficiencies and developing a refactoring plan for energy efficient {target_lang} C++ code.
    \n\nPlease respond with only your detailed descriptive summary of the system. Do not include any code examples or additional extraneous information.
    """
    
    content_prompt = content_prompt + "\n[CPU/GPU SYSTEM DETAILS TO DESCRIBE]\n\n" + gpu_system_details + "\n"

    response_text = call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, 0.2, verifyprompt)

    print("< BEGIN: LLM RESPONSE · GPU system description >")
    print(response_text)
    print("< END: LLM RESPONSE >")

    return response_text


def code_generator_llm(llm_source, model, num_ctx, temp, system_prompt, content_prompt, display_response, verifyprompt=False):

    # Append critical instruction to repsond with complete code
    content_prompt = content_prompt + """\n\nCRITICAL REQUIREMENT: You must output the COMPLETE, FULLY FUNCTIONAL code file. 
    Do not use placeholders like '# ... rest of code unchanged ...' or '// ... existing code ...' or similar shortcuts. 
    Every single line of code must be present in your response, including all unchanged sections. 
    The output will be used directly in an automated pipeline that cannot fill in omitted code. 
    Write out the entire code file from start to finish. If you omit any code, the system will fail. 
    Begin your code output immediately after this instruction.
    """
    
    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE GENERATION" + color.END + " -----\n")
    
    # print("\n\nDEBUG input parameter values:")
    # print("  llm_source: " + str(llm_source))
    # print("  model: " + str(model))
    # print("  num_ctx: " + str(num_ctx))
    # print("  system_prompt: " + str(system_prompt))
    # print("  content_prompt: " + str(content_prompt))
    # print("  display_response: " + str(display_response))
    
    response_text = call_llm(llm_source, model, num_ctx, system_prompt, content_prompt, temp, verifyprompt, stream=False)

    print("< BEGIN: LLM GENERATED CODE · code generator >")
    print(response_text)
    print("< END: LLM GENERATED CODE >")

    # Extract only the code block -
    # Search for the ``` pattern in the text
    #   Regular expression pattern to match text between triple backticks
    #   that represent code blocks within generated responses from the LLM.
    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    # print("RESPONSE TEXT -- ", response_text)
    # print("MATCH -- ", match)
    if match:
        extracted_text = match.group(1)
    else:
        # Fallback: LLM response did not include code fences.
        # Try to extract from the first preprocessor directive or CUDA/C++ keyword.
        # This prevents code_block from being "" when the LLM omits backticks, which
        # would cause compile errors (empty .cu file) and break error-correction prompts.
        fallback_pattern = r'((?:#include|#pragma|#define|__global__|extern\s+"C").*)'
        fallback_match = re.search(fallback_pattern, response_text, re.DOTALL)
        if fallback_match:
            extracted_text = fallback_match.group(1)
            print("\n[WARNING] code_generator_llm: no code fences in LLM response; extracted from first code directive.")
        else:
            extracted_text = response_text
            print("\n[WARNING] code_generator_llm: no code fences or code directives found; using full response as code.")

    return response_text, extracted_text