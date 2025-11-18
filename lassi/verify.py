from IPython.display import display, Markdown
import re
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary
from lassi.llm_config import argo_llm, cerebras_llm, ollama_llm, ollama_llm_nostream

# Load prompt dictionary (to capture recent manual changes to script, if any)
prompts = PromptDictionary()


def code_insert_passthrough(source_code, llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=False):

    content_prompt = r"""
        You are preparing GPU-accelerated scientific code for verification testing via passthrough data comparison.
        
        Given the following source code, your task is to inject a data-dump block that captures the final computational results to stdout for verification analysis.
        
        **CRITICAL REQUIREMENTS:**
        1. The data dump MUST execute every time (no conditional flags)
        2. Insert the dump block AFTER all GPU computation completes but BEFORE any memory deallocations
        3. The dump should NOT affect GPU execution or power measurements
        4. Preserve ALL original code without modification
        5. Use protective comment markers so future LLM refactoring preserves this block
        
        **STEP 1: Identify the computing environment**
        - Detect the programming language: C++ or Fortran
        - Detect the GPU API: CUDA (cudaMalloc/cudaMemcpy) or HIP (hipMalloc/hipMemcpy) or other
        - Identify appropriate memory copy functions and syntax
        
        **STEP 2: Identify primary output buffer(s)**
        - Locate the device pointer(s) holding FINAL computational results (not intermediate/temporary buffers)
        - Determine: buffer name, data type, total element count
        - If multiple distinct output buffers exist, identify all of them
        
        **STEP 3: Determine data characteristics**
        - Infer the exact data type (float, double, int, complex, etc.)
        - Calculate or infer total element count from allocation size or loop bounds
        - Identify appropriate format specifiers for printf (e.g., %f for float, %lf for double, %d for int)
        
        **STEP 4: Inject the passthrough data dump block**
        
        Insert this block at the optimal location:
        - AFTER: all GPU kernels complete and cudaDeviceSynchronize() (or equivalent)
        - BEFORE: any return statements
        - BEFORE: any memory deallocations (cudaFree, hipFree, free, etc.)
        
        Use this template structure, adapting it for the specific code:
        ```cpp
        // >>> BEGIN LASSI-VERIFY PASSTHROUGH (do not edit or delete this block) >>>
        
        printf("\\n=== BEGIN PASSTHROUGH DATA ===\\n");
        
        // For each output buffer identified:
        {
            printf("Buffer: <buffer_name>\\n");
            printf("Type: <data_type>\\n");
            printf("Elements: <total_element_count>\\n");
            
            // Allocate host buffer
            <data_type>* host_buf = (<data_type>*) malloc(<total_elems> * sizeof(<data_type>));
            
            // Copy from device to host using appropriate API
            <memcpy_function>(host_buf, <device_ptr>, <total_elems> * sizeof(<data_type>), <DeviceToHost>);
            
            // Calculate and print statistics
            <data_type> min_val = host_buf[0], max_val = host_buf[0];
            double sum = 0.0;
            for(size_t i = 0; i < <total_elems>; i++) {
                if(host_buf[i] < min_val) min_val = host_buf[i];
                if(host_buf[i] > max_val) max_val = host_buf[i];
                sum += host_buf[i];
            }
            double mean = sum / <total_elems>;
            
            printf("Statistics: min=<format>, max=<format>, mean=<format>, sum=<format>\\n", 
                   min_val, max_val, mean, sum);
            
            // Print sample values (first 10 and last 10)
            int sample_size = (<total_elems> < 10) ? <total_elems> : 10;
            printf("First %d values: ", sample_size);
            for(int i = 0; i < sample_size; i++) {
                printf("<format> ", host_buf[i]);
            }
            printf("\\n");
            
            if(<total_elems> > 10) {
                printf("Last 10 values: ");
                for(size_t i = <total_elems> - 10; i < <total_elems>; i++) {
                    printf("<format> ", host_buf[i]);
                }
                printf("\\n");
            }
            
            free(host_buf);
        }
        
        printf("=== END PASSTHROUGH DATA ===\\n");
        
        // <<< END LASSI-VERIFY PASSTHROUGH <
        ```
        
        **ADAPTATION NOTES:**
        - Replace `<buffer_name>`, `<data_type>`, `<total_elems>`, `<device_ptr>` with actual values from the code
        - Replace `<memcpy_function>` with: cudaMemcpy (CUDA), hipMemcpy (HIP), etc.
        - Replace `<DeviceToHost>` with: cudaMemcpyDeviceToHost (CUDA), hipMemcpyDeviceToHost (HIP), etc.
        - Replace `<format>` with appropriate printf format: %f (float), %lf (double), %d (int), etc.
        - For Fortran: adapt syntax to Fortran conventions (write statements instead of printf, etc.)
        - If data type is complex/custom: extract real components or adapt printing appropriately
        
        **STEP 5: Add required headers**
        Ensure these headers are present (if not already included):
        - C++: `<cstdio>`, `<cstdlib>`, `<algorithm>` (or C equivalents)
        - Fortran: appropriate ISO_C_BINDING if interfacing with C
        
        **PRESERVATION REQUIREMENTS:**
        - Do NOT modify any algorithmic logic, loops, or GPU kernels
        - Do NOT remove or modify existing code or comments
        - Do NOT change performance-critical sections
        - ONLY add: the data dump block and any minimally required headers
        
        **OUTPUT INSTRUCTIONS:**
        Output ONLY the complete, modified source code. Do not include any explanatory text, commentary, or markdown formatting - just the raw source code ready to compile.
        
        ----

    """

    content_prompt = content_prompt + " [BEGIN CODE ANALYSIS]\n" + code_analysis + "\n[/END CODE ANALYSIS]\n\n"
    content_prompt = content_prompt + " [BEGIN SOURCE CODE]\n" + source_code + "\n[/END SOURCE CODE]\n"
    
    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE INSERT - Passthrough Data Dump Blocks" + color.END + " -----\n")
    
    # print("\n\nDEBUG input parameter values:")
    # print("  llm_source: " + str(llm_source))
    # print("  model: " + str(model))
    # print("  num_ctx: " + str(num_ctx))
    # print("  system_prompt: " + str(system_prompt))
    # print("  content_prompt: " + str(content_prompt))
    # print("  display_response: " + str(display_response))

    temp = 0.2
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "cerebras":
        response_text = cerebras_llm(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    else:
        print("Invalid LLM source")
        
    print("\n\n----- " + color.BOLD + color.BLUE + "THE GENERATED CODE" + color.END + " -----\n")
    if display_response:
        print("(hidden)\n\n")
        display(Markdown(response_text))

    # Extract only the code block - 
    # Search for the ``` pattern in the text
    #   Regular expression pattern to match text between triple backticks 
    #   that represent code blocks within generated responses from the LLM.
    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    # print("RESPONSE TEXT -- ", response_text)
    # print("MATCH -- ", match)
    if match:
        extracted_code = match.group(1)
    else:
        extracted_code = ""
    
    return extracted_code


def code_insert_asserts(source_code, llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=False):

    # Append critical instruction to repsond with complete code
    content_prompt = """
    You are a GPU-accelerated scientific CUDA C++ code instrumentation specialist.
    
    Given the following CUDA C++ source code, your task is to insert invariant assertion checks that validate functional equivalence during code transformation. These assertions will verify that mathematical and physical invariants hold before and after GPU computations while being compatible with GPU execution constraints.
    
    **Critical Requirements:**
    
    1. **Insert assertions ONLY on the host side** (CPU code), not inside device kernels
    
    2. **Place assertions at strategic validation points:**
       - After input validation (before cudaMemcpy host-to-device)
       - After output retrieval (after cudaMemcpy device-to-host)
       - Before any `return` statements
       - After kernel completion and synchronization
    
    3. **Surround all assertion blocks with protective comments** so future LLM refactoring tasks will not modify them:
       ```cpp
       // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
       // LASSI_INSTRUMENTATION - [Brief description of what is being checked]
       [your assertion code here]
       // <<< END LASSI-VERIFY ASSERTIONS <<<
       ```
    
    4. **Use ONLY standard library functions:**
       - `#include <cassert>` or `#include <assert.h>` for assert()
       - `#include <cmath>` or `#include <math.h>` for isfinite(), isnan(), isinf(), fabs()
       - Do NOT create or reference any custom assertion libraries
    
    5. **CRITICAL: Insert ONLY invariant assertions (Type A)**
       - Properties that remain true REGARDLESS of parallelization strategy
       - DO NOT assert specific block sizes (e.g., block.x == 16)
       - DO NOT assert specific grid sizes (e.g., grid.x == N/16)
       - DO NOT assert specific thread counts or execution patterns
       - ONLY assert general correctness constraints that survive refactoring
    
    6. **NEVER wrap assertions in conditional compilation flags**
       - DO NOT use #ifdef DEBUG, #ifndef NDEBUG, or any other preprocessor conditionals
       - Assertions MUST execute unconditionally (they can be disabled via -DNDEBUG at compile time)
       - Example of FORBIDDEN pattern:
         ```cpp
         #ifdef DEBUG
             assert(...);  // WRONG - DO NOT DO THIS
         #endif
         ```
       - Correct pattern:
         ```cpp
         // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
         // LASSI_INSTRUMENTATION - [description]
         assert(...);  // CORRECT - unconditional assertion
         // <<< END LASSI-VERIFY ASSERTIONS <<<
         ```
    
    7. **For large arrays (> 1 million elements), use sampling:**
       Instead of checking every element, sample strategic subsets:
       - First 1000 elements
       - Last 1000 elements
       - Random middle elements (if applicable)
       This reduces validation overhead while maintaining effectiveness.
    
    8. **Always validate final output arrays:**
       After the main computation completes and results are copied back to host,
       validate the output array for:
       - Finite values (no NaN/Inf)
       - Physical constraints (if applicable)
       - Boundary conditions preservation (if applicable)
    
    **Step-by-step process:**
    
    **Step 1: Analyze the source code**
    - Review the provided CODE ANALYSIS about this source code
    - Identify input arrays/pointers and their sizes
    - Identify output arrays/pointers and their sizes
    - Identify memory transfer points (cudaMemcpy H→D and D→H)
    - Determine the mathematical/physical domain (CFD, linear algebra, molecular dynamics, etc.)
    - Identify any symmetry, conservation, or invariant properties that should hold
    
    **Step 2: Select minimal relevant assertions**
    - Choose assertions from the catalog that validate correctness for THIS specific code
    - Focus on properties that will remain true regardless of how the code is refactored or optimized
    - Avoid redundant checks
    - AVOID implementation-specific checks (specific thread counts, block sizes, etc.)
    
    **Step 3: Insert assertion blocks**
    - Add assertion blocks with protective commenting at the identified validation points
    - For large arrays, use sampling instead of full validation
    - Ensure all loops over host arrays use correct bounds from the source code
    - Use variable names exactly as they appear in the source code
    
    **Step 4: Kernel parameter validation (general constraints only)**
    When validating kernel launch parameters, use ONLY general constraints:
    ✅ CORRECT (invariant constraints):
       - assert(block.x > 0 && block.y > 0 && "Block dimensions must be positive");
       - assert(block.x * block.y <= 1024 && "Block size exceeds max threads per block");
       - assert(grid.x > 0 && grid.y > 0 && "Grid dimensions must be positive");
       - assert(grid.x * block.x >= N && grid.y * block.y >= N && "Grid must cover domain");
    
    ❌ WRONG (implementation-specific):
       - assert(block.x == 16 && block.y == 16);  // DO NOT CHECK SPECIFIC VALUES
       - assert(grid.x == N/16);  // DO NOT CHECK SPECIFIC GRID SIZE
       - assert(threads == 256);  // DO NOT CHECK SPECIFIC THREAD COUNT
    
    **Step 5: Add required headers**
    - Add `#include <cassert>` if not present
    - Add `#include <cmath>` if not present (for isfinite, etc.)
    
    **Preserve all original code:**
    - Do not modify kernels, algorithmic logic, memory allocations, or performance optimizations
    - Do not remove or edit any existing code or comments
    - Only ADD assertion blocks with protective comments
    - Do not replace any original code with comment placeholders
    
    ---

    """

    content_prompt = content_prompt + prompts.get_codeknowledge("inv_asserts")
    
    content_prompt = content_prompt + """
    ---
        
    **Output Instructions:**
        
    Only output the complete, modified source file with inserted assertion blocks, ready to compile under `nvcc` or `hipcc`.

    Ensure the code is surrounded by three backtick characters: ```
        
    **Do NOT include:**
    - Explanatory commentary about what you changed
    - Markdown code fences with language specifiers (no ```cpp)
    - Descriptions of your reasoning
    - Summaries of insertions
        
    **DO include:**
    - The complete source file with all original code intact
    - Assertion blocks surrounded by protective comments
    - Any necessary additional #include directives at the top
    - All assertions using correct variable names and array bounds from the source
    - Sampling-based validation for arrays larger than 1 million elements
    - Final output array validation after main computation
        
    ---

    """

    content_prompt = content_prompt + " [BEGIN CODE ANALYSIS]\n" + code_analysis + "\n[/END CODE ANALYSIS]\n\n"
    content_prompt = content_prompt + " [BEGIN SOURCE CODE]\n" + source_code + "\n[/END SOURCE CODE]\n"
    
    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE INSERT - Invariant Assertians" + color.END + " -----\n")
    
    # print("\n\nDEBUG input parameter values:")
    # print("  llm_source: " + str(llm_source))
    # print("  model: " + str(model))
    # print("  num_ctx: " + str(num_ctx))
    # print("  system_prompt: " + str(system_prompt))
    # print("  content_prompt: " + str(content_prompt))
    # print("  display_response: " + str(display_response))

    temp = 0.2
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "lmstudio":
        response_text = lmstudio_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "cerebras":
        response_text = cerebras_llm(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    else:
        print("Invalid LLM source")
        
    print("\n\n----- " + color.BOLD + color.BLUE + "THE GENERATED CODE" + color.END + " -----\n")
    if display_response:
        print("(hidden)\n\n")
        display(Markdown(response_text))

    # Extract only the code block - 
    # Search for the ``` pattern in the text
    #   Regular expression pattern to match text between triple backticks 
    #   that represent code blocks within generated responses from the LLM.
    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    # print("RESPONSE TEXT -- ", response_text)
    # print("MATCH -- ", match)
    if match:
        extracted_code = match.group(1)
    else:
        extracted_code = ""
    
    return extracted_code