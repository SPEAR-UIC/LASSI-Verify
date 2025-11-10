from IPython.display import display, Markdown
import re
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary
from lassi.llm_config import argo_llm, cerebras_llm, ollama_llm, ollama_llm_nostream

# Load prompt dictionary (to capture recent manual changes to script, if any)
prompts = PromptDictionary()

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
    
    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE INSERT - Invarient Assertians" + color.END + " -----\n")
    
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