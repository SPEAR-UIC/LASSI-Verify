from IPython.display import display, Markdown
import re
import os
import json
import math
import subprocess
from lassi.format_colors import color
from lassi.prompt_dictionary import PromptDictionary
from lassi.llm_config import argo_llm, cerebras_llm, ollama_llm, ollama_llm_nostream

# Load prompt dictionary (to capture recent manual changes to script, if any)
prompts = PromptDictionary()


def code_insert_passthrough(source_code, llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=False):

    content_prompt = r"""
        You are preparing GPU-accelerated scientific code for Tier 2 empirical equivalence checking
        as part of the LASSI-Verify verification framework.

        Your task is to instrument the source code with **passthrough data blocks** at key observation
        points. These blocks capture computational state to stdout in a structured, labeled format.
        Both the source (SOURCE_verify) and every LASSI-generated refactored version will carry these
        blocks unaltered, enabling deterministic comparison of observable output at each observation
        point to verify that any refactored code preserves the computational behavior of the source.

        **CRITICAL PRESERVATION REQUIREMENT:**
        The source code already contains LASSI-VERIFY ASSERTIONS blocks inserted by Tier 1. You MUST
        preserve ALL existing blocks marked with:
            // >>> BEGIN LASSI-VERIFY ASSERTIONS ... >>>
            // <<< END LASSI-VERIFY ASSERTIONS <<<
        and ALL existing blocks marked with:
            // >>> BEGIN LASSI-VERIFY PASSTHROUGH ... >>>
            // <<< END LASSI-VERIFY PASSTHROUGH <<<
        Do NOT modify, remove, or restructure any of these existing blocks. Only ADD new passthrough
        blocks at additional observation points.

        **STEP 1: Identify the computing environment**
        - Programming language: C++ or Fortran
        - GPU API: CUDA (cudaMalloc/cudaMemcpy) or HIP (hipMalloc/hipMemcpy)
        - Appropriate memory copy function and DeviceToHost constant

        **STEP 2: Select observation points**
        Using the passthrough catalog guidance below, identify the key observation points where the
        source and any refactored code should produce identical output. Aim for 2-5 distinct code
        locations — a block placed inside a loop counts as ONE location even though it executes
        multiple times.

        - REQUIRED: at least one block capturing the primary output buffer at program completion
        - REQUIRED if the code contains an outer loop over configurations or parameters (e.g.,
          different segment sizes, grid levels, tile sizes, time steps): place the passthrough
          block INSIDE that loop so every configuration is captured independently.
          Use the SAME label each iteration — the LASSI-Verify checker compares loop iterations
          positionally and explicitly supports repeated labels. Do NOT hoist to outside the loop
          and capture only a single representative iteration.
        - OPTIONAL: additional intermediate checkpoints after major kernels or phase boundaries
        - Insert AFTER cudaDeviceSynchronize() (or equivalent) and AFTER D→H copy where applicable
        - Insert BEFORE any memory deallocations (cudaFree, hipFree, free)

        **STEP 3: Insert labeled passthrough blocks**

        For each observation point, insert a block with this EXACT structure. Label rules:
        - For a block at a single code location: use a unique, descriptive label (e.g., [final_solution])
        - For a block inside an outer configuration loop: use the SAME label every iteration
          (e.g., [post_kernel_output] repeated for segment_size=16, 32, 64, ...)
        - The label must be consistent in the C++ comment, the BEGIN printf, and the END printf
        - The LASSI-Verify checker handles repeated labels correctly by comparing occurrences positionally

        ```cpp
        // >>> BEGIN LASSI-VERIFY PASSTHROUGH [label] (do not edit or delete this block) >>>
        {
            // --- LASSI-VERIFY PASSTHROUGH: <brief description> ---
            <dtype>* lv_host_buf = (<dtype>*) malloc(<N> * sizeof(<dtype>));
            <memcpy_fn>(lv_host_buf, <device_ptr>, <N> * sizeof(<dtype>), <DeviceToHost>);

            <dtype> lv_min = lv_host_buf[0], lv_max = lv_host_buf[0];
            double lv_sum = 0.0;
            for (int lv_i = 0; lv_i < <N>; lv_i++) {
                if (lv_host_buf[lv_i] < lv_min) lv_min = lv_host_buf[lv_i];
                if (lv_host_buf[lv_i] > lv_max) lv_max = lv_host_buf[lv_i];
                lv_sum += (double)lv_host_buf[lv_i];
            }
            double lv_mean = lv_sum / <N>;

            printf("\n=== BEGIN LASSI-VERIFY PASSTHROUGH [label] ===\n");
            printf("Buffer: <buffer_name>\n");
            printf("Type: <dtype_name>\n");
            printf("Elements: %d\n", (int)<N>);
            printf("Statistics: min=%.17g, max=%.17g, mean=%.17g\n",
                   (double)lv_min, (double)lv_max, lv_mean);

            int lv_sample = (<N> < 10) ? (int)<N> : 10;
            printf("Sample_first10:");
            for (int lv_i = 0; lv_i < lv_sample; lv_i++)
                printf(" %.17g", (double)lv_host_buf[lv_i]);
            printf("\n");

            if (<N> > 10) {
                printf("Sample_last10:");
                for (int lv_i = (int)<N> - 10; lv_i < (int)<N>; lv_i++)
                    printf(" %.17g", (double)lv_host_buf[lv_i]);
                printf("\n");
            }
            printf("=== END LASSI-VERIFY PASSTHROUGH [label] ===\n");
            free(lv_host_buf);
        }
        // <<< END LASSI-VERIFY PASSTHROUGH <<<
        ```

        **Template substitution rules:**
        - Replace `[label]` with a descriptive unique label (e.g., `[final_solution]`, `[post_kernel_A]`)
          — the SAME label must appear in the C++ comment, the BEGIN printf, and the END printf
        - Replace `<dtype>` with the actual C++ data type (double, float, int, etc.)
        - Replace `<dtype_name>` with the string name of the data type (e.g., "double")
        - Replace `<N>` with the element count expression from the code
        - Replace `<device_ptr>` with the actual device pointer variable name
        - Replace `<buffer_name>` with the variable name of the device buffer
        - Replace `<memcpy_fn>` with: cudaMemcpy (CUDA) or hipMemcpy (HIP)
        - Replace `<DeviceToHost>` with: cudaMemcpyDeviceToHost (CUDA) or hipMemcpyDeviceToHost (HIP)
        - Use the `lv_` prefix for ALL passthrough-local variables to avoid name collisions
        - For float buffers: cast to double in printf calls (already shown as `(double)lv_host_buf[lv_i]`)
        - For integer buffers: use `%d` instead of `%.17g`, no cast needed

        **STEP 4: Add required headers if not present**
        - C++: `<cstdio>` / `<stdio.h>` for printf, `<cstdlib>` / `<stdlib.h>` for malloc/free

        **PRESERVATION REQUIREMENTS:**
        - Do NOT modify any algorithmic logic, loops, GPU kernels, or original comments
        - Do NOT remove or alter any existing LASSI-VERIFY ASSERTIONS or PASSTHROUGH blocks
        - Do NOT change performance-critical sections
        - ONLY add: new passthrough blocks and any minimally required headers

        **OUTPUT INSTRUCTIONS:**
        Always provide the complete and fully functional instrumented code without placeholders,
        comments, or references suggesting that parts of the original code should be included.
        Ensure every part of the instrumented code is explicitly written out.
        Do NOT use ellipsis (...), shorthand like "// [other kernels unchanged]",
        "// ... rest of code ...", "// same as before", or ANY comment or notation suggesting
        sections were omitted or skipped. Output ONLY the complete, modified source code ready
        to compile. Do not include explanatory text, commentary, or markdown formatting outside
        of a single code fence.

        ----

    """

    content_prompt = content_prompt + prompts.get_codeknowledge("passthrough_catalog")

    content_prompt = content_prompt + "\n\n----\n\n"
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
    if match:
        extracted_code = match.group(1)
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)  # Strip language specifier (e.g. cpp\n)
    else:
        extracted_code = ""

    # Sanity check: instrumentation only adds lines — same or fewer lines means the LLM
    # omitted sections via placeholder comments (e.g. "// ... [other kernels unchanged] ...").
    input_lines = len(source_code.splitlines())
    output_lines = len(extracted_code.splitlines())
    if extracted_code and output_lines <= input_lines:
        print(f"\n[WARNING] Passthrough instrumentation line count check FAILED: "
              f"input={input_lines} lines, output={output_lines} lines. "
              f"LLM likely omitted code via placeholder comments. Rejecting output.")
        return ""

    return extracted_code


def code_insert_instrumentation_from_template(base_code, template_instrumented_code,
                                               llm_source, model, num_ctx, system_prompt,
                                               display_response, verifyprompt=False):
    """
    Instrument base_code with LASSI-VERIFY blocks (Tier 1 assertions + Tier 2 passthrough)
    by replicating the instrumentation present in template_instrumented_code (SOURCE_verify).

    Used for post-pipeline verification of the zero-shot base LLM code: the same observation
    point labels as SOURCE_verify are inserted into the base LLM code at equivalent locations,
    enabling fair three-tier comparison against the same SOURCE_verify reference output
    (tier2_spec_output) that was used to verify all LASSI-generated codes.

    Unlike code_insert_asserts() + code_insert_passthrough() (which each make one LLM call
    to instrument SOURCE_verify from scratch), this function replicates EXISTING labeled blocks
    from a known-good template in a single LLM call, ensuring label identity across codes.

    Returns:
        str: instrumented base_code with all LASSI-VERIFY blocks, or "" on failure.
    """
    content_prompt = f"""You are given two CUDA programs that implement the same computation:

**PROGRAM A — REFERENCE (instrumented source-as-specification):**
```cuda
{template_instrumented_code}
```

**PROGRAM B — TARGET (base LLM refactored code, not yet instrumented):**
```cuda
{base_code}
```

**YOUR TASK:** Insert the SAME LASSI-VERIFY instrumentation blocks from PROGRAM A into PROGRAM B,
adapted to PROGRAM B's specific code structure.

**RULES:**

1. **Exact label preservation:** Use EXACTLY the same block labels as in PROGRAM A.
   - Every `LASSI-VERIFY ASSERTIONS [label]` block from A must appear in B with the same `[label]`.
   - Every `LASSI-VERIFY PASSTHROUGH [label]` block from A must appear in B with the same `[label]`.
   - Every `LASSI-VERIFY METAMORPHIC [label]` block from A must appear in B with the same `[label]`.

2. **Equivalent placement:** Place each block at the logically equivalent position in B.
   - If A checks a buffer after H→D transfer, find B's equivalent buffer after its H→D transfer.
   - If A checks output after D→H transfer, check B's equivalent output buffer.
   - If A instruments inside a loop, instrument at the equivalent loop location in B.
   - If B has reorganized, renamed, or merged buffers, check the closest functional equivalent.

3. **Statistical format (PASSTHROUGH blocks):** Use the same output format as A:
   ```
   Statistics: min=<val>, max=<val>, mean=<val>
   Sample_first10: <v0> <v1> ... <v9>
   Sample_last10: <vN-9> ... <vN>
   ```
   Use `%.17g` printf format for all floating-point values.

4. **Assertion format (ASSERTIONS blocks):** Use the same one-line PASS/FAIL format as A:
   ```
   printf("LASSI-VERIFY ASSERTIONS [<label>]: %s\\n", <condition> ? "PASS" : "FAIL"); fflush(stdout);
   ```
   Adapt the boolean condition to B's variable names and structure.

5. **Preserve existing blocks in B:** If B already has any LASSI-VERIFY blocks, keep them unchanged.

6. **Do not change B's compute logic:** Only add instrumentation — do not alter kernel calls,
   memory allocations, control flow, or any existing compute logic in PROGRAM B.

7. **Return only the fully instrumented PROGRAM B** in a ```cuda or ```cpp code fence.
   No explanation outside the code fence.
"""
    content_prompt += prompts.get_codeknowledge("passthrough_catalog")

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
        response_text = ""

    print("\n\n----- " + color.BOLD + color.BLUE + "BASE LLM TEMPLATE-GUIDED INSTRUMENTATION" + color.END + " -----\n")
    if display_response:
        print("(hidden)\n\n")
        display(Markdown(response_text))

    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)
    else:
        extracted_code = ""

    return extracted_code


def code_insert_asserts(source_code, llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=False):

    # Append critical instruction to repsond with complete code
    content_prompt = """
    You are a GPU code verification specialist.

    Your task is to instrument the provided CUDA C++ source code with **transformation-invariant
    assertion checks** — properties that must hold in the original code AND any correct refactored
    version, regardless of parallelization strategy, memory layout, or algorithmic optimization.

    These assertions establish **observation points** where the original and any transformed code
    must exhibit **observational equivalence** (identical behavior at program boundaries). They
    function as empirical auxiliary variables, analogous to auxiliary variables in Owicki-Gries
    parallel program verification: instrumentation added solely to observe computation properties
    without altering the computation itself.

    **Critical Requirements:**

    1. **Insert assertions ONLY on the host side** (CPU code), not inside device kernels
    2. **Place assertions at key observation points:**
       - At function entry: validate pointers, dimensions, and configuration parameters
       - Before host-to-device transfers (cudaMemcpy H→D): validate input data
       - After device-to-host transfers (cudaMemcpy D→H): validate output data
       - At function exit: validate final result properties

    3. **Surround all assertion blocks with protective comments** — these are CRITICAL:
       they ensure assertions survive through subsequent LLM-driven transformation steps
       without being removed or modified.

       **Two required block types — use BOTH:**

       **Type 1 — Per-observation-point check (always prints one PASS or FAIL line):**
    ```cpp
       // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
       // LASSI_INSTRUMENTATION - [Brief description of invariant being checked]
       {
           bool _t1_pass = (your_condition_here);
           printf("LASSI-VERIFY ASSERTIONS [label]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
       }
       // <<< END LASSI-VERIFY ASSERTIONS <<<
    ```

       **Type 2 — Completion sentinel (ONE per function, AFTER all Type 1 blocks, OUTSIDE all loops):**
    ```cpp
       // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit or delete this block) >>>
       // LASSI_INSTRUMENTATION - tier1_complete (sentinel — confirms all assertion blocks were reached)
       {
           printf("LASSI-VERIFY ASSERTIONS [tier1_complete]: PASS\n"); fflush(stdout);
       }
       // <<< END LASSI-VERIFY ASSERTIONS <<<
    ```

       **CRITICAL format rules:**
       - Type 1 blocks always print exactly **one line** per block: `PASS` or `FAIL`.
         This line is essential pipeline output — **do not remove it, do not move it inside an `if`**.
         The pipeline detects whether these blocks survive code transformation by reading this output.
       - The sentinel (`[tier1_complete]`) MUST be placed AFTER all Type 1 blocks and OUTSIDE any
         solver or compute loops. It tells the checker that instrumentation ran to completion.
       - **DO NOT place assertion blocks inside solver loops or compute loops.** Place them
         BEFORE the loop (validate inputs) and AFTER the loop (validate outputs/convergence).
         One print per block per program execution is the target — not one print per loop iteration.
       - `[label]` is a short unique identifier matching the LASSI_INSTRUMENTATION comment.
       - Do NOT use C `assert()` — it can be silenced by `-DNDEBUG` compiler flags.
       - Do NOT call `exit()` — the pipeline collects ALL assertion results before deciding.
       - Multiple assertions should each be their own separate Type 1 block with a unique label.

    4. **Use ONLY standard library functions:**
       - `#include <cstdio>` for `printf()` and `fflush()`
       - `#include <cmath>` or `#include <math.h>` for isfinite(), isnan(), isinf(), fabs()
       - Do NOT use `assert()` or include `<cassert>`
       - Do NOT create or reference any custom assertion libraries

    5. **Select minimal relevant assertions** from the provided catalog that match the code's
       computational characteristics — prioritize foundational invariants (Layer 1), then add
       domain-specific ones (Layer 2) where applicable. Avoid over-instrumentation.

    **Your Analysis Approach:**

    Rather than following a fixed procedure, analyze the code by answering these questions:

    - **What is the computational domain?** (e.g., linear algebra, CFD, molecular dynamics,
      iterative solver, stochastic methods) — this determines which domain invariants apply

    - **What properties must be preserved across ANY correct transformation?**
      - Input constraints: pointer validity, array bounds, value ranges, numerical validity
      - Output constraints: correctness bounds, conservation laws, physical limits
      - Mathematical laws: symmetry, conservation, monotonicity — but only where the
        original problem requires them, not as assumptions about this implementation

    - **Where are the natural observation points?** — function entry/exit and memory transfer
      boundaries are the primary candidates; avoid checking intermediate kernel state

    - **Which assertions are truly transformation-invariant?** — an assertion is valid only if
      it would hold in any correct implementation of this computation, not just this specific
      parallelization or optimization. When in doubt, do not insert the assertion.

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

        Always provide the complete and fully functional instrumented code without placeholders,
        comments, or references suggesting that parts of the original code should be included.
        Ensure every part of the instrumented code is explicitly written out.
        Do NOT use ellipsis (...), shorthand like "// [other kernels unchanged]",
        "// ... rest of code ...", "// same as before", or ANY comment or notation suggesting
        sections were omitted or skipped.

        Only output the complete, modified source file with inserted assertion blocks, ready to compile under `nvcc` or `hipcc`.

        **Do NOT include:**
        - Explanatory commentary about what you changed
        - Descriptions of your reasoning
        - Summaries of insertions

        **DO include:**
        - The complete source file with all original code intact
        - Assertion blocks surrounded by protective comments
        - Any necessary additional #include directives at the top
        - All assertions using correct variable names and array bounds from the source

        ---

    """

    content_prompt = content_prompt + " [BEGIN CODE ANALYSIS]\n" + code_analysis + "\n[/END CODE ANALYSIS]"
    content_prompt = content_prompt + " [BEGIN SOURCE CODE]\n" + source_code + "\n[/END SOURCE CODE]"
    
    
    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE INSERT - Invariant Assertions" + color.END + " -----\n")
    
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
        # Strip optional language specifier on the first line (e.g. "cpp\n" or "c++\n")
        # LLMs commonly emit ```cpp ... ``` fences; group(1) captures the specifier too.
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)
    else:
        extracted_code = ""

    # print("\nEXTRACTED CODE: ", extracted_code)  # DEBUG: uncomment to inspect extracted code block

    # Sanity check: instrumentation only adds lines — same or fewer lines means the LLM
    # omitted sections via placeholder comments (e.g. "// ... [other kernels unchanged] ...").
    input_lines = len(source_code.splitlines())
    output_lines = len(extracted_code.splitlines())
    if extracted_code and output_lines <= input_lines:
        print(f"\n[WARNING] Assertion instrumentation line count check FAILED: "
              f"input={input_lines} lines, output={output_lines} lines. "
              f"LLM likely omitted code via placeholder comments. Rejecting output.")
        return ""

    return extracted_code


def compare_passthrough_outputs(spec_stdout, generated_stdout,
                                tolerance=1e-6,
                                llm_source=None, model=None, num_ctx=None,
                                system_prompt=None, verifyprompt=False):
    """
    Tier 2 empirical equivalence check. Compare passthrough output from a generated
    code against the source-as-specification reference (SOURCE_verify stdout).

    Implements the source-as-specification premise: SOURCE_verify's observable behavior
    is the correctness criterion. A transformation is correct if it preserves this
    behavior; the framework makes no claim about whether the source itself is correct.

    Hybrid comparison approach:
    1. Deterministic layer (primary): parse labeled blocks, compare min/max/mean and
       sample values using numpy allclose-style mixed tolerance:
           |a - b| <= atol + rtol * max(|a|, |b|)
       where atol = |spec_max| * rtol (derived per-block from the spec's max statistic).
       This anchors the absolute tolerance to the field's actual dynamic range, preventing
       false positives from near-zero statistics (e.g., converged Jacobi mean ~1e-12).
       `sum` is intentionally excluded: it equals mean×N and accumulates per-element
       floating-point noise at large N, making it an unreliable correctness criterion.
    2. LLM-mediated layer (fallback): invoked when structured parsing fails or is
       incomplete (e.g., LLM did not emit expected format). Returns PASS/FAIL + rationale.

    Returns:
        verdict (str): "PASS" | "PASS (LLM)" | "FAIL" | "FAIL (LLM)" | "SKIP"
        diagnostic (dict): structured result for self-correction context and logs.
            Keys: verdict, method, tolerance, blocks_compared, failures, summary
    """
    # ------------------------------------------------------------------
    # Deterministic layer: extract and match labeled blocks by label
    # ------------------------------------------------------------------
    # Regex: match BEGIN...END pairs by label, capturing label and block body
    block_pattern = re.compile(
        r'=== BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\] ===\n(.*?)\n=== END LASSI-VERIFY PASSTHROUGH \[[^\]]+\] ===',
        re.DOTALL
    )
    # Regex for the Statistics line within a block.
    # Uses [^,\s]+ (non-comma, non-whitespace) so each capture group stops cleanly at
    # the comma delimiter and does not consume subsequent fields.
    # `sum` is optional: older instrumented codes may include it, but it is no longer
    # compared (sum = mean × N accumulates per-element FP noise at large N, making it
    # an unreliable correctness criterion for large grids).
    stats_pattern = re.compile(
        r'Statistics:\s*min=([^,\s]+),\s*max=([^,\s]+),\s*mean=([^,\s]+)'
    )
    # Regex for sample value lines
    sample_pattern = re.compile(
        r'Sample_(first10|last10):(.*)'
    )

    def extract_blocks(stdout_text):
        """Return dict mapping label → ordered list of block body strings.
        Collects all occurrences so blocks inside loops are not silently deduplicated
        by a dict comprehension that keeps only the last occurrence per label."""
        result = {}
        for m in block_pattern.finditer(stdout_text):
            result.setdefault(m.group(1), []).append(m.group(2))
        return result

    def parse_stats(block_body):
        """Parse Statistics line; return dict of {min, max, mean} as floats, or None.
        sum is intentionally excluded: it equals mean×N and accumulates per-element
        floating-point noise at large N, producing unreliable relative errors."""
        m = stats_pattern.search(block_body)
        if not m:
            return None
        try:
            return {
                'min':  float(m.group(1)),
                'max':  float(m.group(2)),
                'mean': float(m.group(3)),
            }
        except ValueError:
            return None

    def parse_samples(block_body):
        """Return list of floats for Sample_first10 and Sample_last10 lines."""
        samples = []
        for m in sample_pattern.finditer(block_body):
            try:
                vals = [float(v) for v in m.group(2).split()]
                samples.extend(vals)
            except ValueError:
                pass
        return samples

    def allclose_pass(a, b, rtol, atol):
        """numpy.allclose-style check: |a-b| <= atol + rtol*max(|a|,|b|).
        atol is anchored to the field's dynamic range (|spec_max| * rtol), so
        near-zero statistics (e.g., converged Jacobi mean ~1e-12) are compared
        against the actual field scale rather than against zero."""
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

    def abs_err(a, b):
        """Absolute error for failure reporting."""
        return abs(a - b)

    spec_blocks = extract_blocks(spec_stdout)
    gen_blocks  = extract_blocks(generated_stdout)

    # If no labeled blocks found in either output, skip Tier 2 gracefully
    if not spec_blocks and not gen_blocks:
        diagnostic = {
            "verdict":         "SKIP",
            "method":          "deterministic",
            "tolerance":       tolerance,
            "blocks_compared": 0,
            "failures":        [],
            "max_relative_error": None,
            "summary":         "SKIP: No LASSI-VERIFY PASSTHROUGH blocks found in either output. "
                               "Passthrough insertion may have failed or blocks were removed during refactoring."
        }
        print("\n----- LASSI-Verify Tier 2: SKIP (no passthrough blocks found) -----")
        return "SKIP", diagnostic

    failures = []
    blocks_compared = 0
    max_rel_err = 0.0   # largest relative error across all blocks and stats (for Table 2)

    for label, spec_bodies in spec_blocks.items():
        gen_bodies = gen_blocks.get(label, [])
        n_spec = len(spec_bodies)
        n_gen  = len(gen_bodies)
        # When a label appears more than once (e.g., inside a loop), tag each occurrence
        # with "#N" so failures are distinguishable in self-correction context.
        multi = n_spec > 1 or n_gen > 1

        if n_gen == 0:
            # Label present in spec but entirely absent from generated output — failure
            failures.append({
                "label":     label,
                "stat":      "block_missing",
                "spec":      "(present)",
                "generated": "(absent)",
                "rel_err":   float('inf'),
            })
            continue

        # Any spec occurrences that exceed the generated count are also block_missing
        for i in range(n_gen, n_spec):
            failure_label = f"{label}#{i+1}" if multi else label
            failures.append({
                "label":     failure_label,
                "stat":      "block_missing",
                "spec":      "(present)",
                "generated": "(absent)",
                "rel_err":   float('inf'),
            })

        # Compare each occurrence positionally (occurrence i in spec vs occurrence i in gen)
        for i, (spec_body, gen_body) in enumerate(zip(spec_bodies, gen_bodies)):
            failure_label = f"{label}#{i+1}" if multi else label
            blocks_compared += 1

            spec_stats = parse_stats(spec_body)
            gen_stats  = parse_stats(gen_body)

            if spec_stats is None or gen_stats is None:
                # Cannot parse statistics — fall through to LLM fallback for this block
                # Mark as unparseable so fallback is triggered below
                failures.append({
                    "label":     failure_label,
                    "stat":      "parse_error",
                    "spec":      spec_body[:200],
                    "generated": gen_body[:200],
                    "rel_err":   float('inf'),
                })
                continue

            # Derive atol from the spec block's max statistic so that near-zero statistics
            # (e.g., converged mean ~1e-12) are compared against the field's actual dynamic
            # range rather than against zero. atol = |spec_max| * tolerance.
            spec_max_val = abs(spec_stats.get('max', 0.0))
            atol = spec_max_val * tolerance

            # Compare each statistic using allclose: |a-b| <= atol + rtol*max(|a|,|b|)
            for stat_name, spec_val in spec_stats.items():
                gen_val = gen_stats.get(stat_name)
                if gen_val is None:
                    max_rel_err = max(max_rel_err, float('inf'))
                    failures.append({
                        "label":     failure_label,
                        "stat":      stat_name,
                        "spec":      spec_val,
                        "generated": "(missing)",
                        "abs_err":   float('inf'),
                        "atol":      atol,
                    })
                    continue
                _denom = max(abs(spec_val), atol)
                _rel = abs_err(spec_val, gen_val) / _denom if _denom > 0 else (0.0 if abs_err(spec_val, gen_val) == 0.0 else float('inf'))
                max_rel_err = max(max_rel_err, _rel)
                if not allclose_pass(spec_val, gen_val, tolerance, atol):
                    failures.append({
                        "label":     failure_label,
                        "stat":      stat_name,
                        "spec":      spec_val,
                        "generated": gen_val,
                        "abs_err":   abs_err(spec_val, gen_val),
                        "atol":      atol,
                    })

            # Compare sample values element-wise with the same allclose tolerance
            spec_samples = parse_samples(spec_body)
            gen_samples  = parse_samples(gen_body)
            if spec_samples and gen_samples:
                for idx, (sv, gv) in enumerate(zip(spec_samples, gen_samples)):
                    _denom = max(abs(sv), atol)
                    _rel = abs_err(sv, gv) / _denom if _denom > 0 else (0.0 if abs_err(sv, gv) == 0.0 else float('inf'))
                    max_rel_err = max(max_rel_err, _rel)
                    if not allclose_pass(sv, gv, tolerance, atol):
                        failures.append({
                            "label":     failure_label,
                            "stat":      f"sample[{idx}]",
                            "spec":      sv,
                            "generated": gv,
                            "abs_err":   abs_err(sv, gv),
                            "atol":      atol,
                        })

    # Check for blocks present in generated output but not in spec (informational only, not a failure)
    extra_labels = set(gen_blocks.keys()) - set(spec_blocks.keys())

    # ------------------------------------------------------------------
    # Decide verdict from deterministic layer
    # ------------------------------------------------------------------
    # Trigger LLM fallback if any parse_error failures exist (structured output was unusable)
    has_parse_errors = any(f["stat"] in ("parse_error",) for f in failures)
    numeric_failures = [f for f in failures if f["stat"] not in ("parse_error",)]

    if has_parse_errors and llm_source is not None:
        # LLM fallback: send raw outputs to LLM for judgment
        fallback_prompt = prompts.get_codeknowledge("passthrough_compare_fallback")
        fallback_prompt += (
            f"\n\n--- PROGRAM A OUTPUT (SOURCE_verify — source-as-specification reference) ---\n"
            f"{spec_stdout[:8000]}\n\n"
            f"--- PROGRAM B OUTPUT (generated/refactored code) ---\n"
            f"{generated_stdout[:8000]}\n"
        )
        temp = 0.0
        if llm_source == "argo":
            llm_response = argo_llm(model, system_prompt, fallback_prompt, temp, verifyprompt)
        elif llm_source == "ollama":
            llm_response = ollama_llm_nostream(model, num_ctx, system_prompt, fallback_prompt, temp, verifyprompt)
        elif llm_source == "cerebras":
            llm_response = cerebras_llm(model, num_ctx, system_prompt, fallback_prompt, temp, verifyprompt)
        else:
            llm_response = ""

        llm_upper = llm_response.upper()
        if "FAIL" in llm_upper:
            llm_verdict = "FAIL (LLM)"
        elif "PASS" in llm_upper:
            llm_verdict = "PASS (LLM)"
        else:
            llm_verdict = "SKIP"  # LLM gave ambiguous response

        summary = (
            f"Tier 2 LLM fallback (deterministic parsing failed for {len([f for f in failures if f['stat']=='parse_error'])} block(s)). "
            f"LLM verdict: {llm_verdict}. LLM rationale: {llm_response[:300]}"
        )
        diagnostic = {
            "verdict":         llm_verdict,
            "method":          "llm_fallback",
            "tolerance":       tolerance,
            "blocks_compared": blocks_compared,
            "failures":        failures,
            "max_relative_error": max_rel_err if max_rel_err != float('inf') else None,
            "summary":         summary,
        }
        return llm_verdict, diagnostic

    # Pure deterministic verdict
    if numeric_failures or any(f["stat"] == "block_missing" for f in failures):
        fail_summary_parts = []
        for f in failures:
            if f["stat"] == "block_missing":
                fail_summary_parts.append(f"  [block_missing] label='{f['label']}': block absent in generated output")
            else:
                fail_summary_parts.append(
                    f"  [{f['label']}] {f['stat']}: spec={f['spec']}, generated={f['generated']}, "
                    f"abs_err={f['abs_err']:.3e} (atol={f.get('atol', tolerance):.3e}, rtol={tolerance:.1e})"
                )
        summary = (
            f"FAIL: {len(failures)} discrepancy(ies) found across {blocks_compared} block(s) compared.\n"
            + "\n".join(fail_summary_parts)
        )
        if extra_labels:
            summary += f"\n  (Note: {len(extra_labels)} label(s) in generated output not present in spec: {extra_labels})"
        verdict = "FAIL"
    else:
        summary = (
            f"PASS: {blocks_compared} passthrough block(s) matched within allclose tolerance "
            f"(rtol={tolerance:.1e}, atol=|spec_max|*rtol per block)."
        )
        if extra_labels:
            summary += f" ({len(extra_labels)} extra label(s) in generated output ignored: {extra_labels})"
        verdict = "PASS"

    diagnostic = {
        "verdict":         verdict,
        "method":          "deterministic",
        "tolerance":       tolerance,
        "blocks_compared": blocks_compared,
        "failures":        failures,
        "max_relative_error": max_rel_err if max_rel_err != float('inf') else None,
        "summary":         summary,
    }
    return verdict, diagnostic


# =============================================================================
# LABEL EXTRACTION HELPERS (used by Table 2 per-attempt logging)
# =============================================================================

def extract_tier1_labels(code_text):
    """Return comma-separated list of Tier 1 assertion labels present in the code source.
    Excludes the sentinel label 'tier1_complete'. Used to document what was checkable
    per attempt and detect cases where the LLM accidentally stripped assertion blocks."""
    labels = re.findall(r'LASSI-VERIFY ASSERTIONS \[([^\]]+)\]', code_text)
    labels = [l for l in labels if l != _TIER1_SENTINEL_LABEL]
    # Deduplicate while preserving order
    seen = set()
    unique = [l for l in labels if not (l in seen or seen.add(l))]
    return ",".join(unique)


def extract_tier2_labels(code_text):
    """Return comma-separated list of Tier 2 passthrough block labels present in the code source.
    Used to document observation points per attempt and detect stripped blocks."""
    labels = re.findall(r'BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\]', code_text)
    # Deduplicate while preserving order
    seen = set()
    unique = [l for l in labels if not (l in seen or seen.add(l))]
    return ",".join(unique)


# =============================================================================
# TIER 1: DETERMINISTIC ASSERTION CHECK
# =============================================================================

_TIER1_SENTINEL_LABEL = "tier1_complete"  # Completion sentinel label (see code_insert_asserts prompt)


def check_tier1_assertions(stdout):
    """
    Tier 1: Deterministic check of invariant assertion results from generated code stdout.

    Two output formats are supported:
      - FAIL-only (current): each Type 1 block prints only on FAIL; a 'tier1_complete' sentinel
        block prints PASS unconditionally to confirm instrumentation ran to completion.
      - Legacy (backward compat): each block prints PASS or FAIL unconditionally.

    Parsing logic:
      - Sentinel ([tier1_complete]: PASS) → stripped from label lists; recorded as sentinel_found.
      - Any FAIL label(s) → verdict FAIL (regardless of sentinel).
      - Sentinel present + no FAILs → verdict PASS (FAIL-only format: all checks passed silently).
      - Sentinel absent + PASS labels present → verdict PASS (legacy format).
      - No assertion output at all → verdict SKIP (no instrumentation or code crashed early).

    All assertion results are collected before returning — giving the LLM full visibility
    into every failing invariant, not just the first one.

    Returns:
        verdict (str): "PASS", "FAIL", or "SKIP"
        diagnostic (dict): {verdict, failing_labels, passing_labels, pass_count, fail_count, summary}
    """
    assertion_pattern = re.compile(
        r'LASSI-VERIFY ASSERTIONS \[([^\]]+)\]:\s*(PASS|FAIL)'
    )

    passing_labels = []
    failing_labels = []
    for m in assertion_pattern.finditer(stdout):
        label, result = m.group(1), m.group(2)
        if result == "FAIL":
            failing_labels.append(label)
        else:
            passing_labels.append(label)

    # Extract and remove sentinel from label lists
    sentinel_found = _TIER1_SENTINEL_LABEL in passing_labels
    passing_labels = [l for l in passing_labels if l != _TIER1_SENTINEL_LABEL]
    failing_labels = [l for l in failing_labels if l != _TIER1_SENTINEL_LABEL]

    total = len(passing_labels) + len(failing_labels)

    if not sentinel_found and total == 0:
        return "SKIP", {
            "verdict": "SKIP", "failing_labels": [], "passing_labels": [],
            "pass_count": 0, "fail_count": 0,
            "summary": "SKIP: No LASSI-VERIFY ASSERTIONS output found in stdout.",
        }

    if failing_labels:
        summary = (
            f"FAIL: {len(failing_labels)} assertion(s) violated: "
            + ", ".join(f"[{l}]" for l in failing_labels)
            + (". Passing: " + ", ".join(f"[{l}]" for l in passing_labels) if passing_labels else "")
        )
        verdict = "FAIL"
    elif sentinel_found:
        # FAIL-only format: sentinel present, no FAILs → all checks passed silently
        summary = "PASS: All assertion(s) passed (sentinel confirmed, no failures detected)."
        verdict = "PASS"
    else:
        # Legacy format: PASS lines found, no sentinel, no FAILs
        summary = f"PASS: All {total} assertion(s) passed."
        verdict = "PASS"

    return verdict, {
        "verdict": verdict,
        "failing_labels": failing_labels,
        "passing_labels": passing_labels,
        "pass_count": len(passing_labels),
        "fail_count": len(failing_labels),
        "summary": summary,
    }


# =============================================================================
# TIER 3: METAMORPHIC TESTING
# =============================================================================

def generate_metamorphic_contract(source_code, llm_source, model, num_ctx, system_prompt,
                                   display_response, verifyprompt=False):
    """
    Tier 3: Generate a metamorphic contract for the given source code.

    Calls the LLM with the metamorphic catalog to select 2-4 applicable relations.
    Returns a contract dict with a 'relations' key, or {} on failure.

    Called once in Stage 0 after SOURCE_verify is constructed (after Tier 1 + Tier 2
    insertion). The contract drives all subsequent Tier 3 work.

    Maps to: Benton's Relational Hoare Logic (relational reasoning over two executions
    under related inputs). The LLM selects validation MRs from mathematical/physical
    semantics, not implementation details.
    """
    content_prompt = (
        "You are preparing GPU-accelerated scientific code for Tier 3 metamorphic testing "
        "as part of the LASSI-Verify verification framework.\n\n"
        "Your task is to analyze the source code and select 2-4 applicable metamorphic "
        "relations from the catalog below. Return your selection as a single JSON block.\n\n"
        "----\n\n"
    )
    content_prompt += prompts.get_codeknowledge("metamorphic_catalog")
    content_prompt += "\n\n----\n\n"
    content_prompt += "[BEGIN SOURCE CODE]\n" + source_code + "\n[END SOURCE CODE]\n\n"
    content_prompt += (
        "Following the Instructions section in the catalog above, analyze the source code "
        "and return a JSON contract. Return ONLY the JSON block — no explanatory text "
        "before or after it.\n\n"
        "```json\n"
        "{ \"relations\": [ ... ] }\n"
        "```"
    )

    print("\n----- " + color.BOLD + color.PURPLE +
          "STARTING Tier 3: Generating Metamorphic Contract" + color.END + " -----\n")

    temp = 0.2
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "cerebras":
        response_text = cerebras_llm(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    else:
        print("Invalid LLM source for generate_metamorphic_contract")
        return {}

    if display_response:
        display(Markdown(response_text))

    # Extract JSON from response: try ```json ... ``` first, then bare { ... }
    contract = {}
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        json_str = json_match.group(0) if json_match else ""

    if json_str:
        try:
            contract = json.loads(json_str)
            # Initialize source_status for each relation (populated in Stage 0 filter run)
            for rel in contract.get("relations", []):
                rel["source_status"] = None
            n = len(contract.get("relations", []))
            print(f"\n----- Tier 3 contract generated: {n} relation(s) selected -----")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n----- Tier 3 contract JSON parse failed: {e} — Tier 3 disabled -----")
            contract = {}
    else:
        print("\n----- Tier 3 contract: no JSON found in LLM response — Tier 3 disabled -----")

    return contract


def code_insert_metamorphic_variant(source_code, relation, llm_source, model, num_ctx,
                                     system_prompt, display_response,
                                     template_code=None, verifyprompt=False):
    """
    Tier 3: Produce a metamorphic variant of source_code for one relation.

    Calls the LLM to insert the input transformation and METAMORPHIC output blocks at the
    designated observation points. Preserves all existing LASSI-VERIFY ASSERTIONS and
    PASSTHROUGH blocks without modification.

    Returns extracted code string (same pattern as code_insert_asserts / code_insert_passthrough).
    Returns "" on failure (causes that relation to be marked SKIP in run_metamorphic_variants).

    Called once per relation per code being tested:
      - Once for SOURCE_verify in Stage 0 (source filter)
      - Once per active relation per generated code in the main loop
    """
    relation_name = relation.get("name", "unknown")
    relation_type = relation.get("type", "UNKNOWN")
    factor        = relation.get("factor")
    input_vars    = relation.get("input_variables", [])
    obs_points    = relation.get("observation_points", [])
    description   = relation.get("description", "")
    expected_rel  = relation.get("expected_relationship", "")
    tolerance     = relation.get("tolerance", 1e-5)

    # --- Relation specification block (human-readable, for the LLM) ---
    rel_spec = (
        f"Relation name: {relation_name}\n"
        f"Relation type: {relation_type}\n"
        f"Description: {description}\n"
    )
    if factor is not None:
        rel_spec += f"Factor (alpha): {factor}\n"
    if input_vars:
        rel_spec += f"Input variables to transform: {', '.join(input_vars)}\n"
    if obs_points:
        rel_spec += (
            f"Observation point labels (must match existing PASSTHROUGH block labels): "
            f"{', '.join(obs_points)}\n"
        )
    rel_spec += f"Expected relationship: {expected_rel}\n"
    rel_spec += f"Tolerance: {tolerance}\n"

    # --- METAMORPHIC block format guidance (relation-type specific) ---
    first_label = obs_points[0] if obs_points else "output"
    if relation_type in ("SCALING", "SYMMETRY"):
        block_format = (
            f"**METAMORPHIC output block format** "
            f"(insert AFTER the final D→H transfer and cudaDeviceSynchronize, "
            f"BEFORE any cudaFree / free calls):\n\n"
            f"```\n"
            f"=== BEGIN LASSI-VERIFY METAMORPHIC [{first_label}] ===\n"
            f"Relation: {relation_name}\n"
            f"Factor: {factor if factor is not None else 'N/A'}\n"
            f"Buffer: <buffer_name>\n"
            f"Elements: <N>\n"
            f"Statistics: min=%.17g, max=%.17g, mean=%.17g\n"
            f"Sample_first10: %.17g %.17g ...  (up to 10 values, space-separated)\n"
            f"=== END LASSI-VERIFY METAMORPHIC [{first_label}] ===\n"
            f"```\n\n"
            f"Print this block using printf with %.17g format. The label MUST exactly match "
            f"the corresponding PASSTHROUGH block label for the same observation point "
            f"(e.g., if the PASSTHROUGH label is `[final_solution]`, the METAMORPHIC label "
            f"must also be `[final_solution]`).\n"
        )
    elif relation_type in ("PERMUTATIVE", "ROTATION"):
        block_format = (
            f"**METAMORPHIC output block format** "
            f"(insert AFTER the final D→H transfer and cudaDeviceSynchronize, "
            f"BEFORE any cudaFree / free calls):\n\n"
            f"```\n"
            f"=== BEGIN LASSI-VERIFY METAMORPHIC [{first_label}] ===\n"
            f"Relation: {relation_name}\n"
            f"Permutation: swap\n"
            f"Swap_idx1: <integer index of first swapped element>\n"
            f"Swap_idx2: <integer index of second swapped element, != Swap_idx1>\n"
            f"Buffer: <buffer_name>\n"
            f"Elements: <N>\n"
            f"Statistics: min=%.17g, max=%.17g, mean=%.17g\n"
            f"Sample_first10: %.17g %.17g ...  (up to 10 values, space-separated)\n"
            f"Sample_at_idx1: %.17g\n"
            f"Sample_at_idx2: %.17g\n"
            f"=== END LASSI-VERIFY METAMORPHIC [{first_label}] ===\n"
            f"```\n\n"
            f"Print this block using printf with %.17g format. "
            f"Swap_idx1 and Swap_idx2 are the LITERAL integer indices used for the swap "
            f"(e.g., 'Swap_idx1: 3' and 'Swap_idx2: N-4'). "
            f"Sample_at_idx1 is the output buffer value at Swap_idx1 (i.e., lv_host_buf[swap_idx1]); "
            f"Sample_at_idx2 is the output buffer value at Swap_idx2 (i.e., lv_host_buf[swap_idx2]). "
            f"For equivariance checking to fire, choose indices from the first-10 range [1,9] OR "
            f"the last-10 range [N-9, N-2] so the checker can find the reference Y values. "
            f"The label MUST exactly match the corresponding PASSTHROUGH block label.\n"
        )
    elif relation_type == "CONSERVATION":
        block_format = (
            "**METAMORPHIC output block format** (print BEFORE and AFTER the kernel, "
            "using 'Sum:' field):\n\n"
            "```\n"
            "=== BEGIN LASSI-VERIFY METAMORPHIC [before] ===\n"
            "Relation: conservation\nBuffer: <buffer_name>\nElements: <N>\nSum: %.17g\n"
            "=== END LASSI-VERIFY METAMORPHIC [before] ===\n"
            "... kernel execution ...\n"
            "=== BEGIN LASSI-VERIFY METAMORPHIC [after] ===\n"
            "Relation: conservation\nBuffer: <buffer_name>\nElements: <N>\nSum: %.17g\n"
            "=== END LASSI-VERIFY METAMORPHIC [after] ===\n"
            "```\n"
            "The labels MUST be EXACTLY `[before]` and `[after]` — do not substitute observation "
            "point names or any other text.\n"
        )
    elif relation_type == "MONOTONICITY":
        block_format = (
            "**METAMORPHIC output block format** (print per-iteration inside the solver loop, "
            "using 'Residual:' field):\n\n"
            "```\n"
            "=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_<k>] ===\n"
            "Relation: monotonicity\nIteration: <k>\nResidual: %.17g\n"
            "=== END LASSI-VERIFY METAMORPHIC [residual_iter_<k>] ===\n"
            "```\n"
        )
    else:
        block_format = (
            "**METAMORPHIC output block format**: Use labeled blocks with "
            "'=== BEGIN LASSI-VERIFY METAMORPHIC [label] ===' and "
            "'=== END LASSI-VERIFY METAMORPHIC [label] ===' delimiters.\n"
        )

    # --- Transformation instruction (relation-type specific) ---
    if relation_type in ("SCALING", "SYMMETRY") and input_vars:
        transform_instruction = (
            f"**Step 1 — Apply the input transformation:**\n"
            f"Insert the transformation code AFTER host array initialization but BEFORE "
            f"the first host-to-device memory copy (cudaMemcpy H→D or hipMemcpy H→D).\n\n"
            f"For {relation_type} with factor={factor}: multiply the values of "
            f"({', '.join(input_vars)}) by {factor}.\n"
            f"Example:\n"
            f"    const double alpha = {factor};\n"
            f"    for (int i = 0; i < N; i++) {input_vars[0]}[i] *= alpha;\n"
            f"Scale ALL input arrays listed in 'Input variables to transform'.\n\n"
            f"**Step 2 — Insert METAMORPHIC output blocks:**\n"
            f"After the final device-to-host transfer, insert METAMORPHIC blocks capturing "
            f"the transformed output at the observation point labels listed above.\n"
        )
    elif relation_type == "PERMUTATIVE" and input_vars:
        transform_instruction = (
            f"**Step 1 — Apply a deterministic swap to the input:**\n"
            f"Insert the swap AFTER host array initialization but BEFORE the first H→D copy.\n"
            f"Choose two fixed indices. For best equivariance coverage, pick one from each boundary:\n"
            f"  idx1 from [2, 5] (captured in Sample_first10)\n"
            f"  idx2 from [N-5, N-2] (captured in Sample_last10, where N is the array length)\n"
            f"Avoid index 0 and index N-1 (often fixed boundary elements). "
            f"If the array is small (N ≤ 20), both indices may come from [1, 9].\n\n"
            f"    const int swap_idx1 = 3;                // from first10 range\n"
            f"    const int swap_idx2 = (int)N - 4;       // from last10 range\n"
            f"    std::swap({input_vars[0]}[swap_idx1], {input_vars[0]}[swap_idx2]);\n"
            f"    // (repeat for each input array listed in 'Input variables to transform')\n\n"
            f"**Step 2 — Insert METAMORPHIC output blocks:**\n"
            f"After the final D→H transfer, insert METAMORPHIC blocks at the observation points above.\n"
            f"Print Swap_idx1/Swap_idx2 as literal integers. Also print Sample_at_idx1 and\n"
            f"Sample_at_idx2 as the output buffer values at those positions:\n"
            f"    printf(\"Sample_at_idx1: %.17g\\n\", (double)lv_host_buf[swap_idx1]);\n"
            f"    printf(\"Sample_at_idx2: %.17g\\n\", (double)lv_host_buf[swap_idx2]);\n"
        )
    elif relation_type == "CONSERVATION":
        transform_instruction = (
            "No input transformation is needed — the relation checks that the computation "
            "preserves a total sum across execution.\n\n"
            "**Step 1 — Insert a [before] block** BEFORE the computation kernel runs:\n"
            "  - Compute the sum of the primary INPUT buffer (the data being reduced/processed).\n"
            "  - Copy d_in (or equivalent) to a temporary host buffer, loop to accumulate the sum,\n"
            "    then print the block with the EXACT label `[before]`.\n\n"
            "**Step 2 — Insert an [after] block** AFTER the kernel and D→H transfer:\n"
            "  - Compute the sum of the primary OUTPUT buffer (the result of the computation).\n"
            "  - Use the already-copied host output buffer (or issue a fresh D→H copy), loop to\n"
            "    accumulate the sum, then print the block with the EXACT label `[after]`.\n\n"
            "The checker verifies: |sum_after − sum_before| / |sum_before| < tolerance.\n"
            "Both labels MUST be exactly `[before]` and `[after]` — not observation point names.\n"
        )
    elif relation_type == "MONOTONICITY":
        transform_instruction = (
            "**Step 1 — Add per-iteration residual output** inside the solver iteration loop.\n"
            "After each convergence update, compute the current residual and print a METAMORPHIC block.\n"
            "No input transformation is needed.\n"
        )
    else:
        transform_instruction = (
            f"**Step 1 — Apply the transformation** appropriate for {relation_type}.\n"
            f"**Step 2 — Insert METAMORPHIC output blocks** at the observation points above.\n"
        )

    content_prompt = (
        "You are preparing a metamorphic variant of a GPU-accelerated source code for "
        "Tier 3 empirical verification as part of the LASSI-Verify framework.\n\n"
        "Your task: modify the source code below to produce a variant that tests the "
        "following metamorphic relation.\n\n"
        "=== METAMORPHIC RELATION ===\n"
        + rel_spec +
        "===========================\n\n"
        "**CRITICAL PRESERVATION REQUIREMENT:**\n"
        "The source code already contains LASSI-VERIFY ASSERTIONS and LASSI-VERIFY "
        "PASSTHROUGH blocks. You MUST preserve ALL existing blocks marked with:\n"
        "    // >>> BEGIN LASSI-VERIFY ASSERTIONS ... >>>\n"
        "    // <<< END LASSI-VERIFY ASSERTIONS <<<\n"
        "and ALL existing blocks marked with:\n"
        "    // >>> BEGIN LASSI-VERIFY PASSTHROUGH ... >>>\n"
        "    // <<< END LASSI-VERIFY PASSTHROUGH <<<\n"
        "Do NOT modify, remove, or restructure any of these blocks.\n\n"
        "**MODIFICATION INSTRUCTIONS:**\n"
        + transform_instruction + "\n"
        + block_format + "\n"
        "**OUTPUT INSTRUCTIONS:**\n"
        "Always provide the complete and fully functional instrumented code without placeholders, "
        "comments, or references suggesting that parts of the original code should be included. "
        "Ensure every part of the instrumented code is explicitly written out. "
        "Do NOT use ellipsis (...), shorthand like '// [other kernels unchanged]', "
        "'// ... rest of code ...', '// same as before', or ANY comment or notation suggesting "
        "sections were omitted or skipped. "
        "Output ONLY the complete, modified source code ready to compile. "
        "Do NOT include explanatory text or commentary outside of a single code fence. "
        "The output must compile with nvcc or hipcc.\n\n"
        + (
            "[REFERENCE TEMPLATE — how this metamorphic relation was inserted into the SOURCE_verify version]\n"
            "The following shows the validated strategy used to insert this same relation into the "
            "reference (SOURCE_verify) version of this code. Use it as a guide for WHERE and HOW "
            "to apply the input transformation and insert the METAMORPHIC output block. "
            "Do NOT copy it verbatim — the source code below may be structured differently. "
            "Replicate the strategy, not the exact lines.\n\n"
            + template_code +
            "\n[END REFERENCE TEMPLATE]\n\n"
            if template_code else ""
        )
        + "[BEGIN SOURCE CODE]\n" + source_code + "\n[END SOURCE CODE]\n"
    )

    print("\n----- " + color.BOLD + color.PURPLE +
          f"STARTING Tier 3: Inserting metamorphic variant ({relation_name})" +
          color.END + " -----\n")

    temp = 0.2
    if llm_source == "argo":
        response_text = argo_llm(model, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "ollama":
        response_text = ollama_llm_nostream(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    elif llm_source == "cerebras":
        response_text = cerebras_llm(model, num_ctx, system_prompt, content_prompt, temp, verifyprompt)
    else:
        print("Invalid LLM source for code_insert_metamorphic_variant")
        return ""

    if display_response:
        display(Markdown(response_text))

    # Extract code block (same pattern as code_insert_asserts / code_insert_passthrough)
    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)
    else:
        extracted_code = ""

    # Sanity check: instrumentation only adds lines — same or fewer lines means the LLM
    # omitted sections via placeholder comments (e.g. "// ... [other kernels unchanged] ...").
    input_lines = len(source_code.splitlines())
    output_lines = len(extracted_code.splitlines())
    if extracted_code and output_lines <= input_lines:
        print(f"\n[WARNING] Metamorphic variant line count check FAILED: "
              f"input={input_lines} lines, output={output_lines} lines. "
              f"LLM likely omitted code via placeholder comments. Rejecting output.")
        return ""

    return extracted_code


def check_metamorphic_relation(variant_stdout, original_passthrough_stdout, relation,
                                tolerance=None):
    """
    Tier 3: Deterministic check of a metamorphic relation between variant and original outputs.

    Parses METAMORPHIC blocks from variant_stdout and PASSTHROUGH blocks from
    original_passthrough_stdout (the generated code's own Tier 2 passthrough output),
    then verifies the expected mathematical relationship.

    Phase 1 implements: SCALING, SYMMETRY, CONSERVATION, MONOTONICITY.
    PERMUTATIVE and ROTATION require element-level array access and are deferred to Phase 3.

    Uses allclose-style tolerance throughout:
        |a - b| <= atol + rtol * max(|a|, |b|)
        atol = |expected_value| * rtol   (anchors tolerance to the field's dynamic range)

    NaN/Inf guard: if either the original or variant value is non-finite, SKIP that check
    (per catalog Design Principle 6 — NaN/Inf guard is mandatory).

    Returns:
        verdict (str): "PASS", "FAIL", or "SKIP"
        diagnostic (dict): {verdict, relation_name, factor, blocks_compared, failures, summary}
    """
    relation_name = relation.get("name", "unknown")
    relation_type = relation.get("type", "UNKNOWN")
    factor        = relation.get("factor", 1.0) if relation.get("factor") is not None else 1.0
    obs_points    = relation.get("observation_points", [])
    rtol          = tolerance if tolerance is not None else relation.get("tolerance", 1e-5)

    # --- Shared parsers (same patterns as compare_passthrough_outputs) ---
    meta_block_pattern = re.compile(
        r'=== BEGIN LASSI-VERIFY METAMORPHIC \[([^\]]+)\] ===\n(.*?)\n=== END LASSI-VERIFY METAMORPHIC \[[^\]]+\] ===',
        re.DOTALL
    )
    pass_block_pattern = re.compile(
        r'=== BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\] ===\n(.*?)\n=== END LASSI-VERIFY PASSTHROUGH \[[^\]]+\] ===',
        re.DOTALL
    )
    stats_pattern  = re.compile(r'Statistics:\s*min=([^,\s]+),\s*max=([^,\s]+),\s*mean=([^,\s]+)')
    sample_pattern = re.compile(r'Sample_(first10|last10):(.*)')
    sum_pattern    = re.compile(r'Sum:\s*([^\s]+)')
    residual_patt  = re.compile(r'Residual:\s*([^\s]+)')
    iter_patt      = re.compile(r'Iteration:\s*(\d+)')

    def extract_meta_blocks(text):
        """Return dict mapping label → ordered list of block body strings (all occurrences)."""
        result = {}
        for m in meta_block_pattern.finditer(text):
            result.setdefault(m.group(1), []).append(m.group(2))
        return result

    def extract_pass_blocks(text):
        """Return dict mapping label → ordered list of block body strings (all occurrences)."""
        result = {}
        for m in pass_block_pattern.finditer(text):
            result.setdefault(m.group(1), []).append(m.group(2))
        return result

    def parse_stats(body):
        m = stats_pattern.search(body)
        if not m:
            return None
        try:
            return {'min': float(m.group(1)), 'max': float(m.group(2)), 'mean': float(m.group(3))}
        except ValueError:
            return None

    def parse_samples(body):
        vals = []
        for m in sample_pattern.finditer(body):
            try:
                vals.extend(float(v) for v in m.group(2).split())
            except ValueError:
                pass
        return vals

    def allclose(a, b, rtol, atol):
        """numpy.allclose-style: |a-b| <= atol + rtol*max(|a|,|b|)"""
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

    def both_finite(a, b):
        return math.isfinite(a) and math.isfinite(b)

    # ==========================================================================
    # SCALING / SYMMETRY: check Z_stat ≈ factor * Y_stat
    # Both types share the same quantitative ratio check; SYMMETRY uses factor=-1.
    # ==========================================================================
    if relation_type in ("SCALING", "SYMMETRY"):
        meta_blocks = extract_meta_blocks(variant_stdout)
        pass_blocks = extract_pass_blocks(original_passthrough_stdout)

        if not meta_blocks and not pass_blocks:
            return "SKIP", {
                "verdict": "SKIP", "relation_name": relation_name, "factor": factor,
                "blocks_compared": 0, "failures": [],
                "summary": "SKIP: No METAMORPHIC or PASSTHROUGH blocks found in outputs.",
            }

        failures = []
        blocks_compared = 0

        # Iterate over designated observation points; fall back to all METAMORPHIC labels
        labels_to_check = obs_points if obs_points else list(meta_blocks.keys())
        for label in labels_to_check:
            if label not in meta_blocks:
                failures.append({"label": label, "stat": "metamorphic_block_missing",
                                  "detail": "METAMORPHIC block absent in variant output"})
                continue
            if label not in pass_blocks:
                failures.append({"label": label, "stat": "passthrough_block_missing",
                                  "detail": "PASSTHROUGH block absent in original output"})
                continue

            # Each value in the dict is now a list of block bodies (all occurrences).
            # Compare positionally so loop-internal blocks are not silently deduplicated.
            pass_bodies = pass_blocks[label]
            meta_bodies = meta_blocks[label]
            n_pairs = min(len(pass_bodies), len(meta_bodies))
            multi = max(len(pass_bodies), len(meta_bodies)) > 1

            for i in range(n_pairs):
                failure_label = f"{label}#{i+1}" if multi else label
                orig_stats = parse_stats(pass_bodies[i])
                var_stats  = parse_stats(meta_bodies[i])
                blocks_compared += 1

                if orig_stats is None or var_stats is None:
                    failures.append({"label": failure_label, "stat": "parse_error",
                                      "detail": "Could not parse Statistics line in one or both blocks"})
                    continue

                # NaN/Inf guard: skip block if any statistic is non-finite
                if not all(math.isfinite(v) for v in list(orig_stats.values()) + list(var_stats.values())):
                    continue

                # atol anchored to |factor * orig_max| * rtol so near-zero fields
                # are judged against the field's actual dynamic range.
                orig_max_val = abs(orig_stats.get('max', 0.0))
                atol = abs(factor) * orig_max_val * rtol

                for stat_name, orig_val in orig_stats.items():
                    var_val = var_stats.get(stat_name)
                    if var_val is None:
                        failures.append({"label": failure_label, "stat": stat_name,
                                          "orig": orig_val, "variant": "(missing)"})
                        continue
                    # When factor < 0, min and max of the field swap under the transformation:
                    # the minimum of the scaled field is factor * orig_max (not factor * orig_min).
                    # Mean is always linear regardless of factor sign.
                    if factor < 0 and stat_name == 'min':
                        expected_var = factor * orig_stats.get('max', orig_val)
                    elif factor < 0 and stat_name == 'max':
                        expected_var = factor * orig_stats.get('min', orig_val)
                    else:
                        expected_var = factor * orig_val
                    if not both_finite(var_val, expected_var):
                        continue  # NaN/Inf guard
                    if not allclose(var_val, expected_var, rtol, atol):
                        failures.append({
                            "label": failure_label, "stat": stat_name,
                            "orig": orig_val, "expected_variant": expected_var,
                            "actual_variant": var_val,
                            "abs_err": abs(var_val - expected_var),
                            "atol": atol, "rtol": rtol,
                        })

                # Element-wise sample comparison
                orig_samples = parse_samples(pass_bodies[i])
                var_samples  = parse_samples(meta_bodies[i])
                for idx, (ov, vv) in enumerate(zip(orig_samples, var_samples)):
                    if not both_finite(ov, vv):
                        continue
                    expected_vv = factor * ov
                    if not both_finite(vv, expected_vv):
                        continue
                    if not allclose(vv, expected_vv, rtol, atol):
                        failures.append({
                            "label": failure_label, "stat": f"sample[{idx}]",
                            "orig": ov, "expected_variant": expected_vv,
                            "actual_variant": vv,
                            "abs_err": abs(vv - expected_vv),
                            "atol": atol, "rtol": rtol,
                        })

        if failures:
            lines = []
            for f in failures:
                if "detail" in f:
                    lines.append(f"  [{f['label']}] {f['stat']}: {f['detail']}")
                else:
                    lines.append(
                        f"  [{f['label']}] {f['stat']}: "
                        f"orig={f.get('orig')}, expected_variant={f.get('expected_variant')}, "
                        f"actual_variant={f.get('actual_variant')}, "
                        f"abs_err={f.get('abs_err', '?'):.3e}"
                    )
            summary = (
                f"FAIL: {len(failures)} discrepancy(ies) across {blocks_compared} block(s). "
                f"Relation: {relation_name} (factor={factor}, rtol={rtol:.1e})\n"
                + "\n".join(lines)
            )
            verdict = "FAIL"
        else:
            summary = (
                f"PASS: {blocks_compared} block(s) satisfied {relation_type} relation "
                f"(factor={factor}, rtol={rtol:.1e}, atol=|factor*orig_max|*rtol per block)."
            )
            verdict = "PASS"

        return verdict, {
            "verdict": verdict, "relation_name": relation_name, "factor": factor,
            "blocks_compared": blocks_compared, "failures": failures, "summary": summary,
        }

    # ==========================================================================
    # CONSERVATION: check |sum_after - sum_before| / |sum_before| < rtol
    # The variant binary prints before/after Sum values in paired METAMORPHIC blocks.
    # ==========================================================================
    elif relation_type == "CONSERVATION":
        meta_blocks = extract_meta_blocks(variant_stdout)

        # Flatten all before/after occurrences into ordered lists so blocks inside
        # loops are not silently deduplicated. extract_meta_blocks returns dict-of-lists.
        before_bodies = []
        after_bodies  = []
        for k, bodies in meta_blocks.items():
            if 'before' in k.lower():
                before_bodies.extend(bodies)
            elif 'after' in k.lower():
                after_bodies.extend(bodies)

        if not before_bodies or not after_bodies:
            return "SKIP", {
                "verdict": "SKIP", "relation_name": relation_name, "factor": None,
                "blocks_compared": 0, "failures": [],
                "summary": "SKIP: Could not find before/after METAMORPHIC blocks for conservation check.",
            }

        failures = []
        blocks_compared = 0

        # Broadcast: if there is one canonical before and many afters (or vice versa),
        # pair the single before/after with every corresponding after/before so that
        # conservation is checked for every loop iteration, not just the last one.
        if len(before_bodies) == 1 and len(after_bodies) > 1:
            pairs = [(before_bodies[0], av) for av in after_bodies]
        elif len(after_bodies) == 1 and len(before_bodies) > 1:
            pairs = [(bv, after_bodies[0]) for bv in before_bodies]
        else:
            pairs = list(zip(before_bodies, after_bodies))

        for pair_idx, (bv, av) in enumerate(pairs):
            bm = sum_pattern.search(bv)
            am = sum_pattern.search(av)
            if not bm or not am:
                continue

            try:
                s_before = float(bm.group(1))
                s_after  = float(am.group(1))
            except ValueError:
                continue

            if not (math.isfinite(s_before) and math.isfinite(s_after)):
                continue  # NaN/Inf guard

            blocks_compared += 1
            pair_label = f"before/after#{pair_idx+1}" if len(pairs) > 1 else "before/after"
            if abs(s_before) < 1e-300:
                # Near-zero baseline: absolute check
                if abs(s_after - s_before) > rtol:
                    failures.append({"label": pair_label, "stat": "sum",
                                      "before": s_before, "after": s_after,
                                      "abs_err": abs(s_after - s_before)})
            else:
                rel_err = abs(s_after - s_before) / abs(s_before)
                if rel_err > rtol:
                    failures.append({"label": pair_label, "stat": "sum_relative_change",
                                      "before": s_before, "after": s_after,
                                      "rel_err": rel_err, "rtol": rtol})

        if failures:
            summary = (
                f"FAIL: Conservation violated for {len(failures)} block pair(s). "
                f"Relation: {relation_name} (rtol={rtol:.1e})\n"
                + "\n".join(
                    f"  [{f['label']}]: before={f['before']:.6g}, after={f['after']:.6g}, "
                    f"err={f.get('rel_err', f.get('abs_err', '?')):.3e}"
                    for f in failures
                )
            )
            verdict = "FAIL"
        else:
            summary = f"PASS: Conservation satisfied across {blocks_compared} block pair(s) (rtol={rtol:.1e})."
            verdict = "PASS"

        return verdict, {
            "verdict": verdict, "relation_name": relation_name, "factor": None,
            "blocks_compared": blocks_compared, "failures": failures, "summary": summary,
        }

    # ==========================================================================
    # MONOTONICITY: check residual[k] >= residual[k+1] - rtol*|residual[k]|
    # The variant binary prints per-iteration residuals in METAMORPHIC blocks.
    # NOTE: Only applies to iterative solvers with monotone convergence guarantees.
    # CG and non-monotone methods should NOT use this relation (catalog Section 6).
    # ==========================================================================
    elif relation_type == "MONOTONICITY":
        meta_blocks = extract_meta_blocks(variant_stdout)

        residuals = []
        for label, bodies in meta_blocks.items():
            for body in bodies:
                im = iter_patt.search(body)
                rm = residual_patt.search(body)
                if im and rm:
                    try:
                        residuals.append((int(im.group(1)), float(rm.group(1))))
                    except ValueError:
                        pass

        if len(residuals) < 2:
            return "SKIP", {
                "verdict": "SKIP", "relation_name": relation_name, "factor": None,
                "blocks_compared": 0, "failures": [],
                "summary": "SKIP: Fewer than 2 residual values found for monotonicity check.",
            }

        residuals.sort(key=lambda x: x[0])
        failures = []
        for i in range(len(residuals) - 1):
            it_curr, r_curr = residuals[i]
            it_next, r_next = residuals[i + 1]
            if not (math.isfinite(r_curr) and math.isfinite(r_next)):
                continue  # NaN/Inf guard
            # Allow small numerical noise: r[k] >= r[k+1] - rtol * |r[k]|
            if r_curr < r_next - rtol * abs(r_curr):
                failures.append({"label": f"iter_{it_curr}_to_{it_next}",
                                  "residual_k": r_curr, "residual_k1": r_next,
                                  "increase": r_next - r_curr})

        if failures:
            summary = (
                f"FAIL: Non-monotone residual at {len(failures)} transition(s). "
                f"Relation: {relation_name} (rtol={rtol:.1e})\n"
                + "\n".join(
                    f"  iter {f['label']}: r[k]={f['residual_k']:.6g}, "
                    f"r[k+1]={f['residual_k1']:.6g} (increase={f['increase']:.3e})"
                    for f in failures
                )
            )
            verdict = "FAIL"
        else:
            summary = (
                f"PASS: Residual monotone non-increasing across {len(residuals)} "
                f"iteration(s) (rtol={rtol:.1e})."
            )
            verdict = "PASS"

        return verdict, {
            "verdict": verdict, "relation_name": relation_name, "factor": None,
            "blocks_compared": len(residuals), "failures": failures, "summary": summary,
        }

    # ==========================================================================
    # PERMUTATIVE / ROTATION:
    # Step A — Aggregate invariance: under any relabeling/permutation, aggregate
    #   stats (min, max, mean) of the output are invariant (factor = 1.0).
    # Step B — Element equivariance: if Swap_idx1/Swap_idx2 and Sample_at_idx1/
    #   Sample_at_idx2 are present in the METAMORPHIC block, check
    #   Z[idx1] ≈ Y[idx2] and Z[idx2] ≈ Y[idx1].
    #   Z values come from Sample_at_idx1/idx2 (printed by variant at swap positions).
    #   Y values come from passthrough Sample_first10/last10, using Elements field
    #   to map last10 indices. Equivariance is skipped for indices outside both
    #   captured ranges (no failure reported — just not checkable).
    # ==========================================================================
    elif relation_type in ("PERMUTATIVE", "ROTATION"):
        # Shared parsers already defined above; add swap-index and targeted-sample patterns
        swap_idx1_patt    = re.compile(r'Swap_idx1:\s*(\d+)')
        swap_idx2_patt    = re.compile(r'Swap_idx2:\s*(\d+)')
        sample_at1_patt   = re.compile(r'Sample_at_idx1:\s*(\S+)')
        sample_at2_patt   = re.compile(r'Sample_at_idx2:\s*(\S+)')
        elements_patt_perm= re.compile(r'Elements:\s*(\d+)')

        def lookup_y_value(pass_body, target_idx, n_elements):
            """Return Y[target_idx] from passthrough Sample_first10/last10, or None if not captured."""
            first10_vals, last10_vals = [], []
            for m in re.finditer(r'Sample_first10:(.*)', pass_body):
                try:
                    first10_vals = [float(v) for v in m.group(1).split()]
                except ValueError:
                    pass
            for m in re.finditer(r'Sample_last10:(.*)', pass_body):
                try:
                    last10_vals = [float(v) for v in m.group(1).split()]
                except ValueError:
                    pass
            if target_idx < len(first10_vals):
                return first10_vals[target_idx]
            if n_elements > 0 and last10_vals:
                last10_start = n_elements - len(last10_vals)
                offset = target_idx - last10_start
                if 0 <= offset < len(last10_vals):
                    return last10_vals[offset]
            return None

        meta_blocks = extract_meta_blocks(variant_stdout)
        pass_blocks = extract_pass_blocks(original_passthrough_stdout)

        if not meta_blocks and not pass_blocks:
            return "SKIP", {
                "verdict": "SKIP", "relation_name": relation_name, "factor": 1.0,
                "blocks_compared": 0, "failures": [],
                "summary": "SKIP: No METAMORPHIC or PASSTHROUGH blocks found in outputs.",
            }

        failures = []
        blocks_compared = 0
        equivariance_ran = False
        # Factor is always 1.0 for PERMUTATIVE/ROTATION (invariance)
        perm_factor = 1.0

        labels_to_check = obs_points if obs_points else list(meta_blocks.keys())
        for label in labels_to_check:
            if label not in meta_blocks:
                failures.append({"label": label, "stat": "metamorphic_block_missing",
                                  "detail": "METAMORPHIC block absent in variant output"})
                continue
            if label not in pass_blocks:
                failures.append({"label": label, "stat": "passthrough_block_missing",
                                  "detail": "PASSTHROUGH block absent in original output"})
                continue

            # Each value in the dict is now a list of block bodies (all occurrences).
            # Compare positionally so loop-internal blocks are not silently deduplicated.
            pass_bodies = pass_blocks[label]
            meta_bodies = meta_blocks[label]
            n_pairs = min(len(pass_bodies), len(meta_bodies))
            multi = max(len(pass_bodies), len(meta_bodies)) > 1

            for i in range(n_pairs):
                failure_label = f"{label}#{i+1}" if multi else label
                orig_stats = parse_stats(pass_bodies[i])
                var_stats  = parse_stats(meta_bodies[i])
                blocks_compared += 1

                if orig_stats is None or var_stats is None:
                    failures.append({"label": failure_label, "stat": "parse_error",
                                      "detail": "Could not parse Statistics line in one or both blocks"})
                    continue

                # NaN/Inf guard
                if not all(math.isfinite(v) for v in list(orig_stats.values()) + list(var_stats.values())):
                    continue

                # Step A: invariance check — each aggregate stat of Z ≈ Y (factor=1.0)
                orig_max_val = abs(orig_stats.get('max', 0.0))
                atol = orig_max_val * rtol
                for stat_name, orig_val in orig_stats.items():
                    var_val = var_stats.get(stat_name)
                    if var_val is None:
                        failures.append({"label": failure_label, "stat": stat_name,
                                          "orig": orig_val, "variant": "(missing)"})
                        continue
                    if not both_finite(var_val, orig_val):
                        continue
                    if not allclose(var_val, orig_val, rtol, atol):
                        failures.append({
                            "label": failure_label, "stat": stat_name,
                            "orig": orig_val, "expected_variant": orig_val,
                            "actual_variant": var_val,
                            "abs_err": abs(var_val - orig_val),
                            "atol": atol, "rtol": rtol,
                        })

                # Step B: element equivariance — use Sample_at_idx1/idx2 from METAMORPHIC block
                # for Z values; look up Y values from passthrough first10/last10 using Elements field.
                meta_body = meta_bodies[i]
                m1 = swap_idx1_patt.search(meta_body)
                m2 = swap_idx2_patt.search(meta_body)
                mat1 = sample_at1_patt.search(meta_body)
                mat2 = sample_at2_patt.search(meta_body)
                if m1 and m2:
                    equivariance_ran = True
                    try:
                        idx1 = int(m1.group(1))
                        idx2 = int(m2.group(1))
                    except ValueError:
                        idx1, idx2 = -1, -1

                    # Determine N for last10 offset computation
                    n_elem = 0
                    me = elements_patt_perm.search(meta_body)
                    if me:
                        try:
                            n_elem = int(me.group(1))
                        except ValueError:
                            pass

                    # Get Z[idx1] and Z[idx2]: prefer Sample_at_idx fields; fall back to Sample_first10
                    z1, z2 = None, None
                    if mat1 and mat2:
                        try:
                            z1 = float(mat1.group(1))
                            z2 = float(mat2.group(1))
                        except ValueError:
                            pass
                    if z1 is None or z2 is None:
                        # Legacy fallback: read Z from concatenated first10/last10 samples
                        var_samples = parse_samples(meta_body)
                        if 0 <= idx1 < len(var_samples):
                            z1 = var_samples[idx1]
                        if 0 <= idx2 < len(var_samples):
                            z2 = var_samples[idx2]

                    # Get Y[idx1] and Y[idx2] from passthrough (first10 or last10)
                    y1 = lookup_y_value(pass_bodies[i], idx1, n_elem)
                    y2 = lookup_y_value(pass_bodies[i], idx2, n_elem)

                    if z1 is not None and z2 is not None and y1 is not None and y2 is not None:
                        # Z[idx1] should ≈ Y[idx2]
                        if both_finite(z1, y2) and not allclose(z1, y2, rtol, atol):
                            failures.append({
                                "label": failure_label, "stat": f"equivariance[Z[{idx1}]≈Y[{idx2}]]",
                                "orig": y2, "expected_variant": y2, "actual_variant": z1,
                                "abs_err": abs(z1 - y2), "atol": atol, "rtol": rtol,
                            })
                        # Z[idx2] should ≈ Y[idx1]
                        if both_finite(z2, y1) and not allclose(z2, y1, rtol, atol):
                            failures.append({
                                "label": failure_label, "stat": f"equivariance[Z[{idx2}]≈Y[{idx1}]]",
                                "orig": y1, "expected_variant": y1, "actual_variant": z2,
                                "abs_err": abs(z2 - y1), "atol": atol, "rtol": rtol,
                            })
                    # If y1/y2 not found in passthrough: equivariance silently skipped for this label
                    # (indices outside captured first10/last10 range — invariance still checked above)

        if failures:
            lines = []
            for f in failures:
                if "detail" in f:
                    lines.append(f"  [{f['label']}] {f['stat']}: {f['detail']}")
                else:
                    lines.append(
                        f"  [{f['label']}] {f['stat']}: "
                        f"orig={f.get('orig')}, expected_variant={f.get('expected_variant')}, "
                        f"actual_variant={f.get('actual_variant')}, "
                        f"abs_err={f.get('abs_err', '?'):.3e}"
                    )
            check_desc = "invariance + equivariance" if equivariance_ran else "invariance only"
            summary = (
                f"FAIL: {len(failures)} discrepancy(ies) across {blocks_compared} block(s). "
                f"Relation: {relation_name} ({check_desc}, rtol={rtol:.1e})\n"
                + "\n".join(lines)
            )
            verdict = "FAIL"
        else:
            check_desc = "invariance + equivariance" if equivariance_ran else "invariance only"
            summary = (
                f"PASS: {blocks_compared} block(s) satisfied {relation_type} relation "
                f"({check_desc}, rtol={rtol:.1e}, atol=|orig_max|*rtol per block)."
            )
            verdict = "PASS"

        return verdict, {
            "verdict": verdict, "relation_name": relation_name, "factor": perm_factor,
            "blocks_compared": blocks_compared, "failures": failures, "summary": summary,
        }

    # ==========================================================================
    # Other relation types: SKIP (not yet implemented).
    # ==========================================================================
    else:
        return "SKIP", {
            "verdict": "SKIP", "relation_name": relation_name, "factor": factor,
            "blocks_compared": 0, "failures": [],
            "summary": (
                f"SKIP: Relation type '{relation_type}' not yet implemented."
            ),
        }


def run_metamorphic_variants(source_code, active_relations, contract,
                              original_passthrough_stdout,
                              llm_source, model, num_ctx, system_prompt,
                              display_response,
                              directory, file_name, code_compiler, code_compiler_kwds,
                              build_script, make_script, code_build_script,
                              exe_input_parameters,
                              experiment_data_directory="generated_codes",
                              code_version=None,
                              verifyprompt=False):
    """
    Tier 3: Orchestrate metamorphic testing for one code (fail-fast variant loop).

    For each active relation (ordered cheapest-first, as arranged by the LLM contract):
      1. Generate a metamorphic variant (code_insert_metamorphic_variant)
      2. Write variant to disk as generated_codes/{file_name}_mt_{relation_name}.cu
      3. Compile with lassi.code_manager.compile_code
      4. Execute with subprocess (no power measurement — Tier 3 must not contaminate timing)
      5. Check with check_metamorphic_relation
      6. Stop at first FAIL (fail-fast)

    Updates relation["source_status"] in-place on each relation dict so Stage 0 can
    filter active relations:
        "PASS"  → relation is active (applicable to this computation)
        "FAIL"  → relation is inactive (filtered out — source fails it, so it's inapplicable)
        "SKIP"  → inactive (variant insertion, compile, or execute failed)

    Called from Stage 0 (source filter, active_relations = all contract relations) and
    from the main loop (active_relations = already-filtered tier3_active_relations).

    Variant execution is lightweight (subprocess.run, no power monitoring, no wait time)
    since Tier 3 uses separate binaries outside the timed performance runs.

    Returns:
        overall_verdict (str): "PASS" | "FAIL" | "SKIP"
        results_list (list): per-relation result dicts
        first_failure_diagnostic (dict): diagnostic from first FAIL, or {}
    """
    import lassi.code_manager  # local import to avoid circular dependencies

    # Infer target language from code_compiler
    target_lang = "HIP" if "hipcc" in code_compiler.lower() else "CUDA"

    # Variant source files are written to experiment_data_directory (same folder as
    # generated codes for this run, e.g. generated_codes/experiment/<custom_folder>/).
    variant_dir = experiment_data_directory
    os.makedirs(variant_dir, exist_ok=True)

    results = []
    first_failure_diagnostic = {}
    any_ran = False  # True if at least one variant reached the execute+check step

    for relation in active_relations:
        relation_name = relation.get("name", "unknown")
        relation_type = relation.get("type", "UNKNOWN")
        print(f"\n----- Tier 3: testing relation '{relation_name}' ({relation_type}) -----")

        # Skip null-factor SCALING/SYMMETRY (spatial reflection etc. require element-wise
        # comparison across the full array — deferred to Phase 3).
        if relation.get("factor") is None and relation_type in ("SCALING", "SYMMETRY"):
            print(f"  [Tier 3] '{relation_name}' has null factor — spatial {relation_type} "
                  f"requires Phase 3 element-wise comparison; SKIP")
            relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "null factor: spatial SYMMETRY/SCALING deferred to Phase 3"})
            continue

        # --- Determine variant file paths ---
        # Stage 0 (code_version=None): write as _source — the validated SOURCE_verify reference.
        # Main loop (code_version=N): write as _vN — versioned per generated code.
        source_variant_path = f"{variant_dir}/{file_name}_mt_{relation_name}_source.cu"
        if code_version is None:
            variant_path = source_variant_path
        else:
            variant_path = f"{variant_dir}/{file_name}_mt_{relation_name}_v{code_version}.cu"

        # --- Step 1: Generate metamorphic variant code ---
        # Template: always read from the _source file (Stage 0 validated reference).
        # During Stage 0 the _source file doesn't exist yet → template_code=None (LLM figures it out).
        # In the main loop the _source file is already on disk → LLM replicates the same strategy.
        template_code = None
        if os.path.exists(source_variant_path):
            try:
                with open(source_variant_path, 'r') as _tf:
                    template_code = _tf.read()
            except OSError:
                template_code = None

        variant_code = code_insert_metamorphic_variant(
            source_code, relation,
            llm_source, model, num_ctx, system_prompt,
            display_response,
            template_code=template_code,
            verifyprompt=verifyprompt
        )
        if not variant_code:
            print(f"  [Tier 3] Variant insertion FAILED for '{relation_name}' — SKIP")
            relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "variant insertion failed"})
            continue

        # --- Step 2: Write variant to disk ---
        with open(variant_path, 'w') as vf:
            vf.write(variant_code)
        print(f"  [Tier 3] Variant written to {variant_path}")

        # --- Step 3: Compile ---
        compile_err, variant_exe = lassi.code_manager.compile_code(
            variant_path, target_lang, code_compiler, code_compiler_kwds
        )
        if compile_err:
            print(f"  [Tier 3] Compile FAILED for '{relation_name}' — SKIP")
            relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "compile failed",
                             "compile_error": compile_err[:300]})
            continue

        # --- Step 4: Execute (lightweight — no power measurement, no wait time) ---
        # Variants run from the binary's directory to match execute_code behavior.
        any_ran = True
        try:
            exe_abs = os.path.abspath(variant_exe)
            exe_dir = os.path.dirname(exe_abs)
            exe_base = os.path.basename(exe_abs)
            result = subprocess.run(
                ["./" + exe_base] + list(exe_input_parameters),
                cwd=exe_dir,
                capture_output=True, text=True, timeout=120
            )
            variant_stdout = result.stdout
            print(f"\n----- < BEGIN: STANDARD OUTPUT FROM Tier 3 VARIANT EXECUTION > -----\n")
            print(variant_stdout)
            print(f"----- < END: Tier 3 Variant Output > -----\n")
            if result.returncode != 0:
                print(f"  [Tier 3] Execution warning: return code {result.returncode}")
                if result.stderr:
                    print(f"  stderr: {result.stderr[:400]}")
        except subprocess.TimeoutExpired:
            print(f"  [Tier 3] Execution TIMEOUT for '{relation_name}' — SKIP")
            relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "execution timeout"})
            continue
        except Exception as e:
            print(f"  [Tier 3] Execution ERROR for '{relation_name}': {e} — SKIP")
            relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": f"execution error: {e}"})
            continue

        # --- Step 5: Check metamorphic relation ---
        tolerance = relation.get("tolerance", 1e-5)
        verdict, diagnostic = check_metamorphic_relation(
            variant_stdout, original_passthrough_stdout, relation, tolerance
        )
        # Only update source_status during Stage 0 (code_version is None).
        # In the main loop, source_status was frozen at Stage 0 and must not be overwritten
        # by per-generated-code verdicts — doing so would corrupt the contract JSON record.
        if code_version is None:
            relation["source_status"] = verdict  # "PASS", "FAIL", or "SKIP"
        results.append({"name": relation_name, "verdict": verdict, "diagnostic": diagnostic})

        print(f"\n----- Tier 3 [{relation_name}]: {verdict} -----")
        if diagnostic.get("summary"):
            print(diagnostic["summary"])

        # Fail-fast: stop at first FAIL
        if verdict == "FAIL":
            first_failure_diagnostic = diagnostic
            break

    # --- Determine overall verdict ---
    verdicts = [r["verdict"] for r in results]
    if "FAIL" in verdicts:
        overall_verdict = "FAIL"
    elif not any_ran or not verdicts:
        overall_verdict = "SKIP"
    elif "PASS" in verdicts:
        overall_verdict = "PASS"
    else:
        overall_verdict = "SKIP"  # All relations skipped

    return overall_verdict, results, first_failure_diagnostic


def strip_verify_blocks(source_code):
    """
    Remove all LASSI-VERIFY ASSERTIONS and LASSI-VERIFY PASSTHROUGH blocks from a
    C++ source string. Returns uninstrumented code for clean performance measurement.

    Block delimiter format (from Tier 1 / Tier 2 instrumentation):
      Opening: // >>> BEGIN LASSI-VERIFY ASSERTIONS ...   >>>
               // >>> BEGIN LASSI-VERIFY PASSTHROUGH ... >>>
      Closing: // <<< END LASSI-VERIFY ASSERTIONS <<<
               // <<< END LASSI-VERIFY PASSTHROUGH <<<
    """
    # Match opening: // >>> BEGIN LASSI-VERIFY (ASSERTIONS|PASSTHROUGH) ... \n
    # Match closing:  // <<< END LASSI-VERIFY (ASSERTIONS|PASSTHROUGH) ...  \n
    # The closing pattern intentionally does NOT require a specific trailing arrow sequence
    # (<<<  vs  >>>) because LLMs occasionally generate the wrong closing arrows.
    # The [^\n]* absorbs any trailing characters (including <<< or >>> variants).
    pattern = re.compile(
        r'// >>> BEGIN LASSI-VERIFY (?:ASSERTIONS|PASSTHROUGH)[^\n]*\n'
        r'.*?'
        r'// <<< END LASSI-VERIFY (?:ASSERTIONS|PASSTHROUGH)[^\n]*\n?',
        re.DOTALL
    )
    return pattern.sub('', source_code)