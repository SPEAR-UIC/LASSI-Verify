from IPython.display import display, Markdown
import re
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
        Using the passthrough catalog guidance below, identify 2-5 observation points where the
        source and any refactored code should produce identical output:
        - REQUIRED: at least one block capturing the primary output buffer at program completion
        - OPTIONAL: intermediate checkpoints after major kernels or at iteration boundaries
        - Insert AFTER cudaDeviceSynchronize() (or equivalent) and AFTER D→H copy where applicable
        - Insert BEFORE any memory deallocations (cudaFree, hipFree, free)

        **STEP 3: Insert labeled passthrough blocks**

        For each observation point, insert a block with this EXACT structure. The label must be
        unique, descriptive, and consistent in the BEGIN and END markers:

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
        Output ONLY the complete, modified source code ready to compile. Do not include explanatory
        text, commentary, or markdown formatting outside of a single code fence.

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
    ```cpp
       // >>> BEGIN LASSI-VERIFY ASSERTIONS (do not edit this block) >>>
       // LASSI_INSTRUMENTATION - [Brief description of invariant being checked]
       [your assertion code here]
       // <<< END LASSI-VERIFY ASSERTIONS <<<
    ```

    4. **Use ONLY standard library functions:**
       - `#include <cassert>` or `#include <assert.h>` for assert()
       - `#include <cmath>` or `#include <math.h>` for isfinite(), isnan(), isinf(), fabs()
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
        """Return dict mapping label → block body string."""
        return {m.group(1): m.group(2) for m in block_pattern.finditer(stdout_text)}

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
            "summary":         "SKIP: No LASSI-VERIFY PASSTHROUGH blocks found in either output. "
                               "Passthrough insertion may have failed or blocks were removed during refactoring."
        }
        print("\n----- LASSI-Verify Tier 2: SKIP (no passthrough blocks found) -----")
        return "SKIP", diagnostic

    failures = []
    blocks_compared = 0

    for label, spec_body in spec_blocks.items():
        if label not in gen_blocks:
            # Label present in spec but missing in generated output — treat as failure
            failures.append({
                "label":     label,
                "stat":      "block_missing",
                "spec":      "(present)",
                "generated": "(absent)",
                "rel_err":   float('inf'),
            })
            continue

        gen_body = gen_blocks[label]
        blocks_compared += 1

        spec_stats = parse_stats(spec_body)
        gen_stats  = parse_stats(gen_body)

        if spec_stats is None or gen_stats is None:
            # Cannot parse statistics — fall through to LLM fallback for this block
            # Mark as unparseable so fallback is triggered below
            failures.append({
                "label":     label,
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
                failures.append({
                    "label":     label,
                    "stat":      stat_name,
                    "spec":      spec_val,
                    "generated": "(missing)",
                    "abs_err":   float('inf'),
                    "atol":      atol,
                })
                continue
            if not allclose_pass(spec_val, gen_val, tolerance, atol):
                failures.append({
                    "label":     label,
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
                if not allclose_pass(sv, gv, tolerance, atol):
                    failures.append({
                        "label":     label,
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
        "summary":         summary,
    }
    return verdict, diagnostic