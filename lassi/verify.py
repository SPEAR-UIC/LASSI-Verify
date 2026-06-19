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


def _strip_prompt_tags(code):
    """Remove prompt wrapper tags that the LLM may echo into its output.

    Tags like [/END SOURCE CODE], [BEGIN SOURCE CODE DESCRIPTION AND ANALYSIS], etc.
    are used in our prompt to delimit context sections. Some LLMs reproduce them
    verbatim inside the code fence, causing compile errors. Strip any line that
    consists solely of one of our bracket tags.
    """
    return re.sub(
        r'^\s*\[/?(?:BEGIN|END)\s+(?:SOURCE CODE|CODE ANALYSIS)[^\]]*\]\s*$',
        '',
        code,
        flags=re.MULTILINE
    )


def code_insert_passthrough(source_code, llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=False, code_description=None, feedback=None):
    """
    Insert Tier 2 passthrough data dump blocks into the source code via LLM call.

    The optional `feedback` parameter is used by the Stage 0 SOURCE_verify retry
    loop: when a previous passthrough-insertion attempt returned empty (no code
    fence detected, or the line-count guard rejected a truncated output), the
    calling code passes a feedback string explaining what went wrong and asking
    for a corrected attempt. The feedback is prepended to the prompt so the LLM
    sees the failure context before reading the rest of the instructions.
    """

    # --- Stage 0 retry feedback (when previous passthrough attempt returned empty) ---
    _feedback_prefix = ""
    if feedback:
        _feedback_prefix = (
            "=== PREVIOUS ATTEMPT FEEDBACK ===\n"
            + feedback + "\n"
            "=== END PREVIOUS ATTEMPT FEEDBACK ===\n\n"
            "Re-instrument the source code below taking the feedback above into account. "
            "Output the COMPLETE instrumented source wrapped in ``` fences. Do NOT truncate "
            "with ellipsis or 'rest unchanged' placeholders. Preserve all existing "
            "LASSI-VERIFY ASSERTIONS blocks; ADD new LASSI-VERIFY PASSTHROUGH blocks.\n\n"
        )

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

        - REQUIRED: at least one block capturing the **kernel-written output buffer** — the device
          buffer that is the DESTINATION of the final (or most significant) kernel call. This is the
          core computational result. **How to find it:** look for where the application performs
          its own D→H cudaMemcpy to bring results back to the host (for printing, validation, or
          returning to the caller) — that is the canonical output location. Alternatively, look at
          the kernel's arguments and identify which pointer it writes to. Do NOT capture a buffer
          that was only written by the host (H→D initialization or cudaMemset) without subsequent
          kernel modification — that would capture input data, not computed results.
        - REQUIRED: before placing each passthrough block, answer this question: "Was this device
          buffer written by a GPU kernel (`<<<...>>>`) between the most recent H→D copy of this
          buffer and this passthrough location?" If NO — you are capturing input, not output. Find
          a later location where the kernel has written to the buffer.
        - REQUIRED if the code contains an outer loop over **configurations or parameters** (e.g.,
          different segment sizes, grid levels, tile sizes, time steps): place the passthrough
          block INSIDE that loop so every configuration is captured independently.
          Use the SAME label each iteration — the LASSI-Verify checker compares loop iterations
          positionally and explicitly supports repeated labels. Do NOT hoist to outside the loop
          and capture only a single representative iteration.

          ⚠️ **BENCHMARK REPEAT LOOP — DO NOT INSTRUMENT INSIDE IT:** Many GPU benchmark codes
          wrap the entire computation in a pure timing/repeat loop (e.g.,
          `for (int i = 0; i < repeat; ++i)` or `for (unsigned int c0 = 0; c0 < repeat; ++c0)`).
          This is NOT a configuration or parameter loop — it repeats the SAME computation
          unchanged for timing accuracy. Do NOT place passthrough blocks inside such a loop.
          Blocks placed there fire once per repeat iteration (often 100–200×), producing a
          massive stdout dump and making the spec comparison unreliable. More critically, any
          refactored code that hoists allocations or reorganizes work outside the loop will
          produce different per-iteration buffer snapshots even when the final result is
          identical — causing guaranteed false Tier 2 failures for every valid optimization.
          **How to identify a benchmark repeat loop:** it resets accumulators or counters at
          the top of each iteration (e.g., `cudaMemset(..., 0, ...)`) and produces the same
          output state at the end of each iteration. Place passthrough blocks AFTER this loop
          closes, capturing the stable final output state once per execution.

        - OPTIONAL: additional intermediate checkpoints after major kernels or phase boundaries
        - Insert AFTER cudaDeviceSynchronize() (or equivalent) and AFTER D→H copy where applicable
        - Insert BEFORE any memory deallocations (cudaFree, hipFree, free)

        **What NOT to instrument (false-failure sources):**
        - DO NOT capture buffers populated by scatter-write patterns — i.e., any buffer
          where the *index being written* depends on thread ID or on the input data value
          observed by a specific thread. Example: `outputBuffer[gpuSrc[xidx]] = 1` inside
          a kernel, especially gated on a single thread condition like `if (tx == 11)`.
          Such buffers record "which input value thread 11 happened to observe" — not what
          the algorithm computed. Any valid optimization that changes launch geometry (block
          count, block size, warp assignment) maps thread 11 to a different array column,
          producing a different set of flagged indices even though the DP result is identical.
          This causes guaranteed Tier 2 FAIL (max_err=1.0) for every optimization, blocking
          all progress. Only instrument buffers whose contents encode algorithm output.
        - DO NOT capture debug, diagnostic, or auxiliary buffers — buffers that track
          algorithm internals (visited-cell flags, thread-local bookkeeping, coverage maps).
          Instrument only primary output buffers that the application uses to report results.

        **STEP 3: Insert labeled passthrough blocks**

        For each observation point, insert a block with this EXACT structure. Label rules:
        - For a block at a single code location: use a unique, descriptive label (e.g., [final_solution])
        - For a block inside an outer configuration loop: use the SAME label every iteration
          (e.g., [post_kernel_output] repeated for segment_size=16, 32, 64, ...)
        - The label must be consistent in the C++ comment, the BEGIN printf, and the END printf
        - The LASSI-Verify checker handles repeated labels correctly by comparing occurrences positionally

        ```cpp
        // >>> BEGIN LASSI-VERIFY PASSTHROUGH [label] (VERIFICATION BLOCK — do not rename, edit, or delete this block; this [label] must remain character-for-character identical) >>>
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

            if ((int)<N> > 10) {
                // Sample_spread10: 10 evenly-spaced interior elements at indices (k+1)*N/11.
                // Covers interior of the buffer regardless of array layout, catching bugs
                // that Sample_first10/last10 miss when boundary elements are unchanged.
                printf("Sample_spread10:");
                for (int lv_k = 0; lv_k < 10; lv_k++) {
                    int lv_spread_idx = (lv_k + 1) * (int)<N> / 11;
                    printf(" %.17g", (double)lv_host_buf[lv_spread_idx]);
                }
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
    if code_description:
        content_prompt = content_prompt + (
            "**SOURCE CODE DESCRIPTION** — Read this before selecting observation points.\n"
            "Use it to determine whether the code has a benchmark repeat loop (timing loop)\n"
            "vs. a configuration/parameter loop, so you can apply the correct placement rule\n"
            "from STEP 2 above.\n\n"
            " [BEGIN SOURCE CODE DESCRIPTION AND ANALYSIS]\n" + code_description + "\n[/END SOURCE CODE DESCRIPTION AND ANALYSIS]\n\n"
        )
    if code_analysis and code_analysis != "None":
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

    # Override with the verification insertion system prompt — preserves all existing code,
    # prevents the performance-optimization system prompt from compressing LASSI-VERIFY blocks.
    system_prompt = prompts.get_system_prompt("verify_insert")

    # Prepend Stage 0 retry feedback (when supplied) so the LLM sees the failure context
    # before reading the rest of the instrumentation prompt.
    content_prompt = _feedback_prefix + content_prompt

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

    print("< BEGIN: LLM GENERATED CODE · Tier 2 passthrough >")
    print(response_text)
    print("< END: LLM GENERATED CODE >")

    # Extract only the code block -
    # Search for the ``` pattern in the text
    #   Regular expression pattern to match text between triple backticks
    #   that represent code blocks within generated responses from the LLM.
    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)  # Strip language specifier (e.g. cpp\n)
        extracted_code = _strip_prompt_tags(extracted_code)             # Remove any echoed prompt wrapper tags
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


def _strip_t2_blocks(code):
    """Remove all Tier 2 PASSTHROUGH blocks from an instrumented code string.
    Returns a T1-only template suitable for use with code_insert_instrumentation_from_template
    when performing the Tier 1 step of zero-shot empirical verification."""
    return re.sub(
        r'// >>> BEGIN LASSI-VERIFY PASSTHROUGH.*?// <<< END LASSI-VERIFY PASSTHROUGH <<<\n?',
        '',
        code,
        flags=re.DOTALL,
    )


def _strip_t1_blocks(code):
    """Remove all Tier 1 ASSERTIONS blocks from an instrumented code string.
    Returns a T2-only template suitable for use with code_insert_instrumentation_from_template
    when performing the Tier 2 step of zero-shot empirical verification."""
    return re.sub(
        r'// >>> BEGIN LASSI-VERIFY ASSERTIONS.*?// <<< END LASSI-VERIFY ASSERTIONS <<<\n?',
        '',
        code,
        flags=re.DOTALL,
    )


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

    print("< BEGIN: LLM GENERATED CODE · template instrumentation >")
    print(response_text)
    print("< END: LLM GENERATED CODE >")

    pattern = r"```(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)
        extracted_code = _strip_prompt_tags(extracted_code)  # Remove any echoed prompt wrapper tags
    else:
        extracted_code = ""

    return extracted_code


# Status tokens emitted/parsed by code_insert_witness_chain(). These mirror the
# witness_chain_status values logged per trial. The current mechanism is output-buffer
# sentinel writing (see planning/chain_verify_output_poisoning_redesign.md for the
# design rationale; "poison" was renamed to "sentinel" and the overall mechanism to
# "witness chain"). The original input-perturbation design is in
# planning/chain_verify_design.md for historical context.
WITNESS_CHAIN_STATUS_NOT_APPLIED = "not_applied_no_timing_loop"
WITNESS_CHAIN_STATUS_SOFT_FLAG   = "soft_flag_no_safe_site"
WITNESS_CHAIN_STATUS_INSERT_ERROR = "insert_error"


def _strip_witness_chain_blocks(code):
    """Remove all witness-chain harness blocks from a code string.

    Strips three categories of harness artifact:

    1. WITNESS_CHAIN comment-delimited blocks (file-scope declarations + helper kernels,
       host init, per-iteration pre/post wrapping).
    2. The dedicated [witness_chain_state] PASSTHROUGH block. Other PASSTHROUGH blocks
       (normal Tier 2 observables) are left untouched.
    3. SAFETY-NET fallback: any raw `_t1_chain_*<<<...>>>(...);` kernel-launch call site
       that escaped the delimiter strip — for example, when the LLM restructured the timed
       loop into a CUDA Graph capture region and placed the sentinel-write / chain-update
       launches without surrounding delimiter comments. The harness template forbids this
       (delimiters are MANDATORY), but the safety net keeps the clean rebuild compilable
       when the template instruction is violated.

    This makes witness-chain (re)insertion idempotent: before installing the harness into
    SOURCE_verify or re-installing it into an already-refactored variant, strip any existing
    (possibly mangled) harness so blocks are never duplicated or compounded. It also makes
    the strip-and-rebuild step that produces uninstrumented runtimes robust against the
    most common LLM template violation.
    """
    code = re.sub(
        r'[ \t]*// >>> BEGIN LASSI-VERIFY WITNESS_CHAIN.*?// <<< END LASSI-VERIFY WITNESS_CHAIN <<<\n?',
        '', code, flags=re.DOTALL)
    code = re.sub(
        r'[ \t]*// >>> BEGIN LASSI-VERIFY PASSTHROUGH \[witness_chain_state\].*?// <<< END LASSI-VERIFY PASSTHROUGH <<<\n?',
        '', code, flags=re.DOTALL)
    # Safety net: remove any raw _t1_chain_* kernel-launch call site (whole line) that
    # survived the delimiter strip. Matches lines like
    #     _t1_chain_write_sentinel<<<1,1>>>(d_output);
    #     _t1_chain_update<<<1,1,0,stream>>>(d_output);
    # No host function in this codebase shares the `_t1_chain_*<<<` launch pattern, so
    # this is unambiguous.
    code = re.sub(
        r'^[ \t]*_t1_chain_\w+<<<[^>]*>>>\s*\([^)]*\);[ \t]*\n?',
        '', code, flags=re.MULTILINE)
    return code


def code_insert_witness_chain(source_code, llm_source, model, num_ctx, system_prompt,
                              display_response, verifyprompt=False, code_description=None,
                              is_variant=False):
    """
    Insert (or re-insert) the witness-chain harness into source_code via a dedicated,
    focused LLM call — kept SEPARATE from Tier 1 invariant-assertion selection (Option D).

    Loads ONLY the Witness_Chain_Harness.md catalog so the LLM focuses on the single mechanical
    task of installing the harness around the timed kernel invocation, rather than competing with
    regular Tier 1 assertion selection for attention (the cause of the no_t1_assertions regression).

    Idempotency / re-insertion: any pre-existing WITNESS_CHAIN harness is stripped first. This is
    what makes re-insertion into an already-refactored variant safe — the framework strips the
    variant's possibly-mangled harness and installs a fresh one around the variant's CURRENT loop
    structure. The optimizing transform is therefore free to restructure the timed loop; the
    framework guarantees the harness is present and computed identically, so a chain-state mismatch
    then genuinely means the variant performed fewer real kernel invocations than the source.

    is_variant=True selects the re-installation framing (the timed loop may have been restructured).

    Returns:
        (instrumented_code, status) where status is one of:
          "applied_decision_tree_case_N" (N=1..8) | "applied_creative" |
          "soft_flag_no_safe_site" | "not_applied_no_timing_loop" | "insert_error"
        When the harness is not applied (no timing loop / no safe site) the returned code is the
        input with any prior harness stripped. On extraction failure, status is "insert_error" and
        the stripped input is returned unchanged.
    """
    # Idempotency: remove any existing harness so we never duplicate/compound blocks.
    stripped = _strip_witness_chain_blocks(source_code)

    if is_variant:
        task_intro = (
            "You are RE-INSTALLING the LASSI-Verify witness-chain harness into a GPU program "
            "that has ALREADY been refactored by an optimizing transform. Its timing loop may have "
            "been restructured, fused, converted to a CUDA Graph, or otherwise changed. Install the "
            "harness around the program's CURRENT timed-kernel structure exactly as it now exists — "
            "do not impose a loop the program does not have, and do not change its compute logic.\n\n"
        )
    else:
        task_intro = (
            "You are installing the LASSI-Verify witness-chain harness into a GPU benchmark "
            "source program. Follow the harness instructions below exactly.\n\n"
        )

    content_prompt = task_intro
    content_prompt += prompts.get_codeknowledge("witness_chain")
    content_prompt += "\n\n----\n\n"
    if code_description:
        content_prompt += (
            "**SOURCE CODE DESCRIPTION** — use this to locate the benchmark timing/repeat loop and "
            "the primary timed-kernel input type for the Decision Tree.\n\n"
            " [BEGIN SOURCE CODE DESCRIPTION AND ANALYSIS]\n" + code_description +
            "\n[/END SOURCE CODE DESCRIPTION AND ANALYSIS]\n\n"
        )
    content_prompt += " [BEGIN SOURCE CODE]\n" + stripped + "\n[/END SOURCE CODE]\n"

    print("\n----- " + color.BOLD + color.PURPLE + "STARTING CODE INSERT - Witness Chain Harness"
          + (" (variant re-install)" if is_variant else "") + color.END + " -----\n")

    # Surgical-insertion system prompt (preserve all existing code exactly).
    system_prompt = prompts.get_system_prompt("verify_insert")

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

    print("< BEGIN: LLM GENERATED CODE · Witness Chain harness >")
    print(response_text)
    print("< END: LLM GENERATED CODE >")

    # --- Parse the WITNESS_CHAIN_STATUS token (emitted on its own line, outside the code fence) ---
    status = None
    _status_match = re.search(r'WITNESS_CHAIN_STATUS:\s*([A-Za-z0-9_]+)', response_text)
    if _status_match:
        status = _status_match.group(1).strip()

    # --- Extract the code fence ---
    match = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1)
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)  # strip language specifier
        extracted_code = _strip_prompt_tags(extracted_code)
    else:
        extracted_code = ""

    # If the LLM reported (or we infer) that the harness was not applied, return the stripped
    # input verbatim — we deliberately do not trust the LLM's echoed body for the unchanged case.
    if status in (WITNESS_CHAIN_STATUS_NOT_APPLIED, WITNESS_CHAIN_STATUS_SOFT_FLAG):
        return stripped, status

    # Applied path: require a real code fence that actually contains the harness.
    harness_present = ("LASSI-VERIFY WITNESS_CHAIN" in extracted_code
                       and "witness_chain_state" in extracted_code)
    if not extracted_code or not harness_present:
        print(f"\n[WARNING] Witness-chain insertion did not yield a harness-bearing code block "
              f"(parsed status={status!r}). Returning stripped input unchanged.")
        return stripped, WITNESS_CHAIN_STATUS_INSERT_ERROR

    # Sanity: instrumentation only adds lines.
    if len(extracted_code.splitlines()) <= len(stripped.splitlines()):
        print(f"\n[WARNING] Witness-chain insertion output is not longer than input "
              f"(input={len(stripped.splitlines())}, output={len(extracted_code.splitlines())}). "
              f"Likely truncated. Returning stripped input unchanged.")
        return stripped, WITNESS_CHAIN_STATUS_INSERT_ERROR

    # Default applied status if the LLM omitted/garbled the token but the harness is present.
    if not status or not status.startswith("applied"):
        status = "applied_creative"
    return extracted_code, status


def code_insert_asserts(source_code, llm_source, model, num_ctx, system_prompt, code_analysis, display_response, verifyprompt=False, code_description=None, feedback=None):
    """
    Insert Tier 1 invariant assertions into the source code via LLM call.

    The optional `feedback` parameter is used by the Stage 0 SOURCE_verify self-check
    retry loop: when a previous instrumentation attempt produced assertions that fired
    FAIL on SOURCE itself (i.e., the LLM asserted invariants that the source legitimately
    violates), the calling code passes a feedback string identifying which assertion
    labels failed. This string is prepended to the prompt so the LLM can revise its
    selection rather than re-issuing the same broken instrumentation.
    """

    # --- Stage 0 retry feedback (when SOURCE_verify failed its own T1 check) ---
    _feedback_prefix = ""
    if feedback:
        _feedback_prefix = (
            "=== PREVIOUS ATTEMPT FEEDBACK ===\n"
            + feedback + "\n"
            "=== END PREVIOUS ATTEMPT FEEDBACK ===\n\n"
            "Re-instrument the source code below taking the feedback above into account. "
            "Do NOT re-introduce the failing assertion(s) in the same form. Either remove "
            "them, or replace them with a different assertion on the same buffer/variable "
            "that captures a genuine invariant (e.g., for a mutable output buffer use "
            "output_finite or output_ptr_valid instead of *_unchanged).\n\n"
        )

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
       - After device-to-host transfers (cudaMemcpy D→H): validate output data — BUT ONLY if that D→H transfer is OUTSIDE any repeat/timing/compute loop (see loop warning below)
       - At function exit: validate final result properties

       ⚠️ **BENCHMARK REPEAT LOOP WARNING:** Many GPU benchmark codes wrap the entire
       computation in an outer timing or repeat loop (e.g., `for (int i = 0; i < repeat; ++i)`
       or `for (auto isoValue : isoValues)`). D→H transfers and kernel launches **inside**
       such a loop are NOT valid assertion points. Assertions placed there fire once per
       repeat iteration (often 100–200×), flooding stdout with repeated PASS/FAIL lines
       and causing the pipeline to count each occurrence as a separate assertion event.
       This is a critical failure mode. Only place assertions BEFORE the outer loop opens
       or AFTER it closes completely.

    3. **Surround all assertion blocks with protective comments** — these are CRITICAL:
       they ensure assertions survive through subsequent LLM-driven transformation steps
       without being removed or modified.

       **Two required block types — use BOTH:**

       **Type 1 — Per-observation-point check (always prints one PASS or FAIL line):**
    ```cpp
       // >>> BEGIN LASSI-VERIFY ASSERTIONS (VERIFICATION BLOCK — do not edit or delete this block; this block and its assertion labels must remain exactly as written) >>>
       // LASSI_INSTRUMENTATION - [Brief description of invariant being checked]
       {
           bool _t1_pass = (your_condition_here);
           printf("LASSI-VERIFY ASSERTIONS [label]: %s\n", _t1_pass ? "PASS" : "FAIL"); fflush(stdout);
       }
       // <<< END LASSI-VERIFY ASSERTIONS <<<
    ```

       **Type 2 — Completion sentinel (ONE per function, AFTER all Type 1 blocks, OUTSIDE all loops):**
    ```cpp
       // >>> BEGIN LASSI-VERIFY ASSERTIONS (VERIFICATION BLOCK — do not edit or delete this block; this block and its assertion labels must remain exactly as written) >>>
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

       **Plausibility range checks — use only source-defined constants:** When writing a
       range or bound check (e.g., `count <= MAX`), the bound MUST come from a named constant
       or expression already present in the source code (e.g., `blockNum`, `N`, `repeat`,
       `n * m`). Do NOT invent arithmetic bounds by reasoning about expected output magnitudes
       or intermediate value ratios — such invented bounds may not hold for the actual code
       and will cause false failures on the unmodified source. When in doubt, use a structural
       check (`> 0`, `!= nullptr`, `std::isfinite()`) rather than an invented range.

    **Your Analysis Approach (follow these steps in order):**

    **Step 1 — Map the loop structure BEFORE selecting any assertion points.**
    Read the source code description (provided at the end) and the source code itself.
    Identify every `for`/`while` loop in `main()` (or the top-level host function). For each
    loop, note its purpose:
    - **Outer timing/benchmark repeat loop** (e.g., `for (int i = 0; i < repeat; ++i)`) —
      repeats the ENTIRE computation for timing purposes. ALL kernel launches, cudaMemcpy
      D→H calls, and intermediate state inside this loop are INSIDE the loop. Assertions
      placed there fire once per repeat, not once per execution. FORBIDDEN zone.
    - **Outer configuration/parameter loop** (e.g., iterating over grid levels, tile sizes,
      isoValues) — similarly, assertions placed inside fire once per parameter value. FORBIDDEN.
    - **Inner compute loops** (solver iterations, reduction loops) — FORBIDDEN.

    After this analysis, explicitly identify the two safe zones:
    - **(A) Pre-loop setup:** code before the first outer loop opens (memory allocation,
      pointer validation, configuration validation)
    - **(B) Post-loop reporting:** code after the outer loop closes (result printing,
      final D→H copies for reporting, validation against expected output)
    All assertions MUST be placed in zone (A) or zone (B) only.

    (Detection of work-dropping cheats on idempotent benchmark timing loops — the chain
    verification harness — is handled by a SEPARATE instrumentation step with its own catalog.
    Do NOT add any perturb/unperturb/chain-state code in this Tier 1 assertion step; focus only
    on selecting transformation-invariant assertions.)

    **Step 2 — Answer these questions about the computation:**
    - **What is the computational domain?** (e.g., linear algebra, CFD, molecular dynamics,
      iterative solver, stochastic methods) — this determines which domain invariants apply
    - **What properties must be preserved across ANY correct transformation?**
      - Input constraints: pointer validity, array bounds, value ranges, numerical validity
      - Output constraints: structural correctness (non-null, non-negative count), finiteness
      - Mathematical laws: only where the original problem requires them
    - **Which assertions are truly transformation-invariant?** — an assertion is valid only if
      it would hold in any correct implementation of this computation, not just this specific
      parallelization or optimization. When in doubt, do not insert the assertion.

    **Step 3 — Select assertions only from the safe zones identified in Step 1.**
    Do NOT place any assertion at a location that falls within the FORBIDDEN zones, even
    if a D→H transfer appears at that location. Use only the pre-loop and post-loop regions.

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

    if code_description:
        content_prompt = content_prompt + (
            "\n**SOURCE CODE DESCRIPTION** — Read this carefully before selecting assertion points.\n"
            "Use it to identify the algorithm's outer loop structure (especially any timing/repeat\n"
            "loop that wraps the full computation) and the pre-loop / post-loop safe zones for\n"
            "assertion placement, as described in Step 1 of the Analysis Approach above.\n\n"
            " [BEGIN SOURCE CODE DESCRIPTION AND ANALYSIS]\n" + code_description + "\n[/END SOURCE CODE DESCRIPTION AND ANALYSIS]\n"
        )
    if code_analysis and code_analysis != "None":
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

    # Override with the verification insertion system prompt — preserves all existing code,
    # prevents the performance-optimization system prompt from compressing LASSI-VERIFY blocks.
    system_prompt = prompts.get_system_prompt("verify_insert")

    # Prepend Stage 0 retry feedback (when supplied) so the LLM sees the failure context
    # before reading the rest of the instrumentation prompt.
    content_prompt = _feedback_prefix + content_prompt

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
        
    print("< BEGIN: LLM GENERATED CODE · Tier 1 assertions >")
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
        extracted_code = match.group(1)
        # Strip optional language specifier on the first line (e.g. "cpp\n" or "c++\n")
        # LLMs commonly emit ```cpp ... ``` fences; group(1) captures the specifier too.
        extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)
        extracted_code = _strip_prompt_tags(extracted_code)  # Remove any echoed prompt wrapper tags
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
                                tolerance=1e-6):
    """
    Tier 2 empirical equivalence check. Compare passthrough output from a generated
    code against the source-as-specification reference (SOURCE_verify stdout).

    Implements the source-as-specification premise: SOURCE_verify's observable behavior
    is the correctness criterion. A transformation is correct if it preserves this
    behavior; the framework makes no claim about whether the source itself is correct.

    Purely deterministic: parse labeled blocks, compare min/max/mean and sample values
    using numpy allclose-style mixed tolerance:
        |a - b| <= atol + rtol * max(|a|, |b|)
    where atol = |spec_max| * rtol (derived per-block from the spec's max statistic).
    This anchors the absolute tolerance to the field's actual dynamic range, preventing
    false positives from near-zero statistics (e.g., converged Jacobi mean ~1e-12).
    `sum` is intentionally excluded: it equals mean×N and accumulates per-element
    floating-point noise at large N, making it an unreliable correctness criterion.

    Parse errors (malformed passthrough output) are treated as FAIL and trigger
    self-correction — no LLM fallback. This keeps Tier 2 verdicts fully deterministic
    and independent of the LLM judge.

    Returns:
        verdict (str): "PASS" | "FAIL" | "SKIP"
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
        r'Sample_(first10|last10|spread10):(.*)'
    )
    # Regex for the Elements count line — parsed and compared exactly (no tolerance).
    # A mismatch here means the generated code operates on a different problem size than
    # the source-as-specification, which is an unambiguous correctness violation
    # regardless of whether min/max/mean statistics happen to match.
    elements_pattern = re.compile(r'Elements:\s*(\d+)')
    # Regex for the Value line — used by the [witness_chain_state] passthrough block,
    # which carries a single 64-bit hex value rather than Statistics/Sample fields.
    # The witness_chain_state value is compared with byte-exact equality (no tolerance):
    # any single-bit difference between source and variant indicates the variant did
    # not perform the same per-iteration work that source did. See the witness-chain
    # harness template and planning/chain_verify_output_poisoning_redesign.md for the
    # full rationale (allclose-style tolerance on a 64-bit hash would falsely accept
    # mismatched chain states because bit-level diffs are dwarfed by the value scale).
    value_pattern = re.compile(r'Value:\s*(\S+)')

    # Label of the witness-chain passthrough block — handled with exact-string comparison
    # instead of the standard Statistics/Sample numerical comparison.
    WITNESS_CHAIN_LABEL = "witness_chain_state"

    def extract_blocks(stdout_text):
        """Return dict mapping label → ordered list of block body strings.
        Collects all occurrences so blocks inside loops are not silently deduplicated
        by a dict comprehension that keeps only the last occurrence per label."""
        result = {}
        for m in block_pattern.finditer(stdout_text):
            result.setdefault(m.group(1), []).append(m.group(2))
        return result

    def parse_elements(block_body):
        """Parse Elements count; return int or None if not found/parseable."""
        m = elements_pattern.search(block_body)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

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

            # --- Special case: witness_chain_state block uses Value: line + exact match ---
            # The witness_chain_state passthrough carries a 64-bit hash that must match
            # byte-exactly between source and variant. Allclose-style numerical tolerance
            # is wrong here (would falsely accept mismatched 64-bit values because bit-level
            # diffs are dwarfed by the magnitude). String comparison of the Value: hex line
            # gives the required exact-match semantics. Skip the stats/samples logic below.
            if label == WITNESS_CHAIN_LABEL:
                spec_val_m = value_pattern.search(spec_body)
                gen_val_m  = value_pattern.search(gen_body)
                # Chain failures use string-based comparison; include abs_err/atol
                # set to inf/0 so the downstream summary formatter (which expects these
                # fields on every failure) doesn't KeyError. The summary formatter has
                # a chain_value_* branch that ignores these numeric fields.
                if spec_val_m is None or gen_val_m is None:
                    failures.append({
                        "label":     failure_label,
                        "stat":      "chain_value_missing",
                        "spec":      spec_val_m.group(1) if spec_val_m else "(missing)",
                        "generated": gen_val_m.group(1) if gen_val_m else "(missing)",
                        "abs_err":   float('inf'),
                        "atol":      0.0,
                        "rel_err":   float('inf'),
                    })
                    max_rel_err = max(max_rel_err, float('inf'))
                    continue
                spec_val_str = spec_val_m.group(1).strip().lower()
                gen_val_str  = gen_val_m.group(1).strip().lower()
                if spec_val_str != gen_val_str:
                    failures.append({
                        "label":     failure_label,
                        "stat":      "chain_value_mismatch",
                        "spec":      spec_val_str,
                        "generated": gen_val_str,
                        "abs_err":   float('inf'),
                        "atol":      0.0,
                        "rel_err":   float('inf'),
                    })
                    max_rel_err = max(max_rel_err, float('inf'))
                # On match, just continue (no failure recorded)
                continue

            # Check element count first — a mismatch means the generated code operates
            # on a different problem size than the source specification. This is an exact
            # integer comparison: no tolerance. Only fail if BOTH blocks have parseable
            # counts that disagree; if either is absent, fall through to stat comparison.
            spec_elements = parse_elements(spec_body)
            gen_elements  = parse_elements(gen_body)
            if spec_elements is not None and gen_elements is not None and spec_elements != gen_elements:
                max_rel_err = max(max_rel_err, float('inf'))
                failures.append({
                    "label":     failure_label,
                    "stat":      "elements_mismatch",
                    "spec":      spec_elements,
                    "generated": gen_elements,
                    "abs_err":   abs(spec_elements - gen_elements),
                    "atol":      0,
                })
                continue  # skip stat/sample comparison — sizes are incommensurate

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
    # Deterministic verdict
    # ------------------------------------------------------------------
    # Any failure (parse_error, block_missing, numeric mismatch) → FAIL.
    # Parse errors mean the generated code did not emit well-formed passthrough output;
    # this triggers Tier 2 self-correction so the LLM can fix the format.
    # No LLM fallback: Tier 2 verdicts must be fully deterministic and independent
    # of the LLM judge.
    if failures:
        fail_summary_parts = []
        for f in failures:
            if f["stat"] == "block_missing":
                fail_summary_parts.append(f"  [block_missing] label='{f['label']}': block absent in generated output")
            elif f["stat"] in ("chain_value_mismatch", "chain_value_missing"):
                # Chain-verify failures are string-based; the abs_err/atol/rtol fields
                # are not meaningful for hex hash comparison. Show the values directly.
                fail_summary_parts.append(
                    f"  [{f['label']}] {f['stat']}: spec={f['spec']}, generated={f['generated']} "
                    f"(64-bit chain hash; exact-match required)"
                )
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
    # Filter out format-specifier artifacts (e.g. '%s', '%d') that can appear if the LLM
    # generates a dynamic label via printf format string instead of a hardcoded label name.
    labels = [l for l in labels if l and '%' not in l]
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
                                   display_response, verifyprompt=False, code_description=None,
                                   feedback=None):
    """
    Tier 3: Generate a metamorphic contract for the given source code.

    Calls the LLM with the metamorphic catalog to select applicable relations per the
    Coverage Principle (catalog §7): at least one relation per applicable type; within
    a type, additional relations only when each catches a distinct fault class.
    Returns a contract dict with a 'relations' key, or {} on failure.

    Called once in Stage 0 after SOURCE_verify is constructed (after Tier 1 + Tier 2
    insertion). The contract drives all subsequent Tier 3 work.

    Maps to: Benton's Relational Hoare Logic (relational reasoning over two executions
    under related inputs). The LLM selects validation MRs from mathematical/physical
    semantics, not implementation details.
    """
    system_prompt = prompts.get_system_prompt("verify_contract")
    content_prompt = (
        "You are preparing GPU-accelerated scientific code for Tier 3 metamorphic testing "
        "as part of the LASSI-Verify verification framework.\n\n"
        "Your task is to analyze the source code and select applicable metamorphic "
        "relations from the catalog below. Follow the Coverage Principle (catalog §7): "
        "for every applicable relation type, propose at least one relation; within a "
        "type, propose additional relations only when each catches a distinct fault "
        "class. Do NOT pad the count to hit a target number, and do NOT cap the count "
        "by dropping an applicable, distinct relation. Return your selection as a "
        "single JSON block.\n\n"
        "----\n\n"
    )
    content_prompt += prompts.get_codeknowledge("metamorphic_catalog")
    content_prompt += "\n\n----\n\n"
    if code_description:
        content_prompt += "[BEGIN SOURCE CODE DESCRIPTION AND ANALYSIS]\n" + code_description + "\n[END SOURCE CODE DESCRIPTION AND ANALYSIS]\n\n"
    content_prompt += "[BEGIN SOURCE CODE]\n" + source_code + "\n[END SOURCE CODE]\n\n"

    # Extract existing PASSTHROUGH block labels from source_code so the LLM can anchor
    # observation_points to the exact vocabulary already present.  These labels are what
    # check_metamorphic_relation() will match against — any name mismatch causes a silent SKIP.
    _pt_labels = re.findall(
        r'=== BEGIN LASSI-VERIFY PASSTHROUGH \[([^\]]+)\] ===', source_code
    )
    if _pt_labels:
        _label_list = ", ".join(f'"{lbl}"' for lbl in _pt_labels)
        content_prompt += (
            "IMPORTANT — observation_points vocabulary:\n"
            "The source code above already contains LASSI-VERIFY PASSTHROUGH blocks with "
            "the following labels (in order):\n"
            f"  {_label_list}\n"
            "For every relation you select, the `observation_points` list MUST use ONLY "
            "these exact label strings. Do not invent new names, do not paraphrase — copy "
            "them character-for-character from the list above. "
            "The matching in check_metamorphic_relation() is an exact-string comparison; "
            "any deviation silently disables that relation.\n\n"
        )

    if feedback:
        content_prompt += (
            "IMPORTANT — previous attempt feedback:\n"
            "The following relations were proposed in a previous attempt but ALL FAILED "
            "when tested against the verified source code. Do NOT re-propose these relations:\n\n"
        )
        for name, description, fail_reason in feedback:
            content_prompt += f"  - {name}: {description}\n    Reason failed: {fail_reason}\n"
        content_prompt += (
            "\nUsing the catalog above, select DIFFERENT relations that are categorically "
            "distinct from those that failed.\n\n"
        )

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

    print("< BEGIN: LLM GENERATED CODE · Tier 3 metamorphic contract >")
    print(response_text)
    print("< END: LLM GENERATED CODE >")

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

            # --- Post-parse validation: correct/drop relations with bad observation_points ---
            # The LLM is told to reuse exact PASSTHROUGH labels, but sometimes hallucinates.
            # check_metamorphic_relation() does exact-string matching, so any mismatch silently
            # disables a relation. Here we auto-correct obvious mismatches and drop the rest.
            if _pt_labels and contract.get("relations"):
                _pt_label_set = set(_pt_labels)

                def _best_label_match(bad_label, candidates):
                    """Return the best-matching candidate label, or None if ambiguous/poor fit.
                    Priority:
                      1. Single candidate → always substitute (unambiguous).
                      2. Token-overlap (split on '_'): pick candidate with highest fraction of
                         bad_label tokens present; require ≥ 50% overlap to auto-correct.
                    """
                    if len(candidates) == 1:
                        return next(iter(candidates))
                    bad_tokens = set(bad_label.split('_'))
                    best, best_score = None, 0
                    for cand in candidates:
                        overlap = len(bad_tokens & set(cand.split('_'))) / max(len(bad_tokens), 1)
                        if overlap > best_score:
                            best_score, best = overlap, cand
                    return best if best_score >= 0.5 else None

                valid_relations = []
                for rel in contract["relations"]:
                    rel_name = rel.get("name", "unknown")
                    obs = rel.get("observation_points", [])
                    corrected_obs = []
                    drop = False
                    for op in obs:
                        if op in _pt_label_set:
                            corrected_obs.append(op)
                        else:
                            fix = _best_label_match(op, _pt_label_set)
                            if fix:
                                print(f"[Tier 3 contract] Auto-corrected observation_point "
                                      f"for '{rel_name}': '{op}' → '{fix}'")
                                corrected_obs.append(fix)
                            else:
                                print(f"[Tier 3 contract] Dropping relation '{rel_name}': "
                                      f"observation_point '{op}' not in source PASSTHROUGH "
                                      f"labels {sorted(_pt_label_set)} and no close match found.")
                                drop = True
                                break
                    if not drop:
                        rel["observation_points"] = corrected_obs
                        valid_relations.append(rel)

                n_dropped = len(contract["relations"]) - len(valid_relations)
                if n_dropped:
                    print(f"[Tier 3 contract] {n_dropped} relation(s) dropped due to "
                          f"unresolvable observation_points; {len(valid_relations)} remain.")
                contract["relations"] = valid_relations

            # --- Post-parse validation: drop relations with degenerate type/factor pairings ---
            # The relation type defines what the factor must mean. For transformation-style
            # relations, certain factor values reduce the relation to an identity that any
            # code (including broken ones) trivially passes — zero discriminating power.
            #
            #   SCALING    → factor is the OUTPUT scaling ratio (variant ≈ factor * baseline).
            #                factor == 1.0 is identity scaling. Drop unconditionally. Cases
            #                where the output is genuinely unchanged under an input
            #                transformation (e.g., co-normalized group scaling on
            #                normalized-distance kernels) belong in INVARIANCE, not SCALING.
            #   ADDITIVITY → factor is the OUTPUT change ratio for a given input offset.
            #                factor == 0.0 is the additivity analog of the SCALING degeneracy
            #                (adding zero to inputs trivially produces unchanged output).
            #                Drop unconditionally — the INVARIANCE category is the right
            #                home for output-unchanged-under-input-shift relations.
            #   SYMMETRY   → factor is fixed at -1.0 (negation/reflection). Any other factor
            #                contradicts the type definition. Drop unconditionally.
            #   INVARIANCE → factor is conventionally 1.0 (output ratio is unity by
            #                definition). Accepted as first-class; the only check that
            #                applies is the input_variables requirement (a zero-input-var
            #                INVARIANCE is just identity replay). The no-op-phrase catch-all
            #                below is BYPASSED for INVARIANCE because legitimate INVARIANCE
            #                descriptions explicitly say "output unchanged" / "leaves
            #                unchanged" — that wording is the correct semantic for this type.
            #
            # Catch-all: even when factor passes the type-specific check, the description
            # may openly admit the transformation is a no-op (e.g. "transformed input is
            # identical", "adding zero", "identity transformation", "leaves unchanged").
            # Drop these regardless of type/factor — the LLM has confessed the relation is
            # vacuous in its own words. EXCEPTION: INVARIANCE relations are exempt (see above).
            if contract.get("relations"):
                # Phrases that indicate the LLM admits the input transformation is a no-op.
                # Matched case-insensitively against description + expected_relationship.
                # Additions over time (cat-and-mouse — LLM keeps inventing new degenerate
                # framings once the obvious ones are blocked):
                #   - "identity replay" / "identity_replay": LLM mislabels as MONOTONICITY
                #     and proposes running source twice with no input change ("self-replay").
                #   - "= 1.0 * original" / "= original_output": exact phrasing the LLM uses
                #     in expected_relationship when the relation is literal identity.
                #   - "unchanged-input", "follow-up execution", "second execution",
                #     "repeated execution": describe re-running the source without any
                #     transformation, the structural signature of identity_replay relations.
                _noop_phrases = (
                    "transformed input is identical",
                    "input is unchanged", "input unchanged",
                    "leaves the input unchanged", "leaves the input identical",
                    "identity transformation",
                    "leaves the mathematical problem unchanged",
                    "adding zero offset", "adding an offset of zero",
                    "adding zero is", "scaling by 1 is",
                    "any finite factor leaves",
                    "identity replay", "identity_replay",
                    "unchanged-input", "unchanged input",
                    "follow-up execution", "second execution", "repeated execution",
                    "= 1.0 * original", "= original_output", "= 1.0*original",
                )
                non_degenerate = []
                for rel in contract["relations"]:
                    rel_type = rel.get("type", "")
                    rel_name = rel.get("name", "unknown")
                    rel_factor = rel.get("factor", None)
                    rel_desc = (rel.get("description", "") + " "
                                + rel.get("expected_relationship", "")).lower()

                    if rel_type == "SCALING" and rel_factor == 1.0:
                        print(f"[Tier 3 contract] Dropping relation '{rel_name}': "
                              f"SCALING with factor=1.0 is identity (no test power). "
                              f"If the kernel's output is genuinely unchanged under an input "
                              f"transformation, use type=INVARIANCE instead. "
                              f"Description: {rel.get('description', '(none)')[:120]}")
                        continue
                    if rel_type == "ADDITIVITY" and rel_factor == 0.0:
                        print(f"[Tier 3 contract] Dropping relation '{rel_name}': "
                              f"ADDITIVITY with factor=0.0 is identity (no test power). "
                              f"If the kernel's output is genuinely unchanged under a uniform "
                              f"input shift, use type=INVARIANCE instead. "
                              f"Description: {rel.get('description', '(none)')[:120]}")
                        continue
                    if rel_type == "SYMMETRY" and rel_factor != -1.0:
                        print(f"[Tier 3 contract] Dropping relation '{rel_name}': "
                              f"SYMMETRY requires factor=-1.0 (negation), got factor={rel_factor}.")
                        continue
                    # Structural check: SCALING/SYMMETRY/ADDITIVITY/INVARIANCE are
                    # transformation-style relations — they require at least one input
                    # variable to transform. An empty input_variables list means "no
                    # transformation defined", which is the structural signature of an
                    # identity-replay degenerate relation (the LLM sometimes mislabels these
                    # as MONOTONICITY too, but for the transformation types it's an outright
                    # contradiction with the type). MONOTONICITY/CONSERVATION legitimately
                    # have empty input_variables (they observe the existing solver run), so
                    # this check excludes them.
                    if (rel_type in ("SCALING", "SYMMETRY", "ADDITIVITY", "INVARIANCE")
                            and not rel.get("input_variables")):
                        print(f"[Tier 3 contract] Dropping relation '{rel_name}': "
                              f"{rel_type} requires at least one input_variable to transform; "
                              f"empty input_variables means no transformation is defined.")
                        continue
                    # Catch-all: description openly admits the transformation is a no-op.
                    # EXEMPTION: INVARIANCE relations legitimately describe themselves with
                    # "output unchanged" / "leaves unchanged" wording — that IS the semantic
                    # of the type, not an admission of degeneracy. Apply the catch-all to
                    # every type EXCEPT INVARIANCE.
                    if rel_type != "INVARIANCE" and any(phrase in rel_desc for phrase in _noop_phrases):
                        print(f"[Tier 3 contract] Dropping relation '{rel_name}': "
                              f"description openly admits the transformation is a no-op. "
                              f"Description: {rel.get('description', '(none)')[:140]}")
                        continue
                    non_degenerate.append(rel)
                n_dropped_deg = len(contract["relations"]) - len(non_degenerate)
                if n_dropped_deg:
                    print(f"[Tier 3 contract] {n_dropped_deg} degenerate relation(s) dropped; "
                          f"{len(non_degenerate)} remain.")
                contract["relations"] = non_degenerate

        except (json.JSONDecodeError, ValueError) as e:
            print(f"\n----- Tier 3 contract JSON parse failed: {e} — Tier 3 disabled -----")
            contract = {}
    else:
        print("\n----- Tier 3 contract: no JSON found in LLM response — Tier 3 disabled -----")

    return contract


def code_insert_metamorphic_variant(source_code, relation, llm_source, model, num_ctx,
                                     system_prompt, display_response,
                                     template_code=None, feedback=None, verifyprompt=False,
                                     repair_mode=False):
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
    input_lines   = len(source_code.splitlines())

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
            "**METAMORPHIC output block format** (using 'Residual:' field):\n\n"
            "```\n"
            "=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_<k>] ===\n"
            "Relation: monotonicity\nIteration: <k>\nResidual: %.17g\n"
            "=== END LASSI-VERIFY METAMORPHIC [residual_iter_<k>] ===\n"
            "```\n\n"
            "**CRITICAL — Block placement depends on the loop structure:**\n\n"
            "If the solver has a TWO-LEVEL loop (outer timestep/sweep loop wrapping an inner "
            "solver loop), place the block AFTER the inner loop exits, once per outer iteration. "
            "Placing it inside the inner loop would emit O(timesteps × inner_iterations) lines — "
            "potentially millions — and will crash the pipeline.\n\n"
            "```cpp\n"
            "// TWO-LEVEL: block goes OUTSIDE the inner loop\n"
            "for (int t = 0; t < num_timesteps; ++t) {\n"
            "    double final_res = 0.0;\n"
            "    for (int iter = 0; iter < max_iter; ++iter) {\n"
            "        // ... solver kernels ...\n"
            "        if (final_res < tol) break;\n"
            "    }\n"
            "    // ✅ ONE block per timestep, after inner loop\n"
            "    printf(\"=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\\n\", t);\n"
            "    printf(\"Relation: monotonicity\\nIteration: %d\\nResidual: %.17g\\n\", t, final_res);\n"
            "    printf(\"=== END LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\\n\", t);\n"
            "}\n"
            "```\n\n"
            "If the solver has only a SINGLE-LEVEL iteration loop, cap the output at 200 "
            "iterations to prevent stdout overflow:\n\n"
            "```cpp\n"
            "// SINGLE-LEVEL: cap at 200\n"
            "#define LASSI_MT_MAX_PRINT 200\n"
            "for (int iter = 0; iter < max_iter; ++iter) {\n"
            "    // ... solver ...\n"
            "    if (iter < LASSI_MT_MAX_PRINT) {\n"
            "        printf(\"=== BEGIN LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\\n\", iter);\n"
            "        printf(\"Relation: monotonicity\\nIteration: %d\\nResidual: %.17g\\n\", iter, residual);\n"
            "        printf(\"=== END LASSI-VERIFY METAMORPHIC [residual_iter_%d] ===\\n\", iter);\n"
            "    }\n"
            "    if (converged) break;\n"
            "}\n"
            "```\n"
        )
    elif relation_type == "INVARIANCE":
        # Build a one-line label for the Invariance_under: field. Prefer the relation
        # name itself (which the LLM already chose as a descriptive identifier).
        _inv_label = relation_name
        block_format = (
            f"**METAMORPHIC output block format** "
            f"(insert AFTER the final D→H transfer and cudaDeviceSynchronize, "
            f"BEFORE any cudaFree / free calls):\n\n"
            f"```\n"
            f"=== BEGIN LASSI-VERIFY METAMORPHIC [{first_label}] ===\n"
            f"Relation: {relation_name}\n"
            f"Invariance_under: {_inv_label}\n"
            f"Buffer: <buffer_name>\n"
            f"Elements: <N>\n"
            f"Statistics: min=%.17g, max=%.17g, mean=%.17g\n"
            f"Sample_first10: %.17g %.17g ...  (up to 10 values, space-separated)\n"
            f"=== END LASSI-VERIFY METAMORPHIC [{first_label}] ===\n"
            f"```\n\n"
            f"Print this block using printf with %.17g format. The label MUST exactly match "
            f"the corresponding PASSTHROUGH block label for this observation point "
            f"(e.g., if the PASSTHROUGH label is `[final_solution]`, the METAMORPHIC label "
            f"must also be `[final_solution]`). The `Invariance_under:` field names the "
            f"specific input transformation applied — it is informational for diagnostic "
            f"output and does not affect the check. The checker compares each variant "
            f"statistic Z_stat to the original statistic Y_stat (factor = 1.0 by definition "
            f"for INVARIANCE).\n"
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
            "No input transformation is needed.\n\n"
            "**Step 1 — Identify the loop structure before inserting any blocks:**\n"
            "Read the source code and determine whether the solver uses:\n"
            "  (A) A single-level iteration loop (the solver loop is the outermost loop), OR\n"
            "  (B) A two-level loop structure (outer timestep/sweep loop wrapping an inner "
            "solver/SOR loop).\n\n"
            "**If (A) — single-level loop:** Insert the METAMORPHIC block inside the loop "
            "but guard it with `if (iter < LASSI_MT_MAX_PRINT)` (define LASSI_MT_MAX_PRINT 200 "
            "above the loop). This caps stdout at 200 blocks regardless of max_iter.\n\n"
            "**If (B) — two-level loop:** Insert the METAMORPHIC block AFTER the inner loop "
            "exits, inside the outer loop body. Print once per outer iteration using the outer "
            "loop counter as the iteration index and the inner loop's final convergence "
            "residual as the value. NEVER place the block inside the inner loop — this "
            "would print O(outer_iterations × inner_iterations) lines and overflow the pipeline.\n"
        )
    elif relation_type == "INVARIANCE" and input_vars:
        transform_instruction = (
            f"**Step 1 — Apply the input transformation:**\n"
            f"Insert the transformation code AFTER host array initialization but BEFORE "
            f"the first host-to-device memory copy (cudaMemcpy H→D or hipMemcpy H→D).\n\n"
            f"The transformation must produce a variant input that is genuinely different "
            f"from the original (no-op transformations are rejected). The specific "
            f"transformation is described in the relation description above:\n"
            f"  {description}\n\n"
            f"Apply the transformation to ALL of these input arrays/scalars: "
            f"{', '.join(input_vars)}.\n\n"
            f"Common INVARIANCE input transformations and their canonical implementations\n"
            f"(translate the description above into the matching pattern):\n"
            f"  - Attribute / column-order reversal within every row of a 2D row-major array:\n"
            f"      for (int r = 0; r < NUM_ROWS; r++)\n"
            f"          for (int c = 0; c < COLS_PER_ROW / 2; c++)\n"
            f"              std::swap({input_vars[0]}[r*COLS_PER_ROW + c],\n"
            f"                        {input_vars[0]}[r*COLS_PER_ROW + (COLS_PER_ROW - 1 - c)]);\n"
            f"  - Value relabeling preserving equality classes (e.g., bitwise NOT, modular shift):\n"
            f"      const int K = 7;  // any non-zero constant\n"
            f"      for (int i = 0; i < N; i++) {input_vars[0]}[i] = ({input_vars[0]}[i] ^ K);\n"
            f"  - Uniform value shift (adds the same constant to every element):\n"
            f"      const int SHIFT = 5;\n"
            f"      for (int i = 0; i < N; i++) {input_vars[0]}[i] += SHIFT;\n"
            f"  - Joint α-scaling of a co-normalized group (data + mean + std together):\n"
            f"      const double alpha = 2.0;\n"
            f"      for (int i = 0; i < N; i++) {{\n"
            f"          {input_vars[0]}[i] *= alpha;\n"
            f"          // ... apply the same scaling to each remaining input variable above\n"
            f"      }}\n\n"
            f"Pick the pattern that matches the relation's description. Apply ONE transformation\n"
            f"per variant (do not chain multiple). If the relation specifies a transformation\n"
            f"not in this list, implement it directly from the description — these patterns are\n"
            f"common examples, not an exhaustive list.\n\n"
            f"**Step 2 — Insert METAMORPHIC output blocks:**\n"
            f"After the final device-to-host transfer, insert a METAMORPHIC block at each\n"
            f"observation point label listed above. The output buffer values must be the\n"
            f"computed result on the transformed input — the checker will compare these against\n"
            f"the original PASSTHROUGH values and expect them to match (output is unchanged).\n"
        )
    else:
        transform_instruction = (
            f"**Step 1 — Apply the transformation** appropriate for {relation_type}.\n"
            f"**Step 2 — Insert METAMORPHIC output blocks** at the observation points above.\n"
        )

    # --- Feedback prefix (Stage 0 retry only) ---
    # Prepended when a previous attempt failed so the LLM can correct its mistake.
    _feedback_prefix = ""
    if feedback:
        _feedback_prefix = (
            "=== PREVIOUS ATTEMPT FEEDBACK ===\n"
            + feedback + "\n"
            "=== END PREVIOUS ATTEMPT FEEDBACK ===\n\n"
        )

    if repair_mode:
        # repair_mode: fix syntax errors in a broken variant; do NOT restructure.
        # source_code = the broken variant to patch.
        # template_code = the source variant that compiled successfully (structural reference).
        content_prompt = _feedback_prefix + (
            "You are fixing a metamorphic variant that failed to compile or execute.\n\n"
            "Your task: find and fix ONLY the syntax or compilation errors in the broken variant "
            "below. This is a bug-fix-only task — do NOT change the metamorphic transformation "
            "logic, observation points, METAMORPHIC block labels, or block format.\n\n"
            "**CRITICAL CONSTRAINTS:**\n"
            "- Fix ONLY syntax errors (missing braces, missing semicolons, malformed expressions, "
            "undeclared variables, wrong types, etc.)\n"
            "- Do NOT change the input transformation code or its scope\n"
            "- Do NOT change METAMORPHIC output block labels or format\n"
            "- Do NOT restructure or reorganize the code\n"
            "- Do NOT remove or modify any LASSI-VERIFY ASSERTIONS or LASSI-VERIFY PASSTHROUGH blocks\n"
            "- The output must be the same program with only the syntax bugs corrected\n\n"
            + (
                "[STRUCTURAL REFERENCE — the same metamorphic variant compiled successfully "
                "for the source code version]\n"
                "Use this to identify what the correct structure should look like. "
                "Your fix should make the broken variant follow this structural pattern closely "
                "without copying it verbatim (the generated code may differ from the source).\n\n"
                + template_code +
                "\n[END STRUCTURAL REFERENCE]\n\n"
                if template_code else ""
            )
            + "[BROKEN VARIANT — fix the syntax errors in this code]\n"
            + source_code
            + "\n[END BROKEN VARIANT]\n\n"
            "Output ONLY the complete, fixed source code ready to compile. "
            "Do NOT include explanatory text or commentary outside of a single code fence. "
            "The output must compile with nvcc or hipcc.\n"
        )
    else:
        content_prompt = _feedback_prefix + (
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
            f"**OUTPUT INSTRUCTIONS:**\n"
            f"The source code below has exactly {input_lines} lines. Your output MUST have MORE than "
            f"{input_lines} lines — you are inserting instrumentation, so the output must be longer.\n"
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

    _mode_label = "REPAIRING" if repair_mode else "Inserting"
    print("\n----- " + color.BOLD + color.PURPLE +
          f"STARTING Tier 3: {_mode_label} metamorphic variant ({relation_name})" +
          color.END + " -----\n")

    # Override with the verification insertion system prompt — preserves all existing code,
    # prevents the performance-optimization system prompt from compressing LASSI-VERIFY blocks.
    system_prompt = prompts.get_system_prompt("verify_insert")

    temp = 0.2
    _prev_output_lines = None
    # Allow one retry with an explicit line count warning if the LLM truncates the output.
    for _attempt in range(2):
        if _attempt == 0:
            _prompt = content_prompt
        else:
            _prompt = (
                f"[CRITICAL CORRECTION] Your previous output had only {_prev_output_lines} lines "
                f"but the source code has {input_lines} lines. You omitted sections of code using "
                f"placeholder comments. You MUST output ALL {input_lines}+ lines of the source code "
                f"verbatim with only the metamorphic instrumentation added. "
                f"Do not use '// ... rest unchanged', ellipsis, or any shorthand notation.\n\n"
            ) + content_prompt

        if llm_source == "argo":
            response_text = argo_llm(model, system_prompt, _prompt, temp, verifyprompt)
        elif llm_source == "ollama":
            response_text = ollama_llm_nostream(model, num_ctx, system_prompt, _prompt, temp, verifyprompt)
        elif llm_source == "cerebras":
            response_text = cerebras_llm(model, num_ctx, system_prompt, _prompt, temp, verifyprompt)
        else:
            print("Invalid LLM source for code_insert_metamorphic_variant")
            return ""

        print("< BEGIN: LLM GENERATED CODE · Tier 3 metamorphic variant >")
        print(response_text)
        print("< END: LLM GENERATED CODE >")

        # Extract code block (same pattern as code_insert_asserts / code_insert_passthrough)
        pattern = r"```(.*?)```"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            extracted_code = match.group(1)
            extracted_code = re.sub(r'^[a-zA-Z+]+\n', '', extracted_code)
            extracted_code = _strip_prompt_tags(extracted_code)  # Remove any echoed prompt wrapper tags
        else:
            # Fallback: LLM response did not include code fences (common for very long files).
            # Try to extract from the first preprocessor directive or CUDA/C++ keyword.
            # Without this, long kernels like lid-driven-cavity always SKIP Tier 3 because
            # the LLM response for a 1000+ line file omits backtick fences.
            fallback_pattern = r'((?:#include|#pragma|#define|__global__|extern\s+"C").*)'
            fallback_match = re.search(fallback_pattern, response_text, re.DOTALL)
            if fallback_match:
                extracted_code = fallback_match.group(1)
                extracted_code = _strip_prompt_tags(extracted_code)  # Remove any echoed prompt wrapper tags
                print(f"\n[WARNING] code_insert_metamorphic_variant: no code fences in LLM response for '{relation_name}'; extracted from first code directive.")
            else:
                extracted_code = ""
                print(f"\n[WARNING] code_insert_metamorphic_variant: no code fences or code directives found for '{relation_name}'; cannot extract variant.")

        # Sanity check: instrumentation only adds lines — same or fewer lines means the LLM
        # omitted sections via placeholder comments (e.g. "// ... [other kernels unchanged] ...").
        # Skipped in repair_mode: a syntax-only fix may not change the line count, and the
        # broken variant itself is already at or above the original line count.
        if not repair_mode:
            output_lines = len(extracted_code.splitlines())
            if extracted_code and output_lines <= input_lines:
                _prev_output_lines = output_lines
                retry_msg = "Retrying with explicit line count correction." if _attempt == 0 else "Rejecting output."
                print(f"\n[WARNING] Metamorphic variant line count check FAILED: "
                      f"input={input_lines} lines, output={output_lines} lines. "
                      f"LLM likely omitted code via placeholder comments. {retry_msg}")
                if _attempt == 0:
                    continue  # retry
                else:
                    return ""  # two failures — give up

        return extracted_code
    return ""  # should not be reached


def check_metamorphic_relation(variant_stdout, original_passthrough_stdout, relation,
                                tolerance=None):
    """
    Tier 3: Deterministic check of a metamorphic relation between variant and original outputs.

    Parses METAMORPHIC blocks from variant_stdout and PASSTHROUGH blocks from
    original_passthrough_stdout (the generated code's own Tier 2 passthrough output),
    then verifies the expected mathematical relationship.

    Phase 1 implements: SCALING, SYMMETRY, CONSERVATION, MONOTONICITY, ADDITIVITY.
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
    sample_pattern = re.compile(r'Sample_(first10|last10|spread10):(.*)')
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
    # ADDITIVITY: check Z_stat ≈ Y_stat + offset for each statistic and sample.
    # Complements SCALING (tests superposition, not homogeneity). Most applicable
    # to reductions (mean, sum) and pointwise-identity-coefficient transforms where
    # adding a constant c to all inputs shifts all output stats by exactly c.
    # The contract's `factor` field holds the offset c (additive interpretation).
    # ==========================================================================
    elif relation_type == "ADDITIVITY":
        meta_blocks = extract_meta_blocks(variant_stdout)
        pass_blocks = extract_pass_blocks(original_passthrough_stdout)

        if not meta_blocks and not pass_blocks:
            return "SKIP", {
                "verdict": "SKIP", "relation_name": relation_name, "factor": factor,
                "blocks_compared": 0, "failures": [],
                "summary": "SKIP: No METAMORPHIC or PASSTHROUGH blocks found in outputs.",
            }

        offset = factor   # contract `factor` field carries the additive offset for ADDITIVITY
        failures = []
        blocks_compared = 0

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

                if not all(math.isfinite(v) for v in list(orig_stats.values()) + list(var_stats.values())):
                    continue  # NaN/Inf guard

                # atol anchored to |Y_stat + offset| * rtol — the expected output magnitude
                expected_max = abs(orig_stats.get('max', 0.0) + offset)
                atol = expected_max * rtol

                for stat_name, orig_val in orig_stats.items():
                    var_val = var_stats.get(stat_name)
                    if var_val is None:
                        failures.append({"label": failure_label, "stat": stat_name,
                                          "orig": orig_val, "variant": "(missing)"})
                        continue
                    expected_var = orig_val + offset
                    if not both_finite(var_val, expected_var):
                        continue
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
                    expected_vv = ov + offset
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
                f"Relation: {relation_name} (offset={offset}, rtol={rtol:.1e})\n"
                + "\n".join(lines)
            )
            verdict = "FAIL"
        else:
            summary = (
                f"PASS: {blocks_compared} block(s) satisfied ADDITIVITY relation "
                f"(offset={offset}, rtol={rtol:.1e}, atol=|Y+offset|_max*rtol per block)."
            )
            verdict = "PASS"

        return verdict, {
            "verdict": verdict, "relation_name": relation_name, "factor": offset,
            "blocks_compared": blocks_compared, "failures": failures, "summary": summary,
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
    # INVARIANCE: check Z_stat ≈ Y_stat (output unchanged under input transformation).
    # Structurally identical to SCALING with factor pinned to 1.0 — the variant input
    # has been transformed by some non-trivial T, but the algorithm's correctness
    # property is that the output is unchanged. Compare aggregate stats (min/max/mean)
    # and per-element samples; PASS if every check is within tolerance.
    # ==========================================================================
    elif relation_type == "INVARIANCE":
        inv_factor = 1.0  # By definition for INVARIANCE — output ratio is unity.
        meta_blocks = extract_meta_blocks(variant_stdout)
        pass_blocks = extract_pass_blocks(original_passthrough_stdout)

        if not meta_blocks and not pass_blocks:
            return "SKIP", {
                "verdict": "SKIP", "relation_name": relation_name, "factor": inv_factor,
                "blocks_compared": 0, "failures": [],
                "summary": "SKIP: No METAMORPHIC or PASSTHROUGH blocks found in outputs.",
            }

        failures = []
        blocks_compared = 0

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

                if not all(math.isfinite(v) for v in list(orig_stats.values()) + list(var_stats.values())):
                    continue  # NaN/Inf guard

                # atol anchored to |orig_max| * rtol — same convention as SCALING with factor=1.
                orig_max_val = abs(orig_stats.get('max', 0.0))
                atol = orig_max_val * rtol

                # Aggregate stats: Z_stat ≈ Y_stat (no factor scaling, no min/max sign-flip).
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

                # Element-wise sample comparison: Z_sample[i] ≈ Y_sample[i].
                orig_samples = parse_samples(pass_bodies[i])
                var_samples  = parse_samples(meta_bodies[i])
                for idx, (ov, vv) in enumerate(zip(orig_samples, var_samples)):
                    if not both_finite(ov, vv):
                        continue
                    if not allclose(vv, ov, rtol, atol):
                        failures.append({
                            "label": failure_label, "stat": f"sample[{idx}]",
                            "orig": ov, "expected_variant": ov,
                            "actual_variant": vv,
                            "abs_err": abs(vv - ov),
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
                f"Relation: {relation_name} (INVARIANCE, rtol={rtol:.1e})\n"
                + "\n".join(lines)
            )
            verdict = "FAIL"
        else:
            summary = (
                f"PASS: {blocks_compared} block(s) satisfied INVARIANCE relation "
                f"(output unchanged within rtol={rtol:.1e}, atol=|orig_max|*rtol per block)."
            )
            verdict = "PASS"

        return verdict, {
            "verdict": verdict, "relation_name": relation_name, "factor": inv_factor,
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
    # Maps relation_name → retry entry for Stage 0 second-pass.
    # Entry is either None (intentional no-retry, e.g. null-factor deferred)
    # or {"feedback_text": str, "failed_code": str|None}.
    # Covers both SKIP (code quality issues) and FAIL (math check failure — may be a code bug).
    _retry_info = {}
    # Max repair attempts for main-loop compile/execute failures (3 total: 1 original + 2 repairs).
    # Stage 0 is unaffected — it uses the separate Stage 0 retry block below.
    _MAX_VARIANT_REPAIR_RETRIES = 2

    for relation in active_relations:
        relation_name = relation.get("name", "unknown")
        relation_type = relation.get("type", "UNKNOWN")
        print(f"\n----- Tier 3: testing relation '{relation_name}' ({relation_type}) -----")
        # Set per-iteration flag — repair loop sets this True to skip the normal execute block
        # when the repaired variant has already been compiled and executed successfully.
        _skip_execute = False
        _repaired_variant_stdout = None

        # Skip null-factor SCALING/SYMMETRY (spatial reflection etc. require element-wise
        # comparison across the full array — deferred to Phase 3).
        if relation.get("factor") is None and relation_type in ("SCALING", "SYMMETRY"):
            print(f"  [Tier 3] '{relation_name}' has null factor — spatial {relation_type} "
                  f"requires Phase 3 element-wise comparison; SKIP")
            if code_version is None:  # Fix A: never overwrite frozen Stage 0 source_status in main loop
                relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "null factor: spatial SYMMETRY/SCALING deferred to Phase 3"})
            _retry_info[relation_name] = None  # Phase 3 deferred — intentional SKIP, no retry
            if code_version is not None:  # Fix B: active relation untestable → FAIL (defensive)
                first_failure_diagnostic = {
                    "verdict": "FAIL", "relation_name": relation_name,
                    "reason": results[-1]["reason"],
                    "summary": f"FAIL: active relation '{relation_name}' could not be tested on generated code ({results[-1]['reason']}).",
                }
                break
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
            if code_version is None:  # Fix A: never overwrite frozen Stage 0 source_status in main loop
                relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "variant insertion failed"})
            _retry_info[relation_name] = {
                "feedback_text": (
                    "Your previous attempt failed to produce valid instrumented code "
                    "(the LLM response contained no code fences or extractable C++ code). "
                    "Output the complete modified source wrapped in ```cpp ... ``` fences. "
                    "Do NOT use placeholder comments or omit any sections."
                ),
                "failed_code": None,  # no code was extracted
            }
            if code_version is not None:  # Fix B: active relation untestable → FAIL
                first_failure_diagnostic = {
                    "verdict": "FAIL", "relation_name": relation_name,
                    "reason": results[-1]["reason"],
                    "summary": f"FAIL: active relation '{relation_name}' could not be tested on generated code ({results[-1]['reason']}).",
                }
                break
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
            if code_version is None:  # Fix A: never overwrite frozen Stage 0 source_status in main loop
                relation["source_status"] = "SKIP"
            results.append({"name": relation_name, "verdict": "SKIP",
                             "reason": "compile failed",
                             "compile_error": compile_err[:300]})
            _retry_info[relation_name] = {
                "feedback_text": (
                    f"Your previous attempt produced code that failed to compile:\n"
                    f"```\n{compile_err[:400]}\n```\n"
                    f"Fix the compile error(s) and output the corrected complete source."
                ),
                "failed_code": variant_code,
            }
            if code_version is None:  # Stage 0: skip relation, continue to next
                continue
            # Main loop: attempt repair before marking as instrument_error.
            # Use repair_mode=True so the LLM patches the broken variant (not regenerates from scratch).
            # template_code (the validated source variant) is the structural reference.
            _repair_succeeded = False
            _current_broken = variant_code
            _current_compile_err = compile_err
            for _repair_idx in range(_MAX_VARIANT_REPAIR_RETRIES):
                print(f"  [Tier 3] Compile repair attempt {_repair_idx + 1}/{_MAX_VARIANT_REPAIR_RETRIES} "
                      f"for '{relation_name}'...")
                _repair_feedback = (
                    f"The previous variant failed to compile:\n"
                    f"```\n{_current_compile_err[:400]}\n```"
                    f"\n\n[PREVIOUS BROKEN VARIANT]\n```cpp\n{_current_broken}\n```"
                )
                _repaired_code = code_insert_metamorphic_variant(
                    _current_broken, relation,
                    llm_source, model, num_ctx, system_prompt, display_response,
                    template_code=template_code,
                    feedback=_repair_feedback,
                    verifyprompt=verifyprompt,
                    repair_mode=True,
                )
                if not _repaired_code:
                    print(f"  [Tier 3] Repair {_repair_idx + 1}: no code extracted — retrying")
                    continue
                _current_broken = _repaired_code
                with open(variant_path, 'w') as _rvf:
                    _rvf.write(_repaired_code)
                _r_compile_err, _r_variant_exe = lassi.code_manager.compile_code(
                    variant_path, target_lang, code_compiler, code_compiler_kwds
                )
                if _r_compile_err:
                    print(f"  [Tier 3] Repair {_repair_idx + 1}: still compile error")
                    _current_compile_err = _r_compile_err
                    continue
                # Repair compiled — execute immediately so we can fall through to Step 5.
                print(f"  [Tier 3] Repair {_repair_idx + 1}: compile OK — executing...")
                try:
                    _r_exe_abs = os.path.abspath(_r_variant_exe)
                    _r_exe_dir = os.path.dirname(_r_exe_abs)
                    _r_exe_base = os.path.basename(_r_exe_abs)
                    _r_result = subprocess.run(
                        ["./" + _r_exe_base] + list(exe_input_parameters),
                        cwd=_r_exe_dir,
                        capture_output=True, text=True, timeout=120
                    )
                    _repaired_variant_stdout = _r_result.stdout
                    any_ran = True
                    print(f"\n----- < BEGIN: Tier 3 repaired variant stdout > -----\n")
                    print(_repaired_variant_stdout)
                    print(f"----- < END: Tier 3 repaired variant stdout > -----\n")
                    if _r_result.returncode != 0:
                        print(f"  [Tier 3] Repair {_repair_idx + 1}: execution warning: "
                              f"return code {_r_result.returncode}")
                    _repair_succeeded = True
                    # Remove the SKIP result appended above — repair succeeded, proceed to check.
                    results.pop()
                    break
                except (subprocess.TimeoutExpired, Exception) as _re:
                    print(f"  [Tier 3] Repair {_repair_idx + 1}: execute failed: {_re}")
                    # Repaired code compiled but didn't run — treat repaired code as next broken base.
                    continue
            if _repair_succeeded:
                # Repair compiled and executed — skip the normal execute block (Step 4) and
                # fall through to Step 5 with the repaired stdout.
                _skip_execute = True
                variant_stdout = _repaired_variant_stdout
            else:
                # All repair attempts failed — cannot verify this relation; instrument_error.
                print(f"  [Tier 3] All {_MAX_VARIANT_REPAIR_RETRIES} repair attempts failed "
                      f"for '{relation_name}' — instrument_error")
                # Update the last SKIP result to reflect the final state.
                if results and results[-1].get("name") == relation_name:
                    results[-1]["reason"] = f"compile failed (instrument_error after {_MAX_VARIANT_REPAIR_RETRIES} repair attempts)"
                first_failure_diagnostic = {
                    "verdict": "FAIL", "relation_name": relation_name,
                    "reason": "instrument_error",
                    "summary": (
                        f"FAIL: active relation '{relation_name}' could not be tested on generated "
                        f"code (compile error; instrument_error after {_MAX_VARIANT_REPAIR_RETRIES} "
                        f"repair attempts). This is an instrumentation failure, not a mathematical "
                        f"property violation."
                    ),
                }
                break

        # --- Step 4: Execute (lightweight — no power measurement, no wait time) ---
        # Variants run from the binary's directory to match execute_code behavior.
        # Skipped when _skip_execute=True (repair loop already compiled and executed the variant).
        if not _skip_execute:
            _exec_error_str = None
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
                any_ran = True
                print(f"\n----- < BEGIN: STANDARD OUTPUT FROM Tier 3 VARIANT EXECUTION > -----\n")
                print(variant_stdout)
                print(f"----- < END: Tier 3 Variant Output > -----\n")
                if result.returncode != 0:
                    print(f"  [Tier 3] Execution warning: return code {result.returncode}")
                    if result.stderr:
                        print(f"  stderr: {result.stderr[:400]}")
            except subprocess.TimeoutExpired:
                _exec_error_str = "execution timeout (>120s)"
            except Exception as e:
                _exec_error_str = f"execution error: {e}"

            if _exec_error_str:
                print(f"  [Tier 3] Execution FAILED for '{relation_name}' ({_exec_error_str}) — SKIP")
                if code_version is None:  # Fix A: never overwrite frozen Stage 0 source_status in main loop
                    relation["source_status"] = "SKIP"
                results.append({"name": relation_name, "verdict": "SKIP",
                                 "reason": _exec_error_str})
                _retry_info[relation_name] = {
                    "feedback_text": (
                        f"Your previous attempt failed at runtime: {_exec_error_str}. "
                        "Check for infinite loops, incorrect cudaDeviceSynchronize placement, "
                        "or unbounded memory access patterns in the instrumentation."
                    ),
                    "failed_code": variant_code,
                }
                if code_version is None:  # Stage 0: skip relation, continue to next
                    continue
                # Main loop: attempt repair before marking as instrument_error.
                _repair_succeeded = False
                _current_broken = variant_code
                for _repair_idx in range(_MAX_VARIANT_REPAIR_RETRIES):
                    print(f"  [Tier 3] Execute repair attempt {_repair_idx + 1}/{_MAX_VARIANT_REPAIR_RETRIES} "
                          f"for '{relation_name}'...")
                    _repair_feedback = (
                        f"The previous variant failed at runtime: {_exec_error_str}.\n"
                        "Check for infinite loops, incorrect cudaDeviceSynchronize placement, "
                        "or unbounded memory access in the instrumentation code."
                        f"\n\n[PREVIOUS BROKEN VARIANT]\n```cpp\n{_current_broken}\n```"
                    )
                    _repaired_code = code_insert_metamorphic_variant(
                        _current_broken, relation,
                        llm_source, model, num_ctx, system_prompt, display_response,
                        template_code=template_code,
                        feedback=_repair_feedback,
                        verifyprompt=verifyprompt,
                        repair_mode=True,
                    )
                    if not _repaired_code:
                        print(f"  [Tier 3] Execute repair {_repair_idx + 1}: no code extracted")
                        continue
                    _current_broken = _repaired_code
                    with open(variant_path, 'w') as _rvf:
                        _rvf.write(_repaired_code)
                    _r_compile_err, _r_variant_exe = lassi.code_manager.compile_code(
                        variant_path, target_lang, code_compiler, code_compiler_kwds
                    )
                    if _r_compile_err:
                        print(f"  [Tier 3] Execute repair {_repair_idx + 1}: compile error after fix")
                        continue
                    try:
                        _r_exe_abs = os.path.abspath(_r_variant_exe)
                        _r_exe_dir = os.path.dirname(_r_exe_abs)
                        _r_exe_base = os.path.basename(_r_exe_abs)
                        _r_result = subprocess.run(
                            ["./" + _r_exe_base] + list(exe_input_parameters),
                            cwd=_r_exe_dir,
                            capture_output=True, text=True, timeout=120
                        )
                        _repaired_variant_stdout = _r_result.stdout
                        any_ran = True
                        print(f"\n----- < BEGIN: Tier 3 execute-repaired variant stdout > -----\n")
                        print(_repaired_variant_stdout)
                        print(f"----- < END: Tier 3 execute-repaired variant stdout > -----\n")
                        if _r_result.returncode != 0:
                            print(f"  [Tier 3] Execute repair {_repair_idx + 1}: warning: "
                                  f"return code {_r_result.returncode}")
                        _repair_succeeded = True
                        # Remove the SKIP result appended above — repair succeeded.
                        results.pop()
                        variant_stdout = _repaired_variant_stdout
                        break
                    except (subprocess.TimeoutExpired, Exception) as _re:
                        print(f"  [Tier 3] Execute repair {_repair_idx + 1}: still failing: {_re}")
                        _exec_error_str = str(_re)
                        continue
                if not _repair_succeeded:
                    print(f"  [Tier 3] All {_MAX_VARIANT_REPAIR_RETRIES} execute repair attempts failed "
                          f"for '{relation_name}' — instrument_error")
                    if results and results[-1].get("name") == relation_name:
                        results[-1]["reason"] = (f"{_exec_error_str} "
                                                  f"(instrument_error after {_MAX_VARIANT_REPAIR_RETRIES} repair attempts)")
                    first_failure_diagnostic = {
                        "verdict": "FAIL", "relation_name": relation_name,
                        "reason": "instrument_error",
                        "summary": (
                            f"FAIL: active relation '{relation_name}' could not be tested on generated "
                            f"code ({_exec_error_str}; instrument_error after {_MAX_VARIANT_REPAIR_RETRIES} "
                            f"repair attempts). This is an instrumentation failure, not a mathematical "
                            f"property violation."
                        ),
                    }
                    break  # stop testing further relations for this code

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

        # Fail-fast on FAIL — but ONLY in the main loop (generated codes, code_version != None).
        # In Stage 0 (source filter, code_version=None) ALL relations must be evaluated
        # independently: just because one relation fails for the source does not mean another
        # should be skipped. Stopping early in Stage 0 leaves subsequent relations with
        # source_status=None, which prevents them from ever being applied to generated codes.
        if verdict == "FAIL":
            first_failure_diagnostic = diagnostic
            if code_version is not None:
                break  # main loop: fail-fast, stop testing further relations
            # Stage 0: record failure, store retry info with full diagnostic, continue
            if code_version is None:
                _diag_summary = diagnostic.get("summary", "No summary available.")
                # Heuristic: if actual ≈ 2× expected, the transformation was applied twice
                # (aliased pointer double-scaling, e.g. data[] + wall[][] in pathfinder).
                _hint = ""
                for _f in diagnostic.get("failures", []):
                    _av = _f.get("actual_variant")
                    _ev = _f.get("expected_variant")
                    if _av is not None and _ev is not None:
                        try:
                            _ratio = abs(float(_av)) / abs(float(_ev)) if float(_ev) != 0 else None
                        except (TypeError, ZeroDivisionError):
                            _ratio = None
                        if _ratio is not None and 1.8 < _ratio < 2.2:
                            _hint = (
                                "\nNote: actual_variant ≈ 2× expected_variant on at least one statistic. "
                                "This pattern typically indicates a double-transformation bug where two "
                                "variables alias the same memory (e.g., `wall[n] = data + cols*n`). "
                                "Transforming both `data[]` and `wall[][]` scales the same elements twice. "
                                "Fix: apply the transformation ONCE — either loop over the flat `data[]` "
                                "array OR iterate via `wall[][]`, but NOT both."
                            )
                            break
                _retry_info[relation_name] = {
                    "feedback_text": (
                        f"Your previous attempt FAILED the metamorphic check:\n"
                        f"{_diag_summary}{_hint}\n\n"
                        "Review the variant code carefully for: aliased-buffer double-transformation, "
                        "incorrect observation point placement, or wrong scaling scope."
                    ),
                    "failed_code": variant_code,
                }

    # --- Stage 0 retry: give SKIP/FAIL relations one more attempt with targeted feedback ---
    # Covers SKIP (code quality: insertion failure, compile error, execute error) AND FAIL
    # (math check failure that may be caused by a code bug, e.g. aliased-buffer double-scaling).
    # Only runs during Stage 0 (code_version is None). Main loop uses pre-filtered relations
    # (all PASS at Stage 0) — failures there are generated-code issues, not Stage 0 filter bugs.
    # Single retry per relation (user requirement: avoid endless retrying on bad math).
    if code_version is None:
        for relation in active_relations:
            if relation.get("source_status") not in ("SKIP", "FAIL"):
                continue

            relation_name = relation.get("name", "unknown")
            relation_type = relation.get("type", "UNKNOWN")

            # Null-factor SYMMETRY/SCALING: Phase 3 deferred — no variant code was generated,
            # no code-quality feedback exists, and the skip is intentional. No retry.
            _retry_entry = _retry_info.get(relation_name)
            if _retry_entry is None:
                continue

            # Build combined feedback: diagnostic text + the actual failed code (if any)
            _feedback_text = _retry_entry["feedback_text"]
            _failed_code = _retry_entry.get("failed_code")
            if _failed_code:
                combined_feedback = (
                    _feedback_text
                    + "\n\n[PREVIOUS ATTEMPT CODE — review this carefully for the issue described above:]\n"
                    "```cpp\n" + _failed_code + "\n```"
                )
            else:
                combined_feedback = _feedback_text

            print(f"\n----- Tier 3 Stage 0 retry: '{relation_name}' ({relation_type}) with feedback -----")

            # source_variant_path may not exist (insertion failed) or contain bad code (compile
            # failed) — always use template_code=None for Stage 0 retries (no valid reference yet).
            source_variant_path = f"{variant_dir}/{file_name}_mt_{relation_name}_source.cu"
            variant_path = source_variant_path

            variant_code = code_insert_metamorphic_variant(
                source_code, relation,
                llm_source, model, num_ctx, system_prompt,
                display_response,
                template_code=None,
                feedback=combined_feedback,
                verifyprompt=verifyprompt
            )

            # Helper: replace the first-pass result entry for this relation
            def _replace_result(new_entry, _results=results):
                for i, r in enumerate(_results):
                    if r["name"] == new_entry["name"]:
                        _results[i] = new_entry
                        return
                _results.append(new_entry)

            if not variant_code:
                print(f"  [Tier 3 retry] Variant insertion FAILED again for '{relation_name}' — remains SKIP")
                _replace_result({"name": relation_name, "verdict": "SKIP",
                                  "reason": "variant insertion failed (retry)"})
                continue

            # Write, compile, execute, check — same flow as first pass
            with open(variant_path, 'w') as _vf:
                _vf.write(variant_code)
            print(f"  [Tier 3 retry] Variant written to {variant_path}")

            _compile_err, _variant_exe = lassi.code_manager.compile_code(
                variant_path, target_lang, code_compiler, code_compiler_kwds
            )
            if _compile_err:
                print(f"  [Tier 3 retry] Compile FAILED again for '{relation_name}' — remains SKIP")
                relation["source_status"] = "SKIP"
                _replace_result({"name": relation_name, "verdict": "SKIP",
                                  "reason": "compile failed (retry)",
                                  "compile_error": _compile_err[:300]})
                continue

            any_ran = True
            try:
                _exe_abs = os.path.abspath(_variant_exe)
                _exe_dir = os.path.dirname(_exe_abs)
                _exe_base = os.path.basename(_exe_abs)
                _result = subprocess.run(
                    ["./" + _exe_base] + list(exe_input_parameters),
                    cwd=_exe_dir,
                    capture_output=True, text=True, timeout=120
                )
                _variant_stdout = _result.stdout
                print(f"\n----- < BEGIN: Tier 3 retry variant stdout > -----\n")
                print(_variant_stdout)
                print(f"----- < END: Tier 3 retry variant stdout > -----\n")
                if _result.returncode != 0:
                    print(f"  [Tier 3 retry] Execution warning: return code {_result.returncode}")
                    if _result.stderr:
                        print(f"  stderr: {_result.stderr[:400]}")
            except subprocess.TimeoutExpired:
                print(f"  [Tier 3 retry] Execution TIMEOUT for '{relation_name}' — remains SKIP")
                relation["source_status"] = "SKIP"
                _replace_result({"name": relation_name, "verdict": "SKIP",
                                  "reason": "execution timeout (retry)"})
                continue
            except Exception as _e:
                print(f"  [Tier 3 retry] Execution ERROR for '{relation_name}': {_e} — remains SKIP")
                relation["source_status"] = "SKIP"
                _replace_result({"name": relation_name, "verdict": "SKIP",
                                  "reason": f"execution error (retry): {_e}"})
                continue

            _tolerance = relation.get("tolerance", 1e-5)
            _verdict, _diagnostic = check_metamorphic_relation(
                _variant_stdout, original_passthrough_stdout, relation, _tolerance
            )
            relation["source_status"] = _verdict  # update for Stage 0 filter
            _replace_result({"name": relation_name, "verdict": _verdict,
                              "diagnostic": _diagnostic})

            print(f"\n----- Tier 3 retry [{relation_name}]: {_verdict} -----")
            if _diagnostic.get("summary"):
                print(_diagnostic["summary"])

            # Track first failure (no fail-fast in Stage 0 — evaluate all independently)
            if _verdict == "FAIL" and not first_failure_diagnostic:
                first_failure_diagnostic = _diagnostic

    # --- Determine overall verdict ---
    verdicts = [r["verdict"] for r in results]
    if "FAIL" in verdicts:
        overall_verdict = "FAIL"
    elif code_version is not None and "SKIP" in verdicts:
        # Main loop: binary PASS/FAIL only — an active relation that could not be tested
        # (variant insertion, compile, or execution failure) counts as FAIL.
        # Stage 0 is unaffected: there, SKIP on one relation is inconclusive for that relation.
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
    Remove all LASSI-VERIFY ASSERTIONS, PASSTHROUGH, and WITNESS_CHAIN blocks from a
    C++ source string. Returns uninstrumented code for clean performance measurement.

    Block delimiter format (from Tier 1 / Tier 2 / witness-chain instrumentation):
      Opening: // >>> BEGIN LASSI-VERIFY ASSERTIONS ...    >>>
               // >>> BEGIN LASSI-VERIFY PASSTHROUGH ...   >>>
               // >>> BEGIN LASSI-VERIFY WITNESS_CHAIN ... >>>
      Closing: // <<< END LASSI-VERIFY ASSERTIONS <<<
               // <<< END LASSI-VERIFY PASSTHROUGH <<<
               // <<< END LASSI-VERIFY WITNESS_CHAIN <<<

    Additionally applies a safety-net pass: any raw `_t1_chain_*<<<...>>>(...);` kernel-launch
    call site that escaped the delimiter strip is removed. The witness-chain harness template
    requires every helper-kernel call to be enclosed in delimiter comments, but LLMs that
    restructure the timed loop into a CUDA Graph capture region (or similar) sometimes place
    the launches without surrounding delimiters. Without this safety net, the device function
    definitions strip correctly but the dangling host call sites cause the clean rebuild to
    fail with "undefined symbol" compile errors. (See planning/chain_verify_output_poisoning_redesign.md
    for the failure-mode case study from KERNEL_TEST2_CHACHA_1 Trial 1 v12.)
    """
    # Match opening: // >>> BEGIN LASSI-VERIFY (ASSERTIONS|PASSTHROUGH|WITNESS_CHAIN) ... \n
    # Match closing:  // <<< END LASSI-VERIFY (ASSERTIONS|PASSTHROUGH|WITNESS_CHAIN) ...  \n
    # The closing pattern intentionally does NOT require a specific trailing arrow sequence
    # (<<<  vs  >>>) because LLMs occasionally generate the wrong closing arrows.
    # The [^\n]* absorbs any trailing characters (including <<< or >>> variants).
    pattern = re.compile(
        r'// >>> BEGIN LASSI-VERIFY (?:ASSERTIONS|PASSTHROUGH|WITNESS_CHAIN)[^\n]*\n'
        r'.*?'
        r'// <<< END LASSI-VERIFY (?:ASSERTIONS|PASSTHROUGH|WITNESS_CHAIN)[^\n]*\n?',
        re.DOTALL
    )
    stripped = pattern.sub('', source_code)
    # Safety net: remove any raw _t1_chain_* kernel-launch call site that survived the
    # delimiter strip (whole line). Matches lines like:
    #   _t1_chain_write_sentinel<<<1,1>>>(d_output);
    #   _t1_chain_update<<<1,1,0,stream>>>(d_output, n);
    stripped = re.sub(
        r'^[ \t]*_t1_chain_\w+<<<[^>]*>>>\s*\([^)]*\);[ \t]*\n?',
        '', stripped, flags=re.MULTILINE)
    return stripped