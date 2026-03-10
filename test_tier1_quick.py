#!/usr/bin/env python3
"""
Quick Tier 1 Validation Test
Tests LASSI-Verify assertion insertion without full EE pipeline
"""

import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# Import LASSI modules
import lassi.verify
import lassi.llm_config
import lassi.code_manager
import lassi.pipeline_methods
from lassi.prompt_dictionary import PromptDictionary
from lassi.gpu_detector import detect_gpu_vendors

def test_tier1_assertions():
    """Minimal test of Tier 1 invariant assertion insertion"""

    print("\n" + "="*80)
    print("TIER 1 VALIDATION TEST - Invariant Assertion Insertion")
    print("="*80 + "\n")

    # Detect GPU
    vendors = detect_gpu_vendors()
    if not vendors:
        print("❌ No GPU detected. Exiting.")
        return False

    detected_vendor = vendors[0]['vendor']
    print(f"✅ Detected GPU: {detected_vendor.upper()}\n")

    # Load prompts
    prompts = PromptDictionary()

    # Configuration
    source_lang = "CUDA"
    app_name = "jacobi"
    source_file = "source_codes/HeCBench/jacobi/jacobi-cuda_main.cu"

    # LLM settings (using minimal config)
    llm_source = "argo"
    model = "gpto4mini"
    num_ctx = 200000
    verifyprompt = True
    display_response = True

    # Read source code
    print(f"📖 Reading source code: {source_file}")
    with open(source_file, 'r') as f:
        input_code = f.read()

    print(f"   Lines of code: {len(input_code.splitlines())}\n")

    # Generate code analysis
    print("🔍 Step 1: Generate code analysis...")
    system_prompt = prompts.get_system_prompt("CUDA_ee")
    original_context_content_prompt = ""

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

    print(f"\n✅ Code analysis generated ({len(code_analysis)} chars)\n")

    # Insert assertions
    print("🔧 Step 2: Insert invariant assertions...")
    instrumented_code = lassi.verify.code_insert_asserts(
        input_code,
        llm_source,
        model,
        num_ctx,
        system_prompt,
        code_analysis,
        display_response,
        verifyprompt=verifyprompt
    )

    print(f"\n✅ Assertions inserted ({len(instrumented_code)} chars)\n")

    # Save instrumented code
    output_dir = "generated_codes/tier1_test"
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now(ZoneInfo("America/Chicago"))
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/jacobi_tier1_{timestamp}.cu"

    with open(output_file, 'w') as f:
        f.write(instrumented_code)

    print(f"💾 Saved instrumented code to: {output_file}\n")

    # Compile test
    print("🔨 Step 3: Compile instrumented code...")
    code_compiler = "nvcc"
    code_compiler_kwds = "-std=c++14 -Xcompiler -Wall -arch=sm_89 -O3"

    compile_result, execute_file = lassi.code_manager.compile_code(
        output_file,
        source_lang,
        code_compiler,
        code_compiler_kwds
    )

    if compile_result == "":
        print(f"✅ Compilation successful: {execute_file}\n")
    else:
        print(f"❌ Compilation failed:\n{compile_result}\n")
        return False

    # Execute test
    print("▶️  Step 4: Execute instrumented code...")
    exe_result = os.system(f"{execute_file}")

    if exe_result == 0:
        print("✅ Execution successful - assertions passed!\n")
    else:
        print(f"❌ Execution failed with exit code: {exe_result}\n")
        return False

    # Verify assertions were inserted
    print("🔍 Step 5: Verify assertion instrumentation...")
    assert_count = instrumented_code.count("LASSI-VERIFY ASSERTIONS")
    assert_instrumentation_count = instrumented_code.count("LASSI_INSTRUMENTATION")

    print(f"   Found {assert_count} LASSI-VERIFY assertion blocks")
    print(f"   Found {assert_instrumentation_count} LASSI_INSTRUMENTATION markers")

    if assert_count > 0 or assert_instrumentation_count > 0:
        print("\n✅ TIER 1 VALIDATION PASSED!\n")
        return True
    else:
        print("\n⚠️  Warning: No assertion markers found in instrumented code\n")
        return False


if __name__ == "__main__":
    success = test_tier1_assertions()
    exit(0 if success else 1)
