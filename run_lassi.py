#!/usr/bin/env python3
"""
run_lassi.py — Standalone entry point for the LASSI-Verify pipeline.

Equivalent to running the LASSI_power_pipeline_v4.ipynb notebook interactively,
but works from the command line without Jupyter.  The monitor web dashboard
(localhost:5007) replaces notebook cell outputs.

Usage:
    python run_lassi.py                         # uses config.yaml (default)
    python run_lassi.py --config my_exp.yaml    # custom config file
    python run_lassi.py --no-monitor            # skip web dashboard
    python run_lassi.py --help
"""

import argparse
import os
import sys

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="LASSI-Verify pipeline runner",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument(
    "--config",
    default="config.yaml",
    metavar="PATH",
    help="Path to YAML config file (default: config.yaml)",
)
parser.add_argument(
    "--no-monitor",
    action="store_true",
    help="Skip launching the live monitor web dashboard",
)
args = parser.parse_args()

# ── Config loading ─────────────────────────────────────────────────────────────
try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed.  Run: pip install pyyaml")
    sys.exit(1)

if not os.path.exists(args.config):
    print(f"ERROR: Config file not found: {args.config}")
    print("Copy config.yaml to your working directory or pass --config PATH")
    sys.exit(1)

with open(args.config, "r") as _f:
    _cfg = yaml.safe_load(_f)

# ── GPU Detection (Cell 1) ─────────────────────────────────────────────────────
import subprocess

from lassi.gpu_detector import detect_gpu_vendors

# Detect available GPUs
vendors = detect_gpu_vendors()
if not vendors:
    print("❌ No GPUs detected! LASSI requires GPU hardware.")
    raise SystemExit("Pipeline stopped - no GPU hardware found")

detected_vendor = vendors[0]['vendor']
gpu_count = vendors[0]['count']
print(f"🔍 Detected: {detected_vendor.upper()} with {gpu_count} GPU(s)")

# Vendor-specific initialization
if detected_vendor == "nvidia":
    # NVIDIA-specific setup
    import pynvml
    pynvml.nvmlInit()
    print("NVIDIA Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    pynvml.nvmlShutdown()

    # Show NVIDIA system info
    subprocess.run(["nvidia-smi"], check=False)
    subprocess.run(["nvcc", "--version"], check=False)

elif detected_vendor == "amd":
    # AMD-specific setup
    from lassi.amd_power import amd_report_gpu_memory

    # Show AMD system info
    subprocess.run(["rocminfo"], check=False)
    subprocess.run(["rocm-smi"], check=False)
    subprocess.run(["hipcc", "--version"], check=False)

print(f"\n✅ {detected_vendor.upper()} GPU environment ready for LASSI!")

# ── Imports (Cell 3, Jupyter magic removed) ────────────────────────────────────
import csv
import json
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# Import LASSI modules
import lassi.llm_config
import lassi.pipeline_v4
from lassi.format_colors import color
from lassi.monitor import log_figure, start_monitor
from lassi.prompt_dictionary import PromptDictionary

# ── Monitor (Cell 4) ──────────────────────────────────────────────────────────
# Starts a local web server and streams all pipeline print() output to a browser
# tab in real-time. Open http://localhost:5007 in a browser tab.
# SSH remote access: forward port 5007, then open http://localhost:5007 locally.
if not args.no_monitor:
    start_monitor()   # default port 5007 — pass port=XXXX to change


# ── Experiment configuration (Cell 7) ─────────────────────────────────────────

def experimental_setup(cfg):
    # Load config file with source application names and input parameters
    with open("source_apps.json", "r") as f:
        source_apps = json.load(f)

    """
    [1] ENTER APPLICATION FOLDER NAME FROM HeCBench
        >> Execution input parameters provided.

    *** USE FOR LASSI-Verfify experiment: ***
    -- colorwheel ["10000", "256", "500000"] ["10000", "8", "1"]  # Usage: <range> <size> <repeat>
     (this one may show a lot of failures)

    HeCBench apps:
    -- matrix-rotate ["30000", "100"]  ["10000", "1"]  # Usage: %s <matrix size> <repeat>
    -- jacobi [""] (none required)
    -- layout ["2500"] ["1"]
    -- atomicCost ["16", "10"] ["1"]
    -- segment-reduce ["16384", "100"] (a replacement for atomicCost, same "Data compression and reduction" category in HeCBench)
       --> (may be trivial for LASSI-Verify)
    -- dense-embedding ["100000", "128", "100"] ["10000", "8", "1"]
    -- pathfinder ["100000", "10000", "2500"] ["100000", "1000", "5"] ["10000", "1000", "1000"]  # Usage: <column length> <row length> <pyramid_height>
    -- bsearch ["8388608", "1000"] ["10000", "1"]  # Usage: <number of elements> <repeat>
    -- entropy ["10000", "1024", "1"] ["8192", "8192", "500"]   # Usage: <width> <height> <repeat>
        --> can't get a passing source_verify
    -- colorwheel ["10000", "256", "500000"] ["10000", "8", "1"]  # Usage: <range> <size> <repeat>
    -- randomAccess ["10"] ["1"]  # Usage: <repeat>
    -- lid-driven-cavity [""]
    -- murmurhash3 ["750000, 500"]
    -- chacha20 ["300000"]
    -- jaccard ["1024, 512, 1000"]
    -- keogh ["256", "22500000", "100"]
    -- floydwarshall ["2048", "100", "16"]
    -- extrema ["750"]
    -- threadfence ["100", "250000000"]
    -- all-pairs-distance ["10000"]
    -- marchingCubes ["100"]

    Altis apps:
    -- level1/bfs ["--size 3"]
    -- level2/raytracing [""]
    -- level2/mandelbrot ["--imageSize 1024 --iterations 100000"]

    Miniapps:
    -- miniMDock ["-lfile", "./input/7cpa/7cpa_ligand.pdbqt", "-nrun 500"]
    -- XSBench ["-m", "event", "-l", "1700000000"]

    """

    # ****************************
    # ---> EXPERIMENT UPDATE FIELD (now loaded from config.yaml)

    # Enter source application to process through LASSI
    source_collection = cfg["app"]["collection"]   # "HeCBench" | "Miniapps"
    source_app        = cfg["app"]["name"]          # e.g. "jacobi"
    make_script       = cfg["app"]["make_script"]   # Used only for Miniapps
    build_script      = cfg["app"]["build_script"]  # False (HeCBench), True for Altis only


    """
    Valid source code folder names (under source_codes):
    1. HeCBench
    2. altis/src/cuda
    3. miniMDock/bin/
    """
    source_folder = source_apps[source_collection]["source_folder"]  # "altis/src/cuda/level2"    #  "HeCBench" "altis/src/cuda/level1"
    app_folder_name = source_apps[source_collection][source_app]["app_folder"]
    lines_of_code = source_apps[source_collection][source_app]["code_lines"]
    if make_script:
        app_name = source_app  # app_folder_name
        file_name = "Main"  # kernels
    else:
        app_name = "mt_" + app_folder_name + "-cuda_main"
        file_name = ""


    """
    [2] SELECT THE CODE TRANSLATION MAPPING DIRECTION EXPERIMENT CONFIGURATION

    CUDA -- OMP -- HIP -- SYCL -- FORTRAN
    """
    # These should be the same if *not* doing translation.
    source_lang = cfg["language"]["source"]
    target_lang = cfg["language"]["target"]

    verifyprompt = cfg["pipeline"]["verifyprompt"]  # True or False -- Print out LLM prompt in stdout to preview

    # TOGGLE USE CODE KNOWLEDGE CONTEXT FOR TARGET
    use_context_summaries_only = True  # True or False
    few_shot_learning = False  # True or False

    # LLM judge code analysis (pre-cursor to LASSI-Verify)
    run_code_analysis = cfg["verification"]["run_code_analysis"]  # True or False

    # THESE WILL NO LONGER BE OPTIONS
    # test_code_first = True  # True or False
    use_context = False  # True or False -- NOT USED IN v2, v3
    use_self_prompting = False  # True or False -- NOT USED IN v2, v3
    baseline_oneshot_goal = True  # True or False -- NOT USED IN v2, v3

    # -----------------------------------------------------------------------
    # TASK MODE CONFIGURATION
    # Set these two flags together depending on the task:
    #
    #   ENERGY EFFICIENCY (EE) task:
    #     power_measure    = True   -- measure GPU power/energy during execution
    #     yes_power_feedback = True -- run LASSI self-improvement loop using energy as the optimization metric
    #
    #   RUNTIME PERFORMANCE task:
    #     power_measure    = False  -- skip power measurement, use wall-clock duration instead
    #     yes_power_feedback = True -- run LASSI self-improvement loop using runtime as the optimization metric
    #
    #   CODE TRANSLATION task (one-shot, no iterative improvement):
    #     power_measure    = False  -- no measurement needed
    #     yes_power_feedback = False -- skip feedback loop entirely (generate once, compile, execute, judge, done)
    #     NOTE: translation mode still has a known infinite-loop-on-success issue when yes_power_feedback=False
    # -----------------------------------------------------------------------
    power_measure = cfg["task"]["power_measure"]  # True (EE task) | False (runtime or translation task)
    vary_llm_temp = cfg["pipeline"]["vary_llm_temp"]  # True or False

    # TOGGLE LASSI SELF-IMPROVEMENT FEEDBACK LOOP
    yes_power_feedback = cfg["task"]["yes_power_feedback"]  # True (EE or runtime task) | False (translation task, one-shot only)

    # -----------------------------------------------------------------------
    # LASSI-VERIFY TIER 2 CONFIGURATION
    # run_tier2=True: insert passthrough observation blocks and validate empirical equivalence
    # tier2_tolerance: relative floating-point tolerance for passthrough comparison (default 1e-6)
    # -----------------------------------------------------------------------
    run_tier2       = cfg["verification"]["run_tier2"]       # True to enable Tier 2 empirical equivalence checking
    tier2_tolerance = cfg["verification"]["tier2_tolerance"]  # Relative tolerance for passthrough block comparison

    # LASSI-VERIFY TIER 3 CONFIGURATION
    # run_tier3=True: generate metamorphic contract and run variant filter (requires run_tier2=True)
    # tier3_tolerance: relative tolerance for metamorphic relation checks (default 1e-5, slightly looser than Tier 2)
    # -----------------------------------------------------------------------
    run_tier3       = cfg["verification"]["run_tier3"]       # True to enable Tier 3 metamorphic testing
    tier3_tolerance = cfg["verification"]["tier3_tolerance"]  # Relative tolerance for metamorphic relation comparison

    # -----------------------------------------------------------------------
    # LLM-AS-A-JUDGE CONFIGURATION
    # run_llm_judge=True: enable LLM functional equivalence judge (pre-Tier-1/2 stop-gap)
    # run_llm_judge=False: pure three-tier mode — Tier 1 + Tier 2 determine validity only
    # -----------------------------------------------------------------------
    run_llm_judge = cfg["verification"]["run_llm_judge"]  # True to enable LLM judge; False for pure three-tier LASSI-Verify mode

    # ASSIGN CODE COMPILE COMMAND
    code_build_script = ""
    code_compiler = ""
    code_compiler_kwds = ""
    if source_lang == "CUDA":
        if build_script:
            code_build_script = "build_" + app_name
        elif make_script:
            if source_app == "miniMDock":
                code_compiler = "make"
                code_compiler_kwds = "API=CUDA DEVICE=GPU CARD=NVIDIA NUMWI=64"
            elif source_app == "XSBench":
                code_compiler = "cuda/make"
        else:
            code_compiler = "nvcc"
            code_compiler_kwds = "-std=c++14 -Xcompiler -Wall -arch=sm_89 -O3"
            # -arch=sm_70 for aiops01 GPU (Argonne)  # -arch=sm_80 for A100s
            # -arch=sm_60 for P100
            # -arch=sm_89 for RTX 5000
        if yes_power_feedback:
            context_knowledge_selection = "cuda_optimization_summary"
            context_fewshot_examples = "cuda_c_few_shot_examples"
        else:
            context_knowledge_selection = "cuda"
    elif source_lang == "OMP":
        # For A100 on Linux_x86_64 or Linux_aarch64
        # code_compiler = "g++" # "/opt/nvidia/hpc_sdk/Linux_aarch64/2024/compilers/bin/nvc++"  #
        # code_compiler_kwds = "-fopenmp -O3" # "-Wall -O3 -Minfo -mp=gpu -gpu=cc80"  # -c
        # !! TODO - REVIEW: code_compiler_kwds = "-Wall -O3 -Minfo=accel -mp=gpu -gpu=cc80"  # "-fopenmp -O3 -c"
        # For RTX 5000 Ada on Linux x86_64:
        code_compiler = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.7/compilers/bin/nvc++"  # "g++"
        code_compiler_kwds = "-Wall -O3 -Minfo=accel -mp=gpu -gpu=cc89"  # "-fopenmp -O3"
        context_knowledge_selection = "omp"
    elif source_lang == "HIP":
        if build_script:
            code_build_script = "build_" + app_name
        elif make_script:
            if source_app == "miniMDock":
                code_compiler = "make"
                code_compiler_kwds = "API=HIP DEVICE=GPU CARD=AMD NUMWI=64 HIP_PATH=/opt/rocm-6.3.3"
            elif source_app == "XSBench":
                code_compiler = "hip/make"
            else:
                code_compiler_kwds = ""
        else:
            code_compiler = "hipcc"
            code_compiler_kwds = "-std=c++14 -Wall -O3 --amdgpu-target=gfx908"
        if yes_power_feedback:
            context_knowledge_selection = "hip_optimization_summary"
            context_fewshot_examples = "hip_c_few_shot_examples"
        else:
            context_knowledge_selection = "hip"  # TODO

    # TOGGLE EXECUTE PIPELINE
    yes_execute = True  # True / False
    if yes_execute:
        # Enter execution input parameters, if needed
        # A list of parameter string values, e.g., ["10000", "1"]
        # ****************************
        # ---> EXPERIMENT UPDATE FIELD - Now, automated:
        exe_input_parameters = source_apps[source_collection][source_app]["app_params"]
    else:
        exe_input_parameters = [""]


    """
    [3] SELECT THE LLM TO USE FOR THE CODE TRANSLATION EXPERIMENT
    """
    # SELECT A MODEL

    # ****************************
    # ---> EXPERIMENT UPDATE FIELD (now loaded from config.yaml)

    # -- FROM ARGO:
    # model = "gpt35"  # (max token = 4096)
    # model = "gpt35large"  # (max token = 16384)
    # model = "gpt4large"  # (max token = 32768)
    # model = "gpt4turbo"  # (max token = 128000)
    # model = "gpto4mini"  # (max token = 128000)
    # model_coder = "gpto4mini"  # (max token = 128000)
    # model = "gpt4olatest"  # (max token = 128000)
    # model_coder = "gpt4olatest"  # (max token = 128000)
    # model_coder = "gpt4o"  # (max token = 128000)
    # model_coder = "gpt4olatest"  # (max token = 128000)
    # model = "gpt4olatest"  # (max token = 200000)
    # model_coder = "gpt4olatest"  # (max token = 200000)
    # model = "gpt54"
    # model_coder = "gpt54"
    # model_judge = "gpt41"  # (max token = 200000)

    # -- FROM CEREBRAS
    # model = "gpt-oss-120b"
    # model_judge = "gpt-oss-120b"
    # model_coder = "qwen-3-coder-480b"

    # -- FROM OLLAMA
    # model = "devstral:24b" # (max token = 128000)
    # model_coder = "devstral:24b" # (max token = 128000)
    # model_judge = "devstral:24b" # (max token = 128000)
    # model = "qwq:latest"  # (max token = 128000)
    # model_coder = "qwen2.5-coder:32b"  # (max token = 32768)
    # model_judge = "qwq:latest"  # (max token = 128000)
    # model = "qwq:latest"  # (max token = 131072)
    # model = "gemma3:27b"  # crashing on code gen
    # model_coder = "gemma3:27b"
    # model = "granite3.2:8b-instruct-q8_0"
    # model_coder = "granite-code:34b-instruct-q8_0"
    # model = "deepseek-r1:70b"  # (max token = 128000)
    # model_coder = "deepseek-r1:70b"  # (max token = 128000)
    # model_coder = "deepseek-coder-v2:16b"  # (max token = 131072)
    # model_coder = "deepseek-coder-v2:16b"  # (max token = 131072)
    # model = "dolphin3"  # (max token = 131072)
    # model = "llama3.3:latest"  # (max token = 131072)
    # model = "phi4" # (max token = 16384)
    # model_coder = "codestral:22b"  # (max token = 32768)
    # model = "codestral:22b-v0.1-q8_0"  # (max token = 32768)
    # model = "wizardcoder:33b-v1.1-q8_0"  # (max token = 16384)
    # model_coder = "deepseek-coder:33b-instruct-q8_0"  # (max token = 16384)
    # model = "deepseek-coder-v2:16b-lite-instruct-fp16"  # (max token = 128000)
    # model = "deepseek-coder-v2:16b-lite-instruct-q8_0"
    # model = "phi3:14b"  # (max token = 131072)
    # model_judge = "llama3.3:70b"  # (max token = 128000)
    # model = "qwen3:latest"
    # model_coder = "qwen3:latest"
    # model_judge = "qwen3:latest"  # (max token = 40960)
    # model_coder = "qwen3-coder:30b"  # context 262144
    # model = "gpt-oss:20b"
    # model_judge = "gpt-oss:20b"

    model       = cfg["llm"]["model"]
    model_coder = cfg["llm"]["model_coder"]
    model_judge = cfg["llm"]["model_judge"]

    # Set maximum context token count size for selected LLM:
    num_ctx       = cfg["llm"]["num_ctx"]
    num_ctx_coder = cfg["llm"]["num_ctx_coder"]

    # Needed only for Argo endpoints
    uri = cfg["llm"]["argo_uri"]  # local  remote
    lassi.llm_config.argo_username = cfg["llm"]["argo_username"]  # Argo Gateway username

    # ASSIGN LLM API SOURCE:
    if model in ["gpt54", "gpt5", "gpt35", "gpt35large", "gpt4", "gpt4large", "gpt4turbo", "gpt4o", "gpt4olatest", "gpt41",
                 "gpto1preview", "gpto1mini", "gpto3mini", "gpto4mini", "claudesonnet4", "claudesonnet46"]:
        llm_source = "argo"
    elif model in ["gpt-oss-120b", "qwen-3-coder-480b"]:
        llm_source = "cerebras"
    elif ("GGUF" in model and "hf.co" not in model):
        llm_source = "lmstudio"
    else:
        llm_source = "ollama"

    """
    [4] SET ADDITIONAL PIPELINE PARAMETERS
    """
    _exp_label = cfg["experiment"].get("label", "").strip()
    if _exp_label:
        experiment = _exp_label
    else:
        experiment = "TEST_EXP_verify_runtime_" + model_coder + "_" + app_folder_name + "_" + detected_vendor.upper()  # Assign any label for experiment name.
    display_response = True
    test_with_error = False
    generate_test_case = False

    # Unload any previously loaded LLMs in the Ollama server
    # (clear out the memory)
    if llm_source == "ollama":
        lassi.llm_config.unload_ollama_model(model)
        lassi.llm_config.unload_ollama_model(model_coder)

    """
    [5] SET DATA COLLECTION CONFIGURATION
    """
    # Data spreadsheet column headers
    DATAFIELDNAMES = [
        "trial_number",
        "timestamp",
        "experiment",
        "gpu_vendor",
        "app_name",
        "code_line_count",
        "llm",
        "llm_coder",
        "llm_judge",
        "source_v0_energy_J",
        "source_v0_avg_power_W",
        "source_v0_runtime_s",
        "source_verify_runtime_s",  # ★ instrumented source runtime
        "base_v1_energy_J",
        "base_v1_avg_power_W",
        "base_v1_runtime_s",
        "base_v1_valid",
        "base_v1_energy_diff",
        "base_v1_avg_power_diff",
        "base_v1_runtime_diff",
        "zeroshot_runtime_s",        # ★ schema-aligned alias for base_v1_runtime_s
        "zeroshot_compile_status",   # ★ "success" | "compile_error" | "skipped"
        "zeroshot_execute_status",   # ★ "success" | "execute_error" | "skipped"
        # "base_v1_optimization_methods",
        # "base_v2_energy_J",
        # "base_v2_avg_power_W",
        # "base_v2_runtime_s",
        # "base_v2_valid",
        # "base_v2_selfcorrections",
        # "base_v2_energy_diff",
        # "base_v2_avg_power_diff",
        # "base_v2_runtime_diff",
        # "base_v2_optimization_methods",
        "lassiee_energy_J",
        "lassiee_avg_power_W",
        "lassiee_runtime_s",
        "lassiee_valid",
        "lassiee_version",
        "lassiee_llm_temp",
        "lassiee_selfcorrections",
        "lassiee_eeiterations",
        "lassiee_energy_diff",
        "lassiee_avg_power_diff",
        "lassiee_runtime_diff",
        "lassiee_optimization_methods",
        "total_selfcorrections",
        "compile_selfcorrections",   # ★
        "execute_selfcorrections",   # ★
        "tier1_selfcorrections",     # ★
        "tier2_selfcorrections",     # ★
        "tier3_selfcorrections",     # ★
        "global_counter_at_end",     # ★ global verify counter at trial end
        "total_code_versions",       # ★ distinct LASSI code block indices
        "total_code_attempts",       # ★ rows in attempt log for this trial
        "total_eeiterations",
        "pipeline_runtime_min",
        "tier2_pass_count",
        "tier2_fail_total",
        "tier3_pass_count",
        "tier3_fail_total",
        "tier3_active_relations",
        "tier3_inactive_relations",  # ★
        "judge_false_positives",
        "judge_false_negatives",
        "lassiee_clean_status",
        "lassiee_clean_runtime_s",
        "lassiee_clean_runtime_diff",
        "lassiee_clean_energy_J",
        "lassiee_clean_energy_diff",
        "zeroshot_tier1_verdict",    # ★ T1 invariant assertion verdict for zero-shot code (PASS/FAIL/SKIP)
        "zeroshot_tier2_verdict",    # ★ T2 empirical equivalence verdict for zero-shot code (PASS/FAIL/SKIP)
        "zeroshot_tier3_verdict",    # ★ T3 metamorphic testing verdict for zero-shot code (PASS/FAIL/SKIP)
        "zeroshot_empirical_verdict" # ★ composite empirical verdict (PASS/FAIL/SKIP)
    ]

    return (app_folder_name, app_name, file_name, lines_of_code, source_lang, target_lang, model, model_coder, model_judge, num_ctx, num_ctx_coder, llm_source, use_context, use_context_summaries_only,
            few_shot_learning, run_code_analysis, context_knowledge_selection, source_folder, code_compiler, code_compiler_kwds, build_script, make_script, code_build_script, yes_execute, exe_input_parameters,
            experiment, display_response, test_with_error, generate_test_case, power_measure, yes_power_feedback, baseline_oneshot_goal, use_self_prompting, vary_llm_temp,
            verifyprompt, DATAFIELDNAMES, detected_vendor.upper(),
            run_tier2, tier2_tolerance, run_tier3, tier3_tolerance,
            run_llm_judge)


# EXPERIMENTAL DATA COLLECTION MANAGEMENT
def init_data_collection(dir_path, DATAFIELDNAMES, filename="result_lassi-verify_pipeline_data.csv"):
    os.makedirs(dir_path, exist_ok=True)
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        with open(fp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DATAFIELDNAMES)
            writer.writeheader()
    return fp


# ── Run loop (Cell 9) ─────────────────────────────────────────────────────────

# Experimental repeat for pass@k metrics
trial_number = _cfg["experiment"]["trial_number"]
total_trials = _cfg["experiment"]["total_trials"]

# Load prompt dictionary (to capture recent manual changes to script, if any)
prompts = PromptDictionary()

all_trials_start = time.time()

# Store the current home working directory (needed for code execution)
original_home_directory = os.getcwd()


# COLLECT THE EXPERIMENT CONFIGURATIONS
(
    app_folder_name, app_name, file_name, lines_of_code, source_lang, target_lang, model, model_coder, model_judge, num_ctx, num_ctx_coder,
    llm_source, use_context, use_context_summaries_only, few_shot_learning, run_code_analysis, context_knowledge_selection, source_folder, code_compiler,
    code_compiler_kwds, build_script, make_script, code_build_script, yes_execute, exe_input_parameters, experiment,
    display_response, test_with_error, generate_test_case, power_measure, yes_power_feedback, baseline_oneshot_goal,
    use_self_prompting, vary_llm_temp, verifyprompt, DATAFIELDNAMES, gpu_vendor,
    run_tier2, tier2_tolerance, run_tier3, tier3_tolerance,
    run_llm_judge
) = experimental_setup(_cfg)

# IMPORT CONTEXT KNOWLEDGE FOR TARGET PROGRAMMING LANGUAGE
if gpu_vendor == "NVIDIA":
    input_context = "[BACKGROUND KNOWLEDGE: CUDA C++ PROGRAMMING BEST PRACTICES] " + prompts.get_codeknowledge("cuda") + "\n\n"  # [/END GENERAL CUDA PROGRAMMING CONTEXT]
    input_context = input_context + "[BACKGROUND KNOWLEDGE: CUDA C++ OPTIMIZATION GUIDE] " + prompts.get_codeknowledge(str(context_knowledge_selection)) + "\n\n"  #  [/END CUDA OPTIMIZATION CONTEXT]
elif gpu_vendor == "AMD":
    input_context = "[BACKGROUND KNOWLEDGE: HIP C++ PROGRAMMING BEST PRACTICES] " + prompts.get_codeknowledge("hip") + "\n\n"  # [/END GENERAL CUDA PROGRAMMING CONTEXT]
    input_context = input_context + "[BACKGROUND KNOWLEDGE: HIP C++ OPTIMIZATION GUIDE] " + prompts.get_codeknowledge(str(context_knowledge_selection)) + "\n\n"  #  [/END CUDA OPTIMIZATION CONTEXT]


# DEFINE PROMPT TEMPLATES
# TODO: MOST TASK PROMPT TEMPLATES WILL BE HANDLED BY THE PIPELINE METHODS
# If we are doing the runtime task, then set the system prompts, otherwise they are for ee
# we need to bring back in the translation systems prompt reference when we create LASSI-X
if (power_measure == False and yes_power_feedback == True):
    system_prompt_selection_index = "general_system_" + source_lang + "_runtime"
else:
    system_prompt_selection_index = source_lang + "_ee"  # "_to_" + target_lang  # Select which prompt to use from the prompt dictionary
this_llm_system_prompt = prompts.get_system_prompt(str(system_prompt_selection_index))

coderefactor_prompt_selection_index = source_lang  # + "_to_" + target_lang  # Select which prompt to use from the prompt dictionary
if (not baseline_oneshot_goal and use_self_prompting):
    # TODO: THIS SHOULD BE A DEAD BRANCH BECAUSE WE DON'T WANT TO USE baseline_oneshot_goal ANYMORE
    his_llm_refactoredcode_prompt = prompts.get_code_ee_refactoring_prompt(str(coderefactor_prompt_selection_index))
else:
    if (power_measure == False and yes_power_feedback == True):
        # RUNTIME TASK
        if use_context:
            this_llm_refactoredcode_prompt = prompts.get_code_ee_refactoring_prompt(str(coderefactor_prompt_selection_index + "_baseline_runtime"))
        else:
            this_llm_refactoredcode_prompt = prompts.get_code_ee_refactoring_prompt(str(coderefactor_prompt_selection_index + "_baseline_nocontext_runtime"))
    else:
        # EE TASK
        if use_context:
            this_llm_refactoredcode_prompt = prompts.get_code_ee_refactoring_prompt(str(coderefactor_prompt_selection_index + "_baseline"))
        else:
            this_llm_refactoredcode_prompt = prompts.get_code_ee_refactoring_prompt(str(coderefactor_prompt_selection_index + "_baseline_nocontext"))

# LOAD INPUT CODE TO REFACTOR
input_code_folder = source_folder + "/" + app_folder_name + "/"
if build_script:
    input_code_filename = app_name
elif make_script:
    input_code_filename = file_name
else:
    input_code_filename = app_folder_name + "-" + source_lang.lower() + "_main"
if source_lang == "OMP" or source_lang == "SYCL":
    input_code_extension = "cpp"
elif source_lang == "CUDA" or source_lang == "HIP":
    if (make_script and source_lang == "HIP"):
        input_code_extension = "cpp"  # custom for miniMDock and XSBench
    else:
        input_code_extension = "cu"
if make_script:
    source_file_directory = "generated_codes/" + input_code_folder
else:
    source_file_directory = "source_codes/" + input_code_folder
if make_script:
    # SELECT FILE TO REFACTOR from miniapp
    if source_lang == "CUDA":
        if app_name == "miniMDock":
            input_code_file_subdirectory = "device/cuda/"
        elif app_name == "XSBench":
            input_code_file_subdirectory = ""
    elif source_lang == "HIP":
        if app_name == "miniMDock":
            input_code_file_subdirectory = "device/hip/"
        elif app_name == "XSBench":
            input_code_file_subdirectory = ""
    source_file_path = source_file_directory + input_code_file_subdirectory + input_code_filename + "." + input_code_extension
    source_file_path = os.path.abspath(source_file_path)
else:
    input_code_file_subdirectory = ""
    source_file_path = source_file_directory + input_code_filename + "." + input_code_extension
    source_file_path = os.path.abspath(source_file_path)
with open(source_file_path, 'r') as file:
    input_code = file.read()
file.close()


# SPECIFY OUTPUT CODE SAVE
if build_script:
    output_code_directory = "generated_codes/" + source_folder + "/" + app_folder_name
    output_code_file_name = app_name
elif make_script:
    output_code_file_name = file_name
    output_code_directory = "generated_codes/" + source_folder + "/" + app_folder_name
else:
    output_code_directory = "generated_codes/" + llm_source + "_" + model.replace('/', '--') + "/promptset_" + str(coderefactor_prompt_selection_index)
    output_code_file_name = input_code_filename + "_ee_refactorcode_" + source_lang + "_exp" + experiment

now = datetime.now(ZoneInfo("America/Chicago"))
dt_string = now.strftime("%m-%d-%Y_t%H_%M_%S")

# SPECIFIY DATA COLLECTION SAVE
experiment_data_directory = "generated_codes/experiment/" + experiment + "__" + dt_string
# Convert to absolute directory::
experiment_data_directory = os.path.abspath(experiment_data_directory)
experiment_data_file_object = init_data_collection(experiment_data_directory, DATAFIELDNAMES)
experiment_data = dict.fromkeys(DATAFIELDNAMES, None)  # all keys start out as None
# Table 2 ★ per-attempt log — one row per code generation attempt (including self-correction retries)
attempt_log_file_path = init_data_collection(
    experiment_data_directory, lassi.pipeline_v4.ATTEMPT_LOG_FIELDNAMES,
    filename="result_lassi-verify_attempt_log.csv"
)
experiment_data["gpu_vendor"] = str(detected_vendor.upper())

while trial_number <= total_trials:
    # For the miniApp, we need to reset the source kernel file
    #  because the pipeline for this scenario must overwrite the
    #  source code in the generated_codes folder.
    if make_script:
        if source_lang == "CUDA":
            if app_name == "miniMDock":
                working_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/miniMDock/device/cuda/kernels.cu"
                clean_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/miniMDock/device/cuda/kernels_orig_merged.cu"
            elif app_name == "XSBench":
                working_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/XSBench/cuda/Main.cu"
                clean_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/XSBench/cuda/Main_orig_merged.cu"
        elif source_lang == "HIP":
            if app_name == "miniMDock":
                working_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/miniMDock/device/hip/kernels.cpp"
                clean_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/miniMDock/device/hip/kernels_orig_merged.cpp"
            elif app_name == "XSBench":
                working_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/XSBench/hip/Main.cpp"
                clean_kernel_file = "/home/cc/lassi/LASSI-EE/generated_codes/Miniapps/XSBench/hip/Main_orig_merged.cpp"
        try:
            print(f"[reset] Copying clean kernel -> working file\n"
                  f"        src: {clean_kernel_file}\n"
                  f"        dst: {working_kernel_file}")
            shutil.copyfile(clean_kernel_file, working_kernel_file)
            print("[reset] Success: working file reset to clean copy.")
        except FileNotFoundError as e:
            print(f"[reset] ERROR: file not found: {e.filename}")
        except PermissionError as e:
            print(f"[reset] ERROR: permission denied: {e.filename}")
        except OSError as e:
            # catches other IO errors (disk full, cross-device issues, etc.)
            print(f"[reset] ERROR: {e.__class__.__name__}: {e}")

    # CREATE METADATA RECORD CONTENT
    meta_data_file = "***** EXPERIMENT CONFIGURATION SUMMARY *****\n"
    meta_data_file += "\nGPU Vendor: " + str(detected_vendor.upper())
    meta_data_file += "\nDate-time: " + dt_string
    meta_data_file += "\nApplication: " + app_folder_name
    meta_data_file += "\n\nRefactoring task: " + system_prompt_selection_index
    meta_data_file += "\n\nLLM Source: " + llm_source
    meta_data_file += "\nModel: " + model
    meta_data_file += "\nModel coder: " + model_coder
    meta_data_file += "\nModel judge: " + model_judge
    meta_data_file += "\nSystem Prompt Index: " + str(coderefactor_prompt_selection_index)
    meta_data_file += "\nCode file template: " + output_code_directory + "/" + output_code_file_name
    meta_data_file += "\n\nLLM System Prompt: " + "\n\n" + this_llm_system_prompt
    # meta_data_file += "\n\nLLM Prompt for Code Generation: " + "\n\n" + this_llm_refactoredcode_prompt
    meta_data_file += "\n\nPower measurement: " + str(power_measure)

    # Display metadata
    print("\n" + meta_data_file + "\n")

    print("\n\n" + color.BLUE + color.BOLD + "***** BEGIN THE LASSI ENERGY EFFICIENCY PIPELINE (v4) *****" + color.END + " Trial #: " + str(trial_number) + "\n\n")

    # RUN LASSI-EE PIPELINE
    complete = lassi.pipeline_v4.auto_code_llmgeneration_pipeline(
        prompts,
        this_llm_system_prompt,
        this_llm_refactoredcode_prompt,
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
        run_code_analysis,
        input_context,
        input_code_file_subdirectory,
        input_code,
        lines_of_code,
        # target_code,
        target_lang,
        input_code_extension,
        source_lang,
        code_compiler,
        code_compiler_kwds,
        build_script,
        make_script,
        code_build_script,
        output_code_directory,
        output_code_file_name,
        source_file_path,
        display_response,
        test_with_error,
        generate_test_case,
        yes_execute,
        exe_input_parameters,
        meta_data_file,
        power_measure,
        yes_power_feedback,
        dt_string,
        baseline_oneshot_goal,
        use_self_prompting,
        vary_llm_temp,
        experiment,
        verifyprompt,
        trial_number,
        experiment_data_directory,
        experiment_data_file_object,
        experiment_data,
        app_name,
        run_tier2=run_tier2,
        tier2_tolerance=tier2_tolerance,
        run_tier3=run_tier3,
        tier3_tolerance=tier3_tolerance,
        run_llm_judge=run_llm_judge,
        attempt_log_file_path=attempt_log_file_path
    )

    # Reset to original working directory
    os.chdir(original_home_directory)

    if complete:
        print("\n\n" + color.GREEN + color.BOLD + "***** LASSI PIPELINE COMPLETED *****\n\n" + color.END)
    else:
        print("\n\n" + color.RED + color.BOLD + "***** LASSI PIPELINE FAILED TO COMPLETE *****\n\n" + color.END)

    # Increment experiment counter
    trial_number = trial_number + 1


all_trials_end = time.time() - all_trials_start

print("\n\n" + color.PURPLE + color.BOLD + "***** ALL TRIALS COMPLETED ***** Total time (min): " + str(round(all_trials_end / 60, 2)) + "\n\n" + color.END)
