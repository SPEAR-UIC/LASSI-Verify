# AGENTS.md

Background and orientation for AI coding assistants working in this repository. This
file documents how the project is put together today; it intentionally does not cover
roadmap items or work in progress.

## Related projects

LASSI-Verify is part of a small family of research tools exploring LLM-based automated
refactoring of parallel scientific GPU/HPC code, from the same research group
(SPEAR-UIC, in collaboration with Argonne National Laboratory). Understanding where this
repository sits in that family helps when a user asks "how does this relate to LASSI" or
wants help citing the work:

- **LASSI** — the original self-correcting pipeline for translating parallel scientific
  code between languages (e.g., CUDA to OpenMP) via an iterative LLM self-correction
  loop. LASSI-Verify's self-correction loop and prompt-driven refactoring approach build
  directly on this design.
  > Dearing, M. T., Tao, Y., Wu, X., Lan, Z., Taylor, V. "LASSI: An LLM-based Automated
  > Self-Correcting Pipeline for Translating Parallel Scientific Codes," International
  > Workshop on Large Language Models and HPC (LLMxHPC), IEEE International Conference
  > on Cluster Computing (CLUSTER) (2024). DOI: 10.1109/CLUSTERWorkshops61563.2024.00029
- **LASSI-EE** — re-targets the same self-correction loop at energy-efficiency
  refactoring, using runtime GPU power profiling and an LLM-as-judge screening step.
  > Dearing, M. T., Tao, Y., Wu, X., Lan, Z., Taylor, V. "LASSI-EE: LLM-based Automated
  > Refactoring of Parallel Scientific Codes for Energy Efficiency," Proceedings of the 3rd
  > International Workshop on Energy Efficiency with Sustainable Performance (EESP '26),
  > held with SC26: The International Conference for High Performance Computing,
  > Networking, Storage and Analysis (Nov. 2026).
- **LASSI-Verify** (this repository) — replaces human review / LLM-judge screening with
  a three-tier empirical verification framework (invariant assertions, observational
  equivalence, metamorphic testing) plus a witness chain for cheat-resistant
  loop-deletion detection, so accepting a refactored kernel doesn't depend on a human or
  a single LLM's judgment call.
  > M. T. Dearing, X. Wu, V. Taylor, and Z. Lan, "LASSI-Verify: Adaptive Empirical
  > Verification of LLM-Generated Parallel Scientific Code Transformations," SC26
  > Workshops (Correctness 2026), Chicago, IL, USA, Nov. 2026.

See [`CITATION.cff`](CITATION.cff) for the machine-readable citation of this repository.

## Project overview

LASSI-Verify is an adaptive empirical verification framework for LLM-generated
CUDA/HIP GPU kernel refactorings. It orchestrates an LLM self-improvement loop that
generates optimized code, instruments it with three verification tiers, compiles/runs
it on GPU hardware, and iterates using diagnostics from failed tiers as feedback for the
next LLM attempt.

## Repository layout

```
run_lassi.py           CLI entry point (reads config.yaml, runs the pipeline)
lassiapp.py            Streamlit GUI — writes config.yaml, then spawns run_lassi.py
config.yaml.example    Template experiment config; copy to config.yaml before running
source_apps.json       Benchmark registry (app names, execution params, lines of code)
CITATION.cff           Machine-readable citation for this repository

lassi/                  Core package
  pipeline_v4.py          Pipeline orchestration (the only pipeline version in use)
  verify.py               Tier 1/2/3 instrumentation + witness-chain harness
  llm_config.py           LLM backend configs (ollama, openai_compatible) + dispatch
  llm_methods.py          LLM call wrappers for each pipeline role (coder, judge, etc.)
  code_manager.py         Compile/execute via subprocess (nvcc for CUDA, hipcc for HIP)
  pipeline_methods.py     Stage 0 source testing, code analysis, refactor-plan generation
  prompt_dictionary.py    All LLM prompt templates
  gpu_detector.py         NVIDIA/AMD GPU vendor detection
  get_power.py            NVIDIA power monitoring (via the nvidia-ml-py package)
  amd_power.py            AMD power monitoring (rocm-smi)
  monitor.py              Live web dashboard — streams pipeline stdout to localhost:5007
  context/                LLM knowledge docs and verification catalogs (see below)

lassi_verify_analysis/  Post-experiment analysis package (no imports from lassi/)
  run_all.py              CLI: python -m lassi_verify_analysis.run_all --data-dir DIR
  pass_at_k.py            Verified pass@k metrics
  verification_analysis.py   Verification-effectiveness metrics
  verification_evidence.py   Per-trial evidence summaries
  (additional modules cover ablation analysis, oracle-anchored confusion-matrix
  scoring, and defect-injection studies behind specific results in the paper)

scripts/
  test_tier1_quick.py     Standalone smoke test for Tier 1 (assertion insertion)

source_codes/HeCBench/   Vendored CUDA kernels used in the reported experiment
                         (see source_apps.json for each kernel's build parameters)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml
```

Requires a Linux system with a CUDA-capable NVIDIA GPU and `nvcc` on `PATH` (the
reported experiment used an NVIDIA RTX 5000 Ada, `sm_89`). AMD GPU support (ROCm/`hipcc`)
is implemented in the codebase (`lassi/amd_power.py`, vendor detection in
`lassi/gpu_detector.py`) but was not exercised by this paper's CUDA-only kernel corpus.

**Warning:** `requirements.txt` depends on `nvidia-ml-py` for NVIDIA power monitoring —
do not also install the `pynvml` package; the two conflict on the same `pynvml` Python
namespace.

## Running

```bash
# CLI (reads config.yaml)
python run_lassi.py
python run_lassi.py --config my_exp.yaml   # custom config file
python run_lassi.py --no-monitor           # skip the live web dashboard (port 5007)

# GUI (writes config.yaml, then launches run_lassi.py for you)
streamlit run lassiapp.py
```

`config.yaml` (gitignored; copy it from `config.yaml.example`) controls everything about
an experiment: app/kernel selection, LLM backend and model, verification tier settings,
tolerances, and trial count. Two LLM backends are supported (`llm.backend` in
config.yaml): `ollama` for a local Ollama server, and `openai_compatible` for OpenAI
itself or any OpenAI-compatible endpoint (vLLM, llama.cpp server, an institutional
gateway, etc.).

## Running analysis

```bash
python -m lassi_verify_analysis.run_all \
  --data-dir path/to/experiment/csvs \
  --output-dir path/to/results \
  --verbose
```

Expects two CSVs in `--data-dir`: `lassi-verify_pipeline_data.csv` (required) and
`lassi-verify_attempt_log.csv` (optional).

## Tests

No test framework (no pytest). `scripts/test_tier1_quick.py` is a standalone script
(not a unit test) that exercises Tier 1 assertion insertion against the real pipeline —
it requires live GPU hardware and a configured LLM backend.

## Key architecture details

- **Pipeline:** `lassi/pipeline_v4.py`, imported by both `run_lassi.py` and
  `lassiapp.py`. For a deep architecture walkthrough (data structures, pipeline flow,
  self-correction budgets, per-tier mechanics), see
  `lassi/context/LASSI_Verify_Pipeline_Overview.md`.
- **Verification tiers** (implemented in `lassi/verify.py`):
  - Tier 1 — Invariant assertions (`code_insert_asserts`): LLM-proposed, catalog-grounded
    runtime checks over program state, inserted into both source and variant.
  - Tier 2 — Observational equivalence (`code_insert_passthrough` +
    `compare_passthrough_outputs`): deterministic comparison of output state between
    source and variant under matched execution; no LLM involved in the comparison.
  - Tier 3 — Metamorphic testing: generates variant `.cu` files (`*_mt_*.cu`) and checks
    relational properties that must hold across systematically perturbed executions.
  - All three tiers are mandatory and run together — there is no partial-tier mode.
  - **Witness chain:** a seeded, cheat-resistant instrumentation mechanism
    (`verification.run_witness_chain` in config.yaml, defaults to `true`) that supplies
    launch-integrity evidence for benchmark-structured (timed-loop) code, guarding
    against an LLM deleting timed work to fabricate a speedup.
  - **LLM-as-judge:** also recorded for every attempt as a comparison point, but it
    never contributes to the framework's own accept/reject decision — the three tiers
    alone determine validity.
- **Verification catalogs** (`lassi/context/`): `Invariant_Assertions_Catalog.md`,
  `Passthrough_Catalog.md`, and `Metamorphic_Relations_Catalog.md` define the properties
  each tier draws from; `Witness_Chain_Harness.md` documents the witness-chain design.
  When modifying tier behavior, prefer changing the relevant catalog or
  `lassi/prompt_dictionary.py` (all LLM prompt templates are centralized there) over
  inline prompt edits elsewhere.
- **Compilation:** code is compiled at runtime via subprocess calls in
  `lassi/code_manager.py` (`nvcc` for CUDA, `hipcc` for HIP). Compiler flags are set in
  `config.yaml`/code, not a build system. Default NVIDIA flags:
  `nvcc -std=c++14 -Xcompiler -Wall -arch=sm_89 -O3`.
- **Generated file naming** (all gitignored): `*_verify.cu`, `*_mt_*.cu`,
  `*_zs_t1_Trial*.cu`, `*_zs_t2_Trial*.cu`, `*_best_clean.cu`, and compiled binaries like
  `*-cuda_main`.

## Conventions

- No linter, formatter, type checker, or CI is configured.
- Python 3.12+ (uses the `zoneinfo` stdlib module).
- CSV result files and compiled binaries are gitignored — do not commit them.
- `source_codes/` contains vendored benchmark source; treat it as read-only input data.
