# LASSI-Verify

<img src="lassi/lassi-verify-logo.png" alt="LASSI-Verify Logo" width="300"/>

> **Status:** This repository is being prepared for public release and is pending institutional open-source clearance. (Remove this line once cleared.)

## What is LASSI-Verify?

LASSI-Verify is an adaptive empirical verification framework for parallel scientific GPU code transformations. Its goal is to determine whether an LLM-generated refactoring of a CUDA kernel (optimized for energy efficiency or runtime) is functionally equivalent to the original source and genuinely better-performing. The framework is adaptive in that verification properties are drawn from structured catalogs and instantiated relative to each specific transformation, while the three-tier architecture and mandatory coverage requirements remain fixed.
It wraps around a multi-iteration LLM self-improvement loop ("LASSI") with three formal-methods-grounded verification tiers that together provide correctness evidence without requiring formal proofs or exhaustive testing.

## How It Works

LASSI-Verify treats the source kernel's observable behavior as its specification and grounds acceptance of each LLM-generated variant in three tiers, all mandatory and none individually sufficient:

1. **Tier 1 — Invariant assertions.** LLM-proposed, catalog-grounded runtime assertions over program state, inserted into both source and variant.
2. **Tier 2 — Observational equivalence.** Empirical comparison of output state between source and variant under matched execution.
3. **Tier 3 — Metamorphic relations.** Property-based checks that must hold across systematically perturbed executions of the variant.

A **witness chain** — a seeded, cheat-resistant instrumentation mechanism — supplies launch-integrity evidence for benchmark-structured (timed-loop) code, added after observing LLMs delete timed work to fabricate speedups. Every tier failure doubles as a repair signal fed back into the next LLM self-correction attempt. An LLM-as-judge verdict is also recorded for comparison, but never contributes to the framework's own accept/reject decision.

See the paper for the full design and preliminary results (citation below).

## Repository Layout

- `run_lassi.py` — CLI entry point for running the pipeline (config via `config.yaml`)
- `lassiapp.py` — Streamlit GUI launcher
- `lassi/` — core package: `pipeline_v4.py` (orchestration), `verify.py` (Tier 1/2/3 instrumentation + witness chain), `prompt_dictionary.py` (LLM prompts), `context/` (property catalogs)
- `source_codes/HeCBench/` — the vendored CUDA kernels used in the reported experiment (see `source_apps.json` for per-kernel build parameters)
- `lassi_verify_analysis/` — post-hoc analysis package that computes the paper's tables/figures from raw per-trial logs
- `scripts/` — standalone utility/validation scripts

## Getting Started

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

A Linux system with a CUDA-capable NVIDIA GPU and `nvcc` on `PATH` is required. The pipeline is not pinned to a specific architecture beyond the `-arch` flag passed to `nvcc` (set in `config.yaml`); the reported experiment used an NVIDIA RTX 5000 Ada (`sm_89`).

### Run via the CLI

```bash
python run_lassi.py
```

Experiment configuration (kernel selection, LLM backend/model, verification tier toggles, trial count, etc.) is read from `config.yaml` on every run.

### Run via the GUI

```bash
streamlit run lassiapp.py
```

Opens a browser-based form at `http://localhost:8501` for configuring and launching a run without editing `config.yaml` directly.

## Citation

If you use LASSI-Verify in your research, please cite:

> M. T. Dearing, X. Wu, V. Taylor, and Z. Lan, "LASSI-Verify: Adaptive Empirical Verification of LLM-Generated Parallel Scientific Code Transformations," in *SC26 Workshops (Correctness 2026)*, Chicago, IL, USA, Nov. 2026.

See [`CITATION.cff`](CITATION.cff) for a machine-readable version.

## License

TBD, pending institutional review.
