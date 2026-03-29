# LASSI-Verify

<img src="lassi/lassi-verify-logo.png" alt="LASSI-Verify Logo" width="300"/>

## What is LASSI-Verify?

LASSI-Verify is an adaptive empirical verification framework for parallel scientific GPU code transformations. Its goal is to determine whether an LLM-generated refactoring of a CUDA kernel (optimized for energy efficiency or runtime) is functionally equivalent to the original source and genuinely better-performing. The framework is adaptive in that verification properties are drawn from structured catalogs and instantiated relative to each specific transformation, while the three-tier architecture and mandatory coverage requirements remain fixed.
It wraps around a multi-iteration LLM self-improvement loop ("LASSI") with three formal-methods-grounded verification tiers that together provide correctness evidence without requiring formal proofs or exhaustive testing.

## Getting Started

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run lassiapp.py
```
