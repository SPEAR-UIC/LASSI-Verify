"""
lassi_verify_analysis — SC26 paper metrics for LASSI-Verify experimental data.

Reads two CSV files produced by the LASSI-Verify pipeline:
  - lassi-verify_pipeline_data.csv  (Table 1: one row per kernel × trial)
  - lassi-verify_attempt_log.csv    (Table 2: one row per code attempt)

No imports from the lassi/ pipeline module — analysis only reads CSV outputs.
"""

__version__ = "0.1.0"
