# INVALID — do not use

Everything in this directory was produced on 2026-08-10 while the CUDA runtime was broken
(`cudaGetDeviceCount`/`cudaMalloc` returned "unknown error" despite `nvidia-smi` reporting
the GPU as healthy — see memory `reference_nvidia_driver_recovery.md`, "New symptom variant
(2026-07-27)"). Every kernel binary ran without crashing but computed nothing: buffers came
back all-zero or as uninitialized-memory garbage. The two "tolerant" classifications
(jaccard `final_weight_j`, threadfence `final_reduction_result`) are uninitialized-memory
artifacts, not genuine floating-point instability. The "exact" classifications are trivially
"stable" zero-equals-zero comparisons, not evidence of real buffer stability. The timing
numbers in timing_spotcheck.csv time a no-op, not real kernel execution.

Kept only as a record of the investigation. Re-run
`lassi_verify_analysis/buffer_stability_classification.py` fresh once the CUDA runtime is
confirmed working again (`nvcc`-compiled `cudaMalloc` test succeeds).
