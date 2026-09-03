"""
Canonical-base registry for Step 4 defect injection.

One framework-VALID (inter)oracle-PASS winning code variant per kernel, identified
by joining each experiment dir's `result_lassi-verify_pipeline_data.csv` (lassiee_valid,
lassiee_version) against `oracle_results.csv` (oracle_verdict) on (trial_number,
code_version) -- confirmed on disk 2026-08-23, see planning/LASSI_Verify_Phase2_Bridge.md's
dated entries for the design context. All file existence checked directly (`main_exists`
was true for all 17 at registry-build time -- re-verify if this module starts throwing
FileNotFoundError, since archived experiment dirs are not expected to move).

matrix-rotate has no buffer_classification.json (postdates the offline classification
pass) -- callers must default its buffers to 'exact' comparison mode.
"""
from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KERNEL_BASES: dict[str, dict] = {
    'all-pairs-distance': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_ALLPAIRS_1__07-20-2026_t00_11_40',
        'trial': 1,
        'code_version': 4,
        'main_file': 'all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_07-20-2026_t00_11_40_v4_Trial1.cu',
        't3_companions': ['all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_mt_attribute_order_invariance_v4.cu', 'all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_mt_attribute_order_reversal_invariance_v4.cu', 'all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_mt_categorical_value_relabeling_invariance_v4.cu', 'all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_mt_instance_row_swap_permutation_v4.cu', 'all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_mt_instance_swap_permutative_v4.cu', 'all-pairs-distance-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ALLPAIRS_1_mt_value_relabeling_invariance_v4.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/all-pairs-distance_buffer_classification.json',
    },
    'atomicCost': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_ATOMIC_1__07-20-2026_t08_52_15',
        'trial': 1,
        'code_version': 4,
        'main_file': 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_07-20-2026_t08_52_15_v4_Trial1.cu',
        't3_companions': ['atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_single_input_size_perturbation_sensitivity_atomic__p_outlier_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_single_input_size_perturbation_sensitivity_nonatomic__p_outlier_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_single_input_size_perturbation_sensitivity_nonatomic__p_small_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_input_sensitivity_atomic__p_outlier_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_input_sensitivity_atomic__p_small_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_input_sensitivity_non_atomic__p_outlier_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_input_sensitivity_non_atomic__p_small_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_sensitivity_atomic_output__p_outlier_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_sensitivity_atomic_output__p_small_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_sensitivity_non_atomic_output__p_outlier_v4.cu', 'atomicCost-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ATOMIC_1_mt_size_sensitivity_non_atomic_output__p_small_v4.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/atomicCost_buffer_classification.json',
    },
    'bsearch': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_BSEARCH_1__07-21-2026_t07_47_59',
        'trial': 1,
        'code_version': 3,
        'main_file': 'bsearch-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_BSEARCH_1_07-21-2026_t07_47_59_v3_Trial1.cu',
        't3_companions': ['bsearch-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_BSEARCH_1_mt_query_input_sensitivity__p_outlier_v3.cu', 'bsearch-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_BSEARCH_1_mt_query_input_sensitivity__p_small_v3.cu', 'bsearch-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_BSEARCH_1_mt_query_permutation_sensitivity__p_outlier_v3.cu', 'bsearch-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_BSEARCH_1_mt_query_permutation_sensitivity__p_small_v3.cu', 'bsearch-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_BSEARCH_1_mt_query_shift_invariance_v3.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/bsearch_buffer_classification.json',
    },
    'chacha20': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_CHACHA20_1__07-22-2026_t17_22_45',
        'trial': 1,
        'code_version': 10,
        'main_file': 'chacha20-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_CHACHA20_1_07-22-2026_t17_22_45_v10_Trial1.cu',
        't3_companions': ['chacha20-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_CHACHA20_1_mt_key_bitflip_sensitivity__p_outlier_v10.cu', 'chacha20-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_CHACHA20_1_mt_key_bitflip_sensitivity__p_small_v10.cu', 'chacha20-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_CHACHA20_1_mt_nonce_bitflip_sensitivity__p_outlier_v10.cu', 'chacha20-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_CHACHA20_1_mt_nonce_bitflip_sensitivity__p_small_v10.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/chacha20_buffer_classification.json',
    },
    'colorwheel': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_COLORW_1__07-22-2026_t13_50_27',
        'trial': 1,
        'code_version': 9,
        'main_file': 'colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_07-22-2026_t13_50_27_v9_Trial1.cu',
        't3_companions': ['colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_mt_output_sensitivity_to_range__p_outlier_v9.cu', 'colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_mt_repeat_invariance_v9.cu', 'colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_mt_size_sensitivity__p_outlier_v9.cu', 'colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_mt_size_sensitivity__p_small_v9.cu', 'colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_mt_truerange_sensitivity__p_outlier_v9.cu', 'colorwheel-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_COLORW_1_mt_truerange_sensitivity__p_small_v9.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/colorwheel_buffer_classification.json',
    },
    'entropy': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_ENTROPY_2__07-19-2026_t15_29_00',
        'trial': 1,
        'code_version': 8,
        'main_file': 'entropy-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ENTROPY_2_07-19-2026_t15_29_00_v8_Trial1.cu',
        't3_companions': ['entropy-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ENTROPY_2_mt_attribute_negation_invariance_v8.cu', 'entropy-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ENTROPY_2_mt_input_symbol_sensitivity__p_outlier_v8.cu', 'entropy-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ENTROPY_2_mt_input_symbol_sensitivity__p_small_v8.cu', 'entropy-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ENTROPY_2_mt_uniform_value_shift_invariance_v8.cu', 'entropy-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_ENTROPY_2_mt_value_relabeling_invariance_v8.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/entropy_buffer_classification.json',
    },
    'extrema': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_EXTREMA_1__07-19-2026_t21_47_11',
        'trial': 2,
        'code_version': 3,
        'main_file': 'extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_07-19-2026_t21_47_11_v3_Trial2.cu',
        't3_companions': ['extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_mt_input_negation_sensitivity_1d__p_outlier_v3.cu', 'extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_mt_input_negation_sensitivity_1d__p_small_v3.cu', 'extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_mt_input_negation_sensitivity_2d__p_outlier_v3.cu', 'extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_mt_input_negation_sensitivity_2d__p_small_v3.cu', 'extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_mt_input_value_shift_invariance_1d_v3.cu', 'extrema-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_EXTREMA_1_mt_input_value_shift_invariance_2d_v3.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/extrema_buffer_classification.json',
    },
    'floydwarshall': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_FLOYDW_1__07-23-2026_t00_31_21',
        'trial': 1,
        'code_version': 4,
        'main_file': 'floydwarshall-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_FLOYDW_1_07-23-2026_t00_31_21_v4_Trial1.cu',
        't3_companions': ['floydwarshall-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_FLOYDW_1_mt_distance_matrix_reversal_invariance_v4.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/floydwarshall_buffer_classification.json',
    },
    'jaccard': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_JACCARD_1__07-24-2026_t10_49_40',
        'trial': 1,
        'code_version': 6,
        'main_file': 'jaccard-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_JACCARD_1_07-24-2026_t10_49_40_v6_Trial1.cu',
        't3_companions': ['jaccard-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_JACCARD_1_mt_csr_value_negation_v6.cu', 'jaccard-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_JACCARD_1_mt_csr_value_scaling_v6.cu', 'jaccard-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_JACCARD_1_mt_csr_value_sensitivity__p_outlier_v6.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/jaccard_buffer_classification.json',
    },
    'layout': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_LAYOUT_1__07-20-2026_t18_40_16',
        'trial': 1,
        'code_version': 3,
        'main_file': 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_07-20-2026_t18_40_16_v3_Trial1.cu',
        't3_companions': ['layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_aos_input_sensitivity__p_outlier_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_aos_input_sensitivity__p_small_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_input_negation_aos_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_input_negation_soa_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_soa_input_sensitivity__p_outlier_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_soa_input_sensitivity__p_small_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_tree_permutation_aos_v3.cu', 'layout-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LAYOUT_1_mt_tree_permutation_soa_v3.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/layout_buffer_classification.json',
    },
    'lid-driven-cavity': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_LIDDRIVENC_1__07-24-2026_t16_03_41',
        'trial': 1,
        'code_version': 10,
        'main_file': 'lid-driven-cavity-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LIDDRIVENC_1_07-24-2026_t16_03_41_v10_Trial1.cu',
        't3_companions': ['lid-driven-cavity-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LIDDRIVENC_1_mt_initial_lid_velocity_sensitivity__p_outlier_v10.cu', 'lid-driven-cavity-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LIDDRIVENC_1_mt_initial_lid_velocity_sensitivity__p_small_v10.cu', 'lid-driven-cavity-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_LIDDRIVENC_1_mt_lid_velocity_sensitivity__p_outlier_v10.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/lid-driven-cavity_buffer_classification.json',
    },
    'matrix-rotate': {
        'exp_dir': 'generated_codes/experiment/KERNEL_WITNESS-TEST_matrixrotate_2__08-20-2026_t23_30_46',
        'trial': 1,
        'code_version': 3,
        'main_file': 'matrix-rotate-cuda_main_ee_refactorcode_CUDA_expKERNEL_WITNESS-TEST_matrixrotate_2_08-20-2026_t23_30_46_v3_Trial1.cu',
        't3_companions': ['matrix-rotate-cuda_main_ee_refactorcode_CUDA_expKERNEL_WITNESS-TEST_matrixrotate_2_mt_global_value_scaling_v3.cu', 'matrix-rotate-cuda_main_ee_refactorcode_CUDA_expKERNEL_WITNESS-TEST_matrixrotate_2_mt_rotation_cycle_repeat_plus_four_v3.cu', 'matrix-rotate-cuda_main_ee_refactorcode_CUDA_expKERNEL_WITNESS-TEST_matrixrotate_2_mt_uniform_value_shift_v3.cu'],
        'buffer_classification': None,
    },
    'murmurhash3': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_MURMURH_1__07-23-2026_t19_45_45',
        'trial': 1,
        'code_version': 6,
        'main_file': 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_07-23-2026_t19_45_45_v6_Trial1.cu',
        't3_companions': ['murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_key_byte_perturbation_sensitivity__p_outlier_v6.cu', 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_key_byte_perturbation_sensitivity__p_small_v6.cu', 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_key_byte_sensitivity__p_outlier_v6.cu', 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_key_byte_sensitivity__p_small_v6.cu', 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_seed_index_sensitivity__p_small_v6.cu', 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_seed_offset_sensitivity__p_outlier_v6.cu', 'murmurhash3-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_MURMURH_1_mt_seed_offset_sensitivity__p_small_v6.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/murmurhash3_buffer_classification.json',
    },
    'pathfinder': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_PATHFINDER_1__07-26-2026_t05_34_18',
        'trial': 1,
        'code_version': 6,
        'main_file': 'pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_07-26-2026_t05_34_18_v6_Trial1.cu',
        't3_companions': ['pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_mt_final_result_sensitivity__p_outlier_v6.cu', 'pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_mt_final_result_sensitivity__p_small_v6.cu', 'pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_mt_input_grid_sensitivity__p_outlier_v6.cu', 'pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_mt_input_grid_sensitivity__p_small_v6.cu', 'pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_mt_row_weight_sensitivity__p_outlier_v6.cu', 'pathfinder-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_PATHFINDER_1_mt_row_weight_sensitivity__p_small_v6.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/pathfinder_buffer_classification.json',
    },
    'randomaccess': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_RANDA_1__07-21-2026_t21_03_54',
        'trial': 1,
        'code_version': 11,
        'main_file': 'randomaccess-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_RANDA_1_07-21-2026_t21_03_54_v11_Trial1.cu',
        't3_companions': ['randomaccess-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_RANDA_1_mt_table_entry_sensitivity__p_outlier_v11.cu', 'randomaccess-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_RANDA_1_mt_uniform_xor_mask_invariance_v11.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/randomaccess_buffer_classification.json',
    },
    'segment-reduce': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_SEGMENTR_1__07-25-2026_t19_56_25',
        'trial': 1,
        'code_version': 6,
        'main_file': 'segment-reduce-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_SEGMENTR_1_07-25-2026_t19_56_25_v6_Trial1.cu',
        't3_companions': ['segment-reduce-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_SEGMENTR_1_mt_input_value_scaling_v6.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/segment-reduce_buffer_classification.json',
    },
    'threadfence': {
        'exp_dir': 'generated_codes/experiment/SC Correctness 26 Workshop/KERNEL_EXP-SC26_THREADF_1__07-20-2026_t14_07_07',
        'trial': 1,
        'code_version': 5,
        'main_file': 'threadfence-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_THREADF_1_07-20-2026_t14_07_07_v5_Trial1.cu',
        't3_companions': ['threadfence-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_THREADF_1_mt_input_scaling_sum_v5.cu', 'threadfence-cuda_main_ee_refactorcode_CUDA_expKERNEL_EXP-SC26_THREADF_1_mt_input_scaling_v5.cu'],
        'buffer_classification': 'lassi_verify_analysis/buffer_classification/threadfence_buffer_classification.json',
    },
}


def base_main_path(kernel: str) -> str:
    """Absolute path to kernel's canonical base main Trial file."""
    b = KERNEL_BASES[kernel]
    return os.path.join(REPO_ROOT, b['exp_dir'], b['main_file'])


def base_t3_paths(kernel: str) -> list[str]:
    """Absolute paths to kernel's canonical base T3 companion files."""
    b = KERNEL_BASES[kernel]
    return [os.path.join(REPO_ROOT, b['exp_dir'], f) for f in b['t3_companions']]


def buffer_classification_path(kernel: str) -> str | None:
    b = KERNEL_BASES[kernel]
    if b['buffer_classification'] is None:
        return None
    return os.path.join(REPO_ROOT, b['buffer_classification'])


def verify_all_bases_exist() -> list[str]:
    """Returns a list of problems found (empty = all clear). Never trust the registry
    blindly -- call this before a sweep run.
    """
    problems = []
    for kernel in KERNEL_BASES:
        p = base_main_path(kernel)
        if not os.path.isfile(p):
            problems.append(f'{kernel}: main file missing: {p}')
        for t3 in base_t3_paths(kernel):
            if not os.path.isfile(t3):
                problems.append(f'{kernel}: T3 companion missing: {t3}')
        bc = buffer_classification_path(kernel)
        if bc is not None and not os.path.isfile(bc):
            problems.append(f'{kernel}: buffer_classification.json missing: {bc}')
    return problems


if __name__ == "__main__":
    problems = verify_all_bases_exist()
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"All {len(KERNEL_BASES)} kernel bases verified OK.")
