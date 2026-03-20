
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__global__ void rotate_matrix_parallel(float *matrix, const int n) {
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int layer_count = n / 2; // Number of layers to process

    for (int layer = global_id; layer < layer_count; layer += blockDim.x * gridDim.x) {
        int first = layer;
        int last = n - 1 - layer;

        for (int i = first + 1; i < last; ++i) {
            // Compute indices for the four corners of this offset
            int top_row = first;
            int right_col = i;
            int bottom_row = last;
            int left_col = last - (i - first);

            // Load values into registers
            float temp_top = matrix[top_row * n + i];
            float temp_right = matrix[right_col * n + last];
            float temp_bottom = matrix[bottom_row * n + left_col];
            float temp_left = matrix[left_col * n + top_row];

            // Perform the swap in a cycle: top -> left, left -> bottom, etc.
            matrix[top_row * n + i] = temp_left;
            matrix[right_col * n + last] = temp_top;
            matrix[bottom_row * n + left_col] = temp_right;
            matrix[left_col * n + top_row] = temp_bottom;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
        return 1;
    }
    const int n = atoi(argv[1]);
    const int repeat = atoi(argv[2]);

    float *parallel_res = (float*) aligned_alloc(1024, n*n*sizeof(float));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            parallel_res[i*n+j] = i*n+j;

    float *d_parallel_res;
    cudaMalloc((void**)&d_parallel_res, n*n*sizeof(float));
    cudaMemcpy(d_parallel_res, parallel_res, n*n*sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    dim3 blocks(1); // Single block
    dim3 threads(1024); // Larger block size for better occupancy

    for (int i = 0; i < repeat; i++) {
        rotate_matrix_parallel<<<blocks, threads>>>(d_parallel_res, n);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    cudaMemcpy(parallel_res, d_parallel_res, n*n*sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) {
        for (int j = 0; j < n && ok; ++j) {
            int expected_value = (n - 1 - j) * n + i;
            if (parallel_res[i*n+j] != expected_value) {
                printf("Mismatch at (%d, %d): expected %d, got %f\n", i, j, expected_value, parallel_res[i*n+j]);
                ok = false;
            }
        }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");

    free(parallel_res);
    cudaFree(d_parallel_res);
    return 0;
}
