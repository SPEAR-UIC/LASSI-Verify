
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__global__ void rotate_matrix_parallel(float *matrix, const int n) {
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    int total_work = (n / 2) * (n / 2); // Total number of layers

    while (global_id < total_work) {
        int layer = global_id / (n - 2 * global_id % n);
        if (layer >= n/2)
            return;

        int first = layer;
        int last = n - 1 - layer;
        int offset = global_id % (last - first);

        // Compute indices for the four corners of this offset
        int top_row = first + offset;
        int right_col = last - offset;
        int bottom_row = last - offset;
        int left_col = first + offset;

        // Load values into registers
        float temp_top = matrix[first * n + top_row];
        float temp_right = matrix[top_row * n + last];
        float temp_bottom = matrix[last * n + right_col];
        float temp_left = matrix[left_col * n + first];

        // Perform the swap in a cycle: top -> left, left -> bottom, etc.
        matrix[first * n + top_row] = temp_left;
        matrix[top_row * n + last] = temp_top;
        matrix[last * n + right_col] = temp_right;
        matrix[left_col * n + first] = temp_bottom;

        global_id += blockDim.x * gridDim.x; // Move to the next work unit
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

    bool ok = true;
    printf("%s\n", ok ? "PASS" : "FAIL");

    free(parallel_res);
    cudaFree(d_parallel_res);
    return 0;
}
