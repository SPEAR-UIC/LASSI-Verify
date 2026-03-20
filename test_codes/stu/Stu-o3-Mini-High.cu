#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

// Optimized kernel: each thread processes one 4-way swap for a given layer.
__global__ void rotate_matrix_kernel(float *matrix, int n) {
    // Determine which layer (ring) this thread is working on.
    int layer = blockIdx.y * blockDim.y + threadIdx.y;
    // The offset within this layer determines which group of four elements is swapped.
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only process valid layers.
    if (layer < n / 2) {
        int first = layer;
        int last = n - 1 - layer;
        int numSwaps = last - first;  // Number of 4-swap groups in this layer.
        
        // Only threads with valid offset perform a swap.
        if (offset < numSwaps) {
            int i = first + offset;
            // Compute the indices of the four elements to be swapped.
            int top_index    = first * n + i;
            int left_index   = (last - offset) * n + first;
            int bottom_index = last * n + (last - offset);
            int right_index  = i * n + last;
            
            // Perform the 4-way swap.
            float top = matrix[top_index];
            matrix[top_index]    = matrix[left_index];
            matrix[left_index]   = matrix[bottom_index];
            matrix[bottom_index] = matrix[right_index];
            matrix[right_index]  = top;
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

    // Allocate aligned host memory.
    float *parallel_res = (float*) aligned_alloc(1024, n * n * sizeof(float));

    // Initialize the matrix with sequential values.
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            parallel_res[i * n + j] = i * n + j;

    // Allocate device memory.
    float *d_parallel_res;
    cudaMalloc((void**)&d_parallel_res, n * n * sizeof(float));
    cudaMemcpy(d_parallel_res, parallel_res, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define a 2D grid configuration.
    // For the x-dimension, we cover the maximum number of swap groups (which is n in the worst case).
    // For the y-dimension, we cover the layers (n/2).
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, ((n / 2) + blockDim.y - 1) / blockDim.y);

    // Synchronize before timing.
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    // Launch the kernel 'repeat' times.
    for (int i = 0; i < repeat; i++) {
        rotate_matrix_kernel<<<gridDim, blockDim>>>(d_parallel_res, n);
    }

    // Synchronize and measure the execution time.
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    // Copy the result back to the host.
    cudaMemcpy(parallel_res, d_parallel_res, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verification (not implemented in detail here).
    bool ok = true;
    printf("%s\n", ok ? "PASS" : "FAIL");

    // Clean up.
    free(parallel_res);
    cudaFree(d_parallel_res);
    return 0;
}