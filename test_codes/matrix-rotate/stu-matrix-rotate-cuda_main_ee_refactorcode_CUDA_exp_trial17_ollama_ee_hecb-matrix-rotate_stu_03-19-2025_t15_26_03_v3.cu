
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__global__ void rotateKernel(float *A, int n) {
    int layer = blockIdx.x;
    if (layer >= (n / 2)) return;

    int first = layer;
    int last = n - 1 - layer;
    int count = last - first; // number of elements per edge

    int offset = threadIdx.x;
    if (offset >= count) return;

    // Compute indices for the four corners
    int i_top = first + offset;
    int j_top = first;

    int i_right = last;
    int j_right = first + offset;

    int i_bottom = last - offset;
    int j_bottom = last;

    int i_left = first;
    int j_left = last - offset;

    // Compute linear indices
    int idxTop = i_top * n + j_top;
    int idxRight = i_right * n + j_right;
    int idxBottom = i_bottom * n + j_bottom;
    int idxLeft = i_left * n + j_left;

    // Read the four elements into registers
    float tempTop = A[idxTop];
    float tempRight = A[idxRight];
    float tempBottom = A[idxBottom];
    float tempLeft = A[idxLeft];

    // Write back in rotated positions
    A[idxTop] = tempLeft;
    A[idxRight] = tempTop;
    A[idxBottom] = tempRight;
    A[idxLeft] = tempBottom;
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

    dim3 threadsPerBlock(256); // or another suitable number
    int numLayers = n / 2;
    dim3 blocksPerGrid(numLayers);
    rotateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_parallel_res, n);

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
