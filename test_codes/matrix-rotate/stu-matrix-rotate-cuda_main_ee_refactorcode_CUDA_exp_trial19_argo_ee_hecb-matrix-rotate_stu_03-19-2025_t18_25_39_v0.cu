
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cuda.h>

__global__ void rotate_matrix_parallel(float *input_matrix, float *output_matrix, const int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < n && y < n) {
    int new_x = n - 1 - y;
    int new_y = x;
    output_matrix[new_y * n + new_x] = input_matrix[y * n + x];
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <matrix size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  float *h_input_matrix, *h_output_matrix;
  cudaMallocHost((void**)&h_input_matrix, n*n*sizeof(float)); // Use pinned memory for faster transfers
  cudaMallocHost((void**)&h_output_matrix, n*n*sizeof(float));

  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      h_input_matrix[i*n+j] = i*n+j;

  float *d_input_matrix, *d_output_matrix;
  cudaMalloc((void**)&d_input_matrix, n*n*sizeof(float));
  cudaMalloc((void**)&d_output_matrix, n*n*sizeof(float));

  cudaMemcpy(d_input_matrix, h_input_matrix, n*n*sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(16, 16); // Use a block size that is a multiple of warp size
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < repeat; i++) {
    rotate_matrix_parallel<<<gridDim, blockDim>>>(d_input_matrix, d_output_matrix, n);
  }
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Average kernel execution time: %f (ms)\n", milliseconds / repeat);

  cudaMemcpy(h_output_matrix, d_output_matrix, n*n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFreeHost(h_input_matrix);
  cudaFreeHost(h_output_matrix);
  cudaFree(d_input_matrix);
  cudaFree(d_output_matrix);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
