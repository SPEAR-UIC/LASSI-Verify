
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <ctime>
#include <chrono>
#include <cuda.h>

// A multiple of thread block size
#define N 16384

#define IDX(i, j) ((i) + (j) * N)

void initialize_data(float* f) {
  // Set up simple sinusoidal boundary conditions
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
      } else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
      } else {
        f[IDX(i,j)] = 0.0f;
      }
    }
  }
}

__global__ void jacobi_step(float* __restrict__ f,
                            const float* __restrict__ f_old,
                            float* __restrict__ error) {
  __shared__ float f_old_tile[18][18];
  __shared__ float reduction_array[16];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Load data into shared memory
  f_old_tile[threadIdx.y + 1][threadIdx.x + 1] = f_old[IDX(i, j)];

  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 0] = f_old[IDX(i - 1, j)];
  }
  if (threadIdx.x == 15 && i <= N - 2) {
    f_old_tile[threadIdx.y + 1][threadIdx.x + 2] = f_old[IDX(i + 1, j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[threadIdx.y + 0][threadIdx.x + 1] = f_old[IDX(i, j - 1)];
  }
  if (threadIdx.y == 15 && j <= N - 2) {
    f_old_tile[threadIdx.y + 2][threadIdx.x + 1] = f_old[IDX(i, j + 1)];
  }

  __syncthreads();

  float err = 0.0f;

  if (j >= 1 && j <= N - 2 && i >= 1 && i <= N - 2) {
    f[IDX(i, j)] = 0.25f * (f_old_tile[threadIdx.y + 1][threadIdx.x + 2] +
                            f_old_tile[threadIdx.y + 1][threadIdx.x + 0] +
                            f_old_tile[threadIdx.y + 2][threadIdx.x + 1] +
                            f_old_tile[threadIdx.y + 0][threadIdx.x + 1]);
    float df = f[IDX(i, j)] - f_old_tile[threadIdx.y + 1][threadIdx.x + 1];
    err = df * df;
  }

  // Warp-level reduction within the block
  for (int offset = 8; offset > 0; offset /= 2) {
    err += __shfl_down_sync(0xffffffff, err, offset);
  }

  if (threadIdx.x == 0) {
    reduction_array[threadIdx.y] = err;
  }

  __syncthreads();

  // Final reduction using the first warp
  if (threadIdx.y == 0) {
    err = reduction_array[threadIdx.x];
    for (int offset = 8; offset > 0; offset /= 2) {
      err += __shfl_down_sync(0xffffffff, err, offset);
    }
    if (threadIdx.x == 0) {
      atomicAdd(error, err);
    }
  }
}

__global__ void swap_data(const float* __restrict__ f,
                          float* __restrict__ f_old) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= 1 && j <= N - 2 && i >= 1 && i <= N - 2) {
    f_old[IDX(i, j)] = f[IDX(i, j)];
  }
}

int main() {
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;

  float* f = (float*)aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*)aligned_alloc(64, N * N * sizeof(float));

  initialize_data(f);
  initialize_data(f_old);

  cudaMalloc((void**)&d_f, N * N * sizeof(float));
  cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_error, sizeof(float));

  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  const int max_iters = 10000;
  int num_iters = 0;

  dim3 grid(N / 16, N / 16);
  dim3 block(16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    cudaMemset(d_error, 0, sizeof(float));

    jacobi_step<<<grid, block>>>(d_f, d_f_old, d_error);

    swap_data<<<grid, block>>>(d_f, d_f_old);

    cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);

    error = sqrtf(error / (N * N));

    if (num_iters % 1000 == 0) {
      std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
    }

    ++num_iters;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  } else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  free(f);
  free(f_old);

  double duration = (std::clock() - start_time) / (double)CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
