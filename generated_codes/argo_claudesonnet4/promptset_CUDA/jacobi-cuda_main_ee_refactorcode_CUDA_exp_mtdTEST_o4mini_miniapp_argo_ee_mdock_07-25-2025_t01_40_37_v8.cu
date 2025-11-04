
// Copyright (c) 2021, NVIDIA CORPORATION.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

void initialize_data (float* f) {
  // Set up simple sinusoidal boundary conditions
  const float inv_n_minus_1 = 1.0f / (N - 1);
  const float two_pi = 2.0f * M_PI;
  
  for (int j = 0; j < N; ++j) {
    const float sin_j = sinf(j * two_pi * inv_n_minus_1);
    for (int i = 0; i < N; ++i) {
      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sin_j;
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * two_pi * inv_n_minus_1);
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }
    }
  }
}

__global__ void jacobi_step_fused (float*__restrict__ f, 
                                   float*__restrict__ f_old, 
                                   float*__restrict__ error,
                                   bool compute_error,
                                   float tolerance,
                                   int*__restrict__ converged_flag) {
  // Use smaller shared memory with better alignment
  __shared__ float f_tile[18][18];

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // Load data with improved coalescing
  if (i < N && j < N) {
    f_tile[ty+1][tx+1] = f_old[IDX(i,j)];
  }

  // Load halo data more efficiently
  if (tx == 0 && i > 0) {
    f_tile[ty+1][0] = f_old[IDX(i-1,j)];
  }
  if (tx == 15 && i < N-1) {
    f_tile[ty+1][17] = f_old[IDX(i+1,j)];
  }
  if (ty == 0 && j > 0) {
    f_tile[0][tx+1] = f_old[IDX(i,j-1)];
  }
  if (ty == 15 && j < N-1) {
    f_tile[17][tx+1] = f_old[IDX(i,j+1)];
  }

  __syncthreads();

  float err = 0.0f;
  
  if (j >= 1 && j <= N-2 && i >= 1 && i <= N-2) {
    // Compute new value
    const float new_val = 0.25f * (f_tile[ty+1][tx+2] + 
                                   f_tile[ty+1][tx] + 
                                   f_tile[ty+2][tx+1] + 
                                   f_tile[ty][tx+1]);
    
    // Store new value
    f[IDX(i,j)] = new_val;
    
    // Swap data in-place to avoid separate kernel
    f_old[IDX(i,j)] = new_val;
    
    // Compute error only when needed
    if (compute_error) {
      const float df = new_val - f_tile[ty+1][tx+1];
      err = df * df;
    }
  }

  // Error reduction only when needed
  if (compute_error) {
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      err += __shfl_down_sync(0xffffffff, err, offset);
    }

    // Block-level reduction using fewer shared memory
    __shared__ float block_err[16];
    if (tx == 0) {
      block_err[ty] = err;
    }

    __syncthreads();

    if (ty == 0) {
      err = (tx < 16) ? block_err[tx] : 0.0f;
      #pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1) {
        err += __shfl_down_sync(0xffffffff, err, offset);
      }
      if (tx == 0) {
        atomicAdd(error, err);
      }
    }
  }
}

__global__ void check_convergence(float*__restrict__ error, 
                                  float tolerance, 
                                  int*__restrict__ converged_flag) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float total_error = sqrtf(*error / (N * N));
    if (total_error <= tolerance) {
      *converged_flag = 1;
    }
    // Store the computed error back for host access
    *error = total_error;
  }
}

int main () {
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;
  int* d_converged_flag;

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = (float*) aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*) aligned_alloc(64, N * N * sizeof(float));

  // Initialize data
  initialize_data(f);
  initialize_data(f_old);

  // Use cudaMallocManaged for better memory management
  cudaMalloc((void**)&d_f, N * N * sizeof(float));
  cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_error, sizeof(float));
  cudaMalloc((void**)&d_converged_flag, sizeof(int));

  // Initialize convergence flag and error
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;
  int converged = 0;

  // Iterate until we're converged
  const int max_iters = 10000;
  const int error_check_interval = 100; // Check error less frequently
  int num_iters = 0;

  dim3 grid (N/16, N/16);
  dim3 block (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (!converged && num_iters < max_iters) {
    bool check_error = (num_iters % error_check_interval == 0);
    
    if (check_error) {
      cudaMemset(d_error, 0, sizeof(float));
      cudaMemset(d_converged_flag, 0, sizeof(int));
    }

    // Fused kernel that combines Jacobi step and data swap
    jacobi_step_fused<<<grid, block>>>(d_f, d_f_old, d_error, check_error, tolerance, d_converged_flag);

    // Only check convergence and transfer data when needed
    if (check_error) {
      // Device-side convergence check
      check_convergence<<<1, 1>>>(d_error, tolerance, d_converged_flag);
      
      // Only transfer convergence flag and error (much smaller transfer)
      cudaMemcpy(&converged, d_converged_flag, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
      
      // Print progress less frequently
      if (num_iters % 1000 == 0) {
        std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
      }
    }

    ++num_iters;
  }

  // Final error check if we haven't checked recently
  if ((num_iters - 1) % error_check_interval != 0) {
    cudaMemset(d_error, 0, sizeof(float));
    cudaMemset(d_converged_flag, 0, sizeof(int));
    jacobi_step_fused<<<grid, block>>>(d_f, d_f_old, d_error, true, tolerance, d_converged_flag);
    check_convergence<<<1, 1>>>(d_error, tolerance, d_converged_flag);
    cudaMemcpy(&converged, d_converged_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  if (converged && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  // Clean up memory allocations
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  cudaFree(d_converged_flag);
  free(f);
  free(f_old);

  // End wall timing
  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
