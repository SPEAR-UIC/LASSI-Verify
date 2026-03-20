
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
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {

      if (i == 0 || i == N-1) {
        f[IDX(i,j)] = sinf(j * 2 * M_PI / (N - 1));
      }
      else if (j == 0 || j == N-1) {
        f[IDX(i,j)] = sinf(i * 2 * M_PI / (N - 1));
      }
      else {
        f[IDX(i,j)] = 0.0f;
      }

    }
  }
}

// Combined kernel that performs Jacobi step and data swap in one pass
__global__ void jacobi_step_and_swap(float*__restrict__ f, 
                                     float*__restrict__ f_old, 
                                     float*__restrict__ error,
                                     bool compute_error) {
  // Use 32-byte aligned shared memory to avoid bank conflicts
  __shared__ float f_old_tile[18][18];

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Early exit for out-of-bounds threads
  if (i >= N || j >= N) return;

  // First read in the "interior" data, one value per thread
  f_old_tile[threadIdx.y+1][threadIdx.x+1] = f_old[IDX(i,j)];

  // Load halo data more efficiently with fewer conditionals
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  
  // Load left halo
  if (tx == 0 && i > 0) {
    f_old_tile[ty+1][0] = f_old[IDX(i-1,j)];
  }
  // Load right halo
  if (tx == 15 && i < N-1) {
    f_old_tile[ty+1][17] = f_old[IDX(i+1,j)];
  }
  // Load bottom halo
  if (ty == 0 && j > 0) {
    f_old_tile[0][tx+1] = f_old[IDX(i,j-1)];
  }
  // Load top halo
  if (ty == 15 && j < N-1) {
    f_old_tile[17][tx+1] = f_old[IDX(i,j+1)];
  }

  __syncthreads();

  float err = 0.0f;
  float new_val = f_old_tile[ty+1][tx+1]; // Default to old value

  // Only compute for interior points
  if (i > 0 && i < N-1 && j > 0 && j < N-1) {
    // Compute new value using shared memory
    new_val = 0.25f * (f_old_tile[ty+1][tx+2] + 
                       f_old_tile[ty+1][tx+0] + 
                       f_old_tile[ty+2][tx+1] + 
                       f_old_tile[ty+0][tx+1]);
    
    // Store new value
    f[IDX(i,j)] = new_val;
    
    // Compute error only when requested
    if (compute_error) {
      const float df = new_val - f_old_tile[ty+1][tx+1];
      err = df * df;
    }
    
    // Update f_old for next iteration (swap operation)
    f_old[IDX(i,j)] = new_val;
  }

  // Error reduction only when needed
  if (compute_error) {
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      err += __shfl_down_sync(0xffffffff, err, offset);
    }

    // Block-level reduction using fewer shared memory operations
    __shared__ float reduction_array[16];
    if (tx == 0) {
      reduction_array[ty] = err;
    }

    __syncthreads();

    // Final reduction
    if (ty == 0) {
      err = reduction_array[tx];
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

int main () {
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;

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

  // Initialize error to a large number
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  // Iterate until we're converged
  const int max_iters = 10000;
  const int error_check_interval = 100; // Check error less frequently
  int num_iters = 0;

  dim3 grid ((N + 15) / 16, (N + 15) / 16);
  dim3 block (16, 16);

  // Create CUDA stream for better performance
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Check error less frequently to reduce memory transfers
    bool check_error = (num_iters % error_check_interval == 0);
    
    if (check_error) {
      cudaMemsetAsync(d_error, 0, sizeof(float), stream);
    }

    // Combined Jacobi step and swap operation
    jacobi_step_and_swap<<<grid, block, 0, stream>>>(d_f, d_f_old, d_error, check_error);

    if (check_error) {
      cudaMemcpyAsync(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      
      // Normalize the L2-norm of the error
      error = sqrtf(error / (N * N));

      // Print progress less frequently
      if (num_iters % 1000 == 0) {
        std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
      }
    }

    ++num_iters;
  }

  cudaStreamSynchronize(stream);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  // Final error check if we haven't checked recently
  if (num_iters % error_check_interval != 0) {
    cudaMemset(d_error, 0, sizeof(float));
    jacobi_step_and_swap<<<grid, block>>>(d_f, d_f_old, d_error, true);
    cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
    error = sqrtf(error / (N * N));
  }

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  // Clean up
  cudaStreamDestroy(stream);
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  free(f);
  free(f_old);

  // End wall timing
  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
