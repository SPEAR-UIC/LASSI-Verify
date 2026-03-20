
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
#include <cooperative_groups.h>

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

// Persistent kernel that runs entire simulation on device
__global__ void jacobi_persistent_kernel(float*__restrict__ f, 
                                        float*__restrict__ f_old, 
                                        float tolerance,
                                        int max_iters,
                                        int* final_iterations,
                                        float* final_error) {
  
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  
  __shared__ float f_old_tile[18][18];
  __shared__ float block_error;
  __shared__ bool block_converged;
  
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  
  // Early exit for out-of-bounds threads
  if (i >= N || j >= N) return;
  
  int num_iters = 0;
  bool converged = false;
  float global_error = 0.0f;
  
  while (num_iters < max_iters && !converged) {
    // Initialize block-level error for this iteration
    if (tid == 0) {
      block_error = 0.0f;
      block_converged = false;
    }
    __syncthreads();
    
    // Load data into shared memory
    f_old_tile[threadIdx.y+1][threadIdx.x+1] = f_old[IDX(i,j)];
    
    // Load halo data efficiently
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    if (tx == 0 && i > 0) {
      f_old_tile[ty+1][0] = f_old[IDX(i-1,j)];
    }
    if (tx == 15 && i < N-1) {
      f_old_tile[ty+1][17] = f_old[IDX(i+1,j)];
    }
    if (ty == 0 && j > 0) {
      f_old_tile[0][tx+1] = f_old[IDX(i,j-1)];
    }
    if (ty == 15 && j < N-1) {
      f_old_tile[17][tx+1] = f_old[IDX(i,j+1)];
    }
    
    __syncthreads();
    
    float err = 0.0f;
    float new_val = f_old_tile[ty+1][tx+1];
    
    // Only compute for interior points
    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
      // Perform the Jacobi update using shared memory
      new_val = 0.25f * (f_old_tile[ty+1][tx+2] + 
                         f_old_tile[ty+1][tx+0] + 
                         f_old_tile[ty+2][tx+1] + 
                         f_old_tile[ty+0][tx+1]);
      
      // Compute error every 10 iterations
      if (num_iters % 10 == 0) {
        const float df = new_val - f_old_tile[ty+1][tx+1];
        err = df * df;
      }
    }
    
    // Update f with new value
    f[IDX(i,j)] = new_val;
    
    // Error reduction and convergence check every 10 iterations
    if (num_iters % 10 == 0) {
      // Warp-level reduction
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        err += __shfl_down_sync(0xffffffff, err, offset);
      }
      
      // Block-level reduction
      __shared__ float reduction_array[16];
      if (tx == 0) {
        reduction_array[ty] = err;
      }
      
      __syncthreads();
      
      if (ty == 0) {
        err = reduction_array[tx];
        #pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1) {
          err += __shfl_down_sync(0xffffffff, err, offset);
        }
        if (tx == 0) {
          atomicAdd(&block_error, err);
        }
      }
      
      __syncthreads();
      
      // Grid-level convergence check using cooperative groups
      if (tid == 0) {
        // Use grid-level reduction for error across all blocks
        cg::grid_group grid = cg::this_grid();
        
        // Simple grid-level error aggregation
        __shared__ float grid_error_sum;
        if (blockIdx.x == 0 && blockIdx.y == 0) {
          grid_error_sum = 0.0f;
        }
        
        grid.sync();
        
        // Each block contributes its error
        if (blockIdx.x == 0 && blockIdx.y == 0) {
          atomicAdd(&grid_error_sum, block_error);
        }
        
        grid.sync();
        
        // Check convergence on the first block
        if (blockIdx.x == 0 && blockIdx.y == 0) {
          global_error = sqrtf(grid_error_sum / (N * N));
          if (global_error <= tolerance) {
            block_converged = true;
          }
        }
      }
      
      __syncthreads();
      
      // Broadcast convergence result to all threads in this block
      converged = block_converged;
      
      // Sync across entire grid to ensure all blocks see convergence
      grid.sync();
    }
    
    // Swap pointers by swapping the arrays themselves
    float* temp = f;
    f = f_old;
    f_old = temp;
    
    ++num_iters;
    
    // Sync grid before next iteration
    grid.sync();
  }
  
  // Store final results (only first thread of first block)
  if (blockIdx.x == 0 && blockIdx.y == 0 && tid == 0) {
    *final_iterations = num_iters;
    *final_error = global_error;
  }
}

int main () {
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  int* d_final_iterations;
  float* d_final_error;

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = (float*) aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*) aligned_alloc(64, N * N * sizeof(float));

  // Initialize data
  initialize_data(f);
  initialize_data(f_old);

  // Allocate device memory
  cudaMalloc((void**)&d_f, N * N * sizeof(float));
  cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_final_iterations, sizeof(int));
  cudaMalloc((void**)&d_final_error, sizeof(float));

  // Initialize parameters
  float tolerance = 1.e-5f;
  int max_iters = 10000;

  dim3 grid ((N + 15) / 16, (N + 15) / 16);
  dim3 block (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Launch persistent kernel that runs entire simulation on device
  void* kernel_args[] = {
    &d_f, &d_f_old, &tolerance, &max_iters, &d_final_iterations, &d_final_error
  };
  
  cudaLaunchCooperativeKernel((void*)jacobi_persistent_kernel, grid, block, kernel_args, 0, 0);
  
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // Copy final results back to host
  int num_iters;
  float current_error;
  cudaMemcpy(&num_iters, d_final_iterations, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&current_error, d_final_error, sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";
  std::cout << "Final error after " << num_iters << " iterations: " << current_error << std::endl;

  if (current_error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  // Clean up
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_final_iterations);
  cudaFree(d_final_error);
  free(f);
  free(f_old);

  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
