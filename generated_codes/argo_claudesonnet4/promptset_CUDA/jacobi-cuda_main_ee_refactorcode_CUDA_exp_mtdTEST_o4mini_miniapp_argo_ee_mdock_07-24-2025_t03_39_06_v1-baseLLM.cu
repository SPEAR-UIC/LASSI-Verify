
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
__global__ void jacobi_step_and_swap (float*__restrict__ f, 
                                      float*__restrict__ f_old, 
                                      float*__restrict__ error) {
  __shared__ float f_old_tile[18][18];

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  // First read in the "interior" data, one value per thread
  // Note the offset by 1, to reserve space for the "left"/"bottom" halo
  f_old_tile[ty+1][tx+1] = f_old[IDX(i,j)];

  // Now read in the halo data; we'll pick the "closest" thread
  // to each element. When we do this, make sure we don't fall
  // off the end of the global memory array.
  if (tx == 0 && i >= 1) {
    f_old_tile[ty+1][0] = f_old[IDX(i-1,j)];
  }
  if (tx == 15 && i <= N-2) {
    f_old_tile[ty+1][17] = f_old[IDX(i+1,j)];
  }
  if (ty == 0 && j >= 1) {
    f_old_tile[0][tx+1] = f_old[IDX(i,j-1)];
  }
  if (ty == 15 && j <= N-2) {
    f_old_tile[17][tx+1] = f_old[IDX(i,j+1)];
  }

  // Synchronize all threads
  __syncthreads();

  float err = 0.0f;
  float new_val = 0.0f;

  if (j >= 1 && j <= N-2 && i >= 1 && i <= N-2) {
    // Perform the read from shared memory
    new_val = 0.25f * (f_old_tile[ty+1][tx+2] + 
                       f_old_tile[ty+1][tx+0] + 
                       f_old_tile[ty+2][tx+1] + 
                       f_old_tile[ty+0][tx+1]);
    
    const float df = new_val - f_old_tile[ty+1][tx+1];
    err = df * df;
    
    // Write new value and perform swap in one operation
    f_old[IDX(i,j)] = new_val;  // This is the swap operation
    f[IDX(i,j)] = new_val;      // Update current array
  }

  // Sum over threads in the warp using warp shuffle
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    err += __shfl_down_sync(0xffffffff, err, offset);
  }

  // Block-level reduction using shared memory
  __shared__ float reduction_array[16];
  if (tx == 0) {
    reduction_array[ty] = err;
  }

  __syncthreads();

  // Final reduction using first warp
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

int main () {
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = (float*) aligned_alloc(64, N * N * sizeof(float));
  float* f_old = (float*) aligned_alloc(64, N * N * sizeof(float));

  // Initialize data (we'll do this on both f and f_old, so that we don't
  // have to worry about the boundary points later)
  initialize_data(f);
  initialize_data(f_old);

  cudaMalloc((void**)&d_f, N * N * sizeof(float));
  cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_error, sizeof(float));

  // Initialize error to a large number
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  int num_iters = 0;

  // Energy optimization: Check convergence less frequently to reduce CPU-GPU transfers
  const int convergence_check_interval = 100;

  dim3 grid (N/16, N/16);
  dim3 block (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Initialize error to zero only when we need to check convergence
    if (num_iters % convergence_check_interval == 0) {
      cudaMemset(d_error, 0, sizeof(float));
    }

    // Perform a Jacobi relaxation step and swap in one kernel
    if (num_iters % convergence_check_interval == 0) {
      jacobi_step_and_swap<<<grid, block>>>(d_f, d_f_old, d_error);
      
      // Only copy error back when we need to check convergence
      cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
      
      // Normalize the L2-norm of the error by the number of data points
      // and then take the square root
      error = sqrtf(error / (N * N));
      
      // Periodically print out the current error
      if (num_iters % 1000 == 0) {
        std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
      }
    } else {
      // Run without error calculation for better energy efficiency
      jacobi_step_and_swap<<<grid, block>>>(d_f, d_f_old, nullptr);
    }

    // Increment the iteration count
    ++num_iters;
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  // If we took fewer than max_iters steps and the error is below the tolerance,
  // we succeeded. Otherwise, we failed.

  if (error <= tolerance && num_iters < max_iters) {
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
  free(f);
  free(f_old);

  // End wall timing
  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
