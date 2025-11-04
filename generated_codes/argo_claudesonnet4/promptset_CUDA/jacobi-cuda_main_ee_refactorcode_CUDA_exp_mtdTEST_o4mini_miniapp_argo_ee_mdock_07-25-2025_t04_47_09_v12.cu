
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
#define M_PI_F 3.14159265358979323846f

__global__ void initialize_data_kernel(float* f, float* f_old) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (i < N && j < N) {
    float value;
    
    if (i == 0 || i == N-1) {
      value = sinf(j * 2.0f * M_PI_F / (N - 1));
    }
    else if (j == 0 || j == N-1) {
      value = sinf(i * 2.0f * M_PI_F / (N - 1));
    }
    else {
      value = 0.0f;
    }
    
    f[IDX(i,j)] = value;
    f_old[IDX(i,j)] = value;
  }
}

__global__ void jacobi_step_fused (float*__restrict__ f, 
                                   float*__restrict__ f_old, 
                                   float*__restrict__ error) {
  __shared__ float f_old_tile[18][18];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // First read in the "interior" data, one value per thread
  // Note the offset by 1, to reserve space for the "left"/"bottom" halo

  f_old_tile[threadIdx.y+1][threadIdx.x+1] = f_old[IDX(i,j)];

  // Now read in the halo data; we'll pick the "closest" thread
  // to each element. When we do this, make sure we don't fall
  // off the end of the global memory array. Note that this
  // code does not fill the corners, as they are not used in
  // this stencil.

  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y+1][threadIdx.x+0] = f_old[IDX(i-1,j)];
  }
  if (threadIdx.x == 15 && i <= N-2) {
    f_old_tile[threadIdx.y+1][threadIdx.x+2] = f_old[IDX(i+1,j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[threadIdx.y+0][threadIdx.x+1] = f_old[IDX(i,j-1)];
  }
  if (threadIdx.y == 15 && j <= N-2) {
    f_old_tile[threadIdx.y+2][threadIdx.x+1] = f_old[IDX(i,j+1)];
  }

  // Synchronize all threads
  __syncthreads();

  float err = 0.0f;
  float new_value = 0.0f;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      // Perform the read from shared memory
      new_value = 0.25f * (f_old_tile[threadIdx.y+1][threadIdx.x+2] + 
                           f_old_tile[threadIdx.y+1][threadIdx.x+0] + 
                           f_old_tile[threadIdx.y+2][threadIdx.x+1] + 
                           f_old_tile[threadIdx.y+0][threadIdx.x+1]);
      
      // Write to f array only - removed redundant write to f_old
      f[IDX(i,j)] = new_value;
      
      // Calculate error
      float df = new_value - f_old_tile[threadIdx.y+1][threadIdx.x+1];
      err = df * df;
    }
  }

  // Sum over threads in the warp
  // For simplicity, we do this outside the above conditional
  // so that all threads participate
  for (int offset = 8; offset > 0; offset /= 2) {
    err += __shfl_down_sync(0xffffffff, err, offset);
  }

  // If we're thread 0 in the warp, update our value to shared memory
  // Note that we're assuming exactly a 16x16 block and that the warp ID
  // is equivalent to threadIdx.y. For the general case, we would have to
  // write more careful code.
  __shared__ float reduction_array[16];
  if (threadIdx.x == 0) {
    reduction_array[threadIdx.y] = err;
  }

  // Synchronize the block before reading any values from smem
  __syncthreads();

  // Using the first warp in the block, reduce over the partial sums
  // in the shared memory array.
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

__global__ void check_convergence(const float*__restrict__ error, 
                                  float tolerance_squared, 
                                  float inv_n_squared,
                                  int*__restrict__ converged) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float normalized_error = (*error) * inv_n_squared;
    if (normalized_error <= tolerance_squared) {
      *converged = 1;
    }
  }
}

__global__ void swap_data (const float*__restrict__ f,
                                 float*__restrict__ f_old) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      f_old[IDX(i,j)] = f[IDX(i,j)];
    }
  }
}

int main () {
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;
  int* d_converged;

  // Allocate GPU memory directly without host arrays
  cudaMalloc((void**)&d_f, N * N * sizeof(float));
  cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  cudaMalloc((void**)&d_error, sizeof(float));
  cudaMalloc((void**)&d_converged, sizeof(int));

  // Initialize data directly on GPU
  dim3 grid (N/16, N/16);
  dim3 block (16, 16);
  initialize_data_kernel<<<grid, block>>>(d_f, d_f_old);

  // Pre-compute constants to avoid repeated calculations
  const float tolerance = 1.e-5f;
  const float tolerance_squared = tolerance * tolerance;
  const float inv_n_squared = 1.0f / (N * N);

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  const int convergence_check_interval = 10; // Check convergence every 10 iterations
  int num_iters = 0;
  int converged = 0;
  float error = std::numeric_limits<float>::max();

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (num_iters < max_iters && !converged) {
    // Initialize error to zero (we'll add to it the following step)
    cudaMemset(d_error, 0, sizeof(float));

    // Perform Jacobi relaxation step (no longer fused with swap)
    jacobi_step_fused<<<grid, block>>>(d_f, d_f_old, d_error);

    ++num_iters;

    // Check convergence only every convergence_check_interval iterations
    if (num_iters % convergence_check_interval == 0) {
      // Reset convergence flag
      cudaMemset(d_converged, 0, sizeof(int));
      
      // Check convergence on device
      check_convergence<<<1, 1>>>(d_error, tolerance_squared, inv_n_squared, d_converged);
      
      // Copy convergence result to host
      cudaMemcpy(&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
      
      // For periodic printing, we need the actual error value
      if (num_iters % 1000 == 0) {
        cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
        error = sqrtf(error * inv_n_squared);
        std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
      }
    }

    // Efficient pointer swap instead of data copying - eliminates redundant memory operations
    float* temp = d_f;
    d_f = d_f_old;
    d_f_old = temp;
  }

  // Get final error for output validation
  cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
  error = sqrtf(error * inv_n_squared);

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
  cudaFree(d_converged);

  // End wall timing
  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
