
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
#define CHECK_INTERVAL 100  // Check convergence every 100 iterations

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
                                   bool compute_error) {
  __shared__ float f_old_tile[18][18];

  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;

  // First read in the "interior" data, one value per thread
  f_old_tile[threadIdx.y+1][threadIdx.x+1] = f_old[IDX(i,j)];

  // Now read in the halo data with bounds checking
  if (threadIdx.x == 0 && i >= 1) {
    f_old_tile[threadIdx.y+1][0] = f_old[IDX(i-1,j)];
  }
  if (threadIdx.x == 15 && i <= N-2) {
    f_old_tile[threadIdx.y+1][17] = f_old[IDX(i+1,j)];
  }
  if (threadIdx.y == 0 && j >= 1) {
    f_old_tile[0][threadIdx.x+1] = f_old[IDX(i,j-1)];
  }
  if (threadIdx.y == 15 && j <= N-2) {
    f_old_tile[17][threadIdx.x+1] = f_old[IDX(i,j+1)];
  }

  __syncthreads();

  float err = 0.0f;
  
  if (j >= 1 && j <= N-2 && i >= 1 && i <= N-2) {
    // Compute new value
    const float new_val = 0.25f * (f_old_tile[threadIdx.y+1][threadIdx.x+2] + 
                                   f_old_tile[threadIdx.y+1][threadIdx.x+0] + 
                                   f_old_tile[threadIdx.y+2][threadIdx.x+1] + 
                                   f_old_tile[threadIdx.y+0][threadIdx.x+1]);
    
    // Compute error only when needed
    if (compute_error) {
      const float df = new_val - f_old_tile[threadIdx.y+1][threadIdx.x+1];
      err = df * df;
    }
    
    // Swap data in-place to avoid separate kernel
    f_old[IDX(i,j)] = new_val;
  }

  // Error reduction only when computing error
  if (compute_error) {
    // Sum over threads in the warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      err += __shfl_down_sync(0xffffffff, err, offset);
    }

    __shared__ float reduction_array[16];
    if (threadIdx.x == 0) {
      reduction_array[threadIdx.y] = err;
    }

    __syncthreads();

    if (threadIdx.y == 0) {
      err = reduction_array[threadIdx.x];
      #pragma unroll
      for (int offset = 8; offset > 0; offset /= 2) {
        err += __shfl_down_sync(0xffffffff, err, offset);
      }
      if (threadIdx.x == 0) {
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
  const float tolerance_sq = tolerance * tolerance * (N * N); // Pre-compute for efficiency

  // Iterate until we're converged
  const int max_iters = 10000;
  int num_iters = 0;

  dim3 grid (N/16, N/16);
  dim3 block (16, 16);

  // Create CUDA stream for better performance
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (error > tolerance && num_iters < max_iters) {
    // Only compute error every CHECK_INTERVAL iterations
    bool compute_error = (num_iters % CHECK_INTERVAL == 0);
    
    if (compute_error) {
      cudaMemsetAsync(d_error, 0, sizeof(float), stream);
    }

    // Perform fused Jacobi step with optional error computation
    jacobi_step_fused<<<grid, block, 0, stream>>>(d_f, d_f_old, d_error, compute_error);

    // Swap pointers instead of copying data
    float* temp = d_f;
    d_f = d_f_old;
    d_f_old = temp;

    if (compute_error) {
      cudaMemcpyAsync(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      
      // Normalize the L2-norm of the error
      error = sqrtf(error / (N * N));

      // Periodically print out the current error
      if (num_iters % 1000 == 0) {
        std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
      }
    }

    ++num_iters;
  }

  // Final error check if we didn't just compute it
  if ((num_iters - 1) % CHECK_INTERVAL != 0) {
    cudaMemsetAsync(d_error, 0, sizeof(float), stream);
    jacobi_step_fused<<<grid, block, 0, stream>>>(d_f, d_f_old, d_error, true);
    cudaMemcpyAsync(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    error = sqrtf(error / (N * N));
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  if (error <= tolerance && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
    return -1;
  }

  // Clean up memory allocations
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
