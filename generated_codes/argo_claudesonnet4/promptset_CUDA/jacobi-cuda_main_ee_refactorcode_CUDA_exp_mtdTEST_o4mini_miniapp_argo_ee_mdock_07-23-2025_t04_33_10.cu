
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

// Device-side convergence check kernel
__global__ void check_convergence(float* error, float tolerance, int* converged) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float current_error = sqrtf(*error / (N * N));
    if (current_error <= tolerance) {
      *converged = 1;
    }
    *error = current_error; // Store normalized error for final reporting
  }
}

// Combined kernel that performs Jacobi step and data swap in one pass
__global__ void jacobi_step_combined (float*__restrict__ f, 
                                     float*__restrict__ f_old, 
                                     float*__restrict__ error,
                                     bool compute_error) {
  // Use smaller shared memory tile to reduce memory usage
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
  float new_val = f_old_tile[ty+1][tx+1]; // Use shared memory value instead of redundant global read

  // Only compute for interior points
  if (i > 0 && i < N-1 && j > 0 && j < N-1) {
    // Perform the Jacobi update using shared memory
    new_val = 0.25f * (f_old_tile[ty+1][tx+2] + 
                       f_old_tile[ty+1][tx+0] + 
                       f_old_tile[ty+2][tx+1] + 
                       f_old_tile[ty+0][tx+1]);
    
    // Compute error only when requested (reduce computation)
    if (compute_error) {
      const float df = new_val - f_old_tile[ty+1][tx+1];
      err = df * df;
    }
  }

  // Update f with new value (combined swap operation)
  f[IDX(i,j)] = new_val;

  // Only perform error reduction when needed
  if (compute_error) {
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      err += __shfl_down_sync(0xffffffff, err, offset);
    }

    // Block-level reduction using fewer shared memory
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
  int* d_converged;
  int* d_final_iterations;

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
  cudaMalloc((void**)&d_converged, sizeof(int));
  cudaMalloc((void**)&d_final_iterations, sizeof(int));

  // Initialize convergence flag and error
  float tolerance = 1.e-5f;
  int max_iters = 10000;

  dim3 grid ((N + 15) / 16, (N + 15) / 16);
  dim3 block (16, 16);

  // Create CUDA stream for better performance
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Adaptive error checking with reduced frequency
  int num_iters = 0;
  int converged = 0;
  float current_error = 0.0f;
  float previous_error = std::numeric_limits<float>::max();
  
  // Start with longer intervals and adapt based on convergence rate
  int base_check_interval = 50;  // Increased from 10 to 50
  int current_check_interval = base_check_interval;
  int iterations_since_check = 0;
  
  while (num_iters < max_iters && !converged) {
    bool check_error = (iterations_since_check >= current_check_interval);
    
    if (check_error) {
      float zero = 0.0f;
      int zero_int = 0;
      cudaMemcpyAsync(d_error, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_converged, &zero_int, sizeof(int), cudaMemcpyHostToDevice, stream);
      iterations_since_check = 0;
    }
    
    // Perform Jacobi step
    jacobi_step_combined<<<grid, block, 0, stream>>>(d_f, d_f_old, d_error, check_error);
    
    // Swap pointers
    float* temp = d_f;
    d_f = d_f_old;
    d_f_old = temp;
    
    if (check_error) {
      // Check convergence on device
      check_convergence<<<1, 1, 0, stream>>>(d_error, tolerance, d_converged);
      
      // Copy convergence result back to host
      cudaMemcpyAsync(&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&current_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      
      // Adaptive interval adjustment based on convergence rate
      if (previous_error != std::numeric_limits<float>::max()) {
        float error_reduction_rate = (previous_error - current_error) / previous_error;
        
        if (error_reduction_rate > 0.1f) {
          // Fast convergence - keep longer intervals for energy efficiency
          current_check_interval = base_check_interval;
        } else if (error_reduction_rate > 0.01f) {
          // Moderate convergence - slightly reduce interval
          current_check_interval = base_check_interval * 3 / 4;
        } else {
          // Slow convergence - reduce interval for better monitoring
          current_check_interval = base_check_interval / 2;
        }
        
        // Ensure minimum interval for energy efficiency
        current_check_interval = std::max(current_check_interval, 25);
      }
      
      previous_error = current_error;
    }
    
    ++num_iters;
    ++iterations_since_check;
  }

  cudaStreamSynchronize(stream);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

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
  cudaStreamDestroy(stream);
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  cudaFree(d_converged);
  cudaFree(d_final_iterations);
  free(f);
  free(f_old);

  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
