
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
#include <cstdlib>

namespace cg = cooperative_groups;

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

__global__ void jacobi_step (float*__restrict__ f, 
                             const float*__restrict__ f_old, 
                             float*__restrict__ error) {
  __shared__ float f_old_tile[18][18];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Bounds check to prevent out-of-bounds access
  if (i >= N || j >= N) return;

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

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      // Perform the read from shared memory
      f[IDX(i,j)] = 0.25f * (f_old_tile[threadIdx.y+1][threadIdx.x+2] + 
                             f_old_tile[threadIdx.y+1][threadIdx.x+0] + 
                             f_old_tile[threadIdx.y+2][threadIdx.x+1] + 
                             f_old_tile[threadIdx.y+0][threadIdx.x+1]);
      float df = f[IDX(i,j)] - f_old_tile[threadIdx.y+1][threadIdx.x+1];
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

__global__ void device_convergence_check(float* error, int* converged, float tolerance, int num_points, int iteration, int check_frequency) {
  // Use cooperative groups for grid-wide synchronization
  cg::grid_group grid = cg::this_grid();
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Only check convergence at specified intervals
    if (iteration % check_frequency == 0) {
      float normalized_error = sqrtf(*error / num_points);
      if (normalized_error <= tolerance) {
        *converged = 1;  // Signal convergence
      }
      *error = 0.0f;  // Reset error for next batch
    }
  }
  
  // Synchronize across the entire grid to ensure convergence flag is visible
  grid.sync();
}

__global__ void check_convergence_and_reset(float* error, int* converged, float tolerance, int num_points) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float normalized_error = sqrtf(*error / num_points);
    *converged = (normalized_error <= tolerance) ? 1 : 0;
    *error = 0.0f;  // Reset error for next batch
  }
}

int main () {
  // Begin wall timing
  std::clock_t start_time = std::clock();

  float* d_f;
  float* d_f_old;
  float* d_error;
  int* d_converged;

  // Reserve space for the scalar field and the "old" copy of the data
  float* f = (float*) malloc(N * N * sizeof(float));
  float* f_old = (float*) malloc(N * N * sizeof(float));

  // Check for allocation failure
  if (!f || !f_old) {
    std::cerr << "Failed to allocate host memory" << std::endl;
    return -1;
  }

  // Initialize data (we'll do this on both f and f_old, so that we don't
  // have to worry about the boundary points later)
  initialize_data(f);
  initialize_data(f_old);

  cudaError_t err;
  
  err = cudaMalloc((void**)&d_f, N * N * sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for d_f: " << cudaGetErrorString(err) << std::endl;
    free(f); free(f_old);
    return -1;
  }
  
  err = cudaMemcpy(d_f, f, N * N * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data to d_f: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_f); free(f); free(f_old);
    return -1;
  }

  err = cudaMalloc((void**)&d_f_old, N * N * sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for d_f_old: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_f); free(f); free(f_old);
    return -1;
  }
  
  err = cudaMemcpy(d_f_old, f_old, N * N * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to copy data to d_f_old: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_f); cudaFree(d_f_old); free(f); free(f_old);
    return -1;
  }

  err = cudaMalloc((void**)&d_error, sizeof(float));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for d_error: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_f); cudaFree(d_f_old); free(f); free(f_old);
    return -1;
  }
  
  err = cudaMalloc((void**)&d_converged, sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "Failed to allocate device memory for d_converged: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_f); cudaFree(d_f_old); cudaFree(d_error); free(f); free(f_old);
    return -1;
  }

  // Initialize convergence flag and error
  const float tolerance = 1.e-5f;
  const int check_frequency = 100;  // Check convergence every 100 iterations
  int converged = 0;
  
  // Initialize device values
  cudaMemset(d_error, 0, sizeof(float));
  cudaMemcpy(d_converged, &converged, sizeof(int), cudaMemcpyHostToDevice);

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  int num_iters = 0;

  dim3 grid (N/16, N/16);
  dim3 block (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // Main iteration loop - no host-device transfers until convergence
  while (num_iters < max_iters && !converged) {
    // Perform a Jacobi relaxation step
    jacobi_step<<<grid, block>>>(d_f, d_f_old, d_error);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      break;
    }

    // Swap pointers instead of copying data - much more efficient!
    // This eliminates the need for the swap_data kernel entirely
    float* temp = d_f;
    d_f = d_f_old;
    d_f_old = temp;

    // Increment the iteration count
    ++num_iters;

    // Check convergence on device at specified intervals
    if (num_iters % check_frequency == 0) {
      // Use simple kernel launch instead of cooperative kernel
      check_convergence_and_reset<<<1, 1>>>(d_error, d_converged, tolerance, N * N);
      
      // Check for kernel launch errors
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "Convergence check kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        break;
      }
      
      // Synchronize before checking convergence
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::cerr << "Device synchronization failed: " << cudaGetErrorString(err) << std::endl;
        break;
      }
      
      // Check convergence status (single transfer)
      err = cudaMemcpy(&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        std::cerr << "Failed to copy convergence status: " << cudaGetErrorString(err) << std::endl;
        break;
      }

      // Periodically print out status (without transferring error value)
      if (num_iters % 1000 == 0) {
        std::cout << "Completed iteration " << num_iters << std::endl;
      }
    }
  }

  // Final error calculation for reporting (only done once at the end)
  float final_error = 0.0f;
  err = cudaMemcpy(&final_error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    final_error = sqrtf(final_error / (N * N));
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  // If we took fewer than max_iters steps and converged, we succeeded
  if (converged && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
    std::cout << "Final error: " << final_error << std::endl;
  }
  else {
    std::cout << "FAIL" << std::endl;
  }

  // Clean up memory allocations
  cudaFree(d_f);
  cudaFree(d_f_old);
  cudaFree(d_error);
  cudaFree(d_converged);
  free(f);
  free(f_old);

  // End wall timing
  double duration = (std::clock() - start_time) / (double) CLOCKS_PER_SEC;
  std::cout << "Total elapsed time: " << std::setprecision(4) << duration << " seconds" << std::endl;

  return 0;
}
