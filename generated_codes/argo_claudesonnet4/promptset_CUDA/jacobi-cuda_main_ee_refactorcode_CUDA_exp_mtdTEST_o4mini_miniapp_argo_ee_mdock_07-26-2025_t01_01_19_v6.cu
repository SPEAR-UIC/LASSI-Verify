
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

__global__ void jacobi_step (float*__restrict__ f, 
                             const float*__restrict__ f_old, 
                             float*__restrict__ error) {
  __shared__ float f_old_tile[18][18];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Check bounds before accessing global memory
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
  for (int offset = 16; offset > 0; offset /= 2) {
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

__global__ void swap_data (const float*__restrict__ f,
                                 float*__restrict__ f_old) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Check bounds before accessing global memory
  if (i >= N || j >= N) return;

  if (j >= 1 && j <= N-2) {
    if (i >= 1 && i <= N-2) {
      f_old[IDX(i,j)] = f[IDX(i,j)];
    }
  }
}

__global__ void check_convergence(float* error, float tolerance, int* converged) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float normalized_error = sqrtf(*error / (N * N));
    if (normalized_error <= tolerance) {
      *converged = 1;
    }
    // Reset error for next batch of iterations
    *error = 0.0f;
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

  // Initialize convergence status and error
  int converged = 0;
  err = cudaMemcpy(d_converged, &converged, sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "Failed to initialize d_converged: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_f); cudaFree(d_f_old); cudaFree(d_error); cudaFree(d_converged); free(f); free(f_old);
    return -1;
  }
  
  float error = std::numeric_limits<float>::max();
  const float tolerance = 1.e-5f;

  // Iterate until we're converged (but set a cap on the maximum number of
  // iterations to avoid any possible hangs)
  const int max_iters = 10000;
  const int convergence_check_frequency = 100; // Check convergence every 100 iterations
  int num_iters = 0;

  dim3 grid (N/16, N/16);
  dim3 block (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  while (num_iters < max_iters) {
    // Reset error at the beginning of each batch
    if (num_iters % convergence_check_frequency == 0) {
      cudaMemset(d_error, 0, sizeof(float));
    }

    // Perform a Jacobi relaxation step
    jacobi_step<<<grid, block>>>(d_f, d_f_old, d_error);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      break;
    }

    // Swap the old data and the new data
    // We're doing this explicitly for pedagogical purposes, even though
    // in this specific application a std::swap would have been OK
    swap_data<<<grid, block>>>(d_f, d_f_old);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
      break;
    }

    // Increment the iteration count
    ++num_iters;

    // Check convergence every convergence_check_frequency iterations
    if (num_iters % convergence_check_frequency == 0) {
      // Check convergence on device
      check_convergence<<<1, 1>>>(d_error, tolerance, d_converged);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        break;
      }
      
      // Copy convergence status to host
      err = cudaMemcpy(&converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
      if (err != cudaSuccess) {
        std::cerr << "Failed to copy convergence status: " << cudaGetErrorString(err) << std::endl;
        break;
      }
      
      if (converged) {
        // Get final error for reporting
        err = cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
          error = sqrtf(error / (N * N));
        }
        break;
      }

      // Periodically print out progress (get error for display)
      if (num_iters % 1000 == 0) {
        err = cudaMemcpy(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
          error = sqrtf(error / (N * N));
          std::cout << "Error after iteration " << num_iters << " = " << error << std::endl;
        }
      }
    }
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average execution time per iteration: " << (time * 1e-9f) / num_iters << " (s)\n";

  // If we took fewer than max_iters steps and converged, we succeeded. Otherwise, we failed.
  if (converged && num_iters < max_iters) {
    std::cout << "PASS" << std::endl;
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
