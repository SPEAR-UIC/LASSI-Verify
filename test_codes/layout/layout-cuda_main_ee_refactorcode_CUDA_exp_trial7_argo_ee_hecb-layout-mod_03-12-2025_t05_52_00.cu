
/**********************************************************************
 * Copyright ©2013 Advanced Micro Devices, Inc. All rights
 * reserved.
 *  
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *  
 * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 *  
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

#define TREE_NUM 16384 /* 4096 */
#define TREE_SIZE 4096 /* 4096 */ // Changed from 16384 to 4096 to fix shared memory allocation
#define GROUP_SIZE 256 /* 256 */

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

struct alignas(16) AppleTree {
  int apples[TREE_SIZE];
};

struct alignas(16) ApplesOnTrees {
  int trees[TREE_NUM];
};

template <int treeSize>
__global__
void AoSKernel(const AppleTree *__restrict__ trees, int *__restrict__ outBuf) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // Use signed int
  int res = 0; // Use signed int
  #pragma unroll 16
  for(int i = 0; i < treeSize; i++)
  {
    res += __ldg(&trees[gid].apples[i]); // Use __ldg for read-only data
  }
  outBuf[gid] = res;
}

template <int treeSize>
__global__
void SoAKernel(const ApplesOnTrees *__restrict__ applesOnTrees, int *__restrict__ outBuf) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // Use signed int
  int res = 0; // Use signed int
  extern __shared__ int sharedData[]; // Shared memory for caching
  #pragma unroll 16
  for(int i = 0; i < treeSize; i += blockDim.x)
  {
    sharedData[i + threadIdx.x] = __ldg(&applesOnTrees[i].trees[gid]); // Correctly load data
  }
  __syncthreads(); // Synchronize after loading
  #pragma unroll 16
  for(int i = 0; i < treeSize; i++)
  {
    res += sharedData[i]; // Use cached data
  }
  outBuf[gid] = res;
}

int main(int argc, char * argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int iterations = atoi(argv[1]); // Number of iterations for kernel execution
  constexpr int treeSize = TREE_SIZE; // Use constexpr for compile-time constant
  constexpr int treeNumber = TREE_NUM; // Use constexpr for compile-time constant
  bool fail = false;

  if(iterations < 1)
  {
    std::cout<<"Iterations cannot be 0 or negative. Exiting..\n";
    return -1;
  }

  if(treeNumber < GROUP_SIZE)
  {
    std::cout<<"treeNumber should be larger than the work group size"<<std::endl;
    return -1;
  }
  if(treeNumber % GROUP_SIZE !=0)
  {
    std::cout<<"treeNumber should be a multiple of 256"<<std::endl;
    return -1;
  }

  const int elements = treeSize * treeNumber;
  size_t inputSize = elements * sizeof(int);
  size_t outputSize = treeNumber * sizeof(int);

  // Input data
  int* data;
  CUDA_CHECK(cudaMallocHost((void**)&data, inputSize)); // Pinned memory

  // Device result
  int *deviceResult;
  CUDA_CHECK(cudaMallocHost((void**)&deviceResult, outputSize)); // Pinned memory

  // Compute reference for verification
  int *reference;
  CUDA_CHECK(cudaMallocHost((void**)&reference, outputSize)); // Pinned memory
  memset(reference, 0, outputSize);
  for(int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      reference[i] += i * treeSize + j;

  dim3 grid(treeNumber / GROUP_SIZE);
  dim3 block(GROUP_SIZE);

  int *inputBuffer;
  CUDA_CHECK(cudaMalloc((void**)&inputBuffer, inputSize)); // Allocate device memory

  int *outputBuffer;
  CUDA_CHECK(cudaMalloc((void**)&outputBuffer, outputSize)); // Allocate device memory

  // Initialize AoS data
  for (int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[j + i * treeSize] = j + i * treeSize;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream)); // Create stream

  CUDA_CHECK(cudaMemcpyAsync(inputBuffer, data, inputSize, cudaMemcpyHostToDevice, stream)); // Batch transfer

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++)
    AoSKernel<treeSize><<<grid, block, 0, stream>>>((AppleTree*)inputBuffer, outputBuffer);

  CUDA_CHECK(cudaGetLastError()); // Check kernel launch
  CUDA_CHECK(cudaStreamSynchronize(stream)); // Synchronize stream

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (AoS): " 
            << (time * 1e-3f) / iterations << " (us)\n";

  CUDA_CHECK(cudaMemcpyAsync(deviceResult, outputBuffer, outputSize, cudaMemcpyDeviceToHost, stream)); // Batch transfer
  CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure copy completed

  for(int i = 0; i < treeNumber; i++)
  {
    if(deviceResult[i] != reference[i])
    {
      fail = true;
      break;
    }
  }

  if (fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  // Initialize SoA data
  for (int i = 0; i < treeNumber; i++)
    for(int j = 0; j < treeSize; j++)
      data[i + j * treeNumber] = j + i * treeSize;

  CUDA_CHECK(cudaMemcpyAsync(inputBuffer, data, inputSize, cudaMemcpyHostToDevice, stream)); // Batch transfer

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++)
    SoAKernel<treeSize><<<grid, block, treeSize * sizeof(int), stream>>>((ApplesOnTrees*)inputBuffer, outputBuffer); // Shared memory size

  CUDA_CHECK(cudaGetLastError()); // Check kernel launch
  CUDA_CHECK(cudaStreamSynchronize(stream)); // Synchronize stream

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average kernel execution time (SoA): " 
            << (time * 1e-3f) / iterations << " (us)\n";

  CUDA_CHECK(cudaMemcpyAsync(deviceResult, outputBuffer, outputSize, cudaMemcpyDeviceToHost, stream)); // Batch transfer
  CUDA_CHECK(cudaStreamSynchronize(stream)); // Ensure copy completed

  for(int i = 0; i < treeNumber; i++)
  {
    if(deviceResult[i] != reference[i])
    {
      fail = true;
      break;
    }
  }

  if (fail)
    std::cout << "FAIL\n";
  else
    std::cout << "PASS\n";

  CUDA_CHECK(cudaFree(inputBuffer));
  CUDA_CHECK(cudaFree(outputBuffer));
  CUDA_CHECK(cudaFreeHost(deviceResult)); // Free pinned memory
  CUDA_CHECK(cudaFreeHost(reference)); // Free pinned memory
  CUDA_CHECK(cudaFreeHost(data)); // Free pinned memory
  CUDA_CHECK(cudaStreamDestroy(stream)); // Destroy stream
  return 0;
}
