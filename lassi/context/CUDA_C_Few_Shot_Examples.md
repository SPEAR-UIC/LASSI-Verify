[BEGIN OPTIMIZATION EXAMPLES] Below are several short CUDA C++ "code pairs," each illustrating a common optimization strategy to improve energy efficiency (and performance) of GPU kernels. For each pair, there is a non-optimized version and an optimized version. The optimized versions demonstrate best practices such as leveraging coalesced memory accesses, shared memory usage, minimizing divergent branches, and reducing unnecessary global memory traffic—all of which can significantly reduce energy consumption on GPU systems.

# 1. Coalesced vs. Non-Coalesced Memory Access

Goal: Improve memory throughput by having threads within a warp access consecutive global memory locations (coalesced access), reducing overhead and energy usage.

## Non-Optimized (Non-Coalesced):

```
#include <stdio.h>

__global__ void nonCoalescedKernel(const float* d_in, float* d_out, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Grid-stride loop to process elements
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        // Access non-consecutive indices across threads
        // e.g., tid 0 reads d_in[0], tid 1 reads d_in[100], etc.
        // Just an example of strided access that leads to non-coalescing
        int idx = i * 100 % size;  // artificial non-coalesced pattern
        d_out[i] = d_in[idx] * 2.0f;
    }
}

int main() {
    const int size = 1 << 20; // 1M floats
    const int bytes = size * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    // Initialize input
    for (int i = 0; i < size; i++) {
        h_in[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    nonCoalescedKernel<<<gridSize, blockSize>>>(d_in, d_out, size);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

## Optimized (Coalesced):

```
#include <stdio.h>

__global__ void coalescedKernel(const float* d_in, float* d_out, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Grid-stride loop to process elements
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        // Here each thread reads a consecutive element: d_in[i]
        // Thus, accesses within a warp will be coalesced
        d_out[i] = d_in[i] * 2.0f;
    }
}

int main() {
    const int size = 1 << 20; // 1M floats
    const int bytes = size * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    coalescedKernel<<<gridSize, blockSize>>>(d_in, d_out, size);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

Key Optimization Point: By making each thread read/write consecutive indices, memory accesses become coalesced, improving throughput and reducing wasted energy.

# 2. Using Shared Memory to Reduce Global Memory Traffic

Goal: Cache repeatedly used data in shared memory to reduce expensive global memory accesses. This can greatly lower energy usage when the same data is repeatedly accessed.

## Non-Optimized (Direct Global Memory Accesses):

```
#include <stdio.h>

// Simple kernel that uses global memory repeatedly
__global__ void globalMemoryKernel(const float* d_in, float* d_out, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        // Each thread performs multiple global reads on the same index
        float val = d_in[tid];
        for (int i = 0; i < 10; i++) {
            val += d_in[tid] * 0.1f; // repeated global reads
        }
        d_out[tid] = val;
    }
}

int main() {
    const int size = 1 << 20; 
    const int bytes = size * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        h_in[i] = static_cast<float>(i % 100);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    globalMemoryKernel<<<gridSize, blockSize>>>(d_in, d_out, size);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

## Optimized (Using Shared Memory):

```
#include <stdio.h>

// Kernel that reduces repeated global reads by caching data in shared memory
__global__ void sharedMemoryKernel(const float* d_in, float* d_out, int size) {
    // Use a shared memory buffer sized to blockDim.x
    extern __shared__ float tile[];
    
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx  = threadIdx.x;

    if (globalIdx < size) {
        // Load the value from global memory into shared memory
        tile[localIdx] = d_in[globalIdx];
    }
    
    // Synchronize to ensure all threads have loaded their data
    __syncthreads();

    if (globalIdx < size) {
        // Now use the shared memory copy instead of repeated global reads
        float val = tile[localIdx];
        for (int i = 0; i < 10; i++) {
            val += tile[localIdx] * 0.1f;
        }
        d_out[globalIdx] = val;
    }
}

int main() {
    const int size = 1 << 20; 
    const int bytes = size * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        h_in[i] = static_cast<float>(i % 100);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;

    // Specify dynamically allocated shared memory using the third parameter
    sharedMemoryKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, size);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

Key Optimization Point: By loading a value from global memory just once into shared memory, we minimize repeated global accesses. This can reduce both latency and energy consumption.

# 3. Minimizing Warp Divergence

Goal: Avoid conditional branches that cause threads within the same warp to follow different execution paths. Divergent branches force some threads to idle while others execute, wasting cycles and energy.

## Non-Optimized (Warp Divergence):

```
#include <stdio.h>

__global__ void divergentKernel(const float* d_in, float* d_out, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Highly divergent branch
        // Example: half the threads do one thing, the other half do another
        if (tid % 2 == 0) {
            d_out[tid] = d_in[tid] * 2.0f;
        } else {
            d_out[tid] = d_in[tid] + 100.0f;
        }
    }
}

int main() {
    const int size = 1 << 20;
    const int bytes = size * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for(int i = 0; i < size; i++) {
        h_in[i] = (float)i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    divergentKernel<<<gridSize, blockSize>>>(d_in, d_out, size);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

## Optimized (Reduced Divergence by Using Arithmetic Masking):

```
#include <stdio.h>

__global__ void lessDivergentKernel(const float* d_in, float* d_out, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Compute both potential results and select via arithmetic
        // This avoids branching: warp threads do the same operations
        float evenVal = d_in[tid] * 2.0f;
        float oddVal  = d_in[tid] + 100.0f;

        // Condition is converted into a mask: 1 for even, 0 for odd
        // or vice versa. Then we combine results in one expression.
        int isEven = ((tid & 1) == 0); // 1 if even, 0 if odd
        d_out[tid] = evenVal * isEven + oddVal * (1 - isEven);
    }
}

int main() {
    const int size = 1 << 20;
    const int bytes = size * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for(int i = 0; i < size; i++) {
        h_in[i] = (float)i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    lessDivergentKernel<<<gridSize, blockSize>>>(d_in, d_out, size);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);

    return 0;
}
```

Key Optimization Point: Converting branching code to a uniform instruction sequence (arithmetic selection) helps keep all threads in a warp doing similar work, mitigating branch divergence and reducing wasted cycles.

# 4. Reducing Unnecessary Host-Device Transfers

Goal: Move data between host and device only when absolutely necessary. Extra transfers not only add overhead time but also consume additional energy.

## Non-Optimized (Multiple, Unnecessary Transfers):

```
#include <stdio.h>

__global__ void simpleKernel(float* d_data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // Some trivial operation
        d_data[tid] += 1.0f;
    }
}

int main() {
    const int size = 1 << 20;
    const int bytes = size * sizeof(float);

    // Allocate and initialize host data
    float* h_data = (float*)malloc(bytes);
    for(int i = 0; i < size; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float* d_data;
    cudaMalloc((void**)&d_data, bytes);

    // (1) Copy data to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    simpleKernel<<<gridSize, blockSize>>>(d_data, size);
    
    // (2) Copy data back to host, even though not needed yet
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    
    // Another kernel launch with the same data
    simpleKernel<<<gridSize, blockSize>>>(d_data, size);

    // (3) Again copy data to device unnecessarily
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // Final copy to host for result
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    free(h_data);

    return 0;
}
```

## Optimized (Minimizing Transfers):

```
#include <stdio.h>

__global__ void simpleKernel(float* d_data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_data[tid] += 1.0f;
    }
}

int main() {
    const int size = 1 << 20;
    const int bytes = size * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    for(int i = 0; i < size; i++) {
        h_data[i] = (float)i;
    }

    float* d_data;
    cudaMalloc((void**)&d_data, bytes);

    // Copy data to device once
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // First kernel
    simpleKernel<<<gridSize, blockSize>>>(d_data, size);

    // Directly launch second kernel without copying data back
    simpleKernel<<<gridSize, blockSize>>>(d_data, size);

    // Final copy to host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyHostToDevice);

    cudaFree(d_data);
    free(h_data);

    return 0;
}

```

Key Optimization Point: Consolidate or eliminate unnecessary host-device copies to reduce data transfer overhead and power consumption.

# Summary of Best Practices

1. Coalesced Memory Access: Align and organize data so that consecutive threads access consecutive memory addresses.
2. Shared Memory: Cache frequently accessed data in shared memory to reduce costly global memory transactions.
3. Minimize Warp Divergence: Use conditional operations carefully; consider arithmetic masking instead of branching when feasible.
4 .Reduce Host-Device Transfers: Perform copies only when necessary, and reuse data on the device to avoid extra transfers.
5. Tune Occupancy: Choose appropriate block and grid dimensions to keep the GPU fully utilized while avoiding excessive registers or shared memory usage per block.
6. Efficient Synchronization: Minimize __syncthreads() calls and ensure they are used only when absolutely necessary.

By systematically applying these strategies, you not only improve performance but also reduce the total energy consumption of your GPU-accelerated applications.

[/END OPTIMIZATION EXAMPLES]