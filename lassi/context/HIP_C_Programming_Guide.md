# HIP C++ Programming Best Practices Guide

This guide introduces the main concepts behind the HIP programming model and provides best practices for efficient GPU programming on AMD hardware. HIP (Heterogeneous-computing Interface for Portability) extends C++ by allowing programmers to create portable applications for AMD and NVIDIA GPUs from a single source code.

## 1. Kernels

HIP C++ extends C++ by allowing programmers to define C++ functions, called kernels, that execute in parallel across multiple HIP threads on AMD GPU devices. A kernel is defined using the `__global__` declaration specifier and is launched using either the triple chevron syntax `<<<...>>>` or the `hipLaunchKernelGGL()` function.

Each thread executing the kernel has unique identifiers accessible through built-in variables. Here's a simple vector addition example:

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main() {
    // ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    // ...
}
```

Each of the N threads executing `VecAdd()` performs one pair-wise addition operation.

**Best Practices:**
- Use meaningful kernel names that describe the operation
- Keep kernels focused on a single computational task
- Minimize parameter passing - use pointers for large data structures
- Ensure kernels have `void` return types (required by HIP)

## 2. Thread Hierarchy

HIP organizes threads in a three-level hierarchy that maps efficiently to AMD GPU hardware:

### 2.1 Thread Organization

- **Threads**: Individual instances of kernel execution, identified by `threadIdx` (1D, 2D, or 3D)
- **Blocks**: Groups of threads that can cooperate through shared memory, identified by `blockIdx`
- **Grids**: Collections of thread blocks for a kernel launch, with dimensions specified in kernel launch parameters

The relationship between thread indices follows this pattern:
- 1D block: thread ID = `threadIdx.x`
- 2D block: thread ID = `threadIdx.x + threadIdx.y * blockDim.x`
- 3D block: thread ID = `threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y`

### 2.2 Matrix Addition Example

```cpp
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
    // ...
    // Kernel invocation with one block of N * N threads
    int numBlocks = 1;
    dim3 threadsPerBlock(N, N);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    // ...
}
```

### 2.3 Multi-Block Processing

For larger datasets, use multiple blocks:

```cpp
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

int main() {
    // ...
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    // ...
}
```

**Best Practices:**
- Use 2D/3D thread organizations for 2D/3D data structures
- Choose block sizes that are multiples of wavefront size (64 for AMD GPUs)
- Common block sizes: 64, 128, 256, 512 threads
- Always include bounds checking when using multiple blocks
- Design for thread independence - blocks must execute independently

### 2.4 Wavefront Considerations

AMD GPUs execute threads in groups called wavefronts (typically 64 threads). Understanding this is crucial for performance:

- **Wavefront Size**: Usually 64 threads on AMD hardware (vs. 32 for NVIDIA)
- **SIMT Execution**: All threads in a wavefront execute the same instruction
- **Divergence**: Conditional branches can cause performance penalties within wavefronts

## 3. Memory Hierarchy

HIP threads access data from multiple memory spaces with different performance characteristics:

### 3.1 Memory Types

- **Local Memory**: Private to each thread, fastest access
- **Shared Memory**: Visible to all threads in a block, fast access with potential bank conflicts
- **Global Memory**: Accessible by all threads, large capacity but high latency
- **Constant Memory**: Read-only global memory with caching optimization
- **Texture Memory**: Read-only with spatial locality caching

### 3.2 Memory Optimization Strategies

**Coalesced Global Memory Access:**
```cpp
// Good: coalesced access pattern
__global__ void coalescedAccess(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid] = tid * 2.0f;  // Each thread accesses consecutive elements
}

// Bad: strided access pattern
__global__ void stridedAccess(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    data[tid * STRIDE] = tid * 2.0f;  // Non-consecutive access
}
```

**Shared Memory Usage:**
```cpp
__global__ void sharedMemoryExample(float* input, float* output) {
    // Allocate shared memory
    __shared__ float shared_data[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    shared_data[tid] = input[gid];
    
    // Synchronize threads in block
    __syncthreads();
    
    // Perform computation using shared data
    float result = 0.0f;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        result += shared_data[i];
    }
    
    output[gid] = result;
}
```

**Best Practices:**
- Minimize global memory transactions through coalescing
- Use shared memory for data reuse within thread blocks
- Avoid bank conflicts in shared memory (pad arrays if necessary)
- Use constant memory for small, frequently accessed read-only data
- Consider texture memory for 2D spatial locality patterns

## 4. Heterogeneous Programming

HIP applications consist of host (CPU) and device (GPU) code that work together:

### 4.1 Host-Device Interaction

```cpp
int main() {
    // 1. Initialize and select GPU device
    int device = 0;
    hipSetDevice(device);
    
    // 2. Allocate host memory
    float *h_A, *h_B, *h_C;
    size_t size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // 3. Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    // 4. Copy data from host to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);
    
    // 5. Launch kernel
    dim3 threadsPerBlock(256);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
    
    // 6. Copy result back to host
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    // 7. Synchronize and check for errors
    hipDeviceSynchronize();
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("HIP error: %s\n", hipGetErrorString(err));
    }
    
    // 8. Cleanup
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
```

### 4.2 Unified Memory (Managed Memory)

HIP supports managed memory that can be accessed from both CPU and GPU:

```cpp
// Allocate managed memory
float *managed_data;
hipMallocManaged(&managed_data, size);

// Use directly in kernel (no explicit copying needed)
VecAdd<<<numBlocks, threadsPerBlock>>>(managed_data, managed_data + N, managed_data + 2*N);

// Prefetch for better performance
hipMemPrefetchAsync(managed_data, size, device, 0);
```

**Best Practices:**
- Always check for HIP errors after API calls
- Use asynchronous memory transfers when possible
- Consider using managed memory for simplified memory management
- Implement proper resource cleanup
- Use streams for overlapping computation and data transfer

## 5. Asynchronous Programming and Streams

HIP provides asynchronous execution capabilities for better performance:

### 5.1 Stream Management

```cpp
// Create streams
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

// Asynchronous operations
hipMemcpyAsync(d_A, h_A, size, hipMemcpyHostToDevice, stream1);
hipMemcpyAsync(d_B, h_B, size, hipMemcpyHostToDevice, stream2);

// Launch kernels on different streams
VecAdd<<<numBlocks, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C1);
VecAdd<<<numBlocks, threadsPerBlock, 0, stream2>>>(d_A2, d_B2, d_C2);

// Synchronize streams
hipStreamSynchronize(stream1);
hipStreamSynchronize(stream2);

// Cleanup
hipStreamDestroy(stream1);
hipStreamDestroy(stream2);
```

### 5.2 Events for Synchronization

```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

// Record events
hipEventRecord(start, 0);
VecAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
hipEventRecord(stop, 0);

// Wait and measure time
hipEventSynchronize(stop);
float milliseconds = 0;
hipEventElapsedTime(&milliseconds, start, stop);

printf("Kernel execution time: %f ms\n", milliseconds);

hipEventDestroy(start);
hipEventDestroy(stop);
```

## 6. Device Properties and Compute Capability

Query device capabilities to optimize your kernels:

```cpp
void queryDeviceProperties() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        hipDeviceProp_t deviceProp;
        hipGetDeviceProperties(&deviceProp, dev);
        
        printf("Device %d: %s\n", dev, deviceProp.name);
        printf("  Compute capability: %d.%d\n", 
               deviceProp.major, deviceProp.minor);
        printf("  Max threads per block: %d\n", 
               deviceProp.maxThreadsPerBlock);
        printf("  Max block dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1], 
               deviceProp.maxThreadsDim[2]);
        printf("  Max grid dimensions: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Warp size: %d\n", deviceProp.warpSize);
        printf("  Shared memory per block: %zu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total global memory: %zu bytes\n",
               deviceProp.totalGlobalMem);
    }
}
```

## 7. Performance Optimization Guidelines

### 7.1 Occupancy Optimization

```cpp
// Use launch bounds to control resource usage
__global__ 
__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_CU)
void optimizedKernel(float* data) {
    // Kernel implementation
}
```

### 7.2 Memory Access Patterns

```cpp
// Prefer coalesced access patterns
__global__ void coalescedKernel(float* input, float* output, int width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid / width;
    int col = tid % width;
    
    // Sequential access pattern
    output[tid] = input[tid] * 2.0f;
}
```

### 7.3 Avoiding Divergent Branching

```cpp
// Bad: causes wavefront divergence
__global__ void divergentKernel(float* data) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        data[tid] *= 2.0f;    // Even threads
    } else {
        data[tid] += 1.0f;    // Odd threads
    }
}

// Better: minimize divergence
__global__ void lessdivergentKernel(float* data) {
    int tid = threadIdx.x;
    // Use arithmetic instead of branching when possible
    float factor = 2.0f - (tid % 2);  // 2.0 for even, 1.0 for odd
    data[tid] = data[tid] * factor + (tid % 2);
}
```

## 8. Multi-GPU Programming

HIP supports multi-GPU systems for increased computational power:

```cpp
void multiGPUExample() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    if (deviceCount > 1) {
        // Use multiple devices
        for (int i = 0; i < deviceCount; i++) {
            hipSetDevice(i);
            
            // Allocate memory and launch kernels on each device
            float *d_data;
            hipMalloc(&d_data, size);
            
            // Launch kernel on current device
            kernel<<<blocks, threads>>>(d_data);
            
            hipFree(d_data);
        }
    }
}
```

## 9. Error Handling Best Practices

Always implement comprehensive error checking:

```cpp
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            printf("HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                   hipGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Usage
HIP_CHECK(hipMalloc(&d_data, size));
HIP_CHECK(hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice));
kernel<<<blocks, threads>>>(d_data);
HIP_CHECK(hipGetLastError());  // Check kernel launch
HIP_CHECK(hipDeviceSynchronize());  // Check kernel execution
```

## 10. Debugging and Profiling

### 10.1 Debugging Tips

- Use `printf()` in kernels for debugging (limited buffer size)
- Utilize `hip-gdb` for device-side debugging
- Check return values of all HIP API calls
- Use `hipDeviceSynchronize()` to catch kernel errors immediately

### 10.2 Profiling Tools

- **ROCProfiler**: AMD's profiling tool for HIP applications
- **rocprof**: Command-line profiler for performance analysis
- **ROC-debugger (ROCgdb)**: GPU debugger for HIP applications

## Summary

This guide covered the essential concepts and best practices for HIP C++ programming:

1. **Kernels**: Use `__global__` functions with proper parameter design
2. **Thread Hierarchy**: Organize threads, blocks, and grids efficiently
3. **Memory Management**: Optimize for coalesced access and proper memory usage
4. **Heterogeneous Programming**: Coordinate CPU and GPU execution
5. **Asynchronous Operations**: Leverage streams and events for performance
6. **Device Properties**: Query and adapt to hardware capabilities
7. **Performance**: Focus on occupancy, memory patterns, and minimizing divergence
8. **Multi-GPU**: Scale applications across multiple devices
9. **Error Handling**: Implement robust error checking
10. **Debugging**: Use appropriate tools and techniques

Following these practices will help you develop efficient, portable, and maintainable HIP applications for AMD GPU hardware.