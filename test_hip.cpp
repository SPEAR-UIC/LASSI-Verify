#include <hip/hip_runtime.h>
#include <iostream>

__global__ void hello_kernel() {
    int tid = threadIdx.x;
    printf("Hello from GPU thread %d\n", tid);
}

int main() {
    std::cout << "Launching HIP kernel..." << std::endl;
    hello_kernel<<<1, 4>>>();
    
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "HIP test completed successfully!" << std::endl;
    return 0;
}
