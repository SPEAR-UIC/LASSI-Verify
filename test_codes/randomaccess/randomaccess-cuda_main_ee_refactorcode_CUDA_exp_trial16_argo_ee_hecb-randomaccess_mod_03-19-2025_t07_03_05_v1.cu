
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

typedef unsigned long long int u64Int;
typedef long long int s64Int;
typedef unsigned int u32;

/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L

#define K1_BLOCKSIZE  256
#define K2_BLOCKSIZE  128

__forceinline__ __device__ u64Int HPCC_starts(s64Int n) {
    int i, j;
    u64Int m2[64];
    u64Int temp, ran;

    while (n < 0) n += PERIOD;
    while (n > PERIOD) n -= PERIOD;
    if (n == 0) return 0x1;

    temp = 0x1;

    #pragma unroll
    for (i=0; i<64; i++) {
        m2[i] = temp;
        temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
        temp = (temp << 1) ^ ((s64Int) temp < 0 ? POLY : 0);
    }

    for (i=62; i>=0; i--)
        if ((n >> i) & 1)
            break;

    ran = 0x2;
    while (i > 0) {
        temp = 0;
        #pragma unroll
        for (j=0; j<64; j++)
            if ((ran >> j) & 1)
                temp ^= m2[j];
        ran = temp;
        i -= 1;
        if ((n >> i) & 1)
            ran = (ran << 1) ^ ((s64Int) ran < 0 ? POLY : 0);
    }

    return ran;
}

__global__ void initTable (u64Int* Table, const u64Int TableSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < TableSize) Table[i] = i;
}

__global__ void update (u64Int*__restrict__ Table,
                        u64Int*__restrict__ ran,
                        const u64Int TableSize) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    u32 updates_per_thread = (8 * (u32)TableSize) / (gridDim.x * blockDim.x); // Use 32-bit for loop counter
    u64Int ran_val = HPCC_starts ((s64Int)(updates_per_thread) * (s64Int)j); // Initialize ran_val
    for (u32 i=0; i<updates_per_thread; i++) {
        ran_val = (ran_val << 1) ^ ((-(s64Int)(ran_val >> 63)) & POLY); // Branchless computation
        atomicXor(&Table[ran_val & (TableSize-1)], ran_val); // Coalesced access pattern
    }
    ran[j] = ran_val; // Store final ran_val back
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

    int failure;
    u64Int i;
    u64Int temp;
    double totalMem;
    u64Int *Table = NULL;
    u64Int logTableSize, TableSize;

    /* calculate local memory per node for the update table */
    totalMem = 1024*1024*512;
    totalMem /= sizeof(u64Int);

    /* calculate the size of update array (must be a power of 2) */
    for (totalMem *= 0.5, logTableSize = 0, TableSize = 1;
         totalMem >= 1.0;
         totalMem *= 0.5, logTableSize++, TableSize <<= 1)
        ; /* EMPTY */

    printf("Table size = %llu\n",  TableSize);

    /* Use pinned memory for faster and energy-efficient transfers */
    cudaHostAlloc((void**)&Table, TableSize * sizeof(u64Int), cudaHostAllocDefault); // Pinned memory

    if (! Table ) {
        fprintf(stderr, "Failed to allocate memory for the update table %llu\n", TableSize);
        return 1;
    }

    /* Print parameters for run */
    fprintf(stdout, "Main table size   = 2^%llu = %llu words\n", logTableSize,TableSize);
    fprintf(stdout, "Number of updates = %llu\n", (8 * TableSize));

    u64Int* d_Table;
    cudaMalloc((void**)&d_Table, TableSize * sizeof(u64Int));

    /* Define number of blocks to utilize GPU resources */
    int num_blocks = 256; // Example block count
    u64Int* d_ran;
    cudaMalloc((void**)&d_ran, num_blocks * K2_BLOCKSIZE * sizeof(u64Int)); // Allocate ran for all blocks

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
        /* initialize the table */
        initTable<<<(TableSize + K1_BLOCKSIZE - 1) / K1_BLOCKSIZE, K1_BLOCKSIZE>>>(d_Table, TableSize);

        /* update the table with multiple blocks to enhance concurrency */
        update<<<num_blocks, K2_BLOCKSIZE>>>(d_Table, d_ran, TableSize);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

    cudaMemcpy(Table, d_Table, TableSize * sizeof(u64Int), cudaMemcpyDeviceToHost); // Efficient transfer with pinned memory

    /* validation */
    temp = 0x1;
    for (i=0; i<(8 * TableSize); i++) { // Adjusted NUPDATE to match macro
        temp = (temp << 1) ^ (((s64Int) temp < 0) ? POLY : 0);
        Table[temp & (TableSize-1)] ^= temp;
    }

    temp = 0;
    for (i=0; i<TableSize; i++)
        if (Table[i] != i) {
            temp++;
        }

    fprintf(stdout, "Found %llu errors in %llu locations (%s).\n",
            temp, TableSize, (temp <= 0.01 * TableSize) ? "PASS" : "FAIL");
    if (temp <= 0.01 * TableSize) failure = 0;
    else failure = 1;

    cudaFreeHost(Table); // Free pinned memory
    cudaFree(d_Table);
    cudaFree(d_ran);
    return failure;
}
