
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>

template <typename T>
void reference(
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    int batch_size,
    const int* offset) {
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    const int range = offset[batch_idx + 1] - offset[batch_idx];
    for (int idx = 0; idx < embedding_dim; idx++) {
      const T dense_elem = dense[batch_idx * embedding_dim + idx];
      for (int nested_idx = idx; nested_idx < range; nested_idx += embedding_dim) {
        output[offset[batch_idx] + nested_idx] =
          input[offset[batch_idx] + nested_idx] + dense_elem;
      }
    }
  }
}

template <typename T>
__global__ void dense_esuhm(
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    const int* offset) {
  extern __shared__ T s_dense[];
  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int range = offset[batch_idx + 1] - offset[batch_idx];
  const int start = offset[batch_idx];

  // Load dense data into shared memory
  for (int i = tid; i < embedding_dim; i += blockDim.x) {
    s_dense[i] = __ldg(&dense[batch_idx * embedding_dim + i]); // Use __ldg for read-only access
  }
  __syncthreads();

  // Process each element in the range with coalesced memory access
  for (int idx = tid; idx < range; idx += blockDim.x) {
    output[start + idx] = input[start + idx] + s_dense[idx % embedding_dim];
  }
}

template <typename T>
__global__ void dense_esuhm2(
    const T* input,
    const T* dense,
          T* output,
    int embedding_dim,
    const int* offset) {
  extern __shared__ T s_dense[];
  const int tid = threadIdx.x;
  const int N = offset[gridDim.x]; // Assuming offset[batch_size] is the total number of elements

  for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
    int low = 0, high = gridDim.x - 1;
    while (low <= high) { // Binary search to find batch index
      int mid = (low + high) / 2;
      if (__ldg(&offset[mid]) > i) {
        high = mid - 1;
      } else if (__ldg(&offset[mid + 1]) <= i) {
        low = mid + 1;
      } else {
        const int batch_idx = mid;
        const int start = offset[batch_idx];
        const int idx = i - start;
        output[i] = input[i] + s_dense[idx % embedding_dim];
        break;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of rows> <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int batch_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  assert(nrows > batch_size * batch_size);

  printf("Number of rows in the embedding table: %d\n", nrows);
  printf("Batch size: %d\n", batch_size);

  const int embed_dims[] = {768, 2048, 12288};

  for (size_t n = 0; n < sizeof(embed_dims)/sizeof(int); n++) {
    int ncols = embed_dims[n];
    printf("\nEmbedding dimension: %d\n", ncols);

    int input_size = nrows * ncols;  // same as output size
    size_t input_size_bytes = input_size * sizeof(float);

    int dense_size = batch_size * ncols ;
    int dense_size_bytes = dense_size * sizeof(float);

    int batch_size_bytes = (batch_size + 1) * sizeof(int); // Corrected to int
    float *input, *dense, *output_k1, *output_k2, *output_ref;
    input = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    dense = (float*) malloc (dense_size_bytes); // [batch_size x embedding_dim]
    output_k1 = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    output_k2 = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    output_ref = (float*) malloc (input_size_bytes); // [sum(*) x embedding_dim]
    int *input_offset = (int*) malloc (batch_size_bytes);  // [batch_size]

    srand(123);
    input_offset[0] = 0;
    for (int i = 1; i <= batch_size; i++)
      input_offset[i] = input_offset[i-1] + (rand() % batch_size + 1) * ncols;

    std::default_random_engine g (123);
    std::uniform_real_distribution<float> distr (-1.f, 1.f);
    for (int i = 0; i < dense_size; i++) {
      dense[i] = distr(g);
    }

    for (int i = 0; i < input_size; i++) {
      input[i] = distr(g);
      output_ref[i] = 0;
    }

    reference(input, dense, output_ref, ncols, batch_size, input_offset);

    float *d_input, *d_dense, *d_output;
    cudaMalloc((void**)&d_input, input_size_bytes);
    cudaMemcpy(d_input, input, input_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_dense, dense_size_bytes);
    cudaMemcpy(d_dense, dense, dense_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_output, input_size_bytes);

    int* d_input_offset;
    cudaMalloc((void**)&d_input_offset, batch_size_bytes);
    cudaMemcpy(d_input_offset, input_offset, batch_size_bytes, cudaMemcpyHostToDevice);

    for (int block_size = 128; block_size <= 1024; block_size = block_size * 2) {
      printf("block size: %d\n", block_size);

      cudaMemset(d_output, 0, input_size_bytes);
      cudaDeviceSynchronize();
      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++)
        dense_esuhm<float><<<batch_size, block_size, ncols * sizeof(float)>>>(d_input, d_dense, d_output, ncols, d_input_offset);
      cudaDeviceSynchronize();
      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of dense embedding kernel (k1): %f (us)\n", (time * 1e-3f) / repeat);
      cudaMemcpy(output_k1, d_output, input_size_bytes, cudaMemcpyDeviceToHost);

      cudaMemset(d_output, 0, input_size_bytes);
      cudaDeviceSynchronize();
      start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++)
        dense_esuhm2<float><<<batch_size, block_size, ncols * sizeof(float)>>>(d_input, d_dense, d_output, ncols, d_input_offset);
      cudaDeviceSynchronize();
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of dense embedding kernel (k2): %f (us)\n", (time * 1e-3f) / repeat);
      cudaMemcpy(output_k2, d_output, input_size_bytes, cudaMemcpyDeviceToHost);

      bool ok = true;
      for (int i = 0; i < input_size; i++) {
        if (fabsf(output_k1[i] - output_ref[i]) > 1e-3f ||
            fabsf(output_k2[i] - output_ref[i]) > 1e-3f) {
          ok = false;
          break;
        }
      }
      printf("%s\n", ok ? "PASS" : "FAIL");
    }

    cudaFree(d_input);
    cudaFree(d_dense);
    cudaFree(d_output);
    cudaFree(d_input_offset);

    free(input);
    free(dense);
    free(output_k1);
    free(output_k2);
    free(output_ref);
    free(input_offset);
  }

  return 0;
}
