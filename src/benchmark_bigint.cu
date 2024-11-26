#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <math.h>
#include <CuBigInt/bigint.cuh>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include "benchmark.cuh"

using namespace std;


__global__ void BigIntSimpleMulTest_kernel(bigint* a) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx != 0) return;

  bigint c[1];
  bigint_init(c);

  DEBUG_PRINT("BigIntInitTest_kernel multiplication before\n");

  DEBUG_PRINT("Address of a %p\n", a);
  DEBUG_PRINT("Address of a.capacity %d\n", a->capacity);
  DEBUG_PRINT("Address of a+1 %p\n", a + 1);
  DEBUG_PRINT("Address of (a+1).capacity %d\n", (a + 1)->capacity);
  DEBUG_PRINT("Address of (a+2).capacity %d\n", (a + 2)->capacity);

  bigint_mul(c, a, a + 1);

  DEBUG_PRINT("BigIntInitTest_kernel multiplication after\n");

  assert(0 == bigint_cmp(c, a + 2));

  bigint_free(c);
}

__global__ void BigIntArrayTest_kernel(bigint* input, bigint* output, int size) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx > size) return;

  DEBUG_PRINT("BigIntInitTest_kernel multiplication before\n");

  DEBUG_PRINT("Address of a %p\n", a);
  DEBUG_PRINT("Address of a.capacity %d\n", a->capacity);
  DEBUG_PRINT("Address of a+1 %p\n", a + 1);
  DEBUG_PRINT("Address of (a+1).capacity %d\n", (a + 1)->capacity);
  DEBUG_PRINT("Address of (a+2).capacity %d\n", (a + 2)->capacity);


  bigint_add(output + idx, input+idx, input + idx);


  DEBUG_PRINT("BigIntInitTest_kernel multiplication after\n");
}

void BM_BigIntSimpleMul(benchmark::State& state)
{
  maybeInitDevice();
  const char *text = "123456790123456790120987654320987654321";
  const char *expected = "15241579027587258039323273891175125743036122542295381801554580094497789971041";
  bigint a[3];
  bigint *device_a;
  for (int i = 0; i < 3; i++) bigint_init(a + i);

  bigint_from_str_base(a, text, 10);
  bigint_from_str_base(a + 1, text, 10);
  bigint_from_str_base(a + 2, expected, 10);

  DEBUG_PRINT("CudaTestBigIntFromStrBase: sizeof(a) %ld\n", sizeof(a));

  CUDA_CHECK(cudaMalloc(&device_a, sizeof(a)));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice));

  for (int i = 0; i < 3; i++) {
    // Allocate memory for words on the device
    bigint_word *device_words;
    DEBUG_PRINT("CudaTestBigIntFromStrBase: a[%d].capacity %d\n", i, a[i].capacity);
    CUDA_CHECK(cudaMalloc(&device_words, a[i].capacity * sizeof(bigint_word)));

    // Copy words data from host to device
    CUDA_CHECK(cudaMemcpy(device_words, a[i].words, a[i].size * sizeof(bigint_word), cudaMemcpyHostToDevice));

    // Update the device `bigint` structure's `words` pointer
    DEBUG_PRINT("Updating a[%d].words to %p\n", i, device_words);
    CUDA_CHECK(cudaMemcpy(&device_a[i].words, &device_words, sizeof(bigint_word *), cudaMemcpyHostToDevice));
  }


  CUDA_CHECK(cudaDeviceSynchronize());

  // CUDA events for GPU timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _: state) {
    CUDA_CHECK(cudaEventRecord(start));

    BigIntSimpleMulTest_kernel<<<1, 1>>>(device_a);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    state.SetIterationTime(gpu_time_ms / 1000.0);

  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaDeviceSynchronize());

  // todo how to free the two pointers in the device?
  CUDA_CHECK(cudaFree(device_a));
}

void BM_BigIntLargeArrayAddition(benchmark::State& state, int size, int threads_per_block)
{

  maybeInitDevice();
  auto num_blocks = size / threads_per_block + 1;

  cout << "BM_CGBNLargeArrayAddition test configuration:" << endl;
  cout << "Array size: " << size << endl;
  cout << "num_blocks: " << num_blocks << endl;
  cout << "threads_per_block: " << threads_per_block << endl;

  const char *text = "922337203685477580881231245678901234567";
  bigint *a, *device_a, *device_output;
  a = (bigint*) malloc(sizeof(bigint) * size);

  if (a == NULL){
    fprintf(stderr, "Memory allocation on host failed!\n");
    exit(EXIT_FAILURE);
  }

  for (auto i = 0; i < size; i++) {
    bigint_init(a + i);
    bigint_from_str_base(a+i, text, 10);
  }

  CUDA_CHECK(cudaMalloc(&device_a, sizeof(bigint) * size));
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(bigint) * size));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice));

  for (int i = 0; i < size; i++) {
    // Allocate memory for words on the device
    bigint_word *device_words, *device_output_words;
    DEBUG_PRINT("CudaTestBigIntFromStrBase: a[%d].capacity %d\n", i, a[i].capacity);
    CUDA_CHECK(cudaMalloc(&device_words, (a+i)->capacity * sizeof(bigint_word)));
    CUDA_CHECK(cudaMalloc(&device_output_words, (a+i)->capacity * sizeof(bigint_word)));

    // Copy words data from host to device
    CUDA_CHECK(cudaMemcpy(device_words, a[i].words, (a+i)->size * sizeof(bigint_word), cudaMemcpyHostToDevice));

    // Update the device `bigint` structure's `words` pointer
    DEBUG_PRINT("Updating a[%d].words to %p\n", i, device_words);
    CUDA_CHECK(cudaMemcpy(&device_a[i].words, &device_words, sizeof(bigint_word *), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(&device_output[i].words, &device_output_words, sizeof(bigint_word *), cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  // CUDA events for GPU timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (auto _: state) {
    CUDA_CHECK(cudaEventRecord(start));

    BigIntArrayTest_kernel<<<num_blocks, threads_per_block>>>(device_a, device_output, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
    state.SetIterationTime(gpu_time_ms / 1000.0);

    // TODO copy back and check results
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(device_a));

  free(a);
}
