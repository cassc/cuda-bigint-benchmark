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

__global__ void BigIntArrayTest_kernel(bigint* a, bigint* output, int size) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= size) return;

  // DEBUG_PRINT("Address of a %p\n", a);
  // DEBUG_PRINT("Address of a.capacity %d\n", a->capacity);
  // DEBUG_PRINT("Address of a+1 %p\n", a + 1);
  // DEBUG_PRINT("Address of (a+1).capacity %d\n", (a + 1)->capacity);
  // DEBUG_PRINT("Address of (a+2).capacity %d\n", (a + 2)->capacity);

  assert(a[idx].size > 0);

  bigint_add(output + idx, a+idx, a + idx);
  // memory allocation on device is very slow
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

void BM_BigIntLargeArrayAddition_heap(benchmark::State& state, int size, int threads_per_block)
{
  maybeInitDevice();
  auto num_blocks = size / threads_per_block + 1;

  // cout << "\nBM_BigIntLargeArrayAddition test configuration:" << endl;
  // cout << "Array size: " << size << endl;
  // cout << "Config: " << num_blocks << " X " << threads_per_block  << endl;

  const char *text = "922337203685477580881231245678901234567";
  const char *expected_text = "1844674407370955161762462491357802469134";

  bigint expected, *a, *device_a, *device_output;
  a = NULL;
  device_a = NULL;
  device_output = NULL;
  a = (bigint*) malloc(sizeof(bigint) * size);

  if (a==NULL){
    cout << "Memory allocation on host failed!" << endl;
    exit(EXIT_FAILURE);
  }

  auto capacity = 0;

  bigint_from_str_base(&expected, expected_text, 10);

  for (auto i = 0; i < size; i++) {
    bigint_init(a + i);
    bigint_from_str_base(a+i, text, 10);
    if (0 == capacity){
      capacity = a[i].capacity;
    }
  }

  CUDA_CHECK(cudaMalloc((void**)&device_a, sizeof(bigint) * size));
  CUDA_CHECK(cudaMalloc((void**)&device_output, sizeof(bigint) * size));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(bigint) * size , cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_output, a, sizeof(bigint) * size, cudaMemcpyHostToDevice));

  for (int i = 0; i < size; i++) {
    // Allocate memory for words on the device
    bigint_word *device_words, *device_output_words;
    DEBUG_PRINT("CudaTestBigIntFromStrBase: a[%d].capacity %d\n", i, a[i].capacity);
    CUDA_CHECK(cudaMalloc((void**)&device_words, (a+i)->capacity * sizeof(bigint_word)));
    CUDA_CHECK(cudaMalloc((void**)&device_output_words, (a+i)->capacity * sizeof(bigint_word)));

    // Copy words data from host to device
    CUDA_CHECK(cudaMemcpy(device_words, a[i].words, (a+i)->size * sizeof(bigint_word), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output_words, a[i].words, (a+i)->size * sizeof(bigint_word), cudaMemcpyHostToDevice));

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
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaDeviceSynchronize());

  // Free memory on the host to be reused
  for (auto i=0; i< size; i++){
    free(a[i].words);
    a[i].words = NULL;
  }

  CUDA_CHECK(cudaMemcpy(a, device_output, sizeof(bigint)*size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    bigint_word *tmp = (bigint_word*)malloc(a[i].capacity * sizeof(bigint_word));
    assert(tmp != NULL);
    assert(capacity == a[i].capacity);  // Device expanded capacity, things will break
    CUDA_CHECK(cudaMemcpy(tmp, a[i].words, a[i].size * sizeof(bigint_word), cudaMemcpyDeviceToHost));
    a[i].words = tmp;
  }

  for (int i = 0; i < size; i++) {
    assert(0 == bigint_cmp(a+i, &expected));
  }


  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_output));
  free(a);
  // todo free internal points a.words, device_a.words,
}

void BM_BigIntLargeArrayAddition(benchmark::State& state, int size, int threads_per_block)
{
  maybeInitDevice();
  auto num_blocks = size / threads_per_block + 1;

  const char *text = "15241579027587258039323273891175125743036122542295381801554580094497789971041";
  const char *expected_text = "30483158055174516078646547782350251486072245084590763603109160188995579942082";
  bigint a[size], expected;
  bigint *device_a, *device_output;
  bigint_init(&expected);
  for (int i = 0; i < size; i++) bigint_init(a + i);

  for (auto i = 0; i < size; i++){
    bigint_from_str_base(a+i, text, 10);
  }


  auto capacity = a[0].capacity;

  bigint_from_str(&expected, expected_text);

  // printf("expected 0 %u\n", *(expected.words+ 0));
  // printf("expected 1 %u\n", *(expected.words+ 1));
  // printf("expected 2 %u\n", *(expected.words+ 2));
  // printf("expected 3 %u\n", *(expected.words+ 3));
  // printf("expected 4 %u\n", *(expected.words+ 4));
  // printf("expected 5 %u\n", *(expected.words+ 5));
  // printf("expected 6 %u\n", *(expected.words+ 6));
  // printf("expected 7 %u\n", *(expected.words+ 7));

  DEBUG_PRINT("CudaTestBigIntFromStrBase: sizeof(a) %ld\n", sizeof(a));

  CUDA_CHECK(cudaMalloc(&device_a, sizeof(a)));
  CUDA_CHECK(cudaMalloc(&device_output, sizeof(a)));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(a), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_output, a, sizeof(a), cudaMemcpyHostToDevice));

  for (int i = 0; i < size; i++) {
    // Allocate memory for words on the device
    bigint_word *device_words, *device_output_words;
    DEBUG_PRINT("CudaTestBigIntFromStrBase: a[%d].capacity %d\n", i, a[i].capacity);
    CUDA_CHECK(cudaMalloc(&device_words, a[i].capacity * sizeof(bigint_word)));
    CUDA_CHECK(cudaMalloc(&device_output_words, a[i].capacity * sizeof(bigint_word)));

    // Copy words data from host to device
    CUDA_CHECK(cudaMemcpy(device_words, a[i].words, a[i].size * sizeof(bigint_word), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output_words, a[i].words, a[i].size * sizeof(bigint_word), cudaMemcpyHostToDevice));

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

  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(a, device_output, sizeof(bigint)*size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++) {
    bigint_word *tmp = (bigint_word*)malloc(a[i].capacity * sizeof(bigint_word));
    assert(tmp != NULL);
    assert(capacity == a[i].capacity);  // Device expanded capacity, things will break
    CUDA_CHECK(cudaMemcpy(tmp, a[i].words, a[i].size * sizeof(bigint_word), cudaMemcpyDeviceToHost));
    a[i].words = tmp;
  }

  // for (int i = 0; i < size; i++) {
  //   printf("a 0 %u\n", *(a[i].words+ 0));
  //   printf("a 1 %u\n", *(a[i].words+ 1));
  //   printf("a 2 %u\n", *(a[i].words+ 2));
  //   printf("a 3 %u\n", *(a[i].words+ 3));
  //   printf("a 4 %u\n", *(a[i].words+ 4));
  //   printf("a 5 %u\n", *(a[i].words+ 5));
  //   printf("a 6 %u\n", *(a[i].words+ 6));
  //   printf("a 7 %u\n", *(a[i].words+ 7));
  //   assert(0 == bigint_cmp(a+i, &expected));
  // }

  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_output));
}
