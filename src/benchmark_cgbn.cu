#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <CGBN/cgbn.h>
#include "benchmark.cuh"

using namespace std;

#define TPI 8
#define BITS 256

// 8 threads and 256 bits integer
#if defined(__CUDA_ARCH__)
typedef cgbn_context_t<TPI>         context_t;
#else
typedef cgbn_host_context_t<TPI>         context_t;
#endif

typedef cgbn_env_t<context_t, BITS> env_t;

typedef env_t::cgbn_t bn_t;

typedef cgbn_mem_t<BITS> word_t;


__global__ void CGBNSimpleMulKernel(cgbn_error_report_t *report, word_t *words, uint32_t count) {
  int32_t tid, instance;

  tid = (blockIdx.x*blockDim.x + threadIdx.x);
  instance = tid/TPI;

  if(instance>=1) return;

  context_t      bn_context(cgbn_report_monitor, report, instance);
  env_t          bn_env(bn_context.env<env_t>());
  bn_t  a, b, result, r;

  if (tid == 0) DEBUG_PRINT("Check first word\n");
  // assert(1558243763 == words->_limbs[3]);
  // assert(1715966102 == words->_limbs[2]);
  // assert(2273913630 == words->_limbs[1]);
  // assert(2079934641 == words->_limbs[0]);

  cgbn_load(bn_env, a, words);
  cgbn_load(bn_env, b, words + 1);
  cgbn_load(bn_env, r, words + 2);
  cgbn_mul(bn_env, result, a, b);

  if (tid==0) DEBUG_PRINT("Computed %u\n", cgbn_get_ui32(bn_env, result));
  if (tid==0) DEBUG_PRINT("Expected %u\n", cgbn_get_ui32(bn_env, r));

  auto equal = cgbn_compare(bn_env, result, r);
  assert(equal == 0);
}

void BM_CGBNSimpleMul(benchmark::State& state)
{
  cgbn_error_report_t *report;

  DEBUG_PRINT("Genereating instances ...\n");

  word_t *a = (word_t *)malloc(sizeof(word_t)* 3);
  for (int i = 0; i < 3; i++) {
    for (auto j = 0; j < 8; j++){
      (a+i)->_limbs[j] = 0;
    }
    (a+i)->_limbs[3] = 1558243763;
    (a+i)->_limbs[2] = 1715966102;
    (a+i)->_limbs[1] = 2273913630;
    (a+i)->_limbs[0] = 2079934641;
  }


  (a+2)->_limbs[7] = 565341586;
  (a+2)->_limbs[6] = 3234757391;
  (a+2)->_limbs[5] = 2935691132;
  (a+2)->_limbs[4] = 300816443;
  (a+2)->_limbs[3] = 895749092;
  (a+2)->_limbs[2] = 1824205869;
  (a+2)->_limbs[1] = 2220097044;
  (a+2)->_limbs[0] = 2465598049;


  DEBUG_PRINT("Copying instances to the GPU ...\n");
  word_t *device_a;
  CUDA_CHECK(cudaMalloc((void **)&device_a, sizeof(word_t)*3));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(word_t)*3, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  DEBUG_PRINT("Running GPU kernel ...\n");

  CUDA_CHECK(cudaDeviceSynchronize());

  // CUDA events for GPU timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _: state) {
    cudaEventRecord(start);

    CGBNSimpleMulKernel<<<1, TPI>>>(report, device_a, 3);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    state.SetIterationTime(gpu_time_ms / 1000.0);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());

  DEBUG_PRINT("Copying results back to CPU ...\n");
  // clean up
  free(a);
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cgbn_error_report_free(report));
}


__global__ void CGBNLargeArrayAddKernel(cgbn_error_report_t *report, word_t *words, word_t *output, uint32_t count) {
  int32_t tid, instance;

  tid = (blockIdx.x*blockDim.x + threadIdx.x);
  instance = tid/TPI;

  if(instance>=count) return;

  context_t bn_context(cgbn_report_monitor, report, instance);
  env_t bn_env(bn_context.env<env_t>());
  bn_t a, b, r;

  if (tid == 0) DEBUG_PRINT("Check first word\n");
  // assert(1558243763 == words->_limbs[3]);

  cgbn_load(bn_env, a, words);
  cgbn_load(bn_env, b, words);
  cgbn_add(bn_env, r, a, b);

  cgbn_store(bn_env, output, r);
}

void BM_CGBNLargeArrayAddition(benchmark::State& state, int size, int threads_per_block)
{

  maybeInitDevice();

  auto num_blocks = (TPI * size) / threads_per_block + 1;

  cout << "BM_CGBNLargeArrayAddition test configuration:" << endl;
  cout << "Array size: " << size << endl;
  cout << "num_blocks: " << num_blocks << endl;
  cout << "threads_per_block: " << threads_per_block << endl;


  cgbn_error_report_t *report;

  word_t *a = (word_t *)malloc(sizeof(word_t)* size);
  for (int i = 0; i < size; i++) {
    for (auto j = 0; j < 8; j++){
      (a+i)->_limbs[j] = 0;
    }
    (a+i)->_limbs[4] = 2;
    (a+i)->_limbs[3] = 3051597590;
    (a+i)->_limbs[2] = 2978480132;
    (a+i)->_limbs[1] = 1733254032;
    (a+i)->_limbs[0] = 962890631;
  }

  DEBUG_PRINT("Copying instances to the GPU ...\n");
  word_t *device_a, *device_output;
  CUDA_CHECK(cudaMalloc((void **)&device_a, sizeof(word_t)* size));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(word_t)*size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void **)&device_output, sizeof(word_t)* size));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  DEBUG_PRINT("Running GPU kernel ...\n");

  CUDA_CHECK(cudaDeviceSynchronize());

  // CUDA events for GPU timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (auto _: state) {
    cudaEventRecord(start);

    CGBNLargeArrayAddKernel<<<num_blocks, threads_per_block>>>(report, device_a, device_output, size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    state.SetIterationTime(gpu_time_ms / 1000.0);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());

  DEBUG_PRINT("Copying results back to CPU ...\n");
  // clean up
  free(a);
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cgbn_error_report_free(report));
}
