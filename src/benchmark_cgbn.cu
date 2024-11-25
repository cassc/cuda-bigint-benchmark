#include <CGBN/cgbn.h>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <CGBN/cgbn.h>

#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

// Define CUDA_CHECK_THROW macro
#define CUDA_CHECK(call) do {					\
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
      std::cerr << "CUDA call failed: " << #call << "\n"                \
                << "Error code: " << err << "\n"                        \
                << "Error string: " << cudaGetErrorString(err)          \
                << " (at " << __FILE__ << ":" << __LINE__ << ")"        \
                << std::endl;                                           \
      throw std::runtime_error(cudaGetErrorString(err));                \
    }                                                                   \
  } while (0)


#ifdef ENABLE_DEBUG
#define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while (0)
#endif

#define TPI 8
#define BITS 256


// 8 threads and 256 bits integer
#if defined(__CUDA_ARCH__)
typedef cgbn_context_t<TPI>         context_t;
#else
typedef cgbn_host_context_t<TPI>         context_t;
#endif
typedef cgbn_env_t<context_t, BITS> env_t;

typedef cgbn_mem_t<BITS> word_t;

// define the kernel
__global__ void CGBNSimpleMathKernel(cgbn_error_report_t *report, word_t *words, uint32_t count) {
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx != 0) return;

  context_t      bn_context(cgbn_report_monitor, report, idx);
  env_t          bn_env(bn_context.env<env_t>());
  env_t::cgbn_t  a, b, result, r;

  printf("First word\n");
  for (auto i = 0; i < 8; i++) {
    printf("%u\n", words->_limbs[i]);
  }

  cgbn_load(bn_env, a, words);
  cgbn_load(bn_env, b, words + 1);
  cgbn_load(bn_env, r, words + 2);
  cgbn_mul(bn_env, result, a, b);


  printf("a %u\n", cgbn_get_ui32(bn_env, a)); // todo this value is incorrect

  cgbn_set_ui32(bn_env, result, 0);
  auto equal = cgbn_compare(bn_env, result, a);
  assert(equal != 0);


  auto match = cgbn_compare(bn_env, result, r);
  assert(match == 0);
}

void BM_CGBNSimpleMath(benchmark::State& state)
{
  cgbn_error_report_t *report;

  DEBUG_PRINT("Genereating instances ...\n");

  word_t *a = (word_t *)malloc(sizeof(word_t)* 3);
  for (int i = 0; i < 2; i++) {
    for (auto j = 0; j < 8; j++){
      (a+i)->_limbs[j] = 0;
    }
    (a+i)->_limbs[4] = 1558243763;
    (a+i)->_limbs[5] = 1715966102;
    (a+i)->_limbs[6] = 2273913630;
    (a+i)->_limbs[7] = 2079934641;
  }


  (a+2)->_limbs[0] = 565341586;
  (a+2)->_limbs[1] = 3234757391;
  (a+2)->_limbs[2] = 2935691132;
  (a+2)->_limbs[3] = 300816443;
  (a+2)->_limbs[4] = 895749092;
  (a+2)->_limbs[5] = 1824205869;
  (a+2)->_limbs[6] = 2220097044;
  (a+2)->_limbs[7] = 2465598049;


  printf("Copying instances to the GPU ...\n");
  word_t *device_a;
  CUDA_CHECK(cudaMalloc((void **)&device_a, sizeof(word_t)*3));
  CUDA_CHECK(cudaMemcpy(device_a, a, sizeof(word_t)*3, cudaMemcpyHostToDevice));

  for (auto i =0; i < 3; i++){
    CUDA_CHECK(cudaMemcpy((device_a+i)->_limbs, (a+i)->_limbs, sizeof(uint32_t)*8, cudaMemcpyHostToDevice));
  }

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("Running GPU kernel ...\n");

  CUDA_CHECK(cudaDeviceSynchronize());

  CGBNSimpleMathKernel<<<1, TPI>>>(report, device_a, 3);

  // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
  CUDA_CHECK(cudaDeviceSynchronize());
  // CGBN_CHECK(report);

  printf("Copying results back to CPU ...\n");
  // clean up
  free(a);
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cgbn_error_report_free(report));
}
