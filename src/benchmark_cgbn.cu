#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdexcept>
#include <iostream>
#include <CGBN/cgbn.h>


// Define CUDA_CHECK_THROW macro
#define CUDA_CHECK_THROW(call) do {					\
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

// define a struct to hold each problem instance
typedef struct {
  cgbn_mem_t<BITS> a;
  cgbn_mem_t<BITS> b;
  cgbn_mem_t<BITS> r;
} instance_t;


// 8 threads and 256 bits integer
typedef cgbn_context_t<TPI>         context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

// define the kernel
__global__ void add_kernel(cgbn_error_report_t *report, instance_t *instances, uint32_t count) {
  int32_t instance;

  instance=(blockIdx.x*blockDim.x + threadIdx.x)/TPI;
  if(instance>=count){
    return;
  }

  context_t      bn_context(cgbn_report_monitor, report, instance);   // construct a context
  env_t          bn_env(bn_context.env<env_t>());                     // construct an environment for 1024-bit math
  env_t::cgbn_t  a, b, r;


  cgbn_load(bn_env, a, &(instances[instance].a));
  cgbn_load(bn_env, b, &(instances[instance].b));
  cgbn_add(bn_env, r, a, b);
  cgbn_store(bn_env, &(instances[instance].r), r);
}
