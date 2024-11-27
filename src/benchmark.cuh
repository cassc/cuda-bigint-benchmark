#ifndef CUDA_BIGINT_CGBN_BENCHMARK_H_
#define CUDA_BIGINT_CGBN_BENCHMARK_H_
#include <stdexcept>
#include <iostream>
#include <benchmark/benchmark.h>

// Define CUDA_CHECK macro
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



void BM_BigIntSimpleMul(benchmark::State& state);
void BM_BigIntLargeArrayAddition(benchmark::State& state, int size=500'000, int threads_per_block=32);

void BM_CGBNSimpleMul(benchmark::State& state);
void BM_CGBNLargeArrayAddition(benchmark::State& state, int size=500'000, int threads_per_block=32);

static bool device_initialized = false;

inline void maybeInitDevice(){
  if (device_initialized){
    return;
  }
  device_initialized = true;
  size_t heap_size = 4l * 1024 * 1024 * 1024;
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));
}

#endif
