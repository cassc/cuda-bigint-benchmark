#ifndef CUDA_BIGINT_CGBN_BENCHMARK_H_
#define CUDA_BIGINT_CGBN_BENCHMARK_H_

#include <benchmark/benchmark.h>

void BM_BigIntSimpleMath(benchmark::State& state);
void BM_CGBNSimpleMath(benchmark::State& state);

#endif
