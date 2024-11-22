#include <benchmark/benchmark.h>
#include "benchmark.cuh"

BENCHMARK(BM_BigIntSimpleMath)->UseManualTime();

BENCHMARK_MAIN();
