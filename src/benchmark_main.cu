#include <benchmark/benchmark.h>
#include "benchmark.cuh"

BENCHMARK(BM_BigIntSimpleMath)->UseManualTime();

BENCHMARK(BM_CGBNSimpleMath)->UseManualTime();

BENCHMARK_MAIN();
