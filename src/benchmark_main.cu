#include <benchmark/benchmark.h>
#include "benchmark.cuh"


BENCHMARK(BM_BigIntSimpleMul)->UseManualTime();

BENCHMARK(BM_BigIntLargeArrayAddition)->UseManualTime();

BENCHMARK(BM_CGBNSimpleMul)->UseManualTime();

BENCHMARK_MAIN();
