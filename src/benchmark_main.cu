#include <benchmark/benchmark.h>
#include "benchmark.cuh"


void BM_CGBNLargeArrayAddition_8(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 8);
}

void BM_CGBNLargeArrayAddition_32(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 32);
}

void BM_CGBNLargeArrayAddition_64(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 64);
}

void BM_CGBNLargeArrayAddition_128(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 128);
}

void BM_CGBNLargeArrayAddition_256(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 256);
}

void BM_CGBNLargeArrayAddition_512(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 512);
}

void BM_CGBNLargeArrayAddition_1024(benchmark::State& state){
  BM_CGBNLargeArrayAddition(state, 5000, 1024);
}

void BM_BigIntLargeArrayAddition_8(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 8);
}

void BM_BigIntLargeArrayAddition_32(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 32);
}

void BM_BigIntLargeArrayAddition_64(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 64);
}

void BM_BigIntLargeArrayAddition_128(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 128);
}

void BM_BigIntLargeArrayAddition_256(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 256);
}

void BM_BigIntLargeArrayAddition_512(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 512);
}

void BM_BigIntLargeArrayAddition_1024(benchmark::State& state){
  BM_BigIntLargeArrayAddition(state, 5000, 1024);
}


BENCHMARK(BM_BigIntSimpleMul)->UseManualTime();
BENCHMARK(BM_CGBNSimpleMul)->UseManualTime();



BENCHMARK(BM_BigIntLargeArrayAddition_8)->UseManualTime();
BENCHMARK(BM_BigIntLargeArrayAddition_32)->UseManualTime();
BENCHMARK(BM_BigIntLargeArrayAddition_64)->UseManualTime();
BENCHMARK(BM_BigIntLargeArrayAddition_128)->UseManualTime();
BENCHMARK(BM_BigIntLargeArrayAddition_256)->UseManualTime();
BENCHMARK(BM_BigIntLargeArrayAddition_512)->UseManualTime();
BENCHMARK(BM_BigIntLargeArrayAddition_1024)->UseManualTime();

BENCHMARK(BM_CGBNLargeArrayAddition_8)->UseManualTime();
BENCHMARK(BM_CGBNLargeArrayAddition_32)->UseManualTime();
BENCHMARK(BM_CGBNLargeArrayAddition_64)->UseManualTime();
BENCHMARK(BM_CGBNLargeArrayAddition_128)->UseManualTime();
BENCHMARK(BM_CGBNLargeArrayAddition_256)->UseManualTime();
BENCHMARK(BM_CGBNLargeArrayAddition_512)->UseManualTime();
BENCHMARK(BM_CGBNLargeArrayAddition_1024)->UseManualTime();


BENCHMARK_MAIN();
