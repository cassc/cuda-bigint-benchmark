# (WIP) Benchmark comparison between CGBN and Bigint

## Requirements

[google-benchmark](https://github.com/google/benchmark). On Arch Linux you can install using `sudo pacman -S benchmark`.



## Build

``` bash
git clone https://github.com/cassc/cuda-bigint-benchmark
cd cuda-bigint-benchmark
git submodule update --init --recursive

# Default architecture is 50, to change it to 86:
cmake -S . -B build -DCUDA_COMPUTE_CAPABILITY=86

# To enable verbose printing
cmake -S . -B build -DCUDA_COMPUTE_CAPABILITY=86 -DENABLE_DEBUG=ON

cmake --build build
```

## Run the benchmark

``` bash
./build/CudaBigIntBenchmark_benchmark
```

## Results

``` bash
❯ ./build/CudaBigIntBenchmark_benchmark
...
--------------------------------------------------------------------------
Benchmark                                Time             CPU   Iterations
--------------------------------------------------------------------------
BM_BigIntSimpleMath/manual_time      18273 ns        24103 ns        37492
BM_CGBNSimpleMath/manual_time         4154 ns         9545 ns       168860
```
