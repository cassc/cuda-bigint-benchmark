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

The following result is from a GTX 3060 Ti GPU.

``` bash
❯ ./build/CudaBigIntBenchmark_benchmark
...
---------------------------------------------------------------------------------------
Benchmark                                             Time             CPU   Iterations
---------------------------------------------------------------------------------------
BM_BigIntSimpleMul/manual_time                    34609 ns        41200 ns        19969
BM_CGBNSimpleMul/manual_time                       4220 ns         9632 ns       164757
BM_BigIntLargeArrayAddition_8/manual_time         10534 ns        16435 ns        65341
BM_BigIntLargeArrayAddition_32/manual_time         8488 ns        14070 ns        82066
BM_BigIntLargeArrayAddition_64/manual_time         8407 ns        14134 ns        80105
BM_BigIntLargeArrayAddition_128/manual_time        9380 ns        14954 ns        73786
BM_BigIntLargeArrayAddition_256/manual_time        9757 ns        15400 ns        71272
BM_BigIntLargeArrayAddition_512/manual_time       13627 ns        19397 ns        51057
BM_BigIntLargeArrayAddition_1024/manual_time      24586 ns        30580 ns        28582
BM_CGBNLargeArrayAddition_8/manual_time           16145 ns        22291 ns        43714
BM_CGBNLargeArrayAddition_32/manual_time           8437 ns        14027 ns        83367
BM_CGBNLargeArrayAddition_64/manual_time           9319 ns        14948 ns        74827
BM_CGBNLargeArrayAddition_128/manual_time         10869 ns        16748 ns        65128
BM_CGBNLargeArrayAddition_256/manual_time         10802 ns        16751 ns        64037
BM_CGBNLargeArrayAddition_512/manual_time          9352 ns        15029 ns        74459
BM_CGBNLargeArrayAddition_1024/manual_time        10549 ns        16183 ns        66655
```
