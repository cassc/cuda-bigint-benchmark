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
BM_BigIntSimpleMul/manual_time                    34449 ns        40507 ns        20104
BM_CGBNSimpleMul/manual_time                       4185 ns         9512 ns       167351
BM_BigIntLargeArrayAddition_8/manual_time      28649292 ns     28659591 ns          103
BM_BigIntLargeArrayAddition_32/manual_time     66325807 ns     66282554 ns           11
BM_BigIntLargeArrayAddition_64/manual_time     83953885 ns     83917369 ns           10
BM_BigIntLargeArrayAddition_128/manual_time    81300141 ns     81244607 ns            8
BM_BigIntLargeArrayAddition_256/manual_time    81600438 ns     81547056 ns            9
BM_BigIntLargeArrayAddition_512/manual_time    91228018 ns     91184577 ns           11
BM_BigIntLargeArrayAddition_1024/manual_time  124203170 ns    124107816 ns            6
BM_CGBNLargeArrayAddition_8/manual_time           15965 ns        21509 ns        43591
BM_CGBNLargeArrayAddition_8/manual_time           15991 ns        21606 ns        43777
BM_CGBNLargeArrayAddition_32/manual_time           8307 ns        13692 ns        83970
BM_CGBNLargeArrayAddition_64/manual_time           9674 ns        15326 ns        72205
BM_CGBNLargeArrayAddition_128/manual_time         10926 ns        16844 ns        63886
BM_CGBNLargeArrayAddition_256/manual_time         10861 ns        16611 ns        64183
BM_CGBNLargeArrayAddition_512/manual_time          9626 ns        15162 ns        72941
BM_CGBNLargeArrayAddition_1024/manual_time        10803 ns        16352 ns        64953
```
