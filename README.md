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
BM_BigIntSimpleMul/manual_time                    34747 ns        41742 ns        19946
BM_BigIntLargeArrayAddition_8/manual_time         10550 ns        16436 ns        66619
BM_BigIntLargeArrayAddition_32/manual_time         8440 ns        14332 ns        80911
BM_BigIntLargeArrayAddition_64/manual_time         8623 ns        14386 ns        83025
BM_BigIntLargeArrayAddition_128/manual_time        9541 ns        15256 ns        75920
BM_BigIntLargeArrayAddition_256/manual_time        9834 ns        15517 ns        70995
BM_BigIntLargeArrayAddition_512/manual_time       13835 ns        19774 ns        51668
BM_BigIntLargeArrayAddition_1024/manual_time      24539 ns        30847 ns        28569
BM_CGBNSimpleMul/manual_time                       3436 ns         8865 ns       203437
BM_CGBNLargeArrayAddition_8/manual_time            9119 ns        14692 ns        76712
BM_CGBNLargeArrayAddition_32/manual_time           5307 ns        10764 ns       128862
BM_CGBNLargeArrayAddition_64/manual_time           4530 ns        10054 ns       154759
BM_CGBNLargeArrayAddition_128/manual_time          4332 ns         9801 ns       161792
BM_CGBNLargeArrayAddition_256/manual_time          4222 ns         9702 ns       166110
BM_CGBNLargeArrayAddition_512/manual_time          4219 ns         9657 ns       166432
BM_CGBNLargeArrayAddition_1024/manual_time         4730 ns        10329 ns       149331
```
