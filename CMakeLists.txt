cmake_minimum_required(VERSION 3.22)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

project("CudaBigIntBenchmark"
    VERSION 0.0.1
    DESCRIPTION "Benchmarking for CudaBigInt vs CGBN"
    LANGUAGES CUDA CXX
)

enable_language(CUDA)

# Define GMP_HOME if not provided
set(GMP_HOME $ENV{GMP_HOME} CACHE PATH "Path to GMP library")

# Check if GMP_HOME is set
if (GMP_HOME)
    message(STATUS "Using GMP from: ${GMP_HOME}")
    set(GMP_INC_DIR ${GMP_HOME}/include)
    set(GMP_LIB_DIR ${GMP_HOME}/lib)
else()
    message(WARNING "GMP_HOME is not set. GMP support will not be included.")
    set(GMP_INC_DIR "")
    set(GMP_LIB_DIR "")
endif()

# Add include and library directories
include_directories(${GMP_INC_DIR})
link_directories(${GMP_LIB_DIR})

option(ENABLE_DEBUG "Enable debug printing" OFF)
if(ENABLE_DEBUG)
    add_compile_definitions(ENABLE_DEBUG)
  endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
add_subdirectory(bigint)


find_package(benchmark REQUIRED)
file(GLOB BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
message(STATUS "Benchmark sources: ${BENCHMARK_SOURCES}")

add_executable(${PROJECT_NAME}_benchmark ${BENCHMARK_SOURCES} $<TARGET_OBJECTS:CuBigInt>)

set_target_properties(${PROJECT_NAME}_benchmark
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       POSITION_INDEPENDENT_CODE ON
     )

target_include_directories(${PROJECT_NAME}_benchmark PRIVATE CGBN/include)

target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE CuBigInt)
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE benchmark::benchmark)
target_link_directories(${PROJECT_NAME}_benchmark PRIVATE ${GMP_LIB_DIR})
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE gmp)
