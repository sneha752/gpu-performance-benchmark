cmake_minimum_required(VERSION 3.18)
project(gpu_performance_benchmark LANGUAGES CXX)

option(ENABLE_CUDA "Build the CUDA kernels when a CUDA compiler is available" ON)
set(MATMUL_TILE_SIZE 16 CACHE STRING "Default tiled-kernel tile size (8, 16 or 32)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- CUDA detection: optional, so the CPU baseline builds anywhere ---------
set(HAVE_CUDA OFF)
if(ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # "native" targets the GPU in this machine but needs one at configure
      # time; otherwise build for common recent architectures (T4, A100, RTX 30/40).
      find_program(NVIDIA_SMI nvidia-smi)
      if(NVIDIA_SMI AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
      endif()
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(HAVE_CUDA ON)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER} (archs: ${CMAKE_CUDA_ARCHITECTURES})")
  else()
    message(WARNING "No CUDA compiler found: building the CPU-only benchmark. "
                    "CUDA rows in the CSV will be marked not_run.")
  endif()
endif()

# ---- Targets ----------------------------------------------------------------
add_library(matmul_cpu STATIC src/cpu/matmul_cpu.cpp)
target_include_directories(matmul_cpu PUBLIC include)

add_executable(matmul_bench src/main.cpp)
target_link_libraries(matmul_bench PRIVATE matmul_cpu)
if(NOT MATMUL_TILE_SIZE MATCHES "^(8|16|32)$")
  message(FATAL_ERROR "MATMUL_TILE_SIZE must be 8, 16 or 32")
endif()
target_compile_definitions(matmul_bench PRIVATE
  BENCH_BUILD_TYPE="$<CONFIG>"
  MATMUL_DEFAULT_TILE=${MATMUL_TILE_SIZE})

if(HAVE_CUDA)
  add_library(matmul_cuda STATIC
    src/cuda/matmul_naive.cu
    src/cuda/matmul_tiled.cu
    src/cuda/cuda_bench.cu)
  target_include_directories(matmul_cuda PUBLIC include PRIVATE src/cuda)
  target_link_libraries(matmul_cuda PUBLIC CUDA::cudart)
  target_compile_options(matmul_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(matmul_bench PRIVATE matmul_cuda)
  target_compile_definitions(matmul_bench PRIVATE HAVE_CUDA)
endif()

if(MSVC)
  target_compile_options(matmul_cpu PRIVATE /W4)
  target_compile_options(matmul_bench PRIVATE /W4)
else()
  target_compile_options(matmul_cpu PRIVATE -Wall -Wextra)
  target_compile_options(matmul_bench PRIVATE -Wall -Wextra)
endif()

# ---- Tests: correctness on sizes that are and are not multiples of the tile -
enable_testing()
add_test(NAME correctness
  COMMAND matmul_bench --validate-only --tile ${MATMUL_TILE_SIZE}
          --sizes 1,2,7,15,16,17,31,32,33,64,100,127,128,129,255,256,257)
foreach(t 8 32)
  if(HAVE_CUDA)
    add_test(NAME correctness_tile${t}
      COMMAND matmul_bench --validate-only --tile ${t} --sizes 1,7,31,33,100,129,257)
  endif()
endforeach()
