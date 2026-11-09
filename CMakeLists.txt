# Standalone drivers of the original grid GPU kernels.
#   cmake -B build -DGPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=70
#   cmake -B build -DGPU_BACKEND=HIP  -DCMAKE_HIP_ARCHITECTURES=gfx942
#   cmake --build build
cmake_minimum_required(VERSION 3.24)
project(grid_gpu_original LANGUAGES CXX)

set(GPU_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS CUDA HIP)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_${GPU_BACKEND}_ARCHITECTURES)
  set(CMAKE_${GPU_BACKEND}_ARCHITECTURES native)
endif()
enable_language(${GPU_BACKEND})

foreach(kernel collocate integrate coefficients)
  set(src ${kernel}_original_standalone.cu)
  set_source_files_properties(${src} PROPERTIES LANGUAGE ${GPU_BACKEND})
  add_executable(${kernel}_original ${src})
endforeach()

target_compile_definitions(coefficients_original PRIVATE __OFFLOAD_${GPU_BACKEND})
if(GPU_BACKEND STREQUAL "CUDA")
  target_compile_options(coefficients_original PRIVATE --diag-suppress=177)
endif()
