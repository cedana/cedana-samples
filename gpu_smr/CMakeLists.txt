cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # SASS from Volta to Hopper, PTX for Hopper (the Dockerfile sets it per CUDA version)
  set(CMAKE_CUDA_ARCHITECTURES "70-real;75-real;80-real;86-real;89-real;90")
endif()

project(cedana-bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA 11.8 REQUIRED)

option(DEBUG "Enable debugging symbols" OFF)
if(DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
endif()

# Add a separate executable for each local workload automatically
file(GLOB WORKLOADS "./*.cu")
foreach(WORKLOAD ${WORKLOADS})
  get_filename_component(WORKLOAD_NAME ${WORKLOAD} NAME_WE)
  add_executable(${WORKLOAD_NAME} ${WORKLOAD})
  target_link_libraries(${WORKLOAD_NAME} ${CUDA_LIBRARIES})
  # cufft_* samples also link libcufft.
  if(WORKLOAD_NAME MATCHES "^cufft_")
    target_link_libraries(${WORKLOAD_NAME} ${CUDA_CUFFT_LIBRARIES})
  endif()
endforeach()

# PTX only: every GPU JIT-compiles it when it loads, as with an app that ships no native code
# for the GPU. That needs a driver at least as new as the CUDA that built it, so on an older
# node it runs on the image's CUDA forward compatibility libcuda (CED-2566).
add_executable(mem-throughput-saxpy-loop-ptx mem-throughput-saxpy-loop.cu)
target_link_libraries(mem-throughput-saxpy-loop-ptx ${CUDA_LIBRARIES})
set_target_properties(mem-throughput-saxpy-loop-ptx PROPERTIES CUDA_ARCHITECTURES 75-virtual)
