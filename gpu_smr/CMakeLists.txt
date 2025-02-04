cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cedana-bench LANGUAGES CXX CUDA)

# XXX: Should be set based on CUDA version. E.g. CUDA v11 requires GCC <= v11
# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-11) set(CMAKE_C_COMPILER
# /usr/bin/gcc-11) set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70) # minimum Volta
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDA 12.4 REQUIRED)

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
endforeach()
