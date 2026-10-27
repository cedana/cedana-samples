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
endforeach()
