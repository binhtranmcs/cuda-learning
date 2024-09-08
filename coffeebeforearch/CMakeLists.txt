cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(coffeebeforearch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(test test.cu)

add_subdirectory(01_vector_addition)
add_subdirectory(02_mat_mul)
