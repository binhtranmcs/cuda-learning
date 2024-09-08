cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES 50 60 72 75 80 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cuda_learning CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(pmpp main.cu)

set_target_properties(pmpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
