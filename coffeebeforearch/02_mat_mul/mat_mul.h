//
// Created by root on 07/09/2024.
//

#ifndef COFFEEBEFOREARCH_MAT_MUL_H
#define COFFEEBEFOREARCH_MAT_MUL_H


#include <vector>


template <typename T>
__global__ void MatMul(const T *a, const T *b, T *c, int N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    T& value = c[row * N + col];
    value = 0;
    for (int k = 0; k < N; k++) {
      value += a[row * N + k] * b[k * N + col];
    }
  }
}


constexpr int TILE_WIDTH = 1 << 5;
template <typename T>
__global__ void MatMulShm(const T *a, const T *b, T *c, int N) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  size_t ty = threadIdx.y, tx = threadIdx.x;

  __shared__ T s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ T s_b[TILE_WIDTH][TILE_WIDTH];

  T value{};
  for (size_t i = 0; i < N; i += blockDim.x) {
    s_a[ty][tx] = a[row * N + i + tx];
    s_b[ty][tx] = b[(i + ty) * N + col];

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      value += s_a[ty][k] * s_b[k][tx];
    }

    __syncthreads();
  }

  // Write back results
  c[row * N + col] = value;
}


constexpr int COARSE = 2;
template <typename T>
__global__ void MatMulCoarse(const T *a, const T *b, T *c, int N) {
  size_t row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  size_t col = blockIdx.x * TILE_WIDTH * COARSE + threadIdx.x;
  size_t ty = threadIdx.y, tx = threadIdx.x;

  __shared__ T s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ T s_b[TILE_WIDTH][TILE_WIDTH];

  T values[COARSE];
  for (auto& value : values) {
    value = 0;
  }
  for (size_t i = 0; i < N; i += TILE_WIDTH) {
    s_a[ty][tx] = a[row * N + i + tx];
    for (int j = 0; j < COARSE; j++) {
      s_b[ty][tx] = b[(i + ty) * N + col + j * TILE_WIDTH];
      __syncthreads();

      for (int k = 0; k < TILE_WIDTH; k++) {
        values[j] += s_a[ty][k] * s_b[k][tx];
      }
      __syncthreads();
    }
  }

  for (int j = 0; j < COARSE; j++) {
    c[row * N + col + j * TILE_WIDTH] = values[j];
  }
}


#endif // COFFEEBEFOREARCH_MAT_MUL_H
