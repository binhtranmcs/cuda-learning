//
// Created by root on 07/09/2024.
//

#include <cassert>
#include <iostream>

#include "mat_mul.h"


void CheckCorrectness(const int *a, const int *b, const int *c, const int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int value = 0;
      for (int k = 0; k < N; k++) {
        value += a[i * N + k] * b[k * N + j];
      }
      if (value != c[i * N + j]) {
        std::cout << i << ' ' << j << ' ' << value << ' ' << c[i * N + j] << '\n';
      }
      assert(value == c[i * N + j]);
    }
  }
  std::cout << "Correct!\n";
}


int main() {
  constexpr int N = 1 << 10;
  //  constexpr int N = 1000;
  size_t bytes = N * N * sizeof(int);

  std::vector<int> h_a(N * N), h_b(N * N), h_c(N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
//      h_a[i * N + j] = h_b[i * N + j] = 1;
      h_a[i * N + j] = rand() % 10;
      h_b[i * N + j] = rand() % 10;
    }
  }

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

  int num_threads = TILE_WIDTH;
  int num_blocks = (N + num_threads - 1) / num_threads;
  dim3 threads(num_threads, num_threads);
  dim3 blocks(num_blocks / COARSE, num_blocks);
  //  MatMul<<<blocks, threads>>>(d_a, d_b, d_c, N);
  //  MatMulShm<<<blocks, threads>>>(d_a, d_b, d_c, N);
  MatMulCoarse<<<blocks, threads>>>(d_a, d_b, d_c, N);
  auto err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
  }
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  CheckCorrectness(h_a.data(), h_b.data(), h_c.data(), N);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}