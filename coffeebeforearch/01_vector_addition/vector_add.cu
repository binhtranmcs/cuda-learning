//
// Created by root on 07/09/2024.
//

#include <cassert>
#include <iostream>
#include <vector>

#include "vector_add.h"


void CheckCorrectness(const int *a, const int *b, const int *c, int N) {
  for (int i = 0; i < N; i++) {
    assert(a[i] + b[i] == c[i]);
  }
  std::cout << "Correct!\n";
}


int main() {
  constexpr int N = 1e6;
  constexpr int bytes = N * sizeof(int);

  // normal
  //  std::vector<int> h_a(N), h_b(N), h_c(N);
  // pinned memory
  int *h_a, *h_b, *h_c;
  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);
  // TODO: unified memory

  for (int i = 0; i < N; ++i) {
    h_a[i] = rand() % 1000;
    h_b[i] = rand() % 1000;
  }

  std::vector<int> cc(N);

  int *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  int num_threads = 1 < 10;
  int num_blocks = (N + num_threads - 1) / num_threads;
  VectorAdd<<<num_blocks, num_threads>>>(d_a, d_b, d_c, N);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cudaDeviceSynchronize();

  CheckCorrectness(h_a, h_b, h_c, N);
}
