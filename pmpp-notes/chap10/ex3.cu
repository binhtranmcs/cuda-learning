

#include <bits/stdc++.h>


const int N = 1001;
const int BLOCK_DIM = 512;


int SumReduction(const int *input, int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}


__global__ void SumReductionKernel(int *input, int n, int *output) {
  unsigned int i = BLOCK_DIM + threadIdx.x;
  for (int stride = BLOCK_DIM; stride >= 1; stride /= 2) {
    if (i + stride >= n && i < n) {
      input[i] += input[i - stride];
      // printf("%d %d %d\n", threadIdx.x, i, input[i]);
    }
    __syncthreads();
  }
  if (i == n - 1) {
    *output = input[i];
  }
}


void TestSingleBlockKernel() {
  std::vector<int> h_inp(N, 1);
  // std::iota(h_inp.begin(), h_inp.end(), 0);
  int h_out{12};

  int *d_inp, *d_out;
  cudaMalloc(&d_inp, sizeof(int) * N);
  cudaMemcpy(d_inp, h_inp.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, sizeof(int));

  SumReductionKernel<<<1, BLOCK_DIM>>>(d_inp, N, d_out);
  int gt = SumReduction(h_inp.data(), N);

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_inp);
  cudaFree(d_out);

  cudaDeviceSynchronize();

  std::cout << "SumReduction: " << h_out << ' ' << gt << '\n';
  assert(h_out == gt);
}


int main() {
  TestSingleBlockKernel();
}
