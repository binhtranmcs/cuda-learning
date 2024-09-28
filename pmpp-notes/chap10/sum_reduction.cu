

#include <bits/stdc++.h>


int SumReduction(const int *input, int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}


__global__ void SimpleSumReductionKernel(int *input, int n, int *output) {
  unsigned int i = 2 * threadIdx.x;
  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    if (threadIdx.x % stride == 0 && i + stride < n) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}


void TestSimpleSumReductionKernel() {
  int n = 1024;
  std::vector<int> h_inp(n, 1);
  std::iota(h_inp.begin(), h_inp.end(), 0);
  int h_out{12};

  int *d_inp, *d_out;
  cudaMalloc(&d_inp, sizeof(int) * n);
  cudaMemcpy(d_inp, h_inp.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, sizeof(int));

  SimpleSumReductionKernel<<<1, n / 2>>>(d_inp, n, d_out);
  int gt = SumReduction(h_inp.data(), n);
  
  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_inp);
  cudaFree(d_out);
  
  cudaDeviceSynchronize();

  std::cout << "SumReduction: " << h_out << ' ' << gt << '\n';
  assert(h_out == gt);
}


int main() {
  TestSimpleSumReductionKernel();
}
