

#include <bits/stdc++.h>


const int MN = 100000;
const int MBLOCK_DIM = 256;
const int CF = 4;


int MaxReduction(const int *input, int n) {
  int mx = 0;
  for (int i = 0; i < n; ++i) {
    mx = max(mx, input[i]);
  }
  return mx;
}


__global__ void MaxReductionKernel(int *input, int n, int *output) {
  __shared__ int smem[MBLOCK_DIM];
  unsigned int i = blockDim.x * blockIdx.x * CF + threadIdx.x;
  unsigned int t = threadIdx.x;
  int mx = input[i];
  for (int cf = 1; cf < CF; cf++) {
    mx = max(mx, input[i + MBLOCK_DIM * cf]);
  }
  smem[t] = mx;
  for (int stride = MBLOCK_DIM / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride && t + stride < n) {
      smem[t] = max(smem[t], smem[t + stride]);
    }
  }
  if (threadIdx.x == 0) {
    atomicMax(output, smem[0]);
  }
}


void TestMultiBlockKernel() {
  std::vector<int> h_inp(MN, 1);
  for (int& val : h_inp) val = rand();
  int h_out{12};

  int *d_inp, *d_out;
  cudaMalloc(&d_inp, sizeof(int) * MN);
  cudaMemcpy(d_inp, h_inp.data(), MN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, sizeof(int));

  int num_blocks = (MN + MBLOCK_DIM * CF - 1) / MBLOCK_DIM / CF;
  MaxReductionKernel<<<num_blocks, MBLOCK_DIM>>>(d_inp, MN, d_out);
  int gt = MaxReduction(h_inp.data(), MN);

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_inp);
  cudaFree(d_out);

  cudaDeviceSynchronize();

  std::cout << "SumReduction: " << h_out << ' ' << gt << '\n';
  assert(h_out == gt);
}


int main() {
  TestMultiBlockKernel();
}
