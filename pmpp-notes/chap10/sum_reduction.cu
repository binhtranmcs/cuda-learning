

#include <bits/stdc++.h>


const int N = 1024;
const int BLOCK_DIM = 512;


const int MN = 256 * 512;
const int MBLOCK_DIM = 256;
const int CF = 4;


int SumReduction(const int *input, int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += input[i];
  }
  return sum;
}


__global__ void SumReductionKernel0(int *input, int n, int *output) {
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


__global__ void SumReductionKernel1(int *input, int n, int *output) {
  unsigned int i = threadIdx.x;
  for (int stride = blockDim.x; stride >= 1; stride /= 2) {
    if (threadIdx.x < stride && i + stride < n) {
      input[i] += input[i + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *output = input[0];
  }
}


__global__ void SumReductionKernel2(int *input, int n, int *output) {
  __shared__ int smem[BLOCK_DIM];
  unsigned int i = threadIdx.x;
  smem[i] = input[i] + input[i + BLOCK_DIM];
  for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride && i + stride < n) {
      smem[i] += smem[i + stride];
    }
  }
  if (threadIdx.x == 0) {
    *output = smem[0];
  }
}


__global__ void SumReductionKernel3(int *input, int n, int *output) {
  __shared__ int smem[MBLOCK_DIM];
  unsigned int i = MBLOCK_DIM * blockIdx.x * 2 + threadIdx.x;
  unsigned int t = threadIdx.x;
  smem[t] = input[i] + input[i + MBLOCK_DIM];
  for (int stride = MBLOCK_DIM / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    // printf("??? %d %d\n", t, smem[t]);
    if (t < stride && t + stride < n) {
      smem[t] += smem[t + stride];
    }
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, smem[0]);
  }
}


__global__ void SumReductionKernel4(int *input, int n, int *output) {
  __shared__ int smem[MBLOCK_DIM];
  unsigned int i = blockDim.x * blockIdx.x * CF + threadIdx.x;
  unsigned int t = threadIdx.x;
  int sum = input[i];
  for (int cf = 1; cf < CF; cf++) {
    sum += input[i + MBLOCK_DIM * cf];
  }
  smem[t] = sum;
  for (int stride = BLOCK_DIM / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (t < stride && t + stride < n) {
      smem[t] += smem[t + stride];
    }
  }
  if (threadIdx.x == 0) {
    atomicAdd(output, smem[0]);
  }
}


void TestSingleBlockKernel() {
  std::vector<int> h_inp(N, 1);
  std::iota(h_inp.begin(), h_inp.end(), 0);
  int h_out{12};

  int *d_inp, *d_out;
  cudaMalloc(&d_inp, sizeof(int) * N);
  cudaMemcpy(d_inp, h_inp.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, sizeof(int));

  // SumReductionKernel0<<<1, n / 2>>>(d_inp, n, d_out);
  // SumReductionKernel1<<<1, BLOCK_DIM>>>(d_inp, N, d_out);
  SumReductionKernel2<<<1, BLOCK_DIM>>>(d_inp, N, d_out);
  int gt = SumReduction(h_inp.data(), N);

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_inp);
  cudaFree(d_out);

  cudaDeviceSynchronize();

  std::cout << "SumReduction: " << h_out << ' ' << gt << '\n';
  assert(h_out == gt);
}


void TestMultiBlockKernel() {
  std::vector<int> h_inp(MN, 1);
  // std::iota(h_inp.begin(), h_inp.end(), 0);
  int h_out{12};

  int *d_inp, *d_out;
  cudaMalloc(&d_inp, sizeof(int) * MN);
  cudaMemcpy(d_inp, h_inp.data(), MN * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc(&d_out, sizeof(int));

  int num_blocks = (MN + MBLOCK_DIM * CF - 1) / MBLOCK_DIM / CF;
  SumReductionKernel4<<<num_blocks, MBLOCK_DIM>>>(d_inp, MN, d_out);
  int gt = SumReduction(h_inp.data(), MN);

  cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_inp);
  cudaFree(d_out);

  cudaDeviceSynchronize();

  std::cout << "SumReduction: " << h_out << ' ' << gt << '\n';
  assert(h_out == gt);
}


int main() {
  // TestSingleBlockKernel();
  TestMultiBlockKernel();
}
