
#include <bits/stdc++.h>


struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;
  std::string func;

  GpuTimer(std::string func_name) : func(std::move(func_name)) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
  }

  ~GpuTimer() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << func << " latency: " << elapsed << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
};

__global__ void MatrixVecMulKernel(int *M, int *N, int *P, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < width) {
    int value = 0;
    for (int k = 0; k < width; k++) {
      value += M[row * width + k] * N[k];
    }
    P[row] = value;
  }
}


void Check(const std::vector<int>& M, const std::vector<int>& N,
    const std::vector<int>& P, int width) {
  for (int i = 0; i < width; i++) {
    int value = 0;
    for (int k = 0; k < width; k++) {
      value += M[i * width + k] * N[k];
    }
    if (value != P[i]) {
      std::cout << i << '\n';
      std::cout << value << ' ' << P[i] << '\n';
      assert(false);
    }
  }
}


int main() {
  const int WIDTH = 256;

  std::vector<int> h_m(WIDTH * WIDTH), h_n(WIDTH), h_p(WIDTH);

  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      h_m[i * WIDTH + j] = rand() % 100;
    }
    h_n[i] = rand() % 100;
  }

  int *d_m, *d_n, *d_p;
  int m_bytes = WIDTH * WIDTH * sizeof(int);
  int v_bytes = WIDTH * sizeof(int);
  cudaMalloc(&d_m, m_bytes);
  cudaMalloc(&d_n, v_bytes);
  cudaMalloc(&d_p, v_bytes);

  cudaMemcpy(d_m, h_m.data(), m_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, h_n.data(), v_bytes, cudaMemcpyHostToDevice);

  const int THREADS = 2;
  const int BLOCKS = (WIDTH + THREADS - 1) / THREADS;
  dim3 threads(1, THREADS);
  dim3 blocks(1, BLOCKS);
  {
    MatrixVecMulKernel<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
    GpuTimer timer("MatrixVecMulKernel");
    MatrixVecMulKernel<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
  }

  cudaMemcpy(h_p.data(), d_p, v_bytes, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  Check(h_m, h_n, h_p, WIDTH);

  cudaFree(&d_m);
  cudaFree(&d_n);
  cudaFree(&d_p);

  std::cout << "Correct\n";
}
