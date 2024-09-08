
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

__global__ void MatrixMulKernelEW(int *M, int *N, int *P, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < width && col < width) {
    int value = 0;
    for (int k = 0; k < width; k++) {
      value += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = value;
  }
}


__global__ void MatrixMulKernelRow(int *M, int *N, int *P, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < width) {
    for (int j = 0; j < width; j++) {
      int value = 0;
      for (int k = 0; k < width; k++) {
        value += M[row * width + k] * N[k * width + j];
      }
      P[row * width + j] = value;
    }
  }
}


__global__ void MatrixMulKernelCol(int *M, int *N, int *P, int width) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < width) {
    for (int i = 0; i < width; i++) {
      int value = 0;
      for (int k = 0; k < width; k++) {
        value += M[i * width + k] * N[k * width + col];
      }
      P[i * width + col] = value;
    }
  }
}


void Check(const std::vector<int>& M, const std::vector<int>& N,
    const std::vector<int>& P, int width) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      int value = 0;
      for (int k = 0; k < width; k++) {
        value += M[i * width + k] * N[k * width + j];
      }
      if (value != P[i * width + j]) {
        std::cout << i << ' ' << j << '\n';
        std::cout << value << ' ' << P[i * width + j] << '\n';
        assert(false);
      }
    }
  }
}


int main() {
  const int WIDTH = 256;

  std::vector<int> h_m(WIDTH * WIDTH), h_n(WIDTH * WIDTH), h_p(WIDTH * WIDTH);

  for (int i = 0; i < WIDTH; i++) {
    for (int j = 0; j < WIDTH; j++) {
      h_m[i * WIDTH + j] = rand() % 100;
      h_n[i * WIDTH + j] = rand() % 100;
    }
  }

  int *d_m, *d_n, *d_p;
  int bytes = WIDTH * WIDTH * sizeof(int);
  cudaMalloc(&d_m, bytes);
  cudaMalloc(&d_n, bytes);
  cudaMalloc(&d_p, bytes);

  cudaMemcpy(d_m, h_m.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_n, h_n.data(), bytes, cudaMemcpyHostToDevice);

  const int THREADS = 16;
  const int BLOCKS = (WIDTH + THREADS - 1) / THREADS; // 4
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);
  {
    MatrixMulKernelEW<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
    GpuTimer timer("MatrixMulKernelEW");
    MatrixMulKernelEW<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
  }

  // const int THREADS = 2;
  // const int BLOCKS = (WIDTH + THREADS - 1) / THREADS;
  // dim3 threads(1, THREADS);
  // dim3 blocks(1, BLOCKS);
  // {
  //   MatrixMulKernelRow<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
  //   GpuTimer timer("MatrixMulKernelRow");
  //   MatrixMulKernelRow<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
  // }

  // const int THREADS = 8;
  // const int BLOCKS = (WIDTH + THREADS - 1) / THREADS;
  // dim3 threads(THREADS, 1);
  // dim3 blocks(BLOCKS, 1);
  // {
  //   MatrixMulKernelCol<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
  //   GpuTimer timer("MatrixMulKernelCol");
  //   MatrixMulKernelCol<<<blocks, threads>>>(d_m, d_n, d_p, WIDTH);
  // }

  cudaMemcpy(h_p.data(), d_p, bytes, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  Check(h_m, h_n, h_p, WIDTH);

  cudaFree(&d_m);
  cudaFree(&d_n);
  cudaFree(&d_p);

  std::cout << "Correct\n";
}
