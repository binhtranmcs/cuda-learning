#include <iostream>

__global__ void print(int *x) {
  *x = blockDim.x * blockIdx.x + threadIdx.x;
  printf("Hello World! %d\n", *x);
}

int main() {
  int *h_x = new int(12);
  int *d_x;
  cudaMalloc(&d_x, sizeof(int));
  print<<<4, 4>>>(d_x);
  cudaDeviceSynchronize();
  cudaMemcpy(h_x, d_x, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << *h_x << '\n';
}
