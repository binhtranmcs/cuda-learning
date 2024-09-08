

template <typename T>
void VectorAdd(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& c, int N) {
  for (int i = 0; i < N; ++i) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
__global__ void VectorAdd(T *a, T *b, T *c, int N) {
  size_t id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) {
    c[id] = a[id] + b[id];
  }
}
