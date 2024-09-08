
add_executable(mat_mul mat_mul.cu)
set_target_properties(mat_mul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
