
CUDA_ARCH := -gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_50,code=sm_50 \
	-gencode arch=compute_72,code=sm_72 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_90,code=sm_90

CXXFLAGS := $(CUDA_ARCH) -std=c++20

main:
	nvcc main.cu ${CXXFLAGS}
	./a.out
	rm a.out

c3_e1:
	nvcc chap3/ex1.cu -o c3_e1.out ${CXXFLAGS}
	./c3_e1.out
	rm c3_e1.out

c3_e2:
	nvcc chap3/ex2.cu -o c3_e2.out ${CXXFLAGS}
	./c3_e2.out
	rm c3_e2.out

c10_ex:
	nvcc chap10/sum_reduction.cu -o c10_ex.out ${CXXFLAGS}
	./c10_ex.out
	rm c10_ex.out
