
CUDA_ARCH := -gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_89,code=sm_89 \
	-gencode=arch=compute_90,code=sm_90 \
	-gencode=arch=compute_90,code=compute_90

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

c10_ex3:
	nvcc chap10/ex3.cu -o c10_ex3.out ${CXXFLAGS}
	./c10_ex3.out
	rm c10_ex3.out

c10_ex4:
	nvcc chap10/ex4.cu -o c10_ex4.out ${CXXFLAGS}
	./c10_ex4.out
	rm c10_ex4.out
