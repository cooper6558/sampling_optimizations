# make -j					lib64/libhybrid.a
# make -j interactive		bin/interactive
# make -j reconstruction	bin/reconstruction_call
# make -j examples			interactive + reconstruction
# make -j tests				bin/tests
# make -j all				all of the above
# make clean				remove all above executables and their intermediate
# 								object files

NVCC = nvcc

NVCCFLAGS = -std=c++11 -Iinclude -arch=sm_30
CXXFLAGS =  -std=c++11 -Iinclude -O3 -Wall

CUDA_INCLUDEPATH = -I/usr/local/cuda/include
CUDA_LD_FLAGS = -L/usr/local/cuda/lib64 -lcuda -lcudart

EXAMPLE_LD = -Llib64 -lhybrid -lm -fopenmp

libhybrid: lib64/libhybrid.a

all: libhybrid tests examples

lib64/libhybrid.a: \
	build/utils.o build/sampling_funcs.o build/sampling_kernels.o
	ar -rc lib64/libhybrid.a \
		build/utils.o \
		build/sampling_funcs.o \
		build/sampling_kernels.o

build/utils.o: src/utils.cpp include/utils.h
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDEPATH) -c \
		-o build/utils.o -fopenmp src/utils.cpp

build/sampling_funcs.o: \
	src/sampling_funcs.cpp include/sampling_funcs.h include/utils.h
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDEPATH) -c \
		-o build/sampling_funcs.o -fopenmp src/sampling_funcs.cpp

build/sampling_kernels.o: \
	src/sampling_kernels.cu include/sampling_kernels.h include/utils.h
	$(NVCC) $(NVCCFLAGS) -c -o build/sampling_kernels.o src/sampling_kernels.cu

examples: interactive reconstruction

interactive: bin/interactive

reconstruction: bin/reconstruction_call

bin/interactive: build/interactive.o lib64/libhybrid.a
	$(CXX) $(CXXFLAGS) -o bin/interactive build/interactive.o \
		$(EXAMPLE_LD) $(CUDA_LD_FLAGS) -lstdc++fs

build/interactive.o: examples/interactive_example.cpp \
	include/sampling_kernels.h include/utils.h include/sampling_funcs.h
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDEPATH) -c \
		-o build/interactive.o examples/interactive_example.cpp

bin/reconstruction_call: build/reconstruction_call.o lib64/libhybrid.a
	$(CXX) $(CXXFLAGS) -o bin/reconstruction_call build/reconstruction_call.o \
		$(EXAMPLE_LD) $(CUDA_LD_FLAGS) -lstdc++fs

build/reconstruction_call.o: examples/reconstruction_caller.cpp \
	include/sampling_kernels.h include/utils.h include/sampling_funcs.h
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDEPATH) -c \
		-o build/reconstruction_call.o examples/reconstruction_caller.cpp

tests: bin/tests

bin/tests: build/tests.o build/sampling_kernels.o build/optimized_kernels.o
	$(NVCC) $(NVCCFLAGS) -o bin/tests \
		build/tests.o build/sampling_kernels.o build/optimized_kernels.o

build/tests.o: src/tests.cu \
	include/sampling_kernels.h include/optimized_kernels.cuh include/utils.h
	$(NVCC) $(NVCCFLAGS) -c -o build/tests.o src/tests.cu

build/optimized_kernels.o: \
	src/optimized_kernels.cu include/optimized_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c -o build/optimized_kernels.o \
	src/optimized_kernels.cu

clean:
	rm -f lib64/* build/* bin/*
