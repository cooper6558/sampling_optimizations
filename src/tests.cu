#include "optimized_kernels.cuh"
#include "sampling_kernels.h"
#include <stdio.h>
#include <stdlib.h>

#define HIST_FUNCTIONS 4
#define NAIVE_BLOCK 10

void launch_data_histogram_global(
    float *, int*, int, int, int, int, float, float
);
void launch_data_histogram_global_linearized(
    float *, int*, int, int, int, int, float, float
);
void launch_data_histogram_global_privatized(
    float *, int*, int, int, int, int, float, float
);
void cpu_histogram(
    float *, int*, int, int, int, int, float, float
);

__global__ void data_histogram_global(
    float *, int *, int, int, int, int, float, float
);

float *rand_array(size_t, int, int);

void test_histogram(size_t, int, int, int);
void print_array(float *, size_t);
void print_hist(int *, size_t);

int main() {
    test_histogram(991, 3, 50, 12);

    return 0;
}

void test_histogram(size_t dim, int min, int max, int bins) {
    printf("Dimensions: %zu X %zu X %zu\n", dim, dim, dim);
    printf("Bounds: [%d, %d]\n", min, max);
    printf("Histogramming on %d bin", bins);
    if (bins > 1)
        printf("s");
    puts("");

    size_t size = dim*dim*dim;

    puts("Generating data...");
    float *data = rand_array(size, min, max);
    
    puts("Copying data to GPU...");
    float *d_Data;
    cudaMalloc(&d_Data, sizeof(float) * size);
    cudaMemcpy(
        d_Data, data, sizeof(float) * size, cudaMemcpyHostToDevice
    );

    int *histograms[HIST_FUNCTIONS];
    for (int i = 0; i < HIST_FUNCTIONS; i++)
        histograms[i] = (int *) calloc(sizeof(int), bins);
    
    // print_array(data, size);

    puts("Running cpu test...");
    cpu_histogram(
        data,   histograms[0], bins, dim, dim, dim, max, min
    );
    
    puts("Running original implementation...");
    launch_data_histogram_global(
        d_Data, histograms[1], bins, dim, dim, dim, max, min
    );

    puts("Running linearized implementation...");
    launch_data_histogram_global_linearized(
        d_Data, histograms[2], bins, dim, dim, dim, max, min
    );

    puts("Running privatized implementation...");
    launch_data_histogram_global_privatized(
        d_Data, histograms[3], bins, dim, dim, dim, max, min
    );

    puts("Running accuracy check...");
    for (int i=1; i<HIST_FUNCTIONS; i++)
        for (int j=0; j<bins; j++)
            if (histograms[0][j] != histograms[i][j])
                printf("histogram %d bad\n", i);
    
    cudaFree(d_Data);
    for (int i = 0; i < HIST_FUNCTIONS; i++) {
        print_hist(histograms[i], bins);
        free(histograms[i]);
    }
}

void cpu_histogram(
    float *data, int *histogram, int bins,
    int xdim, int ydim, int zdim, float max, float min
) {
    int n = xdim * ydim * zdim;
    float width = (max - min) / bins;

    for (int i = 0; i < n; i++) {
        int binId = (data[i] - min) / width;

        if (binId > bins - 1)
            binId = bins - 1;
        else if (binId < 0)
            binId = 0;
        
        histogram[binId] += 1;
    }
}

void launch_data_histogram_global(
    float *data, int *histogram, int bins,
    int xdim, int ydim, int zdim, float max, float min
) {
    int *d_histogram;
    cudaMalloc(&d_histogram, sizeof(int) * bins);
    cudaMemcpy(
        d_histogram, histogram, sizeof(int) * bins, cudaMemcpyHostToDevice
    );

    dim3 gridSize(
        ceil((float) xdim/NAIVE_BLOCK),
        ceil((float) ydim/NAIVE_BLOCK),
        ceil((float) zdim/NAIVE_BLOCK)
    );
    dim3 blockSize(NAIVE_BLOCK, NAIVE_BLOCK, NAIVE_BLOCK);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    data_histogram_global<<<gridSize, blockSize>>>(
        data, d_histogram, bins, xdim, ydim, zdim, max, min
    );
    cudaEventRecord(stop);


    cudaMemcpy(
        histogram, d_histogram, sizeof(int) * bins, cudaMemcpyDeviceToHost
    );
    cudaFree(d_histogram);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Naive kernel ran in %f milliseconds\n", milliseconds);
}

void launch_data_histogram_global_linearized(
    float *data, int *histogram, int bins,
    int xdim, int ydim, int zdim, float max, float min
) {
    int *d_histogram;
    cudaMalloc(&d_histogram, sizeof(int) * bins);
    cudaMemcpy(
        d_histogram, histogram, sizeof(int) * bins, cudaMemcpyHostToDevice
    );

    int size = xdim * ydim * zdim;
    int grid_size = ceil((float) size / 1024);
    if (grid_size > 1 << 16)
        grid_size = 1 << 16;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    data_histogram_global_linearized<<<grid_size, 1024>>>(
        data, d_histogram, bins, size, (max - min)/bins, min, max
    );
    cudaEventRecord(stop);

    cudaMemcpy(
        histogram, d_histogram, sizeof(int) * bins, cudaMemcpyDeviceToHost
    );
    cudaFree(d_histogram);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Linearized kernel ran in %f milliseconds\n", milliseconds);
}

void launch_data_histogram_global_privatized(
    float *data, int *histogram, int bins,
    int xdim, int ydim, int zdim, float max, float min
) {
    int *d_histogram;
    cudaMalloc(&d_histogram, sizeof(int) * bins);
    cudaMemcpy(
        d_histogram, histogram, sizeof(int) * bins, cudaMemcpyHostToDevice
    );

    int size = xdim * ydim * zdim;
    int grid_size = ceil((float) size / 1024);
    if (grid_size > 1 << 16)
        grid_size = 1 << 16;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    data_histogram_global_privatized<<<grid_size, 1024, bins * sizeof(int)>>>(
        data, d_histogram, bins, size, (max - min)/bins, min, max
    );
    cudaEventRecord(stop);

    cudaMemcpy(
        histogram, d_histogram, sizeof(int) * bins, cudaMemcpyDeviceToHost
    );
    cudaFree(d_histogram);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Privatized kernel ran in %f milliseconds\n", milliseconds);
}

float *rand_array(size_t n, int min, int max) {
    srand(time(0));
    float *out = (float *) malloc(n * sizeof(float));
    int range = max - min;
    float scale = (float) range / RAND_MAX;
    for (size_t i=0; i<n; i++)
        out[i] = rand() * scale + min;
    return out;
}

void print_array(float *data, size_t n) {
    for (size_t i=0; i<n; i++)
        printf("%5.1f", data[i]);
    puts("");
}

void print_hist(int *histogram, size_t n) {
    for (size_t i=0; i<n; i++)
        printf("%-10d", histogram[i]);
    puts("");
}