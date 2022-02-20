__global__ void data_histogram_global_linearized(
    float *, int *, int, int, float, float, float
);

__global__ void data_histogram_global_privatized(
    float *, int *, int, int, float, float, float
);

__global__ void data_histogram_global_aggregated(
    float *, int *, int, int, float, float
);

// __global__ void data_histogram_global_