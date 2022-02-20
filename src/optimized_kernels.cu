__global__ void data_histogram_global_linearized(
    float *full_data, int *value_histogram,
    int num_bins, int n, float width, float min, float max
) {
    for (
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < n;
        id += blockDim.x * gridDim.x
    ) {
        int binId = (full_data[id] - min) / width;

        if (binId > num_bins - 1)
            binId = num_bins - 1;
        else if (binId < 0)
            binId = 0;

        atomicAdd(&value_histogram[binId], 1);
    }
}

__global__ void data_histogram_global_privatized(
    float *full_data, int *value_histogram,
    int num_bins, int n, float width, float min, float max
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // declare and initialize shared array
    extern __shared__ int private_hist[];
    for (int binId = threadIdx.x; binId < num_bins; binId += blockDim.x)
        private_hist[binId] = 0;
    __syncthreads();
    
    // private histogram
    for (int i = id; i < n; i += blockDim.x * gridDim.x) {
        int binId = (full_data[id] - min) / width;

        if (binId > num_bins - 1)
            binId = num_bins - 1;
        else if (binId < 0)
            binId = 0;

        atomicAdd(&private_hist[binId], 1);
    }
    __syncthreads();

    // commit to global memory
    for (int binId = threadIdx.x; binId < num_bins; binId += blockDim.x)
        atomicAdd(&value_histogram[binId], private_hist[binId]);
}
