
#include "sampling_kernels.h"

#define BLOCK 10 // max block size is 10
#define K_NEIGHBORS 3



//////////////////////
// KERNEL FUNCTIONS //
//////////////////////

/**
 * nearest_neighbors_reconstruction_global:
 * Reconstructs sample data using nearest neighbors. 
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
__global__
void nearest_neighbors_reconstruction_global(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Determine the location of the reconstructed data point this thread will be working on
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Determine the global id of the reconstructed data point 
  int global_id = x + y*XDIM + z*XDIM*YDIM;
  // Set fake minimum distance
  float min_dist = 999999;
  int min_index = -1;
  // For each sample in the list of samples
  for (int i = 0; i < num_samples; i++){
    // Get coordinates of sample
    int sample_x = d_sample_global_ids[i] % XDIM;
    int sample_y = (d_sample_global_ids[i] / XDIM) % YDIM;
    int sample_z = d_sample_global_ids[i] / (XDIM*YDIM);
    // Determine distance between point and the sample
    float dist = std::pow((x - sample_x), 2) + std::pow((y - sample_y), 2) + std::pow((z - sample_z), 2);
    // Check to see if minimum distance
    if (dist < min_dist || min_index == -1){
      min_dist = dist;
      min_index = i;
    }
  }
  // Write min data to reconstructed data
  reconstructed_data[global_id] = d_sample_data[min_index];
}



/**
 * k_nearest_neighbors_reconstruction_global:
 * Reconstructs sample data using k nearest neighbors. 
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * K Neighbors controlled through constant: K_NEIGHBORS
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
__global__
void k_nearest_neighbors_reconstruction_global(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Determine the location of the reconstructed data point this thread will be working on
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Determine the global id of the reconstructed data point 
  int global_id = x + y*XDIM + z*XDIM*YDIM;
  // Create array of min distances
  float min_dists[K_NEIGHBORS];
  int min_indexs[K_NEIGHBORS];
  for (int i = 0; i < K_NEIGHBORS; i++){
    min_dists[i] = 999999;
    min_indexs[i] = -1;
  }
  // For each sample in the list of samples
  for (int i = 0; i < num_samples; i++){
    // Get coordinates of sample
    int sample_x = d_sample_global_ids[i] % XDIM;
    int sample_y = (d_sample_global_ids[i] / XDIM) % YDIM;
    int sample_z = d_sample_global_ids[i] / (XDIM*YDIM);
    // Determine distance between point and the sample
    float dist = std::pow((x - sample_x), 2) + std::pow((y - sample_y), 2) + std::pow((z - sample_z), 2);
    // Check to see if minimum distance
    for (int j = 0; j < K_NEIGHBORS; j++){
      if (dist < min_dists[j] || min_indexs[j] == -1){
        min_dists[j] = dist;
        min_indexs[j] = i;
        break;
      }
    }
  }
  // Weighted Average of k nearest neighbors based on distance
  float avg = 0;
  float weights = 0;
  for (int i = 0; i < K_NEIGHBORS; i++){
    avg = avg + (d_sample_data[min_indexs[i]] * (1 / (min_dists[i]+0.00000001)));
    weights = weights + (1 / (min_dists[i]+0.00000001));
  }
  avg = avg / weights;
  // Write min data to reconstructed data
  reconstructed_data[global_id] = avg;
}



/**
 * nearest_neighbors_reconstruction_shared:
 * Reconstructs sample data using nearest neighbors. Loads 
 * sample data into shared memory to reduce global memory
 * access times.
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
__global__
void nearest_neighbors_reconstruction_shared(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Determine the location of the reconstructed data point this thread will be working on
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Determine the global id of the reconstructed data point 
  int global_id = x + y*XDIM + z*XDIM*YDIM;

  // Dynamically Allocate Enough Shared Memory to hold all samples
  extern __shared__ int sample_coords[];

  // Determine how many samples each thread should load
  int samples_per_thread = std::ceil((float)num_samples/(float)(BLOCK*BLOCK*BLOCK));

  // Have subset of threads load in sample coordinates
  int local_thread_id = threadIdx.x + threadIdx.y*BLOCK + threadIdx.z*BLOCK*BLOCK;
  int sample_start_location = samples_per_thread * local_thread_id;

  // Have each thread load the nessecary number of samples
  for (int i = 0; i < samples_per_thread; i++){
    int sample_index = sample_start_location + i;
    // Ensure sample index is in range
    if (sample_index < num_samples){
      // Get coordinates of sample
      int sample_x = d_sample_global_ids[sample_index] % XDIM;
      int sample_y = (d_sample_global_ids[sample_index] / XDIM) % YDIM;
      int sample_z = d_sample_global_ids[sample_index] / (XDIM*YDIM);
      // Store coordinates in Shared Memory Array
      sample_coords[sample_index] = sample_x;
      sample_coords[sample_index+num_samples] = sample_y;
      sample_coords[sample_index+num_samples+num_samples] = sample_z;
    }
  }
  
  // Ensure all threads are synchronized
  __syncthreads();

  // Set fake minimum distance
  float min_dist = 999999;
  int min_index = -1;
  // For each sample in the list of samples
  for (int i = 0; i < num_samples; i++){
    // Get coordinates of sample
    int sample_x = sample_coords[i];
    int sample_y = sample_coords[i+num_samples];
    int sample_z = sample_coords[i+num_samples+num_samples];
    // Determine distance between point and the sample
    float dist = std::pow((x - sample_x), 2) + std::pow((y - sample_y), 2) + std::pow((z - sample_z), 2);
    // Check to see if minimum distance
    if (dist < min_dist || min_index == -1){
      min_dist = dist;
      min_index = i;
    }
  }
  // Write min data to reconstructed data
  reconstructed_data[global_id] = d_sample_data[min_index];
}



/**
 * k_nearest_neighbors_reconstruction_shared:
 * Reconstructs sample data using k nearest neighbors. Loads 
 * sample data into shared memory to reduce global memory
 * access times.
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * K Neighbors controlled through constant: K_NEIGHBORS
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
__global__
void k_nearest_neighbors_reconstruction_shared(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Determine the location of the reconstructed data point this thread will be working on
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  // Determine the global id of the reconstructed data point 
  int global_id = x + y*XDIM + z*XDIM*YDIM;

  // Dynamically Allocate Enough Shared Memory to hold all samples
  extern __shared__ int sample_coords[];

  // Determine how many samples each thread should load
  int samples_per_thread = std::ceil((float)num_samples/(float)(BLOCK*BLOCK*BLOCK));

  // Have subset of threads load in sample coordinates
  int local_thread_id = threadIdx.x + threadIdx.y*BLOCK + threadIdx.z*BLOCK*BLOCK;
  int sample_start_location = samples_per_thread * local_thread_id;

  // Have each thread load the nessecary number of samples
  for (int i = 0; i < samples_per_thread; i++){
    int sample_index = sample_start_location + i;
    // Ensure sample index is in range
    if (sample_index < num_samples){
      // Get coordinates of sample
      int sample_x = d_sample_global_ids[sample_index] % XDIM;
      int sample_y = (d_sample_global_ids[sample_index] / XDIM) % YDIM;
      int sample_z = d_sample_global_ids[sample_index] / (XDIM*YDIM);
      // Store coordinates in Shared Memory Array
      sample_coords[sample_index] = sample_x;
      sample_coords[sample_index+num_samples] = sample_y;
      sample_coords[sample_index+num_samples+num_samples] = sample_z;
    }
  }

  // Create array of min distances
  float min_dists[K_NEIGHBORS];
  int min_indexs[K_NEIGHBORS];
  for (int i = 0; i < K_NEIGHBORS; i++){
    min_dists[i] = 999999;
    min_indexs[i] = -1;
  }

  // Ensure all threads are synchronized
  __syncthreads();

  // For each sample in the list of samples
  for (int i = 0; i < num_samples; i++){
    // Get coordinates of sample
    int sample_x = sample_coords[i];
    int sample_y = sample_coords[i+num_samples];
    int sample_z = sample_coords[i+num_samples+num_samples];
    // Determine distance between point and the sample
    float dist = std::pow((x - sample_x), 2) + std::pow((y - sample_y), 2) + std::pow((z - sample_z), 2);
    // Check to see if minimum distance
    for (int j = 0; j < K_NEIGHBORS; j++){
      if (dist < min_dists[j] || min_indexs[j] == -1){
        min_dists[j] = dist;
        min_indexs[j] = i;
        break;
      }
    }
  }
  // Weighted Average of k nearest neighbors based on distance
  float avg = 0;
  float weights = 0;
  for (int i = 0; i < K_NEIGHBORS; i++){
    avg = avg + (d_sample_data[min_indexs[i]] * (1 / (min_dists[i]+0.00000001)));
    weights = weights + (1 / (min_dists[i]+0.00000001));
  }
  avg = avg / weights;
  // Write min data to reconstructed data
  reconstructed_data[global_id] = avg;
}



/**
 * data_histogram_global:
 * Creates a data histogram with num_bins and organizes 
 * data values into these bins.
 * 
 * Input:
 * full_data        - Original Dataset
 * num_bins            - Number of histogram bins to use
 * XDIM, YDIM, ZDIM - Data dimensionality
 * max, min         - Max and min value of dataset
 *
 * Output:
 * value_histogram  - Resulting histogram
 * 
 * Note:
 * Bin Width = (Max - Min) / num_bins
 **/
__global__
void data_histogram_global(float* full_data, int* value_histogram, int num_bins, int XDIM, int YDIM, int ZDIM, float max, float min){
  // Determine the location of the data point this thread will be working on
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  // Ensure data point is a valid location
  if (x_loc < XDIM && y_loc < YDIM && z_loc < ZDIM){
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;

    // Determine which bin this value belongs to
    int binId = (full_data[global_id] - min) / binWidth;
    
    // Handle Edge Cases
    if (binId > num_bins-1){
      binId = num_bins - 1;
    }
    if (binId < 0){
        binId = 0;
    }
    
    // Increment the number of values in this bin
    atomicAdd(&value_histogram[binId], 1);
  }          
}



__global__
void data_histogram_sort_global(int* value_histogram, int *histogram_bin_ids, int num_bins){
  // Set all histogram id bins to -1
  for (int i = 0; i < num_bins; i++){
    histogram_bin_ids[i] = -1;
  }

  // For each histogram bin, check histogram bin value to determine correct sorted index
  for (int i = 0; i < num_bins; i++){
    int id_location = 0;

    for (int j = 0; j < num_bins; j++){
      if(value_histogram[i] > value_histogram[j]){
        id_location++;
      }
    }

    // Ensure no overlap
    int placing = 1;
    while(placing == 1){
      if (histogram_bin_ids[id_location] != -1){
        id_location++;
      }else {
        histogram_bin_ids[id_location] = i;
        placing = 0;
      }
    }
  }
}



__global__
void acceptance_function_global(float* full_data, float *acceptance_histogram, int* value_histogram, int *histogram_bin_ids, int *samples_per_bin, float sample_ratio, int num_bins, int XDIM, int YDIM, int ZDIM){
  // Determine the total number of samples to take
  int tot_samples = sample_ratio * (XDIM * YDIM * ZDIM);

  // Determine the target number of samples per bin
  int target_bin_samples = (tot_samples/num_bins);

  // Distribute samples across bins
  int samples; 
  int remaining_tot_samples = tot_samples;
  for (int i = 0; i < num_bins; i++){
    // Get current bin ID
    int index = histogram_bin_ids[i];

    // If the bin has more data values than target samples set samples to target value
    if(value_histogram[index] > target_bin_samples){
        samples = target_bin_samples;
    // If the bin has less data values than target samples set to total samples in bin
    } else{
        samples = value_histogram[index];
    }

    // Store samples to gather for this bin
    samples_per_bin[index] = samples;

    // Subtract samples taken from total samples
    remaining_tot_samples = remaining_tot_samples - samples;
    // Update target number of samples per bin
    target_bin_samples = remaining_tot_samples/(num_bins-i);
  }

  // Determine acceptance rate for each bin as the samples to take from that bin divided by the total samples in that bin
  for (int i = 0; i < num_bins; i++){
      // If bin has no samples, set acceptance to zero
      if (value_histogram[i] == 0){
          acceptance_histogram[i] = 0;
      }else{
          acceptance_histogram[i] = (float)samples_per_bin[i] / (float)value_histogram[i];
      }
  }     
}

__global__
void curand_initialize(unsigned int seed, curandState_t* states, int XDIM, int YDIM, int ZDIM){
  // Determine the location of the data point this thread needs to make a random number for
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  curand_init(seed, global_id, 0, &states[global_id]);
}

__global__ 
void random_value_array(curandState_t* states, float* numbers, int XDIM, int YDIM, int ZDIM){
  // Determine the location of the data point this thread needs to make a random number for
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  /* curand works like rand - except that it takes a state as a parameter */
  numbers[global_id] = curand_uniform(&states[global_id]);
}



/**
 * value_histogram_importance_sampling_global:
 * Using the acceptance histogram and random value array, 
 * determine which data values to sample, marking these 
 * locations in the stencil.
 * 
 * Input:
 * full_data              - Original dataset
 * num_bins                  - Number of histogram bins
 * max, min               - Max and min value of dataset
 * XDIM, YDIM, ZDIM       - Original data dimensionality
 * XBLOCK, YBLOCK, ZBLOCK - Block dimensionality
 * acceptance_histogram   - Histogram of acceptance rates per bin
 * random_values          - Randomly generated number values
 *
 * Output:
 * stencil                - Overlay to determine which samples to save
 * samples_per_block      - Number of samples saved per block
 * 
 * Note:
 * Random values should be between 0 and 1.
 * TODO: Samples per block could be calculated outside of function to 
 * improve performance.
 **/
__global__
void value_histogram_importance_sampling_global(float* full_data, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float* acceptance_histogram, float* random_values, float* stencil, int* samples_per_block){
  // Determine the location of the data point this thread will be working on
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  // Ensure data point is a valid location
  if (x_loc < XDIM && y_loc < YDIM && z_loc < ZDIM){
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;
    
    // Determine which bin this value belongs to
    int binId = (full_data[global_id] - min) / binWidth;

    // Handle Edge Cases
    if (binId > num_bins-1){
      binId = num_bins - 1;
    }
    if (binId < 0){
        binId = 0;
    }

    // Determine block ID of this data value
    // Block count in x and y direction
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    
    // Data Point X,Y,Z Coordinates
    int x_id = global_id % XDIM;
    int y_id = (global_id / XDIM) % YDIM;
    int z_id =  global_id / (XDIM*YDIM);

    // Block X,Y,Z Coordinates
    int block_x = (x_id/XBLOCK);
    int block_y = (y_id/YBLOCK);
    int block_z = (z_id/ZBLOCK);
    
    // Calculate Block ID
    int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);
    
    // Determine whether to save sample or not
    // If difference is positive, save sample, else dont save
    stencil[global_id] =  acceptance_histogram[binId] - random_values[global_id];
    
    // If sample chosen to be saved, increment samples saved in that block
    if (stencil[global_id] > 0){
      atomicAdd(&samples_per_block[block_id], 1);
    }
  }
}



__global__
void block_histogram_list_global(float* full_data, int* current_histogram_list, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK){
  // Determine the location of the data point this thread will be working on
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  // Ensure data point is a valid location
  if (x_loc < XDIM && y_loc < YDIM && z_loc < ZDIM){
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;

    // Determine which bin this value belongs to
    int binId = (full_data[global_id] - min) / binWidth;
    
    // Handle Edge Cases
    if (binId > num_bins-1){
      binId = num_bins - 1;
    }
    if (binId < 0){
        binId = 0;
    }


    // Determine block ID of this data value
    // Block count in x and y direction
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    
    // Data Point X,Y,Z Coordinates
    int x_id = global_id % XDIM;
    int y_id = (global_id / XDIM) % YDIM;
    int z_id =  global_id / (XDIM*YDIM);

    // Block X,Y,Z Coordinates
    int block_x = (x_id/XBLOCK);
    int block_y = (y_id/YBLOCK);
    int block_z = (z_id/ZBLOCK);
    
    // Calculate Block ID
    int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);
    
    // get offset
    int offset_id = block_id*(num_bins) + binId;

    // Increment the number of values in this bin
    atomicAdd(&current_histogram_list[offset_id], 1);
  }
}

__global__
void utilize_decision_HISTOGRAM_BASED_global(int reuse_flag, int* reference_histogram_list, int* current_histogram_list, int num_bins, int* samples_per_block, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK){
  
  // TODO launch this process with num_blocks threads and have each thread process one block

  int num_blocks = (XDIM*YDIM*ZDIM) / (XBLOCK*YBLOCK*ZBLOCK);

  // compare histograms with this block_id offset
  for (int block_id = 0; block_id < num_blocks; block_id++){

    int offset_id = block_id*num_bins;

    // Check if this block in T - 1 reused samples from T - 2
    if (reference_histogram_list[offset_id] == reuse_flag){
      // If so, do not consider reusing again to avoid domino effects during reconstruction
      //utilize_list[block_id] = 0;
      samples_per_block[block_id] = 0;
    }else{
      // If not, compare histograms through histogram intersection
      float score = 0;
      float q_sum = 0;
      for (int i = 0; i < num_bins; i++){
          if (reference_histogram_list[offset_id+i] < current_histogram_list[offset_id+i]){
              score = score + reference_histogram_list[offset_id+i]; // add the minimum between the two
          }else{
              score = score + current_histogram_list[offset_id+i]; // add the minimum between the two
          }
          q_sum = q_sum + current_histogram_list[offset_id+i];
      }
      
      //Normalize Score
      // 1.0 means exact same, 0.0 means most different
      score = score/q_sum;
      
      // If exactly the same, reuse
      if (score == 1.0){
        //utilize_list[block_id] = 1;
        current_histogram_list[block_id*num_bins] = reuse_flag; // FLAGGED FOR REUSE
        samples_per_block[block_id] = reuse_flag; // FLAGGED FOR REUSE
      }else{ // If not then don't reuse
        //utilize_list[block_id] = 0;
        samples_per_block[block_id] = 0;
      }
    }
  }
}



__global__
void block_error_calculator_global(float* full_data, int* reference_samples_per_block, int total_reference_samples,  int* reference_block_sample_ids, float* reference_block_sample_data, float* reference_block_errors, int* reference_block_errors_ids, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK){
  // Determine which reference sample you are calculating the error for
  int ref_sample_id = blockIdx.x*blockDim.x+threadIdx.x;

  if (ref_sample_id < total_reference_samples){
    int global_id = reference_block_sample_ids[ref_sample_id];

    // Get data point value
    float current_value = full_data[global_id];
    // Get sample data point value
    float reference_value = reference_block_sample_data[ref_sample_id];

    // Calculate squared difference
    float rmse = reference_value - current_value;
    rmse = rmse * rmse;
    
    // Determine block ID of this data value
    // Block count in x and y direction
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    
    // Data Point X,Y,Z Coordinates
    int x_id = global_id % XDIM;
    int y_id = (global_id / XDIM) % YDIM;
    int z_id =  global_id / (XDIM*YDIM);

    // Block X,Y,Z Coordinates
    int block_x = (x_id/XBLOCK);
    int block_y = (y_id/YBLOCK);
    int block_z = (z_id/ZBLOCK);

    // Calculate Block ID
    int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);

    // Add error to error data arrays
    reference_block_errors[ref_sample_id] = rmse;
    reference_block_errors_ids[ref_sample_id] = block_id;

    // Increment the number of samples in the reference block
    atomicAdd(&reference_samples_per_block[block_id], 1);
  }
}



__global__
void utilize_decision_ERROR_BASED_global(int reuse_flag, float* reference_block_errors, int* reference_block_errors_ids, int* reference_samples_per_block, int total_reference_samples, int* samples_per_block, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float error_threshold){
  // Determine number of blocks
  int num_blocks = (XDIM*YDIM*ZDIM) / (XBLOCK*YBLOCK*ZBLOCK);

  // Check error in each block
  for (int block_id = 0; block_id < num_blocks; block_id++){
    float rmse_block = 0;
    if(reference_samples_per_block[block_id] > 0){
      for (int reference = 0; reference < total_reference_samples; reference++){
        if (reference_block_errors_ids[reference] == block_id){
          rmse_block = rmse_block + reference_block_errors[reference];
        }
      }
      float temp = rmse_block / reference_samples_per_block[block_id];
      rmse_block = sqrt(temp);
    } else {
      rmse_block = error_threshold + 1;
    }

    
    // If within tolerance, reuse
    if (rmse_block <= error_threshold){
      samples_per_block[block_id] = reuse_flag; // FLAGGED FOR REUSE
    // If not then don't reuse
    }else{
      samples_per_block[block_id] = 0;
    }
    
  }
}




/**
 * histogram_reuse_method_global:
 * Works similar to value_histogram_importance_sampling_global
 * but will skip blocks that are being reused from the previous
 * timestep.
 * 
 * Input:
 * full_data              - Original dataset
 * num_bins                  - Number of histogram bins
 * max, min               - Max and min value of dataset
 * XDIM, YDIM, ZDIM       - Original data dimensionality
 * XBLOCK, YBLOCK, ZBLOCK - Block dimensionality
 * acceptance_histogram   - Histogram of acceptance rates per bin
 * random_values          - Randomly generated number values
 *
 * Output:
 * stencil                - Overlay to determine which samples to save
 * samples_per_block      - Number of samples saved per block
 * 
 * Note:
 * Random values should be between 0 and 1.
 * Putting -1's in the samples_per_block signal block is to be reused.
 **/
__global__
void histogram_reuse_method_global(float* full_data, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float* acceptance_histogram, float* random_values, float* stencil, int* samples_per_block, int* num_samples_list){
  // Determine the location of the data point this thread will be working on
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  // Ensure data point is a valid location
  if (x_loc < XDIM && y_loc < YDIM && z_loc < ZDIM){ 
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;
    
    // Determine which bin this value belongs to
    int binId = (full_data[global_id] - min) / binWidth;

    // Handle Edge Cases
    if (binId > num_bins-1){
      binId = num_bins - 1;
    }
    if (binId < 0){
        binId = 0;
    }

    // Determine block ID of this data value
    // Block count in x and y direction
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    
    // Data Point X,Y,Z Coordinates
    int x_id = global_id % XDIM;
    int y_id = (global_id / XDIM) % YDIM;
    int z_id =  global_id / (XDIM*YDIM);

    // Block X,Y,Z Coordinates
    int block_x = (x_id/XBLOCK);
    int block_y = (y_id/YBLOCK);
    int block_z = (z_id/ZBLOCK);

    // Calculate Block ID
    int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);

    // Ensure block is not being reused from previous timestep
    // If block is being reused, do not sample
    if (samples_per_block[block_id] == -1){
      stencil[global_id] =  0;
    // If block is not being reused, take samples
    }else{
      // Determine whether to save sample or not
      stencil[global_id] =  acceptance_histogram[binId] - random_values[global_id];
              
      // If sample chosen to be saved, increment samples saved in that block
      if (stencil[global_id] > 0){
        atomicAdd(&samples_per_block[block_id], 1);
        atomicAdd(&num_samples_list[0], 1);
      }
    }
  }
}



__global__
void error_reuse_method_global(float* full_data, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float* acceptance_histogram, float* random_values, float* stencil, int* samples_per_block, int* num_samples_list){
  // Determine the location of the data point this thread will be working on
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  // Ensure data point is a valid location
  if (x_loc < XDIM && y_loc < YDIM && z_loc < ZDIM){ 
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;
    
    // Determine which bin this value belongs to
    int binId = (full_data[global_id] - min) / binWidth;

    // Handle Edge Cases
    if (binId > num_bins-1){
      binId = num_bins - 1;
    }
    if (binId < 0){
        binId = 0;
    }

    // Determine block ID of this data value
    // Block count in x and y direction
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    
    // Data Point X,Y,Z Coordinates
    int x_id = global_id % XDIM;
    int y_id = (global_id / XDIM) % YDIM;
    int z_id =  global_id / (XDIM*YDIM);

    // Block X,Y,Z Coordinates
    int block_x = (x_id/XBLOCK);
    int block_y = (y_id/YBLOCK);
    int block_z = (z_id/ZBLOCK);

    // Calculate Block ID
    int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);

    // Ensure block is not being reused from previous timestep
    // If block is being reused, do not sample
    if (samples_per_block[block_id] == -1){
      stencil[global_id] =  0;
    // If block is not being reused, take samples
    }else{
      // Determine whether to save sample or not
      stencil[global_id] =  acceptance_histogram[binId] - random_values[global_id];
              
      // If sample chosen to be saved, increment samples saved in that block
      if (stencil[global_id] > 0){
        atomicAdd(&samples_per_block[block_id], 1);
        atomicAdd(&num_samples_list[0], 1);
      }
    }
  }
}



__global__
void random_sampling_global(float* full_data, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float sample_ratio, float* random_values, float* stencil, int* samples_per_block, int* num_samples_list){
  
  // Determine the location of the data point this thread will be working on
  int x_loc = blockIdx.x*blockDim.x+threadIdx.x;
  int y_loc = blockIdx.y*blockDim.y+threadIdx.y;
  int z_loc = blockIdx.z*blockDim.z+threadIdx.z;
  // Determine the global id of the data point 
  int global_id = x_loc + y_loc*XDIM + z_loc*XDIM*YDIM;

  int num_blocks = (XDIM*YDIM*ZDIM) / (XBLOCK*YBLOCK*ZBLOCK);

  // TODO I think this is faster than making another variable to pass and doing an atomic add in the utilize_decision function?
  int num_blocks_reused = 0;
  for(int i = 0; i < num_blocks; i++){
    if (samples_per_block[i] == -1){
      num_blocks_reused++;
    }
  }

  // Ensure data point is a valid location
  if (x_loc < XDIM && y_loc < YDIM && z_loc < ZDIM){ 

    // calculate how many samples are remaining
    int samples_remaining = (XDIM*YDIM*ZDIM)*sample_ratio - num_samples_list[0] - num_samples_list[1];
    // calculate new sample ratio

    float new_sample_ratio;
    if (num_blocks_reused != 0){
      new_sample_ratio = (float)samples_remaining/((XBLOCK*YBLOCK*ZBLOCK*(num_blocks-num_blocks_reused)));
      //new_sample_ratio =  ((float)samples_remaining/((float)num_blocks - (float)num_blocks_reused))/(float)(XBLOCK*YBLOCK*ZBLOCK) + 0.001; //TODO FIX THIS
    }else{
      new_sample_ratio = 0;
    }

    if (new_sample_ratio > 0 && samples_remaining > 0){

      /*
      // Calculate width of bins
      float range = max - min;
      float binWidth = range/num_bins;
      
      // Determine which bin this value belongs to
      int binId = (full_data[global_id] - min) / binWidth;

      // Handle Edge Cases
      if (binId > num_bins-1){
        binId = num_bins - 1;
      }
      if (binId < 0){
          binId = 0;
      }
      */

      // Determine block ID of this data value
      // Block count in x and y direction
      int XB_COUNT = (XDIM/XBLOCK);
      int YB_COUNT = (YDIM/YBLOCK);
      
      // Data Point X,Y,Z Coordinates
      int x_id = global_id % XDIM;
      int y_id = (global_id / XDIM) % YDIM;
      int z_id =  global_id / (XDIM*YDIM);

      // Block X,Y,Z Coordinates
      int block_x = (x_id/XBLOCK);
      int block_y = (y_id/YBLOCK);
      int block_z = (z_id/ZBLOCK);

      // Calculate Block ID
      int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);

      // Ensure block is not being reused from previous timestep
      // If block is being reused, do not sample
      if (samples_per_block[block_id] == -1){
        stencil[global_id] =  0;
      // If block is not being reused, take samples
      }else{
        // Determine whether to save sample or not

        // check if stencil has not already sampled this point
        if (stencil[global_id] <= 0){
          // I'm currently using the same random values as before
          // So I'm adding this temp fix to address that for now
          int temp_fix = (global_id+1) % (XDIM*YDIM*ZDIM);

          stencil[global_id] =  new_sample_ratio - random_values[temp_fix];        
          //stencil[global_id] =  new_sample_ratio - random_values[global_id]; // TODO I think this is right but it should be validated
          
          // If sample chosen to be saved, increment samples saved in that block
          // TODO there is a lot of thread divergence here, which will slow us down
          
          if (stencil[global_id] > 0){
            atomicAdd(&samples_per_block[block_id], 1);
            atomicAdd(&num_samples_list[1], 1);
          }
          
        }
      }
    }
  }
}


__global__
void random_sampling_single_thread_global(float* full_data, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float sample_ratio, float* random_values, float* stencil, int* samples_per_block, int* num_samples_list){

  int num_blocks = (XDIM*YDIM*ZDIM) / (XBLOCK*YBLOCK*ZBLOCK);

  // TODO I think this is faster than making another variable to pass and doing an atomic add in the utilize_decision function?
  int num_blocks_reused = 0;
  for(int i = 0; i < num_blocks; i++){
    if (samples_per_block[i] == -1){
      num_blocks_reused++;
    }
  }

  // calculate how many samples are remaining
  int samples_remaining = (XDIM*YDIM*ZDIM)*sample_ratio - num_samples_list[0];
  // calculate new sample ratio

  float new_sample_ratio;
  if (num_blocks_reused != 0){
    //new_sample_ratio = (float)samples_remaining/((XBLOCK*YBLOCK*ZBLOCK*(num_blocks-num_blocks_reused)));
    //new_sample_ratio = (float)samples_remaining/(XDIM*YDIM*ZDIM);
    new_sample_ratio = 10;
  }else{
    new_sample_ratio = 0;
  }

  for (int global_id = 0; global_id < XDIM*YDIM*ZDIM; global_id++){
    samples_remaining = samples_remaining - num_samples_list[1];
    if (new_sample_ratio > 0 && samples_remaining > 0){

      // Determine block ID of this data value
      // Block count in x and y direction
      int XB_COUNT = (XDIM/XBLOCK);
      int YB_COUNT = (YDIM/YBLOCK);
      
      // Data Point X,Y,Z Coordinates
      int x_id = global_id % XDIM;
      int y_id = (global_id / XDIM) % YDIM;
      int z_id =  global_id / (XDIM*YDIM);
  
      // Block X,Y,Z Coordinates
      int block_x = (x_id/XBLOCK);
      int block_y = (y_id/YBLOCK);
      int block_z = (z_id/ZBLOCK);
  
      // Calculate Block ID
      int block_id  = block_x + (block_y*XB_COUNT) + (block_z*XB_COUNT*YB_COUNT);
  
      // Ensure block is not being reused from previous timestep
      // If block is being reused, do not sample
      if (samples_per_block[block_id] == -1){
        stencil[global_id] = 0;
      // If block is not being reused, take samples
      }else{
        // Determine whether to save sample or not
  
        // check if stencil has not already sampled this point
        if (stencil[global_id] <= 0){
          // I'm currently using the same random values as before
          // So I'm adding this temp fix to address that for now
          int temp_fix = (global_id+1) % (XDIM*YDIM*ZDIM);
  
          stencil[global_id] =  new_sample_ratio - random_values[temp_fix];        
          //stencil[global_id] =  new_sample_ratio - random_values[global_id]; // TODO I think this is right but it should be validated
          
          // If sample chosen to be saved, increment samples saved in that block
          // TODO there is a lot of thread divergence here, which will slow us down
          
          if (stencil[global_id] > 0){
            //atomicAdd(&samples_per_block[block_id], 1);
            //atomicAdd(&num_samples_list[1], 1);
            samples_per_block[block_id]++;
            num_samples_list[1]++;
          }
        }
      }
    }else{
      break;
    }
  }
}





//////////////////////
// LAUNCH FUNCTIONS //
//////////////////////

void launch_rand_intialization(curandState_t* states, int XDIM, int YDIM, int ZDIM){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  //unsigned int random_seed = 0;
  unsigned int random_seed = (unsigned int)std::chrono::system_clock::now().time_since_epoch().count();


  // Create an array of curand random values
  dim3 randGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 randBlockSize(BLOCK,BLOCK,BLOCK);
  curand_initialize<<<randGridSize, randBlockSize>>>(random_seed, states, XDIM, YDIM, ZDIM);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}


/**
 * launch_histogram:
 * Launch CUDA histogram creation kernel
 * 
 * Input:
 * full_data        - Original Dataset
 * num_bins            - Number of histogram bins to use
 * XDIM, YDIM, ZDIM - Data dimensionality
 * max, min         - Max and min value of dataset
 *
 * Output:
 * value_histogram  - Resulting histogram
 * 
 * Note:
 * Bin Width = (Max - Min) / num_bins
 **/
void launch_histogram(float* full_data, int* value_histogram, int num_bins, int XDIM, int YDIM, int ZDIM, float max, float min){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Set grid and block dimensions so that each thread sorts a single data value
  dim3 gridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 blockSize(BLOCK,BLOCK,BLOCK);
  // Call histogram creation kernel
  data_histogram_global<<<gridSize, blockSize>>>(full_data, value_histogram, num_bins, XDIM, YDIM, ZDIM, max, min);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}



/**
 * launch_importance_based_sampling:
 * Launch CUDA importance based sampling decision kernel.
 * 
 * Input:
 * full_data              - Original dataset
 * num_bins                  - Number of histogram bins
 * max, min               - Max and min value of dataset
 * XDIM, YDIM, ZDIM       - Original data dimensionality
 * XBLOCK, YBLOCK, ZBLOCK - Block dimensionality
 * acceptance_histogram   - Histogram of acceptance rates per bin
 * random_values          - Randomly generated number values
 *
 * Output:
 * stencil                - Overlay to determine which samples to save
 * samples_per_block      - Number of samples saved per block
 * 
 * Note:
 * Random values should be between 0 and 1.
 **/
void launch_importance_based_sampling(float* full_data, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float* acceptance_histogram, float* random_values, float* stencil, int* samples_per_block){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Set grid and block dimensions so that each thread works on a single data value
  dim3 gridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 blockSize(BLOCK,BLOCK,BLOCK);
  // Call importance based sampling decision kernel
  value_histogram_importance_sampling_global<<<gridSize, blockSize>>>(full_data, num_bins, max, min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram, random_values, stencil, samples_per_block);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}


void launch_importance_based_sampling_method(float* full_data, float sample_ratio, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float *acceptance_histogram, int *histogram_bin_ids, int *samples_per_bin, int *value_histogram, float *stencil, int *samples_per_block, curandState_t* states, float *random_numbers){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Build histogram of entire dataset
  dim3 histGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 histBlockSize(BLOCK,BLOCK,BLOCK);
  data_histogram_global<<<histGridSize, histBlockSize>>>(full_data, value_histogram, num_bins, XDIM, YDIM, ZDIM, max, min);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Sort histogram bins from least to greatest
  thrust::device_ptr<int>th_value_histogram(value_histogram);
  thrust::device_ptr<int>th_histogram_bin_ids(histogram_bin_ids);
  thrust::sort_by_key(th_value_histogram, th_value_histogram + num_bins, th_histogram_bin_ids, thrust::greater<int>());

  // Sort histogram bins from least to greatest
  /**
  dim3 sortGridSize(1,1,1);
  dim3 sortBlockSize(1,1,1);
  data_histogram_sort_global<<<sortGridSize, sortBlockSize>>>(value_histogram, histogram_bin_ids, num_bins);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  **/

  // Create Importance Factor / Acceptance Function
  dim3 acceptGridSize(1,1,1);
  dim3 acceptBlockSize(1,1,1);
  acceptance_function_global<<<acceptGridSize, acceptBlockSize>>>(full_data, acceptance_histogram, value_histogram, histogram_bin_ids, samples_per_bin, sample_ratio, num_bins, XDIM, YDIM, ZDIM);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Create an array of curand random values
  dim3 randGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 randBlockSize(BLOCK,BLOCK,BLOCK);
  random_value_array<<<randGridSize, randBlockSize>>>(states, random_numbers, XDIM, YDIM, ZDIM);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  
  // Begin Sampling Process
  dim3 sampleGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 sampleBlockSize(BLOCK,BLOCK,BLOCK);
  // Call importance based sampling decision kernel
  value_histogram_importance_sampling_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, max, min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram, random_numbers, stencil, samples_per_block);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}



// optimized / all on gpu version
void launch_histogram_based_reuse_sampling_method(float* full_data, float sample_ratio, int num_bins, float data_max, float data_min, float lifetime_max, float lifetime_min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float *acceptance_histogram, int *histogram_bin_ids, int *samples_per_bin, int *value_histogram, float *stencil, int *samples_per_block, int *reference_histogram_list, int *current_histogram_list, int timestep, int *num_samples_list, curandState_t* states, float *random_numbers){

  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Build histogram of entire dataset
  dim3 histGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 histBlockSize(BLOCK,BLOCK,BLOCK);
  data_histogram_global<<<histGridSize, histBlockSize>>>(full_data, value_histogram, num_bins, XDIM, YDIM, ZDIM, data_max, data_min);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Sort histogram bins from least to greatest
  thrust::device_ptr<int>th_value_histogram(value_histogram);
  thrust::device_ptr<int>th_histogram_bin_ids(histogram_bin_ids);
  thrust::sort_by_key(th_value_histogram, th_value_histogram + num_bins, th_histogram_bin_ids, thrust::greater<int>());

  // Sort histogram bins from least to greatest
  /**
  dim3 sortGridSize(1,1,1);
  dim3 sortBlockSize(1,1,1);
  data_histogram_sort_global<<<sortGridSize, sortBlockSize>>>(value_histogram, histogram_bin_ids, num_bins);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  **/

  // Create Importance Factor / Acceptance Function
  dim3 acceptGridSize(1,1,1);
  dim3 acceptBlockSize(1,1,1);
  acceptance_function_global<<<acceptGridSize, acceptBlockSize>>>(full_data, acceptance_histogram, value_histogram, histogram_bin_ids, samples_per_bin, sample_ratio, num_bins, XDIM, YDIM, ZDIM);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // fill current_histogram_list
  // atomic add to fill histogram based on offset of blockid and binid
  block_histogram_list_global<<<histGridSize, histBlockSize>>>(full_data, current_histogram_list, num_bins, lifetime_max, lifetime_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK);

  // Create an array of curand random values
  dim3 randGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 randBlockSize(BLOCK,BLOCK,BLOCK);
  random_value_array<<<randGridSize, randBlockSize>>>(states, random_numbers, XDIM, YDIM, ZDIM);

  // Begin Sampling Process
  if (timestep == 0){
    // Call importance based sampling decision kernel
    dim3 sampleGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
    dim3 sampleBlockSize(BLOCK,BLOCK,BLOCK);
    value_histogram_importance_sampling_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram, random_numbers, stencil, samples_per_block);
  
  }else{

    // call utilize_decision_global
    // returns an array of size num_blocks of 0's and 1's for reuse yes or no
    // compares reference_histogram_list and newly filled current_histogram_list
    dim3 decisionGridSize(1,1,1);
    dim3 decisionBlockSize(1,1,1);
    utilize_decision_HISTOGRAM_BASED_global<<<decisionGridSize, decisionBlockSize>>>(-1, reference_histogram_list, current_histogram_list, num_bins, samples_per_block, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK);

    // Call histogram reuse sampling decision kernel
    dim3 sampleGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
    dim3 sampleBlockSize(BLOCK,BLOCK,BLOCK);
    histogram_reuse_method_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram, random_numbers, stencil, samples_per_block, num_samples_list);
  
    // get current number of samples
    // Call random sampling kernel
    
    // TODO I am reusing the random values list, TODO make a 2nd randoms list
    random_sampling_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_ratio, random_numbers, stencil, samples_per_block, num_samples_list);
    //random_sampling_single_thread_global<<<decisionGridSize, decisionBlockSize>>>(full_data, num_bins, max, min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_ratio, random_numbers, stencil, samples_per_block, num_samples_list);

  }
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

}

// optimized / all on gpu version
void launch_error_based_reuse_sampling_method(float* full_data, float sample_ratio, int num_bins, float data_max, float data_min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, \
  int ZBLOCK, float *acceptance_histogram, int *histogram_bin_ids, int *samples_per_bin, int *value_histogram, float *stencil, int *samples_per_block, int* reference_samples_per_block, \
  int total_reference_samples,  int* reference_block_sample_ids, float* reference_block_sample_data, float* reference_block_errors, int* reference_block_errors_ids, int timestep, int *num_samples_list, float error_threshold, curandState_t* states, float *random_numbers){

  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Build histogram of entire dataset
  dim3 histGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 histBlockSize(BLOCK,BLOCK,BLOCK);
  data_histogram_global<<<histGridSize, histBlockSize>>>(full_data, value_histogram, num_bins, XDIM, YDIM, ZDIM, data_max, data_min);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Sort histogram bins from least to greatest
  thrust::device_ptr<int>th_value_histogram(value_histogram);
  thrust::device_ptr<int>th_histogram_bin_ids(histogram_bin_ids);
  thrust::sort_by_key(th_value_histogram, th_value_histogram + num_bins, th_histogram_bin_ids, thrust::greater<int>());

  // Sort histogram bins from least to greatest
  /**
  dim3 sortGridSize(1,1,1);
  dim3 sortBlockSize(1,1,1);
  data_histogram_sort_global<<<sortGridSize, sortBlockSize>>>(value_histogram, histogram_bin_ids, num_bins);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  **/

  // Create Importance Factor / Acceptance Function
  dim3 acceptGridSize(1,1,1);
  dim3 acceptBlockSize(1,1,1);
  acceptance_function_global<<<acceptGridSize, acceptBlockSize>>>(full_data, acceptance_histogram, value_histogram, histogram_bin_ids, samples_per_bin, sample_ratio, num_bins, XDIM, YDIM, ZDIM);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  // Create an array of curand random values
  dim3 randGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 randBlockSize(BLOCK,BLOCK,BLOCK);
  random_value_array<<<randGridSize, randBlockSize>>>(states, random_numbers, XDIM, YDIM, ZDIM);

  // Begin Sampling Process
  if (timestep == 0){
    // Call importance based sampling decision kernel
    dim3 sampleGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
    dim3 sampleBlockSize(BLOCK,BLOCK,BLOCK);
    value_histogram_importance_sampling_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram, random_numbers, stencil, samples_per_block);

  }else{
    // Calculate error between current and T-1 
    dim3 errorGridSize(std::ceil((float)total_reference_samples/(float)BLOCK),1,1);
    dim3 errorBlockSize(BLOCK,1,1);
    block_error_calculator_global<<<errorGridSize, errorBlockSize>>>(full_data, reference_samples_per_block, total_reference_samples, reference_block_sample_ids, reference_block_sample_data, reference_block_errors, reference_block_errors_ids, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK);

    // call utilize_decision_global
    // returns an array of size num_blocks of 0's and 1's for reuse yes or no
    // compares reference_sample_list and current data points
    dim3 decisionGridSize(1,1,1);
    dim3 decisionBlockSize(1,1,1);
    utilize_decision_ERROR_BASED_global<<<decisionGridSize, decisionBlockSize>>>(-1, reference_block_errors, reference_block_errors_ids, reference_samples_per_block, total_reference_samples, samples_per_block, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, error_threshold);



    // Call error reuse sampling decision kernel
    dim3 sampleGridSize(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
    dim3 sampleBlockSize(BLOCK,BLOCK,BLOCK);
    error_reuse_method_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram, random_numbers, stencil, samples_per_block, num_samples_list);
  
    // get current number of samples
    // Call random sampling kernel
    
    // TODO I am reusing the random values list (as a proof of concept), TODO make a 2nd randoms list for true randomness
    random_sampling_global<<<sampleGridSize, sampleBlockSize>>>(full_data, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_ratio, random_numbers, stencil, samples_per_block, num_samples_list);
    //random_sampling_single_thread_global<<<decisionGridSize, decisionBlockSize>>>(full_data, num_bins, max, min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_ratio, random_numbers, stencil, samples_per_block, num_samples_list);

  }
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());


}



/**
 * launch_nearest_neighbors:
 * Launch CUDA nearest neighbors reconstruction kernel.
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
void launch_nearest_neighbors(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Set grid and block dimensions so that each thread finds minimum distance of a single element
  dim3 grid(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 block(BLOCK, BLOCK, BLOCK);
  // Call nearest neighbors kernel
  nearest_neighbors_reconstruction_global<<<grid, block>>>(d_sample_data, d_sample_global_ids, num_samples, XDIM, YDIM, ZDIM, reconstructed_data);
  cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}



/**
 * launch_k_nearest_neighbors:
 * Launch CUDA k nearest neighbors reconstruction kernel.
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * K Neighbors controlled through constant: K_NEIGHBORS
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
void launch_k_nearest_neighbors(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Set grid and block dimensions so that each thread finds minimum distance of a single element
  dim3 grid(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 block(BLOCK, BLOCK, BLOCK);
  // Call nearest neighbors kernel
  k_nearest_neighbors_reconstruction_global<<<grid, block>>>(d_sample_data, d_sample_global_ids, num_samples, XDIM, YDIM, ZDIM, reconstructed_data);
  cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}



/**
 * launch_nearest_neighbors_shared:
 * Launch shared memory CUDA nearest neighbors reconstruction kernel.
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
void launch_nearest_neighbors_shared(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }

  // Set grid and block dimensions so that each thread finds minimum distance of a single element
  dim3 grid(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 block(BLOCK, BLOCK, BLOCK);
  // Determine amount of shared memory needed to hold coordinates of each sample
	size_t shared_memory_size = num_samples * 3 * sizeof(int);
  // Call nearest neighbors kernel
  nearest_neighbors_reconstruction_shared<<<grid, block, shared_memory_size>>>(d_sample_data, d_sample_global_ids, num_samples, XDIM, YDIM, ZDIM, reconstructed_data);
  cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}



/**
 * launch_k_nearest_neighbors_shared:
 * Launch shared memory CUDA k nearest neighbors reconstruction kernel.
 *
 * Input:
 * d_sample_data        - Sample data values
 * d_sample_global_ids  - Global Offset IDs of sample data values
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Original data dimensionality
 *
 * Output:
 * reconstructed_data   - Reconstructed data array
 *
 * Note:
 * K Neighbors controlled through constant: K_NEIGHBORS
 * Global ID = x + y * XDIM + z * XDIM * YDIM
 **/
void launch_k_nearest_neighbors_shared(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data){
  // Ensure block size is valid
  if (BLOCK*BLOCK*BLOCK >= 1024){
    std::cout << "INVALID BLOCK SIZE: " << BLOCK*BLOCK*BLOCK << "\n";
    exit(0);
  }
  
  // Set grid and block dimensions so that each thread finds minimum distance of a single element
  dim3 grid(std::ceil((float)XDIM/(float)BLOCK),std::ceil((float)YDIM/(float)BLOCK),std::ceil((float)ZDIM/(float)BLOCK));
  dim3 block(BLOCK, BLOCK, BLOCK);
  // Determine amount of shared memory needed to hold coordinates of each sample
	size_t shared_memory_size = num_samples * 3 * sizeof(int);
  // Call nearest neighbors kernel
  k_nearest_neighbors_reconstruction_shared<<<grid, block, shared_memory_size>>>(d_sample_data, d_sample_global_ids, num_samples, XDIM, YDIM, ZDIM, reconstructed_data);
  cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}
