#ifndef SAMP_KERN_H
#define SAMP_KERN_H
#include "utils.h"

void launch_rand_intialization(curandState_t* states, int XDIM, int YDIM, int ZDIM);
void launch_histogram(float* full_data, int* value_histogram, int nbins, int XDIM, int YDIM, int ZDIM, float max, float min);
void launch_importance_based_sampling(float* full_data, int nbins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float* acceptance_histogram, float* random_values, float* stencil, int* samples_per_block);
void launch_importance_based_sampling_method(float* full_data, float sample_ratio, int num_bins, float max, float min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float *acceptance_histogram, int *histogram_bin_ids, int *samples_per_bin, int *value_histogram, float *stencil, int *samples_per_block, curandState_t* states, float *random_numbers);
void launch_histogram_based_reuse_sampling_method(float* full_data, float sample_ratio, int num_bins, float data_max, float data_min, float lifetime_max, float lifetime_min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float *acceptance_histogram, int *histogram_bin_ids, int *samples_per_bin, int *value_histogram, float *stencil, int *samples_per_block, int *reference_histogram_list, int *current_histogram_list, int timestep, int* num_samples, curandState_t* states, float *random_numbers);
void launch_error_based_reuse_sampling_method(float* full_data, float sample_ratio, int num_bins, float data_max, float data_min, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float *acceptance_histogram, int *histogram_bin_ids, int *samples_per_bin, int *value_histogram, float *stencil, int *samples_per_block, int* reference_samples_per_block, int total_reference_samples,  int* reference_block_sample_ids, float* reference_block_sample_data, float* reference_block_errors, int* reference_block_errors_ids, int timestep, int *num_samples_list, float error_threshold, curandState_t* states, float *random_numbers);
void launch_nearest_neighbors(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data);
void launch_k_nearest_neighbors(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data);
void launch_nearest_neighbors_shared(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data);
void launch_k_nearest_neighbors_shared(float* d_sample_data, int *d_sample_global_ids, int num_samples, int XDIM, int YDIM, int ZDIM, float *reconstructed_data);

#endif 
