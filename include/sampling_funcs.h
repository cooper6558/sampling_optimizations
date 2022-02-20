#ifndef SAMP_FUNCS_H
#define SAMP_FUNCS_H
#include "utils.h"


void configure_inputs(vector<string> filenames_list_sorted, int max_threads, int XDIM, int YDIM, int ZDIM, float sample_ratio, int* num_bins, float* error_threshold, int* XBLOCK, int* YBLOCK, int* ZBLOCK, float lifetime_min, float lifetime_max, double* bins_elapsed_seconds, double* err_elapsed_seconds, double* rdims_elapsed_seconds);
void get_block_data_w_global_ids(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_data_global_ids, vector<float> &block_data);
void omp_get_block_data_w_global_ids(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_data_global_ids, vector<float> &block_data);
void get_block_data_w_local_ids(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_data_local_ids, vector<float> &block_data);
void data_histogram(vector<float> &full_data, vector<int> &value_histogram, int nbins, float max, float min);
void omp_data_histogram(vector<float> &full_data, vector<int> &value_histogram, int num_bins, float max, float min);
void acceptance_function(vector<float> &full_data, int num_bins, float sample_ratio, vector<float> &acceptance_histogram, vector<int> &value_histogram, vector<float> &sampling_timers);
void acceptance_function_cuda_inputs(int nbins, int max_samples, float* acceptance_histogram, int* value_histogram);
void utilize_decision_histogram_based(vector<int> current_block_histogram, vector<int> reference_block_histogram, int nbins, int *utilize, int reuse_flag);
void utilize_decision_error_based(vector<float> current_block_data, int reference_samples_per_block, vector<int> reference_block_sample_ids, vector<float> reference_block_sample_data, int *utilize, int reuse_flag, float error_threshold, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK);
void value_histogram_importance_sampling(vector<float> &full_data, int num_blocks, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int num_bins, \
float data_max, float data_min, vector<float> &data_acceptance_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers);
void omp_value_histogram_importance_sampling(int num_threads, vector<float> &full_data, int num_blocks, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int num_bins, \
float data_max, float data_min, vector<float> &data_acceptance_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers);
void add_random_samples(vector<float> &full_data, int reuse_flag, int num_blocks, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, int max_samples, int *tot_samples);
int histogram_reuse_method(vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers);
int omp_histogram_reuse_method(int num_threads, vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers);
int error_reuse_method(vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers, float error_threshold, vector<int> ref_samples_per_block, vector<int> reference_sample_ids, vector<float> reference_sample_data);
int omp_error_reuse_method(int num_threads, vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers, float error_threshold, vector<int> ref_samples_per_block, vector<int> reference_sample_ids, vector<float> reference_sample_data);

// User Sampling Functions
void value_histogram_based_importance_sampling(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float sample_ratio, int num_bins, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers);
void omp_value_histogram_based_importance_sampling(int num_threads, vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float sample_ratio, int num_bins, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers);
int temporal_histogram_based_reuse_sampling(vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers);
int omp_temporal_histogram_based_reuse_sampling(int num_threads, vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers);
int temporal_error_based_reuse_sampling(vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &ref_samples_per_block, vector<int> &reference_sample_ids, vector<float> &reference_sample_data, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers, float error_threshold);
int omp_temporal_error_based_reuse_sampling(int num_threads, vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &ref_samples_per_block, vector<int> &reference_sample_ids, vector<float> &reference_sample_data, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers, float error_threshold);
// User Reconstruction Functions
void nearest_neighbors_reconstruction(vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data);
void omp_nearest_neighbors_reconstruction(int num_threads, vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data);
void k_nearest_neighbors_reconstruction(int k_samples, vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data);
void omp_k_nearest_neighbors_reconstruction(int num_threads, int k_samples, vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data);

// Unused Functions

#endif 