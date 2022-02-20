#ifndef UTILS_H
#define UTILS_H
// C Libraries
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cassert>
// File Libraries
#include <iostream> // I/O
#include <fstream>  // Input file
#include <ostream>  // Output file
// Cuda Libraries
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <curand.h>
#include <curand_kernel.h>
// OpenMP Libraries
#include <omp.h>
// Misc Libraries
#include <iomanip> // Set precision
#include <string>
#include <vector>
#include <chrono>   // timers
#include <numeric>  // std::iota
#include <regex> // for input file sorting
#include <random>


// Namespaces
using namespace std;
// #include <version>
#ifdef __cpp_lib_filesystem
#include <filesystem> // for looping over folder
namespace fs = std::filesystem;
#else
#include <experimental/filesystem> // for looping over folder
namespace fs = std::experimental::filesystem;
#endif



// Util Functions
void coalesce_samples(vector<float> current_timestep_samples, vector<int> current_sample_global_ids, vector<int> current_timestep_samples_per_block, \
vector<float> previous_timestep_samples, vector<int> previous_sample_global_ids, vector<int> previous_timestep_samples_per_block, vector<float> &coalesced_samples, \
vector<int> &coalesced_sample_global_ids, vector<int> &coalesced_samples_per_block);
void data_quality_analysis(vector<float> original_data, vector<float> reconstructed_data, int data_size, vector<double> &stats);
void save_vector_to_bin(int num_samples, int num_blocks, vector<int> sample_data_ids, vector<float> sample_data, vector<int> samples_per_block, int timestep, std::string output_folder_name);
void save_bin_to_vector(vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, int timestep, std::string input_folder_name);
int timestep_sort(std::string timestep_a, std::string timestep_b, std::regex rx);
int timestep_vector_sort(std::vector<std::string> timesteps, std::vector<std::string> &sorted_timesteps, std::regex rx);


// Cuda Macros
#define checkCudaErrors(ans) check( (ans), #ans, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line){
    if(err != cudaSuccess){
        std::cerr << "CUDA error at:: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
} 

template<typename T>
void printArray(T *arr, const int sz){

  for(int i = 0; i < sz; i++){ 
      if(i%9 == 0){ 
          std::cout <<"\n";
      }
     std::cout << arr[i] << "  "; 
  } 
  std::cout <<"\n";

}

#endif 
