// User Libraries
#include "utils.h"
#include "sampling_kernels.h"
#include "sampling_funcs.h"

// #include <version>
#ifdef __cpp_lib_filesystem
#include <filesystem> // for looping over folder
namespace fs = std::filesystem;
#else
#include <experimental/filesystem> // for looping over folder
namespace fs = std::experimental::filesystem;
#endif


// TODO: Make Reuse Flag always -1

bool PRINT = false;
bool PYTHON = false;  // are we saving data binaries to be used with python (true) or c++ (false) reconstruction algorithm
int num_omp_threads = omp_get_max_threads();

/**
 * main:
 * Main function taht takes in various inputs, samples, reconstructs, and
 * gets the effects of sampling and reconstruction on time and quality.
 **/
int main(int argc, char* argv[]){
    // Begin random seed for later
    srand(time(0));

    ////////////////////////////////////////////////
    // Step 0: Get user input data and information//
    ////////////////////////////////////////////////
    // Setup input variables
    int input_data_regular_expression = 0;
    std::string input_folder;
    std::string input_data_dims_str;
    std::string input_region_dims_str;
    float sample_ratio;
    float lifetime_min;
    float lifetime_max;
    int num_bins = 0;
    int sampling_method = 0;
    int reconstruct_method = 0;
    std::string output_folder_name;
    std::string stats_file_name = "statistics.csv";

    int option_index;
    while((option_index = getopt (argc, argv, "h:e:i:d:b:p:m:x:n:s:r:o:f:")) != -1) {
        switch(option_index){
            case 'h':
                std::cout << "Interacitve Hybrid Sampling Example Help:" << std::endl;
                std::cout << "Use -e to specify input data regular expression" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "1: EXaAM Temperature Data" << std::endl;
                std::cout << "2: Hurricane Isabel Pressure Data" << std::endl;
                std::cout << "3: Asteroid Data" << std::endl;
                std::cout << "Use -i to specify input data folder Ex: input_data/" << std::endl;
                std::cout << "Use -d to specify input data dimensions Ex: 20,200,50" << std::endl;
                std::cout << "Use -b to specify desired region dimensions Ex: 4,20,10" << std::endl;
                std::cout << "Use -p to specify sample ratio Ex: 0.01" << std::endl;
                std::cout << "Use -m to specify lifetime expected minimum value Ex: 300.271" << std::endl;
                std::cout << "Use -x to specify lifetime expected maximum value Ex: 927.426" << std::endl;
                std::cout << "Use -n to specify desired number of histogram bins Ex: 20" << std::endl;
                std::cout << "Use -s to specify desired sampling method" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "1. - Importance Based Only (Serial)\n2. - Importance Based Only (OpenMP)\n3. - Importance Based Only (CUDA) \n4. - Histogram Based Reuse (Serial)\n5. - Histogram Based Reuse (OpenMP)\n6. - Histogram Based Reuse (CUDA)\n7. - Error Based Reuse (Serial)\n8. - Error Based Reuse (OpenMP)\n9. - Error Based Reuse (CUDA) (TODO)" << std::endl;
                std::cout << "Use -r to specify desired reconstruction method" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "1. - Nearest Neighbors (Serial)\n2. - Nearest Neighbors (OpenMP)\n3. - Nearest Neighbors (CUDA)\n4. - 3NN (Serial)\n5. - 3NN (OpenMP)\n6. - 3NN (CUDA)" << std::endl;
                std::cout << "Use -o to specify output folder Ex: output_data/" << std::endl;
                std::cout << "Use -f to specify output statistics CSV Ex: statistics.csv" << std::endl;
                break;
            case 'e':
                if(optarg){
                    input_data_regular_expression = std::atoi(optarg);
                }
                break;
            case 'i':
                if(optarg){
                    input_folder = optarg;
                }
                break;
            case 'd':
                if(optarg){
                    input_data_dims_str = optarg;
                }
                break;
            case 'b':
                if(optarg){
                    input_region_dims_str = optarg;
                }
                break;
            case 'p':
                if(optarg){
                    sample_ratio = std::atof(optarg);
                }
                break;
            case 'm':
                if(optarg){
                    lifetime_min = std::atof(optarg);
                }
                break;
            case 'x':
                if(optarg){
                    lifetime_max = std::atof(optarg);
                }
                break;
            case 'n':
                if(optarg){
                    num_bins = std::atoi(optarg);
                }
                break;
            case 's':
                if(optarg){
                    sampling_method = std::atoi(optarg);
                }
                break;
            case 'r':
                if(optarg){
                    reconstruct_method = std::atoi(optarg);
                }
                break;
            case 'o':
                if(optarg){
                    output_folder_name = optarg;
                }
                break;
            case 'f':
                if(optarg){
                    stats_file_name = optarg;
                }
                break;
            default: 
                std::cout << "Incorrect Options! Use -h * to view help information" << std::endl;
                exit(0);
        }
    }



    ////////////////////////////////////////////////////
    // Step 1: Massage user input data to correct form//
    ////////////////////////////////////////////////////
    
    // Determine regular expression to use
    std::string id_string;
    if (input_data_regular_expression == 1){
        // Set ID string for ExaAM temperature variable data set
        id_string = "plt_temperature_([0-9]+).bin";
    } else if (input_data_regular_expression == 2){
        // Set ID string for Hurricane Isabel pressure variable data set
        id_string = "Pf([0-9]+).binLE.raw_corrected_2_subsampled.vti.bin";
    } else if (input_data_regular_expression == 3){
        // Set ID string for zfs Hurricane Isabel pressure variable data set
        id_string = "Pf([0-9]+).bin";
    } else if (input_data_regular_expression == 4){
        // Set ID string for Asteroid vO2 variable data set
        id_string = "pv_insitu_300x300x300_([0-9]+).bin";
    } else {
        // Invalid regular expression, exit
        std::cout << "Incorrect Regular Expression Option! Use -h * to view help information" << std::endl;
        exit(0);
    }
    // Use ID to set correct regex
    std::regex rx(input_folder + id_string);
    
    // Parse dimensions from user input
    std::string delimiter = ",";
    std::string token;

    // Parse data dimensions
    token = input_data_dims_str.substr(0, input_data_dims_str.find(delimiter));
    int XDIM = std::stoi(token);
    input_data_dims_str.erase(0, input_data_dims_str.find(delimiter) + delimiter.length());
    token = input_data_dims_str.substr(0, input_data_dims_str.find(delimiter));
    int YDIM = std::stoi(token);
    input_data_dims_str.erase(0, input_data_dims_str.find(delimiter) + delimiter.length());
    token = input_data_dims_str.substr(0, input_data_dims_str.find(delimiter));
    int ZDIM = std::stoi(token);
    input_data_dims_str.erase(0, input_data_dims_str.find(delimiter) + delimiter.length());

    // Parse region dimensions
    token = input_region_dims_str.substr(0, input_region_dims_str.find(delimiter));
    int XBLOCK = std::stoi(token);
    input_region_dims_str.erase(0, input_region_dims_str.find(delimiter) + delimiter.length());
    token = input_region_dims_str.substr(0, input_region_dims_str.find(delimiter));
    int YBLOCK = std::stoi(token);
    input_region_dims_str.erase(0, input_region_dims_str.find(delimiter) + delimiter.length());
    token = input_region_dims_str.substr(0, input_region_dims_str.find(delimiter));
    int ZBLOCK = std::stoi(token);
    input_region_dims_str.erase(0, input_region_dims_str.find(delimiter) + delimiter.length());
   
    // Check Region Dimension Validity 
    if (XDIM % XBLOCK != 0 && YDIM % YBLOCK != 0 && ZDIM % ZBLOCK != 0){
        std::cout << "Invalid Region Dimensions, Use Evenly Divisible Region Dimensions";
        exit(0);
    }
    


    ///////////////////////////////////////////
    // Step 2: Gather Input Data Information //
    ///////////////////////////////////////////
    
    // Create full data vector
    vector<float> full_data(XDIM*YDIM*ZDIM);

    // Determine number of blocks
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get list of all input files in folder
    vector<std::string> filenames_list;
	std::cout << "Input Folder: " << input_folder << "\n";
    for (const auto & entry : fs::directory_iterator(input_folder)){
        // TODO put if statement here to make sure we only push the file name if it matches the rx
        filenames_list.push_back(entry.path());
    }

    // Determine number of timesteps and sort list
    int num_timesteps = filenames_list.size();
    std::vector<std::string> filenames_list_sorted(num_timesteps);
    int sorted_timesteps = timestep_vector_sort(filenames_list, filenames_list_sorted, rx);

    // Ensure timesteps sorted
    if (sorted_timesteps == 0){
        std::cout << "No Timesteps Found! Check Regular Expression!\n";
        return 0;
    } else {
        std::cout << sorted_timesteps << " Timesteps Found! Continuing!\n";
    }

    std::cout << "Inputs:" << std::endl;
    std::cout << input_data_regular_expression << std::endl;
    std::cout << input_folder << std::endl;
    std::cout << XDIM << " " << YDIM << " " << ZDIM << std::endl;
    std::cout << XBLOCK << " " << YBLOCK << " " << ZBLOCK << std::endl;
    std::cout << sample_ratio << std::endl;
    std::cout << lifetime_min << std::endl;
    std::cout << lifetime_max << std::endl;
    std::cout << num_bins << std::endl;
    std::cout << sampling_method << std::endl;
    std::cout << reconstruct_method << std::endl;
    std::cout << output_folder_name << std::endl;

    std::cout << "Timesteps:" << std::endl;
    for (uint i = 0; i < filenames_list_sorted.size(); i++){
        std::cout << filenames_list_sorted[i] << std::endl;
    }
    // float average_file_size_MB = 0;

    // Ensure output folder exists
    if (!fs::exists(output_folder_name)){
        std::cout << "WARNING: Output results folder does not exist . . . creating directory\n";
        
		fs::create_directory(output_folder_name); // create folder
    }

    ///////////////////////////////////
    // Step 3: Begin Sampling Process//
    ///////////////////////////////////
    
    // Setup variables
    int num_elements = 0;
    // int num_samples;
    vector<int> sample_data_ids;
    vector<float> sample_data;
    vector<int> samples_per_block;

    // Setup output vectors
    vector<float> total_sampling_process_times;
    vector<float> histogram_times;
    vector<float> sample_times;
    vector<float> total_reconstruction_times;
    vector<int> blocks_reused_list;
    vector<int> num_samples_list;
    vector<float> max_diff_list;
    vector<float> avg_diff_list;
    vector<float> PSNR_list;
    vector<float> SNR_list;

    /*

    // For histogram_based_reuse method
    vector<int> reference_histogram_list; 

    // For histogram_based_reuse method on gpu
    int reference_histogram_list_h[num_bins*num_blocks] = {};
    //int current_histogram_list_h[num_bins*num_blocks] = {};

    // For error_based_reuse method
    vector<int> reference_sample_ids;
    vector<float> reference_sample_data;
    vector<int> ref_samples_per_block;
    
    // For each timestep, sample each input file using the corresponding sampling method
	for (int timestep = 0; timestep < num_timesteps; timestep++){
        // Reset each vector for each loop
        sample_data_ids.resize(0,0);
        sample_data.resize(0,0);
        samples_per_block.resize(0,0);

        std::cout << "\n/-*******************************************-/\nTimestep " << timestep << ": " << filenames_list_sorted[timestep] << std::endl;

        // Open corresponding timestep file
        std::ifstream fin(filenames_list_sorted[timestep], std::ios::binary);
        if(!fin){
            std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[timestep] << "\n";
            exit(0);
        }
        
        // Determine number of elements
        fin.seekg(0, std::ios::end);
        int file_size_bytes = fin.tellg();
        num_elements = fin.tellg() / sizeof(float);
        fin.seekg(0, std::ios::beg);

        average_file_size_MB = average_file_size_MB + (file_size_bytes / (1e+6));

        // Check Data Dimension Validity
        if (XDIM*YDIM*ZDIM != num_elements){
            std::cout << "Invalid Data Dimensions..." << XDIM*YDIM*ZDIM << " vs " << num_elements << "\n";
            exit(0);
        }

        // Read timestep data into data vector
        fin.read(reinterpret_cast<char*>(&full_data[0]), num_elements*sizeof(float));

        // Sampling Timers
        vector<float> sampling_timers;

        // Sample using correct sampling process
        if (sampling_method == 1){
            // importance based sampling - Serial Version 
            auto serial_start = std::chrono::steady_clock::now();
            value_histogram_based_importance_sampling(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_ratio, num_bins, sample_data_ids, sample_data, samples_per_block, sampling_timers);
            auto serial_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
            std::cout << "\nSerial Sampling: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

            // save timers
            histogram_times.push_back(sampling_timers[0]);
            sample_times.push_back(sampling_timers[1]);
            total_sampling_process_times.push_back(sampling_timers[2]);


            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            blocks_reused_list.push_back(0);
        } else if (sampling_method == 2){
            //importance based sampling - OpenMP Version 
            auto serial_start = std::chrono::steady_clock::now();
            omp_value_histogram_based_importance_sampling(num_omp_threads, full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_ratio, num_bins, sample_data_ids, sample_data, samples_per_block, sampling_timers);
            auto serial_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
            std::cout << "\nOpenMP Sampling: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

            // save timers
            histogram_times.push_back(sampling_timers[0]);
            sample_times.push_back(sampling_timers[1]);
            total_sampling_process_times.push_back(sampling_timers[2]);


            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            blocks_reused_list.push_back(0);
        } else if (sampling_method == 3){
            //importance based sampling - GPU Version 
            // Start Profiling
            cudaProfilerStart();

            // Move data from vector to float arrays to put on GPU
            float full_data_h[num_elements];
            std::copy(full_data.begin(), full_data.end(), full_data_h);

            // Allocate array of random values
            //float random_values_h[num_elements];
            //for (int i = 0; i < num_elements ; i++){
            //    random_values_h[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            //}

            // Generate random numbers equal to entire data set
            std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
            std::uniform_real_distribution<double> distribution(0.0,1.0);
            float random_values_h[num_elements];
            for (int i = 0; i < num_elements ; i++){
                double r = distribution(generator);
                // Store in random number vector
                random_values_h[i] = r;
            }

            // Get current min and max for GPU function
            float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
            float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

            // Allocate device arrays
            float *full_data_d;
            float *random_values_d;
            float *acceptance_histogram_d;
            int *histogram_bin_ids_d;
            int *samples_per_bin_d;
            int *value_histogram_d;
            float *stencil_d;
            int *samples_per_block_d;

            checkCudaErrors(cudaMalloc((void**)&full_data_d, sizeof(float)*num_elements));
            checkCudaErrors(cudaMalloc((void**)&random_values_d, sizeof(float)*num_elements));
            checkCudaErrors(cudaMalloc((void**)&acceptance_histogram_d, sizeof(float)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&histogram_bin_ids_d, sizeof(int)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&samples_per_bin_d, sizeof(int)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&value_histogram_d, sizeof(int)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&stencil_d, sizeof(float)*num_elements));
            checkCudaErrors(cudaMalloc((void**)&samples_per_block_d, sizeof(int)*num_blocks));

            // Move data into device arrays
            checkCudaErrors(cudaMemcpy(full_data_d, full_data_h, sizeof(float)*num_elements, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(random_values_d, random_values_h, sizeof(float)*num_elements, cudaMemcpyHostToDevice));

            // Initialize other device arrays to zeros
            checkCudaErrors(cudaMemset(acceptance_histogram_d, 0, sizeof(float)*num_bins));
            checkCudaErrors(cudaMemset(histogram_bin_ids_d, 0, sizeof(int)*num_bins)); 
            checkCudaErrors(cudaMemset(samples_per_bin_d, 0, sizeof(int)*num_bins)); 
            checkCudaErrors(cudaMemset(value_histogram_d, 0, sizeof(int)*num_bins));
            checkCudaErrors(cudaMemset(stencil_d, 0, sizeof(int)*num_elements));
            checkCudaErrors(cudaMemset(samples_per_block_d, 0, sizeof(int)*num_blocks));

            // Launch CUDA kernel
            auto timer_start = std::chrono::steady_clock::now();
            launch_importance_based_sampling_method(full_data_d, random_values_d, sample_ratio, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram_d, histogram_bin_ids_d, samples_per_bin_d, value_histogram_d, stencil_d, samples_per_block_d);
            auto timer_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = timer_end-timer_start;
            total_sampling_process_times.push_back(elapsed_seconds.count()*(1e+6)); //us

            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());

            // Copy the final stencil and samples per block back to host
            float stencil_h[num_elements];
            int samples_per_block_h[num_blocks];
            checkCudaErrors(cudaMemcpy(stencil_h, stencil_d, sizeof(float)*num_elements, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(samples_per_block_h, samples_per_block_d, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost));

            // Get samples from stencil
            for (int global_id = 0; global_id < num_elements; global_id++){
                if (stencil_h[global_id] > 0){
                    sample_data.push_back(full_data_h[global_id]);
                    sample_data_ids.push_back(global_id);

                  
                }
            }

            // Copy int array to int vector
            samples_per_block.assign(samples_per_block_h, samples_per_block_h+num_blocks);

            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            int blocks_reused = count(samples_per_block.begin(), samples_per_block.end(), -1);
            blocks_reused_list.push_back(blocks_reused);
            std::cout << blocks_reused << " blocks reused!\n";

            // Free cuda arrays
            cudaFree(full_data_d);
            cudaFree(random_values_d);
            cudaFree(acceptance_histogram_d);
            cudaFree(histogram_bin_ids_d);
            cudaFree(samples_per_bin_d);
            cudaFree(value_histogram_d);
            cudaFree(stencil_d);
            cudaFree(samples_per_block_d);

            // Stop Profiling
            cudaProfilerStop();

        } else if (sampling_method == 4){
            // histogram based reuse - Serial Version 
            auto serial_start = std::chrono::steady_clock::now();
            int blocks_reused = temporal_histogram_based_reuse_sampling(full_data, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);
            auto serial_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
            std::cout << "\nSerial Sampling: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

            // save timers
            histogram_times.push_back(sampling_timers[0]);
            sample_times.push_back(sampling_timers[1]);
            total_sampling_process_times.push_back(sampling_timers[2]);

            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            blocks_reused_list.push_back(blocks_reused);
            std::cout << blocks_reused << " blocks reused!\n";
        } else if (sampling_method == 5){
            // histogram based reuse - OpenMP Version 
            auto serial_start = std::chrono::steady_clock::now();
            int blocks_reused = omp_temporal_histogram_based_reuse_sampling(num_omp_threads, full_data, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);
            auto serial_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
            std::cout << "\nOpenMP Sampling: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

            // save timers
            histogram_times.push_back(sampling_timers[0]);
            sample_times.push_back(sampling_timers[1]);
            total_sampling_process_times.push_back(sampling_timers[2]);

            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            blocks_reused_list.push_back(blocks_reused);
            std::cout << blocks_reused << " blocks reused!\n";
        
        } else if (sampling_method == 6){
            //histogram based reuse - GPU Version 


            // Start Profiling
            cudaProfilerStart();

            // Move data from vector to float arrays to put on GPU
            float full_data_h[num_elements];
            std::copy(full_data.begin(), full_data.end(), full_data_h);

            // Allocate array of random values
            //float random_values_h[num_elements];
            //for (int i = 0; i < num_elements ; i++){
            //    random_values_h[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            //}

            // Generate random numbers equal to entire data set
            std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
            std::uniform_real_distribution<double> distribution(0.0,1.0);
            float random_values_h[num_elements];
            for (int i = 0; i < num_elements ; i++){
                double r = distribution(generator);
                // Store in random number vector
                random_values_h[i] = r;
            }

            // Get current min and max for GPU function
            float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
            float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

            // Allocate device arrays
            float *full_data_d;
            float *random_values_d;
            float *acceptance_histogram_d;
            int *histogram_bin_ids_d;
            int *samples_per_bin_d;
            int *value_histogram_d;
            float *stencil_d;
            int *samples_per_block_d;
            int *reference_histogram_list_d;
            int *current_histogram_list_d;
            int *num_samples_d;

            checkCudaErrors(cudaMalloc((void**)&full_data_d, sizeof(float)*num_elements));
            checkCudaErrors(cudaMalloc((void**)&random_values_d, sizeof(float)*num_elements));
            checkCudaErrors(cudaMalloc((void**)&acceptance_histogram_d, sizeof(float)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&histogram_bin_ids_d, sizeof(int)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&samples_per_bin_d, sizeof(int)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&value_histogram_d, sizeof(int)*num_bins));
            checkCudaErrors(cudaMalloc((void**)&stencil_d, sizeof(float)*num_elements));
            checkCudaErrors(cudaMalloc((void**)&samples_per_block_d, sizeof(int)*num_blocks));
            checkCudaErrors(cudaMalloc((void**)&reference_histogram_list_d, sizeof(int)*num_bins*num_blocks));
            checkCudaErrors(cudaMalloc((void**)&current_histogram_list_d, sizeof(int)*num_bins*num_blocks));
            checkCudaErrors(cudaMalloc((void**)&num_samples_d, sizeof(int)*2));

            // Move data into device arrays
            checkCudaErrors(cudaMemcpy(full_data_d, full_data_h, sizeof(float)*num_elements, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(random_values_d, random_values_h, sizeof(float)*num_elements, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(reference_histogram_list_d, reference_histogram_list_h, sizeof(int)*num_bins*num_blocks, cudaMemcpyHostToDevice));

            // Initialize other device arrays to zeros
            checkCudaErrors(cudaMemset(acceptance_histogram_d, 0, sizeof(float)*num_bins));
            checkCudaErrors(cudaMemset(histogram_bin_ids_d, 0, sizeof(int)*num_bins)); 
            checkCudaErrors(cudaMemset(samples_per_bin_d, 0, sizeof(int)*num_bins)); 
            checkCudaErrors(cudaMemset(value_histogram_d, 0, sizeof(int)*num_bins));
            checkCudaErrors(cudaMemset(stencil_d, 0, sizeof(int)*num_elements));
            checkCudaErrors(cudaMemset(samples_per_block_d, 0, sizeof(int)*num_blocks));
            checkCudaErrors(cudaMemset(current_histogram_list_d, 0, sizeof(int)*num_bins*num_blocks));
            checkCudaErrors(cudaMemset(num_samples_d, 0, sizeof(int)*2));

            // Launch CUDA kernel
            auto timer_start = std::chrono::steady_clock::now();
            launch_histogram_based_reuse_sampling_method(full_data_d, random_values_d, sample_ratio, num_bins, data_max, data_min, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, acceptance_histogram_d, histogram_bin_ids_d, samples_per_bin_d, value_histogram_d, stencil_d, samples_per_block_d, reference_histogram_list_d, current_histogram_list_d, timestep, num_samples_d);
            auto timer_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = timer_end-timer_start;
            total_sampling_process_times.push_back(elapsed_seconds.count()*(1e+6)); //us

            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());

            // Copy the final stencil and samples per block back to host
            float stencil_h[num_elements];
            int samples_per_block_h[num_blocks];
            checkCudaErrors(cudaMemcpy(stencil_h, stencil_d, sizeof(float)*num_elements, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(samples_per_block_h, samples_per_block_d, sizeof(int)*num_blocks, cudaMemcpyDeviceToHost));

            int num_samples_h[2]; 
            checkCudaErrors(cudaMemcpy(num_samples_h, num_samples_d, sizeof(int)*2, cudaMemcpyDeviceToHost));


            // update ref hist with val hist
            checkCudaErrors(cudaMemcpy(reference_histogram_list_h, current_histogram_list_d, sizeof(int)*num_bins*num_blocks, cudaMemcpyDeviceToHost));

            // Get samples from stencil
            for (int global_id = 0; global_id < num_elements; global_id++){
                if (stencil_h[global_id] > 0){
                    sample_data.push_back(full_data_h[global_id]);
                    sample_data_ids.push_back(global_id);

                
                }
            }

            // Copy int array to int vector
            samples_per_block.assign(samples_per_block_h, samples_per_block_h+num_blocks);

            std::cout << "Originally Taken " << num_samples_h[0] << " samples\n";
            std::cout << "Used Random Sampling to take " << num_samples_h[1] << " more samples\n";


            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            int blocks_reused = count(samples_per_block.begin(), samples_per_block.end(), -1);
            blocks_reused_list.push_back(blocks_reused);
            std::cout << blocks_reused << " blocks reused!\n";

            if ((num_samples_h[0] +  num_samples_h[1] != num_samples) && timestep != 0){
                std::cout << "Error in number of samples logic\n";
                exit(0);
            }

            // Free cuda arrays
            cudaFree(full_data_d);
            cudaFree(random_values_d);
            cudaFree(acceptance_histogram_d);
            cudaFree(histogram_bin_ids_d);
            cudaFree(samples_per_bin_d);
            cudaFree(value_histogram_d);
            cudaFree(stencil_d);
            cudaFree(samples_per_block_d);
            cudaFree(reference_histogram_list_d);
            cudaFree(num_samples_d);

            // Stop Profiling
            cudaProfilerStop();

            
        } else if (sampling_method == 7){
            /// error based reuse - Serial Version 
            float error_threshold = 2.67; // TODO user input

            auto serial_start = std::chrono::steady_clock::now();
            int blocks_reused = temporal_error_based_reuse_sampling(full_data, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, ref_samples_per_block, reference_sample_ids, reference_sample_data, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers, error_threshold);
            auto serial_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
            std::cout << "\nSerial Sampling: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

            // save timers
            histogram_times.push_back(sampling_timers[0]);
            sample_times.push_back(sampling_timers[1]);
            total_sampling_process_times.push_back(sampling_timers[2]);

            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            blocks_reused_list.push_back(blocks_reused);
            std::cout << blocks_reused << " blocks reused!\n";

            // reset reference for next loop
            ref_samples_per_block.resize(0,0);
            reference_sample_ids.resize(0,0);
            reference_sample_data.resize(0,0);
            ref_samples_per_block = samples_per_block;
            reference_sample_ids = sample_data_ids;
            reference_sample_data = sample_data;

        } else if (sampling_method == 8){
            // error based reuse - OpenMP Version 
            float error_threshold = 2.67; // TODO user input

            auto serial_start = std::chrono::steady_clock::now();
            int blocks_reused = omp_temporal_error_based_reuse_sampling(num_omp_threads, full_data, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, ref_samples_per_block, reference_sample_ids, reference_sample_data, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers, error_threshold);
            auto serial_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
            std::cout << "\nOpenMP Sampling: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";

            // save timers
            histogram_times.push_back(sampling_timers[0]);
            sample_times.push_back(sampling_timers[1]);
            total_sampling_process_times.push_back(sampling_timers[2]);

            // Determine number of samples
            num_samples = sample_data_ids.size();
            std::cout << "True Taken : " << num_samples << " samples\n";
            num_samples_list.push_back(num_samples);
            blocks_reused_list.push_back(blocks_reused);
            std::cout << blocks_reused << " blocks reused!\n";

            // reset reference for next loop
            ref_samples_per_block.resize(0,0);
            reference_sample_ids.resize(0,0);
            reference_sample_data.resize(0,0);
            ref_samples_per_block = samples_per_block;
            reference_sample_ids = sample_data_ids;
            reference_sample_data = sample_data;

        } else {
            std::cout << "Invalid Sampling Method! Use -h * to view help information" << std::endl;
        }

        // Store nessecary samples to output file
        save_vector_to_bin(sample_data_ids.size(), num_blocks, sample_data_ids, sample_data, samples_per_block, timestep, output_folder_name);
    }

    average_file_size_MB = average_file_size_MB / num_timesteps;
    */

    /////////////////////////////////////////
    // Step 3: Begin Reconstruction Process//
    /////////////////////////////////////////

    if (reconstruct_method != 0){
        for (int current_timestep = 0; current_timestep < num_timesteps; current_timestep++){
            std::cout << "\n/*******************************************/\nTimestep: " << current_timestep << std::endl;

            // Gather sample data from binary files
            if (sampling_method == 3 || sampling_method == 4){
                if (current_timestep != 0){
                    // Get sample data for current timestep
                    vector<int> current_timestep_sample_data_ids;
                    vector<float> current_timestep_sample_data;
                    vector<int> current_timestep_samples_per_block;
                    save_bin_to_vector(current_timestep_sample_data_ids, current_timestep_sample_data, current_timestep_samples_per_block, current_timestep, output_folder_name);
                    
                    // Get sample data for previous timestep
                    int prev_timestep = current_timestep - 1; 
                    vector<int> previous_timestep_sample_data_ids;
                    vector<float> previous_timestep_sample_data;
                    vector<int> previous_timestep_samples_per_block;
                    save_bin_to_vector(previous_timestep_sample_data_ids, previous_timestep_sample_data, previous_timestep_samples_per_block, prev_timestep, output_folder_name);

                    // Put sample data together
                    vector<float> coalesced_samples;
                    vector<int> coalesced_sample_data_ids;
                    vector<int> coalesced_samples_per_block;

                    coalesce_samples(current_timestep_sample_data, current_timestep_sample_data_ids, current_timestep_samples_per_block, previous_timestep_sample_data, previous_timestep_sample_data_ids, previous_timestep_samples_per_block, coalesced_samples, coalesced_sample_data_ids, coalesced_samples_per_block);
                    
                    // Save coalesced sample data for later use
                    int num_samples_coalesced = coalesced_samples.size();
                    save_vector_to_bin(num_samples_coalesced, num_blocks, coalesced_sample_data_ids, coalesced_samples, coalesced_samples_per_block, current_timestep, output_folder_name); // note: this will overwrite the flagged samples data
                }
            }
            
            // Read in current timestep sample data
            vector<int> reconstructed_sample_ids;
            vector<float> reconstructed_sample_data;
            vector<int> reconstructed_samples_per_block;
            save_bin_to_vector(reconstructed_sample_ids, reconstructed_sample_data, reconstructed_samples_per_block, current_timestep, output_folder_name);
            int num_sample_ids = reconstructed_sample_ids.size();

            // Reconstruct using correct method
            if (reconstruct_method == 1){
                // Serial Nearest Neighbors Reconstruction
                vector<float> reconstructed_data;
                auto serial_start = std::chrono::steady_clock::now();
                nearest_neighbors_reconstruction(reconstructed_sample_ids, reconstructed_sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                auto serial_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
                std::cout << "\nSerial Reconstruction: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";
                total_reconstruction_times.push_back(serial_elapsed_seconds.count()*(1e+6));

                // Write out reconstructed data
                std::ofstream fout_data_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
                float f_output;
                for(int i = 0; i < XDIM*YDIM*ZDIM; ++i){
                    f_output = reconstructed_data[i];
                    fout_data_recons.write(reinterpret_cast<const char*>(&f_output), sizeof(f_output));
                }
                fout_data_recons.close();
                std::cout << "Successfully Created Output File:" << output_folder_name << "reconstructed_data.bin\n";

            } else if (reconstruct_method == 2){
                std::cout << "Using " << num_omp_threads << " Threads to Reconstruct!" << std::endl;

                // OMP Nearest Neighbors Reconstruction
                vector<float> reconstructed_data;
                auto serial_start = std::chrono::steady_clock::now();
                omp_nearest_neighbors_reconstruction(num_omp_threads, reconstructed_sample_ids, reconstructed_sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                auto serial_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
                std::cout << "\nSerial Reconstruction: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";
                total_reconstruction_times.push_back(serial_elapsed_seconds.count()*(1e+6));

                // Write out reconstructed data
                std::ofstream fout_data_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
                float f_output;
                for(int i = 0; i < XDIM*YDIM*ZDIM; ++i){
                    f_output = reconstructed_data[i];
                    fout_data_recons.write(reinterpret_cast<const char*>(&f_output), sizeof(f_output));
                }
                fout_data_recons.close();
                std::cout << "Successfully Created Output File:" << output_folder_name << "reconstructed_data.bin\n";

            } else if (reconstruct_method == 3){
                // Start Profiling
                cudaProfilerStart();

                // CUDA Nearest Neighbors Reconstruction
                float sample_data_h[num_sample_ids];
                int sample_ids_h[num_sample_ids]; 
                std::copy(reconstructed_sample_ids.begin(), reconstructed_sample_ids.end(), sample_ids_h); // fill full_data array with vector inputs
                std::copy(reconstructed_sample_data.begin(), reconstructed_sample_data.end(), sample_data_h); // fill full_data array with vector inputs
                float reconstructed_data[num_elements];
                
                // allocate the memories for the device pointers
                float* sample_data_d;
                int* sample_ids_d; 
                float *reconstructed_data_d;
                
                checkCudaErrors(cudaMalloc((void**)&sample_data_d, sizeof(float)*num_sample_ids));
                checkCudaErrors(cudaMalloc((void**)&sample_ids_d, sizeof(float)*num_sample_ids));
                checkCudaErrors(cudaMalloc((void**)&reconstructed_data_d, sizeof(float)*num_elements));
                // move data over to device
                checkCudaErrors(cudaMemcpy(sample_data_d, sample_data_h, sizeof(float)*num_sample_ids, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(sample_ids_d, sample_ids_h, sizeof(float)*num_sample_ids, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemset(reconstructed_data_d, 0, sizeof(int)*num_elements)); //init with zeros
                
                // kernel launch code
                auto cuda_start = std::chrono::steady_clock::now();
                launch_nearest_neighbors(sample_data_d, sample_ids_d, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data_d);
                //launch_nearest_neighbors_shared(sample_data_d, sample_ids_d, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data_d);
                cudaDeviceSynchronize();
                checkCudaErrors(cudaGetLastError());
                auto cuda_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> cuda_elapsed_seconds = cuda_end-cuda_start;
                std::cout << "\nCUDA Reconstruction: " << cuda_elapsed_seconds.count()*(1e+6) << " us\n";
                total_reconstruction_times.push_back(cuda_elapsed_seconds.count()*(1e+6));

                // memcpy the final answer output to the host side.
                checkCudaErrors(cudaMemcpy(reconstructed_data, reconstructed_data_d, sizeof(int)*num_elements, cudaMemcpyDeviceToHost));

                // write out reconstructed data to binary file
                std::ofstream fout_data_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
                float f_output;
                for(int i = 0; i < XDIM*YDIM*ZDIM; ++i){
                    f_output = reconstructed_data[i];
                    fout_data_recons.write(reinterpret_cast<const char*>(&f_output), sizeof(f_output));
                }
                fout_data_recons.close();
                std::cout << "Successfully Created Output File:" << output_folder_name << "reconstructed_data.bin\n";

                // Free device data
                cudaFree(sample_data_d);
                cudaFree(sample_ids_d);
                cudaFree(reconstructed_data_d);

                // Stop Profiling
                cudaProfilerStop();

            } else if (reconstruct_method == 4){
                // Serial K Nearest Neighbors
                vector<float> reconstructed_data;
                auto serial_start = std::chrono::steady_clock::now();
                k_nearest_neighbors_reconstruction(3, reconstructed_sample_ids, reconstructed_sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                auto serial_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
                std::cout << "\nSerial Reconstruction: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";
                total_reconstruction_times.push_back(serial_elapsed_seconds.count()*(1e+6));

                // write out reconstructed data to binary file
                std::ofstream fout_data_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
                float f_output;
                for(int i = 0; i < XDIM*YDIM*ZDIM; ++i){
                    f_output = reconstructed_data[i];
                    fout_data_recons.write(reinterpret_cast<const char*>(&f_output), sizeof(f_output));
                }
                fout_data_recons.close();
                std::cout << "Successfully Created Output File: " << output_folder_name<< "reconstructed_data.bin\n";

            } else if (reconstruct_method == 5){
                std::cout << "Using " << num_omp_threads << " Threads to Reconstruct!" << std::endl;

                // Serial K Nearest Neighbors
                vector<float> reconstructed_data;
                auto serial_start = std::chrono::steady_clock::now();
                omp_k_nearest_neighbors_reconstruction(num_omp_threads, 3, reconstructed_sample_ids, reconstructed_sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                auto serial_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> serial_elapsed_seconds = serial_end-serial_start;
                std::cout << "\nSerial Reconstruction: " << serial_elapsed_seconds.count()*(1e+6) << " us\n";
                total_reconstruction_times.push_back(serial_elapsed_seconds.count()*(1e+6));

                // write out reconstructed data to binary file
                std::ofstream fout_data_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
                float f_output;
                for(int i = 0; i < XDIM*YDIM*ZDIM; ++i){
                    f_output = reconstructed_data[i];
                    fout_data_recons.write(reinterpret_cast<const char*>(&f_output), sizeof(f_output));
                }
                fout_data_recons.close();
                std::cout << "Successfully Created Output File: " << output_folder_name << "reconstructed_data.bin\n";

            } else if (reconstruct_method == 6){
                // Start Profiling
                cudaProfilerStart();

                // CUDA K Nearest Neighbors Reconstruction
                float sample_data_h[num_sample_ids];
                int sample_ids_h[num_sample_ids]; 
                std::copy(reconstructed_sample_ids.begin(), reconstructed_sample_ids.end(), sample_ids_h); // fill full_data array with vector inputs
                std::copy(reconstructed_sample_data.begin(), reconstructed_sample_data.end(), sample_data_h); // fill full_data array with vector inputs
                float reconstructed_data[num_elements];
                
                // allocate the memories for the device pointers
                float* sample_data_d;
                int* sample_ids_d; 
                float *reconstructed_data_d;
                
                checkCudaErrors(cudaMalloc((void**)&sample_data_d, sizeof(float)*num_sample_ids));
                checkCudaErrors(cudaMalloc((void**)&sample_ids_d, sizeof(float)*num_sample_ids));
                checkCudaErrors(cudaMalloc((void**)&reconstructed_data_d, sizeof(float)*num_elements));
                // move data over to device
                checkCudaErrors(cudaMemcpy(sample_data_d, sample_data_h, sizeof(float)*num_sample_ids, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(sample_ids_d, sample_ids_h, sizeof(float)*num_sample_ids, cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemset(reconstructed_data_d, 0, sizeof(int)*num_elements)); //init with zeros
                
                // kernel launch code
                auto cuda_start = std::chrono::steady_clock::now();
                launch_k_nearest_neighbors(sample_data_d, sample_ids_d, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data_d);
                //launch_k_nearest_neighbors_shared(sample_data_d, sample_ids_d, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data_d);
                cudaDeviceSynchronize();
                checkCudaErrors(cudaGetLastError());
                auto cuda_end = std::chrono::steady_clock::now();
                std::chrono::duration<double> cuda_elapsed_seconds = cuda_end-cuda_start;
                std::cout << "\nCUDA Reconstruction: " << cuda_elapsed_seconds.count()*(1e+6) << " us\n";
                total_reconstruction_times.push_back(cuda_elapsed_seconds.count()*(1e+6));

                // memcpy the final answer output to the host side.
                checkCudaErrors(cudaMemcpy(reconstructed_data, reconstructed_data_d, sizeof(int)*num_elements, cudaMemcpyDeviceToHost));

                // write out reconstructed data to binary file
                std::ofstream fout_data_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
                float f_output;
                for(int i = 0; i < XDIM*YDIM*ZDIM; ++i){
                    f_output = reconstructed_data[i];
                    fout_data_recons.write(reinterpret_cast<const char*>(&f_output), sizeof(f_output));
                }
                fout_data_recons.close();
                std::cout << "Successfully Created Output File: " << output_folder_name << "reconstructed_data.bin\n";

                // Free device data
                cudaFree(sample_data_d);
                cudaFree(sample_ids_d);
                cudaFree(reconstructed_data_d);

                // Stop Profiling
                cudaProfilerStop();

            } else {
                std::cout << "Invalid Reconstruction Method! Use -h * to view help information" << std::endl;
            }

            // Post Reconstruction Analysis
            // read in true values
            std::ifstream fin(filenames_list_sorted[current_timestep], std::ios::binary);
            if(!fin)
            {
                std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[current_timestep] << "\n";
                return 0;
            }
            // Determine number of floats in input data
            fin.seekg(0, std::ios::end);
            int num_elements_input = fin.tellg() / sizeof(float);
            fin.seekg(0, std::ios::beg);

            // Create vector and read floats in
            vector<float> full_data_input(num_elements_input); // input timestep data
            fin.read(reinterpret_cast<char*>(&full_data_input[0]), num_elements_input*sizeof(float));
            fin.close();

            // read in reconstructed dataset
            std::ifstream fin_recons(output_folder_name+"reconstructed_data.bin", std::ios::binary);
            if(!fin_recons)
            {
                std::cout << " Error, Couldn't find the file: " << output_folder_name << "reconstructed_data.bin" << "\n";
                return 0;
            }
            // Determine number of floats in input data
            fin_recons.seekg(0, std::ios::end);
            num_elements_input = fin_recons.tellg() / sizeof(float);
            fin_recons.seekg(0, std::ios::beg);

            // Create vector and read floats in
            vector<float> full_data_recons(num_elements_input); // input timestep data
            fin_recons.read(reinterpret_cast<char*>(&full_data_recons[0]), num_elements_input*sizeof(float));

            // TODO quality analysis in cuda?

            vector<double> stats;
            data_quality_analysis(full_data_input, full_data_recons, num_elements_input, stats);
            max_diff_list.push_back(stats[0]);
            avg_diff_list.push_back(stats[1]);
            PSNR_list.push_back(stats[2]);
            SNR_list.push_back(stats[3]);
        }
    }


    std::cout << "Finished Experiments! Writing to file . . .\n";

    /* write results to csv */
    std::ofstream stats_file;
    stats_file.open (stats_file_name); // write to file
    stats_file << "Timestep,Sample Method,File Size (MB),Sample Ratio,Region Dims,Number of Bins,Reconstruction Method,Sampling Bandwidth (MB/s),Total (us),Histogram (us),Sample (us),Recons (us),Recons Bandwidth (MB/s),Blocks Reused,Number of Samples,Max Diff,Avg Diff,PSNR,SNR\n";

    /* Write to Individual Timestep Info File */
    for (int timestep = 0; timestep < num_timesteps; timestep++){

        if (sampling_method == 3 || sampling_method == 6 || sampling_method == 9){ // if we are a GPU sampling method, we do not have histogram or sampling times
            if (reconstruct_method == 0){
                // No Reconstruction -> No statistics values
                stats_file << timestep << "," << std::to_string(sampling_method) << "," << "NA" << "," << std::to_string(sample_ratio) << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins) << ","<< "NA" << "," << "NA" << "," << "NA" << "," <<  "NA" << "," << "NA" << "," << "NA" << "," << "NA" << "," <<  "NA" << "," <<  "NA" << "," << "NA"<< "," << "NA" << "," <<  "NA" << "," << "NA" << "\n";
            } else{
                stats_file << timestep << "," << std::to_string(sampling_method) << "," << "NA" << "," << std::to_string(sample_ratio) << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins)  << ","<< std::to_string(reconstruct_method) << "," << "NA" << "," << "NA" << "," << "NA" << "," << "NA" << "," << total_reconstruction_times[timestep] << "," << "NA" << "," << "NA" << "," <<  "NA" << "," << max_diff_list[timestep] << "," << avg_diff_list[timestep] << "," << PSNR_list[timestep] << "," << SNR_list[timestep] << "\n";
            }
        }else{
            if (reconstruct_method == 0){
                // No Reconstruction -> No statistics values
                stats_file << timestep << "," << std::to_string(sampling_method) << "," << "NA" << "," << std::to_string(sample_ratio) << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins) << ","<< "NA" << "," << "NA" << "," << "NA" << "," <<  "NA" << "," << "NA" << "," << "NA" << "," << "NA" << "," <<  "NA" << "," <<  "NA" << "," << "NA"<< "," << "NA" << "," <<  "NA" << "," << "NA" << "\n";
            } else{
                stats_file << timestep << "," << std::to_string(sampling_method) << "," << "NA" << "," << std::to_string(sample_ratio) << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins)  << ","<< std::to_string(reconstruct_method) << "," << "NA" << "," << "NA" << "," << "NA" << "," << "NA" << "," << total_reconstruction_times[timestep] << "," << "NA" << "," << "NA" << "," <<  "NA" << "," << max_diff_list[timestep] << "," << avg_diff_list[timestep] << "," << PSNR_list[timestep] << "," << SNR_list[timestep] << "\n";
            }
        }
    }
    


    /* Calculate Averages and Save */
    /*
    float average_total_sampling_process_times = (float)(std::accumulate(total_sampling_process_times.begin(), total_sampling_process_times.end(), 0)) / total_sampling_process_times.size();
    float stddev_total_sampling_process_times = (float)std::sqrt((std::inner_product(total_sampling_process_times.begin(), total_sampling_process_times.end(), total_sampling_process_times.begin(), 0.0)) / total_sampling_process_times.size() - average_total_sampling_process_times * average_total_sampling_process_times);

    float average_total_reconstruction_times = (float)(std::accumulate(total_reconstruction_times.begin(), total_reconstruction_times.end(), 0)) / total_reconstruction_times.size();
    float stddev_total_reconstruction_times = (float)std::sqrt((std::inner_product(total_reconstruction_times.begin(), total_reconstruction_times.end(), total_reconstruction_times.begin(), 0.0)) / total_reconstruction_times.size() - average_total_reconstruction_times * average_total_reconstruction_times);

    float average_num_samples_list = (float)(std::accumulate(num_samples_list.begin(), num_samples_list.end(), 0)) / num_samples_list.size();
    float stddev_num_samples_list = (float)std::sqrt((std::inner_product(num_samples_list.begin(), num_samples_list.end(), num_samples_list.begin(), 0.0)) / num_samples_list.size() - average_num_samples_list * average_num_samples_list);

    float average_max_diff_list = (float)(std::accumulate(max_diff_list.begin(), max_diff_list.end(), 0)) / max_diff_list.size();
    float stddev_max_diff_list = (float)std::sqrt((std::inner_product(max_diff_list.begin(), max_diff_list.end(), max_diff_list.begin(), 0.0)) / max_diff_list.size() - average_max_diff_list * average_max_diff_list);

    float average_avg_diff_list = (float)(std::accumulate(avg_diff_list.begin(), avg_diff_list.end(), 0)) / avg_diff_list.size();
    float stddev_avg_diff_list = (float)std::sqrt((std::inner_product(avg_diff_list.begin(), avg_diff_list.end(), avg_diff_list.begin(), 0.0)) / avg_diff_list.size() - average_avg_diff_list * average_avg_diff_list);

    float average_PSNR_list = (float)(std::accumulate(PSNR_list.begin(), PSNR_list.end(), 0)) / PSNR_list.size();
    float stddev_PSNR_list = (float)std::sqrt((std::inner_product(PSNR_list.begin(), PSNR_list.end(), PSNR_list.begin(), 0.0)) / PSNR_list.size() - average_PSNR_list * average_PSNR_list);

    float average_SNR_list = (float)(std::accumulate(SNR_list.begin(), SNR_list.end(), 0)) / SNR_list.size();
    float stddev_SNR_list = (float)std::sqrt((std::inner_product(SNR_list.begin(), SNR_list.end(), SNR_list.begin(), 0.0)) / SNR_list.size() - average_SNR_list * average_SNR_list);

    float average_blocks_reused_list;
    float stddev_blocks_reused_list;

    float average_histogram_times;
    float stddev_histogram_times;
    float average_sample_times;
    float stddev_sample_times;

    if (sampling_method == 3 || sampling_method == 6 || sampling_method == 9){ // if we are a GPU sampling method, we do not have histogram or sampling times
        average_histogram_times = -1;
        stddev_histogram_times = -1;
        average_sample_times = -1;
        stddev_sample_times = -1;
    }else{
        average_histogram_times = (float)(std::accumulate(histogram_times.begin(), histogram_times.end(), 0)) / histogram_times.size();
        stddev_histogram_times = (float)std::sqrt((std::inner_product(histogram_times.begin(), histogram_times.end(), histogram_times.begin(), 0.0)) / histogram_times.size() - average_histogram_times * average_histogram_times);

        average_sample_times = (float)(std::accumulate(sample_times.begin(), sample_times.end(), 0)) / sample_times.size();
        stddev_sample_times = (float)std::sqrt((std::inner_product(sample_times.begin(), sample_times.end(), sample_times.begin(), 0.0)) / sample_times.size() - average_sample_times * average_sample_times);
    }
    */

    /* Write Averages to File */
    /*
    try{
        // ignore T0 because there is no reuse, as it is the first timestep
        blocks_reused_list.erase(blocks_reused_list.begin());
        average_blocks_reused_list = (float)(std::accumulate(blocks_reused_list.begin(), blocks_reused_list.end(), 0)) / blocks_reused_list.size();
        stddev_blocks_reused_list = (float)std::sqrt((std::inner_product(blocks_reused_list.begin(), blocks_reused_list.end(), blocks_reused_list.begin(), 0.0)) / blocks_reused_list.size() - average_blocks_reused_list * average_blocks_reused_list);        
    }catch(int e){
        average_blocks_reused_list = 0;
        stddev_blocks_reused_list = 0;
    }

    float average_bandwidth_mb_s = average_file_size_MB / (average_total_sampling_process_times / (1e+6));
    float average_reconstruction_mb_s = average_file_size_MB / (average_total_reconstruction_times / (1e+6));

    if (reconstruct_method == 0){
        // No Reconstruction -> No statistics values
        stats_file << "Average: " << "," << std::to_string(sampling_method) << "," << std::to_string(average_file_size_MB) << "," << std::to_string(sample_ratio) << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins) << ","<< "NA" << "," << std::to_string(average_bandwidth_mb_s) << "," << average_total_sampling_process_times << "," <<  average_histogram_times << "," << average_sample_times << "," << "NA" << "," << "NA" << "," <<  average_blocks_reused_list << "," <<  "NA" << "," << "NA"<< "," << "NA" << "," <<  "NA" << "," << "NA" << "\n";
        stats_file << "Std Dev: " << "," << std::to_string(sampling_method)<< "," << "NA" << "," << "NA" << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins) << ","<< "NA" << "," << "NA" << "," << stddev_total_sampling_process_times << "," <<  stddev_histogram_times << "," << stddev_sample_times << "," << "NA" << "," << stddev_blocks_reused_list << "," <<  "NA" << "," << "NA"<< "," << "NA" << "," << "NA" << "," << "NA" << "," << "NA" << "\n";

    } else{
        
        stats_file << "Average: " << "," << std::to_string(sampling_method) << "," << std::to_string(average_file_size_MB) << "," << std::to_string(sample_ratio) << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins)  << ","<< std::to_string(reconstruct_method) << "," << std::to_string(average_bandwidth_mb_s) << "," << average_total_sampling_process_times << "," << average_histogram_times << "," << average_sample_times << "," << average_total_reconstruction_times << "," << std::to_string(average_reconstruction_mb_s) << "," << average_blocks_reused_list << "," <<  average_num_samples_list << "," << average_max_diff_list<< "," << average_avg_diff_list << "," <<  average_PSNR_list << "," << average_SNR_list << "\n";
        stats_file << "Std Dev: " << "," << std::to_string(sampling_method) << "," << "NA" << "," << "NA" << "," << XBLOCK << "/" << YBLOCK << "/" << ZBLOCK << "," << std::to_string(num_bins)  << ","<< std::to_string(reconstruct_method) << "," << "NA" << "," << stddev_total_sampling_process_times << "," << stddev_histogram_times << "," << stddev_sample_times << "," << average_total_reconstruction_times << "," << "NA" << "," << stddev_blocks_reused_list << "," <<  stddev_num_samples_list << "," << stddev_max_diff_list<< "," << stddev_avg_diff_list << "," << stddev_PSNR_list << "," << stddev_SNR_list << "\n";

    }
    */

    stats_file.close();

    return 0;
}
