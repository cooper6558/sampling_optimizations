#include "sampling_funcs.h"
#include <numeric> // for standard deviation in hist_bin_scott

bool SAMPLING_PRINT = false;

// Random Seed 2
// Note: Make sure to change seed in interactive_example.cpp as well
//long long int random_seed_2 = 0;
long long int random_seed_2 = std::chrono::system_clock::now().time_since_epoch().count();

//////////////////////
// HELPER FUNCTIONS //
//////////////////////

/* Scott Bin Estimator:  Less robust estimator that that takes into account data variability and data size.
    The binwidth is proportional to the standard deviation of the
    data and inversely proportional to cube root of ``x.size``. Can
    be too conservative for small datasets, but is quite good for
    large datasets. The standard deviation is not very robust to
    outliers. Values are very similar to the Freedman-Diaconis
    estimator in the absence of outliers.
*/
float hist_bin_scott(vector<float> x, int num_elements, float first_edge, float last_edge, double mean, double stdev){
    // note range edges are not used in this method
    //math:: h = \sigma \sqrt[3]{\frac{24 * \sqrt{\pi}}{n}}

    // return width
    return pow((24.0 * sqrt(M_PI) /num_elements),(1.0 / 3.0)) * stdev;
}

/* Doane Bin Estimator: An improved version of Sturges’ estimator that works better with non-normal datasets.  
    An improved version of Sturges' formula that produces better
    estimates for non-normal datasets. This estimator attempts to
    account for the skew of the data.
*/
float hist_bin_doane(vector<float> x, int num_elements, float first_edge, float last_edge, double mean, double stdev){
    float sg1 = std::sqrt(6.0 * (x.size() - 2) / ((x.size() + 1.0) * (x.size() + 3)));

    if (stdev > 0.0){
        std::vector<float> temp(x.size());
        std::copy(x.begin(), x.end(), temp.begin());
        for_each(temp.begin(), temp.end(), [mean](float &c){ c = c - mean; }); //temp = x - mean;
        for_each(temp.begin(), temp.end(), [stdev](float &c){ c /= stdev; }); //temp = temp / stdev;
        int k = 3;
        for_each(temp.begin(), temp.end(), [k](float &c){ pow(c, k); }); //temp = pow(temp, 3);
        double sum_temp = std::accumulate(temp.begin(), temp.end(), 0.0);
        double mean_temp = sum_temp / temp.size();
        float width = float((last_edge - first_edge) / (1.0 + log2(x.size()) + log2(1.0 + abs(mean_temp) / sg1)));
        return width;
    }
    
    return 0;
}


// TODO
/* Freedman Diaconis Estimator: The binwidth is proportional to the interquartile range (IQR)
    and inversely proportional to cube root of a.size. Can be too
    conservative for small datasets, but is quite good for large
    datasets. The IQR is very robust to outliers.
*/
/*
float hist_bin_fd(vector<float> x, int num_elements, float first_edge, float last_edge, double mean, double stdev){
    // range is unused
    //float q1 = 0.50;
    //float q2 = 0.75;
    //vector<float> percentile = np.true_divide(x, 100);
    //vector<float> iqr = subtract(x, percentile);
    
    std::sort(error.begin(), error.end());
    // quartiles
    float q12 = error[num_elements*1/4];
    //float q23 = error[num_elements*2/4];
    float q34 = error[num_elements*3/4];

    //iqr = np.subtract(*np.percentile(x, [75, 25]))
    //return 2.0 * iqr * x.size() ** (-1.0 / 3.0)

    return 0;
}
*/

/* Sturges Bin Estimator: R’s default method, only accounts for data size. 
    Only optimal for gaussian data and underestimates number of bins for large non-gaussian datasets.
    A very simplistic estimator based on the assumption of normality of
    the data. This estimator has poor performance for non-normal data,
    which becomes especially obvious for large data sets. The estimate
    depends only on size of the data.
*/
float hist_bin_sturges(vector<float> x, int num_elements, float first_edge, float last_edge){
    //_ptp(x) / (np.log2(x.size) + 1.0)
    //return (x.max() - x.min()) / (log2(x.size() + 1.0));
    return (last_edge - first_edge) / (log2(x.size() + 1.0));
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in){
  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

/* get multiples of m starting at n*/
vector<int> get_multiples(int m, int n = 2){
    vector<int> multiples;
    for(int i = n; i < m; i++){
        if (m%i == 0) multiples.push_back(i);
    }
    return multiples;
}

void configure_inputs(vector<std::string> filenames_list_sorted, int max_threads, int XDIM, int YDIM, int ZDIM, float sample_ratio, int* num_bins, float* error_threshold, int* XBLOCK, int* YBLOCK, int* ZBLOCK, float lifetime_min, float lifetime_max, double* bins_elapsed_seconds, double* err_elapsed_seconds, double* rdims_elapsed_seconds){
    std::cout << "In Configuring function ...\n";
    float small_sample_ratio = sample_ratio;

    //vector<float> full_data_0, full_data_1

    // read in data
    std::ifstream fin(filenames_list_sorted[0], std::ios::binary);
    if(!fin){
        std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[0] << "\n";
        exit(0);
    }
    
    // Determine number of elements
    fin.seekg(0, std::ios::end);
    // int file_size_bytes = fin.tellg();
    int num_elements = fin.tellg() / sizeof(float);
    fin.seekg(0, std::ios::beg);
    vector<float> full_data_0(XDIM*YDIM*ZDIM);
    fin.read(reinterpret_cast<char*>(&full_data_0[0]), num_elements*sizeof(float));
    fin.close();
    

    // Automatically determine optimal number of bins
    auto time_start = std::chrono::steady_clock::now();
    auto sample_recons_start = std::chrono::steady_clock::now();
    auto sample_recons_end = std::chrono::steady_clock::now();
    if(*num_bins == -1){
        vector<int> num_bins_list;

        // get standard deviation of full_data_0
        double sum = std::accumulate(full_data_0.begin(), full_data_0.end(), 0.0);
        double mean = sum / full_data_0.size();
        double sq_sum = std::inner_product(full_data_0.begin(), full_data_0.end(), full_data_0.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / full_data_0.size() - mean * mean);
        int num_bins_temp;

        if (*XBLOCK == 1 && *YBLOCK == 1 && *ZBLOCK == 1){
            // do not sample and reconstruct bc we dont have the region size yet
            float first_edge = lifetime_min;
            float last_edge = lifetime_max;
            // TODO remove any element outside of the range or it will break
            float width = hist_bin_scott(full_data_0, num_elements, first_edge, last_edge, mean, stdev);
            float range = last_edge - first_edge; // _unsigned_subtract(last_edge, first_edge)
            int n_equal_bins = int(ceil(range / width));
            std::vector<double> bin_edges = linspace(first_edge, last_edge, n_equal_bins + 1); // Return evenly spaced numbers over a specified interval.
            num_bins_list.push_back(bin_edges.size());
            num_bins_temp = bin_edges.size();
            *num_bins = num_bins_temp;
            std::cout << "scott recommendation: " << num_bins_temp << std::endl;
        }else{
            // Setup general output vectors
            vector<int> sample_data_ids;
            vector<float> sample_data;
            vector<int> samples_per_block;
            
            vector<float> max_diff_list;
            vector<float> avg_diff_list;
            vector<float> PSNR_list;
            vector<float> SNR_list;
            vector<float> sampling_timers;
            
            vector<float> reconstructed_data;
            vector<double> stats;
            int num_sample_ids;

            // scott
            float first_edge = lifetime_min;
            float last_edge = lifetime_max;
            // TODO remove any element outside of the range or it will break
            float width = hist_bin_scott(full_data_0, num_elements, first_edge, last_edge, mean, stdev);
            float range = last_edge - first_edge; // _unsigned_subtract(last_edge, first_edge)
            int n_equal_bins = int(ceil(range / width));
            std::vector<double> bin_edges = linspace(first_edge, last_edge, n_equal_bins + 1); // Return evenly spaced numbers over a specified interval.
            num_bins_list.push_back(bin_edges.size());
            num_bins_temp = bin_edges.size();
            std::cout << "/********************/\nscott recommendation: " << num_bins_temp << std::endl;

            /*********/

            // doane
            width = hist_bin_doane(full_data_0, num_elements, first_edge, last_edge, mean, stdev);
            range = last_edge - first_edge; // _unsigned_subtract(last_edge, first_edge)
            n_equal_bins = int(ceil(range / width));
            bin_edges = linspace(first_edge, last_edge, n_equal_bins + 1); // Return evenly spaced numbers over a specified interval.
            num_bins_list.push_back(bin_edges.size());
            num_bins_temp = bin_edges.size();
            std::cout << "/********************/\ndoane recommendation: " << num_bins_temp << std::endl;

            /*********/
            // removed bc too similar results to doane but also works only with gaussian datasets
            /*
            // sturges
            width = hist_bin_sturges(full_data_0, num_elements, first_edge, last_edge);
            range = last_edge - first_edge; // _unsigned_subtract(last_edge, first_edge)
            n_equal_bins = int(ceil(range / width));
            bin_edges = linspace(first_edge, last_edge, n_equal_bins + 1); // Return evenly spaced numbers over a specified interval.
            num_bins_list.push_back(bin_edges.size());
            num_bins_temp = bin_edges.size();
            std::cout << "\nsturges recommendation: " << num_bins_temp << std::endl;
            */

            std::cout << "\n/************************/\n";
            /*** manually add any to test ***/
            /*
            num_bins_list.push_back(3);
            for(int i = 5; i < 1000; i = i + 5){
                num_bins_list.push_back(i); // manually add any to test
            }
            */
            sample_recons_start = std::chrono::steady_clock::now();
            for (uint i = 0; i < num_bins_list.size(); i++){
                // Reset each vector for each loop
                sample_data_ids.resize(0,0);
                sample_data.resize(0,0);
                samples_per_block.resize(0,0);
                sampling_timers.resize(0,0);
                reconstructed_data.resize(0,0);
                stats.resize(0,0);
                
                //sample: Note to be fair when looking at overhead, we need to compare serial sampling with serial sampling configuration
                //omp_value_histogram_based_importance_sampling(max_threads, full_data_0, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, small_sample_ratio, num_bins_list[i], sample_data_ids, sample_data, samples_per_block, sampling_timers);
                value_histogram_based_importance_sampling(full_data_0, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, small_sample_ratio, num_bins_list[i], sample_data_ids, sample_data, samples_per_block, sampling_timers);

                // OMP Nearest Neighbors Reconstruction
                num_sample_ids = sample_data_ids.size();
                omp_nearest_neighbors_reconstruction(max_threads, sample_data_ids, sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                
                // calculate snr
                std::cout << "/********************/\nNumber of Bins: " << num_bins_list[i] << std::endl;
                data_quality_analysis(full_data_0, reconstructed_data, num_elements, stats);
                max_diff_list.push_back(stats[0]);
                avg_diff_list.push_back(stats[1]);
                PSNR_list.push_back(stats[2]);
                SNR_list.push_back(stats[3]);
            }
            sample_recons_end = std::chrono::steady_clock::now();

            // get max SNR index
            int max_index = std::distance (PSNR_list.begin(), std::max_element (PSNR_list.begin(), PSNR_list.end()));
            *num_bins = num_bins_list[max_index];

            /*** Make figure comparisons for paper ***/
            bool PAPER_FIGURES = false;
            if(PAPER_FIGURES){
                // for number of bins in list to test
                int num_timesteps_to_avg = 10;
                // get average quality
                vector<float> avg_stats(num_bins_list.size());
                for (uint i = 0; i < num_bins_list.size(); i++){
                    float avg_PSNR = 0;
                    for (int j = 0; j < num_timesteps_to_avg; j++){
                        // Reset each vector for each loop
                        sample_data_ids.resize(0,0);
                        sample_data.resize(0,0);
                        samples_per_block.resize(0,0);
                        sampling_timers.resize(0,0);
                        reconstructed_data.resize(0,0);
                        stats.resize(0,0);

                        // read in next time-step
                        std::ifstream fin2(filenames_list_sorted[j], std::ios::binary);
                        if(!fin2){
                            std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[i] << "\n";
                            exit(0);
                        }
                        vector<float> full_data_1(XDIM*YDIM*ZDIM);
                        fin2.read(reinterpret_cast<char*>(&full_data_1[0]), num_elements*sizeof(float));
                        fin2.close();
                        
                        //sample
                        omp_value_histogram_based_importance_sampling(max_threads, full_data_1, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, small_sample_ratio, num_bins_list[i], sample_data_ids, sample_data, samples_per_block, sampling_timers);

                        // OMP Nearest Neighbors Reconstruction
                        num_sample_ids = sample_data_ids.size();
                        omp_nearest_neighbors_reconstruction(max_threads, sample_data_ids, sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                        
                        // calculate snr
                        std::cout << "/********************/\nNumber of Bins: " << num_bins_list[i] << std::endl;
                        std::cout << "Time-Step: " << j << std::endl;
                        data_quality_analysis(full_data_1, reconstructed_data, num_elements, stats);
                        avg_PSNR = avg_PSNR + stats[2];
                    }
                    avg_PSNR = avg_PSNR / num_timesteps_to_avg;
                    avg_stats[i] = avg_PSNR;
                }

                /* write results to csv */
                std::ofstream avg_stats_file;
                avg_stats_file.open ("num_bins_configuration_tests.csv"); // write to file
                avg_stats_file << "Num Bins,Max Diff,Avg Diff,PSNR,SNR,AvgPSNR_" << (num_timesteps_to_avg) << "\n";
                for (uint i = 0; i < num_bins_list.size(); i++){
                    avg_stats_file << num_bins_list[i] << "," << max_diff_list[i] << "," << avg_diff_list[i] << "," << PSNR_list[i] << "," << SNR_list[i] << "," << avg_stats[i] << "\n";
                }
                avg_stats_file.close();
            }
        }
        std::cout << "/********************/\nscott recommendation: " << num_bins_list[0] << "doane recommendation: " << num_bins_list[1] << std::endl;
        std::cout << "Updated number of bins to be: " << *num_bins << "\n";
        
    }
    auto time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_elapsed = time_end-time_start; 
    *bins_elapsed_seconds = time_elapsed.count(); //s
    // TODO do something with this timer other than just printing it to screen
    time_elapsed = sample_recons_end-sample_recons_start; 
    std::cout << "Configure Bins - Sample and Recons Sub-Section: " << time_elapsed.count() << " s" << std::endl;
    std::cout << "                                                " << time_elapsed.count() / *bins_elapsed_seconds << " %" << std::endl;


    /*****************************************************/
    // determine error threshold
    time_start = std::chrono::steady_clock::now();
    if(*error_threshold == -1){

        // get second time-step
        std::ifstream fin2(filenames_list_sorted[1], std::ios::binary);
        if(!fin2){
            std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[1] << "\n";
            exit(0);
        }
        vector<float> full_data_1(XDIM*YDIM*ZDIM);
        fin2.read(reinterpret_cast<char*>(&full_data_1[0]), num_elements*sizeof(float));
        fin2.close();
        
        // get difference (error) between first 2 time-steps
        vector<float> error(num_elements);
        #pragma omp parallel for
        for(int i = 0; i < num_elements; i++){
            error[i] = abs(full_data_0[i] - full_data_1[i]);
        }

        
        /*
        // Optional Preprocessing Step: remove all the zero error thresholds except 1 so we don't flood the data
        // or take the range then divide it into 100 regions then do the quartiles?
        float min_err = *std::min_element(error.begin(), error.end());
        float max_err = *std::max_element(error.begin(), error.end());
        std::cout << "err range: " << min_err << " : " << max_err << std::endl;
        error.resize(0,0);
        int n = 100; //100; // == num_elements?
        float diff_err = (max_err - min_err)/n;
        float d = min_err;
        for (int i = 0; i <= n; i++){
            error.push_back(d);
            d = d + diff_err;
            //std::cout << i << " " << error[i] << std::endl;
        }

        // pushing back thresholds for testing
        std::cout << "\n\nErrors: \n";
        vector<float> thresholds;
        for(float i = 0; i < 1; i = i + 0.02){ //0.001 for detailed
            thresholds.push_back(error[int(n*i)]);
            std::cout << i << " " << error[int(n*i)] << " ";
        }
        */
        // End optional preprocessing step

        // TODO openMP sort
        std::sort(error.begin(), error.end());
        
        // percentiles
        //*error_threshold = error[int(num_elements*0.10)]; // 10th percentile
        *error_threshold = error[int(num_elements*0.75)]; // 75th percentile AKA 3rd quartile
        
        /*** Make figure comparisons for paper ***/
        bool PAPER_FIGURES = false;
        if(PAPER_FIGURES){

            vector<float> thresholds;
            for(float i = 0; i <= 1.1; i = i + 0.02){
                if(num_elements*i > num_elements){
                    thresholds.push_back(error[int(num_elements-1)]);
                    std::cout << i << " " << error[num_elements] << std::endl;
                }else{
                    thresholds.push_back(error[int(num_elements*(float(i)))]);
                    std::cout << i << " " << error[int(num_elements*float(i))] << std::endl;
                }
            }
            /* manually add thresholds to test */
            for(int i = 50; i < 1000; i = i + 50){
                thresholds.push_back(i);
            }

            // do sampling and recons and PSNR for all in quartiles
            // Setup general output vectors
            vector<int> sample_data_ids;
            vector<float> sample_data;
            vector<int> samples_per_block;
            
            vector<float> max_diff_list;
            vector<float> avg_diff_list;
            vector<float> PSNR_list;
            vector<float> SNR_list;
            vector<float> sampling_timers;
            vector<int> blocks_reused_list;
            
            vector<float> reconstructed_data;
            vector<double> stats;
            int num_sample_ids;

            // For error_based_reuse method
            // int total_reference_samples;
            vector<int> reference_sample_ids;
            vector<float> reference_sample_data;
            vector<int> ref_samples_per_block;

            //get inital samples to compare against
            omp_value_histogram_based_importance_sampling(max_threads, full_data_0, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, small_sample_ratio, *num_bins, sample_data_ids, sample_data, samples_per_block, sampling_timers);
            ref_samples_per_block = samples_per_block;
            reference_sample_ids = sample_data_ids;
            reference_sample_data = sample_data;

            /* get quality for the first timestep, using this error threshold */
            for (uint i = 0; i < thresholds.size(); i++){
                // Reset each vector for each loop
                sample_data_ids.resize(0,0);
                sample_data.resize(0,0);
                samples_per_block.resize(0,0);
                sampling_timers.resize(0,0);
                reconstructed_data.resize(0,0);
                stats.resize(0,0);

                //sample with error reuse
                int blocks_reused = omp_temporal_error_based_reuse_sampling(max_threads, full_data_1, *num_bins, small_sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, ref_samples_per_block, reference_sample_ids, reference_sample_data, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers, thresholds[i]);
                blocks_reused_list.push_back(blocks_reused);

                // OMP Nearest Neighbors Reconstruction
                num_sample_ids = sample_data_ids.size();
                omp_nearest_neighbors_reconstruction(max_threads, sample_data_ids, sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                
                // calculate snr
                std::cout << "/********************/\nError Threshold: " << thresholds[i] << std::endl;
                std::cout << "Number of blocks reused: " << blocks_reused << std::endl;
                data_quality_analysis(full_data_1, reconstructed_data, num_elements, stats);
                max_diff_list.push_back(stats[0]);
                avg_diff_list.push_back(stats[1]);
                PSNR_list.push_back(stats[2]);
                SNR_list.push_back(stats[3]);
                std::cout << std::endl << std::endl;
            }

            // get max SNR index
            int max_index = std::distance (PSNR_list.begin(), std::max_element (PSNR_list.begin(), PSNR_list.end()));
            std::cout << "best error thresh " << thresholds[max_index] << std::endl;
            //*error_threshold = thresholds[max_index];

            /* test average over multiple timesteps */
            // for number of bins in list to test
            int num_timesteps_to_avg = 10;
            // get average quality
            vector<float> avg_stats(thresholds.size());
            for (uint i = 0; i < thresholds.size(); i++){
                float avg_PSNR = 0;
                ref_samples_per_block.resize(0,0);
                reference_sample_ids.resize(0,0);
                reference_sample_data.resize(0,0);
                for (int j = 0; j < num_timesteps_to_avg; j++){
                    // Reset each vector for each loop
                    sample_data_ids.resize(0,0);
                    sample_data.resize(0,0);
                    samples_per_block.resize(0,0);
                    sampling_timers.resize(0,0);
                    reconstructed_data.resize(0,0);
                    stats.resize(0,0);

                    // read in next time-step
                    std::ifstream fin2(filenames_list_sorted[j], std::ios::binary);
                    if(!fin2){
                        std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[i] << "\n";
                        exit(0);
                    }
                    vector<float> full_data_3(XDIM*YDIM*ZDIM);
                    fin2.read(reinterpret_cast<char*>(&full_data_3[0]), num_elements*sizeof(float));
                    fin2.close();
                    
                    //sample
                    //int blocks_reused = omp_temporal_error_based_reuse_sampling(max_threads, full_data_3, *num_bins, small_sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, ref_samples_per_block, reference_sample_ids, reference_sample_data, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers, thresholds[i]);
                    ref_samples_per_block = samples_per_block;
                    reference_sample_ids = sample_data_ids;
                    reference_sample_data = sample_data;

                    // OMP Nearest Neighbors Reconstruction
                    num_sample_ids = sample_data_ids.size();
                    omp_nearest_neighbors_reconstruction(max_threads, sample_data_ids, sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                    
                    // calculate snr
                    std::cout << "/********************/\nError Threshold: " << thresholds[i] << std::endl;
                    std::cout << "Time-Step: " << j << std::endl;
                    data_quality_analysis(full_data_3, reconstructed_data, num_elements, stats);
                    avg_PSNR = avg_PSNR + stats[2];
                }
                avg_PSNR = avg_PSNR / num_timesteps_to_avg;
                avg_stats[i] = avg_PSNR;
            }

            /* write results to csv */
            std::ofstream avg_stats_file;
            avg_stats_file.open ("error_threshold_configuration_tests.csv"); // write to file
            avg_stats_file << "Error Threshold,Blocks Reused,Max Diff,Avg Diff,PSNR,SNR,AvgPSNR_" << (num_timesteps_to_avg) << "\n";
            for (uint i = 0; i < thresholds.size(); i++){
                avg_stats_file << thresholds[i] << "," << blocks_reused_list[i] << "," << max_diff_list[i] << "," << avg_diff_list[i] << "," << PSNR_list[i] << "," << SNR_list[i] << "," << avg_stats[i] << "\n";
            }
            avg_stats_file.close();
        }

        std::cout << "Updated error theshold to: " << *error_threshold << "\n";
        /*
        std::cout << "Error Percentiles: \n";
        for (float i = 0; i <= 1; i = i + 0.01){
            std::cout << i << ": " << error[int(num_elements*i)] << std::endl;
        }
        */
    }    
    time_end = std::chrono::steady_clock::now();
    time_elapsed = time_end-time_start; 
    *err_elapsed_seconds = time_elapsed.count(); //s

    /*****************************************************/

    time_start = std::chrono::steady_clock::now();
    if((*XBLOCK == 1 && *YBLOCK == 1 && *ZBLOCK == 1)){

        // get second time-step
        std::ifstream fin2(filenames_list_sorted[1], std::ios::binary);
        if(!fin2){
            std::cout << " Error, Couldn't find the file: " << filenames_list_sorted[1] << "\n";
            exit(0);
        }
        vector<float> full_data_1(XDIM*YDIM*ZDIM);
        fin2.read(reinterpret_cast<char*>(&full_data_1[0]), num_elements*sizeof(float));
        fin2.close();
        
        std::cout << "Determining Optimal Number of Regions . . .\n";

        // get all even divisables for each dimension
        vector<int> x_dims = get_multiples(XDIM, 3);
        vector<int> y_dims = get_multiples(YDIM, 3);
        vector<int> z_dims = get_multiples(ZDIM, 3);

        // get all combinations of xyz multiples to form all possible region sizes 
        vector<int> possible_region_dims_x;
        vector<int> possible_region_dims_y;
        vector<int> possible_region_dims_z;
        vector<int> possible_num_regions;
        int num_regions = 0;
        for(uint i = 0; i < x_dims.size(); i++){
            for(uint j = 0; j < y_dims.size(); j++){
                for(uint k = 0; k < z_dims.size(); k++){
                    num_regions = (XDIM*YDIM*ZDIM) / ((x_dims[i]) * (y_dims[j]) * (z_dims[k]));
                    if (!(std::find(possible_num_regions.begin(), possible_num_regions.end(), num_regions) != possible_num_regions.end())){
                        // TODO take smaller subset of num_regions
                        possible_num_regions.push_back(num_regions);
                        possible_region_dims_x.push_back(x_dims[i]);
                        possible_region_dims_y.push_back(y_dims[j]);
                        possible_region_dims_z.push_back(z_dims[k]);
                        std::cout << "Test Dim: " << x_dims[i] << "x" << y_dims[j] << "x" << z_dims[k] <<std::endl;
                    }
                }
            }
        }
       
        /*** Make figure comparisons for paper ***/
        bool PAPER_FIGURES = false;
        bool reconstuct_configure = false; // true to reconstruct and calculate snr
        if(PAPER_FIGURES){
            // do sampling and recons and PSNR for all in quartiles
            // Setup general output vectors
            vector<int> sample_data_ids;
            vector<float> sample_data;
            vector<int> samples_per_block;
            
            vector<float> max_diff_list;
            vector<float> avg_diff_list;
            vector<float> PSNR_list;
            vector<float> SNR_list;
            vector<float> sampling_timers;
            vector<int> blocks_reused_list;
            vector<int> number_of_regions;
            vector<float> sample_bandwidth;

            
            vector<float> reconstructed_data;
            vector<double> stats;
            int num_sample_ids;

            // For error_based_reuse method
            // uint total_reference_samples;
            vector<int> reference_sample_ids;
            vector<float> reference_sample_data;
            vector<int> ref_samples_per_block;
            vector<int> reference_histogram_list;            

            /* get quality for the first timestep, using this error threshold */
            for (uint i = 0; i <= possible_region_dims_x.size(); i++){
                if(i == possible_region_dims_x.size()){
                    *XBLOCK = XDIM;
                    *YBLOCK = YDIM;
                    *ZBLOCK = ZDIM;
                }else{
                    *XBLOCK = possible_region_dims_x[i];
                    *YBLOCK = possible_region_dims_y[i];
                    *ZBLOCK = possible_region_dims_z[i];
                }

                int num_blocks = (XDIM*YDIM*ZDIM) / ((*XBLOCK) * (*YBLOCK) * (*ZBLOCK));
                number_of_regions.push_back(num_blocks);
                std::cout << "Test Dim: " << *XBLOCK << "x" << *YBLOCK << "x" << *ZBLOCK << " : Num regions: " << number_of_regions[i] <<std::endl;

                // Reset each vector for each loop
                sample_data_ids.resize(0,0);
                sample_data.resize(0,0);
                samples_per_block.resize(0,0);
                sampling_timers.resize(0,0);
                reconstructed_data.resize(0,0);
                stats.resize(0,0);
                reference_histogram_list.resize(0,0);

                // control sample
                // running the entire sampling process really isnt necessary if we aren't reconstructing, just getting the reference histograms
                if (reconstuct_configure){
                    int blocks_reused = omp_temporal_histogram_based_reuse_sampling(max_threads, full_data_0, *num_bins, sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);
                
                    // Reset each vector for each loop
                    sample_data_ids.resize(0,0);
                    sample_data.resize(0,0);
                    samples_per_block.resize(0,0);
                    sampling_timers.resize(0,0);
                    reconstructed_data.resize(0,0);
                    stats.resize(0,0);

                    auto sample_start = std::chrono::steady_clock::now();
                    // resample
                    blocks_reused = omp_temporal_histogram_based_reuse_sampling(max_threads, full_data_1, *num_bins, sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);
                    auto sample_end = std::chrono::steady_clock::now();
                    time_elapsed = sample_end-sample_start; 
                    sample_bandwidth.push_back((XDIM*YDIM*ZDIM*4/(1e+9))/time_elapsed.count()); //GB/s
                    blocks_reused_list.push_back(blocks_reused);

                    // OMP Nearest Neighbors Reconstruction
                    num_sample_ids = sample_data_ids.size();
                    omp_nearest_neighbors_reconstruction(max_threads, sample_data_ids, sample_data, num_sample_ids, XDIM, YDIM, ZDIM, reconstructed_data);
                    
                    // calculate snr
                    std::cout << "/********************/\nRegion Dims: " << *XBLOCK << "x"<< *YBLOCK << "x"<< *ZBLOCK << std::endl;
                    std::cout << "Number of Regions: " << num_blocks << std::endl;
                    std::cout << "Percent reused:    " << float(float(blocks_reused) / float(num_blocks)) << std::endl;
                    data_quality_analysis(full_data_1, reconstructed_data, num_elements, stats);
                    max_diff_list.push_back(stats[0]);
                    avg_diff_list.push_back(stats[1]);
                    PSNR_list.push_back(stats[2]);
                    SNR_list.push_back(stats[3]);
                    std::cout << std::endl << std::endl;
                
                }else{
                    /* build reference histogram list with omp */
                    for(int block_id = 0; block_id < num_blocks; block_id++){
                        // Get individual block information
                        vector<int> block_sample_ids;
                        vector<float> block_sample_data;
                        get_block_data_w_global_ids(full_data_0, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, block_id, block_sample_ids, block_sample_data);
                        // Create block histogram
                        vector<int> block_histogram(*num_bins);
                        fill(block_histogram.begin(), block_histogram.end(), 0);
                        omp_data_histogram(block_sample_data, block_histogram, *num_bins, lifetime_max, lifetime_min);
                        // Append to reference histogram
                        reference_histogram_list.insert(std::end(reference_histogram_list), std::begin(block_histogram), std::end(block_histogram));
                    }
                    
                    // Ensure accurate reference histogram
                    if(reference_histogram_list.size() != (uint) (*num_bins*num_blocks)){
                        std::cout << "Invalid Reference Histogram..." << reference_histogram_list.size() << " vs " << *num_bins*num_blocks << "\n";
                        exit(0);
                    }
                    /* end build reference histogram */

                    // Reset each vector for each loop
                    sample_data_ids.resize(0,0);
                    sample_data.resize(0,0);
                    samples_per_block.resize(0,0);
                    sampling_timers.resize(0,0);
                    reconstructed_data.resize(0,0);
                    stats.resize(0,0);

                    auto sample_start = std::chrono::steady_clock::now();
                    // resample: with serial to keep overhead calculations fair
                    //int blocks_reused = omp_temporal_histogram_based_reuse_sampling(max_threads, full_data_1, *num_bins, sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);
                    int blocks_reused = temporal_histogram_based_reuse_sampling(full_data_1, *num_bins, sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);

                    auto sample_end = std::chrono::steady_clock::now();
                    time_elapsed = sample_end-sample_start; 
                    sample_bandwidth.push_back((XDIM*YDIM*ZDIM*4/(1e+9))/time_elapsed.count()); //GB/s
                    blocks_reused_list.push_back(blocks_reused);

                    // calculate snr
                    std::cout << "/********************/\nRegion Dims: " << *XBLOCK << "x"<< *YBLOCK << "x"<< *ZBLOCK << std::endl;
                    std::cout << "Number of Regions: " << num_blocks << std::endl;
                    std::cout << "Percent reused:    " << float(float(blocks_reused) / float(num_blocks)) << std::endl;
                    std::cout << std::endl << std::endl;
                }
            }

            /* write results to csv */
            std::ofstream avg_stats_file;
            avg_stats_file.open ("number_of_regions_configuration_tests.csv"); // write to file
            avg_stats_file << "Number of Regions,Region Dims,Blocks Reused,Blocks Reused Percent,Max Diff,Avg Diff,PSNR,SNR,Bandwidth (GB/s)\n";

            if (reconstuct_configure){
                for (uint i = 0; i < number_of_regions.size(); i++){
                    avg_stats_file << number_of_regions[i] << "," << possible_region_dims_x[i] << "x" << possible_region_dims_y[i] << "x" << possible_region_dims_z[i]<< "," << blocks_reused_list[i] << "," << float(float(blocks_reused_list[i]) / float(number_of_regions[i])) << "," << max_diff_list[i] << "," << avg_diff_list[i] << "," << PSNR_list[i] << "," << SNR_list[i] << "," << sample_bandwidth[i] << "\n";
                }
            }else{
                for (uint i = 0; i < number_of_regions.size(); i++){
                    avg_stats_file << number_of_regions[i] << "," << possible_region_dims_x[i] << "x" << possible_region_dims_y[i] << "x" << possible_region_dims_z[i]<< "," << blocks_reused_list[i] << "," << float(float(blocks_reused_list[i]) / float(number_of_regions[i])) << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << sample_bandwidth[i] << "\n";
                }
            }
            avg_stats_file.close();
        }else{
            // do sampling and recons and PSNR for all in quartiles
            // Setup general output vectors
            vector<int> sample_data_ids;
            vector<float> sample_data;
            vector<int> samples_per_block;
            
            vector<float> max_diff_list;
            vector<float> avg_diff_list;
            vector<float> PSNR_list;
            vector<float> SNR_list;
            vector<float> sampling_timers;
            vector<int> blocks_reused_list;
            vector<int> number_of_regions;
            vector<float> sample_bandwidth;

            
            vector<float> reconstructed_data;
            vector<double> stats;
            // int num_sample_ids;

            // For error_based_reuse method
            // int total_reference_samples;
            vector<int> reference_sample_ids;
            vector<float> reference_sample_data;
            vector<int> ref_samples_per_block;
            vector<int> reference_histogram_list;            

            /* get quality for the first timestep, using this error threshold */
            for (uint i = 0; i <= possible_region_dims_x.size(); i++){
                if(i == possible_region_dims_x.size()){
                    *XBLOCK = XDIM;
                    *YBLOCK = YDIM;
                    *ZBLOCK = ZDIM;
                }else{
                    *XBLOCK = possible_region_dims_x[i];
                    *YBLOCK = possible_region_dims_y[i];
                    *ZBLOCK = possible_region_dims_z[i];
                }

                int num_blocks = (XDIM*YDIM*ZDIM) / ((*XBLOCK) * (*YBLOCK) * (*ZBLOCK));
                number_of_regions.push_back(num_blocks);
                std::cout << "Test Dim: " << *XBLOCK << "x" << *YBLOCK << "x" << *ZBLOCK << " : Num regions: " << number_of_regions[i] <<std::endl;

                // Reset each vector for each loop
                sample_data_ids.resize(0,0);
                sample_data.resize(0,0);
                samples_per_block.resize(0,0);
                sampling_timers.resize(0,0);
                reconstructed_data.resize(0,0);
                stats.resize(0,0);
                reference_histogram_list.resize(0,0);

                /* build reference histogram list with omp */
                for(int block_id = 0; block_id < num_blocks; block_id++){
                    // Get individual block information
                    vector<int> block_sample_ids;
                    vector<float> block_sample_data;
                    get_block_data_w_global_ids(full_data_0, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, block_id, block_sample_ids, block_sample_data);
                    // Create block histogram
                    vector<int> block_histogram(*num_bins);
                    fill(block_histogram.begin(), block_histogram.end(), 0);
                    omp_data_histogram(block_sample_data, block_histogram, *num_bins, lifetime_max, lifetime_min);
                    // Append to reference histogram
                    reference_histogram_list.insert(std::end(reference_histogram_list), std::begin(block_histogram), std::end(block_histogram));
                }
                
                // Ensure accurate reference histogram
                if(reference_histogram_list.size() != (uint) (*num_bins*num_blocks)){
                    std::cout << "Invalid Reference Histogram..." << reference_histogram_list.size() << " vs " << *num_bins*num_blocks << "\n";
                    exit(0);
                }
                /* end build reference histogram */

                // Reset each vector for each loop
                sample_data_ids.resize(0,0);
                sample_data.resize(0,0);
                samples_per_block.resize(0,0);
                sampling_timers.resize(0,0);
                reconstructed_data.resize(0,0);
                stats.resize(0,0);

                auto sample_start = std::chrono::steady_clock::now();
                // resample: with serial to keep overhead calculations fair
                //int blocks_reused = omp_temporal_histogram_based_reuse_sampling(max_threads, full_data_1, *num_bins, sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);
                int blocks_reused = temporal_histogram_based_reuse_sampling(full_data_1, *num_bins, sample_ratio, XDIM, YDIM, ZDIM, *XBLOCK, *YBLOCK, *ZBLOCK, reference_histogram_list, sample_data_ids, sample_data, samples_per_block, lifetime_max, lifetime_min, sampling_timers);

                auto sample_end = std::chrono::steady_clock::now();
                time_elapsed = sample_end-sample_start; 
                sample_bandwidth.push_back((XDIM*YDIM*ZDIM*4/(1e+9))/time_elapsed.count()); //GB/s
                blocks_reused_list.push_back(blocks_reused);

                // calculate snr
                std::cout << "/********************/\nRegion Dims: " << *XBLOCK << "x"<< *YBLOCK << "x"<< *ZBLOCK << std::endl;
                std::cout << "Number of Regions: " << num_blocks << std::endl;
                std::cout << "Percent reused:    " << float(float(blocks_reused) / float(num_blocks)) << std::endl;
                std::cout << std::endl << std::endl;
            }
        }

        // TODO find best compromise between the two functions
        *XBLOCK = possible_region_dims_x[3];
        *YBLOCK = possible_region_dims_y[3];
        *ZBLOCK = possible_region_dims_z[3];
        std::cout << "Updated region dimensions\n";
    }
    time_end = std::chrono::steady_clock::now();
    time_elapsed = time_end-time_start; 
    *rdims_elapsed_seconds = time_elapsed.count(); //s
}


/**
 * get_block_data_w_global_ids:
 * Returns vector of data and global IDs
 * found in block with block_id.
 * 
 * Input:
 * full_data                - Original dataset
 * XDIM, YDIM, ZDIM         - Orignal dataset dimensionality
 * XBLOCK, YBLOCK, ZBLOCK   - Block dimensionality
 * block_id                 - Target block to retrieve data from
 * 
 * Output:
 * block_data_global_ids    - Global IDs for block data values
 * block_data               - Block data values
 **/
void get_block_data_w_global_ids(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_data_global_ids, vector<float> &block_data){
    // Ensure ID vector and data vector are clear
    block_data_global_ids.resize(0,0);
    block_data.resize(0,0);
    
    // Determine block start and end coordinates
    // Calculate number of blocks in each dimension
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    // int ZB_COUNT = (ZDIM/ZBLOCK);
    // Calculate Block X,Y,Z coordinates
    int temp_bid = block_id;
    int block_z = temp_bid / (XB_COUNT * YB_COUNT);
    temp_bid = temp_bid - (block_z * XB_COUNT * YB_COUNT);
    int block_y = temp_bid / XB_COUNT;
    int block_x = temp_bid % XB_COUNT;
    // Calculate block X,Y,Z start coordinates
    int x_start = block_x*XBLOCK;
    int y_start = block_y*YBLOCK;
    int z_start = block_z*ZBLOCK;

    // Iterate over block coordinates to gather all data values and global ID's
    for (int k = z_start; k < z_start + ZBLOCK; k++) {
        for (int j = y_start; j < y_start + YBLOCK; j++) {
            for (int i = x_start; i < x_start + XBLOCK; i++) {
                // get global offset ID
                int global_id = i + j*XDIM + k*XDIM*YDIM;
                // Ensure global ID is valid
                if (global_id >= XDIM*YDIM*ZDIM){
                    std::cout << "Invalid Global Id! " << global_id << ">=" << XBLOCK*YBLOCK*ZBLOCK << "\n";
                    exit(0);
                }
                // Store global ID and data value
                block_data_global_ids.push_back(global_id); // for c++ reconstruction
                block_data.push_back(full_data[global_id]);
            }
        }
    } 
}



/**
 * omp_get_block_data_w_global_ids:
 * Returns vector of data and global IDs
 * found in block with block_id. Uses OpenMP parallelization.
 * 
 * Input:
 * full_data                - Original dataset
 * XDIM, YDIM, ZDIM         - Orignal dataset dimensionality
 * XBLOCK, YBLOCK, ZBLOCK   - Block dimensionality
 * block_id                 - Target block to retrieve data from
 * 
 * Output:
 * block_data_global_ids    - Global IDs for block data values
 * block_data               - Block data values
 **/
void omp_get_block_data_w_global_ids(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_data_global_ids, vector<float> &block_data){
    // Ensure ID vector and data vector are clear
    block_data_global_ids.resize(0,0);
    block_data.resize(0,0);
    
    // Determine block start and end coordinates
    // Calculate number of blocks in each dimension
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    // int ZB_COUNT = (ZDIM/ZBLOCK);
    // Calculate Block X,Y,Z coordinates
    int temp_bid = block_id;
    int block_z = temp_bid / (XB_COUNT * YB_COUNT);
    temp_bid = temp_bid - (block_z * XB_COUNT * YB_COUNT);
    int block_y = temp_bid / XB_COUNT;
    int block_x = temp_bid % XB_COUNT;
    // Calculate block X,Y,Z start coordinates
    int x_start = block_x*XBLOCK;
    int y_start = block_y*YBLOCK;
    int z_start = block_z*ZBLOCK;

    // Iterate over block coordinates to gather all data values and global ID's
    int exit_code = 0;
    #pragma omp parallel for collapse(3)
    for (int k = z_start; k < z_start + ZBLOCK; k++) {
        for (int j = y_start; j < y_start + YBLOCK; j++) {
            for (int i = x_start; i < x_start + XBLOCK; i++) {
                // get global offset ID
                int global_id = i + j*XDIM + k*XDIM*YDIM;
                // Ensure global ID is valid
                if (global_id >= XDIM*YDIM*ZDIM){
                    std::cout << "Invalid Global Id! " << global_id << ">=" << XBLOCK*YBLOCK*ZBLOCK << "\n";
                    #pragma omp critical
                    {
                        exit_code = 1;
                    }
                }
                // Store global ID and data value
                #pragma omp critical
                {
                    block_data_global_ids.push_back(global_id); // for c++ reconstruction
                }
                #pragma omp critical
                {
                    block_data.push_back(full_data[global_id]);
                }
            }
        }
    } 
    if (exit_code == 1){
        exit(0);
    }
}



void get_block_samples_w_global_ids(vector<int> &all_sample_ids, vector<float> &all_sample_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_sample_global_ids, vector<float> &block_sample_data){
    // Ensure ID vector and data vector are clear
    
    // Determine block start and end coordinates
    // Calculate number of blocks in each dimension
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    // int ZB_COUNT = (ZDIM/ZBLOCK);
    // Calculate Block X,Y,Z coordinates
    int temp_bid = block_id;
    int block_z = temp_bid / (XB_COUNT * YB_COUNT);
    temp_bid = temp_bid - (block_z * XB_COUNT * YB_COUNT);
    int block_y = temp_bid / XB_COUNT;
    int block_x = temp_bid % XB_COUNT;
    // Calculate block X,Y,Z start coordinates
    int x_start = block_x*XBLOCK;
    int y_start = block_y*YBLOCK;
    int z_start = block_z*ZBLOCK;

    // Iterate over block coordinates to gather all data values and global ID's
    for (uint a = 0; a < all_sample_ids.size(); a++){
        // get global id
        int global_id = all_sample_ids[a];
        // get x y z components from global_id
        int i = global_id % XDIM;
        int j = (global_id / XDIM) % YDIM;
        int k = global_id / (XDIM*YDIM);
        // double check
        if ((i + j*XDIM + k*XDIM*YDIM) != global_id){
            std::cout << "GLOBAL ID CALCUALTION ERROR!\n";
            exit(0);
        }
        
        // if global id is in this block,
        if ( k >= z_start && k < z_start + ZBLOCK && j >= y_start && j < y_start + YBLOCK && i >= x_start && i < x_start + XBLOCK){
            // Store global ID and data value
            block_sample_global_ids.push_back(all_sample_ids[a]); // for c++ reconstruction
            block_sample_data.push_back(all_sample_data[a]);
        }
    }
}


/**
 * get_block_data_w_local_ids:
 * Returns vector of data and local IDs
 * found in block with block_id.
 * 
 * Input:
 * full_data                - Original dataset
 * XDIM, YDIM, ZDIM         - Orignal dataset dimensionality
 * XBLOCK, YBLOCK, ZBLOCK   - Block dimensionality
 * block_id                 - Target block to retrieve data from
 * 
 * Output:
 * block_data_local_ids     - Local IDs for block data values
 * block_data               - Block data values
 **/
void get_block_data_w_local_ids(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int block_id, \
vector<int> &block_data_local_ids, vector<float> &block_data){
    // Ensure ID vector and data vector are clear
    block_data_local_ids.resize(0,0);
    block_data.resize(0,0);
    
    // Determine block start and end coordinates
    // Calculate number of blocks in each dimension
    int XB_COUNT = (XDIM/XBLOCK);
    int YB_COUNT = (YDIM/YBLOCK);
    // int ZB_COUNT = (ZDIM/ZBLOCK);
    // Calculate Block X,Y,Z coordinates
    int temp_bid = block_id;
    int block_z = temp_bid / (XB_COUNT * YB_COUNT);
    temp_bid = temp_bid - (block_z * XB_COUNT * YB_COUNT);
    int block_y = temp_bid / XB_COUNT;
    int block_x = temp_bid % XB_COUNT;
    // Calculate block X,Y,Z start coordinates
    int x_start = block_x*XBLOCK;
    int y_start = block_y*YBLOCK;
    int z_start = block_z*ZBLOCK;

    // Iterate over block coordinates to gather all data values and global ID's
    int local_id = 0;
    for (int k = z_start; k < z_start + ZBLOCK; k++) {
        for (int j = y_start; j < y_start + YBLOCK; j++) {
            for (int i = x_start; i < x_start + XBLOCK; i++) {
                // get global offset ID
                int global_id = i + j*XDIM + k*XDIM*YDIM;
                // Ensure global ID is valid
                if (global_id >= XDIM*YDIM*ZDIM){
                    std::cout << "Invalid Global Id! " << global_id << ">=" << XBLOCK*YBLOCK*ZBLOCK << "\n";
                    exit(0);
                }
                // Ensure local ID is valid
                if (local_id >= XBLOCK*YBLOCK*ZBLOCK){
                    std::cout << "Invalid Local Id! " << local_id << ">=" << XBLOCK*YBLOCK*ZBLOCK << "\n";
                    exit(0);
                }
                // Store global ID and data value
                block_data_local_ids.push_back(local_id); // for c++ reconstruction
                block_data.push_back(full_data[local_id]);
                // Increment local ID
                local_id = local_id + 1;
            }
        }
    } 
}



/**
 * data_histogram:
 * Construct histogram of input data with num_bins.
 * 
 * Input:
 * full_data                - Original dataset
 * num_bins                    - Number of histogram bins
 * max, min                 - Max and min dataset values
 * 
 * Output:
 * value_histogram          - Histogram of data values
 * 
 * Note:
 * Bin Width = (Max - Min) / num_bins
 **/
void data_histogram(vector<float> &full_data, vector<int> &value_histogram, int num_bins, float max, float min){
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;

    // Iterate over each data value
    int num_elems = full_data.size();
    for (int i = 0; i < num_elems; i++){
        // Determine data values bin ID
        int binId = (full_data[i] - min) / binWidth;
        
        // Handle Edge Cases
        if (binId > num_bins-1){
            binId = num_bins - 1;
        }
        if (binId < 0){
            binId = 0;
        }

        // Increment corresponding bin value
        value_histogram[binId]++;
    }        
}



/**
 * omp_data_histogram:
 * Construct histogram of input data with num_bins using OpenMP.
 * 
 * Input:
 * full_data                - Original dataset
 * num_bins                    - Number of histogram bins
 * max, min                 - Max and min dataset values
 * 
 * Output:
 * value_histogram          - Histogram of data values
 * 
 * Note:
 * Bin Width = (Max - Min) / num_bins
 **/
void omp_data_histogram(vector<float> &full_data, vector<int> &value_histogram, int num_bins, float max, float min){
    // Calculate width of bins
    float range = max - min;
    float binWidth = range/num_bins;

    // Iterate over each data value
    int num_elems = full_data.size();

    #pragma omp parallel
    {
        vector<int> private_histogram(num_bins, 0);

        #pragma omp for
        for (int i = 0; i < num_elems; i++){
            // Determine data values bin ID
            int binId = (full_data[i] - min) / binWidth;
            
            // Handle Edge Cases
            if (binId > num_bins-1){
                binId = num_bins - 1;
            }
            if (binId < 0){
                binId = 0;
            }

            // Increment corresponding bin value
            private_histogram[binId]++;
        } 

        #pragma omp critical
        {
            for (int i = 0; i < num_bins; i++){
                value_histogram[i] = value_histogram[i] + private_histogram[i];

            }
        }
    }      
}



/**
 * acceptance_function:
 * Given data histogram and sample ratio, construct 
 * the importance based probability histogram used 
 * later to determine sampling values from each bin.
 * 
 * Input:
 * full_data                - Original dataset
 * num_bins                    - Number of histogram bins
 * sample_ratio             - Percent of data values to sample
 * value histogram          - Histogram of original dataset values
 * 
 * Output:
 * acceptance_histogram     - Histogram of probabilities each bin will be kept
 **/
void acceptance_function(vector<float> &full_data, int num_bins, float sample_ratio, vector<float> &acceptance_histogram, vector<int> &value_histogram, vector<float> &sampling_timers){
    // Determine the total number of samples to take
    int tot_samples = sample_ratio * full_data.size();
    if (SAMPLING_PRINT){
        std::cout << "Looking for: " << sample_ratio << " * " << full_data.size() << " = " << tot_samples << " samples\n";
    }
    
    // Sort histogram bin IDs from least to greatest based on value histogram
    vector<int> histogram_bin_ids(num_bins);
    iota(histogram_bin_ids.begin(), histogram_bin_ids.end(), 0); // fill with ids 0,1,2...num_bins
    auto histogram_sort_time_start = std::chrono::steady_clock::now();
    stable_sort(histogram_bin_ids.begin(), histogram_bin_ids.end(), [&value_histogram](int i1, int i2) {return value_histogram[i1] < value_histogram[i2];});
    auto histogram_sort_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_sort_seconds = histogram_sort_time_end-histogram_sort_time_start;
    sampling_timers.push_back(histogram_sort_seconds.count()); //s
	
    // Determine the target number of samples per bin
    int target_bin_samples = (tot_samples/num_bins);

    // Distribute samples across bins
    int samples;
    int remaining_tot_samples = tot_samples;
    vector<int> samples_per_bin = value_histogram;
    auto acceptance_function_time_start = std::chrono::steady_clock::now();
    for (int i = 0; i<num_bins; i++){
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

    // Clear acceptance histogram and fill with zeros
    fill(acceptance_histogram.begin(), acceptance_histogram.end(), 0);

    // Determine acceptance rate for each bin as the samples to take from that bin divided by the total samples in that bin
    for (int i = 0; i<num_bins; i++){
        // If bin has no samples, set acceptance to zero
        if (value_histogram[i] == 0){
            acceptance_histogram[i] = 0;
        }else{
            acceptance_histogram[i] = (float)samples_per_bin[i] / (float)value_histogram[i];
        }
    }
    auto acceptance_function_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> acceptance_function_seconds = acceptance_function_time_end-acceptance_function_time_start;
    sampling_timers.push_back(acceptance_function_seconds.count()); //s


    if (SAMPLING_PRINT){
        std::cout << "Acceptance Histogram Created!" << std::endl;
    }
}



/**
 * acceptance_function_cuda_inputs:
 * Same as acceptance_function but using 
 * arrays instead of vectors (nessecary for CUDA)
 * 
 * Input:
 * num_bins                    - Number of histogram bins
 * max_samples              - Maximum samples to take
 * value histogram          - Histogram of dataset values
 * 
 * Output:
 * acceptance_histogram     - Histogram of probabilities each bin will be kept
 **/
void acceptance_function_cuda_inputs(int num_bins, int max_samples, float* acceptance_histogram, int* value_histogram){
    // Determine the total number of samples to take
    if (SAMPLING_PRINT){
        std::cout << "Looking for: " << max_samples << " samples\n";
    }
  
    // Sort histogram bin IDs from least to greatest based on value histogram
    vector<int> histogram_bin_ids(num_bins);
    iota(histogram_bin_ids.begin(), histogram_bin_ids.end(), 0); // fill with ids 0,1,2...num_bins
    stable_sort(histogram_bin_ids.begin(), histogram_bin_ids.end(), [&value_histogram](int i1, int i2) {return value_histogram[i1] < value_histogram[i2];});
  
    // Determine the target number of samples per bin
    int target_bin_samples = (max_samples/num_bins);

    // Distribute samples across bins
    int samples;
    int remaining_samples = max_samples;
    int samples_per_bin[num_bins];
    memcpy(samples_per_bin, value_histogram, sizeof(samples_per_bin));
    for (int i = 0; i<num_bins; i++){
        // Get current bin ID
        int index = histogram_bin_ids[i];
        // If the bin has more data values than target samples set samples to target value
        if(value_histogram[index] > target_bin_samples){
            samples = target_bin_samples;
        // If the bin has less data values than target samples set to total samples in bin
        } else {
            samples = value_histogram[index];
        }

        // Store samples to gather for this bin
        samples_per_bin[index] = samples;
        
        // Subtract samples taken from total samples
        remaining_samples = remaining_samples - samples;
        // Update target number of samples per bin
        target_bin_samples = remaining_samples/(num_bins-i);
    }

    // Determine acceptance rate for each bin as the samples to take from that bin divided by the total samples in that bin
    for (int i = 0; i<num_bins; i++){
        // If bin has no samples, set acceptance to zero
        if (value_histogram[i] == 0){
            acceptance_histogram[i] = 0;
        } else {
            acceptance_histogram[i] = (float)samples_per_bin[i] / (float)value_histogram[i];
        }
    }
    if (SAMPLING_PRINT){
        std::cout << "Acceptance Histogram Created!" << std::endl;
    } 
}



/**
 * acceptance_function_multi_criteria:
 * Given data histogram and sample ratio, construct 
 * the importance based probability histogram used 
 * later to determine sampling values from each bin.
 * This acceptance function utilizes both global and local importance.
 * 
 * Input:
 * full_data                - Original dataset
 * num_bins                    - Number of histogram bins
 * sample_ratio             - Percent of data values to sample
 * value histogram          - Histogram of original dataset values
 * 
 * Output:
 * acceptance_histogram     - Histogram of probabilities each bin will be kept
 **/
void acceptance_function_multi_criteria(vector<float> &full_data, int num_bins, float sample_ratio, vector<float> &acceptance_histogram, vector<int> &value_histogram, vector<float> &sampling_timers){
    // Determine the total number of samples to take
    int tot_samples = sample_ratio * full_data.size();
    if (SAMPLING_PRINT){
        std::cout << "Looking for: " << sample_ratio << " * " << full_data.size() << " = " << tot_samples << " samples\n";
    }
    
    // Sort histogram bin IDs from least to greatest based on value histogram
    vector<int> histogram_bin_ids(num_bins);
    iota(histogram_bin_ids.begin(), histogram_bin_ids.end(), 0); // fill with ids 0,1,2...num_bins
    auto histogram_sort_time_start = std::chrono::steady_clock::now();
    stable_sort(histogram_bin_ids.begin(), histogram_bin_ids.end(), [&value_histogram](int i1, int i2) {return value_histogram[i1] < value_histogram[i2];});
    auto histogram_sort_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_sort_seconds = histogram_sort_time_end-histogram_sort_time_start;
    sampling_timers.push_back(histogram_sort_seconds.count()); //s
	
    // Determine the target number of samples per bin
    int target_bin_samples = (tot_samples/num_bins);

    // Distribute samples across bins
    int samples;
    int remaining_tot_samples = tot_samples;
    vector<int> samples_per_bin = value_histogram;
    auto acceptance_function_time_start = std::chrono::steady_clock::now();
    for (int i = 0; i<num_bins; i++){
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

    // Clear acceptance histogram and fill with zeros
    fill(acceptance_histogram.begin(), acceptance_histogram.end(), 0);

    // Determine acceptance rate for each bin as the samples to take from that bin divided by the total samples in that bin
    for (int i = 0; i<num_bins; i++){
        // If bin has no samples, set acceptance to zero
        if (value_histogram[i] == 0){
            acceptance_histogram[i] = 0;
        }else{
            acceptance_histogram[i] = (float)samples_per_bin[i] / (float)value_histogram[i];
        }
    }
    auto acceptance_function_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> acceptance_function_seconds = acceptance_function_time_end-acceptance_function_time_start;
    sampling_timers.push_back(acceptance_function_seconds.count()); //s


    if (SAMPLING_PRINT){
        std::cout << "Acceptance Histogram Created!" << std::endl;
    }
}





/**
 * utilize_decision_histogram_based:
 * Determines whether previous timestep histogram
 * is similar enough to reuse or not.
 * 
 * Input:
 * current_block_histogram      - Current timestep's block histogram
 * reference_block_histogram    - Previous timestep's block histogram
 * num_bins                        - Number of histogram bins
 * reuse_flag                   - Flag to signal reuse
 * 
 * Output:
 * utilize                      - 1 = Utilize / 0 = Don't Utilize 
 **/
void utilize_decision_histogram_based(vector<int> current_block_histogram, vector<int> reference_block_histogram, int num_bins, int *utilize, int reuse_flag){
    // Check if this block in T - 1 reused samples from T - 2
    if (reference_block_histogram[0] == reuse_flag){
        // If so, do not consider reusing again to avoid domino effects during reconstruction
        *utilize = 0;
    }else{
        // If not, compare histograms through histogram intersection
        float score = 0;
        float q_sum = 0;
        for (int i = 0; i < num_bins; i++){
            if (reference_block_histogram[i] < current_block_histogram[i]){
                score = score + reference_block_histogram[i]; // add the minimum between the two
            }else{
                score = score + current_block_histogram[i]; // add the minimum between the two
            }
            q_sum = q_sum + current_block_histogram[i];
        }
        
        //Normalize Score
        // 1.0 means exact same, 0.0 means most different
        score = score/q_sum;
        
        // If exactly the same, reuse
        if (score == 1.0){
            *utilize = 1;
        // If not then don't reuse
        }else{
            *utilize = 0;
        }
    }
}


/**
 * utilize_decision_error_based:
 * Determines whether previous timestep error
 * is low enough to reuse or not.
 * 
 * Input:
 * 
 * Output:
 * utilize                      - 1 = Utilize / 0 = Don't Utilize 
 **/
void utilize_decision_error_based(vector<float> current_block_data, int reference_samples_per_block, vector<int> reference_block_sample_ids, vector<float> reference_block_sample_data, int *utilize, int reuse_flag, float error_threshold, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK){
    // Check if this block in T - 1 reused samples from T - 2
    if (reference_samples_per_block == reuse_flag || reference_block_sample_ids.size() == 0){
        // If so, do not consider reusing again to avoid domino effects during reconstruction
        *utilize = 0;
    }else{
        // If not, calculate RMSE between T-1 Samples and T current values
        float rmse_sum = 0;
        float local_rmse = 0;
        float rmse_block = 0;
        int local_id;
        int global_id;

        for (uint i = 0; i < reference_block_sample_ids.size(); i++){
            // for each T-1 sample in list
            global_id = reference_block_sample_ids[i];

            // int XB_COUNT = (XDIM/XBLOCK);
            // int YB_COUNT = (YDIM/YBLOCK);
            //int ZB_COUNT = (ZDIM/ZBLOCK); // unused
            // TODO if not evenly divisible, exit(0)
            int x_id = global_id % XDIM;
            int y_id = (global_id / XDIM) % YDIM;
            int z_id =  global_id / (XDIM*YDIM);
            int bx = (x_id/XBLOCK);
            int by = (y_id/YBLOCK);
            int bz = (z_id/ZBLOCK);
            int global_x = global_id % XDIM;
            int global_y = (global_id / XDIM) % YDIM;
            int global_z = global_id / (XDIM*YDIM);
            int local_x = global_x - (bx)*XBLOCK;
            int local_y = global_y - (by)*YBLOCK;
            int local_z = global_z - (bz)*ZBLOCK;
            local_id = local_x + (local_y*XBLOCK) + (local_z*XBLOCK*YBLOCK);

            local_rmse = pow((reference_block_sample_data[i] - current_block_data[local_id]), 2);
            rmse_sum = rmse_sum + local_rmse;
        }


        if (reference_block_sample_ids.size() > 0){
            // if there were samples in this block
            // calculate the RMSE of this block
            float temp = rmse_sum / reference_block_sample_ids.size();
            rmse_block = sqrt(temp);
        }else{
            // if there were no samples in this block
            // Do not reuse this block
            rmse_block = error_threshold + 1;
        }
        
        // If within tolerance, reuse
        if (rmse_block <= error_threshold){
            *utilize = 1;
        // If not then don't reuse
        }else{
            *utilize = 0;
        }
    }
}





////////////////////////
// SAMPLING FUNCTIONS //
////////////////////////

/**
 * value_histogram_importance_sampling:
 * Importance-Based Sampling Method that prioritizes
 * less frequent data values during the sampling 
 * procedure.
 * 
 * Inputs:
 * full_data                        - Original Input Data
 * num_blocks                       - Number of Blocks
 * XDIM,YDIM,ZDIM                   - Input Data Dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block Dimensions
 * num_bins                            - Number of bins in acceptance histogram
 * data_max                         - Maximum value in input data
 * data_min                         - Minimum value in input data
 * data_acceptance_histogram        - Input data acceptance histogram
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
void value_histogram_importance_sampling(vector<float> &full_data, int num_blocks, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int num_bins, \
float data_max, float data_min, vector<float> &data_acceptance_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers){
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    
    if (SAMPLING_PRINT){
        std::cout << "Beginning Importance-Based Sampling Procedure...\n";
    }
    // Calculate width of bins
    float range = data_max - data_min;
    float binWidth = range/num_bins;

    // Set timers
    float random_and_stencil_timer = 0;
    float sample_gathering_timer = 0;
    // Take samples from each block
    for(int block_id = 0; block_id < num_blocks; block_id++){
        // Get individual block data
        vector<int> block_data_ids;
        vector<float> block_data;
        
        // If data is going to be used with python, gather local IDs
        //if (PYTHON){
        //    get_block_data_w_local_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
        // Otherwise gather global IDs
        //} else {
        get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
        //}
        
        // Determine number of values in block
        int num_block_values = block_data_ids.size();
        
        // Get acceptance values vector for this block
        vector<float> prob_vals(num_block_values); 
        for (int i = 0; i < num_block_values; i++){
            // Get data values bin ID
            int binId = (block_data[i] - data_min) / binWidth;
            // Set its corresponding acceptance probability
            prob_vals[i] = data_acceptance_histogram[binId];
        }

        // Create random number vector for this block
        auto random_gen_time_start = std::chrono::steady_clock::now();
        vector<double> rand_vals(num_block_values); 
        for (int i = 0; i < num_block_values; i++){
            // Generate random number
            //float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            double r = distribution(generator);
            // Store in random number vector
            rand_vals[i] = r;
        }
        auto random_gen_time_end = std::chrono::steady_clock::now();

        // Create acceptance stencil
        auto stencil_time_start = std::chrono::steady_clock::now();
        vector<int> stencil(num_block_values);
        // Initialize with zeros
        fill(stencil.begin(), stencil.end(), 0);
        for (int i = 0; i < num_block_values; i++){
            // If the random value is below the probability set to keep data value
            if (rand_vals[i] <  prob_vals[i]){
                stencil[i] = 1;
            }
        }
        // Stop stencil timing
        auto stencil_time_end = std::chrono::steady_clock::now();

        // Store number of samples in current block
        int block_samples = count(stencil.begin(), stencil.end(), 1);
        samples_per_block.push_back(block_samples);  

        // Use stencil to get samples
        auto sample_gather_time_start = std::chrono::steady_clock::now();
        for (int i = 0; i < num_block_values; i++){
            if (stencil[i] == 1){
                // save to list
                sample_data_ids.push_back(block_data_ids[i]);
                sample_data.push_back(block_data[i]);
            }
        }
        auto sample_gather_time_end = std::chrono::steady_clock::now();

        // Add times to running sum
        std::chrono::duration<double> random_gen_seconds = random_gen_time_end-random_gen_time_start;
        std::chrono::duration<double> stencil_seconds = stencil_time_end-stencil_time_start;
        std::chrono::duration<double> sample_gather_seconds = sample_gather_time_end-sample_gather_time_start;
        random_and_stencil_timer = random_and_stencil_timer + random_gen_seconds.count() + stencil_seconds.count();
        sample_gathering_timer = sample_gathering_timer + sample_gather_seconds.count();
    }
    // Store timing information
    sampling_timers.push_back(random_and_stencil_timer); //s
    sampling_timers.push_back(sample_gathering_timer); //s

    if (SAMPLING_PRINT){
        std::cout << "Importance-Based Sampling Procedure Completed!\n";
    }
}



/**
 * omp_value_histogram_importance_sampling:
 * Importance-Based Sampling Method that prioritizes
 * less frequent data values during the sampling 
 * procedure.
 * 
 * Inputs:
 * full_data                        - Original Input Data
 * num_blocks                       - Number of Blocks
 * XDIM,YDIM,ZDIM                   - Input Data Dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block Dimensions
 * num_bins                            - Number of bins in acceptance histogram
 * data_max                         - Maximum value in input data
 * data_min                         - Minimum value in input data
 * data_acceptance_histogram        - Input data acceptance histogram
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
void omp_value_histogram_importance_sampling(int num_threads, vector<float> &full_data, int num_blocks, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, int num_bins, \
float data_max, float data_min, vector<float> &data_acceptance_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers){
    if (SAMPLING_PRINT){
        std::cout << "Beginning Importance-Based Sampling Procedure...\n";
    }

    // Generate random numbers equal to entire data set
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    int num_elements = full_data.size();
    double rand_vals[num_elements];
    auto random_gen_time_start = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        srand(int(time(NULL)) ^ omp_get_thread_num()); // use a different seed per thread
        #pragma omp for
        for (int i = 0; i < num_elements; i++){
            double r = distribution(generator);
            // Store in random number vector
            rand_vals[i] = r;
        }
    }

    auto random_gen_time_end = std::chrono::steady_clock::now();

    // Determine number of elements per thread
    int n_per_thread;
    if (num_elements < num_threads){
        n_per_thread = 1;
    } else {
        n_per_thread = num_elements / num_threads;
    }
    // Set number of threads
	omp_set_num_threads(num_threads);

    // Create acceptance stencil
    double stencil[num_elements] = {0};

    // Resize samples per block vector
    samples_per_block.resize(num_blocks,0);

    // Iterate over all data values
    auto stencil_time_start = std::chrono::steady_clock::now();
    auto stencil_time_end = std::chrono::steady_clock::now();
    auto sample_gather_time_start = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        // Have each thread create individual copies of sample data arrays
        vector<int> private_samples_per_block(num_blocks, 0);
        vector<int> private_sample_data_ids;
        vector<float> private_sample_data;
        int private_total_samples_gathered = 0;

        // Iterate over all elements
        #pragma omp for schedule(static, n_per_thread)
        for (int global_id = 0; global_id < num_elements; global_id++){
            // Calculate width of bins
            float range = data_max - data_min;
            float bin_width = range/num_bins;

            // Determine which bin this value belongs to
            int bin_id = (full_data[global_id] - data_min) / bin_width;

            // Handle Edge Cases
            if (bin_id > num_bins-1){
                bin_id = num_bins - 1;
            }
            if (bin_id < 0){
                bin_id = 0;
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
            stencil[global_id] = data_acceptance_histogram[bin_id] - rand_vals[global_id];
    
            // If sample chosen to be saved, increment samples saved in that block
            if (stencil[global_id] > 0){
                private_samples_per_block[block_id] = private_samples_per_block[block_id] + 1;
                private_sample_data.push_back(full_data[global_id]);
                private_sample_data_ids.push_back(global_id);
                private_total_samples_gathered = private_total_samples_gathered + 1;
            }
        }

        // Have thread zero start and stop timers
        if (omp_get_thread_num() == 0){
            stencil_time_end = std::chrono::steady_clock::now();
            sample_gather_time_start = std::chrono::steady_clock::now();
        }

        #pragma omp critical
        {
            // Have each thread aggreate samples per block taken
            for (int i = 0; i < num_blocks; i++){
                samples_per_block[i] = samples_per_block[i] + private_samples_per_block[i];
            }

            // Have each thread add their samples to the total sample data arrays
            for (int i = 0; i < private_total_samples_gathered; i++){
                sample_data.push_back(private_sample_data[i]);
                sample_data_ids.push_back(private_sample_data_ids[i]);
            }
        }
    }
    auto sample_gather_time_end = std::chrono::steady_clock::now();


    // Add times to running sum
    std::chrono::duration<double> random_gen_seconds = random_gen_time_end-random_gen_time_start;
    std::chrono::duration<double> stencil_seconds = stencil_time_end-stencil_time_start;
    std::chrono::duration<double> sample_gather_seconds = sample_gather_time_end-sample_gather_time_start;
    float random_and_stencil_timer = random_gen_seconds.count() + stencil_seconds.count();
    float sample_gathering_timer = sample_gather_seconds.count();
    
    // Store timing information
    sampling_timers.push_back(random_and_stencil_timer); //s
    sampling_timers.push_back(sample_gathering_timer); //s

    if (SAMPLING_PRINT){
        std::cout << "Importance-Based Sampling Procedure Completed!\n";
    }
}



/**
 * add_random_samples:
 * Add random samples when using the reuse method and extra 
 * space becomes available
 * 
 * Inputs:
 * full_data                        - Original Input Data
 * reuse_flag                       - Flag to signal reuse
 * num_blocks                       - Number of Blocks
 * sample_ratio                     - Percentage of data to keep as samples
 * XDIM,YDIM,ZDIM                   - Input Data Dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block Dimensions
 * max_samples                      - Maximum number of samples to gather
 * tot_samples                      - Total current samples gathered
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
void add_random_samples(vector<float> &full_data, int reuse_flag, int num_blocks, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, int max_samples, int *tot_samples){
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    
    if (SAMPLING_PRINT){
        std::cout << "Adding Additional Random Samples to Fill Budget...\n";
    }
    // For each block, add random samples if the block is not reusing samples
    for(int block_id = 0; block_id < num_blocks; block_id++){
        // If block is not flagged as reusing
        if (samples_per_block[block_id] != reuse_flag){
            // Get individual block data
            vector<int> block_data_ids;
            vector<float> block_data;

            // If data is going to be used with python, gather local IDs
            //if (PYTHON){
            //   get_block_data_w_local_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
            // Otherwise gather global IDs
            //} else {
            get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
            //}
            
            // Determine number of values in block
            int num_block_values = block_data_ids.size();
             
            // Create random number vector for this block
            vector<double> rand_vals(num_block_values);
            for (int i = 0; i < num_block_values; i++){
                // Generate random number
                //float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                double r = distribution(generator);
                // Store in random number vector
                rand_vals[i] = r;
            }

            // Create acceptance stencil
            vector<int> stencil(num_block_values);
            // Initialize with zeros
            fill(stencil.begin(), stencil.end(), 0);
            for (int i = 0; i < num_block_values; i++){
                // If the random value is below the probability set to keep data value
                if (rand_vals[i] <  sample_ratio){
                    stencil[i] = 1;
                }
            }

            // Check to see if the additional samples will hit sample limit
            if (*tot_samples + count(stencil.begin(), stencil.end(), 1) >= max_samples){
                // TODO: Ideally, remove some of them, but for now, just break
                std::cout << "Target samples reached via extra randoms!\n";
                break;
            }

            // Use stencil to get samples
            for (int i = 0; i < num_block_values; i++){
                if (stencil[i] == 1){
                    // save to list
                    sample_data_ids.push_back(block_data_ids[i]);
                    sample_data.push_back(block_data[i]);
                }
            }

            // Update number of samples in current block
            int block_samples = count(stencil.begin(), stencil.end(), 1) + samples_per_block[block_id];
            samples_per_block[block_id] = block_samples;

            // Update total number of samples
            *tot_samples = *tot_samples + count(stencil.begin(), stencil.end(), 1);
        }
    }
    if (SAMPLING_PRINT){
        std::cout << "Random Samples Added!\n";
    }
}



/**
 * histogram_reuse_method:
 * Uses importance-based sampling method approach but
 * first checks to see if previous samples can be kept
 * instead.
 * 
 * Inputs:
 * full_data                        - Current timestep data
 * reuse_flag                       - Flag to signal reuse
 * num_blocks                       - Number of Blocks
 * num_bins                            - Number of bins in acceptance histogram
 * sample_ratio                     - Percentage of data to keep as samples
 * XDIM,YDIM,ZDIM                   - Current timestep data dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block dimensions
 * reference_histogram              - Previous timestep histogram information
 * current_timestep_max             - Maximum value in current timestep
 * current_timestep_min             - Minimum value in current timestep
 * lifetime_max                     - Maximum value expected over entire simulation
 * lifetime_min                     - Minimum value expected over entire simulation
 * data_acceptance_histogram        - Current timestep data acceptance histogram
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
int histogram_reuse_method(vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers){
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    
    if (SAMPLING_PRINT){
        std::cout << "Beginning Histogram Reuse Importance-Based Sampling Procedure...\n";
    }
    // Calculate width of bins
    float range = current_timestep_max - current_timestep_min;
    float binWidth = range/num_bins;

    // Determine maximum samples and setup loop variable
    int max_samples = sample_ratio*full_data.size();
    int tot_samples = 0;
    int num_blocks_reused = 0;

    // Set timers
    float block_histogram_construction_timer = 0;
    float block_comparison_and_utilization_timer = 0;
    float random_and_stencil_timer = 0;
    float additional_random_sampling_timer = 0;
    float sample_gathering_timer = 0;
    // Take samples from each block
    for(int block_id = 0; block_id < num_blocks; block_id++){
        

        // Get individual block data
        vector<int> block_data_ids;
        vector<float> block_data;

        

        // If data is going to be used with python, gather local IDs
        //if (PYTHON){
        //    get_block_data_w_local_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
        // Otherwise gather global IDs
        //} else {
        get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
        //}

        // Determine number of values in block
        int num_block_values = block_data_ids.size();
        
        // Initialize block histogram with all zeros
        vector<int> block_histogram(num_bins);
        fill(block_histogram.begin(), block_histogram.end(), 0);

        // Determine reference block histogram start and end
        const int block_histogram_reference_start = num_bins*block_id;
        const int block_histogram_reference_end = block_histogram_reference_start + num_bins;
        vector<int>::const_iterator first = reference_histogram.begin() + block_histogram_reference_start;
        vector<int>::const_iterator last = reference_histogram.begin() + block_histogram_reference_end;
        vector<int> block_histogram_reference(first, last);
        
        // Ensure histogram is valid
        if (block_histogram_reference_end - block_histogram_reference_start != num_bins){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC A!!"<< block_histogram_reference_end - block_histogram_reference_start << " vs " << num_bins << "\n";
            exit(0);
        }
        if (block_histogram_reference.size() != (uint) num_bins){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC B!!"<< block_histogram_reference.size() << " vs " << num_bins << "\n";
            exit(0);
        }
        if (reference_histogram.size() != (uint) num_bins*num_blocks){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC C!!"<< reference_histogram.size() << " vs " << num_bins*num_blocks << "\n";
            exit(0);
        }


        // Start block histogram timing
        auto block_histogram_time_start = std::chrono::steady_clock::now();

        // Construct block histogram
        data_histogram(block_data, block_histogram, num_bins, lifetime_max, lifetime_min); // is histogram normalized? if not, divide by XDIM*YDIM*ZDIM

        // End block histogram timing
        auto block_histogram_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> block_histogram_seconds = block_histogram_time_end-block_histogram_time_start;
        block_histogram_construction_timer = block_histogram_construction_timer + block_histogram_seconds.count();


        // Compare histograms to determine whether to reuse
        int utilize = 0;
        auto comparison_and_utilization_time_start = std::chrono::steady_clock::now();
        utilize_decision_histogram_based(block_histogram, block_histogram_reference, num_bins, &utilize, reuse_flag);
        auto comparison_and_utilization_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> comparison_and_utilization_seconds = comparison_and_utilization_time_end-comparison_and_utilization_time_start;
        block_comparison_and_utilization_timer = block_comparison_and_utilization_timer + comparison_and_utilization_seconds.count();

        // When utilizing previous data for this block
        if (utilize == 1){
            // Increment number of blocks reused
            num_blocks_reused = num_blocks_reused + 1;
            // append reuse_flag to signify reuse
            samples_per_block.push_back(reuse_flag);
            // Overwrite block in reference histogram with reuse_flag for next timestep
            reference_histogram[block_histogram_reference_start] = reuse_flag;

        // When not utilizing previous data for this block
        }else{
            // Overwrite block in reference histogram with current histogram for next timestep
            int index = 0;
            for(int i = block_histogram_reference_start; i < block_histogram_reference_end; i++){
                reference_histogram[i] = block_histogram[index];
                index++;
            }

            // Gather samples for this block
            // Get acceptance values vector for this block
            vector<float> prob_vals(num_block_values);
            for (int i = 0; i < num_block_values; i++){
                // Get data values bin ID
                int binId = (block_data[i] - current_timestep_min) / binWidth;
                // Set its corresponding acceptance probability
                prob_vals[i] = data_acceptance_histogram[binId];
            }

            // Create random number vector for this block
            auto random_gen_time_start = std::chrono::steady_clock::now();
            vector<double> rand_vals(num_block_values); 
            for (int i = 0; i < num_block_values; i++){
                // Generate random number
                //float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                double r = distribution(generator);
                // Store in random number vector
                rand_vals[i] = r;
            }
            auto random_gen_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> random_gen_seconds = random_gen_time_end-random_gen_time_start;
            random_and_stencil_timer = random_and_stencil_timer + random_gen_seconds.count();

            // Create acceptance stencil
            auto stencil_time_start = std::chrono::steady_clock::now();
            vector<int> stencil(num_block_values);
            // Initialize with zeros
            fill(stencil.begin(), stencil.end(), 0);
            for (int i = 0; i < num_block_values; i++){
                // If the random value is below the probability set to keep data value
                if (rand_vals[i] <  prob_vals[i]){
                    stencil[i] = 1;
                }
            }
            auto stencil_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> stencil_seconds = stencil_time_end-stencil_time_start;
            random_and_stencil_timer = random_and_stencil_timer + stencil_seconds.count();

            // Store number of samples in current block
            int block_samples = count(stencil.begin(), stencil.end(), 1);
            samples_per_block.push_back(block_samples);  

            // Update number of samples taken
            tot_samples = tot_samples + block_samples;

            // Use stencil to get samples
            auto sample_gather_time_start = std::chrono::steady_clock::now();
            for (int i = 0; i < num_block_values; i++){
                if (stencil[i] == 1){
                    // save to list
                    sample_data_ids.push_back(block_data_ids[i]);
                    sample_data.push_back(block_data[i]);
                }
            }
            auto sample_gather_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> sample_gather_seconds = sample_gather_time_end-sample_gather_time_start;
            sample_gathering_timer = sample_gathering_timer + sample_gather_seconds.count();
        }
        
        // Ensure reference histogram is still correct
        if (reference_histogram.size() != (uint) num_bins*num_blocks){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC D!!"<< reference_histogram.size() << " vs " << num_bins*num_blocks << "\n";
            exit(0);
        }
    }

    // Print metrics
    float reused_percentage = ((float)num_blocks_reused/(float)num_blocks)*100;

    // Redisperse extra samples via random sampling
    int samples_remaining = max_samples - tot_samples;
    float new_sample_ratio;
    // Check to make sure some blocks were reused and some samples remain
    if (((num_blocks - num_blocks_reused) != num_blocks) && (samples_remaining > 0)){
        // If only some blocks were reused
        if (num_blocks - num_blocks_reused != 0){
            new_sample_ratio =  ((float)samples_remaining/((float)num_blocks - (float)num_blocks_reused))/(float)(XBLOCK*YBLOCK*ZBLOCK);
        // If all blocks were reused
        }else{
            new_sample_ratio = sample_ratio;
        }
    // If no blocks reused, no extra samples can be added
    }else{
        new_sample_ratio = 0;
    }
    if (SAMPLING_PRINT){
        std::cout << "Histogram Reuse Importance-Based Sampling Procedure Completed!\n";
        std::cout << "True Samples Taken : " << tot_samples << " samples\n";
        std::cout << "Number of Blocks Reused: " << num_blocks_reused << " -> " << reused_percentage << "%\n";
        std::cout << "New Sample ratio: " << new_sample_ratio << "\n";
    }
    
    // Add Random Samples
    if (new_sample_ratio > 0 && num_blocks_reused > 0){
        auto additional_random_time_start = std::chrono::steady_clock::now();
        add_random_samples(full_data, reuse_flag, num_blocks, new_sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, max_samples, &tot_samples);
        auto additional_random_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> additional_random_seconds = additional_random_time_end-additional_random_time_start;
        additional_random_sampling_timer = additional_random_sampling_timer + additional_random_seconds.count();
        if (SAMPLING_PRINT){
            std::cout << "Final True Taken : " << tot_samples << " samples\n";
        }
    }

    // Store timing information
    sampling_timers.push_back(block_histogram_construction_timer); //s
    sampling_timers.push_back(block_comparison_and_utilization_timer); //s
    sampling_timers.push_back(random_and_stencil_timer); //s
    sampling_timers.push_back(additional_random_sampling_timer); //s
    sampling_timers.push_back(sample_gathering_timer); //s

    
    return(num_blocks_reused);
}



/**
 * omp_histogram_reuse_method:
 * Uses importance-based sampling method approach but
 * first checks to see if previous samples can be kept
 * instead.
 * 
 * Inputs:
 * full_data                        - Current timestep data
 * reuse_flag                       - Flag to signal reuse
 * num_blocks                       - Number of Blocks
 * num_bins                            - Number of bins in acceptance histogram
 * sample_ratio                     - Percentage of data to keep as samples
 * XDIM,YDIM,ZDIM                   - Current timestep data dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block dimensions
 * reference_histogram              - Previous timestep histogram information
 * current_timestep_max             - Maximum value in current timestep
 * current_timestep_min             - Minimum value in current timestep
 * lifetime_max                     - Maximum value expected over entire simulation
 * lifetime_min                     - Minimum value expected over entire simulation
 * data_acceptance_histogram        - Current timestep data acceptance histogram
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
int omp_histogram_reuse_method(int num_threads, vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers){
    if (SAMPLING_PRINT){
        std::cout << "Beginning Histogram Reuse Importance-Based Sampling Procedure...\n";
    }

    // Generate random numbers equal to entire data set
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    int num_elements = full_data.size();
    double rand_vals [num_elements];
    auto random_gen_time_start = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        srand(int(time(NULL)) ^ omp_get_thread_num());
        #pragma omp for
        for (int i = 0; i < num_elements; i++){
            double r = distribution(generator);
            // Store in random number vector
            rand_vals[i] = r;
        }
    }
    auto random_gen_time_end = std::chrono::steady_clock::now();

    // Determine number of elements per thread
    int n_per_thread;
    if (num_elements < num_threads){
        n_per_thread = 1;
    } else {
        n_per_thread = num_elements / num_threads;
    }
    // Set number of threads
	omp_set_num_threads(num_threads);

    // Determine maximum samples and setup loop variable
    int max_samples = sample_ratio*full_data.size();
    int tot_samples = 0;
    int num_blocks_reused = 0;

    // Build current block histograms
    vector<int> current_histogram_list(num_bins*num_blocks, 0);

    // Create acceptance stencil
    double stencil[num_elements] = {0};

    // Resize samples per block vector
    samples_per_block.resize(num_blocks,0);

    // Iterate over all data values
    auto block_histogram_construction_time_start = std::chrono::steady_clock::now();
    auto block_histogram_construction_time_end = std::chrono::steady_clock::now();
    auto block_comparison_and_utilization_time_start = std::chrono::steady_clock::now();
    auto block_comparison_and_utilization_time_end = std::chrono::steady_clock::now();
    auto stencil_time_start = std::chrono::steady_clock::now();
    auto stencil_time_end = std::chrono::steady_clock::now();
    auto sample_gather_time_start = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        // Have all threads create private current block histograms
        vector<int> private_current_histogram_list(num_bins*num_blocks, 0);

        // Iterate over all data elements to build histogram
        #pragma omp for schedule(static, n_per_thread)
        for (int global_id = 0; global_id < num_elements; global_id++){
            // Calculate width of bins
            float range = lifetime_max - lifetime_min;
            float bin_width = range/num_bins;

            // Determine which bin this value belongs to
            int bin_id = (full_data[global_id] - lifetime_min) / bin_width;

            // Handle Edge Cases
            if (bin_id > num_bins-1){
                bin_id = num_bins - 1;
            }
            if (bin_id < 0){
                bin_id = 0;
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
            int offset_id = block_id*(num_bins) + bin_id;

            // Increment the number of values in this bin
            private_current_histogram_list[offset_id] = private_current_histogram_list[offset_id] + 1;
        }

        #pragma omp critical
        {
            // Have each thread aggregate histograms together
            for (int i = 0; i < num_bins*num_blocks; i++){
                current_histogram_list[i] = current_histogram_list[i] + private_current_histogram_list[i];
            }
        }

        // Have thread zero start and stop timers
        if (omp_get_thread_num() == 0){
            block_histogram_construction_time_end = std::chrono::steady_clock::now();
            block_comparison_and_utilization_time_start = std::chrono::steady_clock::now();
        }

        // Ensure all threads have completed the previous section
        #pragma omp barrier

        // Have each thread create private arrays
        vector<int> private_block_ids;
        vector<int> private_samples_per_block;
        vector<int> private_reference_histogram(num_bins*num_blocks,0);
        int private_blocks_processed = 0;
        int private_num_blocks_reused = 0;
        
        // Check each block for utilization
        #pragma omp for 
        for(int block_id = 0; block_id < num_blocks; block_id++){
            // Determine current block histogram start and end
            const int block_histogram_current_start = num_bins*block_id;
            const int block_histogram_current_end = block_histogram_current_start + num_bins;
            vector<int>::const_iterator current_first = current_histogram_list.begin() + block_histogram_current_start;
            vector<int>::const_iterator current_last = current_histogram_list.begin() + block_histogram_current_end;
            vector<int> block_histogram_current(current_first, current_last);

            // Determine reference block histogram start and end
            const int block_histogram_reference_start = num_bins*block_id;
            const int block_histogram_reference_end = block_histogram_reference_start + num_bins;
            vector<int>::const_iterator reference_first = reference_histogram.begin() + block_histogram_reference_start;
            vector<int>::const_iterator reference_last = reference_histogram.begin() + block_histogram_reference_end;
            vector<int> block_histogram_reference(reference_first, reference_last);

            // Compare histograms to determine whether to reuse
            int utilize = 0;
            //std::cout << "Block: " << block_id << "/" << num_blocks << ", Num bins: " << num_bins << "\n";
            utilize_decision_histogram_based(block_histogram_current, block_histogram_reference, num_bins, &utilize, reuse_flag);

            // Mark for reuse or not
            if (utilize == 1){
                // Increment number of blocks reused
                private_num_blocks_reused = private_num_blocks_reused + 1;

                // Append reuse_flag to signify reuse
                private_samples_per_block.push_back(reuse_flag);

                // Overwrite block in reference histogram with reuse_flag for next timestep
                private_reference_histogram[block_histogram_reference_start] = reuse_flag;
            } else {
                // Append 0 to signify reuse
                private_samples_per_block.push_back(0);

                // Overwrite block in reference histogram with current histogram for next timestep
                int index = 0;
                for(int i = block_histogram_reference_start; i < block_histogram_reference_end; i++){
                    private_reference_histogram[i] = block_histogram_current[index];
                    index++;
                }
            }

            // Append which block this thread worked on
            private_block_ids.push_back(block_id);
            private_blocks_processed = private_blocks_processed + 1;
        }

        #pragma omp critical
        {
            // Have each thread aggregate their number of blocks reused
            num_blocks_reused = num_blocks_reused + private_num_blocks_reused;

            // Have each thread aggregate their samples per block arrays and reference histograms
            for (int i = 0; i < private_blocks_processed; i++){
                int block_id = private_block_ids[i];
                samples_per_block[block_id] = private_samples_per_block[i];

                // Determine reference block histogram start and end
                const int block_histogram_reference_start = num_bins*block_id;
                const int block_histogram_reference_end = block_histogram_reference_start + num_bins;

                if (private_samples_per_block[i] == reuse_flag){
                    reference_histogram[block_histogram_reference_start] = reuse_flag;
                } else {
                    // Overwrite block in reference histogram with current histogram for next timestep
                    int index = block_histogram_reference_start;
                    for(int i = block_histogram_reference_start; i < block_histogram_reference_end; i++){
                        reference_histogram[i] = private_reference_histogram[index];
                        index++;
                    }
                }
            }
        }

        // Have thread zero start and stop timers
        if (omp_get_thread_num() == 0){
            block_comparison_and_utilization_time_end = std::chrono::steady_clock::now();
            stencil_time_start = std::chrono::steady_clock::now();
        }

        // Ensure all threads have completed the previous section
        #pragma omp barrier

        // Have each thread create individual copies of sample data arrays
        private_samples_per_block.resize(num_blocks, 0);
        vector<int> private_sample_data_ids;
        vector<float> private_sample_data;
        int private_total_samples_gathered = 0;

        // Iterate over all elements
        #pragma omp for schedule(static, n_per_thread)
        for (int global_id = 0; global_id < num_elements; global_id++){
            // Calculate width of bins
            float range = current_timestep_max - current_timestep_min;
            float bin_width = range/num_bins;

            // Determine which bin this value belongs to
            int bin_id = (full_data[global_id] - current_timestep_min) / bin_width;

            // Handle Edge Cases
            if (bin_id > num_bins-1){
                bin_id = num_bins - 1;
            }
            if (bin_id < 0){
                bin_id = 0;
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
            if (samples_per_block[block_id] == reuse_flag){
                stencil[global_id] = 0;
            } else {
                // Determine whether to save sample or not
                // If difference is positive, save sample, else dont save
                stencil[global_id] = data_acceptance_histogram[bin_id] - rand_vals[global_id];

                // If sample chosen to be saved, increment samples saved in that block
                if (stencil[global_id] > 0){
                    private_samples_per_block[block_id] = private_samples_per_block[block_id] + 1;
                    private_sample_data.push_back(full_data[global_id]);
                    private_sample_data_ids.push_back(global_id);
                    private_total_samples_gathered = private_total_samples_gathered + 1;
                }
            }
        }

        // Have thread zero start and stop timers
        if (omp_get_thread_num() == 0){
            stencil_time_end = std::chrono::steady_clock::now();
            sample_gather_time_start = std::chrono::steady_clock::now();
        }

        #pragma omp critical
        {
            // Have each thread aggreate samples per block taken
            for (int i = 0; i < num_blocks; i++){
                samples_per_block[i] = samples_per_block[i] + private_samples_per_block[i];
            }

            // Have each thread add their samples to the total sample data arrays
            for (int i = 0; i < private_total_samples_gathered; i++){
                sample_data.push_back(private_sample_data[i]);
                sample_data_ids.push_back(private_sample_data_ids[i]);   
            }
        }
    }
    auto sample_gather_time_end = std::chrono::steady_clock::now();

    // Print metrics
    float reused_percentage = ((float)num_blocks_reused/(float)num_blocks)*100;

    // Redisperse extra samples via random sampling
    tot_samples = sample_data_ids.size();
    int samples_remaining = max_samples - tot_samples;
    float new_sample_ratio;
    // Check to make sure some blocks were reused and some samples remain
    if (((num_blocks - num_blocks_reused) != num_blocks) && (samples_remaining > 0)){
        // If only some blocks were reused
        if (num_blocks - num_blocks_reused != 0){
            new_sample_ratio =  ((float)samples_remaining/((float)num_blocks - (float)num_blocks_reused))/(float)(XBLOCK*YBLOCK*ZBLOCK);
        // If all blocks were reused
        }else{
            new_sample_ratio = sample_ratio;
        }
    // If no blocks reused, no extra samples can be added
    }else{
        new_sample_ratio = 0;
    }
    if (SAMPLING_PRINT){
        std::cout << "Histogram Reuse Importance-Based Sampling Procedure Completed!\n";
        std::cout << "True Samples Taken : " << tot_samples << " samples\n";
        std::cout << "Number of Blocks Reused: " << num_blocks_reused << " -> " << reused_percentage << "%\n";
        std::cout << "New Sample ratio: " << new_sample_ratio << "\n";
    }
    
    // Add Random Samples
    float additional_random_sampling_timer = 0;
    if (new_sample_ratio > 0 && num_blocks_reused > 0){
        auto additional_random_time_start = std::chrono::steady_clock::now();
        add_random_samples(full_data, reuse_flag, num_blocks, new_sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, max_samples, &tot_samples);
        auto additional_random_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> additional_random_seconds = additional_random_time_end-additional_random_time_start;
        additional_random_sampling_timer = additional_random_sampling_timer + additional_random_seconds.count();

        if (SAMPLING_PRINT){
            std::cout << "Final True Taken : " << tot_samples << " samples\n";
        }
    }

    // Store timing information
    std::chrono::duration<double> block_histogram_construction_seconds = block_histogram_construction_time_end-block_histogram_construction_time_start;
    sampling_timers.push_back(block_histogram_construction_seconds.count()); //s
    std::chrono::duration<double> block_comparison_and_utilization_seconds = block_comparison_and_utilization_time_end-block_comparison_and_utilization_time_start;
    sampling_timers.push_back(block_comparison_and_utilization_seconds.count()); //s
    std::chrono::duration<double> stencil_seconds = stencil_time_end-stencil_time_start;
    std::chrono::duration<double> random_gen_seconds = random_gen_time_end-random_gen_time_start;
    float random_and_stencil_seconds = random_gen_seconds.count() + stencil_seconds.count();
    sampling_timers.push_back(random_and_stencil_seconds); //s
    sampling_timers.push_back(additional_random_sampling_timer); //s
    std::chrono::duration<double> sample_gather_seconds = sample_gather_time_end-sample_gather_time_start;
    sampling_timers.push_back(sample_gather_seconds.count()); //s

    return(num_blocks_reused);
}



/**
 * error_reuse_method:
 * Uses importance-based sampling method approach but
 * first checks to see if previous samples can be kept
 * instead, using RMSE.
 * 
 * Inputs:
 * full_data                        - Current timestep data
 * reuse_flag                       - Flag to signal reuse
 * num_blocks                       - Number of Blocks
 * num_bins                            - Number of bins in acceptance histogram
 * sample_ratio                     - Percentage of data to keep as samples
 * XDIM,YDIM,ZDIM                   - Current timestep data dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block dimensions
 * reference_histogram              - Previous timestep histogram information
 * current_timestep_max             - Maximum value in current timestep
 * current_timestep_min             - Minimum value in current timestep
 * lifetime_max                     - Maximum value expected over entire simulation
 * lifetime_min                     - Minimum value expected over entire simulation
 * data_acceptance_histogram        - Current timestep data acceptance histogram
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
int error_reuse_method(vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers, float error_threshold, vector<int> ref_samples_per_block, vector<int> reference_sample_ids, vector<float> reference_sample_data){
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    
    if (SAMPLING_PRINT){
        std::cout << "Beginning Error Reuse Importance-Based Sampling Procedure...\n";
    }
    // Calculate width of bins
    float range = current_timestep_max - current_timestep_min;
    float binWidth = range/num_bins;

    // Determine maximum samples and setup loop variable
    int max_samples = sample_ratio*full_data.size();
    int tot_samples = 0;
    int num_blocks_reused = 0;

    // Set timers
    float block_comparison_and_utilization_timer = 0;
    float random_and_stencil_timer = 0;
    float additional_random_sampling_timer = 0;
    float sample_gathering_timer = 0;
    // Take samples from each block
    for(int block_id = 0; block_id < num_blocks; block_id++){
        // Start block data timing
        //auto block_data_time_start = std::chrono::steady_clock::now();
        // Get individual block data
        vector<int> block_data_ids;
        vector<float> block_data;

        vector<int> ref_block_data_ids;
        vector<float> ref_block_data;

        // If data is going to be used with python, gather local IDs
        //if (PYTHON){
        //    get_block_data_w_local_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);
        // Otherwise gather global IDs
        //} else {
        get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);

        //}

        // Determine number of values in block
        int num_block_values = block_data_ids.size();

        // get T-1 samples in this block
        get_block_samples_w_global_ids(reference_sample_ids, reference_sample_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, ref_block_data_ids, ref_block_data);

        // Stop block data timing
        //auto block_data_time_end = std::chrono::steady_clock::now();

        // Ensure all samples collected
        /*
        if (ref_samples_per_block[block_id] != ref_block_data_ids.size()){
            std::cout << "Not all reference samples collected!" << std::endl;
            std::cout << "Expected Samples: " << ref_samples_per_block[block_id] << " Found Samples: " << ref_block_data_ids.size() << std::endl;
        }
        */

        // Compare error to determine whether to reuse
        auto comparison_and_utilization_time_start = std::chrono::steady_clock::now();
        int utilize = 0;
        utilize_decision_error_based(block_data, ref_samples_per_block[block_id], ref_block_data_ids, ref_block_data, &utilize, reuse_flag, error_threshold, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK);
        auto comparison_and_utilization_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> comparison_and_utilization_seconds = comparison_and_utilization_time_end-comparison_and_utilization_time_start;
        block_comparison_and_utilization_timer = block_comparison_and_utilization_timer + comparison_and_utilization_seconds.count();

        // When utilizing previous data for this block
        if (utilize == 1){
            // Increment number of blocks reused
            num_blocks_reused = num_blocks_reused + 1;
            // append reuse_flag to signify reuse
            samples_per_block.push_back(reuse_flag);
            // Overwrite block in reference histogram with reuse_flag for next timestep
            // NOTE This cant work because we have no way of knowing for which block this -1 is for so we will have to rely on samples_per_block
            //reference_sample_ids[0] = reuse_flag;
            //reference_sample_data[0] = reuse_flag;
            
        // When not utilizing previous data for this block
        }else{
            // Gather samples for this block
            // Get acceptance values vector for this block
            vector<float> prob_vals(num_block_values);
            for (int i = 0; i < num_block_values; i++){
                // Get data values bin ID
                int binId = (block_data[i] - current_timestep_min) / binWidth;
                // Set its corresponding acceptance probability
                prob_vals[i] = data_acceptance_histogram[binId];
            }

            // Create random number vector for this block
            auto random_gen_time_start = std::chrono::steady_clock::now();
            vector<double> rand_vals(num_block_values); 
            for (int i = 0; i < num_block_values; i++){
                // Generate random number
                //float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                double r = distribution(generator);
                // Store in random number vector
                rand_vals[i] = r;
            }
            auto random_gen_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> random_gen_seconds = random_gen_time_end-random_gen_time_start;
            random_and_stencil_timer = random_and_stencil_timer + random_gen_seconds.count();

            // Create acceptance stencil
            auto stencil_time_start = std::chrono::steady_clock::now();
            vector<int> stencil(num_block_values);
            // Initialize with zeros
            fill(stencil.begin(), stencil.end(), 0);
            for (int i = 0; i < num_block_values; i++){
                // If the random value is below the probability set to keep data value
                if (rand_vals[i] <  prob_vals[i]){
                    stencil[i] = 1;
                }
            }
            auto stencil_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> stencil_seconds = stencil_time_end-stencil_time_start;
            random_and_stencil_timer = random_and_stencil_timer + stencil_seconds.count();

            // Store number of samples in current block
            int block_samples = count(stencil.begin(), stencil.end(), 1);
            samples_per_block.push_back(block_samples);  

            // Update number of samples taken
            tot_samples = tot_samples + block_samples;

            // Use stencil to get samples
            auto sample_gather_time_start = std::chrono::steady_clock::now();
            for (int i = 0; i < num_block_values; i++){
                if (stencil[i] == 1){
                    // save to list
                    sample_data_ids.push_back(block_data_ids[i]);
                    sample_data.push_back(block_data[i]);
                }
            }
            auto sample_gather_time_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> sample_gather_seconds = sample_gather_time_end-sample_gather_time_start;
            sample_gathering_timer = sample_gathering_timer + sample_gather_seconds.count();
        }
    }

    // Print metrics
    float reused_percentage = ((float)num_blocks_reused/(float)num_blocks)*100;

    // Redisperse extra samples via random sampling
    int samples_remaining = max_samples - tot_samples;
    float new_sample_ratio;
    // Check to make sure some blocks were reused and some samples remain
    if (((num_blocks - num_blocks_reused) != num_blocks) && (samples_remaining > 0)){
        // If only some blocks were reused
        if (num_blocks - num_blocks_reused != 0){
            new_sample_ratio =  ((float)samples_remaining/((float)num_blocks - (float)num_blocks_reused))/(float)(XBLOCK*YBLOCK*ZBLOCK);
        // If all blocks were reused
        }else{
            new_sample_ratio = sample_ratio;
        }
    // If no blocks reused, no extra samples can be added
    }else{
        new_sample_ratio = 0;
    }
    if (SAMPLING_PRINT){
        std::cout << "Histogram Reuse Importance-Based Sampling Procedure Completed!\n";
        std::cout << "True Samples Taken : " << tot_samples << " samples\n";
        std::cout << "Number of Blocks Reused: " << num_blocks_reused << " -> " << reused_percentage << "%\n";
        std::cout << "New Sample ratio: " << new_sample_ratio << "\n";
    }
    
    // Add Random Samples
    if (new_sample_ratio > 0 && num_blocks_reused > 0){
        auto additional_random_time_start = std::chrono::steady_clock::now();
        add_random_samples(full_data, reuse_flag, num_blocks, new_sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, max_samples, &tot_samples);
        auto additional_random_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> additional_random_seconds = additional_random_time_end-additional_random_time_start;
        additional_random_sampling_timer = additional_random_sampling_timer + additional_random_seconds.count();
        if (SAMPLING_PRINT){
            std::cout << "Final True Taken : " << tot_samples << " samples\n";
        }
    }

    // Store timing information
    sampling_timers.push_back(block_comparison_and_utilization_timer); // s
    sampling_timers.push_back(random_and_stencil_timer); // s
    sampling_timers.push_back(additional_random_sampling_timer); // s
    sampling_timers.push_back(sample_gathering_timer); // s 

    return(num_blocks_reused);
}



/**
 * omp_error_reuse_method:
 * Uses importance-based sampling method approach but
 * first checks to see if previous samples can be kept
 * instead, using RMSE.
 * 
 * Inputs:
 * full_data                        - Current timestep data
 * reuse_flag                       - Flag to signal reuse
 * num_blocks                       - Number of Blocks
 * num_bins                            - Number of bins in acceptance histogram
 * sample_ratio                     - Percentage of data to keep as samples
 * XDIM,YDIM,ZDIM                   - Current timestep data dimensions
 * XBLOCK, YBLOCK,ZBLOCK            - Block dimensions
 * reference_histogram              - Previous timestep histogram information
 * current_timestep_max             - Maximum value in current timestep
 * current_timestep_min             - Minimum value in current timestep
 * lifetime_max                     - Maximum value expected over entire simulation
 * lifetime_min                     - Minimum value expected over entire simulation
 * data_acceptance_histogram        - Current timestep data acceptance histogram
 * 
 * Outputs:
 * sample_data_ids                  - Sample Location ID
 * sample_data                      - Sample Data
 * samples_per_block                - Samples Per Block
 **/
int omp_error_reuse_method(int num_threads, vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, vector<float> &sampling_timers, float error_threshold, vector<int> ref_samples_per_block, vector<int> reference_sample_ids, vector<float> reference_sample_data){
    if (SAMPLING_PRINT){
        std::cout << "Beginning Error Reuse Importance-Based Sampling Procedure...\n";
    }

    // Generate random numbers equal to entire data set
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    int num_elements = full_data.size();
    double rand_vals [num_elements];
    
    auto random_gen_time_start = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        srand(int(time(NULL)) ^ omp_get_thread_num());
        #pragma omp for
        for (int i = 0; i < num_elements; i++){
            double r = distribution(generator);
            // Store in random number vector
            rand_vals[i] = r;
        }
    }
    auto random_gen_time_end = std::chrono::steady_clock::now();


    // Determine number of elements per thread
    int n_per_thread;
    if (num_elements < num_threads){
        n_per_thread = 1;
    } else {
        n_per_thread = num_elements / num_threads;
    }
    // Set number of threads
	omp_set_num_threads(num_threads);

    // Determine maximum samples and setup loop variable
    int max_samples = sample_ratio*full_data.size();
    int tot_samples = 0;
    int num_blocks_reused = 0;

    // Create acceptance stencil
    double stencil[num_elements] = {0};

    // Resize samples per block vector
    samples_per_block.resize(num_blocks,0);

    // Iterate over all data values
    auto block_comparison_and_utilization_time_start = std::chrono::steady_clock::now();
    auto block_comparison_and_utilization_time_end = std::chrono::steady_clock::now();
    auto stencil_time_start = std::chrono::steady_clock::now();
    auto stencil_time_end = std::chrono::steady_clock::now();
    auto sample_gather_time_start = std::chrono::steady_clock::now();
    #pragma omp parallel
    {
        // Have each thread create private arrays
        vector<int> private_block_ids;
        vector<int> private_samples_per_block;
        int private_blocks_processed = 0;
        int private_num_blocks_reused = 0;
        
        // Check each block for utilization
        #pragma omp for 
        for(int block_id = 0; block_id < num_blocks; block_id++){
            // Get individual block data
            vector<int> block_data_ids;
            vector<float> block_data;
            // Get reference block data
            vector<int> ref_block_data_ids;
            vector<float> ref_block_data;

            get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_data_ids, block_data);

            // Determine number of values in block
            // int num_block_values = block_data_ids.size();

            // get T-1 samples in this block
            get_block_samples_w_global_ids(reference_sample_ids, reference_sample_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, ref_block_data_ids, ref_block_data);

            // Compare blocks to determine whether to reuse
            int utilize = 0;
            utilize_decision_error_based(block_data, ref_samples_per_block[block_id], ref_block_data_ids, ref_block_data, &utilize, reuse_flag, error_threshold, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK);

            // Mark for reuse or not
            if (utilize == 1){
                // Increment number of blocks reused
                private_num_blocks_reused = private_num_blocks_reused + 1;

                // Append reuse_flag to signify reuse
                private_samples_per_block.push_back(reuse_flag);
            } else {
                // Append 0 to signify reuse
                private_samples_per_block.push_back(0);
            }

            // Append which block this thread worked on
            private_block_ids.push_back(block_id);
            private_blocks_processed = private_blocks_processed + 1;
        }

        #pragma omp critical
        {
            // Have each thread aggregate their number of blocks reused
            num_blocks_reused = num_blocks_reused + private_num_blocks_reused;

            // Have each thread aggregate their samples per block arrays and reference histograms
            for (int i = 0; i < private_blocks_processed; i++){
                int block_id = private_block_ids[i];
                samples_per_block[block_id] = private_samples_per_block[i];
            }
        }

        // Have thread zero start and stop timers
        if (omp_get_thread_num() == 0){
            block_comparison_and_utilization_time_end = std::chrono::steady_clock::now();
            stencil_time_start = std::chrono::steady_clock::now();
        }

        // Ensure all threads have completed the previous section
        #pragma omp barrier

        // Have each thread create individual copies of sample data arrays
        private_samples_per_block.resize(num_blocks, 0);
        vector<int> private_sample_data_ids;
        vector<float> private_sample_data;
        int private_total_samples_gathered = 0;

        // Iterate over all elements
        #pragma omp for schedule(static, n_per_thread)
        for (int global_id = 0; global_id < num_elements; global_id++){
            // Calculate width of bins
            float range = current_timestep_max - current_timestep_min;
            float bin_width = range/num_bins;

            // Determine which bin this value belongs to
            int bin_id = (full_data[global_id] - current_timestep_min) / bin_width;

            // Handle Edge Cases
            if (bin_id > num_bins-1){
                bin_id = num_bins - 1;
            }
            if (bin_id < 0){
                bin_id = 0;
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
            if (samples_per_block[block_id] == reuse_flag){
                stencil[global_id] = 0;
            } else {
                // Determine whether to save sample or not
                // If difference is positive, save sample, else dont save
                stencil[global_id] = data_acceptance_histogram[bin_id] - rand_vals[global_id];

                // If sample chosen to be saved, increment samples saved in that block
                if (stencil[global_id] > 0){
                    private_samples_per_block[block_id] = private_samples_per_block[block_id] + 1;
                    private_sample_data.push_back(full_data[global_id]);
                    private_sample_data_ids.push_back(global_id);
                    private_total_samples_gathered = private_total_samples_gathered + 1;
                }
            }
        }

        // Have thread zero start and stop timers
        if (omp_get_thread_num() == 0){
            stencil_time_end = std::chrono::steady_clock::now();
            sample_gather_time_start = std::chrono::steady_clock::now();
        }

        #pragma omp critical
        {
            // Have each thread aggreate samples per block taken
            for (int i = 0; i < num_blocks; i++){
                samples_per_block[i] = samples_per_block[i] + private_samples_per_block[i];
            }

            // Have each thread add their samples to the total sample data arrays
            for (int i = 0; i < private_total_samples_gathered; i++){
                sample_data.push_back(private_sample_data[i]);
                sample_data_ids.push_back(private_sample_data_ids[i]);
            }
        }
    }
    auto sample_gather_time_end = std::chrono::steady_clock::now();

    // Print metrics
    float reused_percentage = ((float)num_blocks_reused/(float)num_blocks)*100;

    // Redisperse extra samples via random sampling
    tot_samples = sample_data_ids.size();
    int samples_remaining = max_samples - tot_samples;
    float new_sample_ratio;
    // Check to make sure some blocks were reused and some samples remain
    if (((num_blocks - num_blocks_reused) != num_blocks) && (samples_remaining > 0)){
        // If only some blocks were reused
        if (num_blocks - num_blocks_reused != 0){
            new_sample_ratio =  ((float)samples_remaining/((float)num_blocks - (float)num_blocks_reused))/(float)(XBLOCK*YBLOCK*ZBLOCK);
        // If all blocks were reused
        }else{
            new_sample_ratio = sample_ratio;
        }
    // If no blocks reused, no extra samples can be added
    }else{
        new_sample_ratio = 0;
    }
    if (SAMPLING_PRINT){
        std::cout << "Error Reuse Importance-Based Sampling Procedure Completed!\n";
        std::cout << "True Samples Taken : " << tot_samples << " samples\n";
        std::cout << "Number of Blocks Reused: " << num_blocks_reused << " -> " << reused_percentage << "%\n";
        std::cout << "New Sample ratio: " << new_sample_ratio << "\n";
    }
    
    // Add Random Samples
    float additional_random_sampling_timer = 0;
    if (new_sample_ratio > 0 && num_blocks_reused > 0){
        auto additional_random_time_start = std::chrono::steady_clock::now();
        add_random_samples(full_data, reuse_flag, num_blocks, new_sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, max_samples, &tot_samples);
        auto additional_random_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> additional_random_seconds = additional_random_time_end-additional_random_time_start;
        additional_random_sampling_timer = additional_random_sampling_timer + additional_random_seconds.count();
        if (SAMPLING_PRINT){
            std::cout << "Final True Taken : " << tot_samples << " samples\n";
        }
    }

    // Store timing information
    std::chrono::duration<double> block_comparison_and_utilization_seconds = block_comparison_and_utilization_time_end-block_comparison_and_utilization_time_start;
    sampling_timers.push_back(block_comparison_and_utilization_seconds.count()); // s
    std::chrono::duration<double> stencil_seconds = stencil_time_end-stencil_time_start;
    std::chrono::duration<double> random_gen_seconds = random_gen_time_end-random_gen_time_start;
    float random_and_stencil_seconds = random_gen_seconds.count() + stencil_seconds.count();
    sampling_timers.push_back(random_and_stencil_seconds); // s
    sampling_timers.push_back(additional_random_sampling_timer); // s
    std::chrono::duration<double> sample_gather_seconds = sample_gather_time_end-sample_gather_time_start;
    sampling_timers.push_back(sample_gather_seconds.count()); // s

    return(num_blocks_reused);
}




////////////////////////////////
// SAMPLING CONTROL FUNCTIONS //
////////////////////////////////

void value_histogram_based_importance_sampling(vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float sample_ratio, int num_bins, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers){
    // Determine number of regions
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get min and max of input data values
    float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
    float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

    // Build histogram of entire dataset
    vector<int> value_histogram(num_bins);
    fill(value_histogram.begin(), value_histogram.end(), 0); // init with zeros

    auto histogram_time_start = std::chrono::steady_clock::now();
    data_histogram(full_data, value_histogram, num_bins, data_max, data_min);
    auto histogram_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_seconds = histogram_time_end-histogram_time_start;
    sampling_timers.push_back(histogram_seconds.count()); // s

    // Create Importance Factor / Acceptance Function   
    vector<float> full_data_acceptance_histogram(num_bins);
    acceptance_function(full_data, num_bins, sample_ratio, full_data_acceptance_histogram, value_histogram, sampling_timers);
        
    // Begin Sampling Process
    value_histogram_importance_sampling(full_data, num_blocks, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, num_bins, data_max, data_min, full_data_acceptance_histogram, sample_data_ids, sample_data, samples_per_block, sampling_timers);
}



void omp_value_histogram_based_importance_sampling(int num_threads, vector<float> &full_data, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, float sample_ratio, int num_bins, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, vector<float> &sampling_timers){
    // Determine number of regions
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get min and max of input data values
    float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
    float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

    // Build histogram of entire dataset
    vector<int> value_histogram(num_bins);
    fill(value_histogram.begin(), value_histogram.end(), 0); // init with zeros

    auto histogram_time_start = std::chrono::steady_clock::now();
    omp_data_histogram(full_data, value_histogram, num_bins, data_max, data_min);
    auto histogram_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_seconds = histogram_time_end-histogram_time_start;
    sampling_timers.push_back(histogram_seconds.count()); // s

    // Create Importance Factor / Acceptance Function   
    vector<float> full_data_acceptance_histogram(num_bins);
    acceptance_function(full_data, num_bins, sample_ratio, full_data_acceptance_histogram, value_histogram, sampling_timers);
        
    // Begin Sampling Process
    omp_value_histogram_importance_sampling(num_threads, full_data, num_blocks, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, num_bins, data_max, data_min, full_data_acceptance_histogram, sample_data_ids, sample_data, samples_per_block, sampling_timers);
}


int temporal_histogram_based_reuse_sampling(vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers){
    // Determine number of regions
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get min and max of input data values
    float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
    float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

    // Build histogram of entire dataset
    vector<int> value_histogram(num_bins);
    fill(value_histogram.begin(), value_histogram.end(), 0); // init with zeros
    auto histogram_time_start = std::chrono::steady_clock::now();
    data_histogram(full_data, value_histogram, num_bins, data_max, data_min);
    auto histogram_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_seconds = histogram_time_end-histogram_time_start;
    sampling_timers.push_back(histogram_seconds.count()); // s

    // Create Importance Factor / Acceptance Function   
    vector<float> full_data_acceptance_histogram(num_bins);
    acceptance_function(full_data, num_bins, sample_ratio, full_data_acceptance_histogram, value_histogram, sampling_timers);

    // Begin Sampling Process
    // Do non-temporal for first timestep
    int num_blocks_reused = 0;
    if (reference_histogram.size() == 0){
        std::cout << "First Timestep Found!" << std::endl;

        // Non-temporal sampling call
        value_histogram_importance_sampling(full_data, num_blocks, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, num_bins, data_max, data_min, full_data_acceptance_histogram, sample_data_ids, sample_data, samples_per_block, sampling_timers);

        // Pull off random number / stencil time and sample gathering times and save for later
        float sample_gathering_timer = sampling_timers.back();
        sampling_timers.pop_back();
        float random_and_stencil_timer = sampling_timers.back();
        sampling_timers.pop_back();

        // Build reference histogram for next timestep
        auto block_histogram_time_start = std::chrono::steady_clock::now();
        for(int block_id = 0; block_id < num_blocks; block_id++){
            // Get individual block information
            vector<int> block_sample_ids;
            vector<float> block_sample_data;
            get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_sample_ids, block_sample_data);
            // Create block histogram
            vector<int> block_histogram(num_bins);
            fill(block_histogram.begin(), block_histogram.end(), 0);
            data_histogram(block_sample_data, block_histogram, num_bins, lifetime_max, lifetime_min);
            // Append to reference histogram
            reference_histogram.insert(std::end(reference_histogram), std::begin(block_histogram), std::end(block_histogram));
        }
        auto block_histogram_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> block_histogram_seconds = block_histogram_time_end-block_histogram_time_start;
        sampling_timers.push_back(block_histogram_seconds.count()); // s

        // Ensure accurate reference histogram
        if(reference_histogram.size() != (uint) (num_bins*num_blocks)){
            std::cout << "Invalid Reference Histogram..." << reference_histogram.size() << " vs " << num_bins*num_blocks << "\n";
            exit(0);
        }

        // Push -1 onto sampling timers as no comparison occurs here
        sampling_timers.push_back(-1);
        // Push random number and stencil times to sampling timers
        sampling_timers.push_back(random_and_stencil_timer);
        // Push -1 onto sampling timers as no additional random samples occur here
        sampling_timers.push_back(-1);
        // Push sample gathering time to sampling timers
        sampling_timers.push_back(sample_gathering_timer);

    // Do temporal method for all future timesteps
    } else {
        num_blocks_reused = histogram_reuse_method(full_data, -1, num_blocks, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, reference_histogram, sample_data_ids, sample_data, samples_per_block, data_max, data_min, lifetime_max, lifetime_min, full_data_acceptance_histogram, sampling_timers);
    }

    return num_blocks_reused;
}



int omp_temporal_histogram_based_reuse_sampling(int num_threads, vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers){
    // Determine number of regions
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get min and max of input data values
    float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
    float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

    // Build histogram of entire dataset
    vector<int> value_histogram(num_bins);
    fill(value_histogram.begin(), value_histogram.end(), 0); // init with zeros
    auto histogram_time_start = std::chrono::steady_clock::now();
    omp_data_histogram(full_data, value_histogram, num_bins, data_max, data_min);
    auto histogram_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_seconds = histogram_time_end-histogram_time_start;
    sampling_timers.push_back(histogram_seconds.count()); // s

    // Create Importance Factor / Acceptance Function   
    vector<float> full_data_acceptance_histogram(num_bins);
    acceptance_function(full_data, num_bins, sample_ratio, full_data_acceptance_histogram, value_histogram, sampling_timers);

    // Begin Sampling Process
    // Do non-temporal for first timestep
    int num_blocks_reused = 0;
    if (reference_histogram.size() == 0){
        // Non-temporal sampling call
        omp_value_histogram_importance_sampling(num_threads, full_data, num_blocks, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, num_bins, data_max, data_min, full_data_acceptance_histogram, sample_data_ids, sample_data, samples_per_block, sampling_timers);

        // Pull off random number / stencil time and sample gathering times and save for later
        float sample_gathering_timer = sampling_timers.back();
        sampling_timers.pop_back();
        float random_and_stencil_timer = sampling_timers.back();
        sampling_timers.pop_back();

        // Build reference histogram for next timestep
        auto block_histogram_time_start = std::chrono::steady_clock::now();
        for(int block_id = 0; block_id < num_blocks; block_id++){
            // Get individual block information
            vector<int> block_sample_ids;
            vector<float> block_sample_data;
            get_block_data_w_global_ids(full_data, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, block_id, block_sample_ids, block_sample_data);
            // Create block histogram
            vector<int> block_histogram(num_bins);
            fill(block_histogram.begin(), block_histogram.end(), 0);
            omp_data_histogram(block_sample_data, block_histogram, num_bins, lifetime_max, lifetime_min);
            // Append to reference histogram
            reference_histogram.insert(std::end(reference_histogram), std::begin(block_histogram), std::end(block_histogram));
        }
        auto block_histogram_time_end = std::chrono::steady_clock::now();
        std::chrono::duration<double> block_histogram_seconds = block_histogram_time_end-block_histogram_time_start;
        sampling_timers.push_back(block_histogram_seconds.count()); // s

        // Ensure accurate reference histogram
        if(reference_histogram.size() != (uint) (num_bins*num_blocks)){
            std::cout << "Invalid Reference Histogram..." << reference_histogram.size() << " vs " << num_bins*num_blocks << "\n";
            exit(0);
        }

        // Push -1 onto sampling timers as no comparison occurs here
        sampling_timers.push_back(-1);
        // Push random number and stencil times to sampling timers
        sampling_timers.push_back(random_and_stencil_timer);
        // Push -1 onto sampling timers as no additional random samples occur here
        sampling_timers.push_back(-1);
        // Push sample gathering time to sampling timers
        sampling_timers.push_back(sample_gathering_timer);

    // Do temporal method for all future timesteps
    } else {
        num_blocks_reused = omp_histogram_reuse_method(num_threads, full_data, -1, num_blocks, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, reference_histogram, sample_data_ids, sample_data, samples_per_block, data_max, data_min, lifetime_max, lifetime_min, full_data_acceptance_histogram, sampling_timers);
    }

    return num_blocks_reused;
}



int temporal_error_based_reuse_sampling(vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &ref_samples_per_block, vector<int> &reference_sample_ids, vector<float> &reference_sample_data, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers, float error_threshold){
    // Determine number of regions
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get min and max of input data values
    float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
    float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

    // Build histogram of entire dataset
    vector<int> value_histogram(num_bins);
    fill(value_histogram.begin(), value_histogram.end(), 0); // init with zeros
    auto histogram_time_start = std::chrono::steady_clock::now();
    data_histogram(full_data, value_histogram, num_bins, data_max, data_min);
    auto histogram_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_seconds = histogram_time_end-histogram_time_start;
    sampling_timers.push_back(histogram_seconds.count()); // s

    // Create Importance Factor / Acceptance Function   
    vector<float> full_data_acceptance_histogram(num_bins);
    acceptance_function(full_data, num_bins, sample_ratio, full_data_acceptance_histogram, value_histogram, sampling_timers);

    // Begin Sampling Process
    // Do non-temporal for first timestep
    int num_blocks_reused = 0;
    if (reference_sample_ids.size() == 0){
        // Non-temporal sampling call
        value_histogram_importance_sampling(full_data, num_blocks, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, num_bins, data_max, data_min, full_data_acceptance_histogram, sample_data_ids, sample_data, samples_per_block, sampling_timers);

        // Pull off random number / stencil time and sample gathering times and save for later
        float sample_gathering_timer = sampling_timers.back();
        sampling_timers.pop_back();
        float random_and_stencil_timer = sampling_timers.back();
        sampling_timers.pop_back();

        // Push -1 onto sampling timers as no comparison occurs here
        sampling_timers.push_back(-1);
        // Push random number and stencil times to sampling timers
        sampling_timers.push_back(random_and_stencil_timer);
        // Push -1 onto sampling timers as no additional random samples occur here
        sampling_timers.push_back(-1);
        // Push sample gathering time to sampling timers
        sampling_timers.push_back(sample_gathering_timer);

    // Do temporal method for all future timesteps
    } else {
        num_blocks_reused = error_reuse_method(full_data, -1, num_blocks, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, data_max, data_min, lifetime_max, lifetime_min, full_data_acceptance_histogram, sampling_timers, error_threshold, ref_samples_per_block, reference_sample_ids, reference_sample_data);
    }

    return num_blocks_reused;
}



int omp_temporal_error_based_reuse_sampling(int num_threads, vector<float> &full_data, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &ref_samples_per_block, vector<int> &reference_sample_ids, vector<float> &reference_sample_data, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float lifetime_max, float lifetime_min, vector<float> &sampling_timers, float error_threshold){
    // Determine number of regions
    int num_blocks = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK);

    // Get min and max of input data values
    float data_min = *std::min_element(std::begin(full_data), std::end(full_data));
    float data_max = *std::max_element(std::begin(full_data), std::end(full_data));

    // Build histogram of entire dataset
    vector<int> value_histogram(num_bins);
    fill(value_histogram.begin(), value_histogram.end(), 0); // init with zeros
    auto histogram_time_start = std::chrono::steady_clock::now();
    omp_data_histogram(full_data, value_histogram, num_bins, data_max, data_min);
    auto histogram_time_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> histogram_seconds = histogram_time_end-histogram_time_start;
    sampling_timers.push_back(histogram_seconds.count()); // s

    // Create Importance Factor / Acceptance Function   
    vector<float> full_data_acceptance_histogram(num_bins);
    acceptance_function(full_data, num_bins, sample_ratio, full_data_acceptance_histogram, value_histogram, sampling_timers);

    // Begin Sampling Process
    // Do non-temporal for first timestep
    int num_blocks_reused = 0;
    if (reference_sample_ids.size() == 0){
        // Non-temporal sampling call
        omp_value_histogram_importance_sampling(num_threads, full_data, num_blocks, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, num_bins, data_max, data_min, full_data_acceptance_histogram, sample_data_ids, sample_data, samples_per_block, sampling_timers);

        // Pull off random number / stencil time and sample gathering times and save for later
        float sample_gathering_timer = sampling_timers.back();
        sampling_timers.pop_back();
        float random_and_stencil_timer = sampling_timers.back();
        sampling_timers.pop_back();

        // Push -1 onto sampling timers as no comparison occurs here
        sampling_timers.push_back(-1);
        // Push random number and stencil times to sampling timers
        sampling_timers.push_back(random_and_stencil_timer);
        // Push -1 onto sampling timers as no additional random samples occur here
        sampling_timers.push_back(-1);
        // Push sample gathering time to sampling timers
        sampling_timers.push_back(sample_gathering_timer);

    // Do temporal method for all future timesteps
    } else {
        num_blocks_reused = omp_error_reuse_method(num_threads, full_data, -1, num_blocks, num_bins, sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, data_max, data_min, lifetime_max, lifetime_min, full_data_acceptance_histogram, sampling_timers, error_threshold, ref_samples_per_block, reference_sample_ids, reference_sample_data);
    }

    return num_blocks_reused;
}




//////////////////////////////
// RECONSTRUCTION FUNCTIONS //
//////////////////////////////

/**
 * nearest_neighbors_reconstruction:
 * Reconstruct data from samples by filling any gaps with 
 * value from nearest neighbor.
 * 
 * Input:
 * sample_global_ids    - Global ID's of samples
 * sample_data          - Samples from original data
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Dimensions of original data
 * 
 * Output:
 * reconstructed_data   - Reconstructed version of original data
 **/

void nearest_neighbors_reconstruction(vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data){
    if (SAMPLING_PRINT){
        std::cout << "Nearest Neighbors Reconstruction Started..." << std::endl;
    }
    // Allocate new vectors for the x y and z global coords of each samples
    vector<int> global_x(num_samples);
    vector<int> global_y(num_samples);
    vector<int> global_z(num_samples);

    // For each sample in sample data, get true global coordinates
    for (int i = 0; i < num_samples; i++){
        global_x[i] = sample_global_ids[i] % XDIM;
        global_y[i] = (sample_global_ids[i] / XDIM) % YDIM;
        global_z[i] = sample_global_ids[i] / (XDIM*YDIM);
    }

    // For each point in the reconstructed dataset, check all samples to find the nearest neighbor
    for (int k = 0; k < ZDIM; k++){
        for (int j = 0; j < YDIM; j++){
            for (int i = 0; i < XDIM; i++){
                // Iterate over all samples and find the nearest neighbor
                int min_index = -1;
                float min_dist = -1;
                for (int x = 0; x < num_samples; x++){
                    // Calculate distance
                    float dist = std::pow((i - global_x[x]), 2) + std::pow((j - global_y[x]), 2) + std::pow((k - global_z[x]), 2);
                    // Save minimum distance and index
                    if (min_dist >= dist || min_dist == -1){
                        min_index = x;
                        min_dist = dist;
                    }
                }

                // Write min value to reconstructed data array
                reconstructed_data.push_back(sample_data[min_index]);
            }
        }
    }
    if (SAMPLING_PRINT){
        std::cout << "Nearest Neighbors Reconstruction Completed!" << std::endl;
    }
}



/**
 * omp_nearest_neighbors_reconstruction:
 * Reconstruct data from samples by filling any gaps with 
 * value from nearest neighbor.
 * 
 * Input:
 * sample_global_ids    - Global ID's of samples
 * sample_data          - Samples from original data
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Dimensions of original data
 * 
 * Output:
 * reconstructed_data   - Reconstructed version of original data
 **/

void omp_nearest_neighbors_reconstruction(int num_threads, vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data){
    if (SAMPLING_PRINT){
        std::cout << "Nearest Neighbors Reconstruction Started..." << std::endl;
    }

    // Set maximum number of threads
    omp_set_num_threads(num_threads);

    // Allocate new vectors for the x y and z global coords of each samples
    vector<int> global_x(num_samples);
    vector<int> global_y(num_samples);
    vector<int> global_z(num_samples);

    

    // For each sample in sample data, get true global coordinates
    #pragma omp parallel for
    for (int i = 0; i < num_samples; i++){
        global_x[i] = sample_global_ids[i] % XDIM;
        global_y[i] = (sample_global_ids[i] / XDIM) % YDIM;
        global_z[i] = sample_global_ids[i] / (XDIM*YDIM);
    }


    // For each point in the reconstructed dataset, check all samples to find the nearest neighbor
    int num_elements = XDIM*YDIM*ZDIM;
    reconstructed_data.resize(num_elements, 0);
    #pragma omp parallel for
    for (int global_id = 0; global_id < num_elements; global_id++){
        int x = global_id % XDIM;
        int y = (global_id / XDIM) % YDIM;
        int z = global_id / (XDIM*YDIM);

        // Iterate over all samples and find the nearest neighbor
        int min_index = -1;
        float min_dist = -1;
        for (int loc = 0; loc < num_samples; loc++){
            // Calculate distance
            float dist = std::pow((x - global_x[loc]), 2) + std::pow((y - global_y[loc]), 2) + std::pow((z - global_z[loc]), 2);
            // Save minimum distance and index
            if (min_dist >= dist || min_dist == -1){
                min_index = loc;
                min_dist = dist;
            }
        }

        // Write min value to reconstructed data array
        #pragma omp critical
        {
            reconstructed_data[global_id] = sample_data[min_index];
        }
    }

    if (SAMPLING_PRINT){
        std::cout << "Nearest Neighbors Reconstruction Completed!" << std::endl;
    }
}



/**
 * k_nearest_neighbors_reconstruction:
 * Reconstruct data from samples by filling any gaps with an 
 * averaged value from k nearest neighbor.
 * 
 * Input:
 * k_samples            - Number of neighboring points to average from
 * sample_global_ids     - Global ID's of samples
 * sample_data    - Samples from original data
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Dimensions of original data
 * 
 * Output:
 * reconstructed_data   - Reconstructed version of original data
 **/
void k_nearest_neighbors_reconstruction(int k_samples, vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data){
    if (SAMPLING_PRINT){
        std::cout << "K Nearest Neighbors Reconstruction Started..." << std::endl;
    }
    // Allocate new vectors for the x y and z global coords of each samples
    vector<int> global_x(num_samples);
    vector<int> global_y(num_samples);
    vector<int> global_z(num_samples);

    // For each sample in sample data, get true global coordinates
    for (int i = 0; i < num_samples; i++){
        global_x[i] = sample_global_ids[i] % XDIM;
        global_y[i] = (sample_global_ids[i] / XDIM) % YDIM;
        global_z[i] = sample_global_ids[i] / (XDIM*YDIM);
    }

    // Create a vector of k samples
    vector<int> k_sample_vector_indices(k_samples);
    vector<float> k_sample_vector_dists(k_samples);

    // For each point in the reconstructed dataset, check all samples to find the nearest neighbor
    for (int k = 0; k < ZDIM; k++){
        for (int j = 0; j < YDIM; j++){
            for (int i = 0; i < XDIM; i++){
                // Set vector of k samples to -1
                fill(k_sample_vector_indices.begin(), k_sample_vector_indices.end(), -1); // init with -1's
                fill(k_sample_vector_dists.begin(), k_sample_vector_dists.end(), -1); // init with -1's

                // Iterate over all samples and find the nearest neighbor
                for (int x = 0; x < num_samples; x++){
                    // Calculate distance
                    float dist = std::pow((i - global_x[x]), 2) + std::pow((j - global_y[x]), 2) + std::pow((k - global_z[x]), 2);
                    for (int y = 0; y < k_samples; y++){
                        // Save minimum distance and index
                        if (k_sample_vector_dists[y] >= dist || k_sample_vector_dists[y] == -1){
                            k_sample_vector_indices[y] = x;
                            k_sample_vector_dists[y] = dist;
                            break;
                        }
                    }
                }

                // Weighted Average of k nearest neighbors based on distance
                float avg = 0;
                float weights = 0;
                for (int y = 0; y < k_samples; y++){
                    avg = avg + (sample_data[k_sample_vector_indices[y]] * (1 / (k_sample_vector_dists[y]+1)));
                    weights = weights + (1 / (k_sample_vector_dists[y]+1));
                }
                avg = avg / weights;

                // Write avg min value to reconstructed data array
                reconstructed_data.push_back(avg);
            }
        }
    }
    if (SAMPLING_PRINT){
        std::cout << "K Nearest Neighbors Reconstruction Completed!" << std::endl;
    }
}



/**
 * omp_k_nearest_neighbors_reconstruction:
 * Reconstruct data from samples by filling any gaps with an 
 * averaged value from k nearest neighbor.
 * 
 * Input:
 * k_samples            - Number of neighboring points to average from
 * sample_global_ids     - Global ID's of samples
 * sample_data    - Samples from original data
 * num_samples          - Number of samples
 * XDIM, YDIM, ZDIM     - Dimensions of original data
 * 
 * Output:
 * reconstructed_data   - Reconstructed version of original data
 **/
void omp_k_nearest_neighbors_reconstruction(int num_threads, int k_samples, vector<int> sample_global_ids, vector<float> sample_data, int num_samples, int XDIM, int YDIM, int ZDIM, vector<float> &reconstructed_data){
    if (SAMPLING_PRINT){
        std::cout << "K Nearest Neighbors Reconstruction Started..." << std::endl;
    }

    // Set maximum number of threads
    omp_set_num_threads(num_threads);

    // Allocate new vectors for the x y and z global coords of each samples
    vector<int> global_x(num_samples);
    vector<int> global_y(num_samples);
    vector<int> global_z(num_samples);

    // For each sample in sample data, get true global coordinates
    #pragma omp parallel for
    for (int i = 0; i < num_samples; i++){
        global_x[i] = sample_global_ids[i] % XDIM;
        global_y[i] = (sample_global_ids[i] / XDIM) % YDIM;
        global_z[i] = sample_global_ids[i] / (XDIM*YDIM);
    }


    // For each point in the reconstructed dataset, check all samples to find the nearest neighbor
    int num_elements = XDIM*YDIM*ZDIM;
    reconstructed_data.resize(num_elements, 0);
    #pragma omp parallel for
    for (int global_id = 0; global_id < num_elements; global_id++){
        int x = global_id % XDIM;
        int y = (global_id / XDIM) % YDIM;
        int z = global_id / (XDIM*YDIM);

        // Create a vector of k samples
        vector<int> k_sample_vector_indices(k_samples);
        vector<float> k_sample_vector_dists(k_samples);

        // Set vector of k samples to -1
        fill(k_sample_vector_indices.begin(), k_sample_vector_indices.end(), -1); // init with -1's
        fill(k_sample_vector_dists.begin(), k_sample_vector_dists.end(), -1); // init with -1's

        // Iterate over all samples and find the nearest neighbor
        for (int loc_a = 0; loc_a < num_samples; loc_a++){
            // Calculate distance
            float dist = std::pow((x - global_x[loc_a]), 2) + std::pow((y - global_y[loc_a]), 2) + std::pow((z - global_z[loc_a]), 2);
            for (int loc_b = 0; loc_b < k_samples; loc_b++){
                // Save minimum distance and index
                if (k_sample_vector_dists[loc_b] >= dist || k_sample_vector_dists[loc_b] == -1){
                    k_sample_vector_indices[loc_b] = loc_a;
                    k_sample_vector_dists[loc_b] = dist;
                    break;
                }
            }
        }

        // Weighted Average of k nearest neighbors based on distance
        float avg = 0;
        float weights = 0;
        for (int loc_b = 0; loc_b < k_samples; loc_b++){
            avg = avg + (sample_data[k_sample_vector_indices[loc_b]] * (1 / (k_sample_vector_dists[loc_b]+1)));
            weights = weights + (1 / (k_sample_vector_dists[loc_b]+1));
        }
        avg = avg / weights;

        // Write avg min value to reconstructed data array
        #pragma omp critical
        {
            reconstructed_data[global_id] = avg;
        }
    }
    if (SAMPLING_PRINT){
        std::cout << "K Nearest Neighbors Reconstruction Completed!" << std::endl;
    }
}


















// Version where block histogramming is serpate but worse due to atomic operation.
/**
int omp_histogram_reuse_method(int num_threads, vector<float> &full_data, int reuse_flag, int num_blocks, int num_bins, float sample_ratio, int XDIM, int YDIM, int ZDIM, int XBLOCK, int YBLOCK, int ZBLOCK, \
vector<int> &reference_histogram, vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, float current_timestep_max, float current_timestep_min, \
float lifetime_max, float lifetime_min, vector<float> &data_acceptance_histogram, double *sampling_time){
    if (SAMPLING_PRINT){
        std::cout << "Beginning Histogram Reuse Importance-Based Sampling Procedure...\n";
    }

    // Generate random numbers equal to entire data set
    std::default_random_engine generator(random_seed_2);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    int num_elements = full_data.size();
    double rand_vals [num_elements];
    for (int i = 0; i < num_elements; i++){
        double r = distribution(generator);
        // Store in random number vector
        rand_vals[i] = r;
    }

    // Determine number of elements per thread
    int n_per_thread;
    if (num_elements < num_threads){
        n_per_thread = 1;
    } else {
        n_per_thread = num_elements / num_threads;
    }
    // Set number of threads
	omp_set_num_threads(num_threads);

    // Determine maximum samples and setup loop variable
    int max_samples = sample_ratio*full_data.size();
    int tot_samples = 0;
    int num_blocks_reused = 0;

    // Build current block histograms
    vector<int> current_histogram_list(reference_histogram.size(), 0);
    #pragma omp parallel for schedule(static, n_per_thread)
    for (int global_id = 0; global_id < num_elements; global_id++){
        // Calculate width of bins
        float range = lifetime_max - lifetime_min;
        float bin_width = range/num_bins;

        // Determine which bin this value belongs to
        int bin_id = (full_data[global_id] - lifetime_min) / bin_width;

        // Handle Edge Cases
        if (bin_id > num_bins-1){
            bin_id = num_bins - 1;
        }
        if (bin_id < 0){
            bin_id = 0;
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
        int offset_id = block_id*(num_bins) + bin_id;

        // Increment the number of values in this bin
        #pragma omp atomic update
        current_histogram_list[offset_id] = current_histogram_list[offset_id] + 1;
    }

    // Check for utilization
    for(int block_id = 0; block_id < num_blocks; block_id++){
        // Determine current block histogram start and end
        const int block_histogram_current_start = num_bins*block_id;
        const int block_histogram_current_end = block_histogram_current_start + num_bins;
        vector<int>::const_iterator current_first = current_histogram_list.begin() + block_histogram_current_start;
        vector<int>::const_iterator current_last = current_histogram_list.begin() + block_histogram_current_end;
        vector<int> block_histogram_current(current_first, current_last);

        // Determine reference block histogram start and end
        const int block_histogram_reference_start = num_bins*block_id;
        const int block_histogram_reference_end = block_histogram_reference_start + num_bins;
        vector<int>::const_iterator reference_first = reference_histogram.begin() + block_histogram_reference_start;
        vector<int>::const_iterator reference_last = reference_histogram.begin() + block_histogram_reference_end;
        vector<int> block_histogram_reference(reference_first, reference_last);

        // Ensure histograms are valid
        if (block_histogram_current_end - block_histogram_current_start != num_bins){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC A!!"<< block_histogram_current_end - block_histogram_current_start << " vs " << num_bins << "\n";
            exit(0);
        }
        if (block_histogram_reference_end - block_histogram_reference_start != num_bins){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC A2!!"<< block_histogram_reference_end - block_histogram_reference_start << " vs " << num_bins << "\n";
            exit(0);
        }
        if (block_histogram_current.size() != num_bins){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC B!!"<< block_histogram_current.size() << " vs " << num_bins << "\n";
            exit(0);
        }
        if (block_histogram_reference.size() != num_bins){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC B2!!"<< block_histogram_reference.size() << " vs " << num_bins << "\n";
            exit(0);
        }
        if (reference_histogram.size() != num_bins*num_blocks){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC C!!"<< reference_histogram.size() << " vs " << num_bins*num_blocks << "\n";
            exit(0);
        }

        // Compare histograms to determine whether to reuse
        int utilize = 0;
        utilize_decision_histogram_based(block_histogram_current, block_histogram_reference, num_bins, &utilize, reuse_flag);

        // Mark for reuse or not
        if (utilize == 1){
            // Increment number of blocks reused
            num_blocks_reused = num_blocks_reused + 1;

            // append reuse_flag to signify reuse
            samples_per_block.push_back(reuse_flag);

            // Overwrite block in reference histogram with reuse_flag for next timestep
            reference_histogram[block_histogram_reference_start] = reuse_flag;
        } else {
            // append 0 to signify reuse
            samples_per_block.push_back(0);

            // Overwrite block in reference histogram with current histogram for next timestep
            int index = block_histogram_current_start;
            for(int i = block_histogram_reference_start; i < block_histogram_reference_end; i++){
                reference_histogram[i] = block_histogram_current[index];
                index++;
            }
        } 

        // Ensure reference histogram is still correct
        if (reference_histogram.size() != num_bins*num_blocks){
            std::cout << "ERROR IN BLOCK HISTOGRAM ID LOGIC D!!"<< reference_histogram.size() << " vs " << num_bins*num_blocks << "\n";
            exit(0);
        }
    }

    // Create acceptance stencil
    double stencil[num_elements] = {0};

    // Begin sampling process
    #pragma omp parallel for schedule(static, n_per_thread)
    for (int global_id = 0; global_id < num_elements; global_id++){
        // Calculate width of bins
        float range = current_timestep_max - current_timestep_min;
        float bin_width = range/num_bins;

        // Determine which bin this value belongs to
        int bin_id = (full_data[global_id] - current_timestep_min) / bin_width;

        // Handle Edge Cases
        if (bin_id > num_bins-1){
            bin_id = num_bins - 1;
        }
        if (bin_id < 0){
            bin_id = 0;
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
        if (samples_per_block[block_id] == reuse_flag){
            stencil[global_id] = 0;
        } else {
            // Determine whether to save sample or not
            // If difference is positive, save sample, else dont save
            stencil[global_id] = data_acceptance_histogram[bin_id] - rand_vals[global_id];

            // If sample chosen to be saved, increment samples saved in that block
            if (stencil[global_id] > 0){
                #pragma omp atomic update
                samples_per_block[block_id] = samples_per_block[block_id] + 1;
            }
        }
    }

    // Determine total samples from all blocks
    for(int block_id = 0; block_id < num_blocks; block_id++){
        if (samples_per_block[block_id] != reuse_flag){
            tot_samples = tot_samples + samples_per_block[block_id];
        }
    }

    // Gather samples from stencil
    for (int global_id = 0; global_id < num_elements; global_id++){
        if (stencil[global_id] > 0){
            sample_data.push_back(full_data[global_id]);
            sample_data_ids.push_back(global_id);
        }
    }

    // Print metrics
    float reused_percentage = ((float)num_blocks_reused/(float)num_blocks)*100;

    // Redisperse extra samples via random sampling
    int samples_remaining = max_samples - tot_samples;
    float new_sample_ratio;
    // Check to make sure some blocks were reused and some samples remain
    if (((num_blocks - num_blocks_reused) != num_blocks) && (samples_remaining > 0)){
        // If only some blocks were reused
        if (num_blocks - num_blocks_reused != 0){
            new_sample_ratio =  ((float)samples_remaining/((float)num_blocks - (float)num_blocks_reused))/(float)(XBLOCK*YBLOCK*ZBLOCK);
        // If all blocks were reused
        }else{
            new_sample_ratio = sample_ratio;
        }
    // If no blocks reused, no extra samples can be added
    }else{
        new_sample_ratio = 0;
    }
    if (SAMPLING_PRINT){
        std::cout << "Histogram Reuse Importance-Based Sampling Procedure Completed!\n";
        std::cout << "True Samples Taken : " << tot_samples << " samples\n";
        std::cout << "Number of Blocks Reused: " << num_blocks_reused << " -> " << reused_percentage << "%\n";
        std::cout << "New Sample ratio: " << new_sample_ratio << "\n";
    }
    
    // Add Random Samples
    if (new_sample_ratio > 0 && num_blocks_reused > 0){
        add_random_samples(full_data, reuse_flag, num_blocks, new_sample_ratio, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, sample_data_ids, sample_data, samples_per_block, max_samples, &tot_samples);
        if (SAMPLING_PRINT){
            std::cout << "Final True Taken : " << tot_samples << " samples\n";
        }
    }

    return(num_blocks_reused);
}
**/
