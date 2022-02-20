#include "utils.h"

bool UTIL_PRINT = false;



/**
 * coalesce_samples:
 * Coalesces samples from two timesteps when 
 * using samples from previous timesteps.
 * 
 * Input:
 * current_timestep_samples             - Samples from current timestep
 * current_sample_global_ids            - Global Ids for current timestep samples
 * current_timestep_samples_per_block   - Samples taken per block in current timestep
 * previous_timestep_samples            - Samples from previous timestep
 * previous_sample_global_ids           - Global Ids for previous timestep samples 
 * previous_timestep_samples_per_block  - Samples taken per block in previous timestep
 * 
 * Output:
 * coalesced_samples                    - Full samples of current timestep to be used in reconstruction
 * coalesced_sample_global_ids          - Global Ids for full samples
 * coalesced_samples_per_block          - Samples taken per block in coalesced sample list
 * 
 * Note:
 * If no samples were taken in a certain block, a value of -1 in the samples_per_block
 * vectors indicate this. However, two consecutive timesteps should not have -1s in the 
 * same block id location of the samples_per_block vectors.
 **/
void coalesce_samples(vector<float> current_timestep_samples, vector<int> current_sample_global_ids, vector<int> current_timestep_samples_per_block, \
vector<float> previous_timestep_samples, vector<int> previous_sample_global_ids, vector<int> previous_timestep_samples_per_block, vector<float> &coalesced_samples, \
vector<int> &coalesced_sample_global_ids, vector<int> &coalesced_samples_per_block){
    if (UTIL_PRINT){
        std::cout << "Sample Coalescing for Reconstruction Started..." << std::endl;
    }
    // Clear coalesced arrays
    coalesced_samples.resize(0,0);
    coalesced_sample_global_ids.resize(0,0);
    coalesced_samples_per_block.resize(0,0);
    // Loop Variables
    int current_sample_index = 0;
    int previous_sample_index = 0;
    // Iterate over each block of the gata
    for(uint block_id = 0; block_id < current_timestep_samples_per_block.size(); block_id++){
        // If the current timestep is reusing regions from previous timestep
        if(current_timestep_samples_per_block[block_id] == -1){
            if(previous_timestep_samples_per_block[block_id] != -1){
                // Loop over previous samples in block
                for (int index = previous_sample_index; index < previous_sample_index + previous_timestep_samples_per_block[block_id]; index++){
                    coalesced_samples.push_back(previous_timestep_samples[index]);
                    coalesced_sample_global_ids.push_back(previous_sample_global_ids[index]);
                }
                // Increment previous sample index
                previous_sample_index = previous_sample_index + previous_timestep_samples_per_block[block_id];
                coalesced_samples_per_block.push_back(previous_timestep_samples_per_block[block_id]);
            } else {
                std::cout << "Coalescing Issue: Block Samples not taken for two consecutive timesteps!";
                exit(0);
            }

        // If samples were taken in current timestep
        } else{
            // Loop over current samples in block
            for (int index = current_sample_index; index < current_sample_index + current_timestep_samples_per_block[block_id]; index++){
                coalesced_samples.push_back(current_timestep_samples[index]);
                coalesced_sample_global_ids.push_back(current_sample_global_ids[index]);
            }
            // Increment current sample index
            current_sample_index = current_sample_index + current_timestep_samples_per_block[block_id];
            coalesced_samples_per_block.push_back(current_timestep_samples_per_block[block_id]);
            
            // Increment previous timestep samples to next block
            if(previous_timestep_samples_per_block[block_id] != -1){
                previous_sample_index = previous_sample_index + previous_timestep_samples_per_block[block_id];
            } 
        }
    }
    if (UTIL_PRINT){
        std::cout << "Sample Coalescing for Reconstruction Completed!" << std::endl;
    }
}



/**
 * data_quality_analysis:
 * Given two floating point vectors, calculate 
 * maximum difference, RMSE, PSNR, and SNR.
 * 
 * Input:
 * original_data        - Original dataset
 * reconstructed_data   - Reconstructed dataset
 * data_size            - Size of both datasets
 * stats                - Output statistics array for csv writing
 **/
void data_quality_analysis(vector<float> original_data, vector<float> reconstructed_data, int data_size, vector<double> &stats){
    if (UTIL_PRINT){
        std::cout << "Data Quality Analysis Started..." << std::endl;
    }
    // Declare Loop Variables
    double max_diff = 0;
    double min_val = -1;
    double max_val = -1;
    double rmse_sum = 0;
    double mean_raw = 0;
    double stdev_raw = 0;
    double mean_sampled = 0;
    double stdev_sampled = 0;
    double mean_error = 0;
    double stdev_error = 0;
    vector<double> differences;

    // Iterate over each data element
    for (int i = 0; i < data_size; i++){
        // Ensure no NAN's 
        if (original_data[i] == original_data[i] && reconstructed_data[i] == reconstructed_data[i]){
            // Get pair of data points
            double original = (double)original_data[i];
            double reconstructed = (double)reconstructed_data[i];

            // RMSE Work
            double rmse_diff = original - reconstructed;
            rmse_diff = rmse_diff * rmse_diff;
            rmse_sum = rmse_sum + rmse_diff;

            // PSNR Work
            if(original > max_val || max_val == -1){
                max_val = original;
            }
            if(original < min_val || min_val == -1){
                min_val = original;
            }

            // Maximum Difference Work
            double diff = fabs(original - reconstructed);
            differences.push_back(diff);
            if(diff > max_diff){
                max_diff = diff;
            }

            // SNR Work
            mean_raw = mean_raw + original;
			mean_sampled = mean_sampled + reconstructed;
			mean_error = mean_error + diff;
        }
    }

    // Calculate SNR
    mean_raw = mean_raw / data_size;
    mean_sampled = mean_sampled / data_size;
    mean_error = mean_error / data_size;
    for (int i = 0; i < data_size; i++){
        // Ensure no NAN's 
        if (original_data[i] == original_data[i] && reconstructed_data[i] == reconstructed_data[i]){
            // Get pair of data points
            double original = (double)original_data[i];
            double reconstructed = (double)reconstructed_data[i];

            stdev_sampled = stdev_sampled + (reconstructed - mean_sampled) * (reconstructed - mean_sampled);
			stdev_raw = stdev_raw + (original - mean_raw) * (original - mean_raw);
			stdev_error = stdev_error + (fabs(original - reconstructed) - mean_error) * (fabs(original - reconstructed) - mean_error);
        }
	}
    stdev_raw = sqrt(stdev_raw / data_size);
    stdev_sampled = sqrt(stdev_sampled / data_size);
    stdev_error = sqrt(stdev_error / data_size);
    double snr = 20 * log10(stdev_raw / stdev_error);

    // Calculate Root Mean Square Error
    double rmse = rmse_sum / (data_size - 1);
	rmse = sqrt(rmse);

    // Check RMSE for bad values
    if (fpclassify(rmse) == FP_INFINITE) {
		rmse = FLT_MAX;
	} else if (fpclassify(rmse) == FP_NAN) {
		rmse = FLT_MAX;
	}

    // Calculate PSNR
    double psnr_control_value = 10000;
	double psnr = 0;
	if (rmse == 0){
		psnr = psnr_control_value;
	} else {
		psnr = 20 * log10((max_val - min_val) / rmse);	
	}	

	// Check PSNR for bad values
    if (fpclassify(psnr) == FP_INFINITE){
		psnr = psnr_control_value * -1;
	} else if (fpclassify(psnr) == FP_NAN) {
		psnr = psnr_control_value * -1;
	}

	// Check Maximum Difference for bad values
    if (fpclassify(max_diff) == FP_INFINITE){
		max_diff = FLT_MAX;
	} else if (fpclassify(max_diff) == FP_NAN) {
		max_diff = FLT_MAX;
	}

    // Temp difference work
    double diff_avg = 0;
    for(int i = 0; i < data_size; i++){
        //std::cout << i << " Difference: " << differences[i];
        diff_avg = diff_avg + differences[i];
    }
    diff_avg = diff_avg / data_size;

	// Print Metrics
    std::cout << "Maximum Absolute Difference: " << max_diff << std::endl;
    std::cout << "Average Absolute Difference: " << diff_avg << std::endl;
    std::cout << "Root Mean Squared Error: " << rmse << std::endl;
    std::cout << "PSNR: " << psnr << std::endl;
    std::cout << "SNR: " << snr << std::endl;


    stats.push_back(max_diff);
    stats.push_back(diff_avg);
    stats.push_back(psnr);
    stats.push_back(snr);

    if (UTIL_PRINT){
        std::cout << "Data Quality Analysis Completed!" << std::endl;
    }
}



/**
 * save_vector_to_bin:
 * Writes samples out to binary file
 * 
 * Input:
 * num_samples          - number of samples to write   
 * num_blocks           - number of blocks
 * sample_data_ids      - vector of sample ID's
 * sample_data          - vector of sample data
 * samples_per_block    - number of samples per block
 * timestep             - current samples timestep
 **/
void save_vector_to_bin(int num_samples, int num_blocks, vector<int> sample_data_ids, vector<float> sample_data, vector<int> samples_per_block, int timestep, std::string output_folder_name){
    std::string timestep_str = std::to_string(timestep);
    std::string id_filename = output_folder_name + "sampled_id_" + timestep_str + ".bin";
    std::string data_filename = output_folder_name + "sampled_data_" + timestep_str + ".bin";
    std::string total_filename = output_folder_name + "sampled_total_" + timestep_str + ".bin";

    // write sample IDs out
    std::ofstream fout_lid(id_filename, std::ios::binary);
    int output_data;
    for(int i = 0; i < num_samples; ++i){
        output_data = sample_data_ids[i];
        fout_lid.write(reinterpret_cast<const char*>(&output_data), sizeof(output_data));
    }
    fout_lid.close();

    // write sample data
    std::ofstream fout_data(data_filename, std::ios::binary);
    float f;
    for(int i = 0; i < num_samples; ++i){
        f = sample_data[i];
        fout_data.write(reinterpret_cast<const char*>(&f), sizeof(f));
    }
    fout_data.close();

    // write samples per block
    std::ofstream fout_total(total_filename, std::ios::binary);
    for(uint i = 0; i < samples_per_block.size(); ++i){ // NOTE: I replaced num_blocks with samples_per_block.size() because it was throwing errors
        output_data = samples_per_block.at(i);
        fout_total.write(reinterpret_cast<const char*>(&output_data), sizeof(output_data));
    }
    fout_total.close();

}



/**
 * save_bin_to_vector:
 * Reads binary file into vector
 * 
 * Input:
 * sample_data_ids      - vector of sample ID's
 * sample_data          - vector of sample data
 * samples_per_block    - number of samples per block
 * timestep             - current samples timestep
 **/
void save_bin_to_vector(vector<int> &sample_data_ids, vector<float> &sample_data, vector<int> &samples_per_block, int timestep, std::string input_folder_name){
    std::string timestep_str = std::to_string(timestep);
    std::string id_filename = input_folder_name + "sampled_id_" + timestep_str + ".bin";
    std::string data_filename = input_folder_name + "sampled_data_" + timestep_str + ".bin";
    std::string total_filename = input_folder_name + "sampled_total_" + timestep_str + ".bin";

    // get location per sample
    std::ifstream fid(id_filename, std::ios::binary);
    if(!fid){
        std::cout << " Error, Couldn't find the file: " << id_filename << "\n";
        exit(0);
    }
    fid.seekg(0, std::ios::end);
    const size_t num_elements = fid.tellg() / sizeof(int);
    fid.seekg(0, std::ios::beg);
    sample_data_ids.resize(num_elements);
    fid.read(reinterpret_cast<char*>(&sample_data_ids[0]), num_elements*sizeof(int));

    // get data per sample
    std::ifstream fdata(data_filename, std::ios::binary);
    if(!fdata){
        std::cout << " Error, Couldn't find the file: " << data_filename << "\n";
        exit(0);
    }
    sample_data.resize(num_elements);
    fdata.read(reinterpret_cast<char*>(&sample_data[0]), num_elements*sizeof(float));

    // get samples per block
    std::ifstream ftotal(total_filename, std::ios::binary);
    if(!ftotal){
        std::cout << " Error, Couldn't find the file: "<< total_filename << "\n";
        exit(0);
    }
    ftotal.seekg(0, std::ios::end);
    const size_t num_blocks = ftotal.tellg() / sizeof(int);
    ftotal.seekg(0, std::ios::beg);
    samples_per_block.resize(num_blocks);
    ftotal.read(reinterpret_cast<char*>(&samples_per_block[0]), num_blocks*sizeof(int));
}



/**
 * timestep_sort:
 * Compares timestep strings to determine which is greater
 * 
 * Input:
 * timestep_a   - first timestep string
 * timestep_b   - second timestep string
 * rx           - timestep string regular expression
 **/
int timestep_sort(std::string timestep_a, std::string timestep_b, std::regex rx){
    std::smatch match;
    // Get value of first timestep
    int timestep_a_value;
    if (std::regex_match(timestep_a, match, rx)){
        // Skip leading 0 in results and get value
        std::ssub_match sub_match = match[1];
        timestep_a_value = std::atoi(sub_match.str().c_str());
    }
    else 
        return -1;
    // Get value of second timestep
    int timestep_b_value;
    if (std::regex_match(timestep_b, match, rx)){
        // Skip leading 0 in results and get value
        std::ssub_match sub_match = match[1];
        timestep_b_value = std::atoi(sub_match.str().c_str());
    }
    else
        return -1;
    // Compare values
    if (timestep_a_value > timestep_b_value){
        // Return 0 if timestep a value is greater
        return 0;
    } else {
        // Return 1 if timestep b value is greater
        return 1;
    }
}



/**
 * timestep_vector_sort:
 * Compares a vector of timestep strings and sorts them.
 * 
 * Input:
 * timesteps        - vector of timestep strings
 * sorted_timesteps - sorted vector of timestep strings
 * rx               - timestep string regular expression
 **/
int timestep_vector_sort(std::vector<std::string> timesteps, std::vector<std::string> &sorted_timesteps, std::regex rx){
    // Allocate space for sorted timesteps
    //sorted_timesteps.resize(timesteps.size(), 0);
    // Keep count of sorted filenames
    int sorted_filenames = 0;
    // Sort timesteps
    for (uint i = 0; i < timesteps.size(); i++){
        // Get current timesteps
        std::string timestep_a = timesteps[i];
        int position = 0;
        // Compare against all other timesteps
        for (uint j = 0; j < timesteps.size(); j++){
            if (i != j){
                // Get next string value
                std::string timestep_b = timesteps[j];
                // Determine which string comes first
                int loc = timestep_sort(timestep_a, timestep_b, rx);
                // If timestep a is greater, 
                if (loc == 0){
                    position++;
                }
            }
        }
        // Store timestep_a at correct position
        sorted_timesteps[position] = timestep_a;
        sorted_filenames++;
    }

    // Return count of sorted filenames
    return sorted_filenames;
}