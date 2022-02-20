# Data Sampling with GPU

## Workflow
0. If working with VTI data file, use python to convert to binary first
1. Have folder input of at least 2 binary data files of a data series
2. Using C++, read in binary file of data and gather samples by use of:
    a. importance-based sampling algorithm, or
    b. histogram-reuse sampling algorithm, or
    c. error-reuse sampling algorithm
3. Write out the samples as lists of locations and information, and their metadata
4. IF DESIRED: Using python, read in the sample information and create the VTP file
5. Reconstruct samples to full resolution

## HOW TO RUN
### Setup on Palmetto: 

** GPU Versions
`qsub -I -l select=1:ncpus=20:ngpus=1:mem=16gb:gpu_model=p100,walltime=48:00:00`. 

** OpenMP Versions
`qsub -I -l select=1:ncpus=20:mpiprocs=20:ngpus=1:mem=16gb:gpu_model=p100,walltime=48:00:00`.

### Local Setup:

module add gcc/7.1.0

## module add cuda/9.2.88-gcc
module add cuda/10.0.130-gcc

module add openmpi/3.1.4-gcc

### Compile with
cd Hybrid_Data_Sampling/

make

cd examples/

make

### Run with

Usage:

`./interactive -e <file_name_regular_expression> -i <input_data_file_FOLDER> -d <input_data_dims> -b <input_region_dims> -p <sample_ratio> -m <input_min> -x <input_max> -n <number_of_bins>  -s <sampling_method> -r <recons_method> -t <max_OpenMP_threads> -o <stats_output_folder> -f <stats_output_csv>`

`<file_name_regular_expression>` - regular expression of input file names.  

    1. - ExaAM "plt_temperature_([0-9]+).bin"  
    2. - Hurricane Pressure "Pf([0-9]+).binLE.raw_corrected_2_subsampled.vti.bin"  
    3. - Hurricane Pressure ZFS "Pf([0-9]+).bin"  
    4. - Asteroid "pv_insitu_300x300x300_([0-9]+).bin"  
    
`<input_data_file_FOLDER>` - path to folder of binary file time-steps to be sampled  
`<input_data_dims>` - X, Y, Z Dimensions of input binary file data set  
`<input_region_dims>` - X, Y, Z Dimensions of regions to divide the data set into  
    1. - ExaAM - [20, 200, 50]
    2. - Hurricane Pressure (sub-sampled) - [250, 250, 50]
    3. - Hurricane Pressure (full-resolution) - [500, 500, 100]
    4. - Asteroid - [300, 300, 300]
`<sample_ratio>` - sample ratio to keep of data  
`<input_min>` `<input_max>` - min/max lifetime value - used when using histogram reuse method, use current timestep range when using single timestep  

    1. - ExaAM - [300.271, 927.426] (NOTE: extracted from ExaAM timestep 52 - TODO find range over entire time series)  
    2. - Hurricane Pressure - [-5471.85791 3225.42578]  
    3. - Asteroid - [0,1]  
    
`<number_of_bins>` - number of bins to use with histograms within the method  
`<sampling_method>` - Sampling Methods:   

    1. - Importance Based Only (Serial)  
    2. - Importance Based Only (OpenMP) 
    3. - Importance Based Only (CUDA) 
    4. - Histogram Based Reuse (Serial)  
    5. - Histogram Based Reuse (OpenMP)
    6. - Histogram Based Reuse (CUDA)  
    7. - Error Based Reuse (Serial)
    8. - Error Based Reuse (OpenMP)
    9. - Error Based Reuse (CUDA) (TODO)
    
`<recons_method>` - Reconstruction Methods:   

    1. - Nearest Neighbors (Serial)  
    2. - Nearest Neighbors (OpenMP)   
    3. - Nearest Neighbors (CUDA)  
    4. - 3NN (Serial) 
    5. - 3NN (OpenMP)   
    6. - 3NN (CUDA)  
    
`<stats_output_folder>` - folder to write final timings and quality metrics statistics and reconstructed binary files to  


#### Example Call

Example: 
`./interactive -e 1 -i example_input/ -d 20,200,50 -b 4,20,10 -p 0.01 -m 300.271 -x 927.426 -n 20 -s 1 -r 1 -t 10 -o output_files/ -f statistics.csv`

./interactive -e 2 -i ../../Datasets/BINARY_INPUT/pressure/ -d 250,250,50 -b 25,25,25 -p 0.02 -m -5471.85791 -x 3225.42578 -n 29 -s 5 -r 3 -t 10 -o output_files/ -f statistics.csv



To gather GPU statistics: 

`nvprof --profile-from-start off ./interactive -e 1 -i example_input/ -d 20,200,50 -b 4,20,10 -p 0.01 -m 300.271 -x 927.426 -n 20 -s 2 -r 2 -t 10 -o output_files/ -f statistics.csv`

nvprof --profile-from-start off --log-file human-readable-output.log ./interactive -e 1 -i example_input/ -d 20,200,50 -b 4,20,10 -p 0.01 -m 300.271 -x 927.426 -n 20 -s 3 -r 3 -o output_files/ -f statistics.csv


## Datasets to Test

* Ten time-steps of the ExaAM dataset are provided for fast debugging and example tests.
* To get access to other datasets, use binary files in `/zfs/fthpc/common/sdrbench/` (NOTE: `<file_name_regular_expression>` needs to reflect the file name convention).


## Using the PBS runner


On login node, use the `whatsfree` command to see what nodes are available.  

Edit the file `experiment_pbs_runner.py` to reflect the node you want to use.




## MISC

TODO - find all '-1' s in all files, and update with `reuse_flag` variable
