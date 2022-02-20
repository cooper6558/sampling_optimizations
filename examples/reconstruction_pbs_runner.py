import subprocess
import re
from datetime import datetime
import time
import math
import os
import sys
from os import walk

def main():
    if len(sys.argv) == 12:
        # Get input needed to run
        dataset_name = sys.argv[1].strip()
        file_name_regular_expression = (sys.argv[2])
        input_folder = (sys.argv[3])
        input_data_dims = (sys.argv[4])
        input_region_dims = (sys.argv[5])
        sample_ratio = (sys.argv[6])
        input_min = (sys.argv[7])
        input_max = (sys.argv[8])
        number_of_bins = (sys.argv[9])
        sampling_method = (sys.argv[10])
        recons_method = (sys.argv[11])
    else:
        print("Incorrect Number of Arguements. . .")
        exit(-1)
	
    # Palmetto PBS script options
    ncpus = "40"
    phase = "18b"
    mem = "62gb"
    walltime = "48:00:00"
    gpu = "v100"


    # Write to a PBS script
    experiment_name = "{}_{}_{}".format(dataset_name, sampling_method, sample_ratio)
    stats_file_name = "{}_recons.csv".format(experiment_name)
    #node_pbs_name = "{}_temp_pbs_node_{}.pbs".format(experiment_name,experiment_id)
    stats_output_folder = "{}/".format(experiment_name)
    node_pbs_name = "{}.pbs".format(experiment_name)
    with open(node_pbs_name, 'w+') as pbs_script:
        pbs_script.write("#!/bin/bash\n\n")
        pbs_script.write("#PBS -N {}\n".format(experiment_name))
        #pbs_script.write("#PBS -N Sampling_Experiment\n")
        
        #select=1:ncpus=40:ngpus=1:mem=62gb:gpu_model=v100:phase=18b,walltime=48:00:00 -q fthpc
        param_line = "#PBS -l select=1:ncpus={}:ngpus=1:mem={}:gpu_model={}:phase={},walltime=48:00:00 -q fthpc\n\n".format(ncpus, mem, gpu, phase, walltime)
        pbs_script.write(param_line)
        pbs_script.write("cd\n")
        pbs_script.write("module add gcc/7.1.0\n")
        pbs_script.write("module add cuda/9.2.88-gcc\n")
        pbs_script.write("module add openmpi/3.1.4-gcc\n")
        #pbs_script.write("conda init bash\n")
        pbs_script.write("cd Hybrid_Data_Sampling/\n")
        pbs_script.write("make\n")
        pbs_script.write("cd examples/\n")
        pbs_script.write("make\n")
        
        run_line = "./reconstruction_call -e {} -i {} -d {} -b {} -p {} -m {} -x {} -n {} -s {} -r {} -o {} -f {}".format(file_name_regular_expression, input_folder, input_data_dims, input_region_dims, sample_ratio, input_min, input_max, number_of_bins, sampling_method, recons_method, stats_output_folder, stats_file_name)


        print(run_line)
        pbs_script.write(run_line)
    pbs_script.close()

    time.sleep(5)
    # Run the PBS script
    #os.system("qsub {}".format(node_pbs_name))
    #print("Jobs Have Been Submitted")

    print("UNCOMMENT TO SUBMIT")
if __name__ == '__main__':
	main()