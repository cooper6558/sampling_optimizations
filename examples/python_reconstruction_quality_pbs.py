import subprocess
import re
from datetime import datetime
import time
import math
import os
import sys
from os import walk

# python python_reconstruction_quality_pbs.py isabel 1 0.005 pressure_full_m1_s0.005_t1/  500 500 100 50 50 50

def main():
    if len(sys.argv) == 11:
        # Get input needed to run
        dataset_name = sys.argv[1].strip()
        sampling_method = sys.argv[2]
        sample_ratio = sys.argv[3]
        in_file_path = sys.argv[4]
        XDIM = int(sys.argv[5])
        YDIM = int(sys.argv[6])
        ZDIM = int(sys.argv[7])
        XBLOCK = int(sys.argv[8])
        YBLOCK = int(sys.argv[9])
        ZBLOCK = int(sys.argv[10])
        output_file_path =  "." # TODO I think this has to be this in order to work with next script
    elif len(sys.argv) == 12:
        # Get input needed to run
        dataset_name = sys.argv[1].strip()
        sampling_method = sys.argv[2]
        sample_ratio = sys.argv[3]
        in_file_path = sys.argv[4]
        XDIM = int(sys.argv[5])
        YDIM = int(sys.argv[6])
        ZDIM = int(sys.argv[7])
        XBLOCK = int(sys.argv[8])
        YBLOCK = int(sys.argv[9])
        ZBLOCK = int(sys.argv[10])
        True_Value_Folder_Path = sys.argv[11]
        region_of_interest_locs_Path = "NA"  # no ROI SNR
        output_file_path =  "." # TODO I think this has to be this in order to work with next script
    elif len(sys.argv) == 13:
        # Get input needed to run
        dataset_name = sys.argv[1].strip()
        sampling_method = sys.argv[2]
        sample_ratio = sys.argv[3]
        in_file_path = sys.argv[4]
        XDIM = int(sys.argv[5])
        YDIM = int(sys.argv[6])
        ZDIM = int(sys.argv[7])
        XBLOCK = int(sys.argv[8])
        YBLOCK = int(sys.argv[9])
        ZBLOCK = int(sys.argv[10])
        True_Value_Folder_Path = sys.argv[11]
        region_of_interest_locs_Path = sys.argv[12]
        output_file_path =  "." # TODO I think this has to be this in order to work with next script
    else:
        print("Incorrect Number of Arguements. . .")
        exit(-1)
	
    # Palmetto PBS script options
    ncpus = "40"
    phase = "18b"
    mem = "372gb"
    walltime = "48:00:00"
    mpiprocs = "40"


    # Write to a PBS script
    experiment_name = "{}_m{}_s{}".format(dataset_name, sampling_method, sample_ratio)
    stats_file_name = "{}.csv".format(experiment_name)
    #node_pbs_name = "{}_temp_pbs_node_{}.pbs".format(experiment_name,experiment_id)
    stats_output_folder = "{}/".format(experiment_name)
    node_pbs_name = "{}.pbs".format(experiment_name)
    with open(node_pbs_name, 'w+') as pbs_script:
        pbs_script.write("#!/bin/bash\n\n")
        pbs_script.write("#PBS -N {}\n".format(experiment_name))
        
        param_line = "#PBS -l select=1:ncpus={}:ngpus=1:mem={}:phase={},walltime={} -q fthpc\n\n".format(ncpus, mem, phase, walltime)
        
        pbs_script.write(param_line)
        pbs_script.write("cd\n")
        pbs_script.write("module add gcc/7.1.0\n")
        pbs_script.write("module add cuda/10.0.130-gcc\n")
        pbs_script.write("module add openmpi/3.1.4-gcc\n")
        #pbs_script.write("conda init bash\n")
        pbs_script.write("cd Hybrid_Data_Sampling/\n")
        pbs_script.write("make\n")
        pbs_script.write("cd examples/\n")
        pbs_script.write("make\n")
        pbs_script.write("source /home/mlhickm/.conda/pkgs/conda-4.8.4-py37_0/lib/python3.7/site-packages/conda/shell/etc/profile.d/conda.sh\n")
        pbs_script.write("conda init bash\n")        
        pbs_script.write("conda activate myenv\n")

        if len(sys.argv) == 11:
            run_line = "python ../src/python_reconstruction_quality.py {} {} {} {} {} {} {} {}".format(in_file_path, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, output_file_path)
        elif len(sys.argv) == 12:
            run_line = "python ../src/python_reconstruction_quality.py {} {} {} {} {} {} {} {} {}".format(in_file_path, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, output_file_path, True_Value_Folder_Path)
        elif len(sys.argv) == 13:
            run_line = "python ../src/python_reconstruction_quality.py {} {} {} {} {} {} {} {} {} {}".format(in_file_path, XDIM, YDIM, ZDIM, XBLOCK, YBLOCK, ZBLOCK, output_file_path, True_Value_Folder_Path, region_of_interest_locs_Path)

        print(run_line)
        pbs_script.write(run_line)

    pbs_script.close()

    time.sleep(5)
    # Run the PBS script
    os.system("qsub {}".format(node_pbs_name))
    print("Jobs Have Been Submitted")

    #print("UNCOMMENT TO SUBMIT")
if __name__ == '__main__':
	main()
