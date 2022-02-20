import sys
import os.path
import Configure_Functions
import subprocess
from subprocess import call
import time
from os import walk
import re
import itertools

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def main():

    if(len(sys.argv) == 11):
        in_file_path = sys.argv[1]
        XDIM = int(sys.argv[2])
        YDIM = int(sys.argv[3])
        ZDIM = int(sys.argv[4])
        XBLOCK = int(sys.argv[5])
        YBLOCK = int(sys.argv[6])
        ZBLOCK = int(sys.argv[7])
        output_file_path = sys.argv[8]  
        True_Value_Folder_Path = sys.argv[9]
        region_of_interest_locs_Path = sys.argv[10]
    else:
        print("Incorrect Number of Arguements Passed!")
        print("python configure_main.py <in_file_path>")
        exit(0)

    # Get file extension
    in_file_ext = (in_file_path).split('.')[-1:]
    print("File extension: ", in_file_ext) 
    in_file_name = ((in_file_path).split('/')[-1:])[0]
    print("File Name: ", in_file_name)
    in_file_dir = (in_file_path)[:-(len(in_file_name))]
    print("File dir:", in_file_dir)

    # create vti files from vtp
    cur_samp = "linear"
    input_folder = in_file_path
    
    if (os.path.isdir(input_folder)): # given a folder
    
        # get list of all input files in folder
        contents = os.listdir(os.path.join(".",input_folder))

        filenames_list = []
        for entry in contents:
            #print(entry)
            if entry.endswith(".vtp"):
                filenames_list.append(entry)
        num_files = len(filenames_list)
        print("Files to Analyze: ", num_files)
        for vtp_name in filenames_list:
            print("Reading in file: ",vtp_name)
            # PATHS TO RECONSTRUCT AND CALCULATE SNR
            Linear_Reconstruction_Path = "./reconstruct_dataset.py"

            # RECONSTRUCT
            # Run reconstruction script and collect the string output
            # TODO input cur_samp value
            cmd = "python {} {} {} {} {} {}".format(Linear_Reconstruction_Path, input_folder+"/"+vtp_name, XDIM, YDIM, ZDIM, input_folder)
            out_str = subprocess.check_output(cmd, shell=True)
            filename, file_extension = os.path.splitext(os.path.basename(input_folder+"/"+vtp_name))
            reconstructed_file_name = input_folder + '/recons_'+filename+'_'+cur_samp+'.vti'
            print("created: ", reconstructed_file_name)

    

    ROI_table = [] # read in this table from file
    # Get ROI locs from file
    if region_of_interest_locs_Path != 'NA':
        with open(region_of_interest_locs_Path) as f:
            for line in f:
                temp = re.compile(r'[^\d,]+').sub('', line)
                temp = temp.split(",")
                if (len(temp) == 7):
                    temp = temp[:-1]
                ROI_table.append(list(map(int, temp)))
    print("\n\n\n")
    print("ROI INPUT:")
    print(ROI_table)
    print("\n\n\n")

    # Calculate SNR
    if True_Value_Folder_Path != 'NA':
        SNR_Analysis_Path = "../../data_sampling/reuse_sampler/compute_SNR.py"

        # get all true files
        contents = os.listdir(os.path.join(".",True_Value_Folder_Path))
        filenames_list_true = []
        for entry in contents:
            #print(entry)
            if entry.endswith(".vti"):
                filenames_list_true.append(entry)
        num_files = len(filenames_list_true)
        filenames_list_true.sort(key=natural_keys)

        # get all reconstructed files
        contents = os.listdir(os.path.join(".",in_file_path))
        filenames_list_recons = []
        for entry in contents:
            #print(entry)
            if entry.endswith(".vti"):
                filenames_list_recons.append(entry)
        num_files = len(filenames_list_recons)
        filenames_list_recons.sort(key=natural_keys)

        statistics_out_file = in_file_path+"/SNR_results.csv"
        f = open(statistics_out_file, 'w') # append results to file
        f.write("filename"+"," + "SNR" + "," + "SNR_ROI" + "\n")
        f.close()
        for timestep in range(num_files):
            #if len(ROI_table) == len(in_files): # if we correctly have the same number of ROIs as we do time-steps
            if len(ROI_table) >= num_files: # if we correctly have the same number of ROIs as we do time-steps
                x_start = ROI_table[timestep][0]
                x_end = ROI_table[timestep][1]
                y_start = ROI_table[timestep][2]
                y_end = ROI_table[timestep][3]
                z_start = ROI_table[timestep][4]
                z_end = ROI_table[timestep][5]

                cmd = "python {} {} {}".format(SNR_Analysis_Path, True_Value_Folder_Path+"/"+filenames_list_true[timestep], in_file_path+"/"+filenames_list_recons[timestep])
                out_str = subprocess.check_output(cmd, shell=True)
                SNR = float((((str(out_str)).split()[-1]))[:-3])
                cmd = "python {} {} {} {} {} {} {} {} {}".format(SNR_Analysis_Path, True_Value_Folder_Path+"/"+filenames_list_true[timestep], in_file_path+"/"+filenames_list_recons[timestep], x_start, x_end, y_start, y_end, z_start, z_end)
                out_str = subprocess.check_output(cmd, shell=True)
                SNR_ROI = float((((str(out_str)).split()[-1]))[:-3])

                print("SNR: ", SNR)
                print("SNR ROI: ", SNR_ROI) 

            else:
                cmd = "python {} {} {}".format(SNR_Analysis_Path, True_Value_Folder_Path+"/"+filenames_list_true[timestep], in_file_path+"/"+filenames_list_recons[timestep])
                out_str = subprocess.check_output(cmd, shell=True)
                SNR = float((((str(out_str)).split()[-1]))[:-3])
                SNR_ROI = 0
                print("SNR: ", SNR)


            # write out to csv
            f = open(statistics_out_file, 'a') # append results to file
            f.write(str(True_Value_Folder_Path+"/"+filenames_list_true[timestep])+"," + str(SNR) + "," + str(SNR_ROI) + "\n")
            f.close()

if __name__ == '__main__':
    main()
