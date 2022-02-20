import sys
import os.path
import Configure_Functions
import subprocess
from subprocess import call

def main():
    
    if (len(sys.argv) == 2):
        in_file_path = sys.argv[1]

    elif(len(sys.argv) == 9):
        in_file_path = sys.argv[1]
        XDIM = int(sys.argv[2])
        YDIM = int(sys.argv[3])
        ZDIM = int(sys.argv[4])
        XBLOCK = int(sys.argv[5])
        YBLOCK = int(sys.argv[6])
        ZBLOCK = int(sys.argv[7])
        output_file_path = sys.argv[8]
    else:
        print("Incorrect Number of Arguements Passed!")
        print("python configure_main.py <in_file_path>")
        exit(0)

    # Get file extension
    in_file_ext = (in_file_path).split('.')[-1:]
    #print("File extension: ", in_file_ext) 
    in_file_name = ((in_file_path).split('/')[-1:])[0]
    #print(in_file_name)
    in_file_dir = (in_file_path)[:-(len(in_file_name))]
    #print(in_file_dir)

    if (in_file_ext[0] == 'vti'): # .vti to .dat
        if not (os.path.isfile(in_file_path)):
            print("Input File Not Found!")
            exit(0)

        # Use LANL class manager to get correct structure
        in_file = Configure_Functions.DataManager(in_file_path, 0)    

        full_data = Configure_Functions.extractData(in_file)
        in_file_path_as_binary = in_file_dir+'binary_files/'+in_file_name[:-4]+".bin"
        if not os.path.exists(in_file_dir):
            os.makedirs(in_file_dir) # Create if doesn't exist
        else:
            shutil.rmtree(in_file_dir) # or delete and create if pre-existing
            os.makedirs(in_file_dir)
            
        Configure_Functions.write_vti_to_binary(in_file_path_as_binary, full_data)
        print("Successfully Created file: "+in_file_path_as_binary)
    
    elif (in_file_ext == 'dat'): # .dat to .vti
        if not (os.path.isfile(in_file_path)):
            print("Input File Not Found!")
            exit(0)

        # Use LANL class manager to get correct structure
        #in_file = Configure_Functions.DataManager(in_file_path, 0)
        #full_data = Configure_Functions.extractData(in_file)
        #in_file_path_as_binary = in_file_dir+'binary_files/'+in_file_name+".bin"
        #Configure_Functions.write_vti_to_binary(in_file_path_as_binary, full_data)
    
    
    elif (os.path.isdir(in_file_path)): # given a folder
    
        # get list of all input files in folder
        contents = os.listdir(os.path.join(".",in_file_path))

        filenames_list = []
        for entry in contents:
            #print(entry)
            if entry.endswith(".bin"):
                filenames_list.append(entry)
        
        num_files = len(filenames_list)
                
        if num_files % 3 != 0: 
            print("ERROR IN NUM FILES!\n")
            exit(0)

        num_timesteps = num_files / 3
        print(in_file_path, num_files, num_timesteps)

        vtp_dir = in_file_path+"/vtp_files"
        # make folder for output vtp files
        if not (os.path.isdir(vtp_dir)):
            os.mkdir(vtp_dir) 
            print("Directory '% s' is built!" % vtp_dir) 

        for timestep in range(int(num_timesteps)):
            # CREATE VTP for each file
            # sample locs and value lists to .vtp
            vtp_name = vtp_dir+'/sampled_'+str(timestep)+'.vtp'
            

            curr_id = in_file_path+"/sampled_id_"+str(timestep)+".bin"
            curr_data = in_file_path+"/sampled_data_"+str(timestep)+".bin"
            curr_total = in_file_path+"/sampled_total_"+str(timestep)+".bin"

            if timestep > 0:
                prev_id = in_file_path+"/sampled_id_"+str(timestep-1)+".bin"
                prev_data = in_file_path+"/sampled_data_"+str(timestep-1)+".bin"
                prev_total = in_file_path+"/sampled_total_"+str(timestep-1)+".bin"
                # TODO Write and read bm and XDIM,YDIM,ZDIM from file
                Configure_Functions.create_vtp_file_from_global_ids(XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, vtp_name, curr_id, curr_data, curr_total, prev_id, prev_data, prev_total)
            else:
                # TODO Write and read bm and XDIM,YDIM,ZDIM from file
                Configure_Functions.create_vtp_file_from_global_ids(XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, vtp_name, curr_id, curr_data, curr_total)
        print("CREATED :", vtp_name)

    elif (in_file_ext[0] == 'vtp'): # .vtp to .vti
        vtp_name = in_file_path
        print("Reading in file: ",vtp_name)
        # PATHS TO RECONSTRUCT AND CALCULATE SNR
        Linear_Reconstruction_Path = "/home/mlhickm/Hybrid_Data_Sampling/src/reconstruct_dataset.py"
        #SNR_Analysis_Path = "/Users/mlhfulp/data_sampling/reuse_sampler/compute_SNR.py"

        # RECONSTRUCT
        out_folder = in_file_dir
        # Run reconstruction script and collect the string output
        cmd = "python {} {} {} {} {} {}".format(Linear_Reconstruction_Path, vtp_name, XDIM, YDIM, ZDIM, out_folder)
        out_str = subprocess.check_output(cmd, shell=True)
    else:
        print("No way to handle input!")
        exit(0)


    return 0

if __name__ == '__main__':
    main()
