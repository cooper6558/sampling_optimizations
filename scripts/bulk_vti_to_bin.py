import sys
import os.path
import shutil
import Configure_Functions

def main():
    
    if (len(sys.argv) == 2):
        in_file_path = sys.argv[1]
    else:
        print("Incorrect Number of Arguements Passed!")
        exit(0)

    # if is existing directory:
    if (os.path.isdir(in_file_path)): # given a folder

        # create binary files folder
        binary_folder = in_file_path+'/binary_files'
        if not os.path.exists(binary_folder):
            os.makedirs(binary_folder) # Create if doesn't exist
        else:
            shutil.rmtree(binary_folder) # or delete and create if pre-existing
            os.makedirs(binary_folder)

        # For file in directory
        for filename in os.listdir(in_file_path):
            if filename.endswith(".vti"):
                print(os.path.join(in_file_path, filename))
                # Use LANL class manager to get correct structure
                in_file = Configure_Functions.DataManager(in_file_path+filename, 0)    

                full_data = Configure_Functions.extractData(in_file)
                in_file_path_as_binary = binary_folder+'/'+filename[:-4]+".bin"
                Configure_Functions.write_vti_to_binary(in_file_path_as_binary, full_data)
                print("Successfully Created file: "+in_file_path_as_binary)
            else:
                continue    
    else:
        print("No way to handle input!")
        exit(0)


    return 0

if __name__ == '__main__':
    main()
