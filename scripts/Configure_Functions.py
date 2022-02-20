
# Confgure_Functions.py

import sys
import shutil
import subprocess
from subprocess import call
import numpy as np
import argparse
import re
from os import walk
import os.path
import vtk
from vtk.util import numpy_support as VN
from array import array
import math

# LANL CLASS CODE
class DataManager(object):
    
    def __init__(self, fname, target_var_id):

        self.fname = fname
        self.varId = target_var_id
        self.f_extension = os.path.splitext(os.path.basename(self.fname))[1]
        self.supported_extenstions = ['.vtp', '.vtk', '.vtu', '.vti']
        self.varName = ""
        self.data = []
        self.grad = []
        self.dataSpacing = []
        self.dataOrigin = []
        self.dataExtent = []

        if self.f_extension in self.supported_extenstions and self.varId > -1:
            self.update()
        else:
            print("check supported file extensions and/or variable id")
            raise SystemExit


    # def __del__(self):
    #     return 1; 
    #     #print("Cluster ", self.no ," Destructor called") 

    def update(self):        

        if self.f_extension=='.vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif self.f_extension=='.vtk':
            reader = vtk.vtkGenericDataObjectReader()
        elif self.f_extension=='.vtu':
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif self.f_extension=='.vti':
            reader = vtk.vtkXMLImageDataReader()
        
        reader.SetFileName(self.fname)
        reader.Update()
        vtk_data = reader.GetOutput()

        self.XDIM, self.YDIM, self.ZDIM = vtk_data.GetDimensions()
        self.dataSpacing = np.asarray(vtk_data.GetSpacing())
        self.dataOrigin = np.asarray(vtk_data.GetOrigin())
        self.dataExtent = np.asarray(vtk_data.GetExtent())
        self.varName = vtk_data.GetPointData().GetArrayName(self.varId)

        self.data = vtk_data.GetPointData().GetArray(self.varId)
        self.data = VN.vtk_to_numpy(self.data)
        self.data = self.data.reshape((self.ZDIM, self.YDIM, self.XDIM))

        #self.data = np.copy(vals)

        self.min = np.min(self.data.flatten())
        self.max = np.max(self.data.flatten())

        self.grad_min = []
        self.grad_max = []


    def get_datafield(self):
        return self.data;

    def get_dimension(self):
        return self.XDIM, self.YDIM, self.ZDIM;

    def get_spacing(self):
        return self.dataSpacing;

    def get_origin(self):
        return self.dataOrigin;

    def get_extent(self):
        return self.dataExtent;

    def get_varName(self):
        return self.varName;

    def get_gradfield(self):
        return self.grad;

    def __str__(self):
        return "\n \
            FILENAME:{} \n \
            XDIM:{} \t YDIM:{} \t ZDIM:{}\n \
            VAR NAME:{} \n \
            SPACING:{} \n \
            ORIGIN:{} \n \
            EXTENT:{} \n \
            SHAPE:{} " \
            .format(self.fname, self.XDIM, self.YDIM, self.ZDIM, self.varName, \
                self.dataSpacing, self.dataOrigin, self.dataExtent, self.data.shape)

    def __repr__(self):
        return self.__str__()

    def getMin(self): 
        return self.min

    def getMax(self): 
        return self.max

    def getGradMin(self): 
        return self.grad_min

    def getGradMax(self): 
        return self.grad_max


# LANL CLASS CODE
class BlockManager(object):

    def __init__(self, partitionType, partitionParameteList):
        
        self.partitionType = partitionType 

        if partitionType=='regular':
            #extract parameters
            self.XBLOCK = np.int(partitionParameteList[0])
            self.YBLOCK = np.int(partitionParameteList[1])
            self.ZBLOCK = np.int(partitionParameteList[2])

            self.XDIM = np.int(partitionParameteList[3])
            self.YDIM = np.int(partitionParameteList[4])
            self.ZDIM = np.int(partitionParameteList[5])

            self.d_origin = partitionParameteList[6]
            self.d_spacing = partitionParameteList[7]
            self.d_extent = partitionParameteList[8]

            self.XB_COUNT = np.int(self.XDIM/self.XBLOCK)
            self.YB_COUNT = np.int(self.YDIM/self.YBLOCK)
            self.ZB_COUNT = np.int(self.ZDIM/self.ZBLOCK)
            
            

            #self.regularPartition(data)

        elif partitionType=='kdtree':
            #extract parameters
            #kdtreePartition(data)
            print("kdtree not implemented")
            raise SystemExit
        elif partitionType=='slic':
            #extract parameters
            #slicPartition(data)
            print("slic not implemented")
            raise SystemExit
        else:
            print("unsupported partitioning")
            raise SystemExit

    def partition(self, data):

        if self.partitionType=='regular':
            print("partitioning removed")
            raise SystemExit

        elif self.partitionType=='kdtree':
            #extract parameters
            #kdtreePartition(data)
            print("kdtree not implemented")
            raise SystemExit
        elif self.partitionType=='slic':
            #extract parameters
            #slicPartition(data)
            print("slic not implemented")
            raise SystemExit
        else:
            print("unsupported partitioning")
            raise SystemExit

    

    def numberOfBlocks(self):
        return self.XB_COUNT * self.YB_COUNT * self.ZB_COUNT;

    def get_blockData(self, data, bid):
        # block_size = self.XBLOCK * self.YBLOCK * self.ZBLOCK
        # block_data = np.zeros((block_size,))

        fid_s, tx_s, ty_s, tz_s = self.func_block_2_full(bid, 0)

        block_data = data[tz_s:tz_s+self.ZBLOCK, ty_s:ty_s+self.YBLOCK, tx_s:tx_s+self.XBLOCK]
        
        # for lid in range(block_size):
        #     fid, tx, ty, tz = self.func_block_2_full(bid, lid)
        #     block_data[lid] = data[tz][ty][tx]

        return block_data.flatten() 


    def func_xyz_2_fid(self, x, y, z):
        if x >= self.XDIM or y >= self.YDIM or z >= self.ZDIM or x < 0 or y < 0 or z < 0:
            print("out of bound_func_xyz_2_fid")
            raise SystemExit
        return z*self.YDIM*self.XDIM + y*self.XDIM + x

    def func_fid_2_xyz(self, fid):
        if fid >= self.XDIM*self.YDIM*self.ZDIM or fid < 0:
            print("out of bound - func_fid_2_xyz")
            raise SystemExit
        fid = np.int(fid)
        z = np.int(fid / (self.XDIM * self.YDIM))
        fid -= np.int(z * self.XDIM * self.YDIM)
        y = np.int(fid / self.XDIM)
        x = np.int(fid % self.XDIM)
        return x, y, z

    def func_block_2_full(self, bid, local_id=0):
        
        
        if bid >= self.XB_COUNT*self.YB_COUNT*self.ZB_COUNT or bid < 0:
            print("out of bound - func_block_2_full")
            #print(bid, " is greater than ", self.XB_COUNT*self.YB_COUNT*self.ZB_COUNT)
            raise SystemExit
        bid = np.int(bid)
        bz = np.int(bid / (self.XB_COUNT * self.YB_COUNT))
        bid -= np.int(bz * self.XB_COUNT * self.YB_COUNT)
        by = np.int(bid / self.XB_COUNT)
        bx = np.int(bid % self.XB_COUNT)
        
        x = np.int(bx*self.XBLOCK)
        y = np.int(by*self.YBLOCK)
        z = np.int(bz*self.ZBLOCK)
        
        
        if int(local_id) >= self.XBLOCK*self.YBLOCK*self.ZBLOCK or int(local_id) < 0:
            print("[local id] out of bound: ", local_id, " ", self.XBLOCK*self.YBLOCK*self.ZBLOCK)
            raise SystemExit
            
        local_id = np.int(local_id)
        local_z = np.int(local_id / (self.XBLOCK * self.YBLOCK))
        local_id -= np.int(local_z * self.XBLOCK * self.YBLOCK)
        local_y = np.int(local_id / self.XBLOCK)
        local_x = np.int(local_id % self.XBLOCK)
        
        fx = x + local_x
        fy = y + local_y
        fz = z + local_z
                        
        
        fid = fz*self.YDIM*self.XDIM + fy*self.XDIM + fx
        
        return fid, fx, fy, fz

    def func_logical_2_physical_location(self, x, y, z):
        px = x*self.d_spacing[0]
        py = y*self.d_spacing[1]
        pz = z*self.d_spacing[2]
        
        return px, py, pz

    def func_physical_2_logical_location(self, x, y, z):
        lx = x/self.d_spacing[0]
        ly = y/self.d_spacing[1]
        lz = z/self.d_spacing[2]
        
        return lx, ly, lz



def extractData(input_data):
    # Extract data information
    #XDIM, YDIM, ZDIM = input_data.get_dimension()
    #d_spacing = input_data.get_spacing()
    #d_origin = input_data.get_origin()
    #d_extent = input_data.get_extent()
    # Extract full data
    full_data = input_data.get_datafield()
    # Create block manager
    #bm_parameter_list = [XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent]
    #bm = FS.BlockManager('regular', bm_parameter_list)
    return full_data


def write_vti_to_binary(out_name, in_file_data):
    in_file_name = ((out_name).split('/')[-1:])[0]
    in_file_dir = (out_name)[:-(len(in_file_name))]

    # Write out vti data as binary file
    output_file = open(out_name, 'wb')
    one_d_data = in_file_data.ravel()
    float_array = array('f', one_d_data)
    float_array.tofile(output_file)
    output_file.close()

def write_samples_to_binary(out_folder, list_sampled_lid, list_sampled_data, list_of_boundary_information):
    
    # Writes out three files:
    # 1. Locations of Samples
    sampled_lid_file = out_folder+"/locs_and_meta/sampled_lid.dat"
    # 2. Data values of those samples
    sampled_data_file = out_folder+"/sampled_data.dat"
    # 3. Metadata for reconstruction
    meta_data_file = out_folder+"/locs_and_meta/meta_data.dat"

    if not os.path.exists(out_folder):
        os.makedirs(out_folder) # Create if doesn't exist
    else:
        shutil.rmtree(out_folder) # or delete and create if pre-existing
        os.makedirs(out_folder)

    if not os.path.exists(out_folder+"/locs_and_meta"):
        os.makedirs(out_folder+"/locs_and_meta") # Create if doesn't exist
    else:
        shutil.rmtree(out_folder+"/locs_and_meta") # or delete and create if pre-existing
        os.makedirs(out_folder+"/locs_and_meta")
    
    output_file = open(sampled_lid_file, 'wb')
    for item in list_sampled_lid:        
        float_array = array('f', item)
        float_array.tofile(output_file)
    output_file.close()

    output_file = open(sampled_data_file, 'wb')
    for item in list_sampled_data:        
        float_array = array('f', item)
        float_array.tofile(output_file)
    output_file.close()
    
    output_file = open(meta_data_file, 'wb')
    for item in range(0,3):
        float_array = array('f', list_of_boundary_information[item])
        float_array.tofile(output_file)
    
    temp = list_of_boundary_information[3:]
    for item in temp:
        float_array = array('f', [item])
        float_array.tofile(output_file)

    #for item in range(3,len(list_of_boundary_information)):
    #    list_of_boundary_information[item].tofile(output_file)
    output_file.close()


def read_samples_from_binary(out_folder):
    
    # Writes out three files:
    # 1. Locations of Samples
    sampled_lid_file = out_folder+"/sampled_lid.bin"
    # 2. Data values of those samples
    sampled_data_file = out_folder+"/sampled_data.bin"
    # 3. Number of Samples Per Block
    sampled_total_file = out_folder+"/sampled_total.bin"
    # 4. Metadata for reconstruction
    #meta_data_file = out_folder+"/locs_and_meta/meta_data.dat"
    
    try:
        input_file = open(sampled_lid_file, 'rb')
        returned_lid = array('i') 
        returned_lid.fromstring(input_file.read())
        input_file.close()
    except:
        print("CANNOT OPEN INPUT FILE: ", sampled_lid_file)
        exit(0)

    try:
        input_file = open(sampled_data_file, 'rb')
        returned_data = array('f')
        returned_data.fromstring(input_file.read())
        input_file.close()
    except:
        print("CANNOT OPEN INPUT FILE: ", sampled_data_file)
        exit(0)

    try:
        input_file = open(sampled_total_file, 'rb')
        returned_total = array('i')
        returned_total.fromstring(input_file.read())
        input_file.close()
    except:
        print("CANNOT OPEN INPUT FILE: ", sampled_total_file)
        exit(0)

    # Put returned data into same format as list_sampled_lid
    list_sampled_lid = []
    lid_sublist = []

    list_sampled_data = []
    data_sublist = []

    nob = len(returned_total)

    #print(returned_total[:250])
    nob = 250 # TODO HELP nob should be len(returned_total) but its not? - fixed on next main push
    
    # given samples per block list
    #counter = 0
    for num_samples_per_block in returned_total[:nob]:
        #print(counter)
        #counter = counter + 1
        for j in range(num_samples_per_block):
            #print(j, " ", num_samples_per_block)
            #print(returned_lid)
            lid_sublist.append(int(returned_lid[j]))
            data_sublist.append(returned_data[j])

        list_sampled_lid.append(lid_sublist)
        list_sampled_data.append(data_sublist)
        lid_sublist = []
        data_sublist = []
        if (num_samples_per_block > 0):
            del returned_lid[:num_samples_per_block]
            del returned_data[:num_samples_per_block]

    return list_sampled_lid, list_sampled_data


def read_samples_from_binary_files(sampled_id_file, sampled_data_file, sampled_total_file, bm_parameter_list):
    
    # TODO read in Metadata for reconstruction
    #meta_data_file = out_folder+"/locs_and_meta/meta_data.dat"
    
    try:
        input_file = open(sampled_id_file, 'rb')
        returned_lid = array('i') 
        returned_lid.fromstring(input_file.read())
        input_file.close()
    except:
        print("CANNOT OPEN INPUT FILE: ", sampled_id_file)
        exit(0)

    try:
        input_file = open(sampled_data_file, 'rb')
        returned_data = array('f')
        returned_data.fromstring(input_file.read())
        input_file.close()
    except:
        print("CANNOT OPEN INPUT FILE: ", sampled_data_file)
        exit(0)

    try:
        input_file = open(sampled_total_file, 'rb')
        returned_total = array('i')
        returned_total.fromstring(input_file.read())
        input_file.close()
    except:
        print("CANNOT OPEN INPUT FILE: ", sampled_total_file)
        exit(0)

    # Put returned data into same format as list_sampled_lid
    list_sampled_lid = []
    lid_sublist = []

    list_sampled_data = []
    data_sublist = []

    nob = len(returned_total)
    [XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent] = bm_parameter_list

    # Calculate number of blocks in each dimension
    XB_COUNT = int(XDIM/XBLOCK)
    YB_COUNT = int(YDIM/YBLOCK)
    ZB_COUNT = int(ZDIM/ZBLOCK)


    # given samples per block list
    reuse_flag = -1
    for block_id in range(nob):
        temp_bid = block_id
        num_samples_per_block = returned_total[block_id]
        if (num_samples_per_block == reuse_flag):
            list_sampled_lid.append(reuse_flag)
            list_sampled_data.append(reuse_flag)
        else:
            block_z = int(temp_bid / (XB_COUNT * YB_COUNT))
            temp_bid = int(temp_bid - (block_z * XB_COUNT * YB_COUNT))
            block_y = int(temp_bid / XB_COUNT)
            block_x = int(temp_bid % XB_COUNT)
            # Calculate block X,Y,Z start coordinates
            x_start = block_x*XBLOCK
            y_start = block_y*YBLOCK
            z_start = block_z*ZBLOCK

            # Iterate over block coordinates to gather all data values and global ID's
            for k in range(z_start, z_start+ZBLOCK):
                for j in range(y_start, y_start+YBLOCK):
                    for i in range(x_start, x_start+XBLOCK):
                        # get global offset ID
                        global_id = i + j*XDIM + k*XDIM*YDIM

                        for i in range(len(returned_lid)):
                            if returned_lid[i] == global_id:
                                # convert global_id to local_id
                                gx_id = int(global_id % XDIM)
                                #gy_id = math.floor((global_id / XDIM)) % YDIM
                                #gz_id = math.floor(global_id / (XDIM*YDIM))
                                gy_id = int((global_id / XDIM) % YDIM)
                                gz_id = int(global_id / (XDIM*YDIM))

                                lx_id = gx_id % XBLOCK
                                ly_id = gy_id % YBLOCK
                                lz_id = gz_id % ZBLOCK

                                local_id = lx_id + (ly_id*XBLOCK) + (lz_id*XBLOCK*YBLOCK)
                                #print(global_id, local_id)

                                lid_sublist.append(local_id)
                                data_sublist.append(returned_data[i])

                                # TODO make more efficient by popping off
                                #    del returned_lid[:num_samples_per_block]
                                #    del returned_data[:num_samples_per_block]

                                break

            list_sampled_lid.append(lid_sublist)
            list_sampled_data.append(data_sublist)
            lid_sublist = []
            data_sublist = []

    return list_sampled_lid, list_sampled_data


# Will construct vtp file from list of samples (can include reuse flags)
def create_vtp_file(in_fname, out_fname, ref_fname='NA'):
    
    # TODO read from metatdata file
    flag = "NA"
    XBLOCK = 4
    YBLOCK = 20
    ZBLOCK = 10
    XDIM = 20
    YDIM = 200
    ZDIM = 50
    d_origin = np.array([0., 0., 0.])
    d_spacing = np.array([1., 1., 1.])
    d_extent = np.array([  0,  19,   0, 199,   0,  49])
    
    
    # TODO THIS IS HARD CODED FOR EXAAM DATA SET
    list_of_boundary_information = [[0, 0, 0, 0, 19, 19, 19, 19], [0, 0, 199, 199, 0, 0, 199, 199], [0, 49, 0, 49, 0, 49, 0, 49], 300.00067065486246, 300.0, 300.0, 300.0, 568.7094662404135, 300.0, 300.0, 300.0]
    print(list_of_boundary_information)


    # HARD CODED FOR ISABEL
    #list_of_boundary_information = [[0, 0, 0, 0, 249, 249, 249, 249], [0, 0, 249, 249, 0, 0, 249, 249], [0, 49, 0, 49, 0, 49, 0, 49], 0.0, 20.057417, 2272.4014, 22.809683, 1063.6722, 6.953186, 1446.9822, 10.676723]

    # HARD CODED FOR ASTEROID
    #list_of_boundary_information = [[0, 0, 0, 0, 299, 299, 299, 299], [0, 0, 299, 299, 0, 0, 299, 299], [0, 299, 0, 299, 0, 299, 0, 299], 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


    # Create block manager
    bm_parameter_list = [XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent]
    bm = BlockManager('regular', bm_parameter_list)
    nob = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK)
    

    # TODO
    # if data flagged
    # LOAD REFERENCE DATA out of ref_fname
    if (ref_fname == 'NA'): ref_fname = in_fname
    list_sampled_lid_ref, list_sampled_data_ref = read_samples_from_binary(ref_fname)


    # load sample lid and data into array
    list_sampled_lid, list_sampled_data = read_samples_from_binary(in_fname)

    #print(list_sampled_lid)

    # For each block ...
    for bid in range(0, int(nob)):
        
        # If flagged, set equal to previous time step's information
        if (np.array_equal(list_sampled_lid[bid], flag)):
            list_sampled_lid[bid] = list_sampled_lid_ref[bid]
            list_sampled_data[bid] = list_sampled_data_ref[bid]
        try:
            if (list_sampled_lid[bid][0] == flag): #starts with flag
                # remove flag and append samples
                list_sampled_lid[bid].pop(0)
                # convert back to numpy
                list_sampled_lid[bid] = np.asarray(list_sampled_lid[bid], dtype=np.int64)
                list_sampled_lid[bid] = np.concatenate((list_sampled_lid[bid], list_sampled_lid_ref[bid]), axis=None)
                list_sampled_data[bid] = np.concatenate((list_sampled_data[bid], list_sampled_data_ref[bid]), axis=None)
        except: 
            continue

    Points = vtk.vtkPoints()

    
    #print("len lid: ", len(list_sampled_lid))
    for b in range(len(list_sampled_lid)):
        
        for i in range(len(list_sampled_lid[b])):
            #try:
            t_fid,t_fx,t_fy,t_fz = bm.func_block_2_full(b,list_sampled_lid[b][i])
            #except:
            #print("{} FAILED: {}".format(b,list_sampled_lid[b]))
            #pass
            Points.InsertNextPoint(t_fx,t_fy,t_fz)

    for i in range(8):
        Points.InsertNextPoint(list_of_boundary_information[0][i], list_of_boundary_information[1][i], list_of_boundary_information[2][i])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)
    
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName("samples")
        
    for b in range(len(list_sampled_data)):
        for i in range(len(list_sampled_data[b])):
            val_arr.InsertNextValue(list_sampled_data[b][i])
            
    for j in range(len(list_of_boundary_information[0])):
        val_arr.InsertNextValue(list_of_boundary_information[3+j])
        
    polydata.GetPointData().AddArray(val_arr)
    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_fname)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()
    print("Created file ", out_fname)
    total_samples = polydata.GetNumberOfPoints()
    
    return 0




# Will construct vtp file from list of samples (can include reuse flags)
def create_vtp_file_from_global_ids(XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, out_fname, curr_id, curr_data, curr_total, prev_id='NA', prev_data='NA', prev_total='NA'):
    
    # TODO read from metatdata file
    flag = -1
    #XBLOCK = 4
    #YBLOCK = 20
    #ZBLOCK = 10
    #XDIM = 20
    #YDIM = 200
    #ZDIM = 50
    d_origin = np.array([0., 0., 0.])
    d_spacing = np.array([1., 1., 1.])
    #d_extent = np.array([  0,  19,   0, 199,   0,  49])
    d_extent = np.array([  0,  XDIM-1,   0, YDIM-1,   0,  ZDIM-1])
    
    # TODO remove hardcode
    if (XDIM == 20 and YDIM == 200 and ZDIM == 50):
        # THIS IS HARD CODED FOR EXAAM DATA SET
        list_of_boundary_information = [[0, 0, 0, 0, 19, 19, 19, 19], [0, 0, 199, 199, 0, 0, 199, 199], [0, 49, 0, 49, 0, 49, 0, 49], 300.00067065486246, 300.0, 300.0, 300.0, 568.7094662404135, 300.0, 300.0, 300.0]
    elif(XDIM == 250 and YDIM == 250 and ZDIM == 50):
        # HARD CODED FOR ISABEL
        list_of_boundary_information = [[0, 0, 0, 0, 249, 249, 249, 249], [0, 0, 249, 249, 0, 0, 249, 249], [0, 49, 0, 49, 0, 49, 0, 49], 0.0, 20.057417, 2272.4014, 22.809683, 1063.6722, 6.953186, 1446.9822, 10.676723]
    elif(XDIM == 500 and YDIM == 500 and ZDIM == 100):
        # HARD CODED FOR ISABEL
        list_of_boundary_information = [[0, 0, 0, 0, 499, 499, 499, 499], [0, 0, 499, 499, 0, 0, 499, 499], [0, 99, 0, 99, 0, 99, 0, 99], 0.0, 20.057417, 2272.4014, 22.809683, 1063.6722, 6.953186, 1446.9822, 10.676723]
    elif(XDIM == 20 and YDIM == 200 and ZDIM == 50):
        # HARD CODED FOR ASTEROID
        list_of_boundary_information = [[0, 0, 0, 0, 299, 299, 299, 299], [0, 0, 299, 299, 0, 0, 299, 299], [0, 299, 0, 299, 0, 299, 0, 299], 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    else:
        print("No known boundary information for given input!\n")
        exit(0)


    print(list_of_boundary_information)

    # Create block manager
    bm_parameter_list = [XBLOCK, YBLOCK, ZBLOCK, XDIM, YDIM, ZDIM, d_origin, d_spacing, d_extent]
    bm = BlockManager('regular', bm_parameter_list)
    nob = (XDIM*YDIM*ZDIM)/(XBLOCK*YBLOCK*ZBLOCK)
    

    # TODO
    # if data flagged
    # LOAD REFERENCE DATA out of ref_fname
    if (prev_id == 'NA' or prev_data == 'NA'):
        # load sample lid and data into array
        list_sampled_lid, list_sampled_data = read_samples_from_binary_files(curr_id, curr_data, curr_total, bm_parameter_list)
        list_sampled_lid_ref, list_sampled_data_ref = read_samples_from_binary_files(curr_id, curr_data, curr_total, bm_parameter_list)
    else:
        list_sampled_lid, list_sampled_data = read_samples_from_binary_files(curr_id, curr_data, curr_total, bm_parameter_list)
        list_sampled_lid_ref, list_sampled_data_ref = read_samples_from_binary_files(prev_id, prev_data, prev_total, bm_parameter_list)

    

    # For each block ...
    for bid in range(0, int(nob)):
        
        # If flagged, set equal to previous time step's information
        if (np.array_equal(list_sampled_lid[bid], flag)):
            #print("FLAGGED: ")
            #print(list_sampled_lid[bid], list_sampled_lid_ref[bid])
            list_sampled_lid[bid] = list_sampled_lid_ref[bid]
            list_sampled_data[bid] = list_sampled_data_ref[bid]


            # NOTE TO NOT SEE T-1 SAMPLES
            #list_sampled_lid[bid] = []
            #list_sampled_data[bid] = []
             
       
    Points = vtk.vtkPoints()
    
    
    #print("len lid: ", len(list_sampled_lid))
    for b in range(len(list_sampled_lid)):
        
        for i in range(len(list_sampled_lid[b])):
            #try:
            t_fid,t_fx,t_fy,t_fz = bm.func_block_2_full(b,list_sampled_lid[b][i])
            #except:
            #print("{} FAILED: {}".format(b,list_sampled_lid[b]))
            #pass
            Points.InsertNextPoint(t_fx,t_fy,t_fz)

    for i in range(8):
        Points.InsertNextPoint(list_of_boundary_information[0][i], list_of_boundary_information[1][i], list_of_boundary_information[2][i])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)
    
    val_arr = vtk.vtkFloatArray()
    val_arr.SetNumberOfComponents(1)
    val_arr.SetName("samples")
        
    for b in range(len(list_sampled_data)):
        for i in range(len(list_sampled_data[b])):
            val_arr.InsertNextValue(list_sampled_data[b][i])
            
    for j in range(len(list_of_boundary_information[0])):
        val_arr.InsertNextValue(list_of_boundary_information[3+j])
        
    polydata.GetPointData().AddArray(val_arr)
    polydata.Modified()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(out_fname)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write()
    print("Created file ", out_fname)
    total_samples = polydata.GetNumberOfPoints()
    

    return 0