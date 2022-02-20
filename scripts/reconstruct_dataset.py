import numpy as np
from sklearn.neighbors import NearestNeighbors
import vtk
from scipy.interpolate import griddata
import sys
import os


#XDIM = 250
#YDIM = 250
#ZDIM = 50

samp_method = 'random'
cur_samp = 'linear' # linear, nearest, cubic


if len(sys.argv)==6:
    infile = sys.argv[1]
    XDIM = int(sys.argv[2])
    YDIM = int(sys.argv[3])
    ZDIM = int(sys.argv[4])
    output_file_path = sys.argv[5]
else:
    print("COULD NOT RECONSTRUCT DATA: "+str(infile)+ " Not enough arguements passed.")
    exit(0)

filename, file_extension = os.path.splitext(os.path.basename(infile))
print(filename)
if file_extension=='.vtp':
    poly_reader = vtk.vtkXMLPolyDataReader()
elif file_extension=='.vtk':
    poly_reader = vtk.vtkXMLGenericDataObjectReader()
elif file_extension=='.vtu':
    poly_reader = vtk.vtkXMLUnstructuredGridReader()
poly_reader.SetFileName(infile)
poly_reader.Update()


#poly_reader = vtk.vtkXMLPolyDataReader()
#poly_reader.SetFileName(infile)
#poly_reader.Update()

data = poly_reader.GetOutput()

var_name = data.GetPointData().GetArrayName(0)

print("total points:",data.GetNumberOfPoints(),data.GetNumberOfElements(0))

pts = data.GetPoints()

pt_data = data.GetPointData().GetArray(var_name).GetTuple1(100)
print('data:',pt_data)

print(pts.GetPoint(0))

tot_pts = data.GetNumberOfPoints()
feat_arr = np.zeros((tot_pts,3))

print('total points:',tot_pts)

data_vals = np.zeros(tot_pts)


for i in range(tot_pts):
    loc = pts.GetPoint(i)
    feat_arr[i,:] = np.asarray(loc)
    pt_data = data.GetPointData().GetArray(var_name).GetTuple1(i)
    data_vals[i] = pt_data

range_min = np.min(feat_arr,axis=0)
range_max = np.max(feat_arr,axis=0)

print("range:",range_min,range_max)

#n_neighbors = 5
#nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(feat_arr)


#grid_x, grid_y, grid_z = np.mgrid[0:XDIM:5j, 0:YDIM:5j, 0:ZDIM:1j]
# grid_x, grid_y, grid_z = np.mgrid[0:XDIM, 0:YDIM, 0:ZDIM]
#
#
# grid_z0 = griddata(feat_arr, data_vals, (grid_x, grid_y, grid_z), method='nearest')
# #grid_z1 = griddata(feat_arr, data_vals, (grid_x, grid_y, grid_z), method='linear')
#
# grid_z0.tofile('out_full.bin')

cur_loc = np.zeros((XDIM*YDIM*ZDIM,3),dtype='double')

ind = 0
for k in range(ZDIM):
    for j in range(YDIM):
        for i in range(XDIM):
            cur_loc[ind,:] = np.array([i,j,k])
            ind = ind+1

grid_z0 = griddata(feat_arr, data_vals, cur_loc, method=cur_samp)
#grid_z1 = griddata(feat_arr, data_vals, (grid_x, grid_y, grid_z), method='linear')
print("writing file:")
#grid_z0.tofile('recons_'+samp_method+'.raw')
grid_z0_3d = grid_z0.reshape((ZDIM,YDIM,XDIM))
# write to a vti file
#filename = 'recons_'+filename+'_'+cur_samp+'.vti'
#filename = 'out_files/reconstructed_files/'+filename

filename = output_file_path + '/recons_'+filename+'_'+cur_samp+'.vti'




imageData = vtk.vtkImageData()
imageData.SetDimensions(XDIM, YDIM, ZDIM)
if vtk.VTK_MAJOR_VERSION <= 5:
    imageData.SetNumberOfScalarComponents(1)
    imageData.SetScalarTypeToDouble()
else:
    imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

dims = imageData.GetDimensions()
print(dims)
# Fill every entry of the image data with "2.0"
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            imageData.SetScalarComponentFromDouble(x, y, z, 0, grid_z0_3d[z,y,x])

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(filename)
if vtk.VTK_MAJOR_VERSION <= 5:
    writer.SetInputConnection(imageData.GetProducerPort())
else:
    writer.SetInputData(imageData)
writer.Write()

print("Created: ", filename)

# interp_data = np.zeros([XDIM,YDIM,ZDIM])
#
# for k in range(ZDIM):
#     for j in range(YDIM):
#         for i in range(XDIM):
#             cur_loc = np.array([i,j,k])
#             distances, indices = nbrs.kneighbors(cur_loc.reshape(1,-1))
#             vals = data_vals[indices]/
#             interp_data[i,j,k] =
#             print(i,j,k)




