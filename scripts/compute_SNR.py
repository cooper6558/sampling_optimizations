import sys

import math 

#direct to the path where the class functions exists
sys.path.append('/Users/mlhfulp/data_sampling/reuse_sampler/src/intelligent_sampler')
import FeatureSampler as FS

import importlib
importlib.reload(FS)

# Import Reuse Sampler Functions
from ReuseSamplerFunctions import *

import shutil
import subprocess
from subprocess import call
import numpy
#import argparse
import re
from os import walk
import os

# ONCE DONE DELETE THE FILE TO SAVE ON SPACE WHEN RUNNING A LOT OF EXPERIMENTS AT ONCE
PARALLEL_DELETE_OPTION = 0

if __name__ == '__main__':
    
    # Get the total number of args passed to the demo.py
    num_args = len(sys.argv)

    if (num_args == 3):
        orig_file = sys.argv[1]
        inpt_file = sys.argv[2]

        print("Orig Data Loc: " + str(orig_file))
        print("Inpt Data Loc: " + str(inpt_file))

        # Read in orig values
        orig = FS.DataManager(orig_file, 0)
        dims = orig.get_dimension()
        print("Working on dimensions ", dims[0],dims[1],dims[2])    


        x_start = 0
        x_stop = (dims[0])
        y_start = 0
        y_stop = (dims[1])
        z_start = 0
        z_stop = (dims[2])


    elif(num_args == 9): 
        orig_file = sys.argv[1]
        inpt_file = sys.argv[2]
        x_start = int(sys.argv[3])
        x_stop = int(sys.argv[4])
        y_start = int(sys.argv[5])
        y_stop = int(sys.argv[6])
        z_start = int(sys.argv[7])
        z_stop = int(sys.argv[8])

        # Read in orig values
        orig = FS.DataManager(orig_file, 0)
        dims = orig.get_dimension()
        print("Working on dimensions ", dims[0],dims[1],dims[2])    


        print("Orig Data Loc: " + str(orig_file))
        print("Inpt Data Loc: " + str(inpt_file))
        print("ROI Data Range: ", x_start, x_stop, y_start, y_stop, z_start, z_stop)
    else: 
        print("INVALID NUMBER OF ARGS PASSED:\n     python compute_SNR.py orig_file inpt_file\n     python compute_SNR.py orig_file inpt_file x_start x_stop y_start y_stop z_start z_stop")
        exit(0)

    print("ENSURE FOLDERS ({}, {}) CONTAIN SAME DATA DIMS".format(orig_file, inpt_file))

    try:
        orig_data, bm = extractData(orig, dims[0], dims[1], dims[2])
    except:
        print("Error Opening File: {}".format(orig_file))
        exit(0)

    try:
        # Read in input values
        inpt = FS.DataManager(inpt_file, 0)
        inpt_data, bm = extractData(inpt, dims[0], dims[1], dims[2])
    except:
        print("Error Opening File: {}".format(inpt_file))
        exit(0)


    data_size = (x_stop - x_start)*(y_stop - y_start)*(z_stop - z_start)

    # Calculate SNR
    mean_raw=0
    stdev_raw=0
    mean_sampled=0
    stdev_sampled=0
    mean_error=0
    stdev_error=0
    
    max_abs_diff = 0
    average_diff = 0

    # Get max_abs_diff and avg_diff
    for kk in range(z_start, z_stop):
        for jj in range(y_start, y_stop):
            for ii in range(x_start, x_stop):
                orig_pixel = orig_data[kk][jj][ii]
                inpt_pixel = inpt_data[kk][jj][ii]

                if (not math.isnan(orig_pixel) and not math.isnan(inpt_pixel)):
                    if (kk == 0 and jj == 0 and ii == 0):
                        global_raw_min = orig_pixel
                        global_raw_max = orig_pixel
                        global_sampled_min = inpt_pixel
                        global_sampled_max = inpt_pixel

                    if global_raw_min > orig_pixel: global_raw_min = orig_pixel
                    if global_raw_max < orig_pixel: global_raw_max = orig_pixel
                    if global_sampled_min > inpt_pixel: global_sampled_min = inpt_pixel
                    if global_sampled_max < inpt_pixel: global_sampled_max = inpt_pixel
                    
                    average_diff += math.fabs(orig_pixel - inpt_pixel)
                    if (max_abs_diff < math.fabs(orig_pixel - inpt_pixel)):
                        max_abs_diff = math.fabs(orig_pixel - inpt_pixel)
    average_diff = average_diff/data_size

    for kk in range(z_start, z_stop):
        for jj in range(y_start, y_stop):
            for ii in range(x_start, x_stop):
                orig_pixel = orig_data[kk][jj][ii]
                inpt_pixel = inpt_data[kk][jj][ii]

                if (not math.isnan(orig_pixel) and not math.isnan(inpt_pixel)):
                    mean_raw += orig_pixel
                    mean_sampled += inpt_pixel
                    mean_error += math.fabs(orig_pixel - inpt_pixel)

    mean_raw /= data_size
    mean_sampled /= data_size
    mean_error /= data_size

    for kk in range(z_start, z_stop):
        for jj in range(y_start, y_stop):
            for ii in range(x_start, x_stop):
                orig_pixel = orig_data[kk][jj][ii]
                inpt_pixel = inpt_data[kk][jj][ii]

                if (not math.isnan(orig_pixel) and not math.isnan(inpt_pixel)):
                    stdev_sampled += (inpt_pixel-mean_sampled)*(inpt_pixel-mean_sampled)
                    stdev_raw += (orig_pixel-mean_raw)*(orig_pixel-mean_raw)
                    stdev_error += (math.fabs((orig_pixel)-(inpt_pixel))-mean_error)*(math.fabs((orig_pixel)-(inpt_pixel))-mean_error)
                

    stdev_raw = math.sqrt(stdev_raw/(data_size))
    stdev_sampled = math.sqrt(stdev_sampled/(data_size))
    stdev_error = math.sqrt(stdev_error/(data_size))

    stats = np.zeros(5)
    stats[0] = mean_raw
    stats[1] = stdev_raw
    stats[2] = mean_sampled
    stats[3] = stdev_sampled
    stats[4] = 20*math.log10(stdev_raw/stdev_error)

    print("/************************************/")
    print("global raw range: [{}, {}]".format(global_raw_min, global_raw_max))
    print("global sampled range: [{}, {}]".format(global_sampled_min, global_sampled_max))
    print("Max Abs Diff: ", max_abs_diff, " Average Diff: ", average_diff)
    print("global raw mean: ", stats[0], " global raw stdev: ", stats[1])
    print("global sampled mean: ", stats[2], " global sampled stdev: ", stats[3])
    print("global signal to noise: ", stats[4])

    # ONCE DONE DELETE THE FILE TO SAVE ON SPACE WHEN RUNNING A LOT OF EXPERIMENTS AT ONCE
    if (PARALLEL_DELETE_OPTION == 1):
        if os.path.exists(inpt_file):
            os.remove(inpt_file)
