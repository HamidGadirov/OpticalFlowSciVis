import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
from progress.bar import Bar
import pickle
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import block_reduce
import random
import pyimof
import yt
from mpl_toolkits.axes_grid1 import ImageGrid
import re
import h5py

def visualize_series(data_to_vis, dir_res="Results", title="Data", show=True, save=False):
    fig = plt.figure()
    columns = 10
    rows = 10

    print("data_to_vis is in range %f to %f" % (np.min(data_to_vis), np.max(data_to_vis)))

    j = 0
    for i in range(1, columns*rows+1):
        index = i

        if (index >= data_to_vis.shape[0]):
            break

        img = data_to_vis[index,...]
        ax = fig.add_subplot(rows, columns, i)

        plt.axis('off')
        # print("range:", data_to_vis.min(), data_to_vis.max())
        # input("x")
        # plt.imshow(img, cmap='viridis', vmin=data_to_vis.min(), vmax=data_to_vis.max())
        plt.imshow(img, cmap='viridis')
        # plt.imshow(img, cmap='viridis', vmin=0, vmax=255)

    fig = plt.gcf()
    plt.suptitle(title) 
    fig.set_size_inches(12, 9) # 30, 9 ?
    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

def visualize_flow(flow_u_gt, flow_v_gt, dir_res="Results", title="Flow", show=True, save=False):
    fig = plt.figure()
    columns = 10
    rows = 10

    max_flow_gt = np.max(np.sqrt(flow_u_gt**2 + flow_v_gt**2))
    quiver_steps = 5
    scale = 0.1 # 1

    print("Flow range: %f to %f" % (np.min(flow_u_gt), np.max(flow_u_gt)))

    j = 0
    for i in range(1, columns * rows + 1):
        index = i

        if (index >= flow_u_gt.shape[0]):
            break

        u_gt = flow_u_gt[index, ...]
        v_gt = flow_v_gt[index, ...]
        norm_gt = np.sqrt(u_gt**2 + v_gt**2)

        ax = fig.add_subplot(rows, columns, i)
        plt.axis('off')

        # Using quiver plot to visualize flow
        pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=None, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale=scale, bg_cmap='gray', step=quiver_steps)

    fig = plt.gcf()
    plt.suptitle(title)
    fig.set_size_inches(12, 9)  # 30, 9 ?
    
    if show:
        plt.show()
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi=300)

def getListOfFiles(dirName):
    # For the given path, get the List of all files in the directory tree 

    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        if entry.endswith(('.npy')):
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
        # fullPath = os.path.join(dirName, entry)
        # if os.path.isdir(fullPath):
        #     allFiles = allFiles + getListOfFiles(fullPath)
        # else:
        #     allFiles.append(fullPath)
                
    return allFiles  

def extract_params(dirName):
    match = re.search(r'Simulation_OmM_(\d+)_OmB_(\d+)_h_(\d+)', dirName)
    if match:
        OmM = float(match.group(1)) / 10000  # Converting from int to float
        OmB = float(match.group(2)) / 10000  # Converting from int to float
        h = float(match.group(3)) / 100      # Converting from int to float
        return OmM, OmB, h
    else:
        raise ValueError("Directory name does not match the expected format")

# Simulation_OmM_012_OmB_00215_h_07_npy 794
# Simulation_OmM_012_OmB_00235_h_07_npy 1099
# Simulation_OmM_0155_OmB_00235_h_055_npy 1600
# Simulation_OmM_0155_OmB_00235_h_07_npy 780
# Simulation_OmM_0155_OmB_00215_h_07_npy 777
# Simulation_OmM_012_OmB_00235_h_055_npy 2400

# Simulation_OmM_012_OmB_00215_h_055_npy 1600
# Simulation_OmM_01375_OmB_00215_h_055_npy 1500
# Simulation_OmM_01375_OmB_00225_h_07_npy 950
# Simulation_OmM_01375_OmB_00235_h_055_npy 2000
# Simulation_OmM_0155_OmB_00215_h_055_npy 1600
# Simulation_OmM_0155_OmB_00225_h_055_npy 1600

# train
dirNames = ['./Simulation_OmM_0155_OmB_00235_h_055_npy', './Simulation_OmM_012_OmB_00235_h_055_npy',
            './Simulation_OmM_012_OmB_00215_h_055_npy', # new
            './Simulation_OmM_0155_OmB_00215_h_055_npy', './Simulation_OmM_0155_OmB_00225_h_055_npy'] # new

dirNames = ['./Simulation_OmM_012_OmB_00215_h_055_npy', './Simulation_OmM_0155_OmB_00215_h_055_npy'] # train
dirNames = ['./Simulation_OmM_0155_OmB_00225_h_055_npy'] # for ends
dirNames = ['./Simulation_OmM_01375_OmB_00225_h_055_npy'] # for ends

# val 
# dirNames = ['./Simulation_OmM_0155_OmB_00215_h_07_npy']
# dirNames = ['./Simulation_OmM_01375_OmB_00215_h_055_npy'] # new
# dirNames = ['./Simulation_OmM_012_OmB_00215_h_055_npy'] # train val
# dirNames = ['./Simulation_OmM_0155_OmB_00215_h_055_npy'] # train val 1

# comoving_OmM = 0.1375
# comoving_OmB = 0.0215
# comoving_h   = 0.55
# './Simulation_OmM_012_OmB_00235_h_07_npy',

# test
# dirNames = ['./Simulation_OmM_0155_OmB_00235_h_07_npy']
# dirNames = ['./Simulation_OmM_01375_OmB_00235_h_055_npy'] # new
# dirNames = ['./Simulation_OmM_012_OmB_00215_h_055_npy'] # ends
# dirNames = ['./Simulation_OmM_0155_OmB_00235_h_055_npy'] # ends
# print("test")

# val
# dirNames = ['./organized_output_1e4']
# dirNames = ['
# 
# ./organized_output_1e2_128', './organized_output_1e2_128_1', './organized_output_1e2_128_2']
# test
# dirNames = ['./organized_output_1e5']
# dirNames = ['./organized_output_1e4_128']
# dirName = '5Jets_density_uchar' # uint8

# data_ensemble = []
# data_ensemble = np.zeros((0, 64, 64, 64, 4), dtype=np.float32) # (2000, 4, 3, 64, 64, 64)
data_ensemble = np.zeros((0, 128, 128, 128, 4), dtype=np.float32) # (2000, 4, 3, 128, 128, 128)
# data_ensemble = np.zeros((0, 128, 128, 128, 5), dtype=np.float32) # (2000, 4, 3, 128, 128, 128)
# tuple: data and meta info
params_ensemble = np.zeros((0, 3), dtype=np.float32)

for dirName in dirNames:
    # for dirName in dirNames:
    print(dirName)
    # input("x")
    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(dirName)
    # listOfFiles.sort()
    listOfFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print("listOfFiles is sorted!")

    # for k in range(10):
    #     print(listOfFiles[k])
        # print(len(listOfFiles))
    # input("files")
    print(len(listOfFiles))

    data = []

    data_dens = []
    count = 0
    # dims = 128 * 128 * 128

    # Print the files
    with Bar("Loading the data...", max=len(listOfFiles) ) as bar:
        for elem in listOfFiles:
            if 'den' in elem:
                if count % 5 == 0: # downloading every 5th time step # 10th
                # print(elem)
                    if count >= 1100: # 1100 700
                        break
                    if count >= 300:
                        tmp_data = np.load(elem)

                        tmp_data.resize(128, 128, 128, 1) # density
                        # print(tmp_data.shape)
                        # tmp_data = np.flip(tmp_data, 1)
                        # input("x")
                        data_dens.append(tmp_data)
                        # data = np.concatenate((data, tmp_data), axis=0)

                count += 1
            print(count)

            bar.next()

    print(len(data_dens))
    print(np.array(data_dens).shape)
    # input("density")

    # data_to_vis = np.array(data_dens)[:, 32, :, :, 0] # lenght, height, width, 
    # visualize_series(data_to_vis, dir_res="Results", title="pool_2D_dens", show=False, save=True)
    # input("viz")

    # data = np.expand_dims(data, axis=1)
    # print(np.array(data).shape)
    # input("data")
    print("adding density")
    if not data:
        data = data_dens
    else:
        data = np.concatenate((data, data_dens), axis=-1)
    print(len(data))
    print(np.array(data).shape)

    # dirName = 'Velocity'

    data_vel = []
    count = 0
    # dims = 128 * 128 * 128

    # Print the files
    with Bar("Loading the data...", max=len(listOfFiles) ) as bar:
        for elem in listOfFiles:
            if 'vel' in elem:
                if count % 5 == 0: # downloading every 5th time step # 10th
                    # print(elem)
                    if count >= 1100: # 1100 700
                        break
                    if count >= 300:
                        tmp_data = np.load(elem)
                        
                        tmp_data.resize(128, 128, 128, 3) # vel
                        # print(tmp_data.shape)
                        # tmp_data = np.flip(tmp_data, 1)
                        # input("x")
                        data_vel.append(tmp_data)

                count += 1
            bar.next()

    print(len(data_vel))
    print(np.array(data_vel).shape)
    # input("data_vel")

    # data_to_vis = np.array(data_vel)[:, 32, :, :, 0] # lenght, height, width, 
    # visualize_series(data_to_vis, dir_res="Results", title="pool_2D_vel", show=False, save=True)
    # input("viz")

    print("combining density and vel")
    data = np.concatenate((data, data_vel), axis=-1)
    print(len(data))
    print(np.array(data).shape)

    ### params
    # Extract simulation parameters
    OmM, OmB, h = extract_params(dirName)
    num_samples = data.shape[0]
    # Create an array to store the parameters for each sample
    params = np.zeros((num_samples, 3), dtype=np.float32)
    params[:, 0] = OmM
    params[:, 1] = OmB
    params[:, 2] = h
    ### params

    # Combine data and parameters into a tuple
    # data_with_params = (data, params)
    # print("Data shape:", data_with_params[0].shape)
    # print("Parameters shape:", data_with_params[1].shape)
    data_ensemble = np.concatenate((data_ensemble, data), axis=0)
    params_ensemble = np.concatenate((params_ensemble, params), axis=0)

    # data_ensemble = np.concatenate((data_ensemble, data_with_params), axis=0)
    print(np.array(data_ensemble).shape)

print(np.array(data_ensemble).shape)
# input("data_ensemble")

data = data_ensemble
del data_ensemble

# data = data[300:] # where something interesting happens
# data = data[300:450] # test data
print("data:", data.shape)

# pkl_filename = "5jets3d" + "_dens" + ".pkl" 
# pkl_filename = "5jets3d" + "_flow" + ".pkl" 
# pkl_filename = "5jets3d" + "_flow" + "_aug" + ".pkl" 

# print("data:", data.shape)
# data = data[:, np.newaxis, ...]
# velocities_x = velocities_x[:, np.newaxis, ...]
# velocities_y = velocities_y[:, np.newaxis, ...]
# data = np.hstack((data, velocities_x))
# data = np.hstack((data, velocities_y))
# print(data.shape)
# input("data")

data = np.float32(data)
# data = np.float16(data)

# print(data.shape)
data = np.transpose(data, (0, 4, 1, 2, 3))
# input(data.shape)

# print(data.shape)
# data_to_vis = data[:, 0, :, :, 64]
# # data_to_vis = data[:, 0, 64, :, :]
# print(data_to_vis.shape)
# visualize_series(data_to_vis, dir_res="Results", title="5Jets_Densities_2D", show=False, save=True)
# # input("viz")

# flow_u_gt = data[300:, 1, :, :, 64]
# flow_v_gt = data[300:, 2, :, :, 64]

# visualize_flow(flow_u_gt, flow_v_gt, dir_res="Results", title="5Jets_Flow_300_yz_2D", show=False, save=True)
# visualize_flow(flow_u_gt, flow_v_gt, dir_res="Results", title="5Jets_Flow_300_xy_2D", show=False, save=True)
# input("viz")

# TODO log norm
# Normalize/scale
print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
print("Vel z range %f to %f" % (np.min(data[:, 3]), np.max(data[:, 3])))
# print("Data is in range %f to %f" % (np.min(data), np.max(data)))
# input(data.shape)
# input("Data range")
# for i in range(4):
data[:, 0] = np.log10(data[:, 0])
print("Density scaled logarithmically!")
data[:, 0] = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
# data = (data - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
# # data[:, 0] = cv2.normalize(data[:, 0], data[:, 0], 0., 1., cv2.NORM_MINMAX)
print("Density scaled to [0, 1] range!")
# # data[:, 1] = cv2.normalize(data[:, 1], data[:, 1], 0., 1., cv2.NORM_MINMAX)
# data = (data - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))
# data[:, 2] = cv2.normalize(data[:, 2], data[:, 2], 0., 1., cv2.NORM_MINMAX)
# data[:, 3] = cv2.normalize(data[:, 3], data[:, 3], 0., 1., cv2.NORM_MINMAX)
for i in range(1, 4):
    data[:, i] = (2 * (data[:, i] - np.min(data[:, i])) / (np.max(data[:, i]) - np.min(data[:, i]))) - 1
# print("Velocities scaled to [0, 1] range!")
print("Velocities scaled to [-1, 1] range!")
print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
print("Vel z range %f to %f" % (np.min(data[:, 3]), np.max(data[:, 3])))

# flow_u_gt = data[300:, 1, :, :, 64]
# flow_v_gt = data[300:, 2, :, :, 64]
# flow_u_gt = data[:, 1, 40, :, :]
# flow_v_gt = data[:, 2, 40, :, :]

# visualize_flow(flow_u_gt, flow_v_gt, dir_res="Results", title="5Jets_FlowScaled_300_yz_2D", show=False, save=True)
# visualize_flow(flow_u_gt, flow_v_gt, dir_res="Results", title="Nyx_FlowScaled_xy_2D", show=False, save=True)
# input("viz")

# Density range 17610006.000000 to 54730323656704.000000
# Vel x range -1902.737793 to 1960.986816
# Vel y range -1920.188721 to 2002.423950
# Vel z range -1890.228638 to 2030.002319

print(data.shape)
test_data_with_params = (data, params_ensemble)
# data = data[200:350]
data = np.array(test_data_with_params[0])
params = np.array(test_data_with_params[1])
print("data shape:", data.shape)
print("params shape:", params.shape)
# input("x")
# filename = "nyx3d_skip5_128_test.pkl"
# filename = "nyx3d_skip5_128_log10norm_test_1.pkl"
# filename = "nyx3d_skip5_128_log10norm_test_params.pkl"
filename = "nyx3d_skip5_128_log10norm_OmM_012_OmB_00215_h_055_test_params.pkl"
# filename = "nyx3d_skip5_128_log10norm_OmM_0155_OmB_00235_h_055_test_params.pkl"
# hdf5_file = 'nyx3d_128_150_test_2.h5'
# with h5py.File(hdf5_file, 'w') as hf:
#     # hf.create_dataset('data', data=data)
#     hf.create_dataset('data', data=data, compression='gzip', compression_opts=9, chunks=True)
# np.save(filename, data.astype(np.float32))
with open(filename, 'wb') as f:
    # pickle.dump(data, f, protocol=4)
    pickle.dump(test_data_with_params, f, protocol=4)
print('Test data saved!')
input("test")
exit()

data_with_params = (data, params_ensemble)
# data = np.array(data_with_params[0])
# params = np.array(data_with_params[1])
# print("data shape:", data.shape)
# print("params shape:", params.shape)
# input("params")

num_members = len(dirNames)
member_size = 160 # data.shape[0] # 160
num_triplets = 200 # 50 200 500
window = 12 # 8 12

# Initialize a tuple to hold the triplet data and parameters
triplet_data = (np.zeros((num_triplets, 4, *data.shape[1:]), dtype=np.float32),
                np.zeros((num_triplets, 3), dtype=np.float32))  # Assuming 3 parameters

print("Triplet data shape (data):", triplet_data[0].shape)
print("Triplet data shape (params):", triplet_data[1].shape)
# size, triplets + t, dens + vel, x, y, z
# size, triplets + t + params, dens + vel, x, y, z - for Hypernetworks
# print("triplet_data:", triplet_data.shape)
# print("triplet_data type:", triplet_data.dtype)
# input("data")

triplet_idx = 0
with Bar("Loading the data...", max=num_triplets) as bar:
    while triplet_idx < num_triplets:

        # Randomly select triplets
        for i in range(num_triplets):
            print("\n")

            member_idx = np.random.randint(num_members) + 1

            # Randomly select k
            k = np.random.randint(0, member_size - 1 - window)
            k = k * member_idx # rand select member
            # print("k", k)
            # Calculate the valid range for l
            max_l = min(member_size * member_idx - 1, k + window)
            # print("max_l", max_l)
            # Randomly select l within the valid range
            l = np.random.randint(k + 1 + 1, max_l + 1)
            # print("l", l)
            # Randomly select t between k and l
            t = np.random.randint(k + 1, l)
            # print("t", t)
            
            # k = t - window
            # l = t + window

            # k, t, l = k * member_idx, t * member_idx, l * member_idx
            print(" ", k, t, l)

            triplet_data[0][triplet_idx, 0] = data[k]
            triplet_data[0][triplet_idx, 1] = data[t]
            triplet_data[0][triplet_idx, 2] = data[l]

            # Calculate t value and create mask
            t_value = (t - k) / (l - k)
            mask = np.zeros((1, *data.shape[1:]))
            mask[0] = 1
            triplet_data[0][triplet_idx, 3] = mask * t_value
            # print("t_mask:", triplet_data[triplet_idx, 3])

            triplet_data[1][triplet_idx] = params_ensemble[t]

            triplet_idx += 1
            bar.next()


print('Triplet data shape:', triplet_data[0].shape)
print('Parameters data shape:', triplet_data[1].shape)

# Save the tuple to a file
# filename = "nyx3d_ens5m_skip5_128_log10norm_triplets_params_W12_200_train.pkl"
# filename = "nyx3d_ens5m_skip5_128_log10norm_triplets_params_W12_500_train.pkl"
# filename = "nyx3d_skip5_128_log10norm_triplets_params_W12_50_val.pkl"
# filename = "nyx3d_skip5_128_log10norm_triplets_params_W12_200_train_OmM_0155_OmB_00215_h_055.pkl"
filename = "nyx3d_skip5_128_log10norm_triplets_params_W12_200_train_OmM_01375_OmB_00225_h_055.pkl"
# filename = "nyx3d_skip5_128_log10norm_triplets_params_W12_200_train_end2m.pkl"
with open(filename, 'wb') as f:
    pickle.dump(triplet_data, f, protocol=4)


print(f"Triplet data and parameters saved to {filename}")

# print('Triplet data shape:', triplet_data.shape)
# print(triplet_data.dtype)
# input("triplet_data")
# filename = "nyx3d_skip5_128_triplets_W12_100_val.pkl"
# filename = "nyx3d_skip5_128_log10norm_triplets_W12_50_val.pkl"
# filename = "nyx3d_skip5_128_log10norm_triplets_W12_50_val_1.pkl"
# filename = "nyx3d_skip5_128_log10norm_triplets_W12_50_val_params.pkl"
print('Saved!')
input("data_train")

# print(data.shape)
data_to_vis = data[:, 0, :, :, 64]
# data_to_vis = data[:, 0, 64, :, :]
# print(data_to_vis.shape)
# # viz
# visualize_series(data_to_vis, dir_res="Results", title="5Jets_Densities_2D", show=False, save=True)
# input("viz")

data_to_vis = data[:, 0]
print(data_to_vis.shape)

"""
# Choose some volumes to visualize
volumes_to_visualize = [0, 10, 20, 30]

# Threshold value for creating boolean array
threshold = 0.75

# Plot each selected volume
fig = plt.figure()
for i, vol_idx in enumerate(volumes_to_visualize):
    volume = data_to_vis[vol_idx, :, :, :]
    volume = block_reduce(volume, block_size=(2, 2, 2), func=np.mean)
    print(volume.shape)
    voxels = volume > threshold  # Convert to boolean array
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    print("yes")
    ax.voxels(voxels, edgecolor='k')
    ax.set_title(f"Volume {vol_idx}")
    print("yes")
plt.show()
dir_res = "Results"
title = "5Jets_Densities_3D"
fig.savefig(os.path.join(dir_res, title), dpi = 300)
# input("viz")
"""

# input("32")
# data = np.float16(data)
# input("16")
data = np.expand_dims(data, axis=1)

# TODO: randomly select
data_train = data[:300] #
# data_val = data[300:372] #
# data_test = data[300:372] #

input("data_train")

# Randomly select 300 indices for the training set
train_indices = random.sample(range(len(data)), 300)
data_train = [data[i] for i in train_indices]

# Remove the selected indices to avoid duplicates in the validation and test sets
remaining_indices = set(range(len(data))) - set(train_indices)

# Randomly select 36 indices for the validation set
val_indices = random.sample(remaining_indices, 36)
data_val = [data[i] for i in val_indices]

# Use the remaining indices for the test set
test_indices = remaining_indices - set(val_indices)
data_test = [data[i] for i in test_indices]

print("random subsets created!")

data_train = np.array(data_train)
data_val = np.array(data_val)
data_test = np.array(data_test)

del data, data_vel, data_to_vis
"""
print("before augmented:", data_train.shape)
data_train = np.append(data_train, data_train[:,:,:,::-1,:], axis=0)
# data_train = np.append(data_train, data_train[:,:,::-1,:,:], axis=0)
# TODO: try add aug with more RAM
print("augmented:", data_train.shape)
# input("x")
# data_train = np.float16(data_train)
"""

data_train_three = []
data_val_three = []
# data_train = np.array(data_train)
for i in range(0, data_train.shape[0]-2, 2): # TODO: try shift only by 1 with more RAM
    data_train_three.append(np.concatenate((data_train[i], data_train[i+2], data_train[i+1]), axis=0)) # img0, img1, gt
data_train = np.array(data_train_three)
print("data_train in three:", data_train.shape)
# input("x")
# data_train = np.array(data_train)
for i in range(0, data_val.shape[0]-2, 2): 
    data_val_three.append(np.concatenate((data_val[i], data_val[i+2], data_val[i+1]), axis=0)) # img0, img1, gt
data_val = np.array(data_val_three)
print("data_val in three:", data_val.shape)
# input("x")
data_test_three = []
# data_train = np.array(data_train)
for i in range(0, data_test.shape[0]-2, 2): 
    data_test_three.append(np.concatenate((data_test[i], data_test[i+2], data_test[i+1]), axis=0)) # img0, img1, gt
data_test = np.array(data_test_three)
print("data_test in three:", data_test.shape)

# input("x")
# input("x")
# input("x")

pkl_train_filename = "5jets3d" + "_fp32" + "_flow" + "_aug" + "_train" + ".pkl" 
pkl_file = open(pkl_train_filename, 'wb')
pickle.dump(data_train_three, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")

pkl_val_filename = "5jets3d" + "_fp32" + "_flow" + "_aug" + "_val" + ".pkl" 
pkl_file = open(pkl_val_filename, 'wb')
pickle.dump(data_val_three, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")

pkl_test_filename = "5jets3d" + "_fp32" + "_flow" + "_aug" + "_test" + ".pkl" 
pkl_file = open(pkl_test_filename, 'wb')
pickle.dump(data_test_three, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")

# pkl_file = open(pkl_train_filename, 'rb')
# data = []
# data = pickle.load(pkl_file)
# pkl_file.close
# print(data.shape)
# input("Data")
input("Done.")


data = np.fromfile('Velocity/ns_1000_v.dat', dtype=np.float32)
# data = np.fromfile('Density/ns_1010_r.dat', dtype=float)
print(data)
print(len(data))

data = np.reshape(data,(128, 128, 128, 3))
print(data.shape)
input("data")

slice_num = 5 # 100
vx = data[:,:,slice_num,0]
vy = data[:,:,slice_num,1]

print(vx)
print(vy)

print(vx.shape)

import cv2

# Use Hue, Saturation, Value colour model 
hsv = np.zeros((128,128,3), dtype=np.uint8)
# print(img1.shape)
# input("x")
hsv[..., 1] = 255

# print(flow_mat[..., 0])
# flow_mat = flow_mat.astype('float32') 

mag, ang = cv2.cartToPolar(vx, vy)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

# print(hsv)

# print(hsv[..., 0])
# hsv = hsv.astype(np.uint8)
# print(cv2.__version__)
# input("x")

bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# print(bgr)
# cv2.imshow("colored flow", bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.gca()
plt.imshow(bgr)
plt.show()
# res_path = os.path.join('result', 'result1.png')
fig.savefig('gt_1000_5')

# def is_float(string):
#     """ True if given string is float else False"""
#     try:
#         return float(string)
#     except ValueError:
#         return False

# data = []
# with open('Velocity/ns_1000_v.dat', 'r') as f:
#     d = f.readlines()
#     for i in d:
#         k = i.rstrip().split(",")
#         data.append([float(i) if is_float(i) else i for i in k])

# data = np.array(data, dtype='O')
# print(data)
