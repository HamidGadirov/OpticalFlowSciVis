# from turtle import shape
# from cv2 import rectangle
from matplotlib.pyplot import axes, title
import matplotlib.pyplot as plt
import os
# import cv2
import math
import time
# import torch
# import torch.distributed as dist
import numpy as np
import pickle
import random
import argparse

# from utils import visualize_ind, visualize_series, visualize_series_flow, visualize_large
def visualize_series(data_to_vis, dir_res="Results", title="Data", show=True, save=False):
    fig=plt.figure()
    columns = 5
    rows = 8
    for i in range(1, columns*rows+1 ):
        if (i == data_to_vis.shape[0]):
            break
        img = data_to_vis[i]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img, cmap='viridis')
        
    fig = plt.gcf()
    plt.suptitle(title) 
    fig.set_size_inches(12, 9)
    if show:
        plt.show()  
    if save:
        title += ".pdf"
        fig.savefig(os.path.join(dir_res, title), dpi = 200)

# create 2d and 3d dataset
# box at random position, moving with random speed in random directions

resolution = 64
grid_x = resolution # 64
grid_y = resolution # 96
grid_z = resolution # 128
grit_t = 1000 # 3000
grid = np.zeros((grit_t, grid_x, grid_y, grid_z), dtype=int)
# grid = np.ones((grit_t, grid_x, grid_y, grid_z), dtype=int)
velocities_x = np.zeros((grit_t, grid_x, grid_y, grid_z), dtype=np.float32)
velocities_y = np.zeros((grit_t, grid_x, grid_y, grid_z), dtype=np.float32)
velocities_z = np.zeros((grit_t, grid_x, grid_y, grid_z), dtype=np.float32)

box_dim_x = 30
box_dim_y = 40
box_dim_z = 40
# box = np.ones((box_dim_x, box_dim_y, box_dim_z), dtype=int) * 255
box = np.ones((box_dim_x, box_dim_y, box_dim_z), dtype=np.float32)
# box = np.zeros((box_dim_x, box_dim_y, box_dim_z), dtype=int)

# add high freq texture to the box
# NxN areas with diff values
box *= 255.
step_size = 16
for i in range(0, box_dim_x, step_size):
    for j in range(0, box_dim_y, step_size):
        rand_value = np.random.randint(50, 256) # 20 30
        # print(rand_value)
        box[i:i+step_size, j:j+step_size] = rand_value
# box /= 255. # to 0...1
# add boundary to check if that forces to learn sharp edges
bound_value = 255
bound_width = 4
box[0:bound_width, :] = bound_value
box[-bound_width:, :] = bound_value

box[:, 0:bound_width] = bound_value
box[:, -bound_width:] = bound_value
box /= 255. # to 0...1
# print(box)
# input("x")

vel_min = -5 # -8 -6
vel_max = 5 # 8 5

pos_x = random.randint(0, grid_x - box_dim_x)
pos_y = random.randint(0, grid_y - box_dim_y)
pos_z = random.randint(0, grid_z - box_dim_z)
# grid[0, pos_x:pos_x+box_dim_x, pos_y:pos_y+box_dim_y] = box

if pos_x < 0: 
    pos_x = 0
if pos_y < 0: 
    pos_y = 0
if pos_z < 0: 
    pos_z = 0
if pos_x > grid_x - box_dim_x:
    pos_x = grid_x - box_dim_x
if pos_y > grid_y - box_dim_y:
    pos_y = grid_y - box_dim_y
if pos_z > grid_z - box_dim_z:
    pos_z = grid_z - box_dim_z
# print("pos_x, pos_y:", pos_x, pos_y)

begin_x = pos_x
begin_y = pos_y
begin_z = pos_z
# print("begin_x, begin_y:", begin_x, begin_y)

end_x = pos_x + box_dim_x
end_y = pos_y + box_dim_y
end_z = pos_z + box_dim_z
# print("end_x, end_y:", end_x, end_y)

vel_x = random.randint(vel_min, vel_max)
vel_y = random.randint(vel_min, vel_max)
vel_z = random.randint(vel_min, vel_max)

max_seq = 15
seq = max_seq
for i in range(grit_t):
    if seq == 0: # till new direction
        vel_x = random.randint(vel_min, vel_max)
        vel_y = random.randint(vel_min, vel_max)
        vel_z = random.randint(vel_min, vel_max)
        seq = max_seq

    pos_x += vel_x
    pos_y += vel_y
    pos_z += vel_z

    if pos_x < 0: 
        pos_x = 0
    if pos_y < 0: 
        pos_y = 0
    if pos_z < 0: 
        pos_z = 0

    if pos_x > grid_x - box_dim_x:
        pos_x = grid_x - box_dim_x
    if pos_y > grid_y - box_dim_y:
        pos_y = grid_y - box_dim_y
    if pos_z > grid_z - box_dim_z:
        pos_z = grid_z - box_dim_z

    begin_x = pos_x
    begin_y = pos_y
    begin_z = pos_z

    end_x = pos_x + box_dim_x
    end_y = pos_y + box_dim_y
    end_z = pos_z + box_dim_z

    # print(i, pos_x, pos_y, pos_z)
    # print(box.shape)

    grid[i, begin_x:end_x, begin_y:end_y, begin_z:end_z] = box
    # print(np.mean(grid[i]))

    # save vel vectors
    velocities_x[i, begin_x:end_x, begin_y:end_y, begin_z:end_z] = vel_x
    velocities_y[i, begin_x:end_x, begin_y:end_y, begin_z:end_z] = vel_y
    velocities_z[i, begin_x:end_x, begin_y:end_y, begin_z:end_z] = vel_z

    seq -= 1
    if pos_x == 0 or pos_y == 0 or pos_z == 0 or pos_x == grid_x - box_dim_x or pos_y == grid_y - box_dim_y or pos_z == grid_z - box_dim_z:
        seq = 0

print(grit_t, "timesteps created")
print(grid.shape)

data = grid

dir_res = "Results"
dataset = "rectangle3d"
# dir_res = os.path.join(dir_res, dataset)
print("Saving at:", dir_res)
# title = "rectange2d"
visualize_series(data[500:,...,50], dir_res, title=dataset, show=True, save=True)
# print(np.mean(data[:,:,5,:]))
# input("x")

# import csv  
# with open('rectangle3d.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)

#     writer.writerow(data)

# input("csv created")

# pkl_filename = dataset + ".pkl" 
pkl_filename = "rectangle3d_big_f_bb" + "_flow" + ".pkl"

print("data:", data.shape)
data = data[:, np.newaxis, ...]
velocities_x = velocities_x[:, np.newaxis, ...]
velocities_y = velocities_y[:, np.newaxis, ...]
velocities_z = velocities_z[:, np.newaxis, ...]
data = np.hstack((data, velocities_x))
data = np.hstack((data, velocities_y))
data = np.hstack((data, velocities_z))
print(data.shape)
input("x")

data = np.float32(data)

pkl_file = open(pkl_filename, 'wb')
pickle.dump(data, pkl_file, protocol=4)
pkl_file.close
print("Pkl file created")

pkl_file = open(pkl_filename, 'rb')
data = []
data = pickle.load(pkl_file)
pkl_file.close
print(data.shape)
print("Data is in range %f to %f" % (np.min(data), np.max(data)))
