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

box_dim_x = 20
box_dim_y = 30
box_dim_z = 40
box = np.ones((box_dim_x, box_dim_y, box_dim_z), dtype=int) * 255 # 255 for better gradients ?
# box = np.zeros((box_dim_x, box_dim_y, box_dim_z), dtype=int)

vel_min = -8
vel_max = 8

pos_x = random.randint(0, grid_x - box_dim_x)
pos_y = random.randint(0, grid_y - box_dim_y)
pos_z = random.randint(0, grid_z - box_dim_z)
# grid[0, pos_x:pos_x+box_dim_x, pos_y:pos_y+box_dim_y] = box

vel_x = random.randint(vel_min, vel_max)
vel_y = random.randint(vel_min, vel_max)
vel_z = random.randint(vel_min, vel_max)

max_seq = 10
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

pkl_filename = dataset + ".pkl" 

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
