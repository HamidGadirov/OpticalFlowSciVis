# from turtle import shape
# from cv2 import rectangle
# from msilib import sequence
from matplotlib.pyplot import axes, title
import matplotlib.pyplot as plt
import pyimof
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
    columns = 6
    rows = 10
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

def visualize_series_flow(data_to_vis, flow_u, flow_v, dataset, dir_res="Results", title="Flow", show=True, save=False):
    fig=plt.figure()
    columns = 10
    rows = 10

    for i in range(1, columns*rows+1 ):
        if (i >= data_to_vis.shape[0] or i >= flow_u.shape[0]):
            break
        # Vector field quiver plot
        u = flow_u[i]
        v = flow_v[i]
        norm = np.sqrt(u*u + v*v)
        # img = data_to_vis[i]
        img = np.zeros((data_to_vis.shape[1], data_to_vis.shape[2]))
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        ax = plt.gca()
        pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray')

    fig = plt.gcf()
    plt.suptitle(title) 
    fig.set_size_inches(12, 9)
    if show:
        plt.show()  
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

# box at random position, moving with random speed in random directions
grid_x = 128
grid_y = 128
grit_t = 3000
grid = np.zeros((grit_t, grid_x, grid_y), dtype=np.float32)
# grid = np.ones((grit_t, grid_x, grid_y), dtype=int)
velocities = np.zeros((grit_t, 2, grid_x, grid_y), dtype=np.float32)

box_dim_x = 20
box_dim_y = 30
box = np.ones((4, box_dim_x, box_dim_y), dtype=np.float32)
for i in range(4):
    # box[i, :, :] = np.ones((box_dim_x, box_dim_y), dtype=np.float32)
    box[i] *= 255.
    # add texture to the boxes
    box[i][8:12, :] = 128.
    box[i][:, 13:17] = 128.
    box[i] /= 255. # to 0...1

vel_min = -8
vel_max = 8

pos = np.zeros((4, 2), dtype=np.int32)
begin = np.zeros((4, 2), dtype=np.int32)
end = np.zeros((4, 2), dtype=np.int32)
vel = np.zeros((4, 2), dtype=np.int32)
for i in range(4):
    pos[i][0] = random.randint(0, grid_x - box_dim_x)
    pos[i][1] = random.randint(0, grid_y - box_dim_y)
    if pos[i][0] < 0: 
        pos[i][0] = 0
    if pos[i][1] < 0: 
        pos[i][1] = 0
    # i am confusing x and y startig positions
    if pos[i][0] > grid_x - box_dim_x:
        pos[i][0] = grid_x - box_dim_x
    if pos[i][1] > grid_y - box_dim_y:
        pos[i][1] = grid_y - box_dim_y

    begin[i][0] = pos[i][0]
    begin[i][1] = pos[i][1]
    end[i][0] = pos[i][0] + box_dim_x
    end[i][1] = pos[i][1] + box_dim_y

    vel[i][0] = random.randint(vel_min, vel_max)
    vel[i][1] = random.randint(vel_min, vel_max)

max_seq = 10
seq = max_seq
for t in range(grit_t):
    if seq == 0: # till new direction
        for i in range(4):
            vel[i][0] = random.randint(vel_min, vel_max)
            vel[i][1] = random.randint(vel_min, vel_max)
        seq = max_seq
    
    # save vel vectors
    for i in range(4):
        # velocities[i, 0, t, end[i][0]-box_dim_x:end[i][0], end[i][1]-box_dim_y:end[i][1]] = vel[i][0]
        # velocities[i, 1, t, end[i][0]-box_dim_x:end[i][0], end[i][1]-box_dim_y:end[i][1]] = vel[i][1]

        pos[i][0] += vel[i][1]
        pos[i][1] += vel[i][0]

        if pos[i][0] < 0: 
            pos[i][0] = 0
        if pos[i][1] < 0: 
            pos[i][1] = 0
        if pos[i][0] > grid_x - box_dim_x:
            pos[i][0] = grid_x - box_dim_x
        if pos[i][1] > grid_y - box_dim_y:
            pos[i][1] = grid_y - box_dim_y

        begin[i][0] = pos[i][0]
        begin[i][1] = pos[i][1]
        end[i][0] = pos[i][0] + box_dim_x
        end[i][1] = pos[i][1] + box_dim_y

        # print(box[i].shape)
        # print(begin[i][0], end[i][0])
        # print(begin[i][1], end[i][1])
        grid[t, begin[i][0]:end[i][0], begin[i][1]:end[i][1]] = box[i]

        velocities[t, 0, begin[i][0]:end[i][0], begin[i][1]:end[i][1]] += vel[i][0]
        velocities[t, 1, begin[i][0]:end[i][0], begin[i][1]:end[i][1]] += vel[i][1]

    seq -= 1
    # this can be improved
    if pos[i][0] == 0 or pos[i][1] == 0 or \
        pos[i][0] == grid_x - box_dim_x or pos[i][1] == grid_y - box_dim_y:
        seq = 0

print(grit_t, "timesteps created")
print(grid.shape)

data = grid

dir_res = "Results"
dataset = "rectange2d"
# dir_res = os.path.join(dir_res, dataset)
print("Saving at:", dir_res)
# title = "rectange2d"
visualize_series(data[2000:], dir_res, title="rectanges2d", show=False, save=True) # 2701::3
visualize_series(velocities[2000:, 0], dir_res, title="rects2d_flow_x", show=False, save=True) # 2701::3
visualize_series(velocities[2000:, 1], dir_res, title="rects2d_flow_y", show=False, save=True) # 2701::3
visualize_series_flow(data[2000:], velocities[2000:, 0], velocities[2000:, 1], 
    dataset, dir_res="Results", title="rects2d_flow_vec", show=False, save=True)
# input("velocities")

pkl_filename = "rectangles2d_text" + "_flow" + ".pkl" 

print("data:", data.shape)
data = data[:, np.newaxis, ...]
velocities_x = velocities[:, 0]
velocities_y = velocities[:, 1]
velocities_x = velocities_x[:, np.newaxis, ...]
velocities_y = velocities_y[:, np.newaxis, ...]
print("vel:", velocities_x.shape)
data = np.hstack((data, velocities_x))
data = np.hstack((data, velocities_y))
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
