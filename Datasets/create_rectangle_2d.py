# from turtle import shape
# from cv2 import rectangle
# from msilib import sequence
from matplotlib.pyplot import axes, title
import matplotlib.pyplot as plt
import pyimof
import os
import cv2
import math
import time
# import torch
# import torch.distributed as dist
import numpy as np
import pickle
import random
import argparse

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.zeros((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

# from utils import visualize_ind, visualize_series, visualize_series_flow, visualize_large
def visualize_series(data_to_vis, dir_res="Results", title="Data", show=True, save=False):
    fig=plt.figure()
    columns = 10
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
velocities_x = np.zeros((grit_t, grid_x, grid_y), dtype=np.float32)
velocities_y = np.zeros((grit_t, grid_x, grid_y), dtype=np.float32)

box_dim_x = 60 # 40
box_dim_y = 80 # 60
# 25x40 is 6% of pixels - loss distill necomes nan, 
box = np.ones((box_dim_x, box_dim_y), dtype=np.float32)
# box = np.zeros((box_dim_x, box_dim_y), dtype=int)

# # add texture to the box
# box *= 255.
# # x: 8-12 18-22 28-32
# # y: 13-17 28-32 43-47
# box[8:12, :] = 128.
# box[18:22, :] = 128.
# box[28:32, :] = 128.
# box[:, 13:17] = 128.
# box[:, 28:32] = 128.
# box[:, 43:47] = 128.
# box /= 255. # to 0...1

# add high freq texture to the box
# 4x4 areas with diff values
box *= 255.
step_size = 10 # 4 5
for i in range(0, box_dim_x, step_size):
    for j in range(0, box_dim_y, step_size):
        rand_value = np.random.randint(30, 256)
        # print(rand_value)
        box[i:i+step_size, j:j+step_size] = rand_value
box /= 255. # to 0...1
# print(box)
# input("x")

vel_min = -6 # -8 
vel_max = 6 # 8

pos_x = random.randint(0, grid_x - box_dim_x)
pos_y = random.randint(0, grid_y - box_dim_y)
# print("pos_x, pos_y:", pos_x, pos_y)
# grid[0, pos_x:pos_x+box_dim_x, pos_y:pos_y+box_dim_y] = box

if pos_x < 0: 
    pos_x = 0
if pos_y < 0: 
    pos_y = 0
if pos_x > grid_x - box_dim_x:
    pos_x = grid_x - box_dim_x
if pos_y > grid_y - box_dim_y:
    pos_y = grid_y - box_dim_y
# print("pos_x, pos_y:", pos_x, pos_y)

begin_x = pos_x
begin_y = pos_y
# print("begin_x, begin_y:", begin_x, begin_y)

end_x = pos_x + box_dim_x
end_y = pos_y + box_dim_y
# print("end_x, end_y:", end_x, end_y)

vel_x = random.randint(vel_min, vel_max)
vel_y = random.randint(vel_min, vel_max)
# print("vel_x, vel_y:", vel_x, vel_y)

max_seq = 15 # 10
seq = max_seq
for i in range(grit_t):
    if seq == 0: # till new direction
        vel_x = random.randint(vel_min, vel_max)
        vel_y = random.randint(vel_min, vel_max)
        seq = max_seq

    # # save vel vectors
    # velocities_x[i, end_x-box_dim_x:end_x, end_y-box_dim_y:end_y] = vel_x
    # velocities_y[i, end_x-box_dim_x:end_x, end_y-box_dim_y:end_y] = vel_y

    # vel_x = 8
    # vel_y = 2

    # this is reversed because matplotlib transposes x and y
    pos_x += vel_y
    pos_y += vel_x
    if pos_x < 0: 
        pos_x = 0
    if pos_y < 0: 
        pos_y = 0
    if pos_x > grid_x - box_dim_x:
        pos_x = grid_x - box_dim_x
    if pos_y > grid_y - box_dim_y:
        pos_y = grid_y - box_dim_y
    # print("pos_x, pos_y:", pos_x, pos_y)

    begin_x = pos_x
    begin_y = pos_y
    # print("begin_x, begin_y:", begin_x, begin_y)
    end_x = pos_x + box_dim_x
    end_y = pos_y + box_dim_y
    # print("end_x, end_y:", end_x, end_y)

    # print(i, pos_x, pos_y, pos_z)
    # print(box.shape)

    # print(end_x - pos_x, end_y - pos_y)
    grid[i, begin_x:end_x, begin_y:end_y] = box

    # grid[i, 20:30, 0:10] = 1 # this is it

    # save vel vectors
    velocities_x[i, begin_x:end_x, begin_y:end_y] = vel_x
    velocities_y[i, begin_x:end_x, begin_y:end_y] = vel_y

    seq -= 1
    if pos_x == 0 or pos_y == 0 or pos_x == grid_x - box_dim_x or pos_y == grid_y - box_dim_y:
        seq = 0
    # input(seq)

    if i>=0 and i<=100:
        print("i, vel_x, vel_y:", i, vel_x, vel_y)

print(grit_t, "timesteps created")
print(grid.shape)

data = grid

dir_res = "Results"
dataset = "rectange2d"
# dir_res = os.path.join(dir_res, dataset)
print("Saving at:", dir_res)

# data_video = data * 255.
# video_name = dataset + "_10fps.mp4"
# fps = 10 # 20
# sizeX = data_video.shape[1]
# sizeY = data_video.shape[2]
# out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (sizeY, sizeX), False)
# for i in range(200):
#     out.write(data_video[i].astype('uint8'))
# out.release()
# print("Created:", video_name)
# input("x")

# title = "rectange2d"
visualize_series(data[:100], dir_res, title=dataset, show=False, save=True) # 2701::3
visualize_series(velocities_x[:100], dir_res, title="rect2d_flow_x", show=False, save=True) # 2701::3
visualize_series(velocities_y[:100], dir_res, title="rect2d_flow_y", show=False, save=True) # 2701::3
visualize_series_flow(data[:100], velocities_x[:100], velocities_y[:100], 
    dataset, dir_res="Results", title="rect2d_flow_vec", show=False, save=True)
# print(velocities_x[2000])
# print(velocities_x[2010])
input("velocities")

# pkl_filename = "rectangle2d" + ".pkl" 
pkl_filename = "rectangle2d_big_hftext" + "_flow" + "_v3" + ".pkl" 

print("data:", data.shape)
data = data[:, np.newaxis, ...]
velocities_x = velocities_x[:, np.newaxis, ...]
velocities_y = velocities_y[:, np.newaxis, ...]
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
