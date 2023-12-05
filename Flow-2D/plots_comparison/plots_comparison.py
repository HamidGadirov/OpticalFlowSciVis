from calendar import c
import os
from platform import java_ver
import cv2
from cv2 import rectangle
import numpy as np
import pickle
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import pyimof
import math
import plotly.graph_objects as go

import io 
from PIL import Image

def visualize_seq_dens(original_data, interpol_data, diffs, interpol_data_RIFE, diffs_RIFE,
    dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    
    columns = 10
    if "droplet2d" in dataset:
        columns = 25
    elif dataset == "lbs2d":
        columns = 8
    rows = 5  # Updated to 5 rows
    skip = 1
    data_to_vis = original_data
    index = 0
    shift = 1

    if "lbs2d" in dataset:
        vmin = 95
        print("vis range:", vmin, data_to_vis.max())
        hspace = -0.14
        wspace = 0.01
        cmap ='turbo'
    if "droplet2d" in dataset:
        vmin = 0
        print("vis range:", vmin, data_to_vis.max())
        hspace = -0.25
        wspace = 0.01
        cmap ='gray'

    # wspace is x; hspace is y
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(30, 5), gridspec_kw={'hspace': hspace, 'wspace': wspace})

    # TODO: fix timesteps, t

    for i in range(columns):
        # First row: original data
        ax = axs[0][i]
        img = original_data[round(index)]
        if dataset == "lbs2d":
            ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        else:
            ax.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
        if dataset == "lbs2d":
            ax_title = "t=" + str(index + shift + 3000 + 1605)
            ax.set_title(ax_title, fontsize=20)
        elif dataset == "droplet2d":
            ax_title = "t=" + str(index + shift + 15)
            ax.set_title(ax_title, fontsize=12, pad=10)
        ax.axis('off')

        # Second row: interpolated data
        ax = axs[1][i]
        img = interpol_data[round(index)]
        if dataset == "lbs2d":
            ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        else:
            ax.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # Third row: difference data
        ax = axs[2][i]
        img = diffs[round(index)]
        ax.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # Fourth row: interpolated data (uses interpol_data_RIFE)
        ax = axs[3][i]
        img = interpol_data_RIFE[round(index)]
        if dataset == "lbs2d":
            ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        else:
            ax.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # Fifth row: difference data (uses diffs_RIFE)
        ax = axs[4][i]
        img = diffs_RIFE[round(index)]
        ax.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        index += skip
        if i % columns == 0:
            shift += 1  # we have only GT here
        if round(index) == round(columns * skip):
            index = 0

    if dataset == "lbs2d":
        normalizer = Normalize(vmin=vmin, vmax=data_to_vis.max())
        im = cm.ScalarMappable(norm=normalizer, cmap='turbo')
        if "lbs2d" in dataset:
            cb_ax = fig.add_axes([0.91, 0.18, 0.005, 0.62])
        elif "rectangle2d" in dataset:
            cb_ax = fig.add_axes([0.75, 0.5, 0.005, 0.4])
        cb_ax.tick_params(labelsize=20)
        fmt = lambda x, pos: '{:0.1f}'.format(x)
        bar = np.linspace(vmin, data_to_vis.max(), 4, endpoint=True)
        plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))

    if show:
        plt.show()
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi=300)


def visualize_seq_flow(flow_gt, flow_pred, flow_u_diff, flow_v_diff,
    flow_pred_RIFE, flow_u_diff_RIFE, flow_v_diff_RIFE, 
    dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    
    columns = 10
    if "droplet2d" in dataset:
        columns = 25
    elif dataset == "lbs2d":
        columns = 8
    rows = 5  # Changed to 5 rows
    skip = 3  # 1
    data_to_vis = flow_gt
    index = 0
    shift = 1

    if "lbs2d" in dataset:
        quiver_steps = 4
        max_flow_gt = flow_gt.max()
        max_flow_pred = 0.5
        scale = None

    flow_u_gt = flow_gt[:, 0, ...]
    flow_v_gt = flow_gt[:, 1, ...]
    flow_u = flow_pred[:, 0, ...]
    flow_v = flow_pred[:, 1, ...]

    flow_u_RIFE = flow_pred_RIFE[:, 0, ...]  # RIFE prediction
    flow_v_RIFE = flow_pred_RIFE[:, 1, ...]  # RIFE prediction

    flow_u_diff_RIFE = flow_u_diff_RIFE  # RIFE difference
    flow_v_diff_RIFE = flow_v_diff_RIFE  # RIFE difference

    # wspace is x; hspace is y
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(30, 5), gridspec_kw={'hspace': -0.14, 'wspace': 0.01})

    # TODO: fix timesteps, t

    for i in range(columns):
        # First row: original flow
        ax = axs[0][i]
        u_gt = flow_u_gt[round(index)]
        v_gt = flow_v_gt[round(index)]
        norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
        img = np.zeros((flow_u_gt.shape[1], flow_u_gt.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale=scale, bg_cmap='gray', step=quiver_steps)
        # if dataset == "lbs2d":
        #     ax_title = "t=" + str(index + shift + 3000 + 1605)
        # elif dataset == "droplet2d":
        #     ax_title = "t=" + str(index + shift + 15)
        # ax.set_title(ax_title, fontsize=20)
        ax.axis('off')

        # Second row: predicted flow
        ax = axs[1][i]
        u = flow_u[round(index)]
        v = flow_v[round(index)]
        norm = np.sqrt(u * u + v * v)
        img = np.zeros((flow_u.shape[1], flow_u.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale=scale, bg_cmap='gray', step=quiver_steps)
        ax.axis('off')

        # Third row: difference flow
        ax = axs[2][i]
        u_diff = flow_u_diff[round(index)]
        v_diff = flow_v_diff[round(index)]
        norm_diff = np.sqrt(u_diff * u_diff + v_diff * v_diff)
        img = np.zeros((flow_u.shape[1], flow_u.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u_diff, v_diff, c=norm_diff, bg=img, ax=ax, clim=(0, max_flow_gt), cmap='turbo', scale=scale, bg_cmap='gray', step=quiver_steps)
        ax.axis('off')

        # Fourth row: predicted flow (RIFE)
        ax = axs[3][i]
        u_RIFE = flow_u_RIFE[round(index)]  # Use RIFE prediction
        v_RIFE = flow_v_RIFE[round(index)]  # Use RIFE prediction
        norm_RIFE = np.sqrt(u_RIFE * u_RIFE + v_RIFE * v_RIFE)
        img = np.zeros((flow_u_RIFE.shape[1], flow_u_RIFE.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u_RIFE, v_RIFE, c=norm_RIFE, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale=scale, bg_cmap='gray', step=quiver_steps)
        ax.axis('off')

        # Fifth row: difference flow (RIFE)
        ax = axs[4][i]
        u_diff_RIFE = flow_u_diff_RIFE[round(index)]  # Use RIFE difference
        v_diff_RIFE = flow_v_diff_RIFE[round(index)]  # Use RIFE difference
        norm_diff_RIFE = np.sqrt(u_diff_RIFE * u_diff_RIFE + v_diff_RIFE * v_diff_RIFE)
        img = np.zeros((flow_u_RIFE.shape[1], flow_u_RIFE.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u_diff_RIFE, v_diff_RIFE, c=norm_diff_RIFE, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale=scale, bg_cmap='gray', step=quiver_steps)
        ax.axis('off')

        index += skip
        if i % columns == 0:
            shift += 1  # we have only GT here
        if round(index) == round(columns * skip):
            index = 0

    normalizer = Normalize(vmin=0., vmax=max_flow_gt)
    im = cm.ScalarMappable(norm=normalizer, cmap='turbo')
    # left bottom width height
    if "lbs2d" in dataset:
        cb_ax = fig.add_axes([0.91, 0.18, 0.005, 0.62])
    elif "rectangle2d" in dataset:
        cb_ax = fig.add_axes([0.75, 0.5, 0.005, 0.4])
    cb_ax.tick_params(labelsize=20)
    fmt = lambda x, pos: '{:0.2f}'.format(x)
    bar = np.linspace(0., max_flow_gt, 4, endpoint=True)
    plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))

    if show:
        plt.show()
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi=300)


# Define the file names
original_data_file = 'original_data.pkl'
interpol_data_LREST_file = 'interpol_data_LREST.pkl'
diffs_LREST_file = 'diffs_LREST.pkl'
interpol_data_RIFE_file = 'interpol_data_RIFE.pkl'
diffs_RIFE_file = 'diffs_RIFE.pkl'

# Load data from the .pkl files
with open(original_data_file, 'rb') as f:
    original_data = pickle.load(f)

with open(interpol_data_LREST_file, 'rb') as f:
    interpol_data_LREST = pickle.load(f)

with open(diffs_LREST_file, 'rb') as f:
    diffs_LREST = pickle.load(f)

with open(interpol_data_RIFE_file, 'rb') as f:
    interpol_data_RIFE = pickle.load(f)

with open(diffs_RIFE_file, 'rb') as f:
    diffs_RIFE = pickle.load(f)

title = "Density_gt_pred_diff"
dir_model = "./"
dataset = "lbs2d"
visualize_seq_dens(original_data, interpol_data_LREST, diffs_LREST, interpol_data_RIFE, diffs_RIFE,
        dataset, dir_model, title=title, show=False, save=True)

# Define FLOW the file names
flow_gt_file = 'flow_gt.pkl'
flow_pred_LREST_file = 'flow_pred_LREST.pkl'
flow_u_diff_LREST_file = 'flow_u_diff_LREST.pkl'
flow_v_diff_LREST_file = 'flow_v_diff_LREST.pkl'
flow_pred_RIFE_file = 'flow_pred_RIFE.pkl'
flow_u_diff_RIFE_file = 'flow_u_diff_RIFE.pkl'
flow_v_diff_RIFE_file = 'flow_v_diff_RIFE.pkl'

# Load ground truth flow data
with open(flow_gt_file, 'rb') as f:
    flow_gt = pickle.load(f)

# Load LREST predicted flow data and differences
with open(flow_pred_LREST_file, 'rb') as f:
    flow_pred_LREST = pickle.load(f)
with open(flow_u_diff_LREST_file, 'rb') as f:
    flow_u_diff_LREST = pickle.load(f)
with open(flow_v_diff_LREST_file, 'rb') as f:
    flow_v_diff_LREST = pickle.load(f)

# Load RIFE predicted flow data and differences
with open(flow_pred_RIFE_file, 'rb') as f:
    flow_pred_RIFE = pickle.load(f)
with open(flow_u_diff_RIFE_file, 'rb') as f:
    flow_u_diff_RIFE = pickle.load(f)
with open(flow_v_diff_RIFE_file, 'rb') as f:
    flow_v_diff_RIFE = pickle.load(f)

dir_model = "./"
dataset = "lbs2d"
if dataset == "lbs2d":
    title = "Flow_gt_pred_diff"
    visualize_seq_flow(flow_gt, flow_pred_LREST, flow_u_diff_LREST, flow_v_diff_LREST, 
                                flow_pred_RIFE, flow_u_diff_RIFE, flow_v_diff_RIFE, 
                            dataset, dir_model, title=title, show=False, save=True)