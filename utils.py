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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import pyimof
import math
import plotly.graph_objects as go
import yt
from yt.visualization.api import Streamlines
import io 
from PIL import Image
from matplotlib.colors import LogNorm
import glob
from natsort import natsorted
from metrics import calculate_mean_of_closest_points_streamline
import pyvista as pv

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.zeros((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

def plot_distribution(data, dir_res, name="dist.png", save=False):
    fig = plt.figure(figsize=(8, 4))
    plt.hist(data.reshape(-1), bins=50)
    plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    if save:
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        res_path = os.path.join(dir_res, name)
        fig.savefig(res_path)

def plot_loss(loss, dir_res, name="loss.png", save=False):
    fig = plt.figure(figsize=(8, 4))
    # labels = ('total', 'lapl', 'tea', 'distill', 'reg', 'photo', 'flow')
    colors = ('r', 'g', 'b', 'c', 'm', 'y', 'orange')
    # markers = ("_", "x", "+", "*", ".", "d", "v")
    print(loss.shape)
    for j in range(loss.shape[1]): # how many loss components
        plt.plot(loss[:, j], colors[j]) # , marker=markers[j])
    # input("x")
    plt.title('validation losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.legend(['total', 'lapl', 'tea', 'dist', 'reg', 'photo', 'flow'], loc='upper right')
    plt.legend(['Total', 'L1', 'Teacher', 'Distill', 'Reg', 'Photo', 'Flow'], loc='upper right') # Laplacian vs L1(MAE)
    # plt.legend()
    if save:
        res_path = os.path.join(dir_res, name)
        fig.savefig(res_path)

def plot_lr(lr, dir_res, name="learning_rate.png", save=False):
    fig = plt.figure(figsize=(8, 4))
    print(lr.shape)
    plt.plot(lr)
    # input("x")
    plt.title('learning rate plot')
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    if save:
        res_path = os.path.join(dir_res, name)
        fig.savefig(res_path)

def visualize_ind(image, dir_res="Results", title="img", show=True, save=False):
    fig = plt.figure()
    plt.axis('off')

    # plt.imshow(image.astype('uint8'))
    plt.imshow(image, cmap='viridis')
    plt.suptitle(title)
    if show:
        plt.show() 
    if save:
        title += ".png"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        dir_res = os.path.join(dir_res, title)
        fig.savefig(dir_res)
        # input("saved")

def visualize_ind_flow(image, max_flow_gt, dir_res="Results", title="flow", show=True, save=False):
    fig, ax = plt.subplots()
    plt.axis('off')
    scale = None
    quiver_steps = 4

    u = image[0]
    v = image[1]
    norm = np.sqrt(u*u + v*v)
    img = np.zeros((image.shape[1], image.shape[2]))
    pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale = scale, bg_cmap='gray', step=quiver_steps)

    # plt.suptitle(title)
    if show:
        plt.show() 
    if save:
        title += ".png"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        dir_res = os.path.join(dir_res, title)
        fig.savefig(dir_res)

def visualize_series(data_to_vis, factor, dataset, dir_res="Results", title="Data", show=True, save=False):
    fig = plt.figure()
    columns = 10
    if "droplet2d" in dataset:
        columns = 10 # 20
    elif "pipedcylinder2d" in dataset:
        columns = 7 #
    elif "cylinder2d" in dataset:
        columns = 7
    rows = 10

    print("data_to_vis is in range %f to %f" % (np.min(data_to_vis), np.max(data_to_vis)))

    j = 0
    for i in range(1, columns*rows+1):
        # index = (i - 1) * 2 # skip each second
        index = i

        if (index >= data_to_vis.shape[0]):
            break

        img = data_to_vis[index,...]
        ax = fig.add_subplot(rows, columns, i)
        
        # if index % factor == 0:
        #     # add bb to the gt image
        #     width = data_to_vis.shape[2]
        #     height = data_to_vis.shape[1]
        #     linewidth=2
            # if "cylinder2d" in dataset:
            #     linewidth=1
            # rect = patches.Rectangle((0, 0), width, height, linewidth=linewidth, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)

        # vis 3 in sequence:
        if j <= 2:
            # add bb to the gt image
            width = data_to_vis.shape[2]
            height = data_to_vis.shape[1]
            linewidth=2
            rect = patches.Rectangle((0, 0), width, height, linewidth=linewidth, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        j += 1
        if j % 6 == 0:
            j = 0

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

def visualize_series_flow(data_to_vis, flow_u, flow_v, dataset, dir_res="Results", title="Flow", show=True, save=False):
    fig=plt.figure()
    columns = 10
    if "droplet2d" in dataset:
        columns = 10
    elif "pipedcylinder2d" in dataset:
        columns = 7
    elif "cylinder2d" in dataset:
        columns = 7
    rows = 10

    for i in range(1, columns*rows+1 ):
        index = (i-1)*2 # skip eaach second
        if (index >= data_to_vis.shape[0] or index >= flow_u.shape[0]):
            break
        # Vector field quiver plot
        u = flow_u[round(index)]
        v = flow_v[round(index)]
        norm = np.sqrt(u*u + v*v)
        img = data_to_vis[round(index)]
        
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        ax = plt.gca()
        pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray')
        # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
        
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

def visualize_seq_dens(original_data, interpol_data, diffs,
    factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    
    columns = 10
    if "droplet2d" in dataset:
        columns = 25
    elif dataset == "lbs2d":
        columns = 8
    rows = 3
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
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(30, 3), gridspec_kw={'hspace': hspace, 'wspace': wspace})

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
            ax_title = "t=" + str(index + shift + 25)
            ax.set_title(ax_title, fontsize=12, pad=10)
        elif dataset == "jets3d":
            ax_title = "t=" + str(index + shift + 25)
            ax.set_title(ax_title, fontsize=12, pad=10)
        elif dataset == "pool3d":
            ax_title = "t=" + str(index + shift + 30)
            ax.set_title(ax_title, fontsize=12, pad=10)
        elif dataset == "nyx3d":
            ax_title = "t=" + str(index + shift + 600)
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

        index += skip
        if i % columns == 0:
            shift += 1 # we have only GT here
        if round(index) == round(columns * skip):
            index = 0

    if dataset == "lbs2d":
        normalizer=Normalize(vmin=vmin, vmax=data_to_vis.max())
        im=cm.ScalarMappable(norm=normalizer, cmap='turbo')
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
        fig.savefig(os.path.join(dir_res, title), dpi = 300)
        # save data for plots
        # """
        original_data_filename = os.path.join(dir_res, 'original_data.pkl')
        with open(original_data_filename, 'wb') as f:
            pickle.dump(original_data, f)
        # interpol_data_filename = os.path.join(dir_res, 'interpol_data_LREST.pkl')
        interpol_data_filename = os.path.join(dir_res, 'interpol_data_RIFE.pkl')
        with open(interpol_data_filename, 'wb') as f:
            pickle.dump(interpol_data, f)
        # diffs_filename = os.path.join(dir_res, 'diffs_LREST.pkl')
        diffs_filename = os.path.join(dir_res, 'diffs_RIFE.pkl')
        with open(diffs_filename, 'wb') as f:
            pickle.dump(diffs, f)
        # """

def visualize_seq_flow(flow_gt, flow_pred, flow_u_diff, flow_v_diff,
    factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
        
    columns = 10
    if "droplet2d" in dataset:
        columns = 25
    elif dataset == "lbs2d":
        columns = 8
    rows = 3
    skip = 3 # 1
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

    # wspace is x; hspace is y
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(30, 3), gridspec_kw={'hspace': -0.14, 'wspace': 0.01})

    # TODO: fix timesteps, t

    for i in range(columns):
        # First row: original flow
        ax = axs[0][i]
        u_gt = flow_u_gt[round(index)]
        v_gt = flow_v_gt[round(index)]
        norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
        img = np.zeros((flow_u_gt.shape[1], flow_u_gt.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale = scale, bg_cmap='gray', step=quiver_steps)
        if dataset == "lbs2d":
            ax_title = "t=" + str(index + shift + 3000 + 1605)
        elif dataset == "droplet2d":
            ax_title = "t=" + str(index + shift + 25)
        elif dataset == "jets3d":
            ax_title = "t=" + str(index + shift + 25)
        elif dataset == "pool3d":
            ax_title = "t=" + str(index + shift + 30)
        elif dataset == "nyx3d":
            ax_title = "t=" + str(index + shift + 600)
        ax.set_title(ax_title, fontsize=20)
        ax.axis('off')

        # Second row: predicted flow
        ax = axs[1][i]
        u = flow_u[round(index)]
        v = flow_v[round(index)]
        norm = np.sqrt(u * u + v * v)
        img = np.zeros((flow_u.shape[1], flow_u.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale = scale, bg_cmap='gray', step=quiver_steps)
        ax.axis('off')

        # Third row: difference flow
        ax = axs[2][i]
        u_diff = flow_u_diff[round(index)]
        v_diff = flow_v_diff[round(index)]
        norm_diff = np.sqrt(u_diff * u_diff + v_diff * v_diff)
        img = np.zeros((flow_u.shape[1], flow_u.shape[2]))
        plt.axis('off')
        pyimof.display.quiver(u_diff, v_diff, c=norm_diff, bg=img, ax=ax, clim=(0, max_flow_gt), cmap='turbo', scale = scale, bg_cmap='gray', step=quiver_steps)
        ax.axis('off')

        index += skip
        if i % columns == 0:
            shift += 1 # we have only GT here
        if round(index) == round(columns * skip):
            index = 0

    normalizer=Normalize(vmin=0., vmax=max_flow_gt)
    im=cm.ScalarMappable(norm=normalizer, cmap='turbo')
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
        fig.savefig(os.path.join(dir_res, title), dpi = 300)
        # """
        flow_gt_filename = os.path.join(dir_res, 'flow_gt.pkl')
        with open(flow_gt_filename, 'wb') as f:
            pickle.dump(flow_gt, f)
        # save data for plots   flow_pred, flow_u_diff, flow_v_diff
        # interpol_data_filename = os.path.join(dir_res, 'flow_pred_LREST.pkl')
        interpol_data_filename = os.path.join(dir_res, 'flow_pred_RIFE.pkl')
        with open(interpol_data_filename, 'wb') as f:
            pickle.dump(flow_pred, f)
        flow_diff = np.stack((flow_u_diff, flow_v_diff), axis=1)
        # diffs_filename = os.path.join(dir_res, 'flow_diff_LREST.pkl')
        diffs_filename = os.path.join(dir_res, 'flow_diff_RIFE.pkl')
        with open(diffs_filename, 'wb') as f:
            pickle.dump(flow_diff, f)
        # # diffs_filename = os.path.join(dir_res, 'flow_u_diff_LREST.pkl')
        # diffs_filename = os.path.join(dir_res, 'flow_u_diff_RIFE.pkl')
        # with open(diffs_filename, 'wb') as f:
        #     pickle.dump(flow_u_diff, f)
        # # diffs_filename = os.path.join(dir_res, 'flow_v_diff_LREST.pkl')
        # diffs_filename = os.path.join(dir_res, 'flow_v_diff_RIFE.pkl')
        # with open(diffs_filename, 'wb') as f:
        #     pickle.dump(flow_v_diff, f)
        # """

def visualize_large(original_data, interpol_data, diffs,
        flow, flow_gt, diffs_flow, flow_u_diff, flow_v_diff, mask,
        factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    # print(original_data.shape, interpol_data.shape, diffs.shape, flow.shape, flow_gt.shape, diffs_flow.shape, u_diff.shape, v_diff.shape)
    fig = plt.figure(figsize=(40, 10))
    columns = 20
    if "droplet2d" in dataset or "FluidSimML2d" in dataset or "rectangle2d" in dataset:
        columns = 24
    elif dataset == "lbs2d":
        columns = 7 # 9 30
    elif "pipedcylinder2d" in dataset or "cylinder2d" in dataset:
        columns = 20
    elif dataset == "vimeo2d":
        columns = 28
    elif dataset == "jets3d":
        columns = 19
    elif dataset == "pool3d":
        columns = 19
    elif dataset == "nyx3d":
        columns = 19

    rows = 7 # 4
    skip = 1
    # if factor == 8:
    #     skip = 2
    # if factor >= 16: # or factor == 32 or factor == 64:
    #     skip = 4
    # if factor == 128:
    #     skip = 8

    print(flow.shape, flow_gt.shape)
    # input("x")

    data_to_vis = original_data
    flow_u_gt = flow_gt[:, 0, ...]
    flow_v_gt = flow_gt[:, 1, ...]
    flow_u = flow[:, 0, ...]
    flow_v = flow[:, 1, ...]

    print("original_data is in range %f to %f" % (np.min(original_data), np.max(original_data)))
    print("interpol_data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))

    print("original_data mean & std:", original_data.mean(), original_data.std())
    print("interpol_data mean & std:", interpol_data.mean(), interpol_data.std())

    print("flow_u_gt is in range %f to %f" % (np.min(flow_u_gt), np.max(flow_u_gt)))
    print("flow_v_gt is in range %f to %f" % (np.min(flow_v_gt), np.max(flow_v_gt)))

    print("flow_u is in range %f to %f" % (np.min(flow_u), np.max(flow_u))) 
    print("flow_v is in range %f to %f" % (np.min(flow_v), np.max(flow_v)))

    print("flow_u_diff is in range %f to %f" % (np.min(flow_u_diff), np.max(flow_u_diff))) 
    print("flow_v_diff is in range %f to %f" % (np.min(flow_v_diff), np.max(flow_v_diff)))
    # input("viz")
    # norm_gt = np.sqrt(np.max(flow_u_gt) * np.max(flow_u_gt) + np.max(flow_v_gt) * np.max(flow_v_gt))
    # input(norm_gt)

    # TODO: set norm as max from gt and pred -
    # norm_gt = np.sqrt(flow_u_gt * flow_u_gt + flow_v_gt * flow_v_gt)
    # norm_pred = np.sqrt(flow_u * flow_u + flow_v * flow_v)
    # print(max(norm_gt, norm_pred))
    # quiver plot params:
    if "rectangle2d" in dataset:
        quiver_steps = 2 # 5
        max_flow_gt = 5.
        # max_flow_pred = ?
        # max_flow_diff = ?
        scale = 1.
        # clim=(0, max_flow_gt)
        vmin=data_to_vis.min()
    elif "lbs2d" in dataset:
        quiver_steps = 4
        max_flow_gt = flow_gt.max() # None # 0.1
        max_flow_pred = 0.5
        scale = None
        vmin = 95 # 100
        print("vis range:", vmin, data_to_vis.max())
        cmap ='turbo'
    elif "droplet2d" in dataset:
        vmin = 0
        max_flow_gt = 15 # flow.max()
        print("vis range:", vmin, data_to_vis.max())
        print("vis flow range:", 0, max_flow_gt)
        cmap ='gray'
        quiver_steps = 4
        # scale = None
        # magnitude = np.sqrt(flow_u**2 + flow_v**2)
        # max_magnitude = np.max(magnitude)
        # scale = 0.5 / max_magnitude  # Adjust the scale factor as needed
        scale = 0.5
    elif "jets3d" in dataset:
        quiver_steps = 4
        max_flow_gt = flow_gt.max() # None # 0.1
        min_flow_gt = flow_gt.min() # None # 0.1
        # max_flow_pred = 0.5
        scale = None
        vmin = 0
        print("vis range:", vmin, data_to_vis.max())
        cmap ='turbo'
    elif "pool3d" in dataset or "nyx3d" in dataset:
        quiver_steps = 4
        max_flow_gt = flow_gt.max() # None # 0.1
        min_flow_gt = flow_gt.min() # None # 0.1
        # max_flow_pred = 0.5
        scale = None
        vmin = 0
        print("vis range:", vmin, data_to_vis.max())
        cmap ='turbo'
    # input("norm")

    # print("flow_u & flow_u_diff:", flow_u.shape, flow_u_diff.shape)

    index = 0
    index_pred = 0
    # fig, axes = plt.subplots(nrows=rows, ncols=columns)
    # for ax in axes.flat:
    #     im = ax.imshow

    for i in range(1, columns*rows+1):
        # if (index >= data_to_vis.shape[0]):
        #     break
        ax = fig.add_subplot(rows, columns, i)
        # print(int(i/columns))
        if int((i-1)/columns) == 0: # gt
            img = original_data[round(index)]
            if dataset == "lbs2d":
                plt.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
                # plt.imshow(img, vmin=original_data.std()-original_data.mean(), vmax=data_to_vis.max())
            elif "3d" in dataset:
                plt.imshow(img, vmin=interpol_data.min(), vmax=interpol_data.max(), cmap=cmap)
            else:
                plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            # print("original_data:", original_data[round(index)])
            # if index % 3 == 0:
            #     ax.set_title('t=0') # can we help model with info of seq?
            # elif index % 3 == 1:
            #     ax.set_title('t=1')
            # else:
            #     ax.set_title('t=2')
            if dataset == "lbs2d":
                ax_title = "t=" + str(index + 3000 + 1605)
                ax.set_title(ax_title, fontsize=10)
            elif dataset == "droplet2d":
                ax_title = "t=" + str(index + 25)
                # ax.set_title(ax_title, fontsize=7)
                ax.set_title(ax_title, fontsize=5, pad=-8)
            elif dataset == "jets3d":
                ax_title = "t=" + str(index + 25)
                ax.set_title(ax_title, fontsize=5, pad=-8)
            elif dataset == "pool3d":
                ax_title = "t=" + str(index + 30)
                ax.set_title(ax_title, fontsize=5, pad=-8)
            elif dataset == "nyx3d":
                ax_title = "t=" + str(index + 600)
                ax.set_title(ax_title, fontsize=5, pad=-8)
            if index % factor != 0: # if index % 3 == 1:
                ax_title += "*"
        if int((i-1)/columns) == 1 and round(index) % factor != 0: # interpol
            # img = data_to_vis[index+int(data_to_vis.shape[0]/3),...]
            img = interpol_data[round(index_pred)] # [index-columns*2,...] # (index - 1) % 3 round((index - 1) / 2)
            if dataset == "lbs2d":
                plt.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
                # plt.imshow(img, vmin=interpol_data.std()-interpol_data.mean(), vmax=data_to_vis.max())
            elif "3d" in dataset:
                plt.imshow(img, vmin=interpol_data.min(), vmax=interpol_data.max(), cmap=cmap)
            else:
                plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            # print("index:", index)
            # print("(index - 1) % 3:", (index - 1) / 3)
            index_pred += 1

        if int((i-1)/columns) == 2 and round(index) % factor != 0: # diff
            # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
            # img = diffs[round((index - 1) / 2)] # [index-columns*2*2,...]
            img = diffs[round(index_pred)] # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            index_pred += 1

        # if int((i-1)/columns) == 3 and round(index) % factor != 0: # mask
        #     # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
        #     img = mask[round((index - 1) / 2)] # [index-columns*2*2,...]
        #     plt.imshow(img, vmin=mask.min(), vmax=mask.max(), cmap=cmap)

        # if int((i-1)/columns) == 6: # flow diff
        #     img = diffs_flow[round(index)] # [index-columns*2*2,...]
        #     plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 3 and round(index) % factor != 0: # flow gt
            if dataset == "vimeo2d" or dataset == "droplet2d": # show pred flow in hsv-rgb instead of gt
                hsv = np.empty(shape=(flow_u.shape[1], flow_u.shape[2], 3), dtype=np.uint8)
                hsv[:,:,1] = 255
                # mag, ang = cv2.cartToPolar(flow_u[round((index - 1) / 3)], flow_v[round((index - 1) / 3)])
                mag, ang = cv2.cartToPolar(flow_u[round(index_pred)], flow_v[round(index_pred)])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                # bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # print("Flow in bgr:", bgr.shape)
                # plt.imshow(bgr, vmin=data_to_vis.min(), vmax=data_to_vis.max())
                plt.imshow(bgr)
                index_pred += 1
            else:
                u_gt = flow_u_gt[round(index_pred)]
                v_gt = flow_v_gt[round(index_pred)]
                norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt) # / norm_coeff
                # print(norm_gt.mean())
                # img = original_data[round(index)] # [index-columns*2*2*2]
                img = np.zeros((original_data.shape[1], original_data.shape[2]))
                plt.axis('off')
                # ax = plt.gca()
                # print("u_gt is in range %f to %f" % (np.min(u_gt), np.max(u_gt)))
                # print("v_gt is in range %f to %f" % (np.min(v_gt), np.max(v_gt)))
                # print("u_gt at 300, 25:", u_gt[25, 300])
                # print("u_gt at 300, 75:", u_gt[75, 300])
                # print("norm_gt is in range %f to %f" % (np.min(norm_gt), np.max(norm_gt)))
                #  pyimof.display.quiver(u, v, c, ax=ax, clim=(-5, 5)) --- use clim
                # pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
                pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap=cmap, clim=(0, max_flow_gt), scale = scale, bg_cmap='gray', step=quiver_steps)
                # pyimof.display.quiver(u_gt, u_gt, c=u_gt, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
                # pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='jet', bg_cmap='gray')
                # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')

                # hsv = np.empty(shape=(flow_u_gt.shape[1], flow_v_gt.shape[2], 3), dtype=np.uint8)
                # hsv[:,:,1] = 255
                # mag, ang = cv2.cartToPolar(flow_u_gt[round(index)], flow_v_gt[round(index)])
                # hsv[..., 0] = ang * 180 / np.pi / 2
                # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                # bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
                # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # # print("Flow in bgr:", bgr.shape)
                # # plt.imshow(bgr, vmin=data_to_vis.min(), vmax=data_to_vis.max())
                # plt.imshow(bgr)

                # bgr = flow2rgb(flow_gt[round(index)].transpose((1,2,0)))
                # plt.imshow(bgr)

                data_range = min(original_data.shape[0], interpol_data.shape[0], flow_u_gt.shape[0])
                # print(data_range)
                if round(index) >= data_range:
                    break
                index_pred += 3
                # print("flow gt, index_pred", index_pred)
        if int((i-1)/columns) == 4 and round(index) % factor != 0: # flow pred; only for interpolated!
            # print("index_pred", index_pred)
            u = flow_u[round(index_pred)]
            v = flow_v[round(index_pred)]
            norm = np.sqrt(u * u + v * v) # / norm_coeff
            # print(norm.mean())
            # img = interpol_data[round(index)] # [index-columns*2*2*2]
            img = np.zeros((original_data.shape[1], original_data.shape[2]))
            plt.axis('off')
            # ax = plt.gca()
            # print("u is in range %f to %f" % (np.min(u), np.max(u)))
            # print("v is in range %f to %f" % (np.min(v), np.max(v)))
            # print("u at 300, 25:", u[25, 300])
            # print("u at 300, 75:", u[75, 300])
            # print("norm is in range %f to %f" % (np.min(norm), np.max(norm)))
            pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale = scale, bg_cmap='gray', step=quiver_steps)
            # pyimof.display.quiver(u, u, c=u, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
            # pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray')
            # pyimof.display.plot(u, v, ax=ax, colorwheel=True) # rgb with colormap. correct direction!
            # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')

            # hsv = np.empty(shape=(flow_u.shape[1], flow_u.shape[2], 3), dtype=np.uint8)
            # hsv[:,:,1] = 255
            # mag, ang = cv2.cartToPolar(flow_u[round(index)], flow_v[round(index)])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            # bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
            # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # # print("Flow in bgr:", bgr.shape)
            # # plt.imshow(bgr, vmin=data_to_vis.min(), vmax=data_to_vis.max())
            # plt.imshow(bgr)

            # bgr = flow2rgb(flow[round(index)].transpose((1,2,0)))
            # plt.imshow(bgr)

            data_range = min(original_data.shape[0], interpol_data.shape[0], flow_u.shape[0])
            # print(data_range)
            if round(index) >= data_range:
                break
            if dataset == "droplet2d":
                index_pred += 1
            if dataset == "lbs2d":
                index_pred += 3
            # print("flow pred, index_pred", index_pred)
        if int((i-1)/columns) == 5 and round(index) % factor != 0 and (dataset == "lbs2d" or "3d" in dataset): # flow diff vec
        # if int((i-1)/columns) == 5 and round(index) % factor != 0 and dataset == "lbs2d": # flow diff vec
            u_diff = flow_u_diff[round(index_pred)]
            v_diff = flow_v_diff[round(index_pred)]
            norm_diff = np.sqrt(u_diff * u_diff + v_diff * v_diff) # / norm_coeff
            # print("u_diff is in range %f to %f" % (np.min(u_diff), np.max(u_diff)))
            # print("u_diff is in range %f to %f" % (np.min(u_diff), np.max(u_diff)))
            # print("v_diff is in range %f to %f" % (np.min(v_diff), np.max(v_diff)))
            # print("norm_diff is in range %f to %f" % (np.min(norm_diff), np.max(norm_diff)))
            # img = interpol_data[round(index)] # [index-columns*2*2*2]
            img = np.zeros((original_data.shape[1], original_data.shape[2]))
            plt.axis('off')
            # ax = plt.gca()
            if "3d" in dataset:
                # max_flow_gt = diffs_flow.max() 
                scale = 0.1
            pyimof.display.quiver(u_diff, v_diff, c=norm_diff, bg=img, ax=ax, clim=(0, max_flow_gt), cmap=cmap, scale = scale, bg_cmap='gray', step=quiver_steps)
            # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
            data_range = min(original_data.shape[0], interpol_data.shape[0], u_diff.shape[0])
            # print(data_range)

            if round(index) >= data_range:
                break

            index_pred += 3

        # if dataset != "vimeo2d":
        #     # if round(index) % factor == 0 and int((i-1)/columns) == 0:
        #     if round(index - 1) % 3 == 0:
        #         # add bb to the gt image
        #         width = data_to_vis.shape[2]
        #         height = data_to_vis.shape[1]
        #         linewidth = 0.5
        #         # if "cylinder2d" in dataset:
        #         #     linewidth = 1
        #         rect = patches.Rectangle((0, 0), width-linewidth, height-linewidth, linewidth=linewidth, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)

        plt.axis('off')
        # cmap='viridis'
        if "rectangle2d" in dataset:
            cmap = "gray"
        # plt.imshow(img, cmap=cmap, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        # if int((i-1)/columns) == 0 or int((i-1)/columns) == 1 or int((i-1)/columns) == 2:
            # plt.imshow(img, cmap=cmap, vmin=0., vmax=1.)
            # plt.imshow(img, cmap=cmap, vmin=data_to_vis.min(), vmax=data_to_vis.max())

        index += skip
        if (i-1) % columns == 0:
            index_pred = 0
            # print("is an integer", index_pred)
            # input("int")

        # index += factor / math.log2(factor) # skip = factor / exp  128/7
        # print(round(index))
        # if round(index) == round(columns * (factor / math.log2(factor))):
        if round(index) == round(columns * skip):
            index = 0

    if dataset == "lbs2d" or "3d" in dataset:
        # each image has a different vmin and vmax
        # cmap=cm.get_cmap('viridis')
        # colorbar for density
        # [left, bottom, width, height]
        normalizer=Normalize(vmin=vmin, vmax=data_to_vis.max())
        im=cm.ScalarMappable(norm=normalizer, cmap='turbo')
        if "lbs2d" in dataset:
            # cb_ax = fig.add_axes([0.91, 0.5, 0.005, 0.3])
            # cb_ax = fig.add_axes([0.82, 0.5, 0.005, 0.35]) # 9 cols
            # cb_ax = fig.add_axes([0.75, 0.5, 0.005, 0.35]) 
            cb_ax = fig.add_axes([0.75, 0.59, 0.005, 0.28]) 
        elif "rectangle2d" in dataset:
            cb_ax = fig.add_axes([0.75, 0.5, 0.005, 0.4])
        elif "3d" in dataset:
            cb_ax = fig.add_axes([0.68, 0.57, 0.003, 0.3]) # 3d
        # cb.ax.tick_params(labelsize=font_size)
        fmt = lambda x, pos: '{:0.1f}'.format(x)
        bar = np.linspace(vmin, data_to_vis.max(), 5, endpoint=True)
        plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))
        # colorbar for flow
        normalizer=Normalize(vmin=0., vmax=max_flow_gt)
        if "3d" in dataset:
            normalizer=Normalize(vmin=min_flow_gt, vmax=max_flow_gt)
        im=cm.ScalarMappable(norm=normalizer, cmap='turbo')
        if "lbs2d" in dataset:
            # cb_ax = fig.add_axes([0.91, 0.2, 0.005, 0.25])
            # cb_ax = fig.add_axes([0.82, 0.15, 0.005, 0.25]) # 9 cols
            # cb_ax = fig.add_axes([0.75, 0.15, 0.005, 0.25])
            cb_ax = fig.add_axes([0.75, 0.23, 0.005, 0.3])
        elif "rectangle2d" in dataset:
            cb_ax = fig.add_axes([0.75, 0.12, 0.005, 0.3])
        elif "3d" in dataset:
            cb_ax = fig.add_axes([0.68, 0.22, 0.003, 0.3]) # 3d
        bar = np.linspace(0., max_flow_gt, 4, endpoint=True)
        if "3d" in dataset:
            bar = np.linspace(min_flow_gt, max_flow_gt, 4, endpoint=True)
        fmt = lambda x, pos: '{:0.2f}'.format(x)
        plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))

    if dataset == "droplet2d":
        # colorbar for density
        normalizer=Normalize(vmin=vmin, vmax=data_to_vis.max())
        im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
        cb_ax = fig.add_axes([0.82, 0.5, 0.005, 0.35])
        fmt = lambda x, pos: '{:0.1f}'.format(x)
        bar = np.linspace(vmin, data_to_vis.max(), 5, endpoint=True)
        plt.colorbar(im, cax=cb_ax, cmap=cmap, ticks=bar, format=FuncFormatter(fmt))
        # colorbar for flow
        normalizer=Normalize(vmin=0., vmax=max_flow_gt)
        im=cm.ScalarMappable(norm=normalizer, cmap=cmap)
        cb_ax = fig.add_axes([0.82, 0.15, 0.005, 0.25])
        bar = np.linspace(0., max_flow_gt, 4, endpoint=True)
        fmt = lambda x, pos: '{:0.2f}'.format(x)
        plt.colorbar(im, cax=cb_ax, cmap=cmap, ticks=bar, format=FuncFormatter(fmt))

    fig = plt.gcf()
    suptitle = dataset + "_" + str(factor) + "x"
    # plt.suptitle(suptitle) 
    # print(matplotlib.get_backend())
    fig.set_size_inches(20, 3)
    # wspace is x; hspace is y 
    if "droplet2d" in dataset:
        plt.subplots_adjust(wspace=-0.9, hspace=0.01)
        x = 0.16
        y = 0.8
        step = (y - 0.15) / 6.
    elif "FluidSimML2d" in dataset:
        plt.subplots_adjust(wspace=-0.21, hspace=-0.01)
    elif "pipedcylinder2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.75)
        x = 0.075
        y = 0.7
        step = (y - 0.25) / 6.
    elif "cylinder2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.945)
        x = 0.075
        y = 0.7
        step = (y - 0.25) / 6.
    elif "rectangle2d" in dataset:
        plt.subplots_adjust(wspace=-0.95, hspace=0.01)
        x = 0.25
        y = 0.8
        step = (y - 0.15) / 6.
    elif "lbs2d" in dataset:
        # plt.subplots_adjust(wspace=0.01, hspace=-0.75) # hspace=-0.9
        # plt.subplots_adjust(wspace=-0.73, hspace=0.02) # 9 cols
        plt.subplots_adjust(wspace=-0.825, hspace=0.02) # 7 cols
        # x = 0.075
        # y = 0.7
        # step = (y - 0.25) / 6.
        x = 0.20 # 0.16 # 9 cols
        y = 0.8
        step = (y - 0.15) / 6.
    elif "vimeo2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        x = 0.075
        y = 0.8
        step = (y - 0.15) / 6.
    elif "3d" in dataset:
        plt.subplots_adjust(wspace=-0.963, hspace=0.01)
        x = 0.3
        y = 0.8
        step = (y - 0.15) / 6.

    font_size = 13
    # if "droplet2d" in dataset:
    #     font_size = 10
    fig.text(x, y, 'Dens GT', size=font_size)
    fig.text(x, y - step, 'Dens pred', size=font_size)
    fig.text(x, y - step*2, 'Dens diff', size=font_size)
    # fig.text(x, y - step*3, 'Dens mask', size=font_size)
    if "droplet2d" in dataset or "vimeo2d" in dataset:
        fig.text(x, y - step*3, 'Flow pred HSV', size=font_size)
    else:
        fig.text(x, y - step*3, 'Flow GT', size=font_size)
    fig.text(x, y - step*4, 'Flow pred', size=font_size)
    if dataset != "droplet2d" and dataset != "vimeo2d":
        fig.text(x, y - step*5, 'Flow diff', size=font_size)

    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

def visualize_large_3d(original_data, interpol_data, diffs, \
        flow, factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    # print(original_data.shape, interpol_data.shape, diffs.shape, flow.shape, flow_gt.shape, diffs_flow.shape, u_diff.shape, v_diff.shape)
    fig = plt.figure(figsize=(30, 10))
    columns = 20
    if "droplet3d" in dataset or "rectangle3d" in dataset:
        columns = 24
    quiver_steps = 5
    rows = 2
    skip = 1
    if factor == 8:
        skip = 2
    if factor >= 16: # or factor == 32 or factor == 64:
        skip = 4
    # if factor == 128:
    #     skip = 8

    print(flow.shape)
    # input("x")

    data_to_vis = original_data
    flow_u = flow[:, 0, ...]
    flow_v = flow[:, 1, ...]
    # flow_u_gt = flow_gt[:, 0, ...]
    # flow_v_gt = flow_gt[:, 1, ...]
    # input("x")

    index = 0
    for i in range(1, columns*rows+1):
        # index = (i-1)*2 # skip eaach second - do it defferently
        # if (index >= data_to_vis.shape[0]):
        #     break
        ax = fig.add_subplot(rows, columns, i)
        # print(int(i/columns))
        if int((i-1)/columns) == 0: # gt
            img = original_data[round(index)]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 1: # interpol
            # img = data_to_vis[index+int(data_to_vis.shape[0]/3),...]
            img = interpol_data[round(index)] # [index-columns*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        # if int((i-1)/columns) == 2: # diffs
        #     # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
        #     img = diffs[round(index)] # [index-columns*2*2,...]
        #     plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        # if int((i-1)/columns) == 2: # flow pred
        #     u = flow_u[round(index)]
        #     v = flow_v[round(index)]
        #     norm = np.sqrt(u * u + v * v)
        #     img = interpol_data[round(index)] # [index-columns*2*2*2]
        #     plt.axis('off')
        #     ax = plt.gca()
        #     pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
        #     # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
        #     data_range = min(original_data.shape[0], interpol_data.shape[0], flow_u.shape[0])
        #     # print(data_range)
        #     if round(index) >= data_range:
        #         break

        if round(index) % factor == 0 and int((i-1)/columns) == 0:
            # add bb to the gt image
            width = data_to_vis.shape[2]
            height = data_to_vis.shape[1]
            linewidth = 2
            # if "cylinder2d" in dataset:
            #     linewidth = 1
            rect = patches.Rectangle((0, 0), width+linewidth, height+linewidth, linewidth=linewidth, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.axis('off')
        cmap='viridis'
        # if "rectangle2d" in dataset:
        #     cmap = "gray"
        # plt.imshow(img, cmap=cmap, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        # if int((i-1)/columns) == 0 or int((i-1)/columns) == 1 or int((i-1)/columns) == 2:
            # plt.imshow(img, cmap=cmap, vmin=0., vmax=1.)
            # plt.imshow(img, cmap=cmap, vmin=data_to_vis.min(), vmax=data_to_vis.max())

        # index += 2
        index += skip
        # index += factor / math.log2(factor) # skip = factor / exp  128/7
        # print(round(index))
        # if round(index) == round(columns * (factor / math.log2(factor))):
        if round(index) == round(columns * skip):
            index = 0

    fig = plt.gcf()
    plt.suptitle(title) 
    # print(matplotlib.get_backend())
    fig.set_size_inches(20, 3)
    # if "droplet2d" in dataset:
    #     plt.subplots_adjust(wspace=0.01, hspace=-0.69)
    # elif "FluidSimML2d" in dataset:
    #     plt.subplots_adjust(wspace=0.01, hspace=-0.15)
    # elif "pipedcylinder2d" in dataset:
    #     plt.subplots_adjust(wspace=0.01, hspace=-0.75)
    # elif "cylinder2d" in dataset:
    #     plt.subplots_adjust(wspace=0.01, hspace=-0.945)
    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)


def visualize_large_3d(original_vols, interpol_vols, diffs, 
                flow, flow_gt, diffs_flow, u_diff, v_diff, mask,
                factor, dataset, dir_model="Results", title="Volumes", show=True, save=False):
    # flow_u = np.ones((1, 64, 64, 64))
    # flow_v = np.ones((1, 64, 64, 64))
    # flow_w = np.ones((1, 64, 64, 64))
    # flow_u = flow[10, 0, ...]
    # flow_v = flow[10, 1, ...]
    # flow_w = flow[10, 2, ...]
    # print("D")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # Make the grid
    # x, y, z = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 64, 1), np.arange(0, 64, 1))
    # print("D")
    # # Make the direction data for the arrows
    # u = flow_u
    # v = flow_v
    # w = flow_w
    # ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
    # title = "droplet3d_flow3d.png"
    # fig.savefig(os.path.join(dir_res, title), dpi = 100)
    # input("x")

    X, Y, Z = np.mgrid[0:original_vols.shape[1], 0:original_vols.shape[2], 0:original_vols.shape[3]]
    values = original_vols[0]
    vol = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        opacity=0.1,
        surface_count=15,
        ))
    # fig.show()
    # title = "tangaroa3d.html" # "droplet3d.html"
    # image_path = os.path.join(dir_res, title)
    # vol.write_html(image_path) 
    # input("x")

    fig = plt.figure(figsize=(30, 10))
    columns = 10
    rows = 3
    step = 1

    print(original_vols.shape, interpol_vols.shape)
    # input("x")
    data_to_vis = original_vols

    flow_u = flow[:, 0, ...]
    flow_v = flow[:, 1, ...]
    flow_w = flow[:, 2, ...]

    X, Y, Z = np.mgrid[0:data_to_vis.shape[1], 0:data_to_vis.shape[2], 0:data_to_vis.shape[3]]
    index = 0
    for i in range(1, columns*rows+1):
        ax = fig.add_subplot(rows, columns, i)
        # print(int(i/columns))
        if int((i-1)/columns) == 0: # gt
            values = original_vols[round(index)]
            vol = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                opacity=0.1,
                surface_count=15,
                ))
            img = plotly_fig2array(vol)
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
            print("0")
        if int((i-1)/columns) == 1: # interpol
            values = interpol_vols[round(index)]
            vol = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                opacity=0.1,
                surface_count=15,
                ))
            img = plotly_fig2array(vol)
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
            print("1")
        if int((i-1)/columns) == 2: # mask
            values = mask[round(index)]
            vol = go.Figure(data=go.Volume(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=values.flatten(),
                opacity=0.1,
                surface_count=15,
                ))
            img = plotly_fig2array(vol)
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
            print("2")
        if int((i-1)/columns) == 3: # flow
            ax = plt.figure().add_subplot(projection='3d')
            # Make the grid
            x, y, z = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 64, 1), np.arange(0, 64, 1))
            # Make the direction data for the arrows
            u = flow_u
            v = flow_v
            w = flow_w
            ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
            print("3")

        plt.axis('off')
        index += step
        if round(index) == round(columns * step):
            index = 0

    fig = plt.gcf()
    plt.suptitle(title) 
    fig.set_size_inches(20, 3)
    if "droplet3d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.1)
    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)
        print("Figure saved")

    # # fig.show()
    # image_path = os.path.join(dir_res, title)
    # fig.write_image(image_path) 

def render_volume_yt(volume, save_path, file_name):

    ds_numpy = yt.load_uniform_grid(dict(density=volume), volume.shape)
    sc = yt.create_scene(ds_numpy, field=("gas", "density"))
    source = sc[0]

    source.set_log(True)
    # bounds = (1e6, 1e15)
    # tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
    # tf.sample_colormap(np.log10(1e7), w=0.005, alpha=1e1, col_bounds=np.log10((1e7, 1e10)), colormap="turbo")
    # tf.sample_colormap(np.log10(1e8), w=0.005, alpha=1e1, col_bounds=np.log10((1e7, 1e10)), colormap="turbo")
    # tf.sample_colormap(np.log10(1e9), w=0.005, alpha=1e1, col_bounds=np.log10((1e7, 1e10)), colormap="turbo")
    # tf.sample_colormap(np.log10(1e10), w=0.005, alpha=1e1, col_bounds=np.log10((1e7, 1e10)), colormap="turbo")

    bounds = (0.1, 0.7)
    tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
    tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
    tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
    tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
    tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

    source.tfh.tf = tf
    source.tfh.bounds = bounds

    cam = sc.camera
    cam.resolution = 256
    # cam.zoom(1.75)
    cam.zoom(1.0)
    # cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
    im = sc.render()
    im.write_png(os.path.join(save_path, file_name), background=[0.0, 0.0, 0.0, 1.0], sigma_clip=4.0) # 2.0
    print("rendered!")
    # omega - overall brightness, to see more details

# Ensure off-screen rendering is possible
pv.start_xvfb()

# Function to visualize volume with streamlines and save the output image
"""
def visualize_volume_with_streamlines(volume, u, v, w, seed_points, output_path):
    dims = volume.shape

    # Create the grid coordinates
    x = np.linspace(0, 1, dims[0])
    y = np.linspace(0, 1, dims[1])
    z = np.linspace(0, 1, dims[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Create a PyVista grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = dims

    # Flatten data to fit grid point format
    grid['density'] = volume.flatten(order='F')
    grid['velocity'] = np.c_[u.flatten(order='F'), v.flatten(order='F'), w.flatten(order='F')]

    # Create seed points on the bottom surface of the cube
    n_points = 5
    x_seed = np.linspace(0.1, 0.9, int(np.sqrt(n_points)))
    y_seed = np.linspace(0.1, 0.9, int(np.sqrt(n_points)))
    xx_seed, yy_seed = np.meshgrid(x_seed, y_seed)
    zz_seed = np.zeros(xx_seed.shape)  # Bottom plane
    seed_points = np.c_[xx_seed.ravel(), yy_seed.ravel(), zz_seed.ravel()]

    # Initialize an empty MultiBlock to store streamlines
    streamline_collection = pv.MultiBlock()

    # Generate streamlines for each seed point
    for seed in seed_points:
        streamline = grid.streamlines(
            vectors='velocity',
            source_center=seed,
            source_radius=0.025,  # Small radius to generate multiple streamlines around the seed
            max_time=1.0,  # Adjusted length of the streamlines
            initial_step_length=0.1,
            integration_direction='forward'  # Integrate in the forward direction only
        )
        
        # Ensure proper distribution of time values for coloring the streamlines
        if streamline.n_points > 0:
            time_scalars = np.linspace(0, 1, streamline.n_points)
            streamline['time'] = time_scalars
            streamline_collection.append(streamline)

    # Combine all streamlines into one PolyData for visualization
    combined_streamlines = streamline_collection.combine().extract_surface().tube(radius=0.005)

    # Create a plotter
    plotter = pv.Plotter(off_screen=True)  # Use off_screen=True for batch rendering

    # Add the streamlines with time-based coloring
    plotter.add_mesh(combined_streamlines, scalars='time', cmap='turbo')

    # Add bounding box
    plotter.add_bounding_box(color="black")

    # Set the camera position
    plotter.camera_position = 'iso'

    # Render and save the image
    plotter.show(screenshot=output_path)
    print(f"Streamlines rendered and saved to {output_path}")
"""

# """
def visualize_volume_with_streamlines(volume, u, v, w, output_path):
    dims = volume.shape

    # Create the grid coordinates
    x = np.linspace(0, 1, dims[0])
    y = np.linspace(0, 1, dims[1])
    z = np.linspace(0, 1, dims[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Create a PyVista grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = dims

    # Flatten data to fit grid point format
    grid['velocity'] = np.c_[u.flatten(order='F'), v.flatten(order='F'), w.flatten(order='F')]

    # Create seed points on the bottom surface of the cube
    n_points = 9
    grid_size = int(np.sqrt(n_points))
    x_seed = np.linspace(0.1, 0.9, grid_size)
    y_seed = np.linspace(0.1, 0.9, grid_size)
    xx_seed, yy_seed = np.meshgrid(x_seed, y_seed)
    zz_seed = np.full(xx_seed.shape, 0.5)
    seed_points = np.c_[xx_seed.ravel(), yy_seed.ravel(), zz_seed.ravel()]

    # Create a visible surface at the seed points' plane
    seed_surface = pv.StructuredGrid()
    seed_surface.points = np.c_[xx_seed.ravel(), yy_seed.ravel(), zz_seed.ravel()]
    seed_surface.dimensions = (grid_size, grid_size, 1)

    # Initialize an empty MultiBlock to store streamlines
    streamline_collection = pv.MultiBlock()

    # Generate streamlines for each seed point
    for seed in seed_points:
        streamline = grid.streamlines(
            vectors='velocity',
            source_center=seed,
            max_time=0.75,  # Adjusted length of the streamlines
            initial_step_length=0.05,
            n_points = 40,
            integration_direction='both',  # Integrate in both forward and backward directions
        )
        # Ensure proper distribution of time values for coloring the streamlines
        if streamline.n_points > 0:
            # time_scalars = np.linspace(0, 1, streamline.n_points)
            # streamline['IntegrationTime'] = time_scalars
            streamline_collection.append(streamline)

    # Combine all streamlines into one PolyData for visualization
    combined_streamlines = streamline_collection.combine().extract_surface().tube(radius=0.005)

    # Create a plotter
    plotter = pv.Plotter(off_screen=True)  # Use off_screen=True for batch rendering

    # Add the streamlines with time-based coloring
    sargs = dict(vertical=True, title_font_size=16)
    plotter.add_mesh(
        combined_streamlines,
        scalars='IntegrationTime',
        cmap='turbo',
        clim=[0, 10],
        scalar_bar_args=sargs,
    )

    # Add the seed surface
    # plotter.add_mesh(seed_surface, color='white', opacity=0.5, show_edges=True)

    # Add bounding box
    plotter.add_bounding_box(color="black")

    # Set the camera position
    plotter.camera_position = 'iso'

    # Render and save the image
    plotter.show(screenshot=output_path)
    print(f"Streamlines rendered and saved to {output_path}")
# """

"""
def visualize_volume_with_glyphs(volume, u, v, w, output_path):
    dims = volume.shape

    # Create the grid coordinates
    x = np.linspace(0, 1, dims[0])
    y = np.linspace(0, 1, dims[1])
    z = np.linspace(0, 1, dims[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    # Create a PyVista grid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = dims

    # Flatten data to fit grid point format
    grid['density'] = volume.flatten(order='F')
    grid['velocity'] = np.c_[u.flatten(order='F'), v.flatten(order='F'), w.flatten(order='F')]

    # Create a subset of seed points
    n_points = 1000  # Number of glyphs to visualize
    np.random.seed(42)
    random_indices = np.random.choice(points.shape[0], n_points, replace=False)
    seed_points = points[random_indices]

    # Create a PyVista PolyData for the seed points
    seed_polydata = pv.PolyData(seed_points)

    # Compute the vectors at the seed points
    vectors = grid['velocity'][random_indices]

    # Add the vectors to the seed PolyData
    seed_polydata['vectors'] = vectors

    # Create glyphs
    glyphs = seed_polydata.glyph(orient='vectors', scale='vectors', factor=1.0)

    # Create a plotter
    plotter = pv.Plotter(off_screen=True)  # Use off_screen=True for batch rendering

    # Add the glyphs
    plotter.add_mesh(glyphs, color='white')

    # Add bounding box
    plotter.add_bounding_box(color="black")

    # Set the camera position
    plotter.camera_position = 'iso'

    # Render and save the image
    plotter.show(screenshot=output_path)
    print(f"Glyphs rendered and saved to {output_path}")
"""

"""
# Define the function to generate streamlines
def generate_streamlines(ds, velocity_field, seed_points):
    streamlines = Streamlines(ds, seed_points, velocity_field[0], velocity_field[1], velocity_field[2])
    streamlines.integrate_through_volume()
    return streamlines

np.random.seed(10)
num_seed_points = 30
# print("num_seed_points", num_seed_points)
seed_points = np.random.rand(num_seed_points, 3)
"""

def visualize_seq_dens_3d(original_data, interpol_data, flow_gt, flow,
                          dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    
    columns = 6 # 2 6
    rows = 4 # 4 6
    skip = 10
    data_to_vis = original_data
    index = 0
    shift = 0

    if "3d" in dataset:
        vmin = 0
        print("vis range:", vmin, data_to_vis.max())
        hspace = 0.01
        wspace = -0.7
        cmap ='gray'

    u_gt = flow_gt[:, 0, ...] 
    v_gt = flow_gt[:, 1, ...] 
    w_gt = flow_gt[:, 2, ...] 
    flow_norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt + w_gt * w_gt)

    u = flow[:, 0, ...] 
    v = flow[:, 1, ...] 
    w = flow[:, 2, ...] 
    flow_norm = np.sqrt(u * u + v * v + w * w)

    # wspace is x; hspace is y
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(20, 8), gridspec_kw={'hspace': hspace, 'wspace': wspace})

    # bounds = (105.0, 160.0)
    # tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
    # tf.add_gaussian(np.log10(1e2), width=2e-4, height=[0.0, 0.25, 1.0, 1e3])

    if "jets3d" in dataset:
        bounds = (0.1, 0.9)
        # 100-200 t:
        # density
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf.add_gaussian(np.log10(0.3), width=1e-4, height=[1.0, 0.1, 0.3, 1e2]) # red 100-200
        # tf.add_gaussian(np.log10(0.4), width=4e-4, height=[0.25, 0.0, 1.0, 1e2]) # blue 100-200
        # tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.25, 0.0, 1.0, 1e2]) # blue 700-800
        tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.05, 0.1, 1.0, 1e2]) # blue 700-800
        # velocity
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.5), width=1e-4, height=[0.2, 0.8, 0.2, 1e2]) # green 100-200
        # tf_vel.add_gaussian(np.log10(0.7), width=4e-4, height=[0.8, 0.6, 0.1, 1e2]) # orange 100-200
        # tf_vel.add_gaussian(np.log10(0.7), width=2e-4, height=[0.2, 0.8, 0.2, 1e2]) # green 700-800
        tf_vel.add_gaussian(np.log10(0.55), width=1e-4, height=[1.0, 0.6, 0.1, 1e2]) # orange 700-800
    elif "nyx3d" in dataset:
        bounds = (0.1, 0.7)
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # velocity
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

    # print("min, max: ", np.min(original_data), np.max(original_data))
    # print("mean: ", np.mean(original_data))
    # print("std: ", np.std(original_data))

    cam_resolution = 256

    def add_zoom(ax):
        # Define the coordinates and size of the region to be zoomed in
        x1, y1, width1, height1 = 55, 120, 40, 40  # For the top left zoomed-in version
        x2, y2, width2, height2 = 108, 80, 40, 40  # For the top right zoomed-in version

        # Create a rectangle patch to highlight the region for the top right zoomed-in version
        rect1 = patches.Rectangle((x1, y1), width1, height1, linewidth=1.5, edgecolor='purple', facecolor='none')
        ax.add_patch(rect1)

        # Create a rectangle patch to highlight the region for the top left zoomed-in version
        rect2 = patches.Rectangle((x2, y2), width2, height2, linewidth=1.5, edgecolor='orange', facecolor='none')
        ax.add_patch(rect2)

        # Create inset axes for the zoomed-in version on the top right
        ax_zoom1 = inset_axes(ax, width="35%", height="35%", loc='upper left')
        ax_zoom1.imshow(img[y1:y1+height1, x1:x1+width1], vmin=0, vmax=1.0, cmap='turbo')
        ax_zoom1.axis('off')

        # Create inset axes for the zoomed-in version on the top left
        ax_zoom2 = inset_axes(ax, width="35%", height="35%", loc='upper right')
        ax_zoom2.imshow(img[y2:y2+height2, x2:x2+width2], vmin=0, vmax=1.0, cmap='turbo')
        ax_zoom2.axis('off')

        # Define the coordinates and size of the frames to match the zoomed-in plots
        frame_width, frame_height = width1 * 0.98, height1 * 0.98  # Adjust the scale factor as needed

        frame1 = patches.Rectangle((0, 0), frame_width, frame_height, linewidth=1.5, edgecolor='purple', facecolor='none')
        frame2 = patches.Rectangle((0, 0), frame_width, frame_height, linewidth=1.5, edgecolor='orange', facecolor='none')

        ax_zoom1.add_patch(frame1)
        ax_zoom2.add_patch(frame2)

    # Initialize lists to store streamline data
    streamlines_data_gt = []
    streamlines_data = []

    # Initialize lists to store mean of closest points for averaging later
    mean_closest_points_gt_list = []
    mean_closest_points_list = []

    # Define reference point (example)
    reference_point = np.array([0.5, 0.5, 0.5])

    for i in range(columns):
        volume = original_data[round(index)]
        # print("volume", volume)
        ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds, field=("gas", "density"))
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "dens_gt_tf.png"), profile_field=("gas", "density"))

        cam = sc.camera
        cam.resolution = cam_resolution
        if "jets3d" in dataset:
            cam.zoom(1.75)
        elif "nyx3d" in dataset:
            cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        # print("im", im)
        im.write_png(os.path.join(dir_res, "Dens_seq.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        # First row: original data
        ax = axs[0][i]
        # img = original_data[round(index)]
        img = plt.imread(os.path.join(dir_res, "Dens_seq.png"))
        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap) # data_to_vis
        if dataset == "jets3d":
            ax_title = "t=" + str(index + shift + 730)
            ax.set_title(ax_title, fontsize=12, pad=10)
        elif dataset == "pool3d":
            ax_title = "t=" + str(index + shift + 30)
            ax.set_title(ax_title, fontsize=12, pad=10)
        elif dataset == "nyx3d":
            ax_title = "t=" + str(5 * 10 * shift + 300)
            shift += 1
            ax.set_title(ax_title, fontsize=12, pad=10)
        ax.axis('off')

        # add zoomings
        if "jetsd" in dataset:
            add_zoom(ax)


        # Second row: interpolated data
        volume = interpol_data[round(index)]
        # print("volume", volume)
        ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds, field=("gas", "density"))
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "dens_tf.png"), profile_field=("gas", "density"))

        cam = sc.camera
        cam.resolution = cam_resolution
        if "jets3d" in dataset:
            cam.zoom(1.75)
        elif "nyx3d" in dataset:
            cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        # print("im", im)
        im.write_png(os.path.join(dir_res, "Dens_seq.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ax = axs[1][i]
        # img = interpol_data[round(index)]
        img = plt.imread(os.path.join(dir_res, "Dens_seq.png"))
        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # add zoomings
        if "jetsd" in dataset:
            add_zoom(ax)


        # Third row: flow gt
        volume = flow_norm_gt[round(index)]
        # print("volume", volume)
        ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds, field=("gas", "density"))
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_gt_tf.png"), profile_field=("gas", "density"))

        # # Generate streamlines
        # streamlines = generate_streamlines(ds, ("velocity_x", "velocity_y", "velocity_z"), seed_points)
        # # streamlines_data_gt.append(streamlines)

        # # Calculate and accumulate mean of closest points for ground truth
        # mean_closest_point_gt = calculate_mean_of_closest_points_streamline(reference_point, [sl for sl in streamlines])
        # mean_closest_points_gt_list.append(mean_closest_point_gt)
        # print(f"Mean of closest points to {reference_point} in GT: {mean_closest_point_gt}")

        cam = sc.camera
        cam.resolution = cam_resolution
        if "jets3d" in dataset:
            cam.zoom(1.75)
        elif "nyx3d" in dataset:
            cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ax = axs[2][i]
        # img = interpol_data[round(index)]
        img = plt.imread(os.path.join(dir_res, "Flow.png"))
        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')


        # Fourth row: flow
        volume = flow_norm[round(index)]
        # print("volume", volume)
        ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds, field=("gas", "density"))
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_tf.png"), profile_field=("gas", "density"))

        # # Generate streamlines
        # # streamlines = generate_streamlines(ds, ("gas", "velocity"), seed_points)
        # streamlines = generate_streamlines(ds, ("velocity_x", "velocity_y", "velocity_z"), seed_points)
        # # streamlines_data.append(streamlines)

        # # Calculate and accumulate mean of closest points for flow
        # mean_closest_point = calculate_mean_of_closest_points_streamline(reference_point, [sl for sl in streamlines])
        # mean_closest_points_list.append(mean_closest_point)
        # print(f"Mean of closest points to {reference_point} in flow: {mean_closest_point}")

        cam = sc.camera
        cam.resolution = cam_resolution
        if "jets3d" in dataset:
            cam.zoom(1.75)
        elif "nyx3d" in dataset:
            cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ax = axs[3][i]
        # img = interpol_data[round(index)]
        img = plt.imread(os.path.join(dir_res, "Flow.png"))
        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')


        # volume = original_data[round(index)]
        # ds = yt.load_uniform_grid({
        #     'velocity_x': u[i],
        #     'velocity_y': v[i],
        #     'velocity_z': w[i],
        #     'density': volume
        # }, volume.shape)
        # sc = yt.create_scene(ds, field=("gas", "density"))
        # source = sc[0]
        # source.set_log(True)
        # source.tfh.tf = tf # yt.ColorTransferFunction((0.1, 1.0))  # Example transfer function
        # source.tfh.bounds = bounds
        # source.tfh.plot(os.path.join(dir_res, "flow_tf.png"), profile_field=("gas", "density"))

        # streamlines = generate_streamlines(ds, ("velocity_x", "velocity_y", "velocity_z"), seed_points)
        # # streamlines_data.append(streamlines_flow)

        # cam = sc.camera
        # cam.resolution = cam_resolution
        # cam.zoom(1.75 if "jets3d" in dataset else 1.0)
        # cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        # im = sc.render()
        # im.write_png(os.path.join(dir_res, "Flow_streamlines.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        
        # mean_closest_point_flow = calculate_mean_of_closest_points_streamline(reference_point, streamlines_flow)
        # mean_closest_points_list.append(mean_closest_point_flow)

        # for streamline in streamlines.streamlines:
        #     if len(streamline) > 1:
        #         ax.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], color='red', alpha=0.7)

        """
        # Fifth row: flow gt streamlines
        output_path = os.path.join(dir_res, "Flow_streamlines.png")
        volume = original_data[round(index)]
        # Call the function to create and save the image
        # visualize_volume_with_streamlines(volume, u_gt[i], v_gt[i], w_gt[i], seed_points, output_path)
        visualize_volume_with_streamlines(volume, u_gt[i], v_gt[i], w_gt[i], output_path)
        # visualize_volume_with_glyphs(volume, u_gt[i], v_gt[i], w_gt[i], output_path)
        print("rendered!")

        ax = axs[4][i]
        # img = interpol_data[round(index)]
        img = plt.imread(output_path)

        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')


        # Sixth row: flow streamlines
        output_path = os.path.join(dir_res, "Flow_streamlines.png")
        volume = original_data[round(index)]
        # Call the function to create and save the image
        # visualize_volume_with_streamlines(volume, u[i], v[i], w[i], seed_points, output_path)
        # visualize_volume_with_streamlines(volume, u[i], v[i], w[i], output_path)
        visualize_volume_with_streamlines(volume, u_gt[i], v_gt[i], w_gt[i], output_path)
        # visualize_volume_with_glyphs(volume, u[i], v[i], w[i], output_path)
        print("rendered!")

        ax = axs[5][i]
        # img = interpol_data[round(index)]
        img = plt.imread(output_path)

        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')
        """

        index += skip

    # # Calculate and accumulate mean of closest points for flow
    # mean_closest_point = calculate_mean_of_closest_points_streamline(reference_point, [sl for sl in streamlines])
    # mean_closest_points_list.append(mean_closest_point)
    # print(f"Mean of closest points to {reference_point} in flow: {mean_closest_point}")

    # # Calculate the average of the mean closest points for both ground truth and flow
    # avg_mean_closest_point_gt = np.mean(mean_closest_points_gt_list, axis=0)
    # avg_mean_closest_point = np.mean(mean_closest_points_list, axis=0)

    # print(f"Average mean of closest points to {reference_point} in GT: {avg_mean_closest_point_gt}")
    # print(f"Average mean of closest points to {reference_point} in flow: {avg_mean_closest_point}")

    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

def render_density(gt, interpol, dataset, dir_res="Results"):

    if "jets3d" in dataset:
        bounds = (0.1, 0.9)
        # 100-200 t:
        # density
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf.add_gaussian(np.log10(0.3), width=1e-4, height=[1.0, 0.1, 0.3, 1e2]) # red 100-200
        # tf.add_gaussian(np.log10(0.4), width=4e-4, height=[0.25, 0.0, 1.0, 1e2]) # blue 100-200
        # tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.25, 0.0, 1.0, 1e2]) # blue 700-800
        tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.05, 0.1, 1.0, 1e2]) # blue 700-800
        # velocity
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.5), width=1e-4, height=[0.2, 0.8, 0.2, 1e2]) # green 100-200
        # tf_vel.add_gaussian(np.log10(0.7), width=4e-4, height=[0.8, 0.6, 0.1, 1e2]) # orange 100-200
        # tf_vel.add_gaussian(np.log10(0.7), width=2e-4, height=[0.2, 0.8, 0.2, 1e2]) # green 700-800
        tf_vel.add_gaussian(np.log10(0.55), width=1e-4, height=[1.0, 0.6, 0.1, 1e2]) # orange 700-800
    elif "nyx3d" in dataset:
        bounds = (0.1, 0.7)
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # 
        tf.sample_colormap(np.log10(0.17), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.42), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.52), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # velocity
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

    # pickle_file = "data_original_interpol.pkl"
    # with open(os.path.join(dir_res, pickle_file), 'rb') as dens_file:
    #     data_test = pickle.load(dens_file)
    data_density = np.array(gt) # [:, 0]  # density
    # data_velocity = np.array(data_test)[:, 1:]  # velocity
    # print(data_velocity.shape)

    save_path = dir_res + "/Renderings_GT"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # render density
    for idx in range(len(data_density)):
        volume = data_density[idx]
        ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds, field=("gas", "density"))
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = bounds
        # source.tfh.plot(f"{save_path}/transfer_function.png", profile_field=("gas", "density"))
        cam = sc.camera
        cam.resolution = 256
        if "jets3d" in dataset:
            cam.zoom(1.75)
        elif "nyx3d" in dataset:
            cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        # cam.rotate(30 / 180 * np.pi, rot_vector=np.array([1.0, 0.0, 0.0]))
        im = sc.render()
        im.write_png(os.path.join(save_path, f'density_{700+idx}.png'), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)

    # pickle_file = "interpol_data.pkl"
    # with open(os.path.join(dir_res, pickle_file), 'rb') as dens_file:
    #     data_test = pickle.load(dens_file)
    data_density = np.array(interpol) # [:, 0]  # density
    # data_velocity = np.array(data_test)[:, 1:]  # velocity
    # print(data_velocity.shape)

    save_path = dir_res + "/Renderings_interpol"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # render density
    for idx in range(len(data_density)):
        volume = data_density[idx]
        ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        sc = yt.create_scene(ds, field=("gas", "density"))
        source = sc[0]
        source.set_log(True)
        source.tfh.tf = tf
        source.tfh.bounds = bounds
        # source.tfh.plot(f"{save_path}/transfer_function.png", profile_field=("gas", "density"))
        cam = sc.camera
        cam.resolution = 256
        if "jets3d" in dataset:
            cam.zoom(1.75)
        elif "nyx3d" in dataset:
            cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        # cam.rotate(30 / 180 * np.pi, rot_vector=np.array([1.0, 0.0, 0.0]))
        im = sc.render()
        im.write_png(os.path.join(save_path, f'density_{700+idx}.png'), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)

def render_flow(flow_gt, flow, dataset, dir_res="Results"):

    cam_resolution = 256

    u_gt = flow_gt[:, 0, ...] 
    v_gt = flow_gt[:, 1, ...] 
    w_gt = flow_gt[:, 2, ...] 
    flow_norm_gt = np.linalg.norm(flow_gt, axis=1)
    u = flow[:, 0, ...] 
    v = flow[:, 1, ...] 
    w = flow[:, 2, ...] 
    flow_norm = np.linalg.norm(flow, axis=1)

    if "jets3d" in dataset:
        bounds = (0.1, 0.9)
        # 100-200 t:
        # density
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf.add_gaussian(np.log10(0.3), width=1e-4, height=[1.0, 0.1, 0.3, 1e2]) # red 100-200
        # tf.add_gaussian(np.log10(0.4), width=4e-4, height=[0.25, 0.0, 1.0, 1e2]) # blue 100-200
        # tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.25, 0.0, 1.0, 1e2]) # blue 700-800
        tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.05, 0.1, 1.0, 1e2]) # blue 700-800
        # velocity
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.5), width=1e-4, height=[0.2, 0.8, 0.2, 1e2]) # green 100-200
        # tf_vel.add_gaussian(np.log10(0.7), width=4e-4, height=[0.8, 0.6, 0.1, 1e2]) # orange 100-200
        # tf_vel.add_gaussian(np.log10(0.7), width=2e-4, height=[0.2, 0.8, 0.2, 1e2]) # green 700-800
        tf_vel.add_gaussian(np.log10(0.55), width=1e-4, height=[1.0, 0.6, 0.1, 1e2]) # orange 700-800
    elif "nyx3d" in dataset:
        bounds = (0.1, 0.7)
        tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # tf.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # tf.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # 
        tf.sample_colormap(np.log10(0.17), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.42), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf.sample_colormap(np.log10(0.52), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        # velocity
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.sample_colormap(np.log10(0.2), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.3), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.4), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")
        tf_vel.sample_colormap(np.log10(0.5), w=0.00025, alpha=1.5e1, col_bounds=np.log10((0.2, 0.6)), colormap="turbo")

    # pickle_file = "data_original_interpol.pkl"
    # with open(os.path.join(dir_res, pickle_file), 'rb') as dens_file:
    #     data_test = pickle.load(dens_file)
    data_flow = np.array(flow_gt) # [:, 0]  # density
    # data_velocity = np.array(data_test)[:, 1:]  # velocity
    # print(data_velocity.shape)

    save_path = dir_res + "/Renderings_flow_GT"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # render density
    for idx in range(len(data_flow)):

        volume = u_gt[idx]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_x=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.15), width=0.0005, height=[1.0, 0.50, 0.05, 1e2]) # orange
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_x_gt_tf.png"), profile_field=("stream", "velocity_x"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_x.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = v_gt[idx]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_y=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_y"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.25), width=0.0005, height=[0.1, 0.8, 0.1, 1e2]) # green
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_y_gt_tf.png"), profile_field=("stream", "velocity_y"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_y.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = w_gt[idx]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_z=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_z"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.35), width=0.0005, height=[0.1, 0.1, 1.0, 1e2]) # blue
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_z_gt_tf.png"), profile_field=("stream", "velocity_z"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_z.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        img_x = plt.imread(os.path.join(dir_res, "Flow_x.png"))
        img_y = plt.imread(os.path.join(dir_res, "Flow_y.png"))
        img_z = plt.imread(os.path.join(dir_res, "Flow_z.png"))
        blended_img = (img_x + img_y + img_z) # / 3 

        # blended_img = blended_img - blended_img.min()  # Shift minimum to 0
        # blended_img = blended_img / blended_img.max()  # Scale maximum to 1
        blended_img = np.clip(blended_img, 0, 1)

        # plt.imsave(os.path.join(save_path, f'flow_{300+idx}.png'), blended_img)
        plt.imsave(os.path.join(save_path, f'flow_{300+idx}.png'), blended_img, vmin=0., vmax=1.)
        
        # blended_img = sc.render(blended_img)
        # blended_img.write_png(os.path.join(save_path, f'flow_{300+idx}.png'), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)


    data_flow = np.array(flow)

    save_path = dir_res + "/Renderings_flow"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # render density
    for idx in range(len(data_flow)):

        volume = u[idx]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_x=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.16), width=0.0005, height=[1.0, 0.50, 0.05, 1e2]) # orange
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_x_gt_tf.png"), profile_field=("stream", "velocity_x"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_x.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = v[idx]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_y=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_y"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.26), width=0.0005, height=[0.1, 0.8, 0.1, 1e2]) # green
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_y_gt_tf.png"), profile_field=("stream", "velocity_y"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_y.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = w[idx]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_z=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_z"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.36), width=0.0005, height=[0.1, 0.1, 1.0, 1e2]) # blue
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_z_gt_tf.png"), profile_field=("stream", "velocity_z"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.0)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_z.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        img_x = plt.imread(os.path.join(dir_res, "Flow_x.png"))
        img_y = plt.imread(os.path.join(dir_res, "Flow_y.png"))
        img_z = plt.imread(os.path.join(dir_res, "Flow_z.png"))
        blended_img = (img_x + img_y + img_z) # / 3 

        # blended_img = blended_img - blended_img.min()  # Shift minimum to 0
        # blended_img = blended_img / blended_img.max()  # Scale maximum to 1
        blended_img = np.clip(blended_img, 0, 1)

        # plt.imsave(os.path.join(save_path, f'flow_{300+idx}.png'), blended_img)
        plt.imsave(os.path.join(save_path, f'flow_{300+idx}.png'), blended_img, vmin=0., vmax=1.)

        # blended_img = sc.render(blended_img)
        # blended_img.write_png(os.path.join(save_path, f'flow_{300+idx}.png'), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)

def make_video(dir_res):

    data_path_gt = dir_res + "/Renderings_GT"
    data_path_rec = dir_res + "/Renderings_interpol"
    image_files_gt = natsorted(glob.glob(f'{data_path_gt}/density_*.png'))
    image_files_rec = natsorted(glob.glob(f'{data_path_rec}/density_*.png'))
    print(len(image_files_gt), len(image_files_rec))
    images_gt = []
    images_rec = []
    for i in range(len(image_files_rec)):
        images_gt.append(cv2.imread(image_files_gt[i]))
        images_rec.append(cv2.imread(image_files_rec[i]))
    video = cv2.VideoWriter(os.path.join(dir_res, f'./GT_interpol.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 15, (np.array(images_gt[0]).shape[1] * 2, np.array(images_gt[0]).shape[0]))

    for idx, (img1, img2) in enumerate(zip(images_gt, images_rec)):
        img1 = cv2.putText(img1, 'GT', (240, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img2 = cv2.putText(img2, 'LREST', (140, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        concatenated_img = np.concatenate((img1, img2), axis=1)
        concatenated_img = cv2.putText(concatenated_img, f'{idx}', (900, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video.write(concatenated_img)
    video.release()
    # cv2.destroyAllWindows()


