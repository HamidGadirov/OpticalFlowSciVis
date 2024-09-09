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

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def visualize_large(original_data, interpol_data, diffs, interpol_data_RIFE, diffs_RIFE,
        flow, flow_RIFE,
        factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    # print(original_data.shape, interpol_data.shape, diffs.shape, flow.shape, flow_gt.shape, diffs_flow.shape, u_diff.shape, v_diff.shape)
    fig = plt.figure(figsize=(60, 10))
    columns = 20
    if "droplet2d" in dataset or "FluidSimML2d" in dataset or "rectangle2d" in dataset:
        columns = 24
    elif dataset == "lbs2d":
        columns = 7 # 9 30
    rows = 9 # 7
    skip = 1
    # if factor == 16:
    #     skip = 3

    print(flow.shape)
    # input("x")

    data_to_vis = original_data
    flow_u = flow[:, 0, ...]
    flow_v = flow[:, 1, ...]

    threshold_u = 0.6  # Adjust based on flow_u's range
    threshold_v = 0.6  # Adjust based on flow_v's range

    # Calculate the magnitude
    magnitude_u = np.abs(flow_u)
    magnitude_v = np.abs(flow_v)

    # Copy the original data to new variables
    filtered_flow_u = np.copy(flow_u)
    filtered_flow_v = np.copy(flow_v)

    # Apply the threshold to filter out small magnitude values
    filtered_flow_u[(magnitude_u < threshold_u)] = 0
    filtered_flow_v[(magnitude_v < threshold_v)] = 0

    threshold = 0.5

    # Calculate the magnitude
    magnitude_u_arrow = np.abs(flow_u)
    magnitude_v_arrow = np.abs(flow_v)

    # Copy the original data to new variables
    filtered_flow_u_arrow = np.copy(flow_u)
    filtered_flow_v_arrow = np.copy(flow_v)

    # Apply the threshold to filter out small magnitude values
    filtered_flow_u_arrow[(magnitude_u_arrow < threshold)] = 0
    filtered_flow_v_arrow[(magnitude_v_arrow < threshold)] = 0

    flow_u_RIFE = flow_RIFE[:, 0, ...]
    flow_v_RIFE = flow_RIFE[:, 1, ...]

    print("original_data is in range %f to %f" % (np.min(original_data), np.max(original_data)))
    print("interpol_data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))

    print("original_data mean & std:", original_data.mean(), original_data.std())
    print("interpol_data mean & std:", interpol_data.mean(), interpol_data.std())

    print("flow_u is in range %f to %f" % (np.min(flow_u), np.max(flow_u))) 
    print("flow_v is in range %f to %f" % (np.min(flow_v), np.max(flow_v)))
    # input("viz")
    # norm_gt = np.sqrt(np.max(flow_u_gt) * np.max(flow_u_gt) + np.max(flow_v_gt) * np.max(flow_v_gt))
    # input(norm_gt)

    if "droplet2d" in dataset:
        vmin = 0
        max_flow_gt = 15 # flow.max()
        print("vis range:", vmin, data_to_vis.max())
        print("vis flow range:", 0, max_flow_gt)
        cmap ='gray'
        quiver_steps = 9
        # scale = None

        # magnitude = np.sqrt(flow_u**2 + flow_v**2)
        # max_magnitude = np.max(magnitude)
        # scale = 0.5 / max_magnitude  # Adjust the scale factor as needed
        scale = None # 0.25

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
        if round(index) % factor == 0: # bug fix
            index_pred += skip

        if int((i-1)/columns) == 0: # gt
            img = original_data[round(index)]
            if dataset == "lbs2d":
                plt.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
                # plt.imshow(img, vmin=original_data.std()-original_data.mean(), vmax=data_to_vis.max())
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
            elif dataset == "droplet2d":
                ax_title = "t=" + str(index + 25) # 35
            if index % factor != 0: # if index % 3 == 1:
                ax_title += "*"
            if dataset == "lbs2d":
                ax.set_title(ax_title, fontsize=5, pad=-10)
            elif dataset == "droplet2d":
                ax.set_title(ax_title, fontsize=4, pad=-10)
        if int((i-1)/columns) == 1 and round(index) % factor != 0: # interpol
            # img = data_to_vis[index+int(data_to_vis.shape[0]/3),...]
            img = interpol_data[round(index_pred)] # [index-columns*2,...] # (index - 1) % 3 round((index - 1) / 2)
            if dataset == "lbs2d":
                plt.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
                # plt.imshow(img, vmin=interpol_data.std()-interpol_data.mean(), vmax=data_to_vis.max())
            else:
                plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            # print("index:", index)
            # print("(index - 1) % 3:", (index - 1) / 3)
            index_pred += skip

        if int((i-1)/columns) == 2 and round(index) % factor != 0: # diff
            # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
            # img = diffs[round((index - 1) / 2)] # [index-columns*2*2,...]
            img = diffs[round(index_pred)] * 5 # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            index_pred += skip

        if int((i-1)/columns) == 3 and round(index) % factor != 0: # interpol RIFE
            # img = data_to_vis[index+int(data_to_vis.shape[0]/3),...]
            img = interpol_data_RIFE[round(index_pred)] # [index-columns*2,...] # (index - 1) % 3 round((index - 1) / 2)
            if dataset == "lbs2d":
                plt.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
            else:
                plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            # print("index:", index)
            # print("(index - 1) % 3:", (index - 1) / 3)
            index_pred += skip

        if int((i-1)/columns) == 4 and round(index) % factor != 0: # diff
            # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
            # img = diffs[round((index - 1) / 2)] # [index-columns*2*2,...]
            img = diffs_RIFE[round(index_pred)] * 5 # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max(), cmap=cmap)
            index_pred += skip

        if int((i-1)/columns) == 5 and round(index) % factor != 0: # flow LREST
            if dataset == "vimeo2d" or dataset == "droplet2d": # show pred flow in hsv-rgb instead of gt
                hsv = np.empty(shape=(flow_u.shape[1], flow_u.shape[2], 3), dtype=np.uint8)
                hsv[:,:,1] = 255
                # mag, ang = cv2.cartToPolar(flow_u[round((index - 1) / 3)], flow_v[round((index - 1) / 3)])
                # mag, ang = cv2.cartToPolar(flow_u[round(index_pred)], flow_v[round(index_pred)])  
                mag, ang = cv2.cartToPolar(filtered_flow_u[round(index_pred)], filtered_flow_v[round(index_pred)])
                
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                # bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # print("Flow in bgr:", bgr.shape)
                # plt.imshow(bgr, vmin=data_to_vis.min(), vmax=data_to_vis.max())
                plt.imshow(bgr)
                index_pred += skip
        if int((i-1)/columns) == 6 and round(index) % factor != 0: # flow RIFE
            if dataset == "vimeo2d" or dataset == "droplet2d": # show pred flow in hsv-rgb instead of gt
                hsv = np.empty(shape=(flow_u_RIFE.shape[1], flow_u_RIFE.shape[2], 3), dtype=np.uint8)
                hsv[:,:,1] = 255
                # mag, ang = cv2.cartToPolar(flow_u[round((index - 1) / 3)], flow_v[round((index - 1) / 3)])
                mag, ang = cv2.cartToPolar(flow_u_RIFE[round(index_pred)], flow_v_RIFE[round(index_pred)])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                # bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # print("Flow in bgr:", bgr.shape)
                # plt.imshow(bgr, vmin=data_to_vis.min(), vmax=data_to_vis.max())
                plt.imshow(bgr)
                index_pred += skip

        if int((i-1)/columns) == 7 and round(index) % factor != 0: # flow arrows LREST
            u = flow_u[round(index_pred)]
            v = flow_v[round(index_pred)]
            norm = np.sqrt(u * u + v * v) # / norm_coeff
            # print(norm.mean())
            # img = interpol_data[round(index)] # [index-columns*2*2*2]
            img = np.zeros((original_data.shape[1], original_data.shape[2]))
            plt.axis('off')
            # ax = plt.gca()
            # print("norm is in range %f to %f" % (np.min(norm), np.max(norm)))
            filtered_u = filtered_flow_u_arrow[round(index_pred)]
            filtered_v = filtered_flow_v_arrow[round(index_pred)]
            # filtered_norm = np.sqrt(filtered_u * filtered_u + filtered_v * filtered_v) # / norm_coeff

            # print("norm ", norm)

            # # Set the threshold for arrow length (magnitude)
            # magnitude_threshold = 0.8  # Adjust this value as needed

            # # Calculate the magnitude of the flow vectors
            # magnitude = np.sqrt(u**2 + v**2)

            # # Mask the arrows that are smaller than the threshold
            # mask = magnitude >= magnitude_threshold
            # u_filtered = np.ma.masked_where(~mask, u)
            # v_filtered = np.ma.masked_where(~mask, v)

            # Threshold value
            threshold = 0.8

            # For vectors with magnitude smaller than the threshold, divide the vectors by 2
            u_filtered = np.where(threshold < threshold, u / 10, u)
            v_filtered = np.where(threshold < threshold, v / 10, v)

            filtered_norm = np.sqrt(filtered_u * filtered_u + filtered_v * filtered_v) # / norm_coeff
            
            pyimof.display.quiver(u_filtered, v_filtered, c=filtered_norm, bg=img, ax=ax, cmap='turbo', clim=(0.5, 12), scale = scale, 
                                  bg_cmap='gray', step=quiver_steps, width=0.3)

            # # Define the region to zoom into (change these coordinates to focus on the region you want)
            # x1, x2, y1, y2 = 30, 50, 30, 50  # These values should be set based on the specific area of interest

            # # Create an inset axis for the zoomed region
            # ax_inset = ax.inset_axes([0.6, 0.6, 0.3, 0.3])  # This defines the location and size of the inset in the main plot
            # ax_inset.axis('off')

            # # Plot the zoomed-in region
            # pyimof.display.quiver(u_filtered[y1:y2, x1:x2], v_filtered[y1:y2, x1:x2], 
            #                     c=filtered_norm[y1:y2, x1:x2], 
            #                     bg=img[y1:y2, x1:x2], 
            #                     ax=ax_inset, 
            #                     cmap='turbo', 
            #                     clim=(0.5, 12), 
            #                     scale=scale, 
            #                     bg_cmap='gray', 
            #                     step=quiver_steps, 
            #                     width=0.3)

            # # Optionally, add a rectangle on the main plot to indicate the zoomed region
            # rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=0.25)
            # ax.add_patch(rect)

            data_range = min(original_data.shape[0], interpol_data.shape[0], flow_u.shape[0])
            # print(data_range)
            if round(index) >= data_range:
                break
            if dataset == "droplet2d":
                index_pred += skip

        if int((i-1)/columns) == 8 and round(index) % factor != 0: # flow arrows RIFE
            u = flow_u_RIFE[round(index_pred)]
            v = flow_v_RIFE[round(index_pred)]
            norm = np.sqrt(u * u + v * v) # / norm_coeff
            # print(norm.mean())
            # img = interpol_data[round(index)] # [index-columns*2*2*2]
            img = np.zeros((original_data.shape[1], original_data.shape[2]))
            plt.axis('off')

            # print("norm ", norm)

            # ax = plt.gca()
            # print("norm is in range %f to %f" % (np.min(norm), np.max(norm)))
            pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='turbo', clim=(0, max_flow_gt), scale = scale, 
                                  bg_cmap='gray', step=quiver_steps, width=0.3)

            data_range = min(original_data.shape[0], interpol_data.shape[0], flow_u.shape[0])
            # print(data_range)
            if round(index) >= data_range:
                break
            if dataset == "droplet2d":
                index_pred += skip

        # if int((i-1)/columns) == 7 and round(index) % factor != 0:  # streamlines pred
        #     u = flow_u[round(index_pred)]
        #     v = flow_v[round(index_pred)]

        #     # Generate a meshgrid for the 2D flow field
        #     y, x = np.mgrid[0:u.shape[0], 0:v.shape[1]]

        #     magnitude = np.sqrt(u**2 + v**2)
        #     # print("magnitude", magnitude)

        #     # # Apply a threshold to filter out low magnitude streamlines
        #     # threshold = 0.1  # Adjust this value based on your data
        #     # u[magnitude < threshold] = 0
        #     # v[magnitude < threshold] = 0

        #     # Plot the streamlines with finer control over density, line width, and arrow size
        #     plt.streamplot(x, y, u, v, color=np.sqrt(u**2 + v**2), cmap='turbo', density=1.5,
        #                 linewidth=0.5, arrowsize=0.2, minlength=0.2)

        #     # Set plot limits and aspect ratio
        #     plt.xlim(0, u.shape[1])
        #     plt.ylim(0, v.shape[0])
        #     plt.gca().set_aspect('equal', adjustable='box')
        #     plt.gca().invert_yaxis()  # To match image origin

        #     index_pred += skip

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

    if dataset == "lbs2d":
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
            cb_ax = fig.add_axes([0.69, 0.54, 0.003, 0.23]) 
        elif "rectangle2d" in dataset:
            cb_ax = fig.add_axes([0.75, 0.5, 0.005, 0.4])
        # cb.ax.tick_params(labelsize=font_size)
        fmt = lambda x, pos: '{:0.1f}'.format(x)
        bar = np.linspace(vmin, data_to_vis.max(), 5, endpoint=True)
        cbar = plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))
        cbar.ax.tick_params(labelsize=6)
        # colorbar for flow
        normalizer=Normalize(vmin=0., vmax=max_flow_gt)
        im=cm.ScalarMappable(norm=normalizer, cmap='turbo')
        if "lbs2d" in dataset:
            # cb_ax = fig.add_axes([0.91, 0.2, 0.005, 0.25])
            # cb_ax = fig.add_axes([0.82, 0.15, 0.005, 0.25]) # 9 cols
            # cb_ax = fig.add_axes([0.75, 0.15, 0.005, 0.25])
            cb_ax = fig.add_axes([0.69, 0.15, 0.003, 0.3])
        elif "rectangle2d" in dataset:
            cb_ax = fig.add_axes([0.75, 0.12, 0.005, 0.3])
        bar = np.linspace(0., max_flow_gt, 4, endpoint=True)
        fmt = lambda x, pos: '{:0.2f}'.format(x)
        cbar = plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))
        cbar.ax.tick_params(labelsize=6)

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

        # colorbar for flow
        normalizer=Normalize(vmin=0., vmax=max_flow_gt)
        im=cm.ScalarMappable(norm=normalizer, cmap='turbo')
        cb_ax = fig.add_axes([0.445, 0.12, 0.002, 0.15])
        bar = np.linspace(0., max_flow_gt, 4, endpoint=True)
        fmt = lambda x, pos: '{:0.1f}'.format(x)
        cbar = plt.colorbar(im, cax=cb_ax, cmap='turbo', ticks=bar, format=FuncFormatter(fmt))
        cbar.ax.tick_params(labelsize=4)

    fig = plt.gcf()
    suptitle = dataset + "_" + str(factor) + "x"
    # plt.suptitle(suptitle) 
    # print(matplotlib.get_backend())
    fig.set_size_inches(20, 3)
    # wspace is x; hspace is y 
    if "droplet2d" in dataset:
        # plt.subplots_adjust(wspace=-0.5, hspace=0.01) # 7 rows
        plt.subplots_adjust(wspace=-0.89, hspace=0.01) # 9 rows
        x = 0.16
        y = 0.8
        step = (y - 0.15) / 6.
    elif "lbs2d" in dataset:
        # plt.subplots_adjust(wspace=0.01, hspace=-0.75) # hspace=-0.9
        # plt.subplots_adjust(wspace=-0.73, hspace=0.02) # 9 cols
        # plt.subplots_adjust(wspace=-0.825, hspace=0.02) # 7 cols
        plt.subplots_adjust(wspace=-0.85, hspace=0.02) # 7 cols 10 rows
        # x = 0.075
        # y = 0.7
        # step = (y - 0.25) / 6.
        x = 0.20 # 0.16 # 9 cols
        y = 0.8
        step = (y - 0.15) / 6.

    # font_size = 10
    # # if "droplet2d" in dataset:
    # #     font_size = 10
    # fig.text(x, y, 'Dens GT', size=font_size)
    # fig.text(x, y - step, 'Dens pred', size=font_size)
    # fig.text(x, y - step*2, 'Dens diff', size=font_size)
    # # fig.text(x, y - step*3, 'Dens mask', size=font_size)
    # if "droplet2d" in dataset or "vimeo2d" in dataset:
    #     fig.text(x, y - step*3, 'Flow pred HSV', size=font_size)
    # else:
    #     fig.text(x, y - step*3, 'Flow GT', size=font_size)
    # fig.text(x, y - step*4, 'Flow pred', size=font_size)
    # if dataset != "droplet2d" and dataset != "vimeo2d":
    #     fig.text(x, y - step*5, 'Flow diff', size=font_size)

    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

    
# Load density data
original_data = load_data('original_data.pkl')

# Load LREST density data
interpol_data = load_data('interpol_data_LREST.pkl')
diffs = load_data('diffs_LREST.pkl')

# Load RIFE density data
interpol_data_RIFE = load_data('interpol_data_RIFE.pkl')
diffs_RIFE = load_data('diffs_RIFE.pkl')

# Load LREST flow data
flow = load_data('flow_pred_LREST.pkl')

# Load RIFE flow data
flow_RIFE = load_data('flow_pred_RIFE.pkl')

factor = 3 # 6
# dataset = "lbs2d"
dataset = "droplet2d"
title = "Droplet_GT_Interpol_Diff_Flow_3x_new"
title = "Droplet_GT_Interpol_Diff_Flow_3x_new_arrows"
# title = "Droplet_GT_Interpol_Diff_Flow_3x_new_streamlines"

# test:
input("didn't forget anything?..")

visualize_large(original_data, interpol_data, diffs, interpol_data_RIFE, diffs_RIFE,
        flow, flow_RIFE,
        factor, dataset, dir_res="./", title=title, show=True, save=True)
