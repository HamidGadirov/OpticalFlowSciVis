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
import io 
from PIL import Image
from matplotlib.colors import LogNorm
import glob
from natsort import natsorted


def visualize_seq_dens_3d(original_data, interpol_data, interpol_data_coordnet, flow_gt, flow,
                          dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    
    columns = 5
    rows = 5
    skip = 14
    data_to_vis = original_data
    index = 0
    shift = 0

    if "3d" in dataset:
        vmin = 0
        print("vis range:", vmin, data_to_vis.max())
        hspace = -0.15
        wspace = -0.8
        cmap ='gray'

    u_gt = flow_gt[:, 0, ...] 
    v_gt = flow_gt[:, 1, ...] 
    w_gt = flow_gt[:, 2, ...] 
    flow_norm_gt = np.linalg.norm(flow_gt, axis=1)
    u = flow[:, 0, ...] 
    v = flow[:, 1, ...] 
    w = flow[:, 2, ...] 
    flow_norm = np.linalg.norm(flow, axis=1)

    # wspace is x; hspace is y
    fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(20, 8), gridspec_kw={'hspace': hspace, 'wspace': wspace})

    # bounds = (105.0, 160.0)
    # tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
    # tf.add_gaussian(np.log10(1e2), width=2e-4, height=[0.0, 0.25, 1.0, 1e3])

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
    # tf_vel.add_gaussian(np.log10(0.55), width=1e-4, height=[1.0, 0.6, 0.1, 1e2]) # orange 700-800

    # tf_vel.add_gaussian(np.log10(0.3), width=1e-4, height=[1.0, 0.6, 0.1, 1e2]) # orange 700-800
    # TODO: try rgb 3 blobs
    # tf_vel.add_gaussian(np.log10(0.4), width=2e-4, height=[1.0, 0.1, 0.1, 1e2]) # red
    # tf_vel.add_gaussian(np.log10(0.5), width=2e-4, height=[0.1, 1.0, 0.1, 1e2]) # green
    # tf_vel.add_gaussian(np.log10(0.6), width=2e-4, height=[0.1, 0.1, 1.0, 1e2]) # blue

    # print("min, max: ", np.min(original_data), np.max(original_data))
    # print("mean: ", np.mean(original_data))
    # print("std: ", np.std(original_data))

    cam_resolution = 512 # 256 512

    def add_zoom(ax, img):
        # Define the coordinates and size of the region to be zoomed in
        if cam_resolution == 256:
            x1, y1, width1, height1 = 55, 120, 40, 40  # For the top left zoomed-in version
            x2, y2, width2, height2 = 108, 80, 40, 40  # For the top right zoomed-in version
        elif cam_resolution == 512:
            x1, y1, width1, height1 = 110, 240, 80, 80  # For the top left zoomed-in version
            x2, y2, width2, height2 = 216, 160, 80, 80  # For the top right zoomed-in version

        # Create a rectangle patch to highlight the region for the top right zoomed-in version
        rect1 = patches.Rectangle((x1, y1), width1, height1, linewidth=1.0, edgecolor='purple', facecolor='none')
        ax.add_patch(rect1)

        # Create a rectangle patch to highlight the region for the top left zoomed-in version
        rect2 = patches.Rectangle((x2, y2), width2, height2, linewidth=1.0, edgecolor='orange', facecolor='none')
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


    for i in range(columns):

        # if i == 4:
        #     index = 70

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
        cam.zoom(1.75)
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
            ax_title = "t=" + str(index + shift + 710)
            # ax_title = "t="
            ax_title += "*"
            ax.set_title(ax_title, fontsize=10, pad=-10)
        elif dataset == "pool3d":
            ax_title = "t=" + str(index + shift + 30)
            ax.set_title(ax_title, fontsize=10, pad=10)
        elif dataset == "nyx3d":
            ax_title = "t=" + str(index + shift + 600)
            ax.set_title(ax_title, fontsize=10, pad=10)    
        ax.axis('off')

        # add zoomings
        add_zoom(ax, img)

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
        cam.zoom(1.75)
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
        add_zoom(ax, img)


        # Third row: interpolated data coordnet
        volume = interpol_data_coordnet[round(index)]
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
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        # print("im", im)
        im.write_png(os.path.join(dir_res, "Dens_seq.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ax = axs[2][i]
        # img = interpol_data[round(index)]
        img = plt.imread(os.path.join(dir_res, "Dens_seq.png"))
        ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # add zoomings
        add_zoom(ax, img)


        # Fourth row: flow gt
        # volume = u_gt[round(index)]
        # # volume = flow_gt[round(index)]
        # # print("volume", volume)
        # # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        # ds = yt.load_uniform_grid(dict(velocity_x=volume), volume.shape)
        # sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        # source = sc[0]
        # source.set_log(True)
        # source.tfh.tf = tf_vel
        # source.tfh.bounds = bounds
        # source.tfh.plot(os.path.join(dir_res, "flow_gt_tf.png"), profile_field=("stream", "velocity_x"))

        # cam = sc.camera
        # cam.resolution = cam_resolution
        # cam.zoom(1.75)
        # cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        # im = sc.render()
        # # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        # im.write_png(os.path.join(dir_res, "Flow.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        # print("rendered!")

        # ax = axs[3][i]
        # # img = interpol_data[round(index)]
        # img = plt.imread(os.path.join(dir_res, "Flow.png"))
        # ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        # ax.axis('off')

        volume = u_gt[round(index)]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_x=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.55), width=2e-4, height=[0.84, 0.15, 0.16, 1e2]) # red
        # tf_vel.add_gaussian(np.log10(0.55), width=3e-4, height=[0.58, 0.40, 0.74, 1e2]) # purple
        tf_vel.add_gaussian(np.log10(0.55), width=3e-4, height=[1.0, 0.50, 0.05, 1e2]) # orange
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_x_gt_tf.png"), profile_field=("stream", "velocity_x"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_x.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = v_gt[round(index)]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_y=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_y"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.55), width=2e-4, height=[0.17, 0.63, 0.17, 1e2]) # green
        tf_vel.add_gaussian(np.log10(0.55), width=3e-4, height=[0.1, 0.8, 0.1, 1e2]) # green
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_y_gt_tf.png"), profile_field=("stream", "velocity_y"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_y.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = w_gt[round(index)]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_z=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_z"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.55), width=2e-4, height=[0.12, 0.47, 0.71, 1e2]) # blue
        tf_vel.add_gaussian(np.log10(0.55), width=3e-4, height=[0.1, 0.1, 1.0, 1e2]) # blue
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_z_gt_tf.png"), profile_field=("stream", "velocity_z"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_z.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ax = axs[3][i]
        # img = interpol_data[round(index)]
        img_x = plt.imread(os.path.join(dir_res, "Flow_x.png"))
        img_y = plt.imread(os.path.join(dir_res, "Flow_y.png"))
        img_z = plt.imread(os.path.join(dir_res, "Flow_z.png"))
        blended_img = (img_x + img_y + img_z) # / 3 
        ax.imshow(blended_img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # add zoomings flow
        add_zoom(ax, blended_img)


        # Fifth row: flow
        # volume = flow_norm[round(index)]
        # # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        # sc = yt.create_scene(ds, field=("gas", "density"))
        # source = sc[0]
        # source.set_log(True)
        # source.tfh.tf = tf_vel
        # source.tfh.bounds = bounds
        # source.tfh.plot(os.path.join(dir_res, "flow_tf.png"), profile_field=("gas", "density"))

        # cam = sc.camera
        # cam.resolution = cam_resolution
        # cam.zoom(1.75)
        # cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        # im = sc.render()
        # # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        # im.write_png(os.path.join(dir_res, "Flow.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        # print("rendered!")

        # ax = axs[4][i]
        # # img = interpol_data[round(index)]
        # img = plt.imread(os.path.join(dir_res, "Flow.png"))
        # ax.imshow(img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        # ax.axis('off')

        volume = u[round(index)]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_x=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_x"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        # tf_vel.add_gaussian(np.log10(0.4), width=2e-4, height=[1.0, 0.1, 0.1, 1e2]) # red
        # tf_vel.add_gaussian(np.log10(0.4), width=3e-4, height=[0.58, 0.40, 0.74, 1e2]) # purple
        tf_vel.add_gaussian(np.log10(0.4), width=3e-4, height=[1.0, 0.50, 0.05, 1e2]) # orange
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_gt_tf.png"), profile_field=("stream", "velocity_x"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_x.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = v[round(index)]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_y=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_y"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.4), width=3e-4, height=[0.1, 0.8, 0.1, 1e2]) # green
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_gt_tf.png"), profile_field=("stream", "velocity_y"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_y.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        volume = w[round(index)]
        # volume = flow_gt[round(index)]
        # print("volume", volume)
        # ds = yt.load_uniform_grid(dict(density=volume), volume.shape)
        ds = yt.load_uniform_grid(dict(velocity_z=volume), volume.shape)
        sc = yt.create_scene(ds, field=("stream", "velocity_z"))
        source = sc[0]
        source.set_log(True)
        tf_vel = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
        tf_vel.add_gaussian(np.log10(0.4), width=3e-4, height=[0.1, 0.1, 1.0, 1e2]) # blue
        source.tfh.tf = tf_vel
        source.tfh.bounds = bounds
        source.tfh.plot(os.path.join(dir_res, "flow_gt_tf.png"), profile_field=("stream", "velocity_z"))

        cam = sc.camera
        cam.resolution = cam_resolution
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        im = sc.render()
        # im.add_background_color(background=([1.0, 1.0, 1.0, 1.0]))
        im.write_png(os.path.join(dir_res, "Flow_z.png"), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)
        print("rendered!")

        ax = axs[4][i]
        # img = interpol_data[round(index)]
        img_x = plt.imread(os.path.join(dir_res, "Flow_x.png"))
        img_y = plt.imread(os.path.join(dir_res, "Flow_y.png"))
        img_z = plt.imread(os.path.join(dir_res, "Flow_z.png"))
        blended_img = (img_x + img_y + img_z) # / 3 
        ax.imshow(blended_img, vmin=vmin, vmax=data_to_vis.max(), cmap=cmap)
        ax.axis('off')

        # add zoomings flow
        add_zoom(ax, blended_img)

        index += skip 
        shift += 6

    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

def render_density(gt, interpol, dir_res="Results"):

    bounds = (0.1, 0.9)
    tf = yt.ColorTransferFunction(np.log10(bounds), grey_opacity=True)
    # density
    tf.add_gaussian(np.log10(0.5), width=1e-4, height=[0.05, 0.1, 1.0, 1e2]) # blue 700-800

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
        cam.zoom(1.75)
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
        cam.zoom(1.75)
        cam.focus = ds.arr([0, 0.3, 0.3], 'unitary')
        # cam.rotate(30 / 180 * np.pi, rot_vector=np.array([1.0, 0.0, 0.0]))
        im = sc.render()
        im.write_png(os.path.join(save_path, f'density_{700+idx}.png'), background=[1.0, 1.0, 1.0, 0.0], sigma_clip=4.0)

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


filenames = ["data_original_interpol.pkl", "interpol_data.pkl", "data_coordnet_interpol.pkl", "flow_original_pred.pkl", "flow.pkl"]
data = []
for fn in filenames:
    with open(fn, 'rb') as f:
        data.append(pickle.load(f))

# Unpack the data
original_data, interpol_data, interpol_data_coordnet, flow_gt, flow = data

# original_data = original_data[30:]
# interpol_data = interpol_data[30:]
# interpol_data_coordnet = interpol_data_coordnet[30:]
# flow_gt = flow_gt[30:]
# flow = flow[30:]

print(original_data.shape)
print(interpol_data.shape)
print(interpol_data_coordnet.shape)
print(flow_gt.shape)
print(flow.shape)

dataset = "jets3d"
visualize_seq_dens_3d(original_data, interpol_data, interpol_data_coordnet, flow_gt, flow,
                          dataset, dir_res="./", title="3D_Dens_FlowXYZ", show=False, save=True)

