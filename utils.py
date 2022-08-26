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
import pyimof
import math
import plotly.graph_objects as go

import io 
from PIL import Image

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

def plot_loss(loss, dir_res, name="loss.png", save=False):
    fig = plt.figure(figsize=(8, 4))
    labels = ('total', 'lapl', 'tea', 'distill', 'reg', 'photo')
    colors = ('b', 'r', 'g', 'c', 'm', 'y')
    print(loss.shape)
    for j in range(loss.shape[1]): # how many loss components
        plt.plot(loss[:, j], colors[j])
    # input("x")
    plt.title('validation losses')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['total', 'lapl', 'tea', 'dist', 'reg', 'photo'], loc='upper right')
    # plt.legend()
    if save:
        res_path = os.path.join(dir_res, name)
        fig.savefig(res_path)

def visualize_ind(image, dir_res="Results", name="result.png", save=False):
    fig = plt.figure()
    # ax = fig.gca()
    plt.imshow(image.astype('uint8'))
    # plt.show()
    if save:
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        dir_res = os.path.join(dir_res, name)
        fig.savefig(dir_res)
        # input("saved")

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

    for i in range(1, columns*rows+1):
        index = (i - 1) * 2 # skip each second
        if (index >= data_to_vis.shape[0]):
            break

        img = data_to_vis[index,...]
        ax = fig.add_subplot(rows, columns, i)

        if index % factor == 0:
            # add bb to the gt image
            width = data_to_vis.shape[2]
            height = data_to_vis.shape[1]
            linewidth=2
            if "cylinder2d" in dataset:
                linewidth=1
            rect = patches.Rectangle((0, 0), width, height, linewidth=linewidth, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        plt.axis('off')
        # print("range:", data_to_vis.min(), data_to_vis.max())
        # input("x")
        plt.imshow(img, cmap='viridis', vmin=data_to_vis.min(), vmax=data_to_vis.max())
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

def visualize_large(original_data, interpol_data, diffs,
        flow, flow_gt, diffs_flow, flow_u_diff, flow_v_diff, mask,
        factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    # print(original_data.shape, interpol_data.shape, diffs.shape, flow.shape, flow_gt.shape, diffs_flow.shape, u_diff.shape, v_diff.shape)
    fig = plt.figure(figsize=(30, 10))
    columns = 20
    if "droplet2d" in dataset or "FluidSimML2d" in dataset or "rectangle2d" in dataset:
        columns = 24
    elif "pipedcylinder2d" in dataset or "cylinder2d" in dataset:
        columns = 20
    elif dataset == "vimeo2d":
        columns = 28
    quiver_steps = 5
    rows = 7 # 4
    skip = 1
    if factor == 8:
        skip = 2
    if factor >= 16: # or factor == 32 or factor == 64:
        skip = 4
    # if factor == 128:
    #     skip = 8

    print(flow.shape, flow_gt.shape)
    # input("x")

    data_to_vis = original_data
    flow_u_gt = flow_gt[:, 0, ...]
    flow_v_gt = flow_gt[:, 1, ...]
    flow_u = flow[:, 0, ...]
    flow_v = flow[:, 1, ...]
    print("flow_u_gt is in range %f to %f" % (np.min(flow_u_gt), np.max(flow_u_gt))) # -2 to 4 -8 to 8
    print("flow_v_gt is in range %f to %f" % (np.min(flow_v_gt), np.max(flow_v_gt)))
    # input("x")
    # norm_gt = np.sqrt(np.max(flow_u_gt) * np.max(flow_u_gt) + np.max(flow_v_gt) * np.max(flow_v_gt))
    # input(norm_gt)

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
        if int((i-1)/columns) == 2: # diff
            # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
            img = diffs[round(index)] # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 3: # mask
            # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
            img = mask[round(index)] # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 6: # flow diff
            img = diffs_flow[round(index)] # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 4: # flow gt
            if dataset == "vimeo2d": # show pred flow in hsv-rgb instead of gt
                hsv = np.empty(shape=(flow_u.shape[1], flow_u.shape[2], 3), dtype=np.uint8)
                hsv[:,:,1] = 255
                mag, ang = cv2.cartToPolar(flow_u[round(index)], flow_v[round(index)])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

                # bgr = np.empty(shape=hsv.shape, dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # print("Flow in bgr:", bgr.shape)
                # plt.imshow(bgr, vmin=data_to_vis.min(), vmax=data_to_vis.max())
                plt.imshow(bgr)
            else:
                u_gt = flow_u_gt[round(index)]
                v_gt = flow_v_gt[round(index)]
                norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
                # img = original_data[round(index)] # [index-columns*2*2*2]
                img = np.zeros((original_data.shape[1], original_data.shape[2]))
                plt.axis('off')
                ax = plt.gca()
                # pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
                pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='jet', bg_cmap='gray')
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
        if int((i-1)/columns) == 5: # flow pred
            # this was flipped. why?
            # u = - flow_u[round(index)]
            # v = - flow_v[round(index)]
            u = flow_u[round(index)]
            v = flow_v[round(index)]
            norm = np.sqrt(u * u + v * v)
            # img = interpol_data[round(index)] # [index-columns*2*2*2]
            img = np.zeros((original_data.shape[1], original_data.shape[2]))
            plt.axis('off')
            ax = plt.gca()
            pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
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
        # if int((i-1)/columns) == 5: # flow diff vec
        #     u_diff = flow_u_diff[round(index)]
        #     v_diff = flow_v_diff[round(index)]
        #     norm_diff = np.sqrt(u_diff * u_diff + v_diff * v_diff)
        #     # print("u_diff is in range %f to %f" % (np.min(u_diff), np.max(u_diff)))
        #     # print("norm_diff is in range %f to %f" % (np.min(norm_diff), np.max(norm_diff)))
        #     img = interpol_data[round(index)] # [index-columns*2*2*2]
        #     plt.axis('off')
        #     ax = plt.gca()
        #     pyimof.display.quiver(u_diff, v_diff, c=norm_diff, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=quiver_steps)
        #     # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
        #     data_range = min(original_data.shape[0], interpol_data.shape[0], u_diff.shape[0])
        #     # print(data_range)
        #     if round(index) >= data_range:
        #         break

        # if dataset != "vimeo2d":
        #     if round(index) % factor == 0 and int((i-1)/columns) == 0:
        #         # add bb to the gt image
        #         width = data_to_vis.shape[2]
        #         height = data_to_vis.shape[1]
        #         linewidth = 2
        #         if "cylinder2d" in dataset:
        #             linewidth = 1
        #         rect = patches.Rectangle((0, 0), width+linewidth, height+linewidth, linewidth=linewidth, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)

        plt.axis('off')
        cmap='viridis'
        if "rectangle2d" in dataset:
            cmap = "gray"
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
    # wspace is x; hspace is y 
    if "droplet2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.69)
    elif "FluidSimML2d" in dataset:
        plt.subplots_adjust(wspace=-0.21, hspace=-0.01)
    elif "pipedcylinder2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.75)
    elif "cylinder2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.945)
    elif "rectangle2d" in dataset:
        plt.subplots_adjust(wspace=-0.95, hspace=0.01)
        x = 0.25
        y = 0.8
    elif "vimeo2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        x = 0.075
        y = 0.8

    step = (0.8 - 0.15) / 6.
    fig.text(x, y, 'GT', size=12)
    fig.text(x, y - step, 'Interpol', size=12)
    fig.text(x, y - step*2, 'Diff', size=12)
    fig.text(x, y - step*3, 'Mask', size=12)
    fig.text(x, y - step*4, 'Flow GT', size=12)
    fig.text(x, y - step*5, 'Flow pred', size=12)
    fig.text(x, y - step*6, 'Flow diff', size=12)

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

def visualize_3d(original_vols, interpol_vols, flow, \
        dataset, dir_res="Results", title="Volumes", show=True, save=False):

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
    title ="tangaroa3d.html" # "droplet3d.html"
    image_path = os.path.join(dir_res, title)
    vol.write_html(image_path) 
    input("x")

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
        # if int((i-1)/columns) == 2: # flow
        #     ax = plt.figure().add_subplot(projection='3d')
        #     # Make the grid
        #     x, y, z = np.meshgrid(np.arange(0, 64, 1), np.arange(0, 64, 1), np.arange(0, 64, 1))
        #     # Make the direction data for the arrows
        #     u = flow_u
        #     v = flow_v
        #     w = flow_w
        #     ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)
        #     print("2")

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