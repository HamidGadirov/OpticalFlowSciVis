import os
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

def plot_loss(loss, dir_res, name="loss.png", save=False):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(loss)
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['val'], loc='upper left')
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

def visualize_large(original_data, interpol_data, diffs, \
        flow, flow_gt, diffs_flow, flow_u_diff, flow_v_diff, factor, dataset, dir_res="Results", title="Lots of Data", show=True, save=False):
    # print(original_data.shape, interpol_data.shape, diffs.shape, flow.shape, flow_gt.shape, diffs_flow.shape, u_diff.shape, v_diff.shape)
    fig = plt.figure(figsize=(30, 10))
    columns = 20
    if "droplet2d" in dataset or "FluidSimML2d" in dataset or "rectangle2d" in dataset:
        columns = 24
    elif "pipedcylinder2d" in dataset or "cylinder2d" in dataset:
        columns = 20
    steps = 5
    rows = 6 # 4
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
    flow_u = flow[:, 0, ...]
    flow_v = flow[:, 1, ...]
    flow_u_gt = flow_gt[:, 0, ...]
    flow_v_gt = flow_gt[:, 1, ...]

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
        if int((i-1)/columns) == 2: #
            # img = data_to_vis[index+int((data_to_vis.shape[0]/3)*2),...]
            img = diffs[round(index)] # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 5: # 
            img = diffs_flow[round(index)] # [index-columns*2*2,...]
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
        if int((i-1)/columns) == 3: # flow gt
            u_gt = flow_u_gt[round(index)]
            v_gt = flow_v_gt[round(index)]
            norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
            img = original_data[round(index)] # [index-columns*2*2*2]
            plt.axis('off')
            ax = plt.gca()
            pyimof.display.quiver(u_gt, v_gt, c=norm_gt, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=steps)
            # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
            data_range = min(original_data.shape[0], interpol_data.shape[0], flow_u_gt.shape[0])
            # print(data_range)
            if round(index) >= data_range:
                break
        if int((i-1)/columns) == 4: # flow pred
            u = flow_u[round(index)]
            v = flow_v[round(index)]
            norm = np.sqrt(u * u + v * v)
            img = interpol_data[round(index)] # [index-columns*2*2*2]
            plt.axis('off')
            ax = plt.gca()
            pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=steps)
            # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
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
        #     pyimof.display.quiver(u_diff, v_diff, c=norm_diff, bg=img, ax=ax, cmap='jet', bg_cmap='gray', step=steps)
        #     # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
        #     data_range = min(original_data.shape[0], interpol_data.shape[0], u_diff.shape[0])
        #     # print(data_range)
        #     if round(index) >= data_range:
        #         break

        if round(index) % factor == 0 and int((i-1)/columns) == 0:
            # add bb to the gt image
            width = data_to_vis.shape[2]
            height = data_to_vis.shape[1]
            linewidth = 2
            if "cylinder2d" in dataset:
                linewidth = 1
            rect = patches.Rectangle((0, 0), width+linewidth, height+linewidth, linewidth=linewidth, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

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
    if "droplet2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.69)
    elif "FluidSimML2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.15)
    elif "pipedcylinder2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.75)
    elif "cylinder2d" in dataset:
        plt.subplots_adjust(wspace=0.01, hspace=-0.945)
    if show:
        plt.show() 
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

def visualize_3d(volumes, dataset, dir_res="Results", title="3d_plot.html"): # (17, 128, 128, 128)
    # x1 = np.linspace(-4, 4, 9) 
    # y1 = np.linspace(-5, 5, 11) 
    # z1 = np.linspace(-5, 5, 11) 
    # X, Y, Z = np.meshgrid(x1, y1, z1)
    # values = (np.sin(X**2 + Y**2))/(X**2 + Y**2)

    x1 = np.linspace(0, 1, 128) 
    y1 = np.linspace(0, 1, 128) 
    z1 = np.linspace(0, 1, 128)
    X, Y, Z = np.meshgrid(x1, y1, z1)
    values = volumes[10, X, Y, Z]
    print(volumes[10, 50, 50, 50])
    
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        opacity=0.5,
        ))
    # fig.show()

    # fig = go.Figure(data=[go.Surface(z=volumes[0])])

    if not os.path.isdir(dir_res):
        os.makedirs(dir_res)
    dir = os.path.join(dir_res, dataset)
    dir = os.path.join(dir, title)
    fig.write_html(dir)