# -*- coding: utf-8 -*-
import os
from utils.tools import tools
import cv2
import numpy as np
from copy import deepcopy
import torch
import warnings  # ignore warnings
import torch.nn.functional as F
import torch.optim as optim
from dataset.kitti_dataset import kitti_train, kitti_flow
# from dataset.scivis_datasets import kitti_train, kitti_flow
from model.upflow import UPFlow_net
from torch.utils.data import DataLoader
import time
import argparse
import pickle

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pyimof

if_cuda = True
device = torch.device("cuda")

def visualize_series_flow(data_to_vis, flow_u, flow_v, dataset, dir_res="Results", title="Flow", show=True, save=False):
    fig=plt.figure()
    columns = 10
    rows = 10

    for i in range(1, columns*rows+1 ):
        # index = (i-1)*2 # skip eaach second
        index = i-1
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

def visualize_series_data_flow(data_to_vis, flow_u, flow_v, dataset, dir_res="Results", title="Flow", show=True, save=False):
    fig=plt.figure()
    columns = 12
    rows = 4
    print("in vis:", data_to_vis.shape, flow_u.shape)

    data_index = 0
    flow_index = 0
    for i in range(1, columns*rows+1 ):
        # index = (i-1)*2 # skip eaach second
        ax = fig.add_subplot(rows, columns, i)
        index = i - 1
        if (index >= data_to_vis.shape[0] or index >= flow_u.shape[0]):
            break
        if (index - 1) % 3 == 0:
            print("flow", index)
            # Vector field quiver plot
            u = flow_u[flow_index]
            v = flow_v[flow_index]
            norm = np.sqrt(u * u + v * v)
            img = data_to_vis[index]
            plt.axis('off')
            fig.add_subplot(rows, columns, i)
            # plt.axis('off')
            ax = plt.gca()
            pyimof.display.quiver(u, v, c=norm, bg=img, ax=ax, cmap='jet', bg_cmap='gray')
            # plt.imshow(pyimof.display.quiver(u, v, c=norm, bg=img, cmap='jet', bg_cmap='gray'), cmap='viridis')
            flow_index += 1
            ax.set_title('flow')
        else:
            print("data", index)
            img = data_to_vis[index]
            plt.axis('off')
            plt.imshow(img, vmin=data_to_vis.min(), vmax=data_to_vis.max())
            # flow_index += 1
            if index % 3 == 0:
                ax.set_title('t=0') # can we help model with info of seq?
            else:
                ax.set_title('t=1')
        
    # plt.axis('off')
    # fig = plt.gcf()
    plt.suptitle(title) 
    fig.set_size_inches(12, 9)
    if show:
        plt.show()  
    if save:
        title += ".pdf"
        if not os.path.isdir(dir_res):
            os.makedirs(dir_res)
        fig.savefig(os.path.join(dir_res, title), dpi = 300)

class Test_model(tools.abs_test_model):
    def __init__(self, pretrain_path='./train_log/upflow_piped_1.pkl'): # './scripts/upflow_kitti2015.pth'):
        super(Test_model, self).__init__()
        param_dict = {
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'if_froze_pwc': False,
            'if_use_cor_pytorch': False,  # speed is very slow, just for debug when cuda correlation is not compiled
            'if_sgu_upsample': True,
        }
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net = net_conf()  # .cuda()
        net.load_model(pretrain_path, if_relax=True, if_print=True)
        if if_cuda:
            net = net.cuda()
        net.eval()
        self.net_work = net

    # def eval_forward(self, im1, im2, gt, *args):
    def eval_forward(self, im1, im2):
        # print("in eval_forward")
        # print("images:", im1.shape, im2.shape)
        # print("Data is in range %f to %f" % (torch.min(im1), torch.max(im1)))
        # input("x")
        # === network output
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = flow_fw
        return pred_flow

    def eval_save_result(self, save_name, predflow, *args, **kwargs):
        # you can save flow results here
        print("in eval_save_result")
        # print(predflow.type)
        # print(predflow.shape)
        flow = predflow.detach().cpu().numpy()

        # img = np.zeros((original_data.shape[1], original_data.shape[2]))
        # print(flow.type)
        print(flow.shape)
        print(save_name)
        return flow


def kitti_2015_test():
    pretrain_path = './scripts/upflow_kitti2015.pth'
    # note that eval batch size should be 1 for KITTI 2012 and KITTI 2015 (image size may be different for different sequence)
    bench = kitti_flow.Evaluation_bench(name='2015_test', if_gpu=if_cuda, batch_size=1)
    testmodel = Test_model(pretrain_path=pretrain_path)
    epe_all, f1, epe_noc, epe_occ = bench(testmodel)
    print('EPE All = %.2f, F1 = %.2f, EPE Noc = %.2f, EPE Occ = %.2f' % (epe_all, f1, epe_noc, epe_occ))

def scivis_test(dataset):
    pretrain_path = './scripts/upflow_kitti2015.pth'
    # bench = kitti_flow.Evaluation_bench(name=dataset, if_gpu=if_cuda, batch_size=1)
    # testmodel = Test_model(pretrain_path=pretrain_path)
    # epe_all, f1, epe_noc, epe_occ = bench(testmodel)
    # print('EPE All = %.2f, F1 = %.2f, EPE Noc = %.2f, EPE Occ = %.2f' % (epe_all, f1, epe_noc, epe_occ))

    filename = "../FlowSciVis/Datasets/"
    if dataset == 'rectangle2d':
        filename += "rectangle2d.pkl"
        flow_fln = "../FlowSciVis/Datasets/rectangle2d_hftext_flow.pkl"
        # flow_fln = "../FlowSciVis/Datasets/rectangle2d_flow.pkl"
    if dataset == "droplet2d":
        filename += "drop2D/droplet2d_test.pkl"
    elif dataset == "pipedcylinder2d":
        filename += "pipedcylinder2d.pkl"
        flow_fln = "../FlowSciVis/Datasets/pipedcylinder2d_flow.pkl"
    elif dataset == "cylinder2d":
        filename += "cylinder2d_nc/cylinder2d.pkl"
        flow_fln = "../Datasets/cylinder2d_nc/cylinder2d_flow.pkl"
    elif dataset == "FluidSimML2d":
        filename += "FluidSimML/FluidSimML_1000_downs_data.pkl" # FluidSimML_1000
        flow_fln = "../Datasets/FluidSimML/FluidSimML_1000_downs_flow.pkl"

    # load data
    data = []
    if dataset == "rectangle2d":
        with open(flow_fln, 'rb') as flow_file:
            data = pickle.load(flow_file)
        data = np.float32(data)
        data = cv2.normalize(data, data, -1., 1., cv2.NORM_MINMAX)
        print("Data is in range %f to %f" % (np.min(data), np.max(data)))
        # print(data.shape)

        data = data[:, 0:1]
        print(data.shape)
    else:
        with open(filename, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        print(data.shape)
        data = np.float32(data)
        print("Data is in range %f to %f" % (np.min(data), np.max(data)))
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # data = data * 255.0
        # data = data.astype(int)
        # data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
        data = cv2.normalize(data, data, -1., 1., cv2.NORM_MINMAX)
        # print(data)
        print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            # input("x")
    if dataset == "pipedcylinder2d":
    #     pkl_file = open(flow_fln, 'rb')
    #     flow_uv = []
    #     flow_uv = pickle.load(pkl_file)
    #     print(flow_uv.shape)
    #     # flow_uv = (flow_uv - np.min(flow_uv)) / (np.max(flow_uv) - np.min(flow_uv))
    #     print("Flow is in range %f to %f" % (np.min(flow_uv), np.max(flow_uv)))
    #     flow_uv = np.float32(flow_uv)
        data = data[540:810]
        data = data[:, np.newaxis, ...]
        # # print(data.shape, flow_uv.shape)
        # # input("x")
        # data_flow = np.hstack((data, flow_uv))
        # print("data_flow", data_flow.shape)
        # data = data_flow
        # data = np.expand_dims(data, axis=-1)
        # print(data.shape, flow_uv.shape)
        # input("x")

    print(data.shape)
    print("Data is in range %f to %f" % (np.min(data), np.max(data)))
    input("x")

    flow_list = []
    data_list = []
    model = Test_model()

    for i in range(48):
        im1_np = data[i * 3]
        im2_np = data[i * 3 + 2]
        gt_np = data[i * 3 + 1]
        # index += 1
        # im1, im2 = batch
        print(im1_np.shape, im2_np.shape)

        # to rgb
        im1_np = np.concatenate((im1_np, im1_np, im1_np), axis=0)
        im2_np = np.concatenate((im2_np, im2_np, im2_np), axis=0)
        gt_np = np.concatenate((gt_np, gt_np, gt_np), axis=0)
        
        im1 = torch.tensor(im1_np)
        im2 = torch.tensor(im2_np)
        gt = torch.tensor(gt_np)
        im1 = im1[None, ...]
        im2 = im2[None, ...]
        gt = gt[None, ...]
        print(im1.shape, im2.shape)
        # print(im1.is_cuda)
        im1 = im1.to(device)
        im2 = im2.to(device)
        # print(im1.is_cuda)
        # input("x")
        # torch.Size([1, 3, 375, 1242])
        predflow = model.eval_forward(im1, im2) 

        save_name = ""
        flow = model.eval_save_result(save_name, predflow)
        print(flow.shape)
        flow_list.append(flow)
        data_list.extend(im1.detach().cpu().numpy())
        data_list.extend(gt.detach().cpu().numpy())
        data_list.extend(im2.detach().cpu().numpy())

    flow_arr = np.squeeze(np.array(flow_list))
    print(flow_arr.shape)
    data_arr = np.squeeze(np.array(data_list))
    print(data_arr.shape)
    flow_u = flow_arr[:, 0]
    flow_v = flow_arr[:, 1]
    # data_to_vis = np.zeros((flow_arr.shape[0], flow_arr.shape[2], flow_arr.shape[3]))
    data_to_vis = data_arr[:, 0] # one channel
    # title = "Flow_trained_" + dataset
    # visualize_series_flow(data_to_vis, flow_u, flow_v, dataset, dir_res="Results", title=title, show=False, save=True)
    title = "Data_flow_trained_" + dataset
    visualize_series_data_flow(data_to_vis, flow_u, flow_v, dataset, dir_res="Results", title=title, show=False, save=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    args = parser.parse_args()
    # assert (not args.dataset is None)
    if args.dataset:
        print(args.dataset)
        scivis_test(args.dataset)
    else:
        kitti_2015_test()
