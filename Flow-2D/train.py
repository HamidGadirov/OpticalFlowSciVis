# ssh -Y hamid@129.125.75.167
# conda activate gpu
# python3 -m torch.distributed.launch --nproc_per_node=1 train.py --world_size=1 --dataset=lbs2d --mode=train

# rsync -avz -e 'ssh' hamid@129.125.75.167:~/Desktop/OpticalFlow/RIFE/train_log/ /Users/hamidgadirov/Desktop/OpticalFlow/RIFE/train_log/

from pickletools import uint2
from matplotlib.pyplot import axes, title
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system
import matplotlib.pyplot as plt
import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import pickle
import random
import argparse
import json
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from load_datasets import load_data
from metrics import calculate_diff, calculate_avg_epe
from metrics import calculate_avg_psnr, calculate_avg_ssim, calculate_avg_lpips
from timestep_selection import calculate_dist_matrix
from utils import plot_distribution, plot_loss, plot_lr, visualize_ind, visualize_ind_flow, visualize_series, visualize_series_flow, visualize_large, \
    visualize_seq_dens, visualize_seq_flow
from baseline_interpol import *

# TODO:
# merge with 3d model (parts) -
# mixed precision model (3D) +

device = torch.device("cuda")

# exp = os.path.abspath('.').split('/')[-1]
log_path = 'train_log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        # return 6e-5 * mul
        # return 1e-4 * mul
        # return 3e-4 * mul # # orig
        # return 4e-4 * mul
        return 6e-4 * mul
        # return 8e-4 * mul
        # return 1e-3 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        # return (6e-5 - 6e-6) * mul + 6e-6
        # return (1e-4 - 1e-5) * mul + 1e-5
        # return (3e-4 - 3e-6) * mul + 3e-6
        # return (3e-4 - 3e-5) * mul + 3e-5 # orig
        # return (4e-4 - 4e-6) * mul + 4e-6
        return (6e-4 - 6e-6) * mul + 6e-6
        # return (6e-4 - 6e-5) * mul + 6e-5
        # return (8e-4 - 8e-6) * mul + 8e-6
        # return (1e-3 - 1e-5) * mul + 1e-5
    # return 1e-6 # worse
    # return 2e-7 # no
    # return 2e-6 # no
    # mul = step / 2000.
    # return 3e-4 * mul
    # return 3e-5
    # return 3e-6
    # return 3e-3
    # mul = np.cos((step) / (args.epoch * args.step_per_epoch) * math.pi) * 0.5 + 0.5
    # return (1e-4 - 1e-5) * mul + 1e-5

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

# early_stop_patience = 1001
# early_stop_k = 0
# val_loss_best = 0.

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return 2
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return 1
        return 0

def train(model, dataset, factor, model_name, mode, local_rank):
    # if local_rank == 0:
        # writer = SummaryWriter('train')
        # writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0

    num_workers = 1 # 8
    if mode == "train":
        if dataset == "vimeo2d": 
            dataset_train = VimeoDataset('train')
            sampler = DistributedSampler(dataset_train)
            train_data = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=sampler)
            args.step_per_epoch = train_data.__len__()
            dataset_val = VimeoDataset('validation')
            val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=num_workers)
        else:
            data_train, data_val = load_data(dataset, factor, mode)
            # print(data_train.shape)
            # sampler = DistributedSampler(data_train)
            # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
            train_data = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
            args.step_per_epoch = train_data.__len__()
            # # input("train loaded...")
            print("len:", len(data_train))
            print(data_train[0].shape)
            # input("train loaded")
            val_data = DataLoader(data_val, args.batch_size, pin_memory=True, num_workers=num_workers)
            print("len:", len(val_data))
    else: # mode == "test"
        if dataset == "vimeo2d": 
            dataset_test = VimeoDataset('test')
            data_test = DataLoader(dataset_test, batch_size=16, pin_memory=True, num_workers=num_workers)
        else:
            if dataset == "lbs2d":
                # data_test, flow_x_mean, flow_y_mean, flow_x_std, flow_y_std = load_data(dataset, factor, mode)
                data_test, dens_min, dens_max, dens_mean, dens_std = load_data(dataset, factor, mode)
            else:
                data_test, data_test_non_overlap = load_data(dataset, factor, mode)
            shuffle = False
            # if dataset == "droplet2d": 
            #     shuffle = True
            data_test = DataLoader(data_test, batch_size=16, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    # model_name = "flownet"
    # model_name += "_lapl" if lapl_loss else ""
    # model_name += "_reg" if l1_reg else ""
    # model_name += "_smooth" if smooth_loss else ""
    # model_name += "_photo" if photo_loss else ""
    # model_name += "_3rd" if each_third else ""
    # model_name += "_aug" if aug else ""
    # model_name += "_" + dataset
    # model_name += ".pkl"

    # model.load_model(model_name, log_path)
    try:
        model.load_model(model_name, log_path)
        # print("Loaded RIFE model:", model_name)
        # print(model) # RIFE is not a neural net
    except:
        print("No weights found, training from scratch.")
        print("model name:", model_name)
    # input("x")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # num_params = count_parameters(model_name)
    # print(num_params)
    # input("x")

    if mode == "train":
        print('training...')
        time_stamp = time.time()
        early_stopper = EarlyStopper(patience=30, min_delta=0.01)
        for epoch in range(args.epoch):
            # sampler.set_epoch(epoch)
            for i, data in enumerate(train_data):
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                # data = torch.zeros([16, 9, 150, 450], dtype=torch.float)
                if dataset == "vimeo2d": 
                    # print("vimeo2d")
                    data_gpu, timestep = data # vimeo
                    data_gpu = data_gpu.to(device, non_blocking=True) / 255.
                else:
                    data_gpu = data
                    data_gpu = data_gpu.to(device, non_blocking=True)
                # data_gpu = data_gpu.permute(0, 3, 1, 2)
                # print("data to gpu:", data_gpu.shape) # b_s, imgs-gt, data-flow. x. y
                # input("x")
                # print(type(data_gpu[0,0,0,0]))
                # print(data_gpu[0, 0])
                # input("data")
                # imgs = data_gpu[:, :2] # :6
                # gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network
                # # print("Training")
                # print(imgs.shape)
                # print(gt.shape)
                # input("x")
                # learning_rate = get_learning_rate(step)
                learning_rate = get_learning_rate(step) * args.world_size / 4
                # save learning_rate to json and plot during inference
                if factor == 2: # # img0, img1, gt
                    imgs = data_gpu[:, :2] # :6
                    gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network
                    # print(imgs.dtype)
                    # gt = data_gpu # use flows from im0 and im1
                    # print(imgs.shape)
                    # print(gt.shape)
                    # input("x")
                    # with torch.cuda.amp.autocast(True):
                    pred, info = model.update(imgs, gt, dataset, learning_rate, training=True)

                    # from torchviz import make_dot
                    # make_dot(pred, params=dict(list(model.named_parameters()))).render("flownet2d_torchviz", format="png")
                    # input("torchviz")

                elif factor == 4: # img0, img1, gt0, gt1, gt2
                    imgs = data_gpu[:, :2]
                    gt = data_gpu[:, 2:5] # gt0, gt1, gt2
                    # print("imgs:", imgs.shape)
                    # print("gt:", gt[:, 1:2].shape)
                    # input("x")
                    pred, info = model.update(imgs, gt[:, 1:2], dataset, learning_rate, training=True) # img0, img1, gt1

                    imgs = torch.cat((data_gpu[:, :1], gt[:, 1:2]), 1)
                    print(imgs.shape)
                    pred, info = model.update(imgs, gt[:, 0:1], dataset, learning_rate, training=True) # img0, gt1, gt0

                    imgs = torch.cat((gt[:, 1:2], data_gpu[:, :1]), 1)
                    print(imgs.shape)
                    pred, info = model.update(imgs, gt[:, 2:3], dataset, learning_rate, training=True) # gt1, img1, gt2
                # if dataset == "pipedcylinder2d" or dataset == "cylinder2d" or dataset == "FluidSimML2d":
                #     imgs = data_gpu[:, :2, 0]
                #     gt = data_gpu[:, 2:3, 0]
                train_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                if step % 200 == 1 and local_rank == 0:
                    print("step 200")
                    # writer.add_scalar('learning_rate', learning_rate, step)
                    # writer.add_scalar('loss/l1', info['loss_l1'], step)
                    # writer.add_scalar('loss/tea', info['loss_tea'], step)
                    # writer.add_scalar('loss/distill', info['loss_distill'], step)
                if step % 1000 == 1 and local_rank == 0:
                    print("step 1000")
                    # gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    # flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                    # for i in range(5):
                    #     print(pred.shape, gt.shape)
                        # print(merged_img.shape, pred.shape, gt.shape)
                        # imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                        # writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                        # writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                        # writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                    # writer.flush()
                if local_rank == 0:
                    print('epoch:{}/{} {}/{} time:{:.2f}+{:.2f} loss_G:{:.4e}' \
                        .format(epoch, args.epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_G']))
                    # print('epoch:{}/{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}' \
                    #     .format(epoch, args.epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
                step += 1
                # evaluate(model, dataset, val_data, step, local_rank) #, writer_val)   
            nr_eval += 1
            # if nr_eval % 5 == 0:
            #     # evaluate(model, val_data, step, local_rank)
            #     evaluate(model, val_data, step, local_rank, writer_val)
            # model.save_model(model_name, log_path, local_rank) # save after checking early stopping
            # input(epoch)
            if epoch == 0:
                print("epoch == 0")
                global val_loss_best 
                val_loss_best = info['loss_G']
                print(val_loss_best)
            evaluate(model, dataset, val_data, step, local_rank, learning_rate, early_stopper) #, writer_val)   
            # model.save_model(model_name, log_path, local_rank) 
            dist.barrier()
    else:
        print("inference...")
        # factor = 2
        # print(factor)
        # input("factor")
        dir_res = "Results"
        dir_res = os.path.join(dir_res, dataset)
        dir_res = os.path.join(dir_res, str(factor) + "x")
        dir_model = os.path.join(dir_res, model_name[:-4])
        # print(dir_model)
        # input("x")
        print("Saving at:", dir_model)
        # data = data_train
        # scale_list = [4, 2, 1]
        # scale_list = [1, 1, 1]
        # print(data.shape)
        # input("x")

        # for i in range(0, data.shape[0], 2):
        #     I0 = data[i].to(device, non_blocking=True) / 255.
        #     I1 = data[i+1].to(device, non_blocking=True) / 255.
        #     middle, flow, mask = model.inference(I0, I1, scale_list) # get the flow too
        #     print(middle.shape)
        interpol_data = []
        flow_combined = []
        mask_combined = []
        flow_gt_combined = []
        # data_test_combined = []
        for i, data in enumerate(data_test):
            # print("test len:", len(data_gpu))
            # print("data:", data.shape)
            # input("x")
            # data = np.expand_dims(data, axis=0)
            if dataset == "vimeo2d":
                # print("vimeo2d")
                data_gpu, timestep = data
                # for b in range(data_gpu.shape[0]):
                #     data_test_combined.append(np.asarray(data_gpu[b, 0] / 255.))
                #     data_test_combined.append(np.asarray(data_gpu[b, 2] / 255.))
                #     data_test_combined.append(np.asarray(data_gpu[b, 1] / 255.))
                # data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            elif dataset == "droplet2d":
                # print("droplet2d")
                # print(data.dtype)
                # input("x")
                data = data.float() # because it was double
                data_gpu = data
                # for b in range(data_gpu.shape[0]):
                #     data_test_combined.append(np.asarray(data_gpu[b, 0] ))
                #     data_test_combined.append(np.asarray(data_gpu[b, 2] ))
                #     data_test_combined.append(np.asarray(data_gpu[b, 1] ))
                # data_gpu = data_gpu.to(device, non_blocking=True) 
            else:
                data_gpu = data
                # for b in range(data_gpu.shape[0]):
                #     data_test_combined.append(np.asarray(data_gpu[b, 0, 0]))
                #     data_test_combined.append(np.asarray(data_gpu[b, 2, 0]))
                #     data_test_combined.append(np.asarray(data_gpu[b, 1, 0]))
                # data_gpu = data_gpu.to(device, non_blocking=True)
            # print("data to gpu:", data_gpu.shape)
            # input("data")
            imgs = data_gpu[:, :2]
            gt = data_gpu[:, 2:3]
            print(imgs.shape)
            print(gt.shape)
            # gt = data_gpu

            if dataset == "pipedcylinder2d"  or dataset == "cylinder2d" or dataset == "FluidSimML2d" \
                or dataset == "rectangle2d" or dataset == "lbs2d":
                flow_gt = data_gpu[:,:,1:3] # only flow in x and y
                # input(flow_gt.shape)
                flow_gt_array = np.asarray(flow_gt.detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    flow_gt_combined.append(flow_gt_array[b, 0])
                    flow_gt_combined.append(flow_gt_array[b, 2])
                    flow_gt_combined.append(flow_gt_array[b, 1])

                imgs = imgs[:, :, 0]
                gt = gt[:, :, 0]

            # print(imgs.shape)
            # print(gt.shape)
            # input("imgs gt")
            I0 = imgs[:, :1]
            I1 = imgs[:, 1:2]

            # arbitrary interpol
            print(f"inference for {factor}x interpolation")
            for j in range(1, factor):
                interpol_step = float(j) / factor
                print(interpol_step)
                merged, flow, mask = model.inference(I0, I1, t=interpol_step)
                merged_last = np.asarray(merged[-1].detach().cpu().numpy()) # from last block
                print(merged_last.shape)

                fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
                axs = axs.flatten()
                # Iterate over the images and plot them on the subplots
                for i in range(16):
                    # Remove the single channel dimension if it exists (assuming grayscale images)
                    image = merged_last[i].squeeze()
                    
                    # Plot the image on the corresponding subplot
                    axs[i].imshow(image, cmap='gray')
                    axs[i].axis('off')

                # Adjust the spacing between the subplots
                plt.tight_layout()

                # Show the plot
                plt.show()
                fig.savefig("droplets.png", dpi = 300)

                input("x")
                for b in range(data_gpu.shape[0]):
                    interpol_data.append(merged_last[b].transpose((1,2,0)))

                flow_last = np.asarray(flow[-1].detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    flow_combined.append(flow_last[b])

                mask_last = np.asarray(mask[-1].detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    mask_combined.extend(mask_last[b])
            # input("inf")
            """
            if factor == 2: # # img0, img1, gt
                # t = 0.5
                # merged, flow, mask = model.inference(I0, I1, scale_list) # get the flow too
                # merged, flow, mask = model.inference(I0, I1, np.squeeze(data_gpu[:, 2:3, 1:3]), scale_list) # get the flow too
                # merged, flow, mask = model.inference(I0, I1, np.squeeze(data_gpu[:, 2:3, 1:3]))
                merged, flow, mask = model.inference(I0, I1)
                # merged, flow, mask = model.inference(I0, I1, t)
                merged_last = np.asarray(merged[-1].detach().cpu().numpy()) # from last block
                # print(merged_last.shape)
                # input("x")
                # interpol_data.extend(middle.transpose((0,2,3,1))) # append
                for b in range(data_gpu.shape[0]):
                    interpol_data.append(merged_last[b].transpose((1,2,0)))
                # input("x")
                flow_last = np.asarray(flow[-1].detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    flow_combined.append(flow_last[b])
                # print(flow_l.shape)
                # input("x")
                # flow_array_ = flow[2].detach().cpu().numpy()
                # flow_combined.extend(flow_l[:,0:4,:,:].squeeze())
                # flow_combined.extend(flow_m[:,0:4,:,:].squeeze())
                # flow_combined.extend(flow_r[:,0:4,:,:].squeeze()) # 0:4 because in x and y for Ft->0 and Ft->1 intermediate flows
                # print(flow.shape)
                mask_last = np.asarray(mask[-1].detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    mask_combined.extend(mask_last[b])
            elif factor == 4: # img0, img1, gt0, gt1, gt2
                middle, flow_middle, mask_middle = model.inference(I0, I1) # get the flow too
                middle = middle[-1]
                print(I1.shape, middle.shape)
                middle_left, flow_left, mask_left = model.inference(I0, middle) # get the flow too
                middle_right, flow_right, mask_right = model.inference(middle, I1) # get the flow too

                middle = middle.detach().cpu().numpy()
                middle_left = middle_left[-1].detach().cpu().numpy()
                middle_right = middle_right[-1].detach().cpu().numpy()
                print("middle:", middle.shape)
                print("middle_left:", middle_left.shape)

                flow_left = flow_left[-1].detach().cpu().numpy()[:, 0:4, :, :].squeeze()
                flow_middle = flow_middle[-1].detach().cpu().numpy()[:, 0:4, :, :].squeeze()
                flow_right = flow_right[-1].detach().cpu().numpy()[:, 0:4, :, :].squeeze()

                mask_middle = mask_middle[-1].detach().cpu().numpy()
                mask_left = mask_left[-1].detach().cpu().numpy()
                mask_right = mask_right[-1].detach().cpu().numpy()
                # print("mask:", mask.shape)

                for b in range(data_gpu.shape[0]):
                    interpol_data.extend([middle_left[b].transpose((1, 2, 0)),
                                        middle[b].transpose((1, 2, 0)),
                                        middle_right[b].transpose((1, 2, 0))])
                    
                    flow_combined.extend([flow_left[b], flow_middle[b], flow_right[b]])
                    mask_combined.extend([mask_left[b], mask_middle[b], mask_right[b]])

                print(flow_middle.shape)
                # input("factor2") 
                """

        interpol_data = np.squeeze(np.array(interpol_data))
        print("interpolated:", interpol_data.shape)

        # reorder based on factor
        if factor > 2:
            interpol_data_new = []
            batch_size = 16 # what was used for inference
            try:
                for fb in range(0, interpol_data.shape[0], (factor - 1) * batch_size): # 0 32 64 ...
                    print("fb", fb)
                    for b in range(batch_size): # 0 1; 0 1; ...
                        for k in range(fb, fb + (factor - 1) * batch_size, batch_size): # 0 16; 32 64; ... 
                            print("k+b:", k + b)
                            interpol_data_new.append(interpol_data[k + b])
            except:
                print("reordered")
            interpol_data = np.squeeze(np.array(interpol_data_new))
            print("interpolated:", interpol_data.shape)

        # title = "inference_" + str(factor) + "x"
        # if "2d" in dataset:
        #     visualize_series(interpol_data, factor, dataset, dir_res, title=title, show=False, save=True)

        # Vector field quiver plot
        flow_combined = np.array(flow_combined)
        mask_combined = np.squeeze(np.array(mask_combined))
        flow_gt_combined = np.array(flow_gt_combined)
        print("Flow:", flow_combined.shape)
        # print(interpol_data.dtype)
        flow_u = flow_combined[:, 2, ...]
        flow_v = flow_combined[:, 3, ...]
        title = "inference_flow_arrows_" + str(factor) + "x"
        # if "2d" in dataset:
        #     visualize_series_flow(interpol_data, flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)

        if factor == 2:
            loss_path = 'loss.json'
            loss_path = os.path.join(dir_model, loss_path)
            try:
                with open(loss_path, 'r') as loss_file:
                    loss_data = json.load(loss_file)
                val_loss = loss_data['val_loss']
                loss_file.close()   
                # from collections import Iterable
                # def flatten(lis):
                #     for item in lis:
                #         if isinstance(item, Iterable) and not isinstance(item, str):
                #             for x in flatten(item):
                #                 yield x
                #         else:        
                #             yield item
                # val_loss = list(flatten(val_loss))
                val_loss = np.array(val_loss)
            except:
                print("loss json doesn't exist")
            plot_loss(val_loss, dir_model, name="val_loss.png", save=True)

            lr_file_name = 'learning_rate.txt'
            lr_path = os.path.join(dir_model, lr_file_name)
            try:
                with open(lr_path, "r") as lr_file:
                    lr_data = lr_file.read()
                    lr_data_into_list = lr_data.replace('\n', ' ').split(' ')
                learning_rate = [float(x) for x in lr_data_into_list[:-1]]
                learning_rate = np.array(learning_rate)
                plot_lr(learning_rate, dir_model, name="learning_rate.png", save=True)
            except:
                print("learning rate file doesn't exists")
        else:
            print("learning rate file is in 2x dir")

        # original_data = np.array(data_test_combined)
        # print("original_data:", original_data.shape)
        # if dataset == "pipedcylinder2d" or dataset == "cylinder2d" or dataset == "FluidSimML2d" or dataset == "rectangle2d":
        #     original_data = data_test[:, :, 0] # gt
        # else:
        #     original_data = data_test
        # print("Data is in range %f to %f" % (np.min(original_data), np.max(original_data)))
        input("x")
        
        if dataset == "lbs2d":
            # statistics = {'flow_x_mean': flow_x_mean, 'flow_y_mean': flow_y_mean, 'flow_x_std': flow_x_std, 'flow_y_std': flow_y_std}
            statistics = {'dens_min': dens_min, 'dens_max': dens_max, 'dens_mean': dens_mean, 'dens_std': dens_std}
        else:
            statistics = {'flow_x_mean': 0, 'flow_y_mean': 0, 'flow_x_std': 0, 'flow_y_std': 0}
        return data_test_non_overlap, interpol_data, flow_combined, flow_gt_combined, mask_combined, statistics
    
def evaluate(model, dataset, val_data, nr_eval, local_rank, learning_rate, early_stopper): #, writer_val):
    print("Evaluate")
    # input(model_name)
    # writer = SummaryWriter('train') #
    # writer_val = SummaryWriter('validate') #
    # psnr_list = []
    # psnr_list_teacher = []
    loss_all_list = [0. for i in range(8)] # all loss components 7
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        # data = torch.zeros([16, 9, 150, 450], dtype=torch.float)
        if dataset == "vimeo2d": 
            print("vimeo2d")
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        else:
            data_gpu = data
            data_gpu = data_gpu.to(device, non_blocking=True)
        # data_gpu = data_gpu.permute(0, 3, 1, 2)
        # print("data to gpu:", data_gpu.shape)
        # input(i)
        imgs = data_gpu[:, :2]
        gt = data_gpu[:, 2:3]
        # gt = data_gpu
        with torch.no_grad():
            pred, info = model.update(imgs, gt, dataset, training=False)
            merged_img = info['merged_tea']
        # loss_l1_list.append(info['loss_l1'].cpu().numpy())
        # loss_tea_list.append(info['loss_tea'].cpu().numpy())
        # loss_distill_list.append(info['loss_distill'].cpu().numpy())
        # for key, value in info.items():
        #     print (key, value)
        # input("x")
        loss_all = [info['loss_G'].cpu().numpy(), info['loss_l1'].cpu().numpy(), info['loss_tea'].cpu().numpy(), 
            info['loss_distill'].cpu().numpy(), info['l1_reg'].cpu().numpy(), info['loss_photo'].cpu().numpy(), 
            # info['loss_smooth'].cpu().numpy(), 
            info['loss_flow'].cpu().numpy()]
        print("loss_G:", info['loss_G'])
        loss_all_list = [a + b for a, b in zip(loss_all_list, loss_all)]
        # for j in range(gt.shape[0]):
        #     max_shape_2 = min(gt.shape[2], pred.shape[2])
        #     max_shape_3 = min(gt.shape[3], pred.shape[3])
        #     # print(max_shape_2, max_shape_3)
        #     gt = gt[:,:,:max_shape_2,:max_shape_3]
        #     pred = pred[:,:,:max_shape_2,:max_shape_3]
        #     psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
        #     psnr_list.append(psnr)
        #     psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
        #     psnr_list_teacher.append(psnr)
        # gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        # flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        # if i == 0 and local_rank == 0:
        #     for j in range(10):
        #         imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
        #         writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    # early stopping
    # val_loss_now = info['loss_G']
    # global val_loss_best 
    # global early_stop_k
    # if val_loss_now > val_loss_best:
    #     print("val_loss_now > val_loss_best:", val_loss_now, val_loss_best)
    #     early_stop_k += 1
    #     if early_stop_k == early_stop_patience:
    #         print("Early stopping")
    #         sys.exit(0)
    # else:
    #     print("val_loss_now < val_loss_best", val_loss_now, val_loss_best)
    #     early_stop_k = 0
    #     val_loss_best = val_loss_now
    #     # input(model_name)
    #     model.save_model(model_name, log_path, local_rank) 

    loss_all = [x / float(i + 1) for x in loss_all_list] # avg over batches
    # input(loss_all)
    validation_loss = loss_all[0] # info['loss_G'].cpu().numpy()
    if early_stopper.early_stop(validation_loss) == 2: 
        model.save_model(model_name, log_path, local_rank) 
        # input("saved")
    elif early_stopper.early_stop(validation_loss) == 1:   
        print("Early stop.")          
        sys.exit(0)

    val_loss = []
    # loss_all = np.array(loss_all).tolist()
    val_loss.append(loss_all)
    # print("loss_all", loss_all)
    loss_path = 'loss.json'
    factor = 2
    dir_res = "Results"
    dir_res = os.path.join(dir_res, dataset)
    dir_res = os.path.join(dir_res, str(factor) + "x")
    dir_model = os.path.join(dir_res, model_name[:-4])
    if not os.path.isdir(dir_model):
        os.makedirs(dir_model)
    # print(dir_model)
    # input("x")
    loss_path = os.path.join(dir_model, loss_path)
    loss_data = {'val_loss': val_loss}

    # load previous loss values if they exist
    if (os.path.exists(loss_path)):
        loss_file = open(loss_path, 'r')
        loss_data_old = json.load(loss_file)
        loss_data_old['val_loss'].extend(val_loss)
        loss_data = loss_data_old
        # print("exists:", loss_data)
        loss_file.close()
    # print("loss_data:", loss_data)
    # input("x")
    with open(loss_path, 'w+') as loss_file:
        json.dump(loss_data, loss_file, indent=4)
        # print("dump loss to json")
    loss_file.close()

    lr_file_name = 'learning_rate.txt'
    lr_path = os.path.join(dir_model, lr_file_name)
    lr_file = open(lr_path, 'a')
    lr_file.write(str(learning_rate) + "\n")
    lr_file.close()

    # lr_path = 'learning_rate.json'
    # lr_path = os.path.join(dir_model, lr_path)
    # if (os.path.exists(lr_path)):
    #     lr_file = open(lr_path, 'r')
    #     lr_data_old = json.load(lr_file)
    #     lr_data_old.extend(learning_rate)
    #     learning_rate = lr_data_old
    #     lr_file.close()
    # with open(lr_path, 'w') as lr_file:
    #     json.dump([learning_rate], lr_file, indent=4)
    # lr_file.close()

    eval_time_interval = time.time() - time_stamp

    del loss_all

    if local_rank != 0:
        return
    # writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":    

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int) # 300 500 1000
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size') # default=16
    parser.add_argument('--local_rank', default=0, type=int, help='local rank') # =local_rank local-rank
    parser.add_argument('--world_size', default=1, type=int, help='world size') # 4
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    parser.add_argument('--mode', dest='mode', type=str, default='test')
    # parser.add_argument('--exp', dest='exp', type=int, default=1)
    parser.add_argument('--factor', dest='factor', type=int, default=2)
    args = parser.parse_args()
    assert (not args.dataset is None)

    # # Remove the local_rank argument
    # if '--local_rank' in sys.argv:
    #     parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    #     args = parser.parse_args()
    #     sys.argv.remove('--local_rank')
    # else:
    #     args = parser.parse_args()

    # args.local_rank = os.environ['LOCAL_RANK']

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    if args.dataset == "vimeo2d" and args.mode == "train":
        args.batch_size = 55
    if args.dataset == "droplet2d":
        args.batch_size = 40
    if args.dataset == "rectangle2d":
        args.batch_size = 100 # 200
    if args.dataset == "lbs2d":
        args.batch_size = 40 # 20 40 55 120
    if args.dataset == "FluidSimML2d":
        args.batch_size = 64
    if args.dataset == "pipedcylinder2d" or args.dataset == "cylinder2d":
        args.batch_size = 64

    """ rectangle2d """
    # model_name = "flownet_lapl_aug_rect.pkl" # bad
    # model_name = "flownet_lapl_aug_rect_255.pkl" # bad
    # model_name = "flownet_lapl_reg_smooth_aug_rect.pkl" # bad
    # model_name = "flownet_l1_aug_rect.pkl"
    # model_name = "flownet_lapl_t_aug_rect.pkl" # bad # loss_distill 0 and nan
    # model_name = "flownet_lapl_photo_aug_rect.pkl" # increased to 15%, 100 ep better than before, but underfit
    # model_name = "flownet_lapl_nodistill_photo_aug_rect.pkl"
    # model_name = "flownet_lapl_nodistill_reg_smooth_photo_aug_rect2.pkl"
    # model_name = "flownet_lapl_modif90_rect.pkl" # good, loss_G = loss_l1 * 10 + loss_tea * 10 + loss_distill * 0.01 + l1_reg + loss_photo * 0.001 + loss_smooth * 0.001
    # model_name = "flownet_lapl1_modif90_rect.pkl" # good 100 ep
    # model_name = "flownet_lapl1_modif64_rect.pkl" # good, correct flow dir, 150-200 ep
    # model_name = "flownet_lapl1_v2_64_rect.pkl" # good, correct flow dir, 150-200 ep - all loss
    # model_name = "flownet_lapl1_all_v2_64_rect.pkl" # not bad but worse than before (64 instead of 128)
    # model_name = "flownet_lapl1_l1only_v2_64_rect.pkl" # not accurate flow, 200 ep
    # model_name = "flownet_lapl1_modif90_scale1_rect.pkl" # good but flow moving in all directions 50 ep
    model_name = "flownet_lapl_all_v2_64_rect.pkl" # good but smooth not enough, wrong flow withing object
    model_name = "flownet_lapl_all_smooth01_v2_64_rect.pkl" # bad flow, too much smooth
    model_name = "flownet_lapl_all_smooth001_v2_64_rect.pkl"
    model_name = "flownet_flowonly_v2_64_rect.pkl"
    model_name = "flownet_flowonly1_v2_64_rect.pkl"
    model_name = "flownet_flowonly_v2_128_rect.pkl"
    model_name = "flownet_flowonly_v2_128_rect_text.pkl" # bad, smth wrong with flow
    model_name = "flownet_lapl_dist_photo_v2_64_rect_text.pkl"
    model_name = "flownet_lapl_dist_v2_64_rect_text.pkl" # 255 range bad
    model_name = "flownet_lapl_dist_v2_64_rect_text1.pkl" # 1 range better, flow not accurate
    model_name = "flownet_lapl_dist_photo_v2_64_rect_text1.pkl" # 1 range better, flow not accurate
    model_name = "flownet_lapl_dist_photo_reg_v2_64_rect_text1.pkl" # 1 range better, flow not accurate
    model_name = "flownet_lapl_dist_photo_reg_v2_256_rect_text1.pkl" # 1 range better, flow not accurate
    model_name = "flownet_lapl_dist_photo_reg_refine_v2_64_rect_text1.pkl" # nothing new
    model_name = "flownet_lapl_dist_photo_reg_refine_v1_64_rect_text1.pkl" # worse than v2
    model_name = "flownet_lapl_refine_v2_128_rect_text1.pkl" # ? good 600 ep
    # model_name = "flownet_lapl_refine_test4x_v2_128_rect_text1.pkl" # 4x interpol
    model_name = "flownet_lapl_dist_photo_v2_128_rect_text1.pkl" # ?
    model_name = "flownet_lapl_dist_photo_refine_v2_128_rect_text1.pkl" # ?
    model_name = "flownet_lapl_dist_reg_photo_refine_v2_128_rect_text1.pkl" # ? interpol good, flow could be better
    model_name = "flownet_lapl_dist_reg_photo_refine_v2_128_rects_text.pkl" # good interpol, inaccurate flow
    model_name = "flownet_flow_lapl_dist_reg_photo_refine_v2_128_rects_text.pkl"
    # model_name = "flownet_lapl_dist_reg_photo1e-4_refine_v2_128_rects_text.pkl" # rects flow bad 1000 ep
    model_name = "flownet_lapl_dist_reg_photo1e-4_refine_v2_128_rect_text.pkl"

    model_name = "flownet_lapl_dist_refine_v2_128_rect_hftext.pkl" # 300 ep bad flow; 1000ep improved?
    # model_name = "flownet_lapl_dist_photo1e-5_refine_v2_128_rect_hftext.pkl" # 1000 ep good flow
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext.pkl" # 1000 ep ~better flow
    model_name = "flownet_lapl_dist_reg1e-4_photo1e-5_refine_v2_128_rect_hftext.pkl" # 800 ep: bad, loss inc
    model_name = "flownet_lapl_dist_reg1e-5_photo1e-5_refine_v3_128_rect_hftext.pkl" # 800 ep: bad, loss inc
    model_name = "flownet_lapl_dist_refine_v3_128_rect_hftext.pkl" # no, don't use 5 blocks
    model_name = "flownet_lapl_dist_reg1e-5_photo1e-5_refine_v2_128_rect_hftext.pkl" # with range; bad
    model_name = "flownet_lapl_dist_refine_v2_128_rect_hftext_range.pkl" # flow bad
    # model_name = "flownet_lapl_dist_refine_reg1e-5_photo1e-5_v2_128_rect_hftext_range.pkl" 1000ep, 3579 range, bad
    # model_name = "flownet_lapl_dist_refine_v2_128_rect_testloss.pkl"
    # model_name = "flownet_lapl_dist_refine_v2_128_rect_hftext_range357shift.pkl"
    model_name = "flownet_lapl_dist_refine_v2_128_rect_hftext_range.pkl" # 2000ep, 3579 range
    model_name = "flownet_lapl_dist_reg1e-5_photo1e-5_refine_v2_128_rect_hftext_bugfix.pkl" # reg loss too high
    model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_bugfix.pkl" # reversed flow and mask

    model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_range.pkl" # 3579 bad 1e-5 reg too much
    model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_range357.pkl" # flow inaccurate
    model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_range3579.pkl" # not good

    model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_1.pkl" 
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_revflow.pkl" 
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_refine_v2_128_rect_hftext_revflowmask.pkl" 
    # model_name = "flownet_lapl_dist_refine_v2_128_rect_hftext_1.pkl" 
    # model_name = "flownet_lapl_dist_refine_v2_128_rect_hftext_range3579.pkl" # flow not accurate, loss jump
    # desired unsupervised flow wasn't achieved, switching to UPFlow... 02.09.22

    model_name = "flownet_lapl_dist_photo1e-5_refine_v2_128_rect_big_hftext.pkl" # no
    model_name = "flownet_lapl_dist_refine_v2_128_rect_big_hftext.pkl" # no
    model_name = "flownet_flowonly_v2_128_rect_big_hftext.pkl" # bad interpol, flow could be better
    model_name = "flownet_lapl_dist_flow_v2_128_rect_big_hftext.pkl" # slowly converges?
    # model_name = "flownet_lapl_flow_v2_128_rect_big_hftext.pkl" # bad
    model_name = "flownet_lapl_dist2e-2_flow2e-1_v2_128_rect_big_hftext.pkl" # not so good, v1
    model_name = "flownet_lapl_dist2e-2_flow2e-1_v2_128_rect_big_hftext_v2.pkl" # v2
    # distill loss increase: why? -> we have flow loss now -> but i apply it to last layer only?
    # check Ft->0 and Ft->1 and to which I apply flow gt
    model_name = "flownet_lapl_flow2e-1_v2_128_rect_big_hftext.pkl" # v2
    model_name = "flownet_lapl_dist2e-2_flow2e-1_v2_128_rect_big_hftext_v3.pkl" # v3: flow loss fix
    model_name = "flownet_lapl_dist2e-2_flow2e-1_v2_128_rect_big_hftext_v4.pkl" # v4: flow loss to all blocks
    model_name = "flownet_lapl_dist2e-2_flow2e-1_v2_128_rect_big_hftext_v4_inv.pkl"

    model_name = "flownet_flowonly_v2_128_rect_big_hftext_allBlocks.pkl" #
    model_name = "flownet_lapl_dist2e-2_flow2e-1_v2_128_rect_big_hftext_allBlocks.pkl" #
    model_name = "flownet_lapl_dist2e-2_reg1e-7_photo1e-5_flow2e-1_v2_128_rect_big_hftext_allBlocks.pkl" #
    model_name = "flownet_lapl_dist2e-2_reg1e-7_photo1e-5_flow2e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" #
    model_name = "flownet_lapl_flow2e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" #
    model_name = "flownet_lapl_flow2e-1_v2_128_rect_big_hftextv3_allBlocks_lr.pkl" # no, fixed lr is bad
    model_name = "flownet_lapl_flow2e-1_v2_128_rect_big_hftextv3_allBlocks_norm.pkl" #
    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_normv2.pkl" #
    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_normv3.pkl" # best supervised!
    model_name = "flownet_lapl_dist2e-2_reg1e-7_photo1e-5_v2_128_rect_big_hftext.pkl" # good interpol, very bad flow
    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_lastBlock_normv3.pkl" # not bad but allBlocks better
    # why unsupervised interpolation better ???
    # add separate layer for supervised flow
    model_name = "flownet_lapl_dist2e-2_reg1e-7_photo1e-5_flow5e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" # distill exploded
    model_name = "flownet_lapl_reg1e-7_photo1e-5_flow5e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" # best interpol, bad flow

    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_nohf_allBlocks.pkl" # bad interpol, very bad flow ???

    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocksv2_normv3.pkl" # blurred flow, not accurate interpol
    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocksEq.pkl" # same as prev: best supervised ? no ???
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocksEq1.pkl" # 1 1 1 is bad
    model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocksEq2.pkl" # 370 ep: not bad +1K: get res again:
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_last2Blocks.pkl" # better than only last
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_lastBlocks.pkl" # - worse than last 2 blocks
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_teaBlocks.pkl" # - blurred flow
    # model_name = "flownet_lapl_flow_v2_128_rect_big_hftextv3_allBlocks.pkl" # good flow, interpol can be better; flow coeff too high?
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocks124.pkl" # 124 weights: good interpol, flow can be more accurate
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_first2Blocks.pkl" # very good interpol, flow not accurate
    # model_name = "flownet_lapl_flow5e-1_1e-1_v2_128_rect_big_hftextv3_first2Blocks_finetuneLastBlock.pkl" # finetuning:
    # model_name = "flownet_lapl_flow1e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" # good interpol, flow can be better
    # model_name = "flownet_lapl_flow2e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" # good interpol, flow can be better
    # model_name = "flownet_lapl_flow3e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" # good interpol, flow can be better
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_GTflowInterpol.pkl" # good interpol but can be better; good/very good flow ?
    # model_name = "flownet_l1_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_GTflowInterpol.pkl" # good interpol, very good flow
    # model_name = "flownet_l1_flow25e-2_v2_128_rect_big_hftextv3_allBlocks_GTflowInterpol.pkl" # again good interpol, very good flow
    # model_name = "flownet_l1_flow25e-2_v2_128_rect_big_hftextv3_allBlocks.pkl" # ? better interpol, good flow, can be better?
    model_name = "flownet_l1_flow5e-1_v2_128_rect_big_hftextv3_allBlocks.pkl" # interpol a tiny bit worse than with flow25e-2, flow25e-2 is preferable
    # model_name = "flownet_l1_flow3e-1_v2_128_rect_big_hftextv3_allBlocks_500ep.pkl"
    # model_name = "flownet_l1_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_gamma.pkl" # flow was reverted, coeff to high
    model_name = "flownet_l1_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_gamma.pkl" # coeff too high
    model_name = "flownet_l1_flow2e-1_v2_128_rect_big_hftextv3_allBlocks_gamma.pkl" # not better than without gamma
    model_name = "flownet_l1_flow2e-1_v2_128_rect_big_hftextv3_allBlocks_gamma_RIFE.pkl" # is it same?? yes.
    model_name = "flownet_l1_flow2e-1_v2_128_rect_big_hftextv3_allBlocks_gamma_RIFE_forw.pkl" # only one direction: a bit worse than before
    model_name = "flownet_l1_flow2e-1_v1_128_rect_big_hftextv3_allBlocks_gamma_RIFE.pkl" 

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks.pkl" # interpol considerably better, flow also good; this was only 1 dir RIFE loss
    # # model_name = "flownet_l1_flow5e-1_v2_128_rect_big_fbb_allBlocks.pkl" # interpol not accurate
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_2.pkl" # two dir flow loss: interpol worse ?
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE.pkl" # one dir flow: + best! both flow & interpol good +
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_eqRIFE.pkl" # one dir flow: flow worse than with gamma

    # model_name = "flownet_l1_flow_v1_128_rect_big_fbb_allBlocks_gamma_RIFE.pkl" # too high flow loss: interpol worsened, flow no better

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_magn.pkl" # test using flow magn loss: No!

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_300ep.pkl" # one dir flow: only 300ep: flow not accurate at edges
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_500ep.pkl" # one dir flow: only 500ep: boundaries still not sharp

    # model_name = "flownet_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_OnlyFlow.pkl" # no better -

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_3K.pkl" # nothing new, still same mistakes in flow

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_MagnDiff.pkl" 
    # model_name = "flownet_l1_flow5e-4_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_MagnDiff2.pkl" # flow no better, intrpol bad
    # model_name = "flownet_l1_flow5e-4_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_MagnDiffLast.pkl" # flow worse, bad interpol

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma05_RIFE.pkl" # 
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_nohf_allBlocks_gamma_RIFE.pkl" # 

    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE.pkl" # one dir flow: + best! both flow & interpol good +

    model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_eqRes.pkl" # not bad! is it better??
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_2_eqRes.pkl" # two dir flow loss: flow a bit worse, interpol bad
    model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_allBlocks_RIFE_eqRes.pkl" # not bad! very similar to the one with gamma
    model_name = "flownet_l1_flow5e-1_v1_128_rect_big_hftextv3_allBlocks_RIFE_eqRes.pkl" # interpol is not very accurate, flow is good but magn errors sometimes
    model_name = "flownet_l1_flow5e-1_v1_128_rect_big_hftextv3_allBlocks_gamma_RIFE_eqRes.pkl" # very similar to the prev., without gamma a tiny bit better

    # TODO: check distill with latest models
    model_name = "flownet_l1_dist2e-2_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_eqRes.pkl" # best? yes.
    # TODO: check stability
    # 2 things helped: instead of pyramid using same scales for all IFBlocks
    # adding distill loss with small coeff
    # model_name = "flownet_l1_dist1e-1_flow5e-1_v1_128_rect_big_fbb_allBlocks_gamma_RIFE_eqRes.pkl" # distill jumps: worse interpol and flow -
    # model_name = "flownet_l1_dist2e-2_flow5e-1_v1_128_rect_big_fbb_allBlocks_RIFE_eqRes.pkl" # no jumps, very similar to the best
    # model_name = "flownet_l1_dist1e-1_flow5e-1_v1_128_rect_big_fbb_allBlocks_RIFE_eqRes.pkl" # jumps in distill
    # model_name = "flownet_l1_dist2e-2_flow5e-1_v1_256_rect_big_fbb_allBlocks_gamma_RIFE_eqRes.pkl" #
    # model_name = "flownet_l1_dist2e-2_flow5e-1_v1_256_rect_big_fbb_allBlocks_gamma_RIFE_eqRes_500ep.pkl" #
    # model_name = "flownet_l1_dist2e-2_flow5e-1_v1_256_rect_big_fbb_allBlocks_gamma_RIFE_eqRes_deconv_500ep.pkl" # ? not better than before
 
    # model_name = "flownet_lapl_dist2e-2_reg1e-7_photo1e-5_v2_128_rect_big_hftext.pkl" # - best interpol unsupervised
    # model_name = "flownet_lapl_flow5e-1_v2_128_rect_big_hftextv3_allBlocks_normv3.pkl" # - best supervised flow
    # L1 is better than Laplacian for rectangle2d

    """ lbs2d """
    # model_name = "flownet_flowonly_v2_128_lbs.pkl" # 665 ep: perf interpol, good flow but it was too easy
    # model_name = "flownet_lapl_dist_photo1e-5_v2_128_lbs.pkl" # 200 ep: very good interpol, bad flow
    # model_name = "flownet_lapl_dist_photo1e-5_v2_128_lbs_skip.pkl" # 250 ep: perf interpol, good flow but it was too easy
    # model_name = "flownet_flowonly_v2_128_lbs_skip.pkl" # 175 ep: 
    # TODO: analyze dataset and ? create ensembles
    # model_name = "flownet_lapl_flow_v2_128_lbs_skip.pkl" # TODO: normalize +
    # model_name = "flownet_test_v2_128_lbs_skip.pkl"
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_NormAllScaleDens.pkl" # flow not accurate
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_NormAllScaleDensv2.pkl" # val and test subsets from middle: flow a bit better ? but not accurate
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_NormDensScaleDens.pkl" # flow flow3e-1; 700 ep: loss descreased, but:
    # model_name = "flownet_l1_flow1e-2_v2_128_lbs_skip_NormAllScaleDens.pkl" # 500 ep: loss descreased;
    # model_name = "flownet_l1_flow5e-2_v2_128_lbs_skip_NormAllScaleDens.pkl" # 500 ep: bad: loss increased -
    # model_name = "flownet_l1_flow1e-1_v2_128_lbs_skip_NormAllScaleAll.pkl" # 500 ep; flow scaled to [-1..1]: loss descreased;
    # model_name = "flownet_l1_flow1e-2_v2_128_lbs_skip_NormAllScaleAll.pkl" # 500 ep; flow scaled to [-1..1]: loss descreased;
    # model_name = "flownet_l1_flow1e-1_v2_128_lbs_skip_ScaleAllNormAll.pkl" # 500 ep; flow1e-1; scale then norm: bad flow
    # model_name = "flownet_l1_flow1e-1_v2_128_lbs_skip_ScaleAllNormVel.pkl" # 500 ep; loss increased -
    # model_name = "flownet_l1_flow1e-2_v2_128_lbs_skip_ScaleAllNormVel.pkl" # 500 ep; flow not accurate
    # model_name = "flownet_l1_flow1e-2_v2_128_lbs_skip_ScaleAllScaleVel10NormVel.pkl" # vel scaled *10;
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_ScaleAllScaleVel10NormVel.pkl" # vel scaled *10; no
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_ScaleVel10.pkl" # vel scaled *10; flow increased after 250ep; flow pos better but direction wrong
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_ScaleVel10v2.pkl" # vel scaled *10;
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_ScaleVel10v3.pkl" # vel scaled *10; flow x and y sep in loss
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_ScaleVel10_rev.pkl"
    # model_name = "flownet_l1_flow5e-1_v2_128_lbs_skip_ScaleVel10_Magn.pkl" # use flow magnitute instead of density: very bad flow?
    # model_name = "flownet_l1_flow2e-1_v2_128_lbs_skip_ScaleVel10_gamma_RIFE.pkl" # loss increased after 200 ep
    # model_name = "flownet_l1_flow2e-1_v2_128_lbs_skip_ScaleVel10_gamma_RIFE_200.pkl"

    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip_ScaleVel10_gamma_RIFE.pkl"
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip_ScaleVel100_gamma_RIFE.pkl"
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleVel10_gamma_RIFE.pkl" # loss stabilized but not accurate
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleVel100_gamma_RIFE.pkl" loss increased -
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE.pkl" # 
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAll_gamma_RIFE.pkl" # loss stab very quickly, flow not accurate.
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleVel10_gamma_RIFE_eqRes_BN.pkl" # flow not accurate
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleVel10_gamma_RIFE_eqRes_BN_lr.pkl" # flow not accurate
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_VelLogNorm_gamma_RIFE_eqRes_BN_lr.pkl" # mo
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_VelLogNorm1_gamma_RIFE_eqRes_BN_lr.pkl" # no
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_VelLogNorm2_gamma_RIFE_eqRes_BN_lr.pkl" # BN worsens interpoltation a bit 
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel300_gamma_RIFE_eqRes_lr.pkl" # no
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_sum_eqRes_lr.pkl" # no
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr3e-3.pkl" # diverge
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr6e-4.pkl" # diverge slowly after 100ep
    # # perhaps start with 3e-4 and change lr after 100 ep? remove loss decay?
    # """
    model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr4e-4.pkl" # start with lr 4e-4, increase to 6e-6 after 100ep: 
    # loss increased a bit in the end, but result best overall +
    # # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr3e-4manual.pkl" # no
    # # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr3e-4mul.pkl" # 3e-4 100ep: decrese, then stop, then increase a bit; 
    # # # then 3e-5 100ep: loss increased a bit again then stoped; 3e-5 100ep: nothing new; 3e-3 100ep: decrese but then increase again...
    # # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr5e-4_v1.pkl" 
    # # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr4e-4_v2.pkl" # flow not bad but can be better; loss inc a bit after 100ep
    model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr4e-4_change.pkl" # start with lr 4e-4, 
    # increase to 6e-4 after 100ep: loss decreasing very slightly, but didn't increase!; 8e-4: fluctuate, then same; 4e-4: val loss not improving
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr2e-7_const.pkl" # no
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr2e-6_const.pkl" # flow loss didn't diverge but flos bad -
    # # increase model capacity: 256 144 96
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr4e-4.pkl" # bad
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleDens_gamma_RIFE_eqRes_lr4e-4.pkl" # 256 192 128;
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleDens_gamma_RIFE_eqRes_lr4e-4_6e-4.pkl"
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr6e-4.pkl" # good loss plot +
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr1e-4_1e-5.pkl" # lr was actually 2e-5: +
    # loss and result good when lr is low in the beginning, then cosine annealing form larger value
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr1e-3_1e-4.pkl" # lr too high, exploded, very bad results -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr1e-4_1e-5_v1.pkl" # no
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr1e-4_1e-5_v2.pkl" # loss started higher, stuck at higher than 0.1
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr1e-4_1e-5_v3.pkl" # same as before
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_lr6e-4_v1.pkl" # 6e-4...6e-6: not bad but before was better
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_lr6e-4.pkl" # 6e-4...6e-6: not bad, is it best? good loss +
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_deconv_lr6e-4.pkl" # good, can be better? +
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_deconv_lr1e-4.pkl" # 1e-4...1e-6: bad loss, bad flow -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_deconv_lr1e-3.pkl" # no, exploded -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_deconv_lr8e-4.pkl" # no, exploded -
    # TODO: try k_s 5x5
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_ks5_lr6e-4.pkl" # no, exploded -
    # reduce lr?+ add grad clipping?
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel10_gamma_RIFE_eqRes_5Blocks_ks5_lr1e-4.pkl" # # good, but not better than before 
    # increase amount of data
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v1.pkl"
    # is the simulation flow values becoming smaller towards simulation?
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v2.pkl" # works, but...
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr3e-4.pkl" # same as lr6e-4
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr1e-3.pkl" # no, exploded -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr1e-4.pkl" #
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_11K_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr7e-4.pkl" # right side is bad
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleDens_gamma_RIFE_eqRes_5Blocks_deconv_lr6e-4.pkl" # 3 blocks: right part is less accurate
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleAllVel100_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # loss didn't decrease properly -
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # accuracy can be better
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr3e-4.pkl" # almost same
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # good?
    ###
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_500ep.pkl" # good?
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ensemble21_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # loss exploded -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ensemble21_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-5.pkl" # loss increasing -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ensemble21_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-6.pkl" # very slowly inc -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ensemble21_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr1e-6.pkl" # not decreasing -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ensemble21_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr3e-6.pkl" # no -
    # # thi is not working, what is the problem ???
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_AMP_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # 16 amp is the problem!
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_AMP32_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # exploded at 25ep -
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_noAMP_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # SAME - autocast was still in IFNet
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_noAMP32_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # +
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ensemble16_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_500ep_ensemble16.pkl" # no better ?
    # model_name = "flownet_lapl_v1_240_lbs_skip3_combi_ScaleAllVel10_lr3e-4_RIFE.pkl" # test RIFE: good interpol, very bad flow; EPE: 0.06
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens7_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_500ep.pkl" # flow worse than best, separate member?
    # test set only parts of members?
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens5_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # not yet close to best...
    # TODO: unsupervised for sim
    # model_name = "flownet_l1_dist_reg1e-6_photo1e-5_v1_256_lbs_skip3_ens5_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # reg too high
    # model_name = "flownet_l1_dist_photo1e-5_v1_256_lbs_skip3_ens5_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # no, bad flow and interpol
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens5_ScaleAllVel10_gamma_RIFE_lr6e-4.pkl" # no flow, distill and photo
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens5_ScaleAllVel10_gamma_RIFE_lr6e-4_v3.pkl" # not bad, but worse?
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens7_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v2.pkl" # not so good ?
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens5_cyl_tau_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # still not as best
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_ScaleAllVel10_gamma_RIFE_lr6e-4.pkl" # # still not as best
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v2.pkl"
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v3.pkl" # train on all data:
    # ###
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # yes! best.
    # 3rdShift - that is important! DensNorm - this too!
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens7_3rdShift_ScaleAllVel10_gamma_RIFE_lr6e-4.pkl" # flow error much higher
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_3rdShift_ScaleAllVel10_gamma_RIFE_lr6e-4.pkl" # flow error higher than best
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_3rdShift_ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # flow error higher than best
    model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_combi_cyl_tau_3rd5thShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # flow error higher than best
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens7_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # all time best?
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # all time best; epe: 0.017
    # model_name = "flownet_lapl_v1_240_lbs_skip3_combi_ScaleAllVel10_lr3e-4_RIFE.pkl" # test RIFE: good interpol, very bad flow; EPE: 0.06
    model_name = "flownet_lapl_v1_240_lbs_skip3_ens10_ScaleAllVel10_lr3e-4_RIFE.pkl" # test RIFE: good interpol, very bad flow; EPE: 0.12
    # PSNR: 48.52 SSIM: 0.999 LPIPS: 0.001 EPE: 0.1800
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_100ep.pkl" # not bad, but 300ep better
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens16_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # not nenough memory
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v2.pkl" # best + epe: 0.017
    # model_name = "flownet_l1_dist_photo1e-5_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_RIFE_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_l1_dist_photo1e-5_reg1e-7_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_RIFE_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_l1_dist_photo1e-5_reg1e-7_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_RIFE_eqRes_deconv_lr6e-4_v2.pkl" # bad flow
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNorm01ScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # good but a bit worse
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # similarly good as best +
    # model_name = "flownet_lapl_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # epe: 0.0171
    # model_name = "flownet_l1_dist_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # epe: 0.0171
    # flow improves with larger ensemble; TODO: run experiment with 30 members on Habrok 21m+
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens10_3rdShift_DensNormScaleAllVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v3.pkl" # works
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # all time best + epe: 0.0136
    # 3Blocks: PSNR: 53.083 EPE: 0.0136
    # untrained: PSNR: 21.8099 SSIM: 0.9965 LPIPS: 0.0074
    # Baseline: avg PSNR: 48.5198 avg EPE: 0.1800
    # TODO: ablation: 1. without flow loss 2. 4,2,1 and lapl 3. 128 instead of 256
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_RIFE_eqRes_deconv_lr6e-4.pkl" # epe: 0.014 
    # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_lr6e-4.pkl" # param s. epe: 0.0228
    # model_name = "flownet_l1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_RIFE_eqRes_deconv_lr6e-4.pkl" # ablation PSNR: 50.38 EPE: 0.01469
    # model_name = "flownet_l1_flow2e-1_v1_128_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # param s.
    # model_name = "flownet_lapl_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # param s. PSNR: 49.668 EPE: 0.0139
    # model_name = "flownet_l1_flow3e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # param s. PSNR: 47.87 EPE: 0.0138
    # model_name = "flownet_l1_flow1e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # param s.
    # model_name = "flownet_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # param s.
    # model_name = "flownet_4Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # ++ 49.42 0.0130 52.84 0.0129 53.24 0.0128
    # 4Blocks: 53.24 0.0128
    model_name = "flownet_4Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v2.pkl" # ++ 50.32 0.0129
    model_name = "flownet_4Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_RIFE_eqRes_deconv_lr6e-4.pkl" # param s. 50ep: 45.99 0.0132
    model_name = "flownet_4Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma7e-1_RIFE_eqRes_deconv_lr6e-4.pkl" # param s.
    # model_name = "flownet_5Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" # ++ 120ep: 57.52 EPE: 0.0130
    # 5Blocks: full test: 48.43 0.0122
    # model_name = "flownet_5Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4_v2.pkl" # PSNR: 46.33 EPE: 0.0121
    # model_name = "flownet_6Blocks_l1_flow2e-1_v1_256_lbs_skip3_ens21_3rdShift_DensNormScaleVel10_gamma_RIFE_eqRes_deconv_lr6e-4.pkl" 
    # 6Blocks: second half of test set: 61 0.014; full test: PSNR: 46.021 EPE: 0.0124
    # TODO: param s.: flow3e-1 + flow1e-1 + no L1 +
    # TODO: num_blocks +- gamma1 - gamma 0.6 - refine -

    """ vimeo2d """
    # model_name = "flownet_lapl_dist_v2_128_vimeo.pkl" # very good interpol, ? good flow
    # model_name = "flownet_lapl_dist_reverse_v2_128_vimeo.pkl" # same ???
    # model_name = "flownet_lapl_dist_noreverse_v2_128_vimeo.pkl" # same ???
    # model_name = "flownet_lapl_dist_zero_v2_128_vimeo.pkl" # empty
    # model_name = "flownet_lapl_dist_reversemask_v2_128_vimeo.pkl" # same ???
    # model_name = "flownet_lapl_dist_reverseflowmask_v2_128_vimeo.pkl" # same ???
    # model_name = "flownet_lapl_dist_photo1e-4_v2_128_vimeo.pkl" # very good interpol, very bad flow
    # model_name = "flownet_lapl_v2_128_vimeo.pkl" # ?
    # model_name = "flownet_lapl_dist_reg_photo1e-5_v2_128_vimeo.pkl" # nan loss
    # model_name = "flownet_lapl_dist01_v2_128_vimeo.pkl"
    # model_name = "flownet_lapl_dist_v2_240_vimeo.pkl" # very good interpol, ? good flow
    # model_name = "flownet_lapl_dist_photo1e-5_v2_128_vimeo.pkl" # very good interpol, ? good flow
    # I can reproduce results, there no additional tricks

    """ droplet2d """
    # model_name = "flownet_lapl_reg_nosmooth_3rd_drop50K.pkl"
    # model_name = "flownet_lapl_3rd_drop50K.pkl"
    # model_name = "flownet_lapl_3rd_c_drop50K.pkl"
    # model_name = "flownet_lapl_3rd_c_drop50K_finetune.pkl"
    # model_name = "flownet_lapl_reg_3rd_c_drop50K.pkl"
    # model_name = "flownet_lapl_smooth_3rd_c_drop50K.pkl"
    # model_name = "flownet_lapl_reg_smooth_3rd_c_drop50K.pkl"
    # model_name = "flownet_lapl_3rd_c_drop50K.pkl" # not so bad before; after shift: similar
    # model_name = "flownet_lapl_reg_smooth_3rd_aug_c_drop50K.pkl" # 300 ep: bad flow
    # model_name = "flownet_lapl_photo_3rd_drop50K.pkl"
    # model_name = "flownet_lapl_all_64_3rd_drop50K.pkl" # small flow vector equally distib, effect of smooth? v2
    # model_name = "flownet_lapl_all_nosmooth_noreg_v2_64_3rd_drop50K.pkl" # bad flow
    # model_name = "flownet_lapl_l1_distill_3rd_drop50K.pkl"
    # model_name = "flownet_lapl_3rd_aug_modif_90_drop50K.pkl" # not so good
    # model_name = "flownet_lapl_all_v2_64_3rd_drop50K.pkl" # coeffs: 
    # model_name = "flownet_lapl_all_10e5_v2_64_3rd_drop50K.pkl" # coeffs: 
    # model_name = "flownet_flowonly_v2_64_tl_infer_drop50K.pkl" # tl flrom pipedc: 
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_v2_128_drop50K.pkl" # reg was too much
    # model_name = "flownet_lapl_dist_photo1e-5_v2_128_drop50K.pkl" # best so far: interpol and flow, how improve?
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-5_v2_128_drop50K.pkl" # not so bad, reg might be higher
    # model_name = "flownet_lapl_v1_240_lbs_drop50K_float_lr3e-4_RIFE.pkl" # ???
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_v2_128_drop50K_v2.pkl" # not so bad
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_v1_256_drop50K.pkl" # reg too high?
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-5_v1_256_drop50K.pkl" # beginning was good but then loss jumped - lr too high? lr6e-4
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-5_v1_256_drop50K_lr3e-4.pkl" # l1: good interpol, some flow, loss unstable +
    ##############
    # model_name = "flownet_lapl_dist_v1_256_drop50K_lr3e-4_RIFEbaseline.pkl" # RIFE Baseline: good interpol and good flow + PSNR: 35.35 LPIPS: 0.026
    # Baseline: PSNR: 34.5642 LPIPS: 0.0299
    # model_name = "flownet_l1_dist_reg1e-8_photo1e-5_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" worse than baeline: flow worse -
    # model_name = "flownet_l1_dist_photo1e-5_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # worse than baeline: distill exploded -
    # model_name = "flownet_l1_photo1e-5_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # similar to baseline: distill is not important
    # model_name = "flownet_l1_photo1e-5AllBlocksGamma_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # now photo to all blocks -
    # model_name = "flownet_l1_photo2e-6AllBlocksGamma_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # worse than B. -
    # model_name = "flownet_l1_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # PSNR: 33.59, LPIPS: 0.035
    # model_name = "flownet_l1_v1_256_drop50K_lr1e-4.pkl" # 2x faster; eqRes and distill is heavy; worse than B&B -
    # model_name = "flownet_l1_dist5e-3_photo1e-6_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # worse, distill jumps -
    # model_name = "flownet_l1_photo1e-6_v1_256_drop50K_eqRes_deconv_lr6e-5.pkl" # worse than before -
    # model_name = "flownet_l1_photo1e-7_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # good but worse than best
    # # model_name = "flownet_l1_v1_256_drop50K_eqRes_deconv_test.pkl" # untrained: PSNR: 33.59, LPIPS: 0.040
    # model_name = "flownet_l1AllBlocksGamma_photo1e-7_v1_256_drop50K_eqRes_deconv_lr1e-4.pkl" # PSNR: 30.19 LPIPS: 0.053
    # model_name = "flownet_l1_dist_v1_256_drop50K_eqRes_deconv_lr3e-4.pkl" # PSNR: 31.19 LPIPS: 0.046 -
    # model_name = "flownet_l1_v1_256_drop50K_eqRes_deconv_lr3e-4.pkl" # worse than before -
    # model_name = "flownet_l1_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr3e-4.pkl" # best + PSNR: 35.82 LPIPS: 0.022
    # model_name = "flownet_l1_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr1e-4.pkl" # worse than before -
    # model_name = "flownet_l1_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # best + PSNR: 36.13 LPIPS: 0.022
    # model_name = "flownet_l1_dist_reg1e-9_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # loss increased -
    # model_name = "flownet_l1_dist_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # ablation
    # model_name = "flownet_l1_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # ablation
    # model_name = "flownet_l1_dist_reg1e-8_v1_256_drop50K_lr6e-4.pkl" # ablation
    # model_name = "flownet_l1_dist_reg1e-8_photo1e-6_v1_128_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # all time best ++ PSNR: 37.381 LPIPS: 0.0144
    # got worse after training more... 36.65 0.0170 
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-6_smooth1e-8_v1_256_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_100ep.pkl" # PSNR: 32.13 LPIPS: 0.199
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_v2.pkl" # reproduce: PSNR: 37.381 LPIPS: 0.0144
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_6e-5.pkl" # reproduce: PSNR: 33 LPIPS: 0.0143
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_6e-6.pkl" # reproduce: all time best ++ PSNR: 35.818 LPIPS: 0.0086
    # 35.34 0.0157 ??? 36.93 0.0188 full test set
    # model_name = "flownet_lapl_dist_reg1e-8_photo1e-5_v1_256_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_lapl_dist2e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_lapl_dist5e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s.
    # model_name = "flownet_5Blocks_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s.: loss jumped -
    # model_name = "flownet_5Blocks_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr3e-4_3e-6.pkl" # distill loss jumped -
    # model_name = "flownet_5Blocks_lapl_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # worse than before -
    # model_name = "flownet_5Blocks_lapl_dist1e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # 33.34 0.032 - scale = [1, 1, 1, 1, 1], other bugs
    # model_name = "flownet_5Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_scale.pkl" # scale = [4, 2, 2, 1, 1] 36.37 0.0164
    # model_name = "flownet_2Blocks_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # scale = [2, 1] 34.78 0.0291
    # model_name = "flownet_4Blocks_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # [8, 4, 2, 1] 34.42 LPIPS: 0.0153
    # model_name = "flownet_4Blocks_4221_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # # distill loss jumped -
    # model_name = "flownet_4Blocks_4221_lapl_dist1e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # 33.97 0.0175 35.048 0.0164 35.41 0.0190
    # model_name = "flownet_4Blocks_4221_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # 35.50 0.0234
    # model_name = "flownet_6Blocks_lapl_dist_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # scale = [4, 4, 2, 2, 1, 1] 35.80 0.0202
    # model_name = "flownet_refine_5Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # scale = [4, 2, 2, 1, 1] very bad -
    # model_name = "flownet_refine_lapl_dist_v1_256_drop50K_lr3e-4_RIFEbaseline.pkl" # 20.60 0.0483 baseline
    # model_name = "flownet_5Blocks_lapl_dist5e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # scale = [4, 2, 2, 1, 1] # 31.53 0.0157
    # model_name = "flownet_5Blocks_lapl_dist1e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # scale = [4, 2, 2, 1, 1] # 30.51 0.0186
    # model_name = "flownet_5Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # TODO
    # model_name = "flownet_5Blocks_l1_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_eqRes_deconv_lr6e-4.pkl" # TODO
    # model_name = "flownet_4Blocks_l1_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_eqRes_deconv_lr6e-4.pkl" # distill jump -
    # model_name = "flownet_4Blocks_l1_dist1e-3_reg1e-8_photo1e-6_v1_256_drop50K_eqRes_deconv_lr6e-4.pkl" # distill jump -
    # model_name = "flownet_4Blocks_l1_dist1e-4_reg1e-8_photo1e-6_v1_256_drop50K_eqRes_deconv_lr6e-4.pkl" # distill jump -
    # model_name = "flownet_4Blocks_l1_reg1e-8_photo1e-6_v1_256_drop50K_eqRes_deconv_lr6e-4.pkl" # PSNR: 31.003 0.0469 
    model_name = "flownet_4Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # ++ 36.814 0.0169
    # model_name = "flownet_4Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_flowscale.pkl" # * scale * 2 distill jump -
    # model_name = "flownet_4Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_interpol_t.pkl" 
    # model_name = "flownet_4Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_no_interpol_t.pkl"
    # model_name = "flownet_4Blocks_lapl_dist1e-2_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4_no_interpol_t_v2.pkl"
    # model_name = "flownet_4Blocks_lapl_dist5e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s. 35.49 0.016
    # model_name = "flownet_4Blocks_lapl_dist5e-3_reg1e-8_photo1e-5_v1_256_drop50K_lr6e-4.pkl" # param s. 34.35 0.027
    # model_name = "flownet_4Blocks_lapl_dist1e-2_reg1e-7_photo1e-6_v1_256_drop50K_lr6e-4.pkl" # param s. 35.25 0.0243
    model_name = "flownet_4Blocks_lapl_dist1e-3_reg1e-8_photo1e-6_v1_256_drop50K_lr6e-4.pkl"
    # TODO: param s.: dist5e-3 + dist2e-2 + no Lapl + reg1e-7 + reg1e-9 + photo1e-5 + photo1e-7 +
    # TODO: num_blocks +- eqRes +- refine +

    """ pipedcylinder2d """
    # model_name = "flownet_lapl_3rd_c_piped.pkl"
    # model_name = "flownet_lapl_3rd_aug_c_piped.pkl" # 300 ep underfit; 600 ep better than before;
    # model_name = "flownet_lapl_photo_3rd_aug_c_piped.pkl" # 100 ep underfit;
    # model_name = "flownet_lapl_reg_photo_3rd_aug_c_piped.pkl" # 
    # model_name = "flownet_lapl_reg_photo_smooth_3rd_aug_piped.pkl" # 
    # model_name = "flownet_lapl_reg_photo_smooth_3rd_aug_coeffs_piped.pkl" # 500 ep, nothing
    # model_name = "flownet_lapl_photo_3rd_aug_coeffs_piped.pkl" # 400 ep, not good flow
    # model_name = "flownet_lapl_all_v2_64_3rd_aug_piped.pkl" # good interpol, no flow
    # model_name = "flownet_lapl_all_nosmooth_v2_64_3rd_aug_piped.pkl" # good interpol, flow correct dir, not accurate
    # model_name = "flownet_lapl_all_nosmooth_noreg_v2_64_3rd_aug_piped.pkl" # even 1K ep not very accurate flow
    # model_name = "flownet_lapl_smooth_3rd_aug_c_piped.pkl" # 300 ep bad - chaotic flow where it should be
    # model_name = "flownet_lapl1_noreg_modif90_piped.pkl" # no
    # model_name = "flownet_lapl1_only_modif90_piped.pkl" # similar to before
    # model_name = "flownet_madif90_piped.pkl" # bad
    # model_name = "flownet_lapl_3rd_aug_90_piped.pkl"
    # model_name = "flownet_lapl_3rd_aug_inv_piped.pkl"
    # model_name = "flownet_lapl_3rd_aug_modif_90_piped.pkl" # very bad, flow at empty
    # model_name = "flownet_lapl_all_1e6_v2_64_3rd_aug_piped.pkl" # flow opposite direction
    # model_name = "flownet_lapl_all_coeffs1_v2_64_3rd_aug_piped.pkl"
    # model_name = "flownet_lapl_all_coeffs2_v2_64_3rd_aug_piped.pkl"
    # model_name = "flownet_lapl_all_nosmooth2_v2_64_3rd_aug_piped.pkl" # very bad
    # model_name = "flownet_lapl_l1_distill_v2_64_3rd_aug_piped.pkl"
    # model_name = "flownet_lapl_flow_v2_64_piped.pkl"
    # model_name = "flownet_lapl_flow2_v2_64_piped.pkl"
    # model_name = "flownet_lapl_flow3_v2_64_piped.pkl" # good
    # model_name = "flownet_flowonly_v2_64_piped.pkl" # good
    # model_name = "flownet_flowonly1_v2_64_piped.pkl" # very good +
    # model_name = "flownet_flowonly1_v2_128_piped.pkl" # 128 96 extra good
    # model_name = "flownet_lapl_distill_v2_128_tl_piped.pkl" # tl on same data: perf interpol, bad flow
    # model_name = "flownet_lapl_dist_refine_v2_128_piped.pkl" # 128 96 64 450ep very good interpol, inaccurate flow
    # model_name = "flownet_lapl_dist_photo_refine_v2_128_piped.pkl" # 128 96 64 no refine, flow not accurate 
    # model_name = "flownet_lapl_dist_reg_photo_v2_128_piped.pkl"
    # model_name = "flownet_lapl_dist_photo1e-6_v2_128_piped.pkl"

    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_v2_128_piped.pkl" # 600 ep: very good interpol, not accurate flow
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_v2_128_piped.pkl" # this was not so bad, improve???
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-5_v2_128_1K_piped.pkl" # best result for unsupervised!
    # model_name = "flownet_lapl_dist_reg1e-5_photo1e-5_v2_128_1K_piped.pkl" # same, reg1e-6 is netter for loss values
    # model_name = "flownet_lapl_dist_v2_128_1K_piped.pkl" # very good interpol, not accurate flow (wrong direction)
    # model_name = "flownet_lapl_dist_reg1e-6_photo1e-4_v2_128_1K_piped.pkl" # photo1e-5 was better
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-5_v2_128_1K_piped.pkl" # best result for unsupervised!
    # supervised: very good interpol and flow
    # unsupervised: very good interpol, some flow
    # model_name = "flownet_flowonly_v2_128_piped_allBlocks.pkl" # very good flow and interpol
    # model_name = "flownet_flowonly_v2_128_piped_allBlocks_inv.pkl" # test, inverse
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-5_v2_128_1K_piped_photofix.pkl" # best result for unsupervised!

    # model_name = "flownet_lapl_flow5e-1_v2_128_piped_allBlocks.pkl" # 1 1 1 is worse

    """ FluidSimML2d """
    # model_name = "flownet_lapl_3rd_c_Fluid.pkl" # not good
    # model_name = "flownet_lapl_3rd_cd_Fluid.pkl" # good 300 ep +
    # model_name = "flownet_lapl_l1_distill_v2_64_3rd_Fluid.pkl" # good ~300 ep 
    # model_name = "flownet_lapl_all_v2_64_3rd_Fluid.pkl" # 150 ep, small flow eq, effect of smooth
    # model_name = "flownet_lapl_nosmooth_noreg_v2_64_3rd_Fluid.pkl"
    # model_name = "flownet_flowonly_v2_64_fluid.pkl" # ?
    # model_name = "flownet_flowonly1_v2_128_tl_piped.pkl"
    # model_name = "flownet_flowonly_v2_96_fluid.pkl"
    # model_name = "flownet16_flowonly_v2_96_fluid.pkl" # error
    # model_name = "flownet_flowonly_v2_64_tl_infer_fluid.pkl" # tl flrom pipedc: good interpol and ? good flow
    # model_name = "flownet_lapl_dist_photo_v2_256_fluid.pkl" # 256 128 64 good interpol, bad flow (can be unstable, nan loss)
    # model_name = "flownet_lapl_dist1_v2_256_fluid.pkl"
    # model_name = "flownet_flowonly_v2_256_fluid.pkl"
    
    """ cylinder2d """
    # model_name = "flownet_lapl_3rd_c_cylinder.pkl" # bad, 600 ep, small loss
    # model_name = "flownet_lapl_all_v2_64_3rd_cylinder.pkl"
    # model_name = "flownet_lapl_l1_distill_v2_64_3rd_cylinder.pkl" # very bad flow, small loss
    # model_name = "flownet_flowonly_v2_64_cylinder.pkl" # good flow
    # model_name = "flownet_flowonly_v2_128_cylinder.pkl" # good flow
    # model_name = "flownet_flowonly_v2_128_tl_cylinder.pkl" # good, nothing surprising
    # model_name = "flownet_lapl_distill_v2_128_tl_cylinder.pkl" # from piped: no
    # model_name = "flownet_lapl_distill_v2_128_tl1_cylinder.pkl" # from cylinder: no, same
    # model_name = "flownet_flowonly_v2_64_tl_infer_cylinder.pkl" # tl from pipedc: good interpol, not good flow
    # model_name = "flownet_lapl_dist_reg1e-7_photo1e-5_v2_128_1K_cylinder.pkl" # very good interpol, but not flow
    # supervised: very good interpol and flow
    # unsupervised: very good interpol, not good flow

    # It also sounds quite promising from your description what you got in terms of results. 
    # In general, I would recommend to concentrate on getting good supervised flow first across datasets before going back to unsupervised 
    # (I think it probably makes sense not to have both unsupervised and supervised in one paper, 
    # let's concentrate on one solution first, unsupervised could maybe be done in a follow-up paper). 
    # Once we get good results with supervised, let's do a closer analysis of its accuracy, and then I think we should 
    # concentrate on demonstrating the utility of supervised for visualization purposes.

    # print(model_name)
    # input("x")

    if args.mode == "train":
        train(model, args.dataset, args.factor, model_name, args.mode, args.local_rank)
    else:
        dataset = args.dataset
        print("inference...")
        # factor = 2
        # factor = 2 ** args.exp
        factor = args.factor
        dir_res = "Results"
        dir_res = os.path.join(dir_res, dataset)
        dir_res = os.path.join(dir_res, str(factor) + "x")
        dir_model = os.path.join(dir_res, model_name[:-4])
        # print(dir_model)f
        # input("x")
        print("Saving at:", dir_model)

        data_test, interpol_data, flow, flow_test, mask, statistics = train(model, args.dataset, args.factor, model_name, args.mode, args.local_rank)
        # data_test is data_test_non_overlap
        print("data_test:", data_test.shape)
        print("interpol_data:", interpol_data.shape)
        print("flow:", flow.shape)
        print("mask:", mask.shape)
        # input("inf")
        # original_data = data_test[:, 2, :interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
        data_test = data_test[:, :interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
        print("data_test:", data_test.shape)
        print("flow_test:", flow_test.shape)
        # print(type(data_test))
        # print(type(flow_test))
        # input("x")

        data_original = [] # I0, I1 for interpol
        data_original_interpol = [] # gt for interpol comparison
        n = args.factor # arbitrary interpol factor

        data_original = [d for i, d in enumerate(data_test) if i % n == 0]
        data_original = np.array(data_original)
        print("data_original:", data_original.shape)
        print(type(data_original))
        flow_original = [f for i, f in enumerate(flow_test) if i % n == 0]
        flow_original = np.array(flow_original)
        print("flow_original:", flow_original.shape)

        data_original_interpol = [d for i, d in enumerate(data_test) if i % n != 0]
        data_original_interpol = np.array(data_original_interpol)
        print("data_original_interpol:", data_original_interpol.shape)
        print(type(data_original_interpol))
        flow_original_interpol = [f for i, f in enumerate(flow_test) if i % n == 0]
        flow_original_interpol = np.array(flow_original_interpol)
        print("flow_original_interpol:", flow_original_interpol.shape)
        flow_gt = flow_original_interpol
        input("n")
        
        """
        if args.exp == 1:
            # remove overlapping
            # original_data_new = []
            # for i, element in enumerate(original_data):
            #     if i < 2 or (i + 1) % 3 != 0:
            #         original_data_new.append(element)
            # original_data = np.array(original_data_new)
            original_data = [data for i, data in enumerate(original_data) if i % 2 == 0]
            print("original_data:", original_data.shape)
            # original_flow_new = []
            # for i, element in enumerate(flow_original):
            #     if i < 2 or (i + 1) % 3 != 0:
            #         original_flow_new.append(element)
            # flow_original = np.array(original_flow_new)
            flow_original = [f for i, f in enumerate(flow_original) if i % 2 == 0]
            print("flow_original:", flow_original.shape)

            # original_data_gt - only GT densities, original_data - all timesteps from orig. dens.
            original_data_gt = [d for i, d in enumerate(original_data) if i % 2 != 0]
            original_data_gt = np.array(original_data_gt)
            

            print("flow_original:", flow_original.shape)
            flow_gt = [f for i, f in enumerate(flow_original) if i % 2 != 0]
            flow_gt = np.array(flow_gt)
        if args.exp == 2:
            # # remove overlapping
            # original_data_new = []
            # for i, element in enumerate(original_data):
            #     if i < 4 or (i + 1) % 5 != 0:
            #         original_data_new.append(element)
            # original_data = np.array(original_data_new)
            # print("original_data:", original_data.shape)
            # original_flow_new = []
            # for i, element in enumerate(flow_original):
            #     if i < 4 or (i + 1) % 5 != 0:
            #         original_flow_new.append(element)
            # flow_original = np.array(original_flow_new)
            # print("flow_original:", flow_original.shape)
            original_data_gt = [data for i, data in enumerate(original_data) if i % 4 != 0]
            original_data_gt = np.array(original_data_gt)
            original_data = np.array(original_data)

            print("flow_original:", flow_original.shape)
            flow_gt = [f for i, f in enumerate(flow_original) if i % 4 != 0]
            flow_gt = np.array(flow_gt)
        """

        print("flow_pred:", flow.shape)
        # viz Ft->1 (forward), that is flow[2][:, 2:4]
        flow = flow[:, 2:4, ...]

        data_original_interpol = data_original_interpol[:interpol_data.shape[0]]

        print("######")
        print("data_test:", data_test.shape) 
        print("data_original:", data_original.shape) # I0, I1 for interpol
        print("data_original_interpol:", data_original_interpol.shape) # gt for interpol comparison
        print("interpol_data:", interpol_data.shape)
        print("flow_gt:", flow_gt.shape)
        print("flow pred & mask:", flow.shape, mask.shape)
        # input("n")

        """
        ##### Baseline evaluation
        lin_interpol = linear_interpolation(data_original)
        lin_interpol = lin_interpol[:data_original_interpol.shape[0]]
        print(lin_interpol.shape)

        avg_psnr = calculate_avg_psnr(data_original_interpol, lin_interpol)
        avg_lpips = calculate_avg_lpips(data_original_interpol, lin_interpol)
        # Droplets: linear: PSNR: 32.21 LPIPS: 0.0356; nearest: PSNR: 30.76 LPIPS: 0.0373
        # LBS: linear: PSNR: 44.88; nearest: PSNR: 47.99
        input("x")
        #####
        """

        if "lbs2d" in dataset: # denormalize
            # print("flow_gt mean std", np.mean(flow_gt), np.std(flow_gt))
            flow_gt = flow_gt / 10.
            flow = flow / 10.
            print("Vel rescaled to /10")
            # flow_gt = np.exp(flow_gt) - 0.3
            # flow = np.exp(flow) - 0.3
            # print("Vel log denormalized")
            # print("flow_gt mean std", np.mean(flow_gt), np.std(flow_gt))

            print("original data is in range %f to %f" % (np.min(data_original_interpol), np.max(data_original_interpol)))
            print("interpol data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))

            # original_data = cv2.normalize(original_data, original_data, 0., statistics['dens_max'], cv2.NORM_MINMAX)
            # original_data = original_data * statistics['dens_max'] 
            # original_data_gt = original_data_gt * statistics['dens_max'] 
            # interpol_data = interpol_data * statistics['dens_max'] 

            # data[:, 0] = (data[:, 0] - data[:, 0].mean()) / data[:, 0].std() # it is important!
            # original_data = original_data * statistics['dens_std'] + statistics['dens_mean']
            # original_data_gt = original_data_gt * statistics['dens_std'] + statistics['dens_mean']
            interpol_data = interpol_data * statistics['dens_std'] + statistics['dens_mean']

            # print("Density range %f to %f" % (np.min(original_data), np.max(original_data)))
            # input("x")

        #     flow_x_mean = statistics['flow_x_mean']
        #     flow_y_mean = statistics['flow_y_mean']
        #     flow_x_std = statistics['flow_x_std']
        #     flow_y_std = statistics['flow_y_std']
        #     # data_t, flow_x_mean, flow_y_mean, flow_x_std, flow_y_std = load_data(dataset, args.factor, args.mode)
        #     # original_data = original_data * original_data.std() + original_data.mean()
        #     # interpol_data = interpol_data * interpol_data.std() + interpol_data.mean()
        #     flow_original[:, 0] = flow_original[:, 0] * flow_x_std + flow_x_mean
        #     flow_original[:, 1] = flow_original[:, 1] * flow_y_std + flow_y_mean
        #     c[:, 2] = flow[:, 2] * flow_x_std + flow_x_mean
        #     flow[:, 3] = flow[:, 3] * flow_y_std + flow_y_mean
        #     print("Denormalized from mean 0 and std 1!")
        #     print("Vel x mean std", flow_original[:, 1].mean(), flow_original[:, 1].std())

            # print("flow_gt:", flow_gt[10, 0])
            # print("flow:", flow[10, 2:3])
            # flow[0][:, 2:3]), gt_flow[:, 0])
            # # loss_flow = torch.nn.functional.l1_loss(flow_original[10, :], flow[10, :] )
            # l1_loss = abs(flow_original[10, :] - flow[10, 2:4] )
            # mae_cost = l1_loss.mean()
            # print("loss_flow:", mae_cost)
            # input("x")
        
        # interpol_data = interpol_data
        # print(original_data.shape, interpol_data.shape, flow.shape, flow_gt.shape)
        print("original data is in range %f to %f" % (np.min(data_original_interpol), np.max(data_original_interpol)))
        print("interpol data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))
        # input("x")

        # calc diff, metrics
        diffs = calculate_diff(data_original_interpol, interpol_data, dataset, factor, dir_model)
        print("diffs:", diffs.shape)

        # test
        # print(flow.shape)
        # flow = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
        # print(flow_gt.shape)
        # flow_gt = np.zeros((flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2], flow_gt.shape[3]), dtype=np.float32)

        if dataset != "vimeo2d" and dataset != "droplet2d":
            print("GT flow available")
            flow_gt = flow_gt[:, :, :flow.shape[2], :flow.shape[3]]
            u_gt = flow_gt[:, 0, ...]
            v_gt = flow_gt[:, 1, ...]
            norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
            u = flow[:, 0, ...]
            v = flow[:, 1, ...] 

            norm = np.sqrt(u * u + v * v)
            norm_gt = norm_gt[:, :norm.shape[1], :norm.shape[2]]
            # print(norm_gt.shape, norm.shape)
            # check the range
            # u_gt = (u_gt - np.min(u_gt)) / (np.max(u_gt) - np.min(u_gt))
            # u = (u - np.min(u)) / (np.max(u) - np.min(u))
            # v_gt = (v_gt - np.min(v_gt)) / (np.max(v_gt) - np.min(v_gt))
            # v = (v - np.min(v)) / (np.max(v) - np.min(v))
            # norm = (norm - np.min(norm)) / (np.max(norm) - np.min(norm))
            # norm_gt = (norm_gt - np.min(norm_gt)) / (np.max(norm_gt) - np.min(norm_gt))
            print("u_gt is in range %f to %f" % (np.min(u_gt), np.max(u_gt)))
            print("u is in range %f to %f" % (np.min(u), np.max(u)))
            print("norm is in range %f to %f" % (np.min(norm), np.max(norm)))
            print("norm_gt is in range %f to %f" % (np.min(norm_gt), np.max(norm_gt)))

            print("norm:", norm_gt.shape)
            # print(norm_gt[0, 10, 10])
            # print(norm_gt[0, 100, 100])
            # print(norm_gt[100, 10, 10])
            # print(norm_gt[100, 100, 100])
            # print("10", norm_gt[10])
            # print("100", norm_gt[100])
            # print("200", norm_gt[200])
            # norm = max(norm, norm_gt)

            diffs_flow = calculate_diff(norm_gt, norm, dataset, factor, dir_model)

            # u_gt = (u_gt - np.min(u_gt)) / (np.max(u_gt) - np.min(u_gt))
            # v_gt = (v_gt - np.min(v_gt)) / (np.max(v_gt) - np.min(v_gt))
            u_gt = u_gt[:, :u.shape[1], :u.shape[2]]
            v_gt = v_gt[:, :v.shape[1], :v.shape[2]]
            u_diff = calculate_diff(u_gt, u, dataset, factor, dir_model)
            v_diff = calculate_diff(v_gt, v, dataset, factor, dir_model)
            # print(u_gt, u, u_diff)
            # input("x")
            # u_diff = (u_diff - np.min(u_diff)) / (np.max(u_diff) - np.min(u_diff))
            # v_diff = (v_diff - np.min(v_diff)) / (np.max(v_diff) - np.min(v_diff))
        else:
            print("no GT flow available")
            # diffs = np.zeros((original_data.shape[0], original_data.shape[1], original_data.shape[2]), dtype=np.float32)
            diffs_flow = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2]), dtype=np.float32)
            flow_gt = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
            # u_diff = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
            # v_diff = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
            u_diff = np.zeros((flow.shape[0], flow.shape[2], flow.shape[3]), dtype=np.float32)
            v_diff = np.zeros((flow.shape[0], flow.shape[2], flow.shape[3]), dtype=np.float32)

        print("Flow gt mean and std:", np.mean(flow_gt), np.std(flow_gt))
        print("Flow pred mean and std:", np.mean(flow), np.std(flow))

        """
        plot_distribution(data_original_interpol, dir_model, name="dist_density_gt.png", save=True)
        plot_distribution(interpol_data, dir_model, name="dist_density_interpol.png", save=True)
        plot_distribution(flow_gt[:, 0], dir_model, name="dict_flow_gt_x.png", save=True)
        plot_distribution(flow_gt[:, 1], dir_model, name="dict_flow_gt_y.png", save=True)
        plot_distribution(flow[:, 0], dir_model, name="dist_flow_pred_x.png", save=True)
        plot_distribution(flow[:, 1], dir_model, name="dist_flow_pred_y.png", save=True)
        """

        # # visualize_ind
        # max_flow_gt = flow_gt.max()
        # title = "Flow_gt_ind_0"
        # visualize_ind_flow(flow_gt[0], max_flow_gt, dir_model, title=title, show=False, save=True)
        # title = "Flow_pred_ind_0"
        # visualize_ind_flow(flow[0], max_flow_gt, dir_model, title=title, show=False, save=True)
        # title = "Flow_gt_ind_300"
        # visualize_ind_flow(flow_gt[300], max_flow_gt, dir_model, title=title, show=False, save=True)
        # title = "Flow_pred_ind_300"
        # visualize_ind_flow(flow[300], max_flow_gt, dir_model, title=title, show=False, save=True)
        # title = "Flow_gt_ind_600"
        # visualize_ind_flow(flow_gt[600], max_flow_gt, dir_model, title=title, show=False, save=True)
        # title = "Flow_pred_ind_600"
        # visualize_ind_flow(flow[600], max_flow_gt, dir_model, title=title, show=False, save=True)

        # """
        title = "GT_Interpol_Diff_Flow_" + str(factor) + "x"
        if args.factor == 2:
            visualize_large(data_original_interpol, interpol_data, diffs,
                flow, flow_gt, diffs_flow, u_diff, v_diff, mask,
                factor, dataset, dir_model, title=title, show=False, save=True)

        title = "Density_gt_pred_diff"
        visualize_seq_dens(data_original_interpol, interpol_data, diffs,
                factor, dataset, dir_model, title=title, show=False, save=True)
        if dataset == "lbs2d":
            title = "Flow_gt_pred_diff"
            visualize_seq_flow(flow_gt, flow, u_diff, v_diff,
                    factor, dataset, dir_model, title=title, show=False, save=True)
        # """
        
        # input("metrics")
        del model
        torch.cuda.empty_cache()
        # input("metrics")
        # Desnity metrics
        # print("original data is in range %f to %f" % (np.min(original_data_gt), np.max(original_data_gt)))
        # print("interpol data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))
        # if args.exp == 4:
        #     print(original_data_gt.shape, interpol_data.shape)
        #     original_data_gt = original_data_gt[:2000]
        #     interpol_data = interpol_data[:2000]
        avg_psnr = calculate_avg_psnr(data_original_interpol, interpol_data)
        # avg_ssim = calculate_avg_ssim(original_data_gt, interpol_data)
        avg_lpips = calculate_avg_lpips(data_original_interpol, interpol_data)
        if dataset != "vimeo2d" and dataset != "droplet2d":
            # Flow metric
            print(flow_gt.shape, flow.shape)
            avg_epe = calculate_avg_epe(flow_gt, flow)
        # save metrics score
        filename = os.path.join(dir_model, 'metrics.txt')
        with open(filename, 'w') as f:
            f.write('PSNR: {:.3f}\n'.format(avg_psnr))
            # f.write('SSIM: {:.4f}\n'.format(avg_ssim))
            f.write('LPIPS: {:.6f}\n'.format(avg_lpips))
            if dataset != "vimeo2d" and dataset != "droplet2d":
                f.write('EPE: {:.5f}\n'.format(avg_epe))

        # # 300-600 subset works
        # TODO for a member of DD
        # dist_matrix = calculate_dist_matrix(original_data_gt, interpol_data)
        # print(dist_matrix.shape)
        # print(dist_matrix)

        """
        # arbitrary interpol: factor = n
        print("arbitrary interpolation")
        combined_for_interpol = []
        gt_for_interpol_error = []
        n = 5
        for i in range(0, data_test.shape[0], n):
            # got concatenated and gt_for_interpol_error
            if i % n == 0:
                combined_for_interpol.append(data_test[i]) 
            else:
                gt_for_interpol_error.append(data_test[i]) 

            # get interpolation from LREST:
            # data_test, interpol_data, flow, flow_original, mask, statistics = train(model, args.dataset, args.factor, model_name, args.mode, args.local_rank)
        print(combined_for_interpol.shape)
        print(gt_for_interpol_error.shape)
        """

# print("Dataset:", dataset)

# min_exp = args.exp # 2 # 1
# max_exp = 7 # 7
# factors = []

# for exp in range(min_exp, max_exp + 1):
#     args.exp = exp
#     factor = 2 ** exp
#     factors.append(factor)
#     print("{}x interpolation".format(factor)) # 2 4 8 16 32 64 128
#     dir_res = "Results"
#     dir_res = os.path.join(dir_res, dataset)
#     dir_res = os.path.join(dir_res, str(factor) + "x")
#     print("Saving at:", dir_res)

#     # GT, create interpol data
#     original_data, video_name = create_gt_interpol(dataset, factor)

#     # interpol, flow
#     print("video_name:", video_name)
#     # interpol_data, flow_combined = interpolate(args, video_name, dataset, factor)
#     data_testinterpol_data, flow_combined = interpolate(args, video_name, dataset, factor)

#     # # calc diff, metrics
#     diffs = calculate_diff(original_data, interpol_data, dataset, factor, dir_res)

#     # add baseline
#     # data_to_vis = np.concatenate((np.array(original_data), np.array(interpol_data), np.array(diffs)), axis=0)
#     # print(data_to_vis.shape)
#     title = "GT_Interpol_Diff_Flow_" + str(factor) + "x"
#     if "2d" in dataset:
#         visualize_large(original_data, interpol_data, flow_combined, diffs, factor, dataset, dir_res, title=title, show=False, save=True)
        
