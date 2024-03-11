# ssh -Y hamid@129.125.75.167
# conda activate gpu
# python3 -m torch.distributed.launch --nproc_per_node=1 train.py --world_size=1 --dataset=pool3d --mode=train

# rsync -avz -e 'ssh' hamid@129.125.75.167:~/Desktop/OpticalFlow/RIFE/train_log/ /Users/hamidgadirov/Desktop/OpticalFlow/RIFE/train_log/

from matplotlib.pyplot import axes, title
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
import yt

from model.RIFE import Model
# from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from load_datasets import load_data
from metrics import calculate_diff, calculate_avg_epe
from metrics import calculate_avg_psnr, calculate_avg_ssim, calculate_avg_lpips, calculate_psnr
from utils import plot_distribution, plot_loss, plot_lr, visualize_ind, visualize_ind_flow, visualize_series, visualize_series_flow, visualize_large, \
    visualize_seq_dens, visualize_seq_flow
from baseline_interpol import *

# If your model is returning NaNs, you could set 
# torch.autograd.detect_anomaly(True) 
# at the beginning of your script to get a stack trace, which would hopefully point to the operation, which is creating the NaNs.

# TODO:
# implement smooth loss +
# implement photometric loss +
# implement laplacian pyramid loss +
# merge with 2D model (parts) -
# fix bug with 3rd dim +
# mixed precision model (3D) +
# remove amp and try on Habrok +

device = torch.device("cuda")

# exp = os.path.abspath('.').split('/')[-1]
log_path = 'train_log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        # return 1e-4 * mul
        # return 3e-4 * mul
        # return 6e-4 * mul # 2D
        return 6e-5 * mul
        # return 1e-5 * mul
        # return 6e-6 * mul
        # return 1e-6 * mul 
        # return 3e-7 * mul
        # return 1e-7 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        # return (1e-4 - 1e-6) * mul + 1e-6
        # return (3e-4 - 3e-6) * mul + 3e-6
        # return (6e-4 - 6e-6) * mul + 6e-6 # 2D
        return (6e-5 - 6e-7) * mul + 6e-7
        # return (1e-5 - 1e-7) * mul + 1e-7
        # return (6e-6 - 6e-7) * mul + 6e-7
        # return (1e-6 - 1e-8) * mul + 1e-8
        # return (3e-7 - 3e-8) * mul + 3e-8
        # return (1e-7 - 1e-9) * mul + 1e-9
    # return 1e-6 # worse

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
    #     writer = SummaryWriter('train')
    #     writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0
    early_stopper = EarlyStopper(patience=50, min_delta=0.01) # 30
    batch_size_inf = 2 # 6 3

    num_workers = 1 # 8
    if mode == "train":
        data_train, data_val = load_data(dataset, factor, mode)
        # sampler = DistributedSampler(data_train)
        # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
        # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
        train_data = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        args.step_per_epoch = train_data.__len__()
        # input("train loaded")
        print(len(data_train))
        print(data_train[0].shape)
        val_data = DataLoader(data_val, batch_size=args.batch_size, pin_memory=True, num_workers=num_workers)
        print("len:", len(val_data))
        # input("x")
        # print(len(train_data.dataset))
        # for i, data in enumerate(train_data):
        #     print(i)
    else:
        # data_test = load_data(dataset, mode)
        data_test, data_test_non_overlap = load_data(dataset, factor, mode)
        shuffle = False
        data_test = DataLoader(data_test, batch_size=batch_size_inf, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    # input("x")

    lapl_loss = True
    l1_reg = True
    smooth_loss = False
    each_third = True
    aug = False

    # model_name = "flownet_lapl_reg_nosmooth_3rd_drop50K.pkl"
    # model_name = "flownet"
    # model_name += "_lapl" if lapl_loss else ""
    # model_name += "_reg" if l1_reg else ""
    # model_name += "_smooth" if smooth_loss else ""
    # model_name += "_3rd" if each_third else ""
    # model_name += "_aug" if aug else ""
    # model_name += "_" + dataset
    # model_name += ".pkl"
    # print(model_name)
    # input("x")

    # model.load_model(model_name, log_path)
    try:
        model.load_model(model_name, log_path)
        # print("Loaded RIFE model.")
        # print(model) # RIFE is not a neural net
    except:
        print("No weights found, training from scratch.")
        print("model name:", model_name)
    # input("x")

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # num_params = count_parameters(model_name)
    # print(num_params)
    # input("x")

    if mode == "train":
        print('training...')
        loss_values = []
        time_stamp = time.time()
        for epoch in range(args.epoch):
            # sampler.set_epoch(epoch)
            for i, data in enumerate(train_data):
                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                # data = torch.zeros([16, 9, 150, 450], dtype=torch.float)
                data_gpu = data.to(device, non_blocking=True) # / 255.
                # data_gpu = data_gpu.permute(0, 3, 1, 2)
                # print("data to gpu:", data_gpu.shape)
                # print(type(data_gpu[0,0,0,0,0]))
                # data_gpu = data_gpu.float16()
                # input("x")
                # input("data")

                # imgs = data_gpu[:, :2] # :6
                # gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network

                # data_gpu = data_gpu.float() # to float32
                # print("Data type of the tensor:", data_gpu.dtype)

                # random triplets:
                img0 = data_gpu[:, :1]
                gt = data_gpu[:, 1:2]
                img1 = data_gpu[:, 2:3]
                t_mask = data_gpu[:, 3:4]
                imgs = torch.cat((img0, img1), dim=1)
                
                # print("Training")
                # print("imgs, gt:", imgs.shape, gt.shape)
                # input("x")
                # imgs = imgs.permute(0, 1, 4, 2, 3) # 3rd is dim0
                # gt = gt.permute(0, 1, 4, 2, 3)

                # imgs = imgs.permute(0, 1, 2, 5, 3, 4) # 3rd is dim0
                # gt = gt.permute(0, 1, 2, 5, 3, 4)
                # t_mask = t_mask.permute(0, 1, 2, 5, 3, 4)
                # print("permuted:", imgs.shape, gt.shape)

                # g_t = gt.detach().cpu().numpy()
                # print(np.mean(imgs.detach().cpu().numpy()))
                # print(np.mean(gt.detach().cpu().numpy()))
                # print(g_t.shape)
                # input("imgs gt before iter")

                # learning_rate = get_learning_rate(step)
                learning_rate = get_learning_rate(step) * args.world_size / 4
                # with torch.autograd.detect_anomaly():
                # model.half()
                    # with torch.cuda.amp.autocast(True):
                pred, info = model.update(imgs, gt, t_mask, dataset, learning_rate, training=True)

                # with torch.cuda.amp.autocast(True):
                #     pred, info = model.update(imgs, gt, t_mask, dataset, learning_rate, training=True)
                
                # print("pred:", pred.shape, gt.shape, info['mask'].shape)
                # pred = pred.permute(0, 1, 3, 4, 2) # (0, 1, 3, 4, 2)
                # gt = gt.permute(0, 1, 3, 4, 2)
                # info['mask'] = info['mask'].permute(0, 1, 3, 4, 2)
                # info['mask_tea'] = info['mask_tea'].permute(0, 1, 3, 4, 2)
                # info['merged_tea'] = info['merged_tea'].permute(0, 1, 3, 4, 2)
                # info['flow'] = info['flow'].permute(0, 1, 3, 4, 2)
                # info['flow_tea'] = info['flow_tea'].permute(0, 1, 3, 4, 2)
                # # print("permute back:", pred.shape, gt.shape)

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
                    # gt = (gt.permute(0, 2, 3, 4, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 4, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # pred = (pred.permute(0, 2, 3, 4, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # merged_img = (info['merged_tea'].permute(0, 2, 3, 4, 1).detach().cpu().numpy() * 255).astype('uint8')
                    # flow0 = info['flow'].permute(0, 2, 3, 4, 1).detach().cpu().numpy()
                    # flow1 = info['flow_tea'].permute(0, 2, 3, 4, 1).detach().cpu().numpy()
                    # for i in range(5):
                    #     # print(merged_img.shape, pred.shape, gt.shape)
                    #     imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    #     print(imgs.shape)
                    #     input("img")
                    #     writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    #     writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    #     writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                    # writer.flush()
                if local_rank == 0:
                    print('epoch:{}/{} {}/{} time:{:.2f}+{:.2f} loss_G:{:.4e}' \
                        .format(epoch, args.epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_G']))
                step += 1
            nr_eval += 1
            # if nr_eval % 5 == 0:
            #     # evaluate(model, val_data, step, local_rank)
            #     evaluate(model, dataset, val_data, step, local_rank, writer_val)
            # model.save_model(model_name, log_path, local_rank) # save after checking early stopping
            if epoch == 0:
                print("epoch == 0")
                global val_loss_best 
                val_loss_best = info['loss_G']
                print(val_loss_best)
            evaluate(model, dataset, val_data, step, local_rank, learning_rate, early_stopper) #, writer_val)   
            # evaluate(model, dataset, val_data, step, local_rank, learning_rate) #, writer_val)
            # print("evaluated")
            # dist.barrier() # ?
    else:
        print("inference...")
        # factor = 2
        dir_res = "Results"
        dir_res = os.path.join(dir_res, dataset)
        dir_res = os.path.join(dir_res, str(factor) + "x")
        dir_model = os.path.join(dir_res, model_name[:-4])
        # print(dir_model)
        # input("x")
        print("Saving at:", dir_model)
        # input("x")
        # data = data_train
        # scale_list = [4, 2, 1]
        # scale_list = [1, 1, 1]
        # scale_list = [2, 1]
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
        # flow_gt_combined = []
        # data_test_combined = []
        print(f"inference for {factor}x interpolation")
        for i, data in enumerate(data_test):
            print("data:", data.shape)
            # input("x")
            data_gpu = data

            # data_gpu = data_gpu.permute(0, 1, 2, 5, 3, 4) # 3rd is dim0

            # data_gpu = data_gpu.float() # to float32

            imgs = data_gpu[:, :2, 0]
            gt = data_gpu[:, 2:3, 0]
            # print(imgs.shape)
            # print(gt.shape)
            # input("imgs gt")
            
            I0 = imgs[:, :1]
            I1 = imgs[:, 1:2]
            # arbitrary interpol
            arbitrary_interpol = True
            if arbitrary_interpol:
                for j in range(1, factor):
                    interpol_step = float(j) / factor # t
                    # print(interpol_step)
                    t_mask = (I0.clone() * 0 + 1) * interpol_step # create mask for arb interpol
                    # print(t_mask)
                    # input("t_mask")
                    # visualize_ind(I0.detach().cpu().numpy()[1, 0, 40], dir_res, title="I0", show=False, save=True)
                    merged, flow, mask = model.inference(I0, I1, t_mask)
                    # merged, flow, mask, warped_img0 = model.inference(I0, I1, t_mask)
                    merged_last = np.asarray(merged[-1].detach().cpu().numpy()) # from last block
                    # visualize_ind(merged_last[0, 0, 40], dir_res, title="I0_pred", show=False, save=True)
                    # visualize_ind(warped_img0.detach().cpu().numpy()[0, 0, 40], dir_res, title="warped_img0", show=False, save=True)
                    # visualize_ind(warped_img0.detach().cpu().numpy()[0, 0, 40], dir_res, title="img0", show=False, save=True)

                    # visualize_ind(np.asarray(mask[-1].detach().cpu().numpy())[0, 0, 40], dir_res, title="mask_pred", show=False, save=True)
                    # print(merged_last.shape)

                    # fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
                    # axs = axs.flatten()
                    # # Iterate over the images and plot them on the subplots
                    # for i in range(16):
                    #     # Remove the single channel dimension if it exists (assuming grayscale images)
                    #     image = merged_last[i].squeeze()

                    #     # Plot the image on the corresponding subplot
                    #     axs[i].imshow(image, cmap='gray')
                    #     axs[i].axis('off')

                    # # Adjust the spacing between the subplots
                    # plt.tight_layout()

                    # # Show the plot
                    # plt.show()
                    # # fig.savefig("droplets.png", dpi = 300)
                    # # input("x")

                    for b in range(data_gpu.shape[0]):
                        # print("merged_last[b]:", merged_last[b].shape)
                        # input("merged_last[b]:")
                        interpol_data.append(merged_last[b].transpose((1, 2, 3, 0)))

                    flow_last = np.asarray(flow[-1].detach().cpu().numpy())
                    for b in range(data_gpu.shape[0]):
                        # print("flow_last[b]:", flow_last[b].shape)
                        # input("flow_last[b]:")
                        flow_combined.append(flow_last[b])

                    mask_last = np.asarray(mask[-1].detach().cpu().numpy())
                    for b in range(data_gpu.shape[0]):
                        mask_combined.extend(mask_last[b])
            else:
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
                    for b in range(data_gpu.shape[0]):
                        interpol_data.append(merged_last[b].transpose((1, 2, 3, 0)))
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

        # for b in range(data_gpu.shape[0]):
        #     data_test_combined.append(np.asarray(data_gpu[b, 0, 0]))
        #     data_test_combined.append(np.asarray(data_gpu[b, 2, 0]))
        #     data_test_combined.append(np.asarray(data_gpu[b, 1, 0]))

        print(f"inference for {factor}x interpolation completed!")
        interpol_data = np.squeeze(np.array(interpol_data))
        print("interpolated:", interpol_data.shape)
        # input("x")

        # reorder based on factor
        if factor > 2:
            interpol_data_new = []
            batch_size = batch_size_inf # what was used for inference
            try:
                for fb in range(0, interpol_data.shape[0], (factor - 1) * batch_size): # 0 32 64 ...
                    # print("fb", fb)
                    for b in range(batch_size): # 0 1; 0 1; ...
                        for k in range(fb, fb + (factor - 1) * batch_size, batch_size): # 0 16; 32 64; ... 
                            # print("k+b:", k + b)
                            interpol_data_new.append(interpol_data[k + b])
            except:
                print("dens reordered")
            interpol_data = np.squeeze(np.array(interpol_data_new))
            print("interpolated:", interpol_data.shape)

        flow_combined = np.array(flow_combined)
        print("flow pred is in range %f to %f" % (np.min(flow_combined), np.max(flow_combined)))
        # input("flow")

        # reorder based on factor
        if factor > 2:
            flow_combined_new = []
            batch_size = batch_size_inf # what was used for inference
            try:
                for fb in range(0, flow_combined.shape[0], (factor - 1) * batch_size): # 0 32 64 ...
                    # print("fb", fb)
                    for b in range(batch_size): # 0 1; 0 1; ...
                        for k in range(fb, fb + (factor - 1) * batch_size, batch_size): # 0 16; 32 64; ... 
                            # print("k+b:", k + b)
                            flow_combined_new.append(flow_combined[k + b])
            except:
                print("flow reordered")
            flow_combined = np.squeeze(np.array(flow_combined_new))
            print("flow_combined:", flow_combined.shape)

        mask_combined = np.squeeze(np.array(mask_combined))
        # flow_gt_combined = np.array(flow_gt_combined)
        print("Flow:", flow_combined.shape)
        # input("x")

        # """
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
        # """
        # print(data_test_non_overlap.shape)
        # data_test_non_overlap = np.transpose(data_test_non_overlap, (0, 1, 4, 2, 3)) # 3rd is dim0
        return data_test_non_overlap, interpol_data, flow_combined, mask_combined

        """
            data_gpu = data_gpu.to(device, non_blocking=True)
            # data = np.expand_dims(data, axis=0)
            # data_gpu = torch.from_numpy(data).to(device, non_blocking=True) # / 255.
            print("data to gpu:", data_gpu.shape)
            # input("data")
            imgs = data_gpu[:, :2] # :6
            gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network
            print(imgs.shape, gt.shape)

            flow_gt = data_gpu[:,:,1:4] # only flow in x, y, z
            # input(flow_gt.shape)
            flow_gt_array = np.asarray(flow_gt.detach().cpu().numpy())
            for b in range(data_gpu.shape[0]):
                flow_gt_combined.append(flow_gt_array[b, 0])
                flow_gt_combined.append(flow_gt_array[b, 2])
                flow_gt_combined.append(flow_gt_array[b, 1])

            imgs = imgs[:, :, 0]
            gt = gt[:, :, 0]
            # imgs = imgs[..., 0]
            # gt = gt[..., 0]
            # (299, 3, 4, 128, 128, 128) means: batch, three, fields, x, y, z

            print(imgs.shape, gt.shape) 
            # imgs = imgs.permute(0, 1, 3, 4, 2) # (0, 1, 2, 5, 3, 4) # 3rd is dim0
            # gt = gt.permute(0, 1, 3, 4, 2)
            # print("permuted:", imgs.shape, gt.shape)
            # input("x")
            I0 = imgs[:, :1]
            I1 = imgs[:, 1:2]
            # merged, flow, mask = model.inference(I0, I1, scale_list) # get the flow too
            merged, flow, mask = model.inference(I0, I1, scale_list) # get the flow too
            merged_2 = np.asarray(merged[2].detach().cpu().numpy())
            print(merged_2.shape)
            for b in range(data_gpu.shape[0]):
                interpol_data.append(merged_2[b].transpose((1,2,3,0)))

            flow_2 = np.asarray(flow[2].detach().cpu().numpy())

            for b in range(data_gpu.shape[0]):
                flow_combined.append(flow_2[b])

            mask_2 = np.asarray(mask[2].detach().cpu().numpy())
            for b in range(data_gpu.shape[0]):
                mask_combined.extend(mask_2[b])
            
        interpol_data = np.squeeze(np.array(interpol_data))
        print("interpolated:", interpol_data.shape)

        # Vector field quiver plot
        flow_combined = np.array(flow_combined)
        mask_combined = np.squeeze(np.array(mask_combined))
        flow_gt_combined = np.array(flow_gt_combined)
        print("Flow:", flow_combined.shape)
        # input("x")

        # merged = np.asarray(merged.detach().cpu().numpy())
        # print(merged.shape)
        # interpol_data.append(merged[2].transpose((0, 2, 3, 4, 1)))
        # gt = np.asarray(gt.detach().cpu().numpy())

        # merged = merged.permute(0, 1, 3, 4, 2)
        # gt = gt.permute(0, 1, 3, 4, 2)
        # flow[2] = flow[2].permute(0, 1, 3, 4, 2)
        # mask = mask.permute(0, 1, 3, 4, 2)
        # # print("permute back:", middle.shape, flow.shape)
        
        # flow = np.asarray(flow)
        # flow_array_ = flow[2].detach().cpu().numpy()
        # flow_combined.append(flow_array_[:,0:6,:,:].squeeze()) # 0:6 because in x, y, z and for F_t->0, F_t->1 intermediate flows
        # print(flow.shape)
        # mask = np.asarray(mask.detach().cpu().numpy())
        # print(mask.shape)

        interpol_data = np.squeeze(np.array(interpol_data))
        # interpol_data = np.array(interpol_data)
        # interpol_data = interpol_data.reshape(17, 120, 180, 300, 1)
        print("Interpolated:", interpol_data.shape)
        slice_num = 40
        # print(interpol_data[:, slice_num, :, :])

        # print("gt:",original_data[:, slice_num, :, :])
        title = title = "inference_" + str(factor) + "x"
        if "2d" in dataset:
            visualize_series(interpol_data, factor, dataset, dir_res, title=title, show=False, save=True)
        if "3d" in dataset:
            visualize_series(interpol_data[..., slice_num], factor, dataset, dir_res, title=title, show=False, save=True)
            # visualize_series(original_data[..., slice_num], factor, dataset, dir_res, title=title+"GT", show=False, save=True)

        print(interpol_data[0].mean(axis=1))

        # Interpolated: (300, 160, 224)                                                                             
        # Flow: (300, 4, 160, 224) 

        # Vector field quiver plot
        flow_combined = np.array(flow_combined)
        print("Flow:", flow_combined.shape)
        # input("infer")
        flow_u = flow_combined[:, 2, :, :, slice_num]
        flow_v = flow_combined[:, 3, :, :, slice_num]
        title = "inference_flow_arrows_" + str(factor) + "x"
        if "2d" in dataset:
            visualize_series_flow(interpol_data, flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)
        if "3d" in dataset:
            visualize_series_flow(interpol_data[..., slice_num], flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)

        loss_path = 'loss.json'
        loss_path = os.path.join(dir_model, loss_path)
        try:
            with open(loss_path, 'r') as loss_file:
                loss_data = json.load(loss_file)
            val_loss = loss_data['val_loss']
            loss_file.close()   
            # val_loss = list(flatten(val_loss))
            val_loss = np.array(val_loss)
            # print(val_loss.shape)
            # input("x")
        except:
            print("loss json doesn't exist")

        plot_loss(val_loss, dir_model, name="val_loss.png", save=True)

        original_data = np.array(data_test_combined)
        print("original_data:", original_data.shape)
        print("Data is in range %f to %f" % (np.min(original_data), np.max(original_data)))

        # return data_test, interpol_data, flow_combined
        return original_data, interpol_data, flow_combined, flow_gt_combined, mask_combined
    
    """
    
def evaluate(model, dataset, val_data, nr_eval, local_rank, learning_rate, early_stopper): #, writer_val):
    print("Evaluate")
    # writer = SummaryWriter('train') #
    # writer_val = SummaryWriter('validate') #
    # psnr_list = []
    # psnr_list_teacher = []
    loss_all_list = [0. for i in range(7)] # all loss components
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu = data.to(device, non_blocking=True) # / 255.
        # imgs = data_gpu[:, :2]
        # gt = data_gpu[:, 2:3]

        # data_gpu = data_gpu.float() # to float32

        img0 = data_gpu[:, :1]
        gt = data_gpu[:, 1:2]
        img1 = data_gpu[:, 2:3]
        t_mask = data_gpu[:, 3:4]
        imgs = torch.cat((img0, img1), dim=1)

        # imgs = imgs.permute(0, 1, 2, 5, 3, 4) # 3rd is dim0
        # gt = gt.permute(0, 1, 2, 5, 3, 4)
        # t_mask = t_mask.permute(0, 1, 2, 5, 3, 4)
        # print("permuted:", imgs.shape, gt.shape)

        with torch.no_grad():
            # pred, info = model.update(imgs, gt, dataset, training=False)
            pred, info = model.update(imgs, gt, t_mask, dataset, training=False)
            merged_img = info['merged_tea']

        # pred = pred.permute(0, 1, 3, 4, 2)
        # gt = gt.permute(0, 1, 3, 4, 2)
        # merged_img = merged_img.permute(0, 1, 3, 4, 2)
        # info['flow'] = info['flow'].permute(0, 1, 3, 4, 2)
        # info['flow_tea'] = info['flow_tea'].permute(0, 1, 3, 4, 2)
        # print("permute back:", pred.shape, info['flow'].shape)

        loss_all = (info['loss_G'].cpu().numpy(), info['loss_l1'].cpu().numpy(), info['loss_tea'].cpu().numpy(), 
            info['loss_distill'].cpu().numpy(), info['l1_reg'].cpu().numpy(), info['loss_photo'].cpu().numpy(), 
            info['loss_flow'].cpu().numpy())
        print("loss_G:", info['loss_G'])
        loss_all_list = [a + b for a, b in zip(loss_all_list, loss_all)]
        # loss_l1_list.append(info['loss_l1'].cpu().numpy())
        # loss_tea_list.append(info['loss_tea'].cpu().numpy())
        # loss_distill_list.append(info['loss_distill'].cpu().numpy())
        # loss_G_list.append(info['loss_G'].cpu().numpy())
        # for j in range(gt.shape[0]):
        #     max_shape_2 = min(gt.shape[2], pred.shape[2])
        #     max_shape_3 = min(gt.shape[3], pred.shape[3])
        #     max_shape_4 = min(gt.shape[4], pred.shape[4])
        #     # print(max_shape_2, max_shape_3)
        #     gt = gt[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
        #     pred = pred[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
        #     psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
        #     psnr_list.append(psnr)
        #     psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
        #     psnr_list_teacher.append(psnr)
        # gt = (gt.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype('uint8')
        # pred = (pred.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype('uint8')
        # merged_img = (merged_img.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype('uint8')
        # flow0 = info['flow'].permute(0, 2, 3, 4, 1).cpu().numpy()
        # flow1 = info['flow_tea'].permute(0, 2, 3, 4, 1).cpu().numpy()
        # if i == 0 and local_rank == 0:
        #     for j in range(10):
        #         imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
        #         writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    # # early stopping
    # val_loss_now = info['loss_G']
    # global val_loss_best 
    # global early_stop_k
    # if val_loss_now > val_loss_best:
    #     early_stop_k += 1
    #     if early_stop_k == early_stop_patience:
    #         print("Early stopping")
    #         sys.exit(0)
    # else:
    #     early_stop_k = 0
    #     val_loss_best = val_loss_now
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
    #     json.dump(learning_rate, lr_file, indent=4)
    # lr_file.close()
    
    eval_time_interval = time.time() - time_stamp

    del loss_all

    if local_rank != 0:
        return
    # writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int) # 300 500
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size') # default=16
    parser.add_argument('--local-rank', default=0, type=int, help='local rank') # local-rank
    parser.add_argument('--world_size', default=1, type=int, help='world size') # 4
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    parser.add_argument('--mode', dest='mode', type=str, default='test')
    parser.add_argument('--exp', dest='exp', type=int, default=1)
    args = parser.parse_args()
    assert (not args.dataset is None)

    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank)
    if args.dataset == "rectangle3d":
        args.batch_size = 40
    if args.dataset == "jets3d":
        args.batch_size = 4 # 3 6 8
    if args.dataset == "pool3d":
        args.batch_size = 2 # 2 4 32
    if args.dataset == "droplet3d":
        args.batch_size = 20
    # if args.dataset == "tangaroa3d":
    #     args.batch_size = 15

    """ rectangle """
    model_name = "flownet_l1_nodistill_90_rect3d.pkl" # 0
    model_name = "flownet_l1_nodistill_90_aug_rect3d.pkl" # 0
    model_name = "flownet_l1_nodistill_modif90_aug_rect3d.pkl" #
    model_name = "flownet_l1_nodistill_modif90_100_aug_rect3d.pkl" #
    model_name = "flownet_l1_modif90_aug_rect3d_.pkl" # not 0 but bad
    model_name = "flownet2_l1only_aug_rect3d_.pkl" # not 0 but not good; 1000+ ep: a bit better but not good
    model_name = "flownet2_l1only_64_aug_rect3d_.pkl" # not 0 but not good
    model_name = "flownet2_l1only_64_kernel4_aug_rect3d_.pkl" #
    # model_name = "flownet2_l1only_64_2IFblocks_aug_rect3d_.pkl" # empty
    # model_name = "flownet2_l1_distillnomask_aug_rect3d_.pkl" # not 0 but not good
    model_name = "flownet2_l1_smooth_64_kernel4_aug_rect3d_.pkl" #
    model_name = "flownet2_l1_photo_64_aug_rect3d_.pkl" #
    model_name = "flownet2_lapl_test_64_aug_rect3d.pkl" #
    model_name = "flownet2_l1_test_128_aug_rect3d.pkl" #
    model_name = "flownet2_l1_dim_64_aug_rect3d.pkl" # first version after fixing dim0
    model_name = "flownet2_l1_distill_dim_64_aug_rect3d.pkl"
    model_name = "flownet2_l1_fix_dim_64_aug_rect3d.pkl" # fixed / 255 bug
    model_name = "flownet2_l1_photo_dim_64_aug_rect3d.pkl"
    # model_name = "flownet2_l1_distill_photo_dim_64_aug_rect3d.pkl"
    # model_name = "flownet2_l1_dim_90_aug_rect3d.pkl" # 
    model_name = "flownet_l1_dist_64_rect3d.pkl"
    model_name = "flownet_l1_64_rect3d.pkl"

    model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_RIFE_eqRes.pkl"
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_gamma_RIFE_eqRes.pkl" # loss nan
    # model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_gamma_RIFE_eqRes_test.pkl"
    model_name = "flownet_l1_flow5e-1_v1_128_rect_big_fbb_gamma_RIFE_eqRes_lr.pkl" # loss diverges, lr too high
    model_name = "flownet_l1_reg_flow5e-1_v1_128_rect_big_fbb_gamma_RIFE_eqRes_lr.pkl"
    model_name = "flownet_l1_reg_flow1e-1_v1_128_rect_big_fbb_gamma_RIFE_eqRes_lr_test.pkl"
    model_name = "flownet_l11e-1_reg1e-6_flow1e-2_v1_128_rect_big_fbb_gamma_RIFE_eqRes_lr_test.pkl" # loss constant, lr too low

    """ jets """
    # model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes.pkl"
    # model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test.pkl" 
    # model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test2.pkl" 
    # model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test_BN.pkl" 
    # model_name = "flownet_l1_reg_flow5e-1_v1_128_jets_RIFE_eqRes_test_lr.pkl" 
    model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test3.pkl" 
    model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test4.pkl" # wrong preprocessing?
    model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test5.pkl" # loss increases -
    model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test6.pkl" # lr3e-5: loss increases -
    model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test7.pkl" # lr6e-6: loss increases -
    model_name = "flownet_l1_flow5e-1_v1_128_jets_RIFE_eqRes_test8.pkl" # lr3e-7: loss constant -
    model_name = "flownet_3Blocks_l1_flow2e-1_v1_128_jets_gamma_eqRes_deconv_lr6e-4_test.pkl"
    model_name = "flownet_3Blocks_l1_flow2e-1_v1_128_jets_gamma_eqRes_deconv_lr6e-4_test1.pkl"
    model_name = "flownet_3Blocks_l1_flow2e-1_v1_128_jets_gamma_eqRes_deconv_lr6e-4_test2.pkl"
    model_name = "flownet_3Blocks_l1_v1_128_jets_gamma_eqRes_deconv_lr6e-4_test.pkl"
    model_name = "flownet_3Blocks_l1_v1_128_jets_gamma_eqRes_deconv_lr1e-6_test.pkl"
    # is there a bug? why loss does not decrease? - yes.
    # fixed bug with the data
    model_name = "flownet_3Blocks_l1_flow2e-1_v1_128_jets_gamma_eqRes_deconv_lr6e-4_databugfix.pkl"
    model_name = "flownet_3Blocks_l1_v1_128_jets_gamma_eqRes_deconv_lr6e-4_databugfix.pkl"
    # model_name = "flownet_3Blocks_l1_flow2e-2_v1_128_jets_gamma_eqRes_deconv_lr6e-4_databugfix.pkl"
    model_name = "flownet_3Blocks_l1_flow2e-2_v1_128_jets_gamma_eqRes_deconv_lr6e-4_databugfix_1.pkl"
    # model_name = "flownet_3Blocks_l1_flow2e-3_v1_128_jets_gamma_eqRes_deconv_lr6e-4_databugfix.pkl"
    # model_name = "flownet_3Blocks_l1_flow2e-3_v1_128_jets_gamma_eqRes_deconv_lr1e-6_databugfix.pkl"
    # model_name = "flownet_3Blocks_l1_flow2e-1_v1_128_jets_gamma_eqRes_deconv_lr1e-6_databugfix.pkl"
    model_name = "flownet_3Blocks_l1_flow2e-1_v1_128_jets_gamma_eqRes_deconv_lr1e-6_databugfix_noamp.pkl"
    # dataset size (150) is too small, TODO: random triplet sampling as in 2D
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6.pkl" # 2e-3
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6_1.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6.pkl"
    model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6.pkl" # no flow
    model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6_1.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-5.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow1_v1_128_jets_triplets_W12simple_500_gamma_eqRes_deconv_lr1e-6.pkl"
    # [-1, 1] scaling flow
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_lr1e-6.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_noPReLU_lr1e-6.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_Tanh_lr1e-6.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_lr1e-7.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_lr6e-4.pkl" # better!
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W60_1000_gamma_eqRes_deconv_lr6e-4.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_lr6e-4_testdims3.pkl" 
    # use_deconv was False
    # bug fixed with wapring dims 
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_jets_triplets_W60_500_gamma_eqRes_deconv_lr6e-4_bugfixdims.pkl"
    # very good results 2x: 33.158 0.03766, but los jumped after epoch 30; 8 conv were used!
    model_name = "flownet_3Blocks_8conv_l1_flow2e-1_v1_128_jets_triplets_W60_1000_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 36.14 0.0374 loss jumped at ~20 epoch ???
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W60_1000_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 55.55 0.0034; 3x: PSNR: 52.98 0.0034; 5x: 46.73 .00341; 8x: 47.09 0.0034; 16x: PSNR: 44.50 0.0034; 

    """ pool3d """
    # untrained: 2x: 35.01 0.506
    ### +++
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 46.21 0.126; 3x: 44.07 0.126
    # 128 - 2x: 34.77 0.167; 3x: 33.24 0.167
    ### +++
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_4K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 42.17 0.108
    model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_4K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 46.09 0.120
    # W8
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W8_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 47.26 0.125
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W8_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 46.84 0.125

    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_pool_triplets_W12_4K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 45.82 0.125 exploded -

    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 45.09 0.125
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 40.32 0.135
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 42.68 0.125
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 45.50 0.127 # this was wrong - flow for 3 blocks!
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4_1.pkl"
    # 2x: 45.24 0.122 3x: 42.98 0.122
    # no student-teacher
    # model_name = "flownet_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 46.14 0.125 3x: 44.29 0.125
    # TODO: remove teacher completely, add ?
    # 128
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl" # explode
    # 2x: 35.37 0.154
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_triplets_W12_1K_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 34.71 0.122
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_pool128_triplets_W12_1K_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 33.75 0.136
    # this won't work - not enough data
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_triplets_W12_1500_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 32.97 0.153 # 30 ep 2x: 34.85 0.132 200 ep

    # float 16 with amp
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_float16amp_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverge -
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16amp_triplets_W12_1K_gamma_eqRes_deconv_lr1e-5.pkl"
    # diverge -
    # float 16 withput amp
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_float16_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 39.64 0.136 lr 1e-5
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_500_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 32.36 0.209
    # PSNR: 32.359 32.365 # float16, convert to 32 vs float32
    # LPIPS: 0.019055
    # EPE: 0.20927 0.20929
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 33.86 0.207 30ep
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverge - # 2x: 33.75 3x: 0.216
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr1e-5.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_triplets_W12_500_gamma_eqRes_deconv_lr6e-5.pkl"
    # 2x: 35.19 0.205
    # untrained: 2x: 34.47 0.557

    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4_1.pkl"
    # 35.025 0.166 trained on upsamled, inf on 128 orig; 
    # vs:
    # 50.49 0.1248 trained on upsamled, inf on upsamled 5x: 44.00 0.1245 90ep
    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverege -
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 50.65 0.1254 5x: 44.38 0.12496 3 blocks 170ep
    # ~100 ep is enough!
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4_1.pkl"
    # diverege -
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_90_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverege -
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_32_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 48.70 0.123


    """ tangaroa """
    # model_name = "flownet_lapl_tangaroa3d.pkl"
    # model_name = "flownet_l1_tangaroa3d.pkl"

    """ droplet """
    # model_name = "flownet_lapl_droplet3d.pkl"
    # model_name = "flownet_l1_droplet3d.pkl"
    # model_name = "flownet_l1_90_drop3d.pkl" # 0 distill
    # model_name = "flownet_l1_nodistill_90_drop3d.pkl" #
    # model_name = "flownet_l1_modif90_drop3d.pkl" #
    # model_name = "flownet_l1_64_drop3d.pkl" # not empty but bad
    # model_name = "flownet_l1_distill_64_drop3d.pkl" # not nan but bad
    # model_name = "flownet_l1_distill_64_500ep_drop3d.pkl"
    # model_name = "flownet16_l1_64_drop3d.pkl" # nan loss

    # print(model_name)
    # input("x")

    if args.mode == "train":
        train(model, args.dataset, args.exp, model_name, args.mode, args.local_rank)
    else:
        dataset = args.dataset
        print("inference...")
        # factor = 2

        if dataset == "jets3d" or dataset == "pool3d": 
            # factors = [2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64]
            # factors = [2, 3, 5, 8, 16, 32]
            factors = [2, 3, 5, 8, 16]
        for factor in factors:
            dir_res = "Results"
            dir_res = os.path.join(dir_res, dataset)
            dir_res = os.path.join(dir_res, str(factor) + "x")
            dir_model = os.path.join(dir_res, model_name[:-4])
            # print(dir_model)
            # input("x")
            print("Saving at:", dir_model)

            # train(model, args.dataset, args.mode, args.local_rank)
            # data_test, interpol_data, flow, flow_original, mask = train(model, args.dataset, args.exp, model_name, args.mode, args.local_rank)
            data_test, interpol_data, flow, mask = train(model, args.dataset, factor, model_name, args.mode, args.local_rank)

            # data_test = np.transpose(data_test, (0, 1, 4, 2, 3)) # 3rd is dim0

            print("data_test:", data_test.shape)
            print("interpol_data:", interpol_data.shape)
            print("flow:", flow.shape)
            print("mask:", mask.shape)
            # input("inf")
            # original_data = data_test[:,1,:interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
            # original_data = data_test[:,1,...]
            # original_data = data_test[:, :interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
            if dataset == "jets3d" or dataset == "pool3d": 
                flow_test = data_test[:, 1:] # flow
                data_test = data_test[:, 0] # density
            else:
                data_test = data_test[:, :interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
                flow_test = np.zeros((data_test.shape[0], data_test.shape[1], data_test.shape[2]), dtype=np.float32)
            print("data_test:", data_test.shape)
            print("flow_test:", flow_test.shape)

            data_original = [] # I0, I1 for interpol
            data_original_interpol = [] # gt for interpol comparison
            n = factor # arbitrary interpol factor

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
            flow_original_pred = [f for i, f in enumerate(flow_test) if i % n != 0]
            flow_original_pred = np.array(flow_original_pred)
            print("flow_original_pred:", flow_original_pred.shape)
            # flow_gt = flow_original_interpol
            # input("n")

            flow = flow[:, 3:6, ...]
            # flow = np.transpose(flow, (0, 1, 2, 4, 3)) # 3rd is dim0

            """
            original_data_gt = [data for i, data in enumerate(original_data) if (i - 1) % 3 == 0]
            original_data_gt = np.array(original_data_gt)

            print("flow_original:", flow_original.shape)
            flow_gt = [flow for i, flow in enumerate(flow_original) if (i - 1) % 3 == 0]
            flow_gt = np.array(flow_gt)

            print("original_data_gt:", original_data_gt.shape)
            print("interpol_data:", interpol_data.shape)
            print("flow_gt:", flow_gt.shape)
            print("flow pred & mask:", flow.shape, mask.shape)

            # interpol_data = interpol_data
            print("original data is in range %f to %f" % (np.min(original_data_gt), np.max(original_data_gt)))
            print("interpol data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))
            # input("x")
            """

            # # TODO: denormalize
            # print("Data gt mean and std:", np.mean(original_data_gt), np.std(original_data_gt))
            # print("Data pred mean and std:", np.mean(interpol_data), np.std(interpol_data))
            # print("Flow gt mean and std:", np.mean(flow_gt), np.std(flow_gt))
            # print("Flow pred mean and std:", np.mean(flow), np.std(flow))

            # calc diff, metrics
            diffs = calculate_diff(data_original_interpol, interpol_data, dataset, factor, dir_model)
            print("diffs:", diffs.shape)

            if dataset == "jets3d" or dataset == "pool3d":
                print("GT flow available")
                # flow_gt = flow_gt[:, :, :flow.shape[2], :flow.shape[3]]
                flow_original_pred = flow_original_pred[:, :, :flow.shape[2], :flow.shape[3], :flow.shape[4]]

                print("flow original pred is in range %f to %f" % (np.min(flow_original_pred), np.max(flow_original_pred)))
                print("flow pred is in range %f to %f" % (np.min(flow), np.max(flow)))
                # input("x")

                u_gt = flow_original_pred[:, 0, ...]
                v_gt = flow_original_pred[:, 1, ...]
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

            print("Flow original pred mean and std:", np.mean(flow_original_pred), np.std(flow_original_pred))
            print("Flow pred mean and std:", np.mean(flow), np.std(flow))

            """
            flow_gt = flow_gt[:, :, :flow.shape[2], :flow.shape[3], :flow.shape[4]]
            u_gt = flow_gt[:, 0, ...]
            v_gt = flow_gt[:, 1, ...]
            norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
            # viz Ft->1 in the end, that is flow[2][:, 3:6] in 3D
            u = flow[:, 2, ...]
            v = flow[:, 3, ...]

            norm = np.sqrt(u * u + v * v)
            norm_gt = norm_gt[:, :norm.shape[1], :norm.shape[2], :norm.shape[3]]
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

            diffs_flow = calculate_diff(norm_gt, norm, dataset, factor, dir_model)

            # u_gt = (u_gt - np.min(u_gt)) / (np.max(u_gt) - np.min(u_gt))
            # v_gt = (v_gt - np.min(v_gt)) / (np.max(v_gt) - np.min(v_gt))
            u_gt = u_gt[:, :u.shape[1], :u.shape[2], :u.shape[3]]
            v_gt = v_gt[:, :v.shape[1], :v.shape[2], :v.shape[3]]
            u_diff = calculate_diff(u_gt, u, dataset, factor, dir_model)
            v_diff = calculate_diff(v_gt, v, dataset, factor, dir_model)
            # print(u_gt, u, u_diff)
            # input("x")
            # u_diff = (u_diff - np.min(u_diff)) / (np.max(u_diff) - np.min(u_diff))
            # v_diff = (v_diff - np.min(v_diff)) / (np.max(v_diff) - np.min(v_diff))

            print("Flow gt mean and std:", np.mean(flow_gt), np.std(flow_gt))
            print("Flow pred mean and std:", np.mean(flow), np.std(flow))

            # title = "GT_Interpol_Diff_Flow_" + str(factor) + "x" + "_3D"
            # visualize_large_3d(original_data, interpol_data, diffs, 
            #         flow, flow_gt, diffs_flow, u_diff, v_diff, mask,
            #         factor, dataset, dir_model, title=title, show=False, save=True)

            print(original_data.shape, interpol_data.shape, diffs.shape)
            print(flow.shape, flow_gt.shape, diffs_flow.shape)
            print(u_diff.shape, v_diff.shape, mask.shape)
            """

            print("xxxxxxx")
            print("data_original_interpol:", data_original_interpol.shape)
            print("interpol_data:", interpol_data.shape)
            print("flow_original_pred:", flow_original_pred.shape)
            print("flow:", flow.shape)

            yt_viz = False
            if yt_viz:
                # data_test_vis = interpol_data[0]
                # ds = yt.load_uniform_grid({"density": data_test_vis}, data_test_vis.shape)
                # # Create a scene
                # sc = yt.create_scene(ds, lens_type="perspective")
                # # Get a reference to the VolumeSource associated with this scene
                # source = sc[0]
                # # Set the bounds of the transfer function based on your data
                # # source.tfh.set_bounds((0., data_test_vis.max()))
                # # Make underdense regions appear opaque
                # # source.tfh.grey_opacity = True
                # # Plot the transfer function
                # # filename = os.path.join(dir_model, 'transfer_function_pool.png')
                # # source.tfh.plot(filename, profile_field=("gas", "density"))
                # source.set_field(("gas", "density"))
                # # Save the rendered image
                # filename = os.path.join(dir_model, 'yt_rendering_pool.png')
                # sc.save(filename, sigma_clip=6.0)
                # # input("yt_rendering_grid")
                # print("density rendered")

                def linramp(vals, minval, maxval):
                    return (vals - vals.min()) / (vals.max() - vals.min())

                data_test_vis = flow[0, 0]
                ds = yt.load_uniform_grid({"density": data_test_vis}, data_test_vis.shape)
                # Create a scene
                sc = yt.create_scene(ds, lens_type="perspective")
                """
                # sc = yt.create_scene(ds, field=("gas", "density"))
                        source = sc.get_source()
                source.transfer_function.map_to_colormap(
                    1e-10, 1007.0, scale=15, colormap="cmyt.arbre", scale_func=linramp
                )
                source.transfer_function.grey_opacity = True
                cam = sc.camera
                cam.resolution = 256
                cam.zoom(0.8)

                im = sc.render()
                im.write_png(os.path.join(save_path, f'{file_name[:-4]}_{frame}.png'), background=[0.0, 0.0, 0.0, 0.0], sigma_clip=6.0)
                """

                # Get a reference to the VolumeSource associated with this scene
                source = sc[0]
                # Set the bounds of the transfer function based on your data
                # source.tfh.set_bounds((0., data_test_vis.max()))
                # Make underdense regions appear opaque
                # source.tfh.grey_opacity = True
                # Plot the transfer function
                # filename = os.path.join(dir_model, 'transfer_function_flow_pool.png')
                # source.tfh.plot(filename, profile_field=("gas", "density"))
                # source.set_field(('gas', 'velocity_magnitude'))
                source.set_field(('gas', 'density'))
                # Save the rendered image
                filename = os.path.join(dir_model, 'yt_rendering_flow_pool.png')
                sc.save(filename, sigma_clip=6.0)
                # input("yt_rendering_grid")
                print("flow rendered")

            data_original_interpol = data_original_interpol[:interpol_data.shape[0]]
            flow_original_pred = flow_original_pred[:flow.shape[0]]

            # 3D metrics
            avg_psnr = calculate_avg_psnr(data_original_interpol, interpol_data, dir_model, factor)
            if "3d" in dataset:
                # Flow metric
                # print(flow_gt.shape, flow.shape)
                print(flow_original_pred.shape, flow.shape)
                avg_epe = calculate_avg_epe(flow_original_pred, flow, dir_model, factor)

            # 2D for VIS
            # TODO check dims
            slice_num = 40
            data_test = data_test[..., slice_num, :, :]
            data_original_interpol = data_original_interpol[..., slice_num, :, :]
            interpol_data = interpol_data[..., slice_num, :, :]
            diffs = diffs[..., slice_num, :, :]
            flow = flow[..., slice_num, :, :]
            # flow = flow[..., slice_num, :]
            flow_original_pred = flow_original_pred[..., slice_num, :, :]
            # flow_original_pred = flow_original_pred[..., slice_num, :]
            diffs_flow = diffs_flow[..., slice_num, :, :]
            u_diff = u_diff[..., slice_num, :, :]
            v_diff = v_diff[..., slice_num, :, :]
            mask = mask[..., slice_num, :, :]

            # title = "original_data_" + str(factor) + "x" + "_2D"
            # visualize_series(original_data, factor, dataset, dir_model, title=title, show=False, save=True)
            # title = "interpol_data" + str(factor) + "x" + "_2D"
            # visualize_series(interpol_data, factor, dataset, dir_model, title=title, show=False, save=True)

            # title = "GT_Interpol_Diff_Flow_" + str(factor) + "x" + "_2D"
            # visualize_large(original_data, interpol_data, diffs,
            #     flow, flow_gt, diffs_flow, u_diff, v_diff, mask,
            #     factor, dataset, dir_model, title=title, show=False, save=True)

            # visualize_series(interpol_data, factor, dataset, dir_res, title="Density_pred", show=False, save=True)
            # visualize_series(flow[:, 0], factor, dataset, dir_res, title="Velocity_x_GT", show=False, save=True)
            # visualize_series(flow[:, 0], factor, dataset, dir_res, title="Velocity_x", show=False, save=True)

            title = "GT_Interpol_Diff_Flow_" + str(factor) + "x"
            # if factor == 2:
            visualize_large(data_test, interpol_data, diffs,
                flow, flow_original_pred, diffs_flow, u_diff, v_diff, mask,
                factor, dataset, dir_model, title=title, show=False, save=True)
            # title = "Density_gt_pred_diff"
            # visualize_seq_dens(data_original_interpol, interpol_data, diffs,
            #         factor, dataset, dir_model, title=title, show=False, save=True)
            # if dataset == "jets3d" or dataset == "pool3d":
            #     title = "Flow_gt_pred_diff"
            #     visualize_seq_flow(flow_original_pred, flow, u_diff, v_diff,
            #             factor, dataset, dir_model, title=title, show=False, save=True)

            # if dataset == "droplet2d":
            #     # flow_filename = os.path.join(dir_model, 'flow_pred_LREST.pkl')
            #     flow_filename = os.path.join(dir_model, 'flow_pred_RIFE.pkl')
            #     with open(flow_filename, 'wb') as f:
            #         pickle.dump(flow, f)

            # avg_psnr = calculate_avg_psnr(data_original_interpol, interpol_data, dir_model, factor)
            # avg_ssim = calculate_avg_ssim(original_data_gt, interpol_data)
            avg_lpips = calculate_avg_lpips(data_original_interpol, interpol_data, dir_model, factor)
            # if dataset == "jets3d" or dataset == "pool3d": 
            #     # Flow metric
            #     # print(flow_gt.shape, flow.shape)
            #     print(flow_original_pred.shape, flow.shape)
            #     avg_epe = calculate_avg_epe(flow_original_pred, flow, dir_model, factor)
            # save metrics score
            filename = os.path.join(dir_model, 'metrics.txt')
            with open(filename, 'w') as f:
                f.write('PSNR: {:.3f}\n'.format(avg_psnr))
                # f.write('SSIM: {:.4f}\n'.format(avg_ssim))
                f.write('LPIPS: {:.6f}\n'.format(avg_lpips))
                if dataset != "vimeo2d" and dataset != "droplet2d":
                    f.write('EPE: {:.5f}\n'.format(avg_epe))

            # slice_num = 40
            # # calc diff, metrics
            # # diffs = calculate_diff(original_data[..., slice_num], interpol_data[..., slice_num], dataset, factor, dir_model)
            # # print("diffs:", diffs.shape)

            # title = "GT_Interpol_" + str(factor) + "x"
            # # visualize_large_3d(original_data[..., slice_num], interpol_data[..., slice_num], \
            # #     # flow_combined[:, :4, :, :, slice_num], diffs, factor, dataset, dir_model, title=title, show=False, save=True)
            # #     flow_combined[..., slice_num], diffs, factor, dataset, dir_model, title=title, show=False, save=True)
            # visualize_3d(original_data, interpol_data, flow_combined, dataset, dir_model, title=title, show=False, save=True)
