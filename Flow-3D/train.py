# ssh -Y hamid@129.125.75.167
# conda activate gpu
# python3 -m torch.distributed.launch --nproc_per_node=1 train.py --world_size=1 --dataset=nyx3d --mode=train

# rsync -avz -e 'ssh' hamid@129.125.75.167:~/Desktop/OpticalFlow/LREST/train_log/ /Users/hamidgadirov/Desktop/OpticalFlow/LREST/train_log/

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
# from torch.cuda.amp import autocast

hyper = False # hyperLREST

if hyper == False:
    from model.LREST import Model
else:
    from model.HyperLREST import Model
# from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from load_datasets import load_data
from metrics import calculate_diff, calculate_avg_epe
from metrics import calculate_avg_psnr, calculate_avg_ssim, calculate_avg_lpips, calculate_psnr
from utils import plot_distribution, plot_loss, plot_lr, visualize_ind, visualize_ind_flow, visualize_series, visualize_series_flow, visualize_large, \
    visualize_seq_dens, visualize_seq_flow, render_volume_yt, visualize_seq_dens_3d, render_density, make_video
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# exp = os.path.abspath('.').split('/')[-1]
log_path = 'train_log'

def get_learning_rate(step):
    if step < 2000: # 2000
        mul = step / 2000.
        # return 1e-3 * mul
        # return 6e-4 * mul # 2D
        # return 3e-4 * mul
        return 1e-4 * mul
        # return 6e-5 * mul
        # return 1e-5 * mul
        # return 6e-6 * mul
        # return 1e-6 * mul 
        # return 3e-7 * mul
        # return 1e-7 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        # return (1e-3 - 1e-4) * mul + 1e-4
        # return (1e-3 - 1e-5) * mul + 1e-5
        # return (6e-4 - 6e-6) * mul + 6e-6 # 2D
        # return (3e-4 - 3e-6) * mul + 3e-6
        return (1e-4 - 4e-4) * mul + 4e-4
        # return (1e-4 - 1e-5) * mul + 1e-5
        # return (1e-4 - 1e-5) * mul + 1e-5
        # return (1e-4 - 5e-5) * mul + 5e-5
        # return (1e-4 - 1e-6) * mul + 1e-6
        # return (6e-5 - 6e-7) * mul + 6e-7
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

class TripletDataset(Dataset):
    def __init__(self, data, params):
        """
        Args:
            data (numpy.ndarray): The triplet data of shape (num_triplets, 4, 4, 128, 128, 128).
            params (numpy.ndarray): The parameters of shape (num_triplets, 3).
        """
        self.data = data
        self.params = params

    def __len__(self):
        # Return the number of triplets
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches the data and parameters for a given index.
        
        Args:
            idx (int): The index of the sample to retrieve.
            
        Returns:
            tuple: (data_sample, params_sample)
                data_sample (torch.Tensor): The triplet data.
                params_sample (torch.Tensor): The corresponding parameters.
        """
        data_sample = torch.tensor(self.data[idx], dtype=torch.float32)
        params_sample = torch.tensor(self.params[idx], dtype=torch.float32)
        return data_sample, params_sample

def train(model, dataset, factor, model_name, mode, local_rank):
    # if local_rank == 0:
    #     writer = SummaryWriter('train')
    #     writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0
    early_stopper = EarlyStopper(patience=50, min_delta=0.01) # 30
    batch_size_inf = 1 # 1 is corrent, otherwise reorder?

    num_workers = 1 # 8
    if mode == "train":
        # sampler = DistributedSampler(data_train)
        # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
        # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
        if hyper:
            data_train_tuple, data_val_tuple = load_data(dataset, factor, mode)
            data_train = data_train_tuple[0]  # Triplet data
            params_train = data_train_tuple[1]  # Parameters
            # print("len data_train:", len(data_train))
            dataset_train = TripletDataset(data_train, params_train)
            train_data = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            data_train, data_val = load_data(dataset, factor, mode)
            train_data = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        args.step_per_epoch = train_data.__len__()
        # input("train loaded")
        print(len(data_train))
        print(data_train[0].shape)
        if hyper:
            data_val = data_val_tuple[0]  # Triplet data
            params_val = data_val_tuple[1]  # Parameters
            dataset_val = TripletDataset(data_val, params_val)
            val_data = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            val_data = DataLoader(data_val, batch_size=args.batch_size, pin_memory=True, num_workers=num_workers)
        print("len:", len(val_data))
        # input("x")
        # print(len(train_data.dataset))
        # for i, data in enumerate(train_data):
        #     print(i)
    else:
        shuffle = False
        if hyper:
            data_test_tuple, data_test_non_overlap = load_data(dataset, factor, mode)
            data_test = data_test_tuple[0]  # Triplet data
            params_test = data_test_tuple[1]  # Parameters
            print("len:", len(data_test))
            dataset_test = TripletDataset(data_test, params_test)
            data_test = DataLoader(dataset_test, batch_size=batch_size_inf, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            data_test, data_test_non_overlap = load_data(dataset, factor, mode)
            data_test = DataLoader(data_test, batch_size=batch_size_inf, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
        print("len:", len(data_test))


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
        # print("Loaded LREST model.")
        # print(model) # LREST is not a neural net
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
            data_iterator = enumerate(train_data)
            # for i, data in enumerate(train_data):
            # for i, (data_batch, params_batch) in enumerate(train_data):
            for i, data in data_iterator:
                if hyper == False:
                    data_batch = data
                else:
                    data_batch, params_batch = data

                data_time_interval = time.time() - time_stamp
                time_stamp = time.time()
                # data = torch.zeros([16, 9, 150, 450], dtype=torch.float)
                if hyper:
                    data_gpu = data_batch.to(device, non_blocking=True)
                    params_gpu = params_batch.to(device, non_blocking=True)
                else:
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

                # print("data_gpu:", data_gpu.shape)

                # random triplets:
                img0 = data_gpu[:, :1]
                gt = data_gpu[:, 1:2]
                img1 = data_gpu[:, 2:3]
                t_mask = data_gpu[:, 3:4]
                imgs = torch.cat((img0, img1), dim=1)

                if hyper:
                    # params = data_gpu[:, 0, 4, :3, 0, 0]  # hypernet
                    params = params_gpu  # hypernet
                # print("Shape of params tensor:", params.shape)
                # params_ = params.detach().cpu().numpy()
                # print("params_", params_)
                # input("x")
                
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
                if hyper == False:
                    pred, info = model.update(imgs, gt, t_mask, dataset, learning_rate, training=True)
                else:
                    # with autocast():
                    pred, info = model.update(imgs, gt, t_mask, params, dataset, learning_rate, training=True) # hyper
                
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
        # for i, data in enumerate(data_test):
        # for i, (data_batch, params_batch) in enumerate(data_test):
        data_iterator = enumerate(data_test)
        for i, data in data_iterator:
            if hyper == False:
                data_gpu = data
                # print("data:", data.shape)
            else:
                data_batch, params_batch = data
                data_gpu = data_batch.to(device, non_blocking=True)
                params_gpu = params_batch.to(device, non_blocking=True)
                
            # data_gpu = data_gpu.permute(0, 1, 2, 5, 3, 4) # 3rd is dim0
            # data_gpu = data_gpu.float() # to float32
            # print("data_gpu:", data_gpu.shape)

            imgs = data_gpu[:, :2, 0]
            # gt = data_gpu[:, 2:3, 0]
            # print("imgs", imgs.shape)
            # print("params_gpu", params_gpu.shape)
            # print(gt.shape)
            # input("imgs gt")
            if hyper:
                # params = data_gpu[:, 0, 4, :3, 0, 0]  # hypernet
                params = params_gpu  # hypernet
            # print("Shape of params tensor:", params.shape)
            # params_ = params.detach().cpu().numpy()
            # print("params_", params_)

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
                    if hyper == False:
                        merged, flow, mask = model.inference(I0, I1, t_mask)
                    else:
                        merged, flow, mask = model.inference(I0, I1, t_mask, params) # hyper
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
        if factor > 100: # 100
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
            input("reordered")

        flow_combined = np.array(flow_combined)
        print("flow pred is in range %f to %f" % (np.min(flow_combined), np.max(flow_combined)))
        # input("flow")

        # reorder based on factor
        if factor > 100:
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
    data_iterator = enumerate(val_data)
    # for i, data in enumerate(val_data):
    # for i, (data_batch, params_batch) in enumerate(val_data):
    for i, data in data_iterator:
        if hyper == False:
            data_batch = data
            data_gpu = data.to(device, non_blocking=True) # / 255.
        else:
            data_batch, params_batch = data
            data_gpu = data_batch.to(device, non_blocking=True)
            params_gpu = params_batch.to(device, non_blocking=True)
            
        # imgs = data_gpu[:, :2]
        # gt = data_gpu[:, 2:3]

        # data_gpu = data_gpu.float() # to float32

        # print("data_gpu:", data_gpu.shape)

        img0 = data_gpu[:, :1]
        gt = data_gpu[:, 1:2]
        img1 = data_gpu[:, 2:3]
        t_mask = data_gpu[:, 3:4]
        imgs = torch.cat((img0, img1), dim=1)

        if hyper:
            # params = data_gpu[:, 0, 4, :3, 0, 0]  # hypernet
            params = params_gpu
        # print("Shape of params tensor:", params.shape)

        # imgs = imgs.permute(0, 1, 2, 5, 3, 4) # 3rd is dim0
        # gt = gt.permute(0, 1, 2, 5, 3, 4)
        # t_mask = t_mask.permute(0, 1, 2, 5, 3, 4)
        # print("permuted:", imgs.shape, gt.shape)

        with torch.no_grad():
            # pred, info = model.update(imgs, gt, dataset, training=False)
            if hyper == False:
                pred, info = model.update(imgs, gt, t_mask, dataset, training=False)
            else:
                pred, info = model.update(imgs, gt, t_mask, params, dataset, training=False) # hyper
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
    parser.add_argument('--epoch', default=200, type=int) # 120 200
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
        args.batch_size = 8 # 3 4 6 8
    if args.dataset == "pool3d":
        args.batch_size = 2 # 2 4 32
    if args.dataset == "nyx3d":
        args.batch_size = 3 # 1 2 4 32
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

    """ jets """
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
    ### +++
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_triplets_W60_1000_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 55.55 0.0034; 3x: 52.98 0.0034; 5x: 46.73 .00341; 8x: 47.09 0.0034; 16x: PSNR: 44.50 0.0034; 
    # 3d metrics:
    # 2x: 46.50 0.0423; 3x: 46.59 0.0423; 5x: 46.09 0.0424;
    ### +++
    model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip5_triplets_W12_200_gamma_eqRes_deconv_lr6e-4.pkl"

    model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_128_jets_skip10_triplets_W12_200_gamma_eqRes_deconv_lr6e-4.pkl"

    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_128_jets_skip10_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 10x: 49.40 0.09932
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_128_jets_skip20_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 20x: 46.37 0.0941
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-2_v1_128_jets_skip20_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 20x: 38.97 0.0969
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_128_jets_skip20_triplets_W6_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # whole simulation
    # 2x: 41.83 0.466 10x: 30.74 0.471
    # till 1000:
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_128_jets_skip10_triplets_W6_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 27.92 0.424 10x: 30.36 0.422
    # continue training on 200 diff triplets with W5: (lr1e-4)
    # 2x: 15 0.4; 10x: 30.56 0.4442

    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow5e-2_v1_128_jets_skip10_triplets_W6_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 29.89 0.0818

    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_jets_skip10_triplets_W6_200_gamma_eqRes_deconv_lr1e-4.pkl"

    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverge
    # 128 96 64, check 256 - 128 !
    ### +++
    model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 45.67 0.08512; 10x: 46.72 0.0853
    # 64 model, 128 in teacher block
    # 100-800 (200): 43.08 0.7346
    ### +++
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow3e-1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 700-800: 2x: 36.15 0.3112 10x: 34.12 0.3112
    # 64 might be best
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_64_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 64 model, 128 in teacher block
    # 700-800: 2x: 36.00 0.0841; 10x: 33.26 0.0843;

    # 4 blocks: 256 192 192 128 
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 32.76 0.0877; 10x: 33.96 0.0869

    ### +++
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 40.45 0.7363; 10x: 46.05 0.7364; 15x: 45.41 0.7365; 20x: 39.36 0.7366
    ### +++ flow to 3 blocks...
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_1K_gamma_eqRes_deconv_lr1e-4.pkl"
    # 10x: PSNR: 40.77 0.7311

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr3e-4.pkl"
    # Early stop.
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma06_eqRes_deconv_lr1e-4.pkl"
    # 10x: 42.12 0.7361 20x: 45.33 0.7345

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma07_eqRes_deconv_lr1e-4.pkl"
    # 10x: 46.08 0.7384; 15x: avg PSNR: 45.04 0.7385; 20x: 42.65 0.7386
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma09_eqRes_deconv_lr1e-4.pkl"
    # 2x: 49.32 0.7326; 10x: 37.84 0.7327; 15x: 33.73 0.7334; 20x: 35.93 0.7335
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_lapl_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 34.89 0.7382; 10x: 35.82 0.7387 20x: 38.92 0.7392

    ### +++
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W6_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 46.39 0.732; 10x: 41.76 0.7326; 15x: 41.11 0.7328; 20x: 47.32 0.7328
    # LREST 20x
    ### +++

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_best.pkl"
    # 
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_flow2.pkl"
    # 10x: 35.44 0.7315 15x: 44.41 0.7311 20x: 42.07 0.7305

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 34.42 0.7337; 10x: 37.55 0.7336; 20x: 39.89 0.7338
    # model_name = "lrest_5Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 31.59 0.742; 10x: 33.70 0.742 20x: 41.50 0.7417

    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W15_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 34.35 0.08498; 10x: 29.86 0.0852
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W10_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 33.35 0.0877; 10x: 33.31 0.0876
    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 10x: 43.30 0.082
    # 100-200: 15x: linear: 30.05 lrest: 31.96
    # 700-800: 20x: linear: 39.39 LREST: 45.15
    # 100-800 (300): 2x: 41.85 0.736; 10x: 41.83 0.7358 linear 49; 15x: linear 43.974 42.946 0.7351; 20x: linear: 38.393 41.9 0.7359
    ###

    # # ablations +
    # no flow:
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 34.98 3.649; 10x: 37.80; 3.4985
    # no rec:
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 10x: 32.32 0.7332
    # w/o s-t:
    # model_name = "flownet_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_jets_skip10_1000_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 10x: 41.99 0.7323


    """ pool3d """
    # untrained: 2x: 35.01 0.506
    ### +++
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 46.21 0.126; 3x: 44.07 0.126
    # 128 - 2x: 34.77 0.167; 3x: 33.24 0.167
    ### +++
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_4K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 42.17 0.108
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W12_4K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 46.09 0.120
    # W8
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_triplets_W8_2K_gamma_eqRes_deconv_lr6e-4.pkl"
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
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_float16amp_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverge -
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16amp_triplets_W12_1K_gamma_eqRes_deconv_lr1e-5.pkl"
    # diverge -
    # float 16 withput amp
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool_float16_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 39.64 0.136 lr 1e-5
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_500_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 32.36 0.209
    # PSNR: 32.359 32.365 # float16, convert to 32 vs float32
    # LPIPS: 0.019055
    # EPE: 0.20927 0.20929
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr1e-5.pkl"
    # 2x: 33.86 0.207 30ep
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # diverge - # 2x: 33.75 3x: 0.216
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_float16_triplets_W12_2K_gamma_eqRes_deconv_lr1e-5.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool64to128_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_pool128_triplets_W12_500_gamma_eqRes_deconv_lr6e-5.pkl"
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

    """ nyx3d """
    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_triplets_W12_500_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 82.49 0.0133; 3x: 81.83 0.0133; 8x: 76.36 0.0133
    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_triplets_W12_1K_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 83.23 0.0998; 3x: 83.07 0.0997; 8x: 80.97 0.0998
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_triplets_W12_500_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 79.53 0.0139; 8x: 77.19 0.0139
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_triplets_W12_500_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 83.74 0.0126; 5x: 79.718 0.0126; 8x: 75.96 0.0126; 16x: 72.48 0.0126
    # model_name = "flownet_untrained.pkl"
    # 2x: 81.585 0.1769; 5x: 78.99 0.17692; 16x: 72.396 0.17696
    # 2x: 73.18 0.19652
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_triplets_skip5_W12_500_gamma_eqRes_deconv_lr6e-4.pkl"
    # 2x: 66.57 0.17978
    # model_name = "flownet_4Blocks_3convDown3convUp3conv_l1_flow2e-2_v1_64_nyx_triplets_skip5_W12_500_gamma_eqRes_deconv_lr6e-4.pkl"
    # 
    # 64 model, 128 in teacher block
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 54.21 0.1441; 3x: 53.30 0.1447; 8x: 45.56 0.1446
    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 54.88 0.1439; 3x: 53.70 0.1437; 5x: 49.80 0.1438; 8x: 45.77 0.1437; 16x: 39.15 0.1438
    ###
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 54.71 0.1421; 8x: 45.92 0.1417
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr6e-4.pkl"
    # explode
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr3e-4.pkl"
    # 2x: 52.88 0.1359; 3x: 52.58 0.1357; 5x: 49.52 0.1353; 8x: 45.61 0.1352; 16x: 38.90 0.1350
    # model_name = "flownet_3Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr3e-4.pkl"
    # 2x: 54.15 0.1343; 3x: 52.98 0.1339; 8x: 45.83 0.1334
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr3e-4.pkl"
    # explode 2x: 54.14 0.1429
    ###
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 54.80 0.1408; 5x: 50.48 0.1405; 8x: 46.52 0.1404; 16x: 39.60 0.1402
    ###
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4.pkl"
    # 2x: 54.42 0.1408; 5x: 50.16 0.1405; 8x: 46.20 0.1404
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 54.70 0.1389; 5x: 49.72 0.1387; 8x: 45.56 0.1386

    ### +++
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_3b_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # num_blocks = 3 flow
    # 2x: 57.13 0.0979; 3x: 55.13 0.0985; 5x: 51.30 0.0986; 8x: 47.02 0.0988; 16x: 39.32 0.0990
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_2b_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # num_blocks = 2 flow
    # 2x: 56.73 0.0935; 3x: 54.93 0.0933; 5x: 51.50 0.0927; 8x: 47.13 0.0925; 16x: 39.69 0.0921
    model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_3b_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 58.02 0.0912; 3x: 56.74 0.0917; 5x: 53.60 0.0923; 8x: 49.73 0.0928; 16x: 42.60 0.0935
    ### +++
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_3b_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 60.47 0.1241
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 51.29 0.0290 8x: 48.98 0.0284

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow5e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 53.29 0.0306; 3x: 53.68 0.0307; 5x: 52.44 0.0305; 8x: 48.57 0.0304; 16x: 41.63 0.0302
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow3e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 52.75 0.0312; 8x: 49.15 0.0311; 16x: 42.46 0.0310

    ### +++
    model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 52.62 0.0310; 5x: 52.31 0.0310; 8x: 49.39 0.0309; 16x: 42.88 0.0309
    ### +++
    model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep_v1.pkl"
    # 2x: 51.82 0.0293; 5x: 51.19 0.0291; 8x: 49.07 0.0290; 16x: 42.53 0.0289
    model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep_v2.pkl"
    # 2x: 50.66 0.0284; 5x: 50.50 0.0279; 8x: 50.05 0.0300; 16x: 42.07 0.0275
    # noTea:
    ### +++
    model_name = "lrest_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 54.01 0.0313; 5x: 53.39 0.0312; 8x: 50.03 0.0312; 16x: 43.35 0.0312
    ### +++
    model_name = "lrest_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 52.92 0.0319; 5x: 53.55 0.0312; 8x: 49.44 0.0320; 16x: 42.97 0.0321
    model_name = "lrest_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_1K_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 53.30 0.0305; 5x: 53.34 0.0305; 8x: 50.25 0.0305; 16x: 43.63 0.0304
    # a bit better, use only for final experiments

    # lrest train as hyperLREST: 
    model_name = "lrest_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 49.86 0.0294; 5x: 51.07 0.0294; OmM_0155_OmB_00235_h_055
    # 2x: 50.52 0.0212; 5x: 52.93 0.0210; OmM_012_OmB_00215_h_055
    model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 48.55 0.0260; 5x: 50.02 0.0259; OmM_0155_OmB_00235_h_055
    # 2x: 49.33 0.0191; 5x: 52.81 0.0189; OmM_012_OmB_00215_h_055
    model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4_4e-4_200ep.pkl"
    # 2x: 50.06 0.0253; 5x: 50.16 0.0251; OmM_0155_OmB_00235_h_055
    # 2x: 52.48 0.0177; 5x: 52.13 0.0176; OmM_012_OmB_00215_h_055
    model_name = "lrest_5Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4_4e-4_200ep.pkl"

    model_name = "lrest_6Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4_4e-4_200ep.pkl"


    # hyper best:
    # 2x: 51.79 0.0264; 5x: 50.64 0.0262; OmM_0155_OmB_00235_h_055
    # 2x: 51.91 0.0166; 5x: 52.22 0.0165; OmM_012_OmB_00215_h_055 

    # model_name = "lrest_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 52.11 0.0312; 5x: 52.25 0.0311; 8x: 49.43 0.0310; 16x: 42.90 0.0309
    # model_name = "lrest_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 52.71 0.0334; 5x: 52.39 0.0334; 8x: 49.28 0.0333; 16x: 42.84 0.0331

    # Tea:
    # model_name = "lrest_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep_v1.pkl"
    # 2x: 51.83 0.0308; 5x: 51.04 0.0309; 8x: 48.70 0.0309; 16x: 42.50 0.0310

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_3b_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep_v1.pkl"
    # 2x: 60.48 0.1239; 5x: 55.11 0.1215; 8x: 50.90 0.1211; 

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma09_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # 2x: 51.29 0.0307; 8x: 49.26 0.0306;

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow1e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 50.29 0.0318; 8x: 49.68 0.0319;
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow1e-1_3b_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 58.35 0.1182; 8x: 50.12 0.1143;

    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_1b_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 55.71 0.0832; 8x: 45.95 0.0817
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_flow2e-1_3b_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-6.pkl"
    # 2x: 56.76 0.0969; 8x: 46.87 0.0972


    # model_name = "lrest_5Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 55.29 0.1365; 8x: 45.69 0.1354
    # model_name = "lrest_5Blocks_3convDown3convUp3conv_l1_flow2e-1_4b_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 56.76 0.1020; 8x: 46.19 0.0982

    # best: 2x: 52.62 0.0310; 5x: 52.31 0.0310; 8x: 49.39 0.0309; 16x: 42.88 0.0309

    # # ablations
    # no flow:
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_noflow_v1_128_nyx_skip5_log10norm_triplets_W12_500_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 64.95 0.1582; 8x: 51.83 0.1586
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_l1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 60.95 0.1455; 3x: 56.67 0.1458; 5x: 53.17 0.1461; 8x: 48.14 0.1465; 16x: 39.54 0.1475
    # no rec:
    # model_name = "lrest_4Blocks_3convDown3convUp3conv_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 47.15 0.1454
    # w/o s-t:
    # model_name = "lrest_noTea_4Blocks_3convDown3convUp3conv_l1_flow2e-1_3b_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # 2x: 59.44 0.1389; 8x: 47.37 0.1391
    # model_name = "lrest_noTea_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep_v1.pkl"
    # 2x: 51.51 0.0299; 5x: 53.38 0.0300; 8x: 50.05 0.0300

    # TODO:
    # try: replace PReLU with sine (SIREN)

    # HYPER:

    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_200ep.pkl"
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5.pkl"
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-5_1e-7.pkl"
    # 2x: 19
    # change arch - only some blocks hypernet
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-5_1e-7_1.pkl"
    # 2x: 44.38
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest.pkl"
    # 2x: 44.04
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest_freeze1.pkl"
    # 2x: 53.08 0.0317 - nothing frozen
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest_freeze2.pkl"
    # None grads
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest_removedNograd.pkl"
    # 2x: 53.08 0.0319; 5x: 48.39 0.0318
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_128_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest_removedNograd.pkl"
    # OOM
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest_removedNograd.pkl"
    # 2x: 53.83 0.0285; 5x: 48.29 0.0283
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN.pkl"
    # ???
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtest_removedNograd1.pkl"
    # 2x: 52.69  0.0342 - increased MLP


    ### +++
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D.pkl"
    # 2x: 52.95 0.0211; 5x: 52.33 0.0211;
    ### +++
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_1.pkl"
    # 2x: 52.91 0.0219; 5x: 52.35 0.0219;

    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b6.pkl"
    # 2x: 53.74 0.0469; 5x: 48.25 0.0467;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b4.pkl"
    # 2x: 53.82 0.0349;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_c16_b4_1.pkl"
    # 2x: 52.51 0.0353; 5x: 51.71 0.0355;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_c16_b4_val_test.pkl"
    # still 
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_lr1e-4_1e-5_archtestCNN1D_c16_b4.pkl"
    # 2x: 53.69 0.0332; 5x: 53.02 0.0334; 500 better than 200 triplets

    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b1_reproduce.pkl"
    # 2x: 51.88 0.0218; 5x: 48.12 0.0218; yes!! val same as train, 50 triplets, batch_size = 1 (look at lr plot)
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b1_reproduce.pkl"
    # 2x: 50.22 0.0215; 5x: 47.68 0.0214;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b1_trainval.pkl"
    # 2x: 51.34 0.0281; 5x: 48.19 0.0281;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_conv01hyper_trainval.pkl"
    # 2x: 52.40 0.0309; 5x: 47.65 0.0306;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_prev+012convhyper_trainval.pkl"
    # 2x: 53.41 0.0314; 5x: 51.69 0.0312;
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_conv01convblock012hyper_trainval.pkl"
    # 2x: 53.94 0.0274; 5x: 51.88 0.0278;
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_lr1e-4_1e-5_archtestCNN1D_b6_conv01convblock012hyper_trainval.pkl"
    # 2x: 54.15 0.0357; 5x: 51.05 0.0375;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_lr1e-3_1e-4_archtestCNN1D_b6_conv01convblock012hyper_trainval.pkl"
    # 2x: 52.67 0.0326; loss jump
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_lr1e-4_1e-6_archtestCNN1D_amp_b10_conv01convblock012hyper_trainval.pkl"
    # 2x: 53.71 0.0369; 5x: 48.38 0.0367;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_lr1e-4_5e-4_archtestCNN1D_amp_b10_conv01convblock012hyper_trainval.pkl"

    # experiemnt - train on one member with furtherst params from test
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_conv01convblock012hyper.pkl"
    # 2x: 52.73 0.0278; 5x: 51.41 0.0284;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 51.55 0.0277; 5x: 52.02 0.0278;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_end2m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 53.81 0.0317; 5x: 52.68 0.0318;
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_5e-4_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 52.18 0.0259; 5x: 51.90 0.0259; lr1e-4_5e-4 actually
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_end2m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_5e-5_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 53.90 0.0293; 5x: 52.86 0.0294; lr1e-4_5e-4 actually
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_end2m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_5e-5_archtestCNN1D_b4_conv01convblock012hyper_1.pkl"
    # 2x: 53.61 0.0315; 5x: 52.74 0.0316; 
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_end2m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_5e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 53.85 0.0602; 5x: 54.15 0.0602; overfit?
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_5e-4_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 52.64 0.0252; 5x: 52.64 0.0251;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-3_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # explode -
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_3e-4_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 52.18 0.0261; 5x: 52.50 0.0262;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_5e-4_AdamW_1e-3_1e-2_archtestCNN1D_b4_conv01convblock012hyper.pkl"
    # 2x: 51.92 0.0255; 5x: 52.57 0.0255;
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_5e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.47 0.0238; 5x: 52.70 0.0237;
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_oneMember_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_5e-5_step4000_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.63 0.0254; 5x: 52.31 0.0253;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_5e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 53.50 0.0263; 5x: 51.53 0.0269;
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_5e-4_archtestCNN1D_b1_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.87 0.0132; 5x: 51.81 0.0134;
    # 2x: 51.34 0.0197; 5x: 51.19 0.0196; OmM_012_OmB_00215_h_055
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_5e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.96 0.0157; 5x: 53.27 0.0156; explode -
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_3e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 54.22 0.0143; 5x: 53.26 0.0142;
    # 2x: 50.43 0.0265; 5x: 51.10 0.0264; OmM_0155_OmB_00235_h_055
    # 2x: 52.89 0.0189; 5x: 52.11 0.0188; OmM_012_OmB_00215_h_055
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.06 0.0210; 5x: 52.00 0.0208; OmM_012_OmB_00215_h_055
    ### 
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 50.29 0.0274; 5x: 50.76 0.0271; OmM_0155_OmB_00235_h_055
    # 2x: 51.27 0.0177; 5x: 52.21 0.0179; OmM_012_OmB_00215_h_055
    ###
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock012hyper_300ep.pkl"
    # 2x: 51.52 0.0255; 5x: 49.98 0.0254; OmM_0155_OmB_00235_h_055
    # 2x: 52.53 0.0181; 5x: 52.02 0.0180; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock01hyper_200ep.pkl"
    # 2x: 50.98 0.0196; 5x: 52.36 0.0194; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_2Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 50.60 0.0179; 5x: 51.42 0.0178; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_4Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 50.96 0.0173; 5x: 51.52 0.0171; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_32_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.51 0.0206; 5x: 51.81 0.0204; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_80_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.53 0.0185; 5x: 52.66 0.0184; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_l2_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.47 0.0174; 5x: 52.59 0.0172; OmM_012_OmB_00215_h_055
    ### +++
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_l3_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.43 0.0266; 5x: 50.87 0.0265; OmM_0155_OmB_00235_h_055
    # 2x: 51.91 0.0166; 5x: 52.22 0.0165; OmM_012_OmB_00215_h_055 
    # b_s = 3, mlp32, 80
    ### +++
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_mlp16_CNN1D_l3_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.35 0.0173; 5x: 52.76 0.0172; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_mlp32_cnn1D_l3_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 50.06 0.0260; 5x: 50.58 0.0259; OmM_0155_OmB_00235_h_055
    # 2x: 50.34 0.0188; 5x: 51.78 0.0186; OmM_012_OmB_00215_h_055
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_mlp32_cnn1D_l3_b3_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.79 0.0264; 5x: 50.64 0.0262; OmM_0155_OmB_00235_h_055
    # 2x: 52.53 0.0178; 5x: 52.09 0.0177; OmM_012_OmB_00215_h_055



    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN3D_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.20 0.0190; 5x: 52.64 0.0190;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN3D_4layers_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 50.71 0.0177; 5x: 51.56 0.0177;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_SELU_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.24 0.0216; 5x: 51.71 0.0215;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_Sine_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 50.86 0.0729; -
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_SineInit_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.79 0.0201; 5x: 52.04 0.0200;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_PReLUinit_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 51.42 0.0212 ; 5x: 51.54 0.0211; exploded
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_l6_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 53.31 0.0180; 5x: 52.31 0.0178;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_OmM_01375_OmB_00225_h_055_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_4e-4_archtestCNN1D_l8_b4_conv01convblock012hyper_200ep.pkl"
    # 2x: 52.61 0.0185; 5x: 52.53 0.0184;

    
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_conv01convblock012deconvhyper_trainval.pkl"
    # 2x: 53.10 0.0322; 5x: 51.81 0.0324;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_conv01convblock0123456hyper_trainval.pkl"
    # 2x: 48.18 0.1517;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_conv01convblock012_456hyper_trainval.pkl"
    # 2x: 52.44 0.0386;
    # hyper layers in the end:
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_lr1e-4_1e-5_archtestCNN1D_b1_convblock789deconvhyper_trainval.pkl"
    # 2x: 51.86 0.0474


    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_50_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b1_train.pkl"
    # 2x: 53.75 0.0351; 5x: 52.04 0.0357;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_b4_train.pkl"
    # 2x: 54.51 0.0354; 5x: 52.65 0.0356;
    # 200 and 50 triplets, same results


    # ensemble with param tuple
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D.pkl"
    # 2x: 53.49 0.0366; 5x: 52.19 0.0368;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_lr1e-4_1e-5_archtestCNN1D.pkl"
    # 2x: 53.80 0.0331; 5x: 52.64 0.0338;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b2.pkl"
    # 2x: 54.51 0.0337; 5x: 53.25 0.0337; 500
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b4.pkl"

    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_noamp_b4.pkl"
    # 2x: 52.19 0.0355
    # why EPE is not 0.02 ??? what is different???
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_noamp_b4_1.pkl"
    # 2x: 51.96 0.0349
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b4_20eprestart.pkl" 
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_b1.pkl"
    # 2x: 53.02 0.0336
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_noamp_b4_2.pkl"
    # 2x: 52.05 0.0352
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_noamp_b4_val1.pkl"
    # 2x: 51.97 0.0372;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_c16_b4.pkl"
    # 2x: 52.86 0.0392; 5x: 52.54 0.0397; 8x: 48.91 0.0395
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-4_1e-5_archtestCNN1D_amp_c16_b4.pkl"
    # 2x: 54.28 0.0382; 5x: 52.21 0.0401; 8x: 48.67 0.0404
    # similar results amp and noamp
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr1e-5_1e-7_archtestCNN1D_c16_b4.pkl"
    # 2x: 54.83 0.0536;
    # loss scaler was used
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr3e-4_3e-6_archtestCNN1D_c16_b4.pkl"
    # 2x: 51.94 0.0328; 5x: 52.02 0.0330
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_200_gamma_lr3e-4_3e-6_archtestCNN1D_c16_b4_1.pkl"

    # amp
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_amp1.pkl"
    # 2x: 53.92 0.0375; 5x: 48.26 0.0374;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_amp2.pkl"
    # 2x: 52.48 0.0266; 5x: 52.14 0.0265;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_2.pkl"
    # no amp:
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_noamp.pkl"
    # 2x: 52.85 0.0266; 5x: 52.16 0.0266;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_3.pkl"
    # 2x: 53.37 0.0262; 5x: 52.01 0.0265;
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_4.pkl"
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_5.pkl"
    # 2x: 52.25 0.0263; 5x: 52.05 0.0263
    # model_name = "hyperLREST_noTea_3Blocks_3convDown3convUp3conv_l1_flow2e-1_v1_64_nyx_ens5m_skip5_log10norm_triplets_W12_500_gamma_eqRes_deconv_lr1e-4_1e-5_archtestCNN1D_6.pkl"
    # 2x: 54.40 0.0371; 5x: 48.30 0.0368


    # untrained: 2x: 52.68 0.1707; 8x: 41.26 0.1710   
    # hyper, untrained: 2x: 39.66 0.3174
    # baseline: skip5 - 2x: 53.19; 3x: 47.49; 5x: 41.92; 8x: 37.51; 16x: 31.90 32x: 27.68
    # 200 and 50 triplets, same results


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

        if "3d" in dataset: 
            # factors = [2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64]
            factors = [2, 3, 5, 8, 16, 32]
            # factors = [2, 10, 15, 20] # 30, 40, 60]
            # factors = [20]
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

            # print("Density range %f to %f" % (np.min(data_test[:, 0]), np.max(data_test[:, 0])))
            # print("Vel x range %f to %f" % (np.min(data_test[:, 1]), np.max(data_test[:, 1])))
            
            if "3d" in dataset: 
                if hyper == False:
                    flow_test = data_test[:, 1:] # flow
                else:
                    flow_test = data_test[:, 1:4] # flow
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

            flow = flow[:, 3:6, ...]
            # flow = np.transpose(flow, (0, 1, 2, 4, 3)) # 3rd is dim0

            if not os.path.isdir(dir_model):
                os.makedirs(dir_model)

            """
            # save data for 3d figure:
            # data = [data_original_interpol, interpol_data, flow_original_pred, flow]
            data = [interpol_data]
            save_path = "Results"
            save_path = os.path.join(save_path, dataset)
            save_path = os.path.join(save_path, "paper_figure")
            # filenames = ["data_original_interpol.pkl", "interpol_data.pkl", "flow_original_pred.pkl", "flow.pkl"]
            filenames = ["data_coordnet_interpol.pkl"]
            for d, fn in zip(data, filenames):
                with open(os.path.join(save_path, fn), 'wb') as f:
                    pickle.dump(d, f, protocol=4)
            input("x")
            """

            # 3D rendering function
            # visualize_seq_dens_3d(data_original_interpol, interpol_data, flow_original_pred, flow,
            #             dataset, dir_res=dir_model, title="Dens_GT_Interpol", show=False, save=True)
            # print("visualize_seq_dens_3d done!")

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
            # diffs = calculate_diff(data_original_interpol, interpol_data, dataset, factor, dir_model)
            # print("diffs:", diffs.shape)

            """
            ##### Baseline evaluation
            print("Baseline evaluation!")
            lin_interpol = linear_interpolation(data_original, factor)
            lin_interpol = lin_interpol[:data_original_interpol.shape[0]]
            print(lin_interpol.shape)

            avg_psnr = calculate_avg_psnr(data_original_interpol, lin_interpol, dir_model, factor)
            # avg_lpips = calculate_avg_lpips(data_original_interpol, lin_interpol, dir_model, factor)
            # Nyx: 2x PSNR: 79.892
            # skip5: 2x: 53.19; 3x: 47.49; 5x: 41.92; 8x: 37.51; 16x: 31.90 32x: 27.68

            # jets: PSNR 2x: 77; 5x: 64; 10x: 60.8; 20x: 56.8
            # 100-200: 2x: 51.4; 5x: 40; 10x: 34.27; 15x: 30; 20x: 30.989
            # 700-800: 2x: 70; 5x: 59.28; 10x: 52.98; 15x: 49.94; 20x: 39.39
            # 700-800 (200): 2x: 66.8; 10x 48.7; 15x: 44.28; 20x: 38.11; 30x: 37.09; 40x: 37.27; 60x: 37.40
            # input("baseline")
            print("Baseline evaluation")
            # visualize_seq_dens_3d(data_original_interpol, lin_interpol, flow_original_pred, flow,
            #             dataset, dir_res=dir_model, title="Dens_GT_Linear", show=False, save=True)
            #####
            """

            if "3d" in dataset:
                print("GT flow available")
                # flow_gt = flow_gt[:, :, :flow.shape[2], :flow.shape[3]]
                flow_original_pred = flow_original_pred[:, :, :flow.shape[2], :flow.shape[3], :flow.shape[4]]

                # print("flow original pred is in range %f to %f" % (np.min(flow_original_pred), np.max(flow_original_pred)))
                # print("flow pred is in range %f to %f" % (np.min(flow), np.max(flow)))
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

                # print("u_gt is in range %f to %f" % (np.min(u_gt), np.max(u_gt)))
                # print("u is in range %f to %f" % (np.min(u), np.max(u)))
                # print("norm is in range %f to %f" % (np.min(norm), np.max(norm)))
                # print("norm_gt is in range %f to %f" % (np.min(norm_gt), np.max(norm_gt)))

                print("norm:", norm_gt.shape)
                # print(norm_gt[0, 10, 10])
                # print(norm_gt[0, 100, 100])
                # print(norm_gt[100, 10, 10])
                # print(norm_gt[100, 100, 100])
                # print("10", norm_gt[10])
                # print("100", norm_gt[100])
                # print("200", norm_gt[200])
                # norm = max(norm, norm_gt)

                # diffs_flow = calculate_diff(norm_gt, norm, dataset, factor, dir_model)

                # u_gt = (u_gt - np.min(u_gt)) / (np.max(u_gt) - np.min(u_gt))
                # v_gt = (v_gt - np.min(v_gt)) / (np.max(v_gt) - np.min(v_gt))

                # u_gt = u_gt[:, :u.shape[1], :u.shape[2]]
                # v_gt = v_gt[:, :v.shape[1], :v.shape[2]]
                # u_diff = calculate_diff(u_gt, u, dataset, factor, dir_model)
                # v_diff = calculate_diff(v_gt, v, dataset, factor, dir_model)

                # print(u_gt, u, u_diff)
                # input("x")
                # u_diff = (u_diff - np.min(u_diff)) / (np.max(u_diff) - np.min(u_diff))
                # v_diff = (v_diff - np.min(v_diff)) / (np.max(v_diff) - np.min(v_diff))

            # print("Flow original pred mean and std:", np.mean(flow_original_pred), np.std(flow_original_pred))
            # print("Flow pred mean and std:", np.mean(flow), np.std(flow))

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

            data_original_interpol = data_original_interpol[:interpol_data.shape[0]]
            flow_original_pred = flow_original_pred[:flow.shape[0]]

            # 3D metrics
            avg_psnr = calculate_avg_psnr(data_original_interpol, interpol_data, dir_model, factor)
            if "3d" in dataset:
                # Flow metric
                # print(flow_gt.shape, flow.shape)
                print(flow_original_pred.shape, flow.shape)
                avg_epe = calculate_avg_epe(flow_original_pred, flow, dir_model, factor)

            # # """
            # # fn = "data_original_interpol.pkl"
            # # with open(os.path.join(dir_model, fn), 'wb') as f:
            # #     pickle.dump(data_original_interpol, f, protocol=4)
            # # fn = "interpol_data.pkl"
            # # with open(os.path.join(dir_model, fn), 'wb') as f:
            # #     pickle.dump(interpol_data, f, protocol=4)
            # # print("saved for renderings!")
            # render_density(data_original_interpol, interpol_data, dir_model)
            # make_video(dir_model)
            # print("video created!")
            # # input("xxx")
            # # """

            """
            original_min = 17610006. # 17327866.
            original_max = 54730323656704 # 54857272655872.

            # Scale the data back to the original range
            data_original_interpol = (data_original_interpol * (original_max - original_min)) + original_min
            interpol_data = (interpol_data * (original_max - original_min)) + original_min
            print("Density range %f to %f" % (np.min(data_original_interpol), np.max(data_original_interpol)))

            for i in range(0, 50, 10):
                    fn = "GT_dens_" + str(i) + ".png"
                    render_volume_yt(data_original_interpol[i], dir_model, fn)
                    fn = "interpol_dens_" + str(i) + ".png"
                    render_volume_yt(interpol_data[i], dir_model, fn)
            input("rendered")
            """

            # 2D for VIS
            # TODO check dims
            slice_num = 40
            data_test = data_test[..., slice_num, :, :]
            data_original_interpol = data_original_interpol[..., slice_num, :, :]
            interpol_data = interpol_data[..., slice_num, :, :]
            # diffs = diffs[..., slice_num, :, :]
            flow = flow[..., slice_num, :, :]
            # flow = flow[..., slice_num, :]
            flow_original_pred = flow_original_pred[..., slice_num, :, :]
            # flow_original_pred = flow_original_pred[..., slice_num, :]
            # diffs_flow = diffs_flow[..., slice_num, :, :]
            # u_diff = u_diff[..., slice_num, :, :]
            # v_diff = v_diff[..., slice_num, :, :]
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
            # visualize_large(data_test, interpol_data, diffs,
            #     flow, flow_original_pred, diffs_flow, u_diff, v_diff, mask,
            #     factor, dataset, dir_model, title=title, show=False, save=True)
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
