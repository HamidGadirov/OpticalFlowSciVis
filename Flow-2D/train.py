# ssh -Y hamid@129.125.75.167
# conda activate gpu
# python3 -m torch.distributed.launch --nproc_per_node=1 train.py --world_size=1 --dataset=rectangle2d --mode=train

# rsync -avz -e 'ssh' hamid@129.125.75.167:~/Desktop/OpticalFlow/RIFE/train_log/ /Users/hamidgadirov/Desktop/OpticalFlow/RIFE/train_log/

from pickletools import uint2
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

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from load_datasets import load_data
from error import calculate_diff
from utils import plot_loss, visualize_ind, visualize_series, visualize_series_flow, visualize_large

# early stopping is deactivated
# TODO:
# merge with 3d model -
# convert model to float 16 +
# try mixed precision model -

device = torch.device("cuda")

# exp = os.path.abspath('.').split('/')[-1]
log_path = 'train_log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

early_stop_patience = 1001
early_stop_k = 0
val_loss_best = 0.

def train(model, dataset, exp, model_name, mode, local_rank):
    # if local_rank == 0:
        # writer = SummaryWriter('train')
        # writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0

    if mode == "train":
        if dataset == "vimeo2d": 
            dataset_train = VimeoDataset('train')
            sampler = DistributedSampler(dataset_train)
            train_data = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
            args.step_per_epoch = train_data.__len__()
            dataset_val = VimeoDataset('validation')
            val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
        else:
            data_train, data_val = load_data(dataset, exp, mode)
            # print(data_train.shape)
            # sampler = DistributedSampler(data_train)
            # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
            train_data = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
            args.step_per_epoch = train_data.__len__()
            # # input("train loaded...")
            print("len:", len(data_train))
            print(data_train[0].shape)
            # input("train loaded")
            val_data = DataLoader(data_val, args.batch_size, pin_memory=True, num_workers=8)
            print("len:", len(val_data))
    else:
        if dataset == "vimeo2d": 
            dataset_test = VimeoDataset('test')
            data_test = DataLoader(dataset_test, batch_size=16, pin_memory=True, num_workers=8)
        else:
            data_test = load_data(dataset, exp, mode)
            data_test = DataLoader(data_test, batch_size=16, pin_memory=True, num_workers=8)

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
        model.load_model(model_name, log_path) # won't work - we changed loss and pyramid
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
                if exp == 1: # # img0, img1, gt
                    imgs = data_gpu[:, :2] # :6
                    gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network
                    # print(imgs.shape)
                    # print(gt.shape)
                    # input("x")
                    pred, info = model.update(imgs, gt, dataset, learning_rate, training=True)
                elif exp == 2: # img0, img1, gt0, gt1, gt2
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
            evaluate(model, dataset, val_data, step, local_rank) #, writer_val)   
            # model.save_model(model_name, log_path, local_rank) 
            dist.barrier()
    else:
        print("inference...")
        factor = 2
        dir_res = "Results"
        dir_res = os.path.join(dir_res, dataset)
        dir_res = os.path.join(dir_res, str(factor) + "x")
        dir_model = os.path.join(dir_res, model_name[:-4])
        # print(dir_model)
        # input("x")
        print("Saving at:", dir_model)
        # data = data_train
        # scale_list = [1, 1, 1]
        scale_list = [4, 2, 1]
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
        data_test_combined = [] # vimeo

        for i, data in enumerate(data_test):
            # print("test len:", len(data_gpu))
            # print("data:", data.shape)
            # input("x")
            # data = np.expand_dims(data, axis=0)
            if dataset == "vimeo2d":
                # print("vimeo2d")
                data_gpu, timestep = data
                for b in range(data_gpu.shape[0]):
                    data_test_combined.append(np.asarray(data_gpu[b, 0] / 255.))
                    data_test_combined.append(np.asarray(data_gpu[b, 2] / 255.))
                    data_test_combined.append(np.asarray(data_gpu[b, 1] / 255.))
                data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            elif dataset == "droplet2d":
                # print("droplet2d")
                data_gpu = data
                for b in range(data_gpu.shape[0]):
                    data_test_combined.append(np.asarray(data_gpu[b, 0] ))
                    data_test_combined.append(np.asarray(data_gpu[b, 2] ))
                    data_test_combined.append(np.asarray(data_gpu[b, 1] ))
                data_gpu = data_gpu.to(device, non_blocking=True) 
            else:
                data_gpu = data
                for b in range(data_gpu.shape[0]):
                    data_test_combined.append(np.asarray(data_gpu[b, 0, 0]))
                    data_test_combined.append(np.asarray(data_gpu[b, 2, 0]))
                    data_test_combined.append(np.asarray(data_gpu[b, 1, 0]))
                data_gpu = data_gpu.to(device, non_blocking=True)
            # print("data to gpu:", data_gpu.shape)
            # input("data")
            imgs = data_gpu[:, :2]
            gt = data_gpu[:, 2:3]

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
            if exp == 1: # # img0, img1, gt
                # middle, flow, mask = model.inference(I0, I1, scale_list) # get the flow too
                merged, flow, mask = model.inference(I0, I1, scale_list) # get the flow too
                left = np.asarray(merged[0].detach().cpu().numpy())
                right = np.asarray(merged[1].detach().cpu().numpy())
                middle = np.asarray(merged[2].detach().cpu().numpy())
                # print(middle.shape)
                # input("x")
                # interpol_data.extend(middle.transpose((0,2,3,1))) # append
                for b in range(data_gpu.shape[0]):
                    interpol_data.append(left[b].transpose((1,2,0)))
                    interpol_data.append(middle[b].transpose((1,2,0)))
                    interpol_data.append(right[b].transpose((1,2,0)))
                # print(middle.shape)
                # input("x")
                flow_l = np.asarray(flow[0].detach().cpu().numpy())
                flow_r = np.asarray(flow[1].detach().cpu().numpy())
                flow_m = np.asarray(flow[2].detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    flow_combined.append(flow_l[b])
                    flow_combined.append(flow_m[b])
                    flow_combined.append(flow_r[b])
                # print(flow_l.shape)
                # input("x")
                # flow_array_ = flow[2].detach().cpu().numpy()
                # flow_combined.extend(flow_l[:,0:4,:,:].squeeze())
                # flow_combined.extend(flow_m[:,0:4,:,:].squeeze())
                # flow_combined.extend(flow_r[:,0:4,:,:].squeeze()) # 0:4 because in x and y for Ft->0 and Ft->1 intermediate flows
                # print(flow.shape)
                mask_l = np.asarray(mask[0].detach().cpu().numpy())
                mask_r = np.asarray(mask[1].detach().cpu().numpy())
                mask_m = np.asarray(mask[2].detach().cpu().numpy())
                for b in range(data_gpu.shape[0]):
                    mask_combined.extend(mask_l[b])
                    mask_combined.extend(mask_m[b])
                    mask_combined.extend(mask_r[b])
                # print("mask:", mask.shape)
                # input("mask")
                # mask_combined.extend(mask_l)
                # mask_combined.extend(mask_m)
                # mask_combined.extend(mask_r)
            elif exp == 2: # img0, img1, gt0, gt1, gt2
                middle, flow_middle, mask = model.inference(I0, I1, scale_list) # get the flow too
                # pred, info = model.update(imgs, gt[:, 1:2], dataset, learning_rate, training=True) # img0, img1, gt1
                middle_left, flow_left, mask = model.inference(I0, middle, scale_list) # get the flow too
                middle_right, flow_right, mask = model.inference(middle, I1, scale_list) # get the flow too
                middle = np.asarray(middle.detach().cpu().numpy())
                middle_left = np.asarray(middle_left.detach().cpu().numpy())
                middle_right = np.asarray(middle_right.detach().cpu().numpy())
                interpol_data.append(middle_left.transpose((0,2,3,1)))
                interpol_data.append(middle.transpose((0,2,3,1)))
                interpol_data.append(middle_right.transpose((0,2,3,1)))
                print(middle.shape)
                flow_left = np.asarray(flow_left)
                flow_middle = np.asarray(flow_middle)
                flow_right = np.asarray(flow_right)
                flow_array_ = flow_left[2].detach().cpu().numpy()
                flow_combined.append(flow_array_[:,0:4,:,:].squeeze()) # 0:4 because in x and y for Ft->0, Ft->1 intermediate flows
                flow_array_ = flow_middle[2].detach().cpu().numpy()
                flow_combined.append(flow_array_[:,0:4,:,:].squeeze())
                flow_array_ = flow_right[2].detach().cpu().numpy()
                flow_combined.append(flow_array_[:,0:4,:,:].squeeze())
                print(flow_middle.shape)
                # input("exp2") 

        interpol_data = np.squeeze(np.array(interpol_data))
        print("interpolated:", interpol_data.shape)
        title = "inference_" + str(factor) + "x"
        # if "2d" in dataset:
        #     visualize_series(interpol_data, factor, dataset, dir_res, title=title, show=False, save=True)

        # Vector field quiver plot
        flow_combined = np.array(flow_combined)
        mask_combined = np.squeeze(np.array(mask_combined))
        flow_gt_combined = np.array(flow_gt_combined)
        print("Flow:", flow_combined.shape)
        # print(interpol_data.dtype)
        flow_u = flow_combined[:, 0, ...]
        flow_v = flow_combined[:, 1, ...]
        title = "inference_flow_arrows_" + str(factor) + "x"
        # if "2d" in dataset:
        #     visualize_series_flow(interpol_data, flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)

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
            # print(val_loss.shape)
            # input("x")
        except:
            print("loss json doesn't exist")

        plot_loss(val_loss, dir_model, name="val_loss.png", save=True)

        data_test = np.array(data_test_combined)
        print("data_test:", data_test.shape)
        # if dataset == "pipedcylinder2d" or dataset == "cylinder2d" or dataset == "FluidSimML2d" or dataset == "rectangle2d":
        #     original_data = data_test[:, :, 0] # gt
        # else:
        #     original_data = data_test
        original_data = data_test
        print(original_data.shape)
        print("Data is in range %f to %f" % (np.min(original_data), np.max(original_data)))
        # input("x")
        return original_data, interpol_data, flow_combined, flow_gt_combined, mask_combined
    
def evaluate(model, dataset, val_data, nr_eval, local_rank): #, writer_val):
    print("Evaluate")
    # input(model_name)
    # writer = SummaryWriter('train') #
    # writer_val = SummaryWriter('validate') #
    # loss_l1_list = []
    # loss_distill_list = []
    # loss_tea_list = []
    loss_G_list = []
    psnr_list = []
    psnr_list_teacher = []
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
        with torch.no_grad():
            pred, info = model.update(imgs, gt, dataset, training=False)
            merged_img = info['merged_tea']
        # loss_l1_list.append(info['loss_l1'].cpu().numpy())
        # loss_tea_list.append(info['loss_tea'].cpu().numpy())
        # loss_distill_list.append(info['loss_distill'].cpu().numpy())
        # for key, value in info.items():
        #     print (key, value)
        # input("x")
        loss_all = (info['loss_G'].cpu().numpy(), info['loss_l1'].cpu().numpy(), info['loss_tea'].cpu().numpy(), 
            info['loss_distill'].cpu().numpy(), info['l1_reg'].cpu().numpy(), info['loss_photo'].cpu().numpy(), 
            info['loss_flow'].cpu().numpy())
        # loss_G_list.append(loss_all)
        # loss_G_list.append(info['loss_G'].cpu().numpy())
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

    model.save_model(model_name, log_path, local_rank) 

    val_loss = []
    # val_loss.append(float(np.array(loss_G_list).mean()))
    loss_all = np.array(loss_all).tolist()
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

    eval_time_interval = time.time() - time_stamp

    del loss_all

    if local_rank != 0:
        return
    # writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=1000, type=int) # 300 1000
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size') # default=16
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
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
    if args.dataset == "vimeo2d" and args.mode == "train":
        args.batch_size = 55
    if args.dataset == "droplet2d" or args.dataset == "rectangle2d":
        args.batch_size = 128
    if args.dataset == "rectangle2d":
        args.batch_size = 180
    if args.dataset == "lbs2d":
        args.batch_size = 100
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

    """ lbs2d """
    # model_name = "flownet_flowonly_v2_128_lbs.pkl" # 665 ep: perf interpol, good flow but it was too easy
    # model_name = "flownet_lapl_dist_photo1e-5_v2_128_lbs.pkl" # 200 ep: very good interpol, bad flow
    # model_name = "flownet_lapl_dist_photo1e-5_v2_128_lbs_skip.pkl" # 250 ep: perf interpol, good flow but it was too easy
    # model_name = "flownet_flowonly_v2_128_lbs_skip.pkl" # 175 ep: 
    # TODO: analyze dataset and ? create ensembles

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
    # we don't have flow for supervision

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
    model_name = "flownet_flowonly_v2_128_piped_allBlocks.pkl" # veru good flow and interpol
    # model_name = "flownet_flowonly_v2_128_piped_allBlocks_inv.pkl" # test, inverse

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
        train(model, args.dataset, args.exp, model_name, args.mode, args.local_rank)
    else:
        dataset = args.dataset
        print("inference...")
        factor = 2
        dir_res = "Results"
        dir_res = os.path.join(dir_res, dataset)
        dir_res = os.path.join(dir_res, str(factor) + "x")
        dir_model = os.path.join(dir_res, model_name[:-4])
        # print(dir_model)f
        # input("x")
        print("Saving at:", dir_model)

        data_test, interpol_data, flow, flow_gt, mask = train(model, args.dataset, args.exp, model_name, args.mode, args.local_rank)
        print("data_test:", data_test.shape)
        print("interpol_data:", interpol_data.shape)
        print("flow:", flow.shape)
        print("mask:", mask.shape)
        # input("inf")
        # original_data = data_test[:, 2, :interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
        original_data = data_test[:, :interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
        print("original_data:", original_data.shape)
        print("interpol_data:", interpol_data.shape)
        print("flow & mask:", flow.shape, mask.shape)

        # flow droplet2d:
        # print(flow.shape, flow_gt.shape)
        
        # interpol_data = interpol_data
        # print(original_data.shape, interpol_data.shape, flow.shape, flow_gt.shape)
        print("original data is in range %f to %f" % (np.min(original_data), np.max(original_data)))
        print("interpol data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))
        # input("x")

        # calc diff, metrics
        diffs = calculate_diff(original_data, interpol_data, dataset, factor, dir_model)
        print("diffs:", diffs.shape)

        if dataset != "vimeo2d" and dataset != "droplet2d":
            print("GT flow available")
            flow_gt = flow_gt[:, :, :flow.shape[2], :flow.shape[3]]
            u_gt = flow_gt[:, 0, ...]
            v_gt = flow_gt[:, 1, ...]
            norm_gt = np.sqrt(u_gt * u_gt + v_gt * v_gt)
            # viz Ft->1 in the end, that is flow[2][:, 2:4]
            u = flow[:, 2, ...] # flow[:, 0, ...]
            v = flow[:, 3, ...] # flow[:, 1, ...]
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
            diffs_flow = np.zeros((original_data.shape[0], original_data.shape[1], original_data.shape[2]), dtype=np.float32)
            flow_gt = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
            # u_diff = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
            # v_diff = np.zeros((flow.shape[0], flow.shape[1], flow.shape[2], flow.shape[3]), dtype=np.float32)
            u_diff = np.zeros((flow.shape[0], flow.shape[2], flow.shape[3]), dtype=np.float32)
            v_diff = np.zeros((flow.shape[0], flow.shape[2], flow.shape[3]), dtype=np.float32)

        print("Flow gt mean and std:", np.mean(flow_gt), np.std(flow_gt))
        print("Flow pred mean and std:", np.mean(flow), np.std(flow))

        title = "GT_Interpol_Diff_Flow_" + str(factor) + "x"
        if "2d" in dataset:
            visualize_large(original_data, interpol_data, diffs,
                flow, flow_gt, diffs_flow, u_diff, v_diff, mask,
                factor, dataset, dir_model, title=title, show=False, save=True)

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
        
