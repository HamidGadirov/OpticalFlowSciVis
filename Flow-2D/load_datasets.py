from curses.textpad import rectangle
# from turtle import shape
from matplotlib.pyplot import axes, title
from dataset import *
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
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# from utils import visualize_ind, visualize_series, visualize_series_flow, visualize_large
from utils import plot_distribution

# from skimage.transform import rescale, resize, downscale_local_mean

# TODO:
# remove RGB +
# add augemntaion: flip +
# add more data from other datasets +-
# check patching -
# FluidSimML: downsample to 256x256? +
# try some optimization (TensorRT, ONNX) -

def load_data(dataset, factor, mode):
    combined_data_train = []
    combined_data_val = []
    if dataset == 'rectangle2d':
        datasets = ["rectangle2d"]
    if dataset == 'lbs2d':
        datasets = ["lbs2d"]
    if dataset == 'droplet2d':
        datasets = ["droplet2d"]
    elif dataset == 'pipedcylinder2d':
        datasets = ["pipedcylinder2d"]
    elif dataset == 'cylinder2d':
        datasets = ["cylinder2d"]
    elif dataset == 'FluidSimML2d':
        datasets = ["FluidSimML2d"]
    elif dataset == 'all':
        datasets = ["droplet2d", "FluidSimML2d", "pipedcylinder2d", "cylinder2d"]
    
    # datasets = ["FluidSimML2d", "pipedcylinder2d", "cylinder2d"]
    # datasets = ["droplet2d", "pipedcylinder2d"]
    
    for dataset in datasets:
        print(dataset)
        filename = "../Datasets/"
        if dataset == 'rectangle2d':
            filename += "rectangle2d.pkl"
            # flow_fln = "../Datasets/rectangle2d_flow.pkl"
            # flow_fln = "../Datasets/rectangle2d_text_flow.pkl"
            # flow_fln = "../Datasets/rectangle2d_hftext_flow.pkl"
            # flow_fln = "../Datasets/rectangle2d_big_hftext_flow.pkl"
            flow_fln = "../Datasets/rectangle2d_big_hftext_flow_v3.pkl"
            flow_fln = "../Datasets/rectangle2d_big_f_bb_flow.pkl"
            # flow_fln = "../Datasets/rectangle2d_big_nohf_flow.pkl"
            # flow_fln = "../Datasets/rectangles2d_text_flow.pkl"
        if dataset == 'lbs2d':
            # filename += "LatticeBoltzmannSim/lbs2d.pkl"
            # filename += "LatticeBoltzmannSim/lbs2d_skip.pkl"
            # filename += "LatticeBoltzmannSim/lbs2d_skip5.pkl"
            # filename += "LatticeBoltzmannSim/lbs2d_skip5_11K.pkl"
            # filename += "LatticeBoltzmannSim/lbs2d_skip3_combi.pkl"
            # filename = "../../../DataSSD/Datasets/LBS/lbs2d_skip3_combi_cyl_tau.pkl"
            # filename = "../../../DataSSD/Datasets/LBS/lbs2d_skip3_ens7.pkl"
            # filename = "../../../DataSSD/Datasets/LBS/lbs2d_skip3_ensemble10.pkl"
            # filename = "../../../DataSSD/Datasets/LBS/Spare_members/lbs2d_skip3_ensemble21.pkl"
            # filename = "../../../DataSSD/Datasets/LBS/lbs2d_skip3_ensemble16m_f32.pkl"
            filename = "../../../DataSSD/Datasets/LBS/lbs2d_skip3_ensemble21m_f32.pkl"
            # filename = "../../../DataSSD/Datasets/LBS/lbs2d_skip3_Cy1r1t1.pkl"
        
        if dataset == "droplet2d":
            filename += "drop2D/droplet2d.pkl" if mode == "train" else "drop2D/droplet2d_test.pkl"
        elif dataset == "pipedcylinder2d":
            filename += "pipedcylinder2d.pkl"
            flow_fln = "../Datasets/pipedcylinder2d_flow.pkl"
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
            print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            print(data.shape)
            data[:, 0] = cv2.normalize(data[:, 0], data[:, 0], 0., 1., cv2.NORM_MINMAX)
            print("Density scaled to [0, 1] range!")
            # data[:, 1] = cv2.normalize(data[:, 1], data[:, 1], -1., 1., cv2.NORM_MINMAX)
            # data[:, 2] = cv2.normalize(data[:, 2], data[:, 2], -1., 1., cv2.NORM_MINMAX)
            print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
            print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
            # input("Data range")
            # flip flow directions
            # data[:, 1] *= -1
            # data[:, 2] *= -1
            # input("x")
        # if dataset == "droplet2d" or dataset == "cylinder2d" or dataset == "FluidSimML":
        elif dataset == "lbs2d":
            with open(filename, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            print(data.shape)
            print("data type:", data.dtype)
            # input("x")
            # 3rdShift and DensNorm important for this dataset!
            # data = data[18000:21000]
            # data = np.float32(data)
            # data = np.float16(data)
            # print("Converted to float16")
            # print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            # print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
            # print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
            # print("Density mean std", np.mean(data[:, 0]), np.std(data[:, 0]))
            # print("Vel x mean std", np.mean(data[:, 1]), np.std(data[:, 1]))
            # data = (data - np.min(data)) / (np.max(data) - np.min(data))
            # print(np.min(data[:, 0]), np.max(data[:, 0]), data[:, 0].mean(), data[:, 0].std())
            # dens_min = np.min(data[:, 0])
            # dens_max = np.max(data[:, 0])
            dens_mean = data[:, 0].mean()
            dens_std = data[:, 0].std()
            # input("x")
            data[:, 0] = (data[:, 0] - dens_mean) / dens_std # it is important!
            print("Density normalized to 0 mean and 1 variance!")
            print(dens_mean, dens_std)
            print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))

            dens_min = np.min(data[:, 0])
            dens_max = np.max(data[:, 0])

            # data = data * 255.0
            # data = data.astype(int)
            # data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
            # data[:, 0] = cv2.normalize(data[:, 0], data[:, 0], 0., 1., cv2.NORM_MINMAX) # this affects flow ???
            # data[:, 0] = data[:, 0] / np.max(data[:, 0]) # to [0, 1] which is not!
            # data[:, 0] = (data[:, 0] - dens_min) / (dens_max - dens_min) # to [0, 1] is a tiny bit worse?
            # data[:, 1] = cv2.normalize(data[:, 1], data[:, 1], -1., 1., cv2.NORM_MINMAX)
            # data[:, 2] = cv2.normalize(data[:, 2], data[:, 2], -1., 1., cv2.NORM_MINMAX)
            # data[:, 0] = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
            # data[:, 1] = (data[:, 1] - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))
            # data[:, 2] = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
            # print(data)
            # print("Density scaled to [0, 1] range!")

            # print("Vel x mean std", np.mean(data[:, 1]), np.std(data[:, 1]))
            # factor = 2
            # dir_res = "Results"
            # dir_res = os.path.join(dir_res, dataset)
            # dir_res = os.path.join(dir_res, str(factor) + "x")
            # model_name = "flownet_l1_flow2e-1_v1_256_lbs_skip5_ScaleDens_gamma_RIFE_eqRes_lr4e-4.pkl"
            # dir_model = os.path.join(dir_res, model_name[:-4])
            # # plot_distribution(data[:, 1:], dir_model, name="dict_flow_gt_before.png", save=True)
            data[:, 1] = data[:, 1] * 10.
            data[:, 2] = data[:, 2] * 10.
            print("Vel scaled to *10")
            # # data[:, 1] = np.log(data[:, 1] + 0.3) # +1 to avoid nan
            # # data[:, 2] = np.log(data[:, 2] + 0.3) # +1 to avoid nan
            # # print("Vel log normalized")
            # # print("Vel x mean std", np.mean(data[:, 1]), np.std(data[:, 1]))
            # plot_distribution(data[:, 1:], dir_model, name="dict_flow_gt_after.png", save=True)

            # flow_x_mean = data[:, 1].mean()
            # flow_y_mean = data[:, 2].mean()
            # flow_x_std = data[:, 1].std()
            # flow_y_std = data[:, 2].std()
            # data[:, 1] = (data[:, 1] - flow_x_mean) / flow_x_std
            # data[:, 2] = (data[:, 2] - flow_y_mean) / flow_y_std
            # print("Normalized to mean 0 and std 1!")
            # print("Vel x mean std", np.mean(data[:, 1]), np.std(data[:, 1]))

            # print("data[:, 1]):", data[:, 1])

            # # test
            # data[:, 1]  = data[:, 1]  * flow_x_std + flow_x_mean
            # data[:, 2]  = data[:, 2]  * flow_y_std + flow_y_mean
            # print("Denormalized from mean 0 and std 1!")
            # print("Vel x mean std", np.mean(data[:, 1]), np.std(data[:, 1]))
            # print("data[:, 1]):", data[:, 1])
            # input("norm")

            # print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            # print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
            # print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
            # print("Density mean std", np.mean(data[:, 0]), np.std(data[:, 0]))
            # print("Vel x mean std", np.mean(data[:, 1]), np.std(data[:, 1]))
            # print("Data shape:", data.shape)
            # input("Data range")
        else:
            with open(filename, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            print(data.shape)
            print("data type:", data.dtype) # droplet2d was in float64 ???
            data = np.float32(data)
            print("data type:", data.dtype)
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            print("Normalized to mean 0 and std 1!")
            # data = data * 255.0
            # data = data.astype(int)
            data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
            print("Scaled to [0, 1] range!")
            # print(data)
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            # input("x")
        if dataset == "pipedcylinder2d" or dataset == "cylinder2d" or dataset == "FluidSimML2d":
            pkl_file = open(flow_fln, 'rb')
            flow_uv = []
            flow_uv = pickle.load(pkl_file)
            print(flow_uv.shape)
            # flow_uv = (flow_uv - np.min(flow_uv)) / (np.max(flow_uv) - np.min(flow_uv))
            print("Flow is in range %f to %f" % (np.min(flow_uv), np.max(flow_uv)))
            flow_uv = np.float32(flow_uv)

            data = data[:, np.newaxis, ...]
            # print(data.shape, flow_uv.shape)
            # input("x")
            data_flow = np.hstack((data, flow_uv))
            print("data_flow", data_flow.shape)
            data = data_flow
            data = np.expand_dims(data, axis=-1)
            # input("x")

        # data_ = data
        # data_ /= 255.
        # print(np.mean(data_), np.std(data_))
        # input("x")

        # factor = 2
        # dir_res = "Results"
        # dir_res = os.path.join(dir_res, dataset)
        # dir_res = os.path.join(dir_res, str(factor) + "x")
        # print("Saving at:", dir_res)
        # title = "train_set"
        # visualize_series(data[:210], factor, dataset, dir_res, title=title, show=False, save=True)
        # input("x")

        # data = data[5:] # skip empty
        # data = np.squeeze(data)
        if data.ndim == 3 or dataset == "rectangle2d" or dataset == "lbs2d":
            data = np.expand_dims(data, axis=-1)

        # convert to RGB for training
        # data = np.float32(data)
        # data_rgb = []
        # for i in range(data.shape[0]):
        #     data_rgb.append(cv2.cvtColor(data[i], cv2.COLOR_GRAY2RGB))
        # data = np.array(data_rgb)
        # data = np.moveaxis(data, -1, 1)
        # print("data in RGB:", data.shape)

        # data = np.float32(data)
        data = np.moveaxis(data, -1, 1)
        print("data in grayscale:", data.shape)
        # input("x")

        if mode == "train":
            # print("a train subset of the member is selected")
            if "rectangle2d" in filename:
                data_train = data[:2205] # div to 3, 5, 9 and 7
                data_val = data[2370:2685] # div to 3, 5, 9 and 7
            if "lbs2d" in filename:

                # # data_train = data
                # data_train = data[:1500]
                # data_train = np.append(data_train, data[1710:4500], axis=0)
                # data_train = np.append(data_train, data[4710:7500], axis=0)
                # # data_train = np.append(data_train, data[7710:], axis=0)
                # data_train = np.append(data_train, data[7710:10500], axis=0)
                # data_train = np.append(data_train, data[10710:13500], axis=0)
                # data_train = np.append(data_train, data[13710:], axis=0) # 13950 in total

                """
                # ens10
                data_train = data[:3000]
                data_train = np.append(data_train, data[6000:15000], axis=0)
                data_train = np.append(data_train, data[18000:], axis=0)
                """

                """
                # ens16
                data_train = data[:18000]
                data_train = np.append(data_train, data[21000:30000], axis=0)
                data_train = np.append(data_train, data[33000:48000], axis=0)
                """

                # """
                # ens21
                data_train = data[:21000]
                data_train = np.append(data_train, data[24000:60000], axis=0)
                # """
         
                # data_train = data[3000:18000] # ens7
                # data_train = np.append(data_train, data[21000:30000], axis=0)
                # data_train = np.append(data_train, data[33000:48000], axis=0)

                # data_train = data[:3000]
                # data_train = np.append(data_train, data[3300:5000], axis=0)
                # data_train = np.append(data_train, data[5210:7000], axis=0) # 8490
                # data_train = np.append(data_train, data[7210:10800], axis=0) # 10080 in total, div to 9
                # data_train = np.append(data_train, data[-1205:], axis=0)

                # data_val = np.append(data_val, data[1500:1605], axis=0)
                # data_val = np.append(data_val, data[2500:2605], axis=0) # 315 in total 

                # data_val = data[1500:1605]
                # data_val = np.append(data_val, data[4500:4605], axis=0)
                # data_val = np.append(data_val, data[7500:7605], axis=0)
                # data_val = np.append(data_val, data[10500:10605], axis=0)
                # data_val = np.append(data_val, data[13500:13605], axis=0)

                # data_val = data[30000:33000] # ens16
                # data_val = data[18000:21000] # ens7
                # data_val = data[3000:6000] # ens10
                data_val = data[60000:63000] # ens21

                # data_val = data[3000:3210]
                # data_val = np.append(data_val, data[5000:5210], axis=0)
                # data_val = np.append(data_val, data[7000:7210], axis=0) # 630 in total 
                # data_train = data[:300]
                # data_val = data[:300]

                del data
                # input("data del")
            if "droplet2d" in filename:
                # data = np.float16(data)
                data_train = data[2700:51300] # 46800 15120
                data_val = data[51300:54000] # 18000 54000
            if "pipedcylinder2d" in filename or "cylinder2d" in filename: # 1501 in total
                data_train = data[:540] # 1080 div to 27 and 5
                data_train = np.append(data_train, data[-540:], axis=0)
                data_val = data[540:810] # 1008:1296
                # invert magnitude
                # data_train = (data_train * 255).astype(int)
                # data_train = np.invert(data_train)
                # data_train = (data_train - np.min(data_train)) / (np.max(data_train) - np.min(data_train))
                # data_train = np.float32(data_train)
                # data_val = (data_val * 255).astype(int)
                # data_val = np.invert(data_val)
                # data_val = (data_val - np.min(data_val)) / (np.max(data_val) - np.min(data_val))
                # data_val = np.float32(data_val)
                # input("x")

            elif "FluidSimML" in filename:
                data_train = data[100:820] 
                data_val = data[820:964]

            if dataset != "rectangle2d":
                # prepare data for training - use only each third frame while shifting the sampling; this is important!!
                print("use 3rd while shifting the sampling")
                # data_train_ = []
                # data_val_ = []
                # for shift in range(3):
                #     for i in range(shift, data_train.shape[0], 3): # 10
                #         data_train_.append(data_train[i])
                #     for i in range(shift, data_val.shape[0], 3): 
                #         data_val_.append(data_val[i])
                # data_train = data_train_
                # data_val = data_val_
                # data_train = np.asarray(data_train)
                # data_val = np.asarray(data_val)

                data_train_new = data_train[::3] # array is more efficient than list in case of large
                data_val_new = data_val[::3]
                for shift in range(1, 3):
                    data_train_new = np.vstack([data_train_new, data_train[shift::3]])
                    data_val_new = np.vstack([data_val_new, data_val[shift::3]])
                data_train = data_train_new
                data_val = data_val_new

                print(data_train.shape)
                print(data_val.shape)
                # input("3rd")
                del data_train_new, data_val_new
                # input("del")

                # print("use 5th while shifting the sampling")
                # data_train_5 = []
                # data_val_5 = []
                # for shift in range(5):
                #     for i in range(shift, data_train.shape[0], 5): # 10
                #         data_train_5.append(data_train[i])
                #     for i in range(shift, data_val.shape[0], 5): 
                #         data_val_5.append(data_val[i])
                # append 3rd and 5th
                # data_train = data_train_
                # print("3rd:", len(data_train))
                # data_train.extend(data_train_5)
                # print("5th:", len(data_train_5))
                # print("3rd + 5th:", len(data_train))
                # data_val = data_val_
                # data_val.extend(data_val_5)
                # data_train = np.asarray(data_train)
                # data_val = np.asarray(data_val)
                # print(data_train.shape)
                # print(data_val.shape)
                # input("x")

            print(data_train.shape) # (2205, 1, 3, 128, 128)
            # print("before aug, data_train 0 is in range %f to %f" % (np.min(data_train[:, :, 0]), np.max(data_train[:, :, 0])))

            # TODO: add other augment: diff steps for picking data ? 
            if "pipedcylinder2d" in filename or "cylinder2d" in filename or "FluidSimML" in filename or \
                "rectangle2d" in filename: #  or "lbs2d" in filename: no aug for lbs2d_skip5_11K
                print("Augmenting the data...")
                data_train_flip = data_train[..., ::-1] # [:,:,:,::-1] wrong!
                data_train = np.append(data_train, data_train_flip, axis=0)
                data_train_flip = data_train[..., ::-1, :]  # [:,:,::-1,:] wrong!
                data_train = np.append(data_train, data_train_flip, axis=0)

                print(data_train.shape)
                print("after aug, data_train is in range %f to %f" % (np.min(data_train[:, :, 0]), np.max(data_train[:, :, 0])))
                # input("x")

            # data_train = torch.from_numpy(data_train.copy()).permute(2, 0, 1)
            # data_val = torch.from_numpy(data_val.copy()).permute(2, 0, 1)
            # print(data_train.shape)

            # if dataset == "pipedcylinder2d":
            #     data_train = data_train[:, np.newaxis, ...]
            #     data_val = data_val[:, np.newaxis, ...]

            data_train_three = []
            data_val_three = []
            if factor == 2: # img0, img1, gt
                if dataset == "rectangle2d__":
                    range_list = [3, 5, 7, 9]
                    repeat = 1
                    # range_max = random.choice(range_list)
                    for n in range(len(range_list)):
                        range_max = range_list[n]
                        # for shift in range(range_max): # no need, too much data then
                        for shift in range(repeat):
                            for i in range(0 + shift, data_train.shape[0] - range_max, range_max): 
                                data_train_three.append(np.concatenate((data_train[i], 
                                    data_train[i + range_max-1], data_train[i + int((range_max-1)/2)]), axis=0)) # img0, img1, gt
                            for i in range(0 + shift, data_val.shape[0] - range_max, range_max): 
                                data_val_three.append(np.concatenate((data_val[i], 
                                    data_val[i + range_max-1], data_val[i + int((range_max-1)/2)]), axis=0)) # img0, img1, gt
                            # print(range_max, repeat) # np.array(data_train_three).shape)
                        # repeat += 1
                    data_train = np.array(data_train_three)
                    print("data_train in three:", data_train.shape)
                    data_val = np.array(data_val_three)
                    print("data_val in three:", data_val.shape)

                    # # visualize
                    # factor = 2
                    # dir_res = "Results"
                    # dir_res = os.path.join(dir_res, dataset)
                    # dir_res = os.path.join(dir_res, str(factor) + "x")
                    # print("Saving at:", dir_res)
                    # title = title = "train_threeseq_aug_range"
                    # print(data_train.shape)
                    # data_train_vis = data_train[14000:14100, :, 0, ...]
                    # data_train_vis_unthree = []
                    # for i in range(data_train_vis.shape[0]):
                    #     data_train_vis_unthree.append(np.concatenate((data_train_vis[i, 0], data_train_vis[i, 2], data_train_vis[i, 1]), axis=0))
                    # data_train_vis_unthree = np.array(data_train_vis_unthree)
                    # print(data_train_vis_unthree.shape)
                    # # input("x")
                    # visualize_series(data_train_vis_unthree, factor, dataset, dir_res, title=title, show=False, save=True)
                    # input("3579")
                else:
                    # prepare img0, gt, img1 (2x interpolation)
                    print("prepare img0, gt, img1 (2x interpolation)")
                    # TODO: shift by 2 each iter +
                    for i in range(0, data_train.shape[0], 2): 
                        try:
                            data_train_three.append(np.concatenate((data_train[i], data_train[i+2], data_train[i+1]), axis=0)) # img0, img1, gt
                        except:
                            print("data_train_three ready")
                    data_train = np.array(data_train_three)
                    print("data_train in three:", data_train.shape)
                    for i in range(0, data_val.shape[0], 2): 
                        try:
                            data_val_three.append(np.concatenate((data_val[i], data_val[i+2], data_val[i+1]), axis=0)) # img0, img1, gt
                        except:
                            print("data_val_three ready")
                    data_val = np.array(data_val_three)
                    print("data_val in three:", data_val.shape)

                    # # on the fly with less memory
                    # for i in range(0, data_train.shape[0]-2, 2):
                    #     data_train[i] = np.concatenate((data_train[i], data_train[i+2], data_train[i+1]), axis=0)
                    # data_train = data_train[:data_train.shape[0]:3]
                    # print("data_train in three:", data_train.shape)

                    # for i in range(0, data_val.shape[0]-2, 2):
                    #     data_val[i] = np.concatenate((data_val[i], data_val[i+2], data_val[i+1]), axis=0)
                    # data_val = data_val[:data_val.shape[0]:3]
                    # print("data_val in three:", data_val.shape)

                    # input("data_train data_train in three")

                    # # visualize
                    # factor = 2
                    # dir_res = "Results"
                    # dir_res = os.path.join(dir_res, dataset)
                    # dir_res = os.path.join(dir_res, str(factor) + "x")
                    # print("Saving at:", dir_res)
                    # title = title = "train_threeseq_aug_range"
                    # print(data_train.shape)
                    # data_train_vis = data_train[:100, :, 0, ...]
                    # data_train_vis_unthree = []
                    # for i in range(data_train_vis.shape[0]):
                    #     data_train_vis_unthree.append(np.concatenate((data_train_vis[i, 0], data_train_vis[i, 2], data_train_vis[i, 1]), axis=0))
                    # data_train_vis_unthree = np.array(data_train_vis_unthree)
                    # print(data_train_vis_unthree.shape)
                    # # input("x")
                    # visualize_series(data_train_vis_unthree, factor, dataset, dir_res, title=title, show=False, save=True)
                    # input("x")
                # input("x")
            elif factor == 4:
                print("4x interpolation") 
                for i in range(0, data_train.shape[0], 5): 
                    data_train_three.append(np.concatenate((data_train[i], data_train[i+4],
                        data_train[i+1], data_train[i+2], data_train[i+3]), axis=0)) # img0, img1, gt0, gt1, gt2
                data_train = np.array(data_train_three)
                print("data_train in three:", data_train.shape)
                for i in range(0, data_val.shape[0], 5): 
                    data_val_three.append(np.concatenate((data_val[i], data_val[i+4],
                        data_val[i+1], data_val[i+2], data_val[i+3]), axis=0)) # img0, img1, gt0, gt1, gt2
                data_val = np.array(data_val_three)
                print("data_val in three:", data_val.shape)
                # input("x")
            elif factor == 8:
                print("8x interpolation") # img0, img1, gt0, gt1, gt2, gt3, gt4, gt5, gt6
                for i in range(0, data_train.shape[0], 9):
                    # print(i)
                    data_train_three.append(np.concatenate((data_train[i], data_train[i+8],
                        data_train[i+1], data_train[i+2], data_train[i+3],
                        data_train[i+4], data_train[i+5], data_train[i+6], data_train[i+7]), axis=0)) 
                data_train = np.array(data_train_three)
                print("data_train in three:", data_train.shape)
                for i in range(0, data_val.shape[0], 9): 
                    data_val_three.append(np.concatenate((data_val[i], data_val[i+8],
                        data_val[i+1], data_val[i+2], data_val[i+3],
                        data_val[i+4], data_val[i+5], data_val[i+6], data_val[i+7]), axis=0)) 
                data_val = np.array(data_val_three)
                print("data_val in three:", data_val.shape)
                # input("x")

            # data_train = np.float32(data_train)
            # data_val = np.float32(data_val)
            # print("Converted back to float32")

            combined_data_train = torch.utils.data.ConcatDataset([combined_data_train, data_train])
            combined_data_val = torch.utils.data.ConcatDataset([combined_data_val, data_val])
        
            # # combined_data_train.extend(data_train.squeeze())
            # # combined_data_val.extend(data_val.squeeze())
            # combined_data_train = np.array(combined_data_train)
            # combined_data_val = np.array(combined_data_val)
            # print(combined_data_train.shape, combined_data_val.shape)
            # input("x")
            # print("data to combined_data_train:", combined_data_train.shape)
            return combined_data_train, combined_data_val
            # return data_train, data_val
        else: # inference
            if "rectangle2d" in filename:
                data_test = data[2685:3000] # div to 3, 5, 9 and 7
            if "lbs2d" in filename:
                # data_test = data[2685:3000] # [5685:6000]
                # data_test = data[1315:1630]
                # data_test = data[600:915] # flow not super accurate but much better than on unseen (before)
                # data_test = data[2000:2315]
                # data_test = data[605:710]
                # data_test = np.append(data_test, data[1605:1710], axis=0)
                # data_test = np.append(data_test, data[2605:2710], axis=0) # 315 in total 
                # data_test = data[2605:2695] # viz last subset
                # data_test = data[1605:1710]
                # data_test = data[:3000] # ens7
                # data_test = data[9000:12000] # ens10
                # data_test = data[1605:1905]
                # data_test = data[4605:4710]
                # data_test = data[1000:1300]
                # data_test = data[10605:10710]
                # data_test = data[15000:18000] # ens10
                # data_test = data[16605:18000] # ens10
                # data_test = data[18000:21000] # ens16
                data_test = data[21000:24000] # ens21
                # data_test = data[22607:24000] # ens21 22605

                # data_test = data[30100:30400] # data[30000:33000] 
                # data_test = data[40100:40400]
                # data_test = np.float32(data_test)
                # print("Converted back to float32")

                # data_test = data[4605:4710]
                # data_test = data[7605:7710]
                # data_test = data[3210:3300] # viz small subset
                # data_test = data[9210:9300] # viz small subset # values becomeing even smaller...
                # data_test = data[1000:1315]
                # print("data_test shape:", data_test.shape) # (315, 1, 3, 100, 400)
                # print(data_test[0, :, 1, 50, 100], data_test[0, :, 1, 50, 200], data_test[0, :, 1, 50, 300])
                # print(data_test[0, :, 2, 50, 100], data_test[0, :, 2, 50, 200], data_test[0, :, 2, 50, 300])
                # print(data_test[100, :, 1, 50, 100], data_test[100, :, 1, 50, 200], data_test[100, :, 1, 50, 300])
                # print(data_test[100, :, 2, 50, 100], data_test[100, :, 2, 50, 200], data_test[100, :, 2, 50, 300])
                # print(data_test[200, :, 1, 50, 100], data_test[200, :, 1, 50, 200], data_test[200, :, 1, 50, 300])
                # print(data_test[200, :, 2, 50, 100], data_test[200, :, 2, 50, 200], data_test[200, :, 2, 50, 300])
                # input("Data range")

                # print("##########")
                flow_x = data_test[:, :, 1]
                # print("Data data_test:", flow_x.mean(), flow_x.std())
                flow_x_mean = flow_x.mean()
                flow_x_std = flow_x.std()
                # flow_x = (flow_x - flow_x.mean()) / flow_x.std()
                # print("Normalized to mean 0 and std 1!")
                # print("Data data_test:", flow_x.mean(), flow_x.std())
                flow_y = data_test[:, :, 2]
                flow_y_mean = flow_y.mean()
                flow_y_std = flow_y.std()
                # print("Data data_test:", flow_y.mean(), flow_y.std())
                # flow_y = (flow_y - flow_y.mean()) / flow_y.std()
                # print("Normalized to mean 0 and std 1!")
                # print("Data data_test:", flow_y.mean(), flow_y.std())
                # # print("Vel x mean std", np.mean(data[:, :, 1]), np.std(data[:, :, 1]))
                # input("xxx")

                # # flow_x_mean = data_test[:, :, 1].mean()
                # # flow_y_mean = data_test[:, :, 2].mean()
                # # flow_x_std = data_test[:, :, 1].std()
                # # flow_y_std = data_test[:, :, 2].std()
                # # data_test[:, :, 1] = (data_test[:, :, 1] - flow_x_mean) / flow_x_std
                # # data_test[:, :, 2] = (data_test[:, :, 2] - flow_y_mean) / flow_y_std
                # data_test[:, :, 1] = flow_x
                # data_test[:, :, 2] = flow_y
                # print("Normalized to mean 0 and std 1!")
                # print("Vel x mean std", data_test[:, :, 1].mean(), data_test[:, :, 1].std())
                # print("Vel y mean std", data_test[:, :, 2].mean(), data_test[:, :, 2].std())
                # input("xxxx")
            elif "droplet2d" in filename:
                # data_test = data[:5000] # 35.02
                # data_test = data[:3000] # 36.82
                # data_test = data[:2700] # 36.81
                # data_test = data[2700:5400] # 34.26
                # data_test = data[350:2700] # 35.12
                # data_test = data[350:950] # 29.98
                data_test = data[350:650] # 35.12
                # data_test = data[:350] # 31.07
                # data_test = data[1500:2100] # 30.57
                # data_test = data[2400:2700] # 28.07
                # data_test = data[0:100] # 27.46
                # 450: 2m, 600: 7m, 900: 27m for dist_matrix
            elif "pipedcylinder2d" in filename or "cylinder2d" in filename:
                data_test = data[810:1080]
                # invert magnitude
                # print(data_test[0])
                # data_test = (data_test * 255).astype(int)
                # data_test = np.invert(data_test)
                # data_test = (data_test - np.min(data_test)) / (np.max(data_test) - np.min(data_test))
                # data_test = np.float32(data_test)
                # print(data_test[0])
                # input("x")
            elif "FluidSimML" in filename:
                data_test = data[820:964]

            print(data_test.shape)
            # input("x")

            # if dataset != "rectangle2d" or dataset != "lbs2d":
            if dataset != "rectangle2d": # droplet2d
                # prepare data for training - use only each third frame while shifting the sampling
                print("3rd, test time")
                data_test_ = []
                for shift in range(3):
                    for i in range(shift, data_test.shape[0], 3): # 10
                        data_test_.append(data_test[i])
                data_test = data_test_
                data_test = np.asarray(data_test)

            print(data_test.shape)
            data_test_non_overlap = np.squeeze(data_test)
            print(data_test_non_overlap.shape)
            # input("data_test_non_overlap")

            # if dataset == "pipedcylinder2d":
            #     data_test = data_test[:, np.newaxis, ...]

            # data_test = np.squeeze(data_test)
            data_test_combined = []
            n = factor  # or any desired value for interpolation factor
            print(f"preparing data for {n}x interpolation")

            for i in range(0, data_test.shape[0], n):
                try:
                    concat_in_between = [np.expand_dims(d, axis=0) for j, d in enumerate(data_test[i+1:i+n])]
                    concat_in_between = np.concatenate(concat_in_between, axis=0)
                    # ensure same number of dimensions
                    data_test_i = np.expand_dims(data_test[i], axis=0)
                    data_test_i_n = np.expand_dims(data_test[i+n], axis=0)
                    # print(np.array(concat_in_between).shape)
                    # print(np.array(data_test_i).shape)
                    # print(np.array(data_test_i_n).shape)
                    data_test_combined.append(np.concatenate((data_test_i, data_test_i_n, concat_in_between), axis=0))
                except:
                    print("done")

            data_test = np.array(data_test_combined)
            data_test = np.squeeze(data_test)
            print("data_test combined:", data_test.shape)
            # input("n")

            """
            data_test_three = []
            if exp == 1: # img0, img1, gt
                if dataset == "rectangle2d__":
                    range_list = [3, 5, 7, 9]
                    repeat = 1
                    # range_max = random.choice(range_list)
                    for n in range(len(range_list)):
                        range_max = range_list[n]
                        for shift in range(repeat):
                            for i in range(0 + shift, data_test.shape[0] - range_max, range_max): 
                                data_test_three.append(np.concatenate((data_test[i], 
                                    data_test[i + range_max-1], data_test[i + int((range_max-1)/2)]), axis=0)) # img0, img1, gt
                        # repeat += 1
                    data_test = np.array(data_test_three)
                    print("data_test in three:", data_test.shape)
                    # input("x")
                else:
                    # TODO: shift by 2 +
                    for i in range(0, data_test.shape[0], 2): 
                        try:
                            data_test_three.append(np.concatenate((data_test[i], data_test[i+2], data_test[i+1]), axis=0)) # img0, img1, gt
                        except:
                            print("done")
                    data_test = np.array(data_test_three)
                    print("data_test in three:", data_test.shape)
                    # input("x")
            elif exp == 2:
                print("4x interpolation")  # hift by 4, not 5
                for i in range(0, data_test.shape[0], 4): 
                    try:
                        data_test_three.append(np.concatenate((data_test[i], data_test[i+4],
                        data_test[i+1], data_test[i+2], data_test[i+3]), axis=0)) # img0, img1, gt0, gt1, gt2
                    except:
                        print("done")
                data_test = np.array(data_test_three)
                print("data_test in three:", data_test.shape)
            elif exp == 3:  
                print("8x interpolation") # img0, img1, gt0, gt1, gt2, gt3, gt4, gt5, gt6
                for i in range(0, data_test.shape[0], 8):
                    # print(i)
                    data_test_three.append(np.concatenate((data_test[i], data_test[i+8],
                        data_test[i+1], data_test[i+2], data_test[i+3],
                        data_test[i+4], data_test[i+5], data_test[i+6], data_test[i+7]), axis=0)) 
                data_test = np.array(data_test_three)
                print("data_test in three:", data_test.shape)
            """

        if dataset == "lbs2d":
            # return data_test, flow_x_mean, flow_y_mean, flow_x_std, flow_y_std
            return data_test, dens_min, dens_max, dens_mean, dens_std
        else:
            return data_test, data_test_non_overlap