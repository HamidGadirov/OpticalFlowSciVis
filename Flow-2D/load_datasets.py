from curses.textpad import rectangle
from turtle import shape
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

from utils import visualize_ind, visualize_series, visualize_series_flow, visualize_large

from skimage.transform import rescale, resize, downscale_local_mean

# TODO:
# remove RGB +
# add augemntaion: flip +
# add more data from other datasets +-
# check patching -
# FluidSimML: downsample to 256x256? +
# try some optimization (TensorRT, ONNX) -

def load_data(dataset, exp, mode):
    combined_data_train = []
    combined_data_val = []
    if dataset == 'rectangle2d':
        datasets = ["rectangle2d"]
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
            flow_fln = "../Datasets/rectangle2d_hftext_flow.pkl"
            # flow_fln = "../Datasets/rectangles2d_text_flow.pkl"
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
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            print(data.shape)
            # flip flow directions
            # data[:, 1] *= -1
            # data[:, 2] *= -1
            # input("x")
        # if dataset == "droplet2d" or dataset == "cylinder2d" or dataset == "FluidSimML":
        else:
            with open(filename, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            print(data.shape)
            data = np.float32(data)
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            # data = data * 255.0
            # data = data.astype(int)
            data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
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
        if data.ndim == 3 or dataset == "rectangle2d":
            data = np.expand_dims(data, axis=-1)

        # convert to RGB for training
        # data = np.float32(data)
        # data_rgb = []
        # for i in range(data.shape[0]):
        #     data_rgb.append(cv2.cvtColor(data[i], cv2.COLOR_GRAY2RGB))
        # data = np.array(data_rgb)
        # data = np.moveaxis(data, -1, 1)
        # print("data in RGB:", data.shape)

        data = np.float32(data)
        data = np.moveaxis(data, -1, 1)
        print("data in grayscale:", data.shape)
        # input("x")

        if mode == "train":
            # print("a train subset of the member is selected")
            if "rectangle2d" in filename:
                data_train = data[:2205] # div to 3, 5, 9 and 7
                data_val = data[2370:2685] # div to 3, 5, 9 and 7
            if "droplet2d" in filename:
                data_train = data[:51300] # 46800 15120
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
                # prepare data for training - use only each third frame while shifting the sampling
                data_train_ = []
                data_val_ = []
                for shift in range(3):
                    for i in range(shift, data_train.shape[0], 3): # 10
                        data_train_.append(data_train[i])
                    for i in range(shift, data_val.shape[0], 3): 
                        data_val_.append(data_val[i])
                data_train = data_train_
                data_val = data_val_
                data_train = np.asarray(data_train)
                data_val = np.asarray(data_val)
                print(data_train.shape)
                print(data_val.shape)
                # input("x")

            # TODO: add other augment: diff steps for picking data ? 
            if "pipedcylinder2d" in filename or "cylinder2d" in filename or "FluidSimML" in filename or "rectangle2d" in filename:
                print("Augmenting the data...")
                data_train_flip = data_train[:,:,:,::-1]
                data_train = np.append(data_train, data_train_flip, axis=0)
                data_train_flip = data_train[:,:,::-1,:]
                data_train = np.append(data_train, data_train_flip, axis=0)

            print(data_train.shape)
            # input("x")

            # factor = 2
            # dir_res = "Results"
            # dir_res = os.path.join(dir_res, dataset)
            # dir_res = os.path.join(dir_res, str(factor) + "x")
            # print("Saving at:", dir_res)
            # title = title = "inference_" + str(factor) + "x"
            # visualize_series(data_train[600:,1,...], factor, dataset, dir_res, title=title, show=False, save=True)
            # input("x")

            # data_train = torch.from_numpy(data_train.copy()).permute(2, 0, 1)
            # data_val = torch.from_numpy(data_val.copy()).permute(2, 0, 1)
            # print(data_train.shape)

            # if dataset == "pipedcylinder2d":
            #     data_train = data_train[:, np.newaxis, ...]
            #     data_val = data_val[:, np.newaxis, ...]

            data_train_three = []
            data_val_three = []
            if exp == 1: # img0, img1, gt
                if dataset == "rectangle2d":
                    range_list = [3, 5, 7, 9]
                    # range_max = random.choice(range_list)
                    for n in range(len(range_list)):
                        range_max = range_list[n]
                        for i in range(0, data_train.shape[0], range_max): 
                            data_train_three.append(np.concatenate((data_train[i], 
                                data_train[i + range_max-1], data_train[i + int((range_max-1)/2)]), axis=0)) # img0, img1, gt
                        for i in range(0, data_val.shape[0], range_max): 
                            data_val_three.append(np.concatenate((data_val[i], 
                                data_val[i + range_max-1], data_val[i + int((range_max-1)/2)]), axis=0)) # img0, img1, gt
                    data_train = np.array(data_train_three)
                    print("data_test in three:", data_train.shape)
                    data_val = np.array(data_val_three)
                    print("data_val in three:", data_val.shape)
                    # input("x")
                else:
                    # prepare img0, gt, img1 (2x interpolation)
                    for i in range(0, data_train.shape[0], 3): 
                        data_train_three.append(np.concatenate((data_train[i], data_train[i+2], data_train[i+1]), axis=0)) # img0, img1, gt
                    data_train = np.array(data_train_three)
                    print("data_train in three:", data_train.shape)
                    for i in range(0, data_val.shape[0], 3): 
                        data_val_three.append(np.concatenate((data_val[i], data_val[i+2], data_val[i+1]), axis=0)) # img0, img1, gt
                    data_val = np.array(data_val_three)
                    print("data_val in three:", data_val.shape)
                # input("x")
            elif exp == 2:
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
            elif exp == 3:
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
        else:
            if "rectangle2d" in filename:
                data_test = data[2685:3000] # div to 3, 5, 9 and 7
            elif "droplet2d" in filename:
                data_test = data[:2700]
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

            if dataset != "rectangle2d":
                # prepare data for training - use only each third frame while shifting the sampling
                data_test_ = []
                for shift in range(3):
                    for i in range(shift, data_test.shape[0], 3): # 10
                        data_test_.append(data_test[i])
                data_test = data_test_
                data_test = np.asarray(data_test)

            print(data_test.shape)

            # if dataset == "pipedcylinder2d":
            #     data_test = data_test[:, np.newaxis, ...]

            data_test_three = []
            if exp == 1: # img0, img1, gt
                if dataset == "rectangle2d":
                    range_list = [3, 5, 7, 9]
                    # range_max = random.choice(range_list)
                    for n in range(len(range_list)):
                        range_max = range_list[n]
                        for i in range(0, data_test.shape[0], range_max): 
                            data_test_three.append(np.concatenate((data_test[i], 
                                data_test[i + range_max-1], data_test[i + int((range_max-1)/2)]), axis=0)) # img0, img1, gt
                    data_test = np.array(data_test_three)
                    print("data_test in three:", data_test.shape)
                    # input("x")
                else:
                    for i in range(0, data_test.shape[0], 3): 
                        data_test_three.append(np.concatenate((data_test[i], data_test[i+2], data_test[i+1]), axis=0)) # img0, img1, gt
                    data_test = np.array(data_test_three)
                    print("data_test in three:", data_test.shape)
                    # input("x")
            elif exp == 2:
                print("4x interpolation") 
                for i in range(0, data_test.shape[0], 5): 
                    data_test_three.append(np.concatenate((data_test[i], data_test[i+4],
                        data_test[i+1], data_test[i+2], data_test[i+3]), axis=0)) # img0, img1, gt0, gt1, gt2
                data_test = np.array(data_test_three)
                print("data_test in three:", data_test.shape)
            elif exp == 3:
                print("8x interpolation") # img0, img1, gt0, gt1, gt2, gt3, gt4, gt5, gt6
                for i in range(0, data_test.shape[0], 9):
                    # print(i)
                    data_test_three.append(np.concatenate((data_test[i], data_test[i+8],
                        data_test[i+1], data_test[i+2], data_test[i+3],
                        data_test[i+4], data_test[i+5], data_test[i+6], data_test[i+7]), axis=0)) 
                data_test = np.array(data_test_three)
                print("data_test in three:", data_test.shape)
        
            return data_test