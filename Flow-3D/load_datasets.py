from turtle import shape
from matplotlib.pyplot import axes, title
# from dataset import *
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
from skimage.transform import rescale, resize, downscale_local_mean

import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils import visualize_ind, visualize_series, visualize_series_flow, visualize_large, visualize_3d

# TODO:
# normalize +-
# downsample +-
# 16777216
# 6480000

def load_data(dataset, mode):
    combined_data_train = []
    combined_data_val = []
    if dataset == 'rectangle3d':
        datasets = ["rectangle3d"]
    if dataset == 'jets3d':
        datasets = ["jets3d"]
    if dataset == 'droplet3d':
        datasets = ["droplet3d"]
    elif dataset == 'tangaroa3d':
        datasets = ["tangaroa3d"]
    elif dataset == 'tornado3d':
        datasets = ["tornado3d"]
    
    for dataset in datasets:
        print(dataset)
        filename = "../Datasets/"
        if dataset == 'rectangle3d':
            filename += "rectangle3d.pkl"
            flow_fln = "../Datasets/rectangle3d_big_f_bb_flow.pkl"
        if dataset == 'jets3d':
            filename = "../../../DataSSD/Datasets/5Jets/5jets3d_flow.pkl"
            flow_fln = "../../../DataSSD/Datasets/5Jets/5jets3d_flow.pkl"
            pkl_train_filename = "../../../DataSSD/Datasets/5Jets/5jets3d_flow_aug_train.pkl"
            pkl_val_filename = "../../../DataSSD/Datasets/5Jets/5jets3d_flow_aug_val.pkl"
            pkl_test_filename = "../../../DataSSD/Datasets/5Jets/5jets3d_flow_aug_test.pkl"
        if dataset == "droplet3d":
            filename_train = filename + "droplet3d_64_train.pkl"
            filename_val = filename + "droplet3d_64_val.pkl"
        elif dataset == "tangaroa3d":
            filename += "tangaroa3d_downs.pkl"
        elif dataset == 'tornado3d':
            filename += "tornado3d.pkl"

        if dataset == "droplet3d":
            data_train = []
            pkl_file = open(filename_train, 'rb')
            data_train = pickle.load(pkl_file)
            print(data_train.shape)
            data_val = []
            pkl_file = open(filename_val, 'rb')
            data_val = pickle.load(pkl_file)
            print(data_val.shape)
            print("Data is in range %f to %f" % (np.min(data_train[0]), np.max(data_train[0])))

            data_train = np.float32(data_train)
            data_val = np.float32(data_val)
            # data_train = np.float16(data_train)
            # data_val = np.float16(data_val)
            
            # data_train_down = []
            # for i in range(data_train.shape[0]):
            #     for j in range(data_train.shape[1]):
            #         data_train_down.append(downscale_local_mean(data_train[i,j,...], (2, 2, 2)))
            # data_train_down = np.array(data_train_down)
            # print(data_train_down.shape)
            # input("x")
        # else:
        #     data = []
        #     pkl_file = open(filename, 'rb')
        #     data = pickle.load(pkl_file)
        #     print(data.shape)
        #     print("Data is in range %f to %f" % (np.min(data[0]), np.max(data[0])))

        #     if not np.isfinite(data).all():
        #         data = np.nan_to_num(data)
        #         print(np.isfinite(data).all())
        #         print("Data is in range %f to %f" % (np.min(data[10]), np.max(data[10])))
        #         # input("x")

        #     # for i in range(data.shape[0]):
        #     #     data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
        #         # data[i] =  (data[i] - np.mean(data[i])) / np.std(data[i])

        #     data = np.float32(data)

        #     if data.ndim == 4: # expand dims for later concatenating
        #         data = np.expand_dims(data, axis=1)

        # load data
        data = []
        if dataset == "rectangle3d":
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
            # data[:, 3] = cv2.normalize(data[:, 3], data[:, 3], -1., 1., cv2.NORM_MINMAX)
            print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
            print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
            print("Vel z range %f to %f" % (np.min(data[:, 3]), np.max(data[:, 3])))
            # input("Data range")
        if dataset == "jets3d":
            # just load data, all steps were done in data creation (read_data.py) to save time
            # with open(flow_fln, 'rb') as flow_file:
            #     data = pickle.load(flow_file)
            with open(pkl_train_filename, 'rb') as flow_file:
                data_train = pickle.load(flow_file)
            with open(pkl_val_filename, 'rb') as flow_file:
                data_val = pickle.load(flow_file)
            with open(pkl_test_filename, 'rb') as flow_file:
                data_test = pickle.load(flow_file)
            
            # # data = np.float32(data)
            # print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            # print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            # print(data.shape)
            # # input("Data range")
            # data[:, 0] = cv2.normalize(data[:, 0], data[:, 0], 0., 1., cv2.NORM_MINMAX)
            # print("Density scaled to [0, 1] range!")
            # data[:, 1] = cv2.normalize(data[:, 1], data[:, 1], 0., 1., cv2.NORM_MINMAX)
            # data[:, 2] = cv2.normalize(data[:, 2], data[:, 2], 0., 1., cv2.NORM_MINMAX)
            # data[:, 3] = cv2.normalize(data[:, 3], data[:, 3], 0., 1., cv2.NORM_MINMAX)
            # print("Velocities scaled to [0, 1] range!")
            # print("Density range %f to %f" % (np.min(data[:, 0]), np.max(data[:, 0])))
            # print("Vel x range %f to %f" % (np.min(data[:, 1]), np.max(data[:, 1])))
            # print("Vel y range %f to %f" % (np.min(data[:, 2]), np.max(data[:, 2])))
            # print("Vel z range %f to %f" % (np.min(data[:, 3]), np.max(data[:, 3])))
            # # input("Data range")
        
        if dataset == "rectangle3d": # or dataset == "jets3d": # expand dims for later concatenating
            data = np.expand_dims(data, axis=1)
        
        # if dataset == "jets3d":
        #     data = np.float16(data)
        # data = np.moveaxis(data, -1, 1)
        # print("data in grayscale:", data.shape)
        # input("x")

        # add aug

        # print(data[1])
        # print("Data is in range %f to %f" % (np.min(data[0]), np.max(data[0])))
        # # data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # # data = data * 255.0
        # # data = data.astype(int)
        # data = cv2.normalize(data,  data, 0, 255, cv2.NORM_MINMAX)
        # # print(data)
        # print("Data is in range %f to %f" % (np.min(data[0]), np.max(data[0])))

        # print(np.mean(data[0], axis=0), np.mean(data[0], axis=1), np.mean(data[0], axis=2))
        # print(np.mean(data[0]), np.std(data[0]))

        # print(np.mean(data[0]), np.std(data[0]))

        # print("Data is in range %f to %f" % (np.min(data[0]), np.max(data[0])))

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

        # print(np.isfinite(data).all())
        # data = np.nan_to_num(data)
        # print(np.isfinite(data).all())
        # print("Data is in range %f to %f" % (np.min(data[10]), np.max(data[10])))
        # input("x")

        if mode == "train":
            if "rectangle3d" in filename:
                data_train = data[:750] #
                data_val = data[750:870] #
            # if "jets3d" in filename:
            #     data_train = data[:300] #
            #     data_val = data[300:372] #
            if "tangaroa3d" in filename: # (201, 120, 180, 300)
                data_train = data[:150] #
                data_val = data[150:201] #
            elif "tornado3d" in filename: # (128, 128, 128)
                data_train = data 
                data_val = data

            if dataset == "rectangle3d" or dataset == "droplet3d": # or dataset == "jets3d" 
                print("before augmented:", data_train.shape)
                data_train = np.append(data_train, data_train[:,:,:,::-1,:], axis=0)
                data_train = np.append(data_train, data_train[:,:,::-1,:,:], axis=0)

                print("augmented:", data_train.shape)
                data_train = np.float16(data_train)
            # input("augmented")

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

            # prepare img0, gt, img1
            data_train_three = []
            data_val_three = []
            if "tangaroa3d" in filename or "rectangle3d" in filename: # or "jets3d" in filename:
                # data_train = np.array(data_train)
                for i in range(0, data_train.shape[0], 3): 
                    data_train_three.append(np.concatenate((data_train[i], data_train[i+2], data_train[i+1]), axis=0)) # img0, img1, gt
                data_train = np.array(data_train_three)
                print("data_train in three:", data_train.shape)
                # data_train = np.array(data_train)
                for i in range(0, data_val.shape[0], 3): 
                    data_val_three.append(np.concatenate((data_val[i], data_val[i+2], data_val[i+1]), axis=0)) # img0, img1, gt
                data_val = np.array(data_val_three)
                print("data_val in three:", data_val.shape)
                # input("x")

            combined_data_train = torch.utils.data.ConcatDataset([combined_data_train, data_train])
            combined_data_val = torch.utils.data.ConcatDataset([combined_data_val, data_val])
        
            return combined_data_train, combined_data_val
            # return data_train, data_val
        else:
            if "tangaroa3d" in filename:
                data_test = data[150:201]

                # if "pipedcylinder2d" in filename or "cylinder2d" in filename or "droplet2d" in filename:
                # prepare data for training - use only each third frame ?
                # data_test_ = []
                # for i in range(0, data_test.shape[0], 3): # 10
                #     data_test_.append(data_test[i])
                # data_test = data_test_
                # data_test = np.asarray(data_test)

                # prepare img0, gt, img1
                data_test_three = []
                # data_train = np.array(data_train)
                for i in range(0, data_test.shape[0], 3): 
                    data_test_three.append(np.concatenate((data_test[i], data_test[i+2], data_test[i+1]), axis=0)) # img0, img1, gt
                data_test = np.array(data_test_three)
                print("data_test in three:", data_test.shape)

            if dataset == "droplet3d":
                data_test = data_val

            if "rectangle3d" in filename:
                data_test = data[870:990] #
            # if "jets3d" in filename:
            #     data_test = data[300:372] # same as val for now

            #     # prepare img0, gt, img1
            #     data_test_three = []
            #     # data_train = np.array(data_train)
            #     for i in range(0, data_test.shape[0], 3): 
            #         data_test_three.append(np.concatenate((data_test[i], data_test[i+2], data_test[i+1]), axis=0)) # img0, img1, gt
            #     data_test = np.array(data_test_three)
            #     print("data_test in three:", data_test.shape)

                # visualize_3d(data_test, dataset, dir_res="Results", title="droplet3d.html")
                # input("3d")

            return data_test