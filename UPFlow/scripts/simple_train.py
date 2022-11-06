# -*- coding: utf-8 -*-
# conda activate upflow_new2

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils.tools import tools
import cv2
import numpy as np
from copy import deepcopy
import torch
import warnings  # ignore warnings
import torch.nn.functional as F
import torch.optim as optim
# from dataset.kitti_dataset_2012 import kitti_train, kitti_flow
# from dataset.kitti_dataset import kitti_train, kitti_flow # kitti
# from dataset.scivis_datasets import kitti_train, kitti_flow # our data
from model.upflow import UPFlow_net
from torch.utils.data import DataLoader
import time
import argparse
import pickle
# import tensorflow as tf
# tf.compat.v1.enable_eager_execution

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir) 
# sys.path.append('../../FlowSciVis/Flow2D/')

# sys.path.insert(1, '../../FlowSciVis/Flow-2D')
# sys.path.insert(2, '../../FlowSciVis/Datasets')
# from load_datasets import load_data

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
device = torch.device("cuda")

''' scripts for training：
1. simply using photo loss and smooth loss
2. add occlusion checking
3. add teacher-student loss(ARFlow)
'''

# save and log loss value during training
class Loss_manager():
    def __init__(self):
        self.error_meter = tools.Avg_meter_ls()

    def fetch_loss(self, loss, loss_dict, name, batch_N, short_name=None):
        if name not in loss_dict.keys():
            pass
        elif loss_dict[name] is None:
            pass
        else:
            this_loss = loss_dict[name].mean()
            self.error_meter.update(name=name, val=this_loss.item(), num=batch_N, short_name=short_name)
            loss = loss + this_loss
        return loss

    def prepare_epoch(self):
        self.error_meter.reset()

    def log_info(self):
        p_str = self.error_meter.print_all_losses()
        return p_str

    def compute_loss(self, loss_dict, batch_N):
        loss = 0
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='photo_loss', short_name='ph', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='smooth_loss', short_name='sm', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='census_loss', short_name='cen', batch_N=batch_N)
        # photo_loss, smooth_loss, census_loss = output_dict['photo_loss'].mean(), output_dict['smooth_loss'], output_dict['census_loss']
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='msd_loss', short_name='msd', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='eq_loss', short_name='eq', batch_N=batch_N)
        loss = self.fetch_loss(loss=loss, loss_dict=loss_dict, name='oi_loss', short_name='oi', batch_N=batch_N)
        return loss

class Eval_model(tools.abs_test_model):
    def __init__(self):
        super(Eval_model, self).__init__()
        self.net_work = None

    def eval_forward(self, im1, im2, gt, *args):
        if self.net_work is None:
            raise ValueError('not network for evaluation')
        # === network output
        print("Eval_model, eval_forward")
        # input("x")
        with torch.no_grad():
            input_dict = {'im1': im1, 'im2': im2, 'if_loss': False}
            output_dict = self.net_work(input_dict)
            flow_fw, flow_bw = output_dict['flow_f_out'], output_dict['flow_b_out']
            pred_flow = flow_fw
        return pred_flow

    def eval_save_result(self, save_name, predflow, *args, **kwargs):
        # you can save flow results here
        # print(save_name)
        pass

    def change_model(self, net):
        net.eval()
        self.net_work = net


class Trainer():
    class Config(tools.abstract_config):
        def __init__(self, **kwargs):
            self.exp_dir = './demo_exp'
            self.if_cuda = True

            self.batchsize = 20
            self.NUM_WORKERS = 8 # 4
            self.n_epoch = 1000
            self.batch_per_epoch = 500
            self.batch_per_print = 20
            self.lr = 1e-4
            self.weight_decay = 1e-4
            self.scheduler_gamma = 1

            # init
            self.update(kwargs)

        def __call__(self, dataset):
            t = Trainer(self, dataset)
            return t

    def __init__(self, conf: Config, dataset):
        self.conf = conf

        tools.check_dir(self.conf.exp_dir)

        # load network
        self.net = self.load_model()

        # for evaluation
        # self.bench = self.load_eval_bench()
        self.eval_model = Eval_model()


        print("in Trainer init")
        input("loading...")
        # load training dataset
        # self.train_set = self.load_training_dataset() # kitti
        self.train_set = self.load_scivis_training_dataset(dataset)

        print("self.train_set:", type(self.train_set))
        # input("xxx")


    def training(self, dataset):
        print("in training")
        # print("self.train_set:", np.array(self.train_set['im1']).shape)
        print("self.train_set:", type(self.train_set))
        train_loader = tools.data_prefetcher(self.train_set, batch_size=self.conf.batchsize, shuffle=True, num_workers=self.conf.NUM_WORKERS, pin_memory=True, drop_last=True)
        optimizer = optim.Adam(self.net.parameters(), lr=self.conf.lr, amsgrad=True, weight_decay=self.conf.weight_decay)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.conf.scheduler_gamma)
        loss_manager = Loss_manager()
        timer = tools.time_clock()
        print("start training" + '=' * 10)
        i_batch = 0
        epoch = 0
        loss_manager.prepare_epoch()
        current_val, best_val, best_epoch = 0, 0, 0
        timer.start()
        while True:
            # prepare batch 
            # data
            print("train_loader", type(train_loader))
            input("prepare batch data")
            batch_value = train_loader.next()
            # batch_value_ = np.array(batch_value)
            # print("batch_value_", batch_value_.shape)
            print("batch_value", type(batch_value))
            input("batch_value")
            if batch_value is None:
                batch_value = train_loader.next()
                assert batch_value is not None
            # batchsize = batch_value['im1'].shape[0] # check if the im1 exists
            batchsize = 20 # batch_value.shape[0] # check if the im1 exists
            print("batchsize", batchsize)
            i_batch += 1
            # train batch
            self.net.train()
            optimizer.zero_grad()
            out_data = self.net(batch_value)  #

            loss_dict = out_data['loss_dict']
            loss = loss_manager.compute_loss(loss_dict=loss_dict, batch_N=batchsize)

            loss.backward()
            optimizer.step()
            if i_batch % self.conf.batch_per_print == 0:
                pass
            if i_batch % self.conf.batch_per_epoch == 0:
                # do eval  and check if save model todo===
                epoch+=1
                timer.end()
                print(' === epoch use time %.2f' % timer.get_during())
                scheduler.step(epoch=epoch)
                timer.start()

    def evaluation(self):
        self.eval_model.change_model(self.net)
        epe_all, f1, epe_noc, epe_occ = self.bench(self.eval_model)
        print('EPE All = %.2f, F1 = %.2f, EPE Noc = %.2f, EPE Occ = %.2f' % (epe_all, f1, epe_noc, epe_occ))
        print_str = 'EPE_%.2f__F1_%.2f__Noc_%.2f__Occ_%.2f' % (epe_all, f1, epe_noc, epe_occ)
        return epe_all, print_str

    # ======
    def load_model(self):
        param_dict = {
            # use cost volume norm
            'if_norm_before_cost_volume': True,
            'norm_moments_across_channels': False,
            'norm_moments_across_images': False,
            'if_froze_pwc': False,
            'if_use_cor_pytorch': False,  # speed is very slow, just for debug when cuda correlation is not compiled
            'if_sgu_upsample': False,  # 先把这个关掉跑通吧
        }
        pretrain_path = None  # pretrain path
        net_conf = UPFlow_net.config()
        net_conf.update(param_dict)
        net = net_conf()  # .cuda()
        if pretrain_path is not None:
            net.load_model(pretrain_path, if_relax=True, if_print=False)
        if self.conf.if_cuda:
            net = net.cuda()
        return net

    # def load_eval_bench(self):
        # bench = kitti_flow.Evaluation_bench(name='2015_train', if_gpu=self.conf.if_cuda, batch_size=1)
        # bench = kitti_flow.Evaluation_bench(name='2012_train', if_gpu=self.conf.if_cuda, batch_size=1)
        # return bench

    # def load_training_dataset(self):
    #     print("in load_training_dataset")
    #     data_config = {
    #         'crop_size': (256, 832),
    #         'rho': 8,
    #         'swap_images': True,
    #         'normalize': True,
    #         'horizontal_flip_aug': True,
    #     }
    #     data_conf = kitti_train.kitti_data_with_start_point.config(mv_type='2015', **data_config)
    #     # data_conf = kitti_train.kitti_data_with_start_point.config(mv_type='2012', **data_config)
    #     print("in load_training_dataset")
    #     dataset = data_conf()
    #     print("in load_training_dataset")
    #     return dataset

    def load_scivis_training_dataset(self, dataset):
        filename = "../../FlowSciVis/Datasets/"
        if dataset == 'rectangle2d':
            filename += "rectangle2d.pkl"
            flow_fln = "../../FlowSciVis/Datasets/rectangle2d_hftext_flow.pkl"
            # flow_fln = "../FlowSciVis/Datasets/rectangle2d_flow.pkl"
        if dataset == "droplet2d":
            filename += "drop2D/droplet2d_test.pkl"
        elif dataset == "pipedcylinder2d":
            filename += "pipedcylinder2d.pkl"
            flow_fln = "../../FlowSciVis/Datasets/pipedcylinder2d_flow.pkl"
        elif dataset == "cylinder2d":
            filename += "cylinder2d_nc/cylinder2d.pkl"
            flow_fln = "../../Datasets/cylinder2d_nc/cylinder2d_flow.pkl"
        elif dataset == "FluidSimML2d":
            filename += "FluidSimML/FluidSimML_1000_downs_data.pkl" # FluidSimML_1000
            flow_fln = "../../Datasets/FluidSimML/FluidSimML_1000_downs_flow.pkl"

        # load data
        data = []
        if dataset == "rectangle2d":
            with open(flow_fln, 'rb') as flow_file:
                data = pickle.load(flow_file)
            data = np.float32(data)
            data = data[:, 0:1] # only data
            data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            print("rectangle2d data:", data.shape)

            im1 = []
            im2 = []
            for i in range(100):
                im1_np = data[i * 3]
                im2_np = data[i * 3 + 2]
                # print(im1_np.shape, im2_np.shape)

                # to rgb
                im1_np = np.concatenate((im1_np, im1_np, im1_np), axis=0)
                im2_np = np.concatenate((im2_np, im2_np, im2_np), axis=0)
                # print(im1_np.shape)

                im1.append(im1_np)
                im2.append(im2_np)
                
            data_dict = {'im1': im1, 'im2': im2, 'if_loss': False}
            print(np.array(data_dict['im1']).shape)
            print(np.array(data_dict['im2']).shape)
            print("created data dictionary")
            data = data_dict
            # data = data_dict['im2']
            # data = im1, im2
        else:
            with open(filename, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
            print("else data:", data.shape)
            data = np.float32(data)
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            # data = data * 255.0
            # data = data.astype(int)
            # data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
            data = cv2.normalize(data, data, 0., 1., cv2.NORM_MINMAX)
            # print(data)
            print("Data is in range %f to %f" % (np.min(data), np.max(data)))
                # input("x")
        if dataset == "pipedcylinder2d":
            data = data[:, np.newaxis, ...]
        if "pipedcylinder2d" in filename or "cylinder2d" in filename: # 1501 in total
                data_train = data[:540] # 1080 div to 27 and 5
                data_train = np.append(data_train, data[-540:], axis=0)
                data_val = data[540:810] # 1008:1296
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
        if "pipedcylinder2d" in dataset or "cylinder2d" in dataset or "FluidSimML" in dataset or "rectangle2d" in dataset:
                print("Augmenting the data...")
                data_train_flip = data_train[..., ::-1] # [:,:,:,::-1] wrong!
                data_train = np.append(data_train, data_train_flip, axis=0)
                data_train_flip = data_train[..., ::-1, :]  # [:,:,::-1,:] wrong!
                data_train = np.append(data_train, data_train_flip, axis=0)
                print(data_train.shape)

        data_train_three = []
        data_val_three = []
        # prepare img0, gt, img1 (2x interpolation)
        for i in range(0, data_train.shape[0], 3): 
            data_train_three.append(np.concatenate((data_train[i], data_train[i+2], data_train[i+1]), axis=0)) # img0, img1, gt
        data_train = np.array(data_train_three)
        print("data_train in three:", data_train.shape)
        for i in range(0, data_val.shape[0], 3): 
            data_val_three.append(np.concatenate((data_val[i], data_val[i+2], data_val[i+1]), axis=0)) # img0, img1, gt
        data_val = np.array(data_val_three)
        print("data_val in three:", data_val.shape)

        # load from FlowSciVis loader
        # exp = 1
        # mode = "train"
        # data_train, data_val = load_data(dataset, exp, mode)
        data = data_train

        # print("load_scivis_training_dataset, data:", data.shape)
        # print("Data is in range %f to %f" % (np.min(data), np.max(data)))
        # input("x")
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    args = parser.parse_args()
    # assert (not args.dataset is None)
    if args.dataset:
        print(args.dataset)

    training_param = {}  # change param here
    # print("x")
    conf = Trainer.Config(**training_param)
    # print("xx")
    trainer = conf(args.dataset)
    # print("xxx")

    trainer.training(args.dataset)
