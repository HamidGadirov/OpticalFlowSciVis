# ssh -Y hamid@129.125.75.167
# python3 -m torch.distributed.launch --nproc_per_node=1 train.py --world_size=1 --dataset=droplet3d --mode=train

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

from model.RIFE import Model
# from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from load_datasets import load_data
from error import calculate_diff
from utils import plot_loss, visualize_ind, visualize_series, visualize_series_flow, visualize_large_3d, visualize_3d

# If your model is returning NaNs, you could set 
# torch.autograd.detect_anomaly(True) 
# at the beginning of your script to get a stack trace, which would hopefully point to the operation, which is creating the NaNs.

# TODO:
# implement smooth loss +
# implement photometric loss +
# implement laplacian pyramid loss +
# merge with 2D model -
# fix bug with 3rd dim -

device = torch.device("cuda")

# exp = os.path.abspath('.').split('/')[-1]
log_path = 'train_log'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-5) * mul + 3e-5

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

early_stop_patience = 1000
early_stop_k = 0
val_loss_best = 0.

def train(model, dataset, model_name, mode, local_rank):
    # if local_rank == 0:
    #     writer = SummaryWriter('train')
    #     writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0

    if mode == "train":
        data_train, data_val = load_data(dataset, mode)

        # sampler = DistributedSampler(data_train)
        # train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
        train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
        args.step_per_epoch = train_data.__len__()
        # input("train loaded")
        # print(len(data_train))
        # print(data_train[0].shape)
        val_data = DataLoader(data_val, batch_size=args.batch_size, pin_memory=True, num_workers=8)

        # print(len(train_data.dataset))

        # for i, data in enumerate(train_data):
        #     print(i)
    else:
        data_test = load_data(dataset, mode)

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
        model.load_model(model_name, log_path) # won't work - we changed loss and pyramid
        print("Loaded RIFE model.")
        # print(model) # RIFE is not a neural net
    except:
        print("No weights found, training from scratch.")
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
                print("data to gpu:", data_gpu.shape)
                # print(type(data_gpu[0,0,0,0,0]))
                # data_gpu = data_gpu.float16()
                # input("x")
                # input("data")
                imgs = data_gpu[:, :2] # :6
                gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network
                # print("Training")
                print(imgs.shape, gt.shape)
                # input("x")
                imgs = imgs.permute(0, 1, 4, 2, 3) # 3rd is dim0
                gt = gt.permute(0, 1, 4, 2, 3)
                # print("permuted:", imgs.shape, gt.shape)

                # g_t = gt.detach().cpu().numpy()
                # print(np.mean(imgs.detach().cpu().numpy()))
                # print(np.mean(gt.detach().cpu().numpy()))
                # print(g_t.shape)
                # input("imgs gt before iter")

                # learning_rate = get_learning_rate(step)
                learning_rate = get_learning_rate(step) * args.world_size / 4
                # with torch.autograd.detect_anomaly():
                pred, info = model.update(imgs, gt, learning_rate, training=True)
                
                pred = pred.permute(0, 1, 3, 4, 2)
                gt = gt.permute(0, 1, 3, 4, 2)
                info['mask'] = info['mask'].permute(0, 1, 3, 4, 2)
                info['mask_tea'] = info['mask_tea'].permute(0, 1, 3, 4, 2)
                info['merged_tea'] = info['merged_tea'].permute(0, 1, 3, 4, 2)
                info['flow'] = info['flow'].permute(0, 1, 3, 4, 2)
                info['flow_tea'] = info['flow_tea'].permute(0, 1, 3, 4, 2)
                # print("permute back:", pred.shape, gt.shape)

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
            evaluate(model, dataset, val_data, step, local_rank) #, writer_val)
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
        # input("x")
        # data = data_train
        scale_list = [4, 2, 1]
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
        original_data = []
        flow_combined = []
        for i, data in enumerate(data_test):
            print("data:", data.shape)
            # input("x")
            data = np.expand_dims(data, axis=0)
            data_gpu = torch.from_numpy(data).to(device, non_blocking=True) # / 255.
            print("data to gpu:", data_gpu.shape)
            # input("data")
            imgs = data_gpu[:, :2] # :6
            gt = data_gpu[:, 2:3] # 6:9 this is GT for teacher network
            # print("Training")
            print(imgs.shape, gt.shape)
            imgs = imgs.permute(0, 1, 4, 2, 3) # 3rd is dim0
            gt = gt.permute(0, 1, 4, 2, 3)
            # print("permuted:", imgs.shape, gt.shape)
            # input("x")
            I0 = imgs[:, :1]
            I1 = imgs[:, 1:2]
            middle, flow, mask = model.inference(I0, I1, scale_list) # get the flow too

            middle = middle.permute(0, 1, 3, 4, 2)
            gt = gt.permute(0, 1, 3, 4, 2)
            flow[2] = flow[2].permute(0, 1, 3, 4, 2)
            mask = mask.permute(0, 1, 3, 4, 2)
            # print("permute back:", middle.shape, flow.shape)

            middle = np.asarray(middle.detach().cpu().numpy())
            interpol_data.append(middle.transpose((0, 2, 3, 4, 1)))
            gt = np.asarray(gt.detach().cpu().numpy())
            original_data.append(gt.transpose((0, 2, 3, 4, 1)))
            print(middle.shape)
            flow = np.asarray(flow)
            flow_array_ = flow[2].detach().cpu().numpy()
            flow_combined.append(flow_array_[:,0:6,:,:].squeeze()) # ? 0:6 because in x, y, z and for F_t->0, F_t->1 intermediate flows
            print(flow.shape)
            mask = np.asarray(mask.detach().cpu().numpy())
            print(mask.shape)

        interpol_data = np.squeeze(np.array(interpol_data))
        # interpol_data = np.array(interpol_data)
        # interpol_data = interpol_data.reshape(17, 120, 180, 300, 1)
        print("Interpolated:", interpol_data.shape)
        slice_num = 40
        # print(interpol_data[:, slice_num, :, :])

        original_data = np.squeeze(original_data)
        print("gt:", original_data.shape)
        # print("gt:",original_data[:, slice_num, :, :])
        title = title = "inference_" + str(factor) + "x"
        if "2d" in dataset:
            visualize_series(interpol_data, factor, dataset, dir_res, title=title, show=False, save=True)
        if "3d" in dataset:
            visualize_series(interpol_data[..., slice_num], factor, dataset, dir_res, title=title, show=False, save=True)
            visualize_series(original_data[..., slice_num], factor, dataset, dir_res, title=title+"GT", show=False, save=True)

        print(interpol_data[0].mean(axis=1))
        print(original_data[0].mean(axis=1))

        # Interpolated: (300, 160, 224)                                                                             
        # Flow: (300, 4, 160, 224) 

        # Vector field quiver plot
        flow_combined = np.array(flow_combined)
        print("Flow:", flow_combined.shape)
        # input("infer")
        flow_u = flow_combined[:, 0, :, :, slice_num]
        flow_v = flow_combined[:, 1, :, :, slice_num]
        title = "inference_flow_arrows_" + str(factor) + "x"
        if "2d" in dataset:
            visualize_series_flow(interpol_data, flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)
        if "3d" in dataset:
            visualize_series_flow(interpol_data[..., slice_num], flow_u, flow_v, dataset, dir_res, title=title, show=False, save=True)

        loss_path = 'loss.json'
        loss_path = os.path.join(dir_model, loss_path)

        with open(loss_path, 'r') as loss_file:
            loss_data = json.load(loss_file)
        val_loss = loss_data['val_loss']
        loss_file.close()   

        from collections import Iterable
        def flatten(lis):
            for item in lis:
                if isinstance(item, Iterable) and not isinstance(item, str):
                    for x in flatten(item):
                        yield x
                else:        
                    yield item

        val_loss = list(flatten(val_loss))
        val_loss = np.array(val_loss)
        print(val_loss.shape)
        # input("x")

        plot_loss(val_loss, dir_model, name="val_loss.png", save=True)

        return data_test, interpol_data, flow_combined
    
def evaluate(model, dataset, val_data, nr_eval, local_rank): #, writer_val):
    print("Evaluate")
    # writer = SummaryWriter('train') #
    # writer_val = SummaryWriter('validate') #

    loss_l1_list = []
    loss_G_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu = data.to(device, non_blocking=True) # / 255.
        imgs = data_gpu[:, :2]
        gt = data_gpu[:, 2:3]
        imgs = imgs.permute(0, 1, 4, 2, 3) # 3rd is dim0
        gt = gt.permute(0, 1, 4, 2, 3)
        print("permuted:", imgs.shape, gt.shape)
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']

        pred = pred.permute(0, 1, 3, 4, 2)
        gt = gt.permute(0, 1, 3, 4, 2)
        merged_img = merged_img.permute(0, 1, 3, 4, 2)
        info['flow'] = info['flow'].permute(0, 1, 3, 4, 2)
        info['flow_tea'] = info['flow_tea'].permute(0, 1, 3, 4, 2)
        print("permute back:", pred.shape, info['flow'].shape)

        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        loss_G_list.append(info['loss_G'].cpu().numpy())
        for j in range(gt.shape[0]):
            max_shape_2 = min(gt.shape[2], pred.shape[2])
            max_shape_3 = min(gt.shape[3], pred.shape[3])
            max_shape_4 = min(gt.shape[4], pred.shape[4])
            # print(max_shape_2, max_shape_3)
            gt = gt[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            pred = pred[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 4, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 4, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 4, 1).cpu().numpy()
        # if i == 0 and local_rank == 0:
        #     for j in range(10):
        #         imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
        #         writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')

    # early stopping
    val_loss_now = info['loss_G']
    global val_loss_best 
    global early_stop_k
    if val_loss_now > val_loss_best:
        early_stop_k += 1
        if early_stop_k == early_stop_patience:
            print("Early stopping")
            sys.exit(0)
    else:
        early_stop_k = 0
        val_loss_best = val_loss_now
        model.save_model(model_name, log_path, local_rank) 

    val_loss = []
    val_loss.append(float(np.array(loss_G_list).mean()))
    # val_loss = val_loss.item().tolist()
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
        loss_data_old['val_loss'].append(val_loss)
        loss_data = loss_data_old
        # print("exists:", loss_data)
        loss_file.close()

    # print(loss_data)
    # input("x")

    with open(loss_path, 'w') as loss_file:
        json.dump(loss_data, loss_file, indent=4)
        # print("dump loss to json")
    loss_file.close()

        # train_loss = info['loss_G']
        # print(type(train_loss))
        # # train_loss = train_loss.detach().cpu().numpy()
        # train_loss = train_loss.item()
        # loss_path = 'loss.json'
        # factor = 2
        # dir_res = "Results"
        # dir_res = os.path.join(dir_res, dataset)
        # dir_res = os.path.join(dir_res, str(factor) + "x")
        # loss_path = os.path.join(dir_res, loss_path)
        # # print("Saving at:", dir_res)
        # # load previous loss values if they exist
        # # if (os.path.exists(loss_path)):
        # #     loss_file = open(loss_path, 'r')
        # #     loss_data = json.load(loss_file)
        # #     loss = loss_data['loss']
        # #     loss_file.close()

        # loss_file = open(loss_path, 'w')
        # loss_data = {'loss': train_loss, 'val_loss': train_loss}
        # print(type(loss_data))
        # json.dump(loss_data, loss_file, indent=4)
        # loss_file.close()

    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    # writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size') # default=16
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size') # 4
    parser.add_argument('--dataset', dest='dataset', type=str, default=None)
    parser.add_argument('--mode', dest='mode', type=str, default='test')
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
    if args.dataset == "droplet3d":
        args.batch_size = 20
    if args.dataset == "rectangle3d":
        args.batch_size = 30
    if args.dataset == "tangaroa3d":
        args.batch_size = 15

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
    
    """ tangaroa """
    # model_name = "flownet_lapl_tangaroa3d.pkl"
    model_name = "flownet_l1_tangaroa3d.pkl"

    """ droplet """
    # model_name = "flownet_lapl_droplet3d.pkl"
    # model_name = "flownet_l1_droplet3d.pkl"
    # model_name = "flownet_l1_90_drop3d.pkl" # 0 distill
    # model_name = "flownet_l1_nodistill_90_drop3d.pkl" #
    # model_name = "flownet_l1_modif90_drop3d.pkl" #
    # model_name = "flownet_l1_64_drop3d.pkl" # not empty but bad
    # model_name = "flownet_l1_distill_64_drop3d.pkl" # not nan but bad
    # model_name = "flownet_l1_distill1_64_drop3d.pkl" # not nan but bad
    # model_name = "flownet_l1_distill2_64_drop3d.pkl" # not nan but bad
    # model_name = "flownet_l1_distill3_64_drop3d.pkl" # not nan but bad
    # model_name = "flownet_l1_distill_64_500ep_drop3d.pkl"
    # model_name = "flownet16_l1_64_drop3d.pkl" # nan loss

    # print(model_name)
    # input("x")

    dataset = args.dataset
    print("inference...")
    factor = 2
    dir_res = "Results"
    dir_res = os.path.join(dir_res, dataset)
    dir_res = os.path.join(dir_res, str(factor) + "x")
    dir_model = os.path.join(dir_res, model_name[:-4])
    # print(dir_model)
    # input("x")
    print("Saving at:", dir_model)

    # train(model, args.dataset, args.mode, args.local_rank)
    data_test, interpol_data, flow_combined = train(model, args.dataset, model_name, args.mode, args.local_rank)
    print(data_test.shape, interpol_data.shape, flow_combined.shape)
    # input("inf")
    # original_data = data_test[:,1,:interpol_data.shape[1], :interpol_data.shape[2]] # / 255.
    original_data = data_test[:,1,...]
    # interpol_data = interpol_data
    print(original_data.shape, interpol_data.shape, flow_combined.shape)
    print("original data is in range %f to %f" % (np.min(original_data), np.max(original_data)))
    print("interpol data is in range %f to %f" % (np.min(interpol_data), np.max(interpol_data)))
    # input("x")

    # print(original_data[0])
    # print(interpol_data[0])
    # input("x")

    slice_num = 40
    # calc diff, metrics
    # diffs = calculate_diff(original_data[..., slice_num], interpol_data[..., slice_num], dataset, factor, dir_model)
    # print("diffs:", diffs.shape)

    title = "GT_Interpol_" + str(factor) + "x"
    # visualize_large_3d(original_data[..., slice_num], interpol_data[..., slice_num], \
    #     # flow_combined[:, :4, :, :, slice_num], diffs, factor, dataset, dir_model, title=title, show=False, save=True)
    #     flow_combined[..., slice_num], diffs, factor, dataset, dir_model, title=title, show=False, save=True)
    visualize_3d(original_data, interpol_data, flow_combined, dataset, dir_model, title=title, show=False, save=True)
