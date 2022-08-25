import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import plot_loss, visualize_ind, visualize_series, visualize_series_flow, visualize_large

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose3d(in_channels=in_planes, out_channels=out_planes, 
                                kernel_size=kernel_size, stride=stride, padding=padding),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

version = 2

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        print("IFBlock init")
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential( # downscale x4; c=128
            conv(in_planes, c//2, 4, 2, 1), # stride=2 -> donwscale
            conv(c//2, c, 4, 2, 1), # # stride=2 -> donwscale  c=128 now
            )
        if version == 1:
            self.convblock = nn.Sequential(
                conv(c, c),
                conv(c, c),
                conv(c, c),
                conv(c, c),
                conv(c, c),
                conv(c, c),
                conv(c, c),
                conv(c, c),
            )
            self.lastconv = nn.ConvTranspose3d(c, 7, 4, 2, 1) # (c, 5, 4, 2, 1)
        if version == 2:
            self.convblock0 = nn.Sequential(
                conv(c, c),
                conv(c, c)
            )
            self.convblock1 = nn.Sequential(
                conv(c, c),
                conv(c, c)
            )
            self.convblock2 = nn.Sequential(
                conv(c, c),
                conv(c, c)
            )
            self.convblock3 = nn.Sequential(
                conv(c, c),
                conv(c, c)
            )
            self.conv1 = nn.Sequential(
                nn.ConvTranspose3d(c, c//2, 4, 2, 1),
                nn.PReLU(c//2),
                nn.ConvTranspose3d(c//2, 6, 4, 2, 1), # flow (c//2, 4, 4, 2, 1)
            )
            self.conv2 = nn.Sequential(
                nn.ConvTranspose3d(c, c//2, 4, 2, 1),
                nn.PReLU(c//2),
                nn.ConvTranspose3d(c//2, 1, 4, 2, 1), # mask
                # input(c//2)
            )

    def forward(self, x, flow, scale):
        # print("scale:", scale)
        if scale != 1:
            # input("here dim bug starts")
            print("before interpolate, ", x.shape)
            x = F.interpolate(x, scale_factor = 1. / scale, mode="trilinear", align_corners=False)
        if flow != None:
            print("flow != None ", flow.shape)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="trilinear", align_corners=False) * 1. / scale
            # print(flow.shape)
            # input("flow")
            x = torch.cat((x, flow), 1)
        print("IFBlock forward, ", x.shape)
        x = self.conv0(x)
        # print("conv0", x.shape)
        if version == 1:
            # print("conv0", x.shape)
            x = self.convblock(x) + x
            # print("convblock", x.shape)
            tmp = self.lastconv(x)
            # print("lastconv", tmp.shape)
            tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="trilinear", align_corners=False)
            # print("interpolate", tmp.shape)
            # input("tmp")
            flow = tmp[:, :6] * scale * 2
            mask = tmp[:, 6:7]
        if version == 2:
            # print(x.shape, self.convblock0(x).shape)
            x = self.convblock0(x) + x
            # print("convblock0", x.shape)
            x = self.convblock1(x) + x
            x = self.convblock2(x) + x
            x = self.convblock3(x) + x   
            # print("convblock3", x.shape)
            flow = self.conv1(x)
            # print("conv1", flow.shape)
            # input("flow")
            mask = self.conv2(x)
            flow = F.interpolate(flow, scale_factor=scale, mode="trilinear", align_corners=False, recompute_scale_factor=False) * scale
            mask = F.interpolate(mask, scale_factor=scale, mode="trilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask

class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(2, c=128) # 240
        self.block1 = IFBlock(5+6, c=64) # 150
        self.block2 = IFBlock(5+6, c=64) # 90
        self.block_tea = IFBlock(6+6, c=64) # 6 from cat, 6 from flow  c=90
        # modified
        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5): # scale=[4,2,1] scale=[1, 1, 1]
        img0 = x[:, :1] 
        img1 = x[:, 1:2]
        gt = x[:, 2:] # In inference time, gt is None
        # img0 = x[:, :1] 
        # img1 = x[:, 1:2]
        # gt = x[:, 2:] # In inference time, gt is None

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        # stu = [self.block0, self.block2]
        for i in range(3):
            if flow != None:
                # correct shapes
                max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
                max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
                max_shape_4 = min(img0.shape[4], warped_img0.shape[4])
                # print(max_shape_2, max_shape_3)
                img0 = img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                img1 = img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                mask = mask[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                flow = flow[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                # gt = gt[:,:,:max_shape_2,:max_shape_3]
                # print(i)
                # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow_d = flow_d[:,:,:img0.shape[2],:img0.shape[3],:img0.shape[4]]
                mask_d = mask_d[:,:,:img0.shape[2],:img0.shape[3],:img0.shape[4]]
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                # print(i)
                print("cat", torch.cat((img0, img1), 1).shape)
                # input("flow == None")
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i]) # stu[0]
                # print(flow, mask) # nan
                # input("flow")
            # correct shapes
            max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
            max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
            max_shape_4 = min(img0.shape[4], warped_img0.shape[4])
            flow = flow[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            mask = mask[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            img0 = img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            img1 = img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            # print("flow:", flow.shape)
            # input("flow")
            warped_img0 = warp(img0, flow[:, :3])
            warped_img1 = warp(img1, flow[:, 3:6])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)

            # debug
            # print("mask:", mask.shape)
            # print("flow:", flow.shape)
            # dir_res = "/home/hamid/Desktop/OpticalFlow/RIFE-3D/Results/rectangle3d/2x"
            # # "../Results/rectangle3d/2x"
            # visualize_ind(mask[0, ..., 50].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="rect3d_mask.png", save=True)
            # # input(flow.shape)
            # visualize_ind(warped_img0[0, ..., 50].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="warped_img0.png", save=True)
            # # visualize_ind(warped_img1[0, ..., 50].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="warped_img1.png", save=True)
            # visualize_ind(flow[0, 0, ..., 50].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="flow.png", save=True)

            print("gt:", gt.shape[1])
            # input("gt")
            
        if gt.shape[1] == 1: # 3
            # input("gt")
            print("gt.shape[1] = 1")
            print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
            # correct shapes
            max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
            max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
            max_shape_4 = min(img0.shape[4], warped_img0.shape[4])
            img0 = img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            img1 = img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            mask = mask[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            flow = flow[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            gt = gt[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
            print("flow:", flow.shape)

            print("before block tea", torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1).shape)
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_d = flow_d[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            mask_d = mask_d[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            
            flow_teacher = flow + flow_d
            warped_img0_teacher = warp(img0, flow_teacher[:, :3])
            warped_img1_teacher = warp(img1, flow_teacher[:, 3:6])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            print("flow_teacher = None")
            flow_teacher = None
            merged_teacher = None
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            merged[i] = merged[i][:,:,:gt.shape[2],:gt.shape[3],:gt.shape[4]]

            # torch.cuda.list_gpu_processes()
            # torch.cuda.memory_allocated()
            # import gc
            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             print(type(obj), obj.size())
            #     except:
            #         pass
            # input("x")
            
            if gt.shape[1] == 1: # 3
                flow_list[i] = flow_list[i][:,:,:flow_teacher.shape[2],:flow_teacher.shape[3],:flow_teacher.shape[4]]

                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                # this is 0 tensor
                # loss_distill += ((flow_teacher.detach() - flow_list[i]).abs() * loss_mask).mean()
                # print(loss_distill)
                # print(flow_teacher.detach())
                # print(flow_list[i])
                # print(np.mean(loss_mask.detach().cpu().numpy()))
                # input("loss_mask")
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
                # loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5).mean()
                # print( (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5).mean() )
                # print(loss_mask.mean())
                # input("loss_distill")

        print("in IFNet, forward")
        # c0 = self.contextnet(img0, flow[:, :2])
        # c1 = self.contextnet(img1, flow[:, 2:4])
        # tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        # res = tmp[:, :3] * 2 - 1
        # merged[2] = torch.clamp(merged[2] + res, 0, 1)
        # print("refined")
        return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill # mask_list[2]
