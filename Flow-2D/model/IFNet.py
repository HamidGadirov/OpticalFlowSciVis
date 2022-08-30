from ensurepip import version
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
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

version = 2
five_blocks = False
blocks_range = 3 if not five_blocks else 5
refine = True

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        # print("IFBlock init")
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
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
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
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
                nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                nn.PReLU(c//2),
                nn.ConvTranspose2d(c//2, 4, 4, 2, 1), # flow
            )
            self.conv2 = nn.Sequential(
                nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                nn.PReLU(c//2),
                nn.ConvTranspose2d(c//2, 1, 4, 2, 1), # mask
                # input(c//2)
            )

    def forward(self, x, flow, scale):
        if scale != 1:
            # here was error
            # print("before interpolate, ", x.shape)
            # print(type(x))
            # x = x.type(torch.float32)
            x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        if flow != None:
            # print("flow != None ", flow.shape)
            flow = F.interpolate(flow, scale_factor = 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        # print("IFBlock forward, ", x.shape)
        x = self.conv0(x)
        # print("conv0:", x.shape)
        if version == 1:
            x = self.convblock(x) + x
            # print("convblock:", x.shape)
            # input("x")
            tmp = self.lastconv(x)
            tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False)
            flow = tmp[:, :4] * scale * 2
            mask = tmp[:, 4:5]
        if version == 2:
            x = self.convblock0(x) + x
            x = self.convblock1(x) + x
            x = self.convblock2(x) + x
            x = self.convblock3(x) + x  
            # print("convblock3", x.shape)
            flow = self.conv1(x)
            # print("conv1", flow.shape)
            # input("flow")
            mask = self.conv2(x)
            flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
            mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        # try reversing flow here
        # flow = - flow
        # mask = 1. - mask
        # flow *= 0. # that breaks
        # print("reversed mask!")
        return flow, mask
    
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(2, c=128) # 6 c=240 128
        self.block1 = IFBlock(5+4, c=96) # 13 c=150 96
        self.block2 = IFBlock(5+4, c=64) # 13 c=90 96 64

        if five_blocks:
            self.block0 = IFBlock(2, c=128)
            self.block1 = IFBlock(5+4, c=96)
            self.block2 = IFBlock(5+4, c=96)
            self.block3 = IFBlock(5+4, c=64)
            self.block4 = IFBlock(5+4, c=64)

        self.block_tea = IFBlock(6+4, c=64) # 16 from cat, 4 from flow, replaces by 6 because grayscale  c=90 96 64
        # modified
        if refine:
            self.contextnet = Contextnet()
            self.unet = Unet()

    def forward(self, x, scale=[4, 2, 1], timestep=0.5): # scale=[4,2,1] scale=[1, 1, 1]
        img0 = x[:, :1] # :3
        img1 = x[:, 1:2] # 3:6
        gt = x[:, 2:3] # 6:   In inference time, gt is None

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        if five_blocks:
            stu = [self.block0, self.block1, self.block2, self.block3, self.block4]
            scale = [4, 2, 2, 1, 1]
        for i in range(blocks_range):
            if flow != None:
                # correct shapes
                # print(max_shape_2, max_shape_3)
                max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
                max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
                img0 = img0[:,:,:max_shape_2,:max_shape_3]
                img1 = img1[:,:,:max_shape_2,:max_shape_3]
                warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3]
                warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3]
                mask = mask[:,:,:max_shape_2,:max_shape_3]
                flow = flow[:,:,:max_shape_2,:max_shape_3]
                # gt = gt[:,:,:max_shape_2,:max_shape_3]
                # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
                flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow_d = flow_d[:,:,:img0.shape[2],:img0.shape[3]]
                mask_d = mask_d[:,:,:img0.shape[2],:img0.shape[3]]
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                # print("cat", torch.cat((img0, img1), 1).shape)
                flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i])
            # correct shapes
            flow = flow[:,:,:img0.shape[2],:img0.shape[3]]
            mask = mask[:,:,:img0.shape[2],:img0.shape[3]]
            max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
            max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
            img0 = img0[:,:,:max_shape_2,:max_shape_3]
            img1 = img1[:,:,:max_shape_2,:max_shape_3]
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)

            # debug - this takes time and CPU memory
            # print("mask:", mask.shape)
            # print("flow:", flow.shape)
            # dir_res = "/home/hamid/Desktop/OpticalFlow/RIFE/Results/rectangle2d/2x"
            # visualize_ind(mask[0, ...].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="rect2d_mask.png", save=True)
            # visualize_ind(warped_img0[0, ...,].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="warped_img0.png", save=True)
            # visualize_ind(warped_img1[0, ...,].detach().cpu().numpy().squeeze(), dir_res=dir_res, name="warped_img1.png", save=True)

            # print("gt:", gt.shape[1])
            
        if gt.shape[1] == 1: # 3
            # print("gt.shape[1] = 1")
            # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
            # correct shapes
            max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
            max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
            img0 = img0[:,:,:max_shape_2,:max_shape_3]
            img1 = img1[:,:,:max_shape_2,:max_shape_3]
            warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3]
            warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3]
            mask = mask[:,:,:max_shape_2,:max_shape_3]
            flow = flow[:,:,:max_shape_2,:max_shape_3]
            gt = gt[:,:,:max_shape_2,:max_shape_3]
            # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
            # print("flow:", flow.shape)

            # print("before block tea", torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1).shape)
            flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
            flow_d = flow_d[:,:,:max_shape_2,:max_shape_3]
            mask_d = mask_d[:,:,:max_shape_2,:max_shape_3]
            
            flow_teacher = flow + flow_d
            # print(flow_teacher.size())
            # input("x")
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
        else:
            # print("flow_teacher = None")
            flow_teacher = None
            merged_teacher = None

        for i in range(blocks_range):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            merged[i] = merged[i][:,:,:gt.shape[2],:gt.shape[3]]
            
            if gt.shape[1] == 1: # 3
                flow_list[i] = flow_list[i][:,:,:flow_teacher.shape[2],:flow_teacher.shape[3]]

                loss_mask = ( (merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01 ).float().detach()
                # loss_distill += ((flow_teacher.detach() - flow_list[i]).abs() * loss_mask).mean()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
                # print( (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5).mean() )
                # print(loss_mask)
                # print(loss_mask.mean())
                # input("loss_distill")

        # print("in IFNet, forward")
        if refine:
            # max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
            # max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
            # img0 = img0[:,:,:max_shape_2,:max_shape_3]
            # img1 = img1[:,:,:max_shape_2,:max_shape_3]
            # warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3]
            # warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3]
            # flow = flow[:,:,:max_shape_2,:max_shape_3]
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            # max_shape_2 = min(c0.shape[2], c1.shape[2])
            # max_shape_3 = min(c0.shape[3], c1.shape[3])
            # c0 = c0[:,:,:max_shape_2,:max_shape_3]
            # c1 = c1[:,:,:max_shape_2,:max_shape_3]
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged[2] = torch.clamp(merged[2] + res, 0, 1)
            # print(merged[2].shape)
            # input("x")
            # print("refined")
        # return flow_list, mask_list[2], merged, flow_teacher, merged_teacher, loss_distill
        return flow_list, mask_list, merged, flow_teacher, merged_teacher, loss_distill


# (block2): IFBlock(                                                                                  
#       (conv0): Sequential(                                                                              
#         (0): Sequential(                                                                                
#           (0): Conv2d(9, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))                         
#           (1): PReLU(num_parameters=16)                                                                 
#         )                                                                                               
#         (1): Sequential(                                                                                
#           (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#       )                                                                                                 
#       (convblock): Sequential(                                                                          
#         (0): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (1): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (2): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (3): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (4): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (5): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (6): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#         (7): Sequential(                                                                                
#           (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                        
#           (1): PReLU(num_parameters=32)                                                                 
#         )                                                                                               
#       )                                                                                                 
#       (lastconv): ConvTranspose2d(32, 5, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))             
#     )
#     (block_tea): IFBlock(
#       (conv0): Sequential( 
#         (0): Sequential(
#           (0): Conv2d(10, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#           (1): PReLU(num_parameters=16)
#         )
#         (1): Sequential(
#           (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#           (1): PReLU(num_parameters=32)
#         )
