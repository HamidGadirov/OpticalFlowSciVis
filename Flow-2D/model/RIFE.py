import torch
import torch.nn as nn
import numpy as np
import math
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1, arbitrary=False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
        self.device()
        # self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-3, weight_decay=1e-2)
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
            # self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, model_name, path, rank=0):
        def convert(param):
            # for key, value in param.items():
            #     print(key)
            # input("x")
            return {
            # k.replace("module.", ""): v
            k.replace("", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/{}'.format(path, model_name))))
            print("Loaded {}".format(model_name))
            # print("loaded flownet_l1_reg.pkl") flownet_lapl_reg_
        
    def save_model(self, model_name, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/{}'.format(path, model_name))
            print("saved {}".format(model_name))

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        print("in RIFE, inference")
        # print(self.flownet) # IFNet summary
        # input("x")
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        print("back to RIFE from IFNet")
        if TTA == False:
            # return merged[2], flow, mask # get the flow too
            return merged, flow, mask # get all 3 frames
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(self, imgs, gt, dataset, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        if dataset == "pipedcylinder2d" or dataset == "cylinder2d" or dataset == "FluidSimML2d" \
            or dataset == "rectangle2d" or dataset == "lbs2d":
            # print(imgs.shape)
            # print(gt.shape)
            # input("x")
            gt_data = gt[:, 0, :1]
            gt_flow = gt[:, 0, 1:3]
            img0 = imgs[:, :1]
            img1 = imgs[:, 1:2]
            img0_data = img0[:, 0, :1]
            img0_flow = img0[:, 0, 1:3]
            img1_data = img1[:, 0, :1]
            img1_flow = img1[:, 0, 1:3]
            # print(img0_data.shape, img1_flow.shape)
            # input("x")
            # img0_flow_uv = imgs[:, :2] # (b_s, 3, 3, 450, 150)
            img0 = img0_data
            img1 = img1_data
            imgs = torch.cat((img0, img1), 1)
            gt = gt_data
            # print(imgs.shape)
            # input("x")
        else:
            img0 = imgs[:, :1]
            img1 = imgs[:, 1:2]

        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[1, 1, 1])
        mask = mask[2] # only gt (pred) durig training
         # correct shapes
        max_shape_2 = min(img0.shape[2], mask.shape[2])
        max_shape_3 = min(img0.shape[3], mask.shape[3])
        gt = gt[:,:,:max_shape_2,:max_shape_3]
        if dataset == "pipedcylinder2d" or dataset == "cylinder2d" or dataset == "FluidSimML2d" \
            or dataset == "rectangle2d" or dataset == "lbs2d":
            gt_flow = gt_flow[:,:,:max_shape_2,:max_shape_3]
            # Flow loss
            # flow and flow_uv
            # print("flow[2]:", flow[2].shape) # 2nd IFBlock, 4 channels: Ft->0 and Ft->1 x and y
            # print("gt_flow", gt_flow.shape)
            # input("flow")
            # loss_flow = torch.nn.functional.l1_loss(flow[2][:, :2], gt_flow) # this was wrong!
            # flow[:, :2] is Ft->0 and flow[:, 2:4] is Ft->1
            # TODO: add to all blocks +
            loss_flow = torch.nn.functional.l1_loss(flow[0][:, 2:4], gt_flow)
            loss_flow += torch.nn.functional.l1_loss(flow[1][:, 2:4], gt_flow)
            loss_flow += torch.nn.functional.l1_loss(flow[2][:, 2:4], gt_flow)
            # TODO: add Ft->0 to all blocks using reverted gt_flow
            loss_flow += torch.nn.functional.l1_loss(flow[0][:, :2], -gt_flow)
            loss_flow += torch.nn.functional.l1_loss(flow[1][:, :2], -gt_flow)
            loss_flow += torch.nn.functional.l1_loss(flow[2][:, :2], -gt_flow)
            # TODO: add this to Tea block +
            loss_flow += torch.nn.functional.l1_loss(flow_teacher[:, 2:4], gt_flow)
            loss_flow += torch.nn.functional.l1_loss(flow_teacher[:, :2], -gt_flow)
            loss_flow /= 8.

        # replacing laplacian loss with l1
        # print("Laplace pyramid + L1 + AdamW (L2) + smoothness loss")
        # print("Laplace pyramid + L1 + AdamW (L2)")
        # print("Laplace pyramid + AdamW (L2)")
        # print(merged[2].shape, gt.shape)
        loss_l1 = (self.lap(merged[2], gt)).mean()
        # print("L1 loss")
        # print(merged[0].shape, gt.shape)
        # loss_l1 = torch.nn.functional.l1_loss(merged[0], gt) # merged[2]
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        # print(merged_teacher.shape, gt.shape)
        # loss_tea = torch.nn.functional.l1_loss(merged_teacher, gt)
        # sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

        # adding L1 reg: ? only to the last block
        # l1_lambda = 1e-6 # 1e-5 1e-4
        # l1_norm = sum(torch.norm(p, 1) for p in self.flownet.parameters())
        # l1_reg = l1_lambda * l1_norm

        # num_params =  sum(p.numel() for p in self.flownet.parameters() if p.requires_grad)
        # print("Parameters in IFNet:", num_params) # 2526410
        # input("p")

        # from torchsummary import summary
        # summary(self.flownet, (2, 128, 128))
        # print(self.flownet)
        # input("p")
        # for p in self.flownet.parameters():
        #     print(p)
            #     for key, value in param.items():
            # print(key)
        # Print model's state_dict
        # print("Model's state_dict:")

        block2_params = 0.
        for param_tensor in self.flownet.state_dict():
            # print(param_tensor, "\t", self.flownet.state_dict()[param_tensor].size())
            if "block2" in param_tensor or "block_tea" in param_tensor:
                p = self.flownet.state_dict()[param_tensor]
                # print(p)
                block2_params += torch.norm(p, 1)
        # print(block2_params)
        # input("x")
        # l1_lambda = 1e-5 # 1e-4
        l1_norm = block2_params
        l1_reg = l1_norm # l1_lambda * l1_norm

        def charbonnier(x, alpha=0.25, epsilon=1.e-9): # for photo or smooth loss
            return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)

        """ Smoothness consistency """
        """
        def smoothness_loss(flow):
            # print("Smoothness loss")
            flow_0 = flow[0]
            # print(flow_0.size())
            flow_1 = flow[1]
            # print(flow_1.size())
            flow_last = flow[2]
            # print(flow_last.size())
            # input("x")

            flow = flow_last
            # print(flow.size()) # torch.Size([64, 4, 150, 448])
            # input("x")
            b, c, h, w = flow.size() # do it for all blocks
            v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w, device=flow.device)), dim=-2)
            # v_translated = torch.cat((flow[:, :, 1:, :], torch.zeros(b, c, 1, w)), dim=-2)
            h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1, device=flow.device)), dim=-1)
            # h_translated = torch.cat((flow[:, :, :, 1:], torch.zeros(b, c, h, 1)), dim=-1)

            s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated)
            s_loss = torch.sum(s_loss, dim=1) / 2

            return torch.sum(s_loss) / b

        loss_smooth = smoothness_loss(flow)
        # smooth_lambda = 0.01
        # loss_smooth = smooth_lambda * smooth
        """
        """ end Smoothness consistency """

        """ Photometric consistency """
        def generate_grid(B, H, W, device):
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            # print(xx.size())
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            # print(yy.size)
            # input("xx yy")

            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            grid = torch.transpose(grid, 1, 2)
            grid = torch.transpose(grid, 2, 3)
            grid = grid.to(device)
            return grid

        def backwrd_warp(flow, frame):
            # frame is img2
            # b, _, h, w = flow.shape
            b, c, h, w = flow.size()
            frame = F.interpolate(frame, size=(h, w), mode='bilinear', align_corners=True)
            flow = torch.transpose(flow, 1, 2)
            flow = torch.transpose(flow, 2, 3)

            # print(flow.size(), generate_grid(b, h, w, flow.device).size())
            # input("x")

            grid = flow + generate_grid(b, h, w, flow.device)
            # print("grid:", grid.size())

            factor = torch.FloatTensor([[[[2 / w, 2 / h]]]]).to(flow.device)
            grid = grid * factor - 1
            warped_frame = F.grid_sample(frame, grid)

            return warped_frame

        # Photometric loss
        # computed as the difference between the first image and the backward/inverse warped second image
        # in this case: can combine img1, img2 (pred); img2 (pred), img3
        # loss, bce_loss, smooth_loss = criterion(pred_flows, wraped_imgs, imgs[:, :3, :, :])
        def photometric_loss(wraped, frame1):
            h, w = wraped.shape[2:]
            frame1 = F.interpolate(frame1, (h, w), mode='bilinear', align_corners=False)
            p_loss = charbonnier(wraped - frame1)
            p_loss = torch.sum(p_loss, dim=1) / 3 # ?
            return torch.sum(p_loss) / frame1.size(0)
        
        frame1 = img0
        warped_frame2 = backwrd_warp(flow[2][:, 2:4, ...], merged[2]) # :2 was wrong!
        loss_photo = photometric_loss(warped_frame2, frame1)
        frame3 = img1
        warped_frame2 = backwrd_warp(flow[2][:, :2, ...], merged[2]) # 2:4 was wrong!
        loss_photo += photometric_loss(warped_frame2, frame3)
        loss_photo /= 2
        # print(loss_photo)
        # input("x")
        """ end Photometric consistency """

        lambda_l1 = 1 # 1
        lambda_tea = 1 # 1
        lambda_distill = 0.02 # 0.01 0.1 # without is bad # 0.01 best
        lambda_reg = 1e-7 # 1e-6 best on rectangle 1e-7
        lambda_photo = 1e-5 # 1e-5 # 2 3 4 5 # 1e-5 best
        lambda_smooth = 0 # 1e-8 not important
        lambda_flow = 0.2 # 0.01 0.2 1
        # automatic parameter study
        # keep simple loss: 3 parameters
        # change in interpol func - additional parameter

        # check if distill los is nan or overflow
        if math.isnan(loss_distill) or loss_distill > 10.:
            loss_distill = torch.tensor(0.)
        if dataset == "droplet2d" or dataset == "vimeo2d":
             loss_flow = torch.tensor(0.)

        loss_G = loss_l1 * lambda_l1 + loss_tea * lambda_tea + loss_distill * lambda_distill + \
                l1_reg * lambda_reg + loss_photo * lambda_photo + loss_flow * lambda_flow # + loss_smooth * lambda_smooth

        if training:
            self.optimG.zero_grad()

            # print("loss_l1:", loss_l1 * lambda_l1)
            # print("loss_tea:", loss_tea * lambda_tea)
            # print("loss_distill:", loss_distill * lambda_distill)
            # print("l1_reg:", l1_reg * lambda_reg)
            # print("loss_photo:", loss_photo * lambda_photo)

            # print("loss_smooth:", loss_smooth * lambda_smooth)
            # print("loss_flow:", loss_flow * lambda_flow)
            # input("x")
            
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
            merged_teacher = merged[2]
        

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1 * lambda_l1,
            'loss_tea': loss_tea * lambda_tea,
            'loss_distill': loss_distill * lambda_distill,
            'l1_reg': l1_reg * lambda_reg,
            'loss_photo': loss_photo * lambda_photo,
            'loss_flow': loss_flow * lambda_flow,
            'loss_G': loss_G
            }
