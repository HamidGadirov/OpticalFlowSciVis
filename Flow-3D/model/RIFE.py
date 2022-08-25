import torch
import torch.nn as nn
import numpy as np
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
        
        # half precision
        # self.flownet.half()

        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
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
            return {
            # k.replace("module.", ""): v
            k.replace("", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/{}'.format(path, model_name))))
            print("loaded {}".format(model_name))
            # print("loaded flownet_l1_reg.pkl") flownet_lapl_reg_
        
    def save_model(self, model_name, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/{}'.format(path, model_name))
            print("saved {}".format(model_name))

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5): # [4, 2, 1]
        print("in RIFE, inference")
        # print(self.flownet) # IFNet summary
        # input("x")
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        print("back to RIFE from IFNet")
        if TTA == False:
            return merged[2], flow, mask # get the flow too
        else:
            print("not implemented")
            # flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            # return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):

        # torch.autograd.detect_anomaly(True) 
        # torch.autograd.anomaly_mode(True)
        
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :1]
        img1 = imgs[:, 1:2]

        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        # input("x")

        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[1, 1, 1])
        # print(np.isfinite(flow.detach().cpu().numpy()).all())
        # print(np.isfinite(mask.detach().cpu().numpy()).all())
        # print(np.isfinite(merged.detach().cpu().numpy()).all())
        # print(np.isfinite(flow_teacher.detach().cpu().numpy()).all())
        # print(np.isfinite(merged_teacher.detach().cpu().numpy()).all())
        # print(np.isfinite(loss_distill.detach().cpu().numpy()).all())
        # input("x")
         # correct shapes
        max_shape_2 = min(img0.shape[2], mask.shape[2])
        max_shape_3 = min(img0.shape[3], mask.shape[3])
        max_shape_4 = min(img0.shape[4], mask.shape[4])
        # print(max_shape_2, max_shape_3)
        gt = gt[:,:,:max_shape_2,:max_shape_3,:max_shape_4]

        # replacing laplacian loss with l1
        # print("Laplace pyramid + L1 + AdamW (L2) + smoothness loss")
        # print("Laplace pyramid + L1 + AdamW (L2)")
        # print("Laplace pyramid + AdamW (L2)")
        # print("L1 loss + AdamW (L2)")
        # print(merged[2].shape, gt.shape)
        # loss_l1 = (self.lap(merged[2], gt)).mean()
        # print(np.min(merged[2].detach().cpu().numpy()), np.max(merged[2].detach().cpu().numpy()))
        # print(np.mean(merged_teacher.detach().cpu().numpy()))
        # print(np.min(gt.detach().cpu().numpy()), np.max(gt.detach().cpu().numpy()))
        # input("l1_loss")

        loss_l1 = torch.nn.functional.l1_loss(merged[2], gt)
        # loss_tea = (self.lap(merged_teacher, gt)).mean()
        loss_tea = torch.nn.functional.l1_loss(merged_teacher, gt)
        # print(merged_teacher.shape, gt.shape)
        # input("loss")

        # num_params =  sum(p.numel() for p in self.flownet.parameters() if p.requires_grad)
        # print("Parameters in IFNet:", num_params) # 9641368
        # input("p")

        # adding L1 reg:
        l1_lambda = 1e-5 # 1e-4
        l1_norm = sum(torch.norm(p, 1) for p in self.flownet.parameters())
        l1_reg = l1_lambda * l1_norm

        # def charbonnier(x, alpha=0.25, epsilon=1.e-9):
        #     return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)

        # def smoothness_loss(flow):
        #     # print("Smoothness loss")
        #     flow_0 = flow[0]
        #     flow_1 = flow[1]
        #     flow_last = flow[2]
        #     # print(flow_last.size())
        #     # input("x")

        #     flow = flow_last
        #     b, c, h, w, z = flow.size() # do it for all blocks
        #     v_translated = torch.cat((flow[:, :, 1:, :, :], torch.zeros(b, c, 1, w, z, device=flow.device)), dim=-3)
        #     h_translated = torch.cat((flow[:, :, :, 1:, :], torch.zeros(b, c, h, 1, z, device=flow.device)), dim=-2)
        #     z_translated = torch.cat((flow[:, :, :, :, 1:], torch.zeros(b, c, h, w, 1, device=flow.device)), dim=-1)

        #     s_loss = charbonnier(flow - v_translated) + charbonnier(flow - h_translated) + charbonnier(flow - z_translated)
        #     s_loss = torch.sum(s_loss, dim=1) / 3

        #     return torch.sum(s_loss) / b

        # loss_smooth = smoothness_loss(flow)
        # # smooth_lambda = 0.001 # 0.01
        # # loss_smooth = smooth_lambda * smooth


        # def generate_grid(B, H, W, Z, device):
        #     # xx = torch.arange(0, W, Z).view(1, W, Z).repeat(H, 1, 1)
        #     # yy = torch.arange(H, 0, Z).view(H, 1, Z).repeat(1, W, 1)
        #     # zz = torch.arange(H, W, 0).view(H, W, 1).repeat(1, 1, Z)
        #     xx = torch.arange(0, W * Z).view(1, -1).repeat(H, 1)
        #     yy = torch.arange(0, H * Z).view(1, -1).repeat(W, 1)
        #     zz = torch.arange(0, H * W).view(1, -1).repeat(Z, 1)
        #     # print("xx:", xx.size)

        #     xx = xx.view(1, 1, H, W, Z).repeat(B, 1, 1, 1, 1)
        #     yy = yy.view(1, 1, H, W, Z).repeat(B, 1, 1, 1, 1)
        #     zz = zz.view(1, 1, H, W, Z).repeat(B, 1, 1, 1, 1)
        #     # print("xx:", xx.size)

        #     grid = torch.cat((xx, yy, zz), 1).float()
        #     grid = torch.transpose(grid, 1, 2)
        #     grid = torch.transpose(grid, 2, 3)
        #     grid = torch.transpose(grid, 3, 4)
        #     grid = grid.to(device)
        #     return grid

        # def backwrd_warp(flow, frame):
        #     # frame is img2
        #     # b, _, h, w = flow.shape
        #     b, c, h, w, z = flow.size()
        #     frame = F.interpolate(frame, size=(h, w, z), mode='trilinear', align_corners=True)
        #     flow = torch.transpose(flow, 1, 2)
        #     flow = torch.transpose(flow, 2, 3)
        #     flow = torch.transpose(flow, 3, 4)

        #     # print(flow.size(), generate_grid(b, h, w, z, flow.device).size())
        #     # input("x")

        #     grid = flow + generate_grid(b, h, w, z, flow.device)
        #     # print("grid:", grid.size())

        #     factor = torch.FloatTensor([[[[2 / w, 2 / h, 2 / z]]]]).to(flow.device)
        #     grid = grid * factor - 1
        #     warped_frame = F.grid_sample(frame, grid)

        #     return warped_frame

        # # Photometric loss
        # # computed as the difference between the first image and the backward/inverse warped second image
        # def photometric_loss(wraped, frame1):
        #     h, w, z = wraped.shape[2:]
        #     frame1 = F.interpolate(frame1, (h, w, z), mode='trilinear', align_corners=False)
        #     p_loss = charbonnier(wraped - frame1)
        #     p_loss = torch.sum(p_loss, dim=1) / 3 # ?
        #     return torch.sum(p_loss) / frame1.size(0)
        
        # frame1 = img0
        # warped_frame2 = backwrd_warp(flow[2][:, :3, ...], merged[2])
        # loss_photo = photometric_loss(warped_frame2, frame1)
        # frame3 = img1
        # warped_frame2 = backwrd_warp(flow[2][:, 3:6, ...], merged[2])
        # loss_photo += photometric_loss(warped_frame2, frame3)
        # loss_photo /= 2
        # # print(loss_photo)
        # # input("x")

        # Flow loss

        lambda_l1 = 1
        lambda_tea = 1
        lambda_distill = 0.1
        lambda_reg = 0 # 0.01
        lambda_photo = 0 # 1e-6
        lambda_smooth = 0 # 1e-8

        loss_G = loss_l1 * lambda_l1 + loss_tea * lambda_tea + loss_distill * lambda_distill # + \
                    # l1_reg * lambda_reg + loss_smooth * lambda_smooth + loss_photo * lambda_photo

        if training:
            self.optimG.zero_grad()
            
             # ? another loss component - flow ?
            # print("loss_l1:", loss_l1 * lambda_l1)
            # print("loss_tea:", loss_tea * lambda_tea)
            # print("loss_distill:", loss_distill * lambda_distill)
            # print("l1_reg:", l1_reg * lambda_reg)
            # print("loss_photo:", loss_photo * lambda_photo)
            # print("loss_smooth:", loss_smooth * lambda_smooth)
            # input("x")
            loss_G.backward()
            self.optimG.step()
        else: # evaluate
            flow_teacher = flow[2]
            merged_teacher = merged[2]
            # loss_G = loss_l1 + loss_tea # + loss_distill * 0.01

        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'loss_G': loss_G
            }
