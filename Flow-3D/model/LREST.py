import torch
import torch.nn as nn
import numpy as np
import math
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.LRESTNet import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
# from model.refine import *

device = torch.device("cuda")

flow_available = True # True False

student_teacher = True # True False
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = LRESTNet()
        
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

    # def inference(self, img0, img1, scale_list=[1, 1, 1], TTA=False, timestep=0.5): # [4, 2, 1]
    def inference(self, img0, img1, t_mask, TTA=False): # t=0.5
        # print("in RIFE, inference")
        # # print(self.flownet) # LRESTNet summary
        # # input("x")
        # imgs = torch.cat((img0, img1), 1)
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list, timestep=timestep)
        # print("back to RIFE from LRESTNet")

        imgs = torch.cat((img0, img1), 1)
        gt = img0.clone() * 0 # no GT during inference
        if student_teacher:
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt, t_mask), 1))
            # flow, mask, merged, warped_img0, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt, t_mask), 1))
        else:
            flow, mask, merged = self.flownet(torch.cat((imgs, gt, t_mask), 1))

        if TTA == False:
            # return merged[2], flow, mask # get the flow too
            return merged, flow, mask # get all 3 frames
            # return merged, flow, mask, warped_img0
        else:
            print("not implemented")
            # flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            # return (merged[2] + merged2[2].flip(2).flip(3)) / 2

    # def update(self, imgs, gt, dataset, learning_rate=0, mul=1, training=True, flow_gt=None):
    def update(self, imgs, gt, t_mask, dataset, learning_rate=0, mul=1, training=True, flow_gt=None):

        # torch.autograd.detect_anomaly(True) 
        # torch.autograd.anomaly_mode(True)
        
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate

        # print("imgs, gt:", imgs.shape, gt.shape) # imgs, gt: torch.Size([30, 2, 64, 64, 64]) torch.Size([30, 1, 64, 64, 64])

        if "3d" in dataset:
            gt_data = gt[:, 0, :1]
            gt_flow = gt[:, 0, 1:4]
            img0 = imgs[:, :1]
            img1 = imgs[:, 1:2]
            img0_data = img0[:, 0, :1]
            img0_flow = img0[:, 0, 1:4]
            img1_data = img1[:, 0, :1]
            img1_flow = img1[:, 0, 1:4]
            t_mask = t_mask[:, 0, :1]

            # using density:
            img0 = img0_data
            img1 = img1_data
            imgs = torch.cat((img0, img1), 1)
            gt = gt_data

            imgs_ = imgs.detach().cpu().numpy()
            gt_ = gt.detach().cpu().numpy()
            t_mask_ = t_mask.detach().cpu().numpy()
            # print("imgs_ is in range %f to %f" % (np.min(imgs_), np.max(imgs_)))
            # print("gt_ is in range %f to %f" % (np.min(gt_), np.max(gt_)))
            # print("t_mask_ is in range %f to %f" % (np.min(t_mask_), np.max(t_mask_)))

        else:
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
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        # print("imgs, gt:", imgs.shape, gt.shape)
        # with torch.cuda.amp.autocast(True):
        # flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[1, 1, 1])
        if student_teacher:
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt, t_mask), 1))
        else:
            flow, mask, merged = self.flownet(torch.cat((imgs, gt, t_mask), 1)) # # try without student-teacher: gt is not used
        
        # print(np.isfinite(flow.detach().cpu().numpy()).all())
        # print(np.isfinite(mask.detach().cpu().numpy()).all())
        # print(np.isfinite(merged.detach().cpu().numpy()).all())
        # print(np.isfinite(flow_teacher.detach().cpu().numpy()).all())
        # print(np.isfinite(merged_teacher.detach().cpu().numpy()).all())
        # print(np.isfinite(loss_distill.detach().cpu().numpy()).all())
        # input("x")
        # mask = mask[2] # only gt (pred) durig training
        mask = mask[-1] # n blocks
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

        # # RIFE loss: same as l1 that I had.
        # # i_weight = gamma**(n_predictions - i - 1)
        # # i_loss = (flow_preds[i] - flow_gt).abs()
        # # flow_loss += i_weight * (valid[:, None] * i_loss).mean()
        # loss_flow = 0.
        # loss_flow += (flow[0][:, 3:6] - gt_flow).abs().mean() * 0.64
        # loss_flow += (flow[1][:, 3:6] - gt_flow).abs().mean() * 0.8
        # loss_flow += (flow[2][:, 3:6] - gt_flow).abs().mean() * 1.
        # # loss_flow += (flow[0][:, :2] - (-gt_flow)).abs().mean() * 0.64
        # # loss_flow += (flow[1][:, :2] - (-gt_flow)).abs().mean() * 0.8
        # # loss_flow += (flow[2][:, :2] - (-gt_flow)).abs().mean() * 1.
        # loss_flow += (flow_teacher[:, 3:6] - gt_flow).abs().mean() * 1.
        # # loss_flow += (flow_teacher[:, :2] - (-gt_flow)).abs().mean() * 1.
        # # loss_flow /= 2.

        # flow_pred = flow[-1].detach().cpu().numpy()
        # flow_gt = gt_flow.detach().cpu().numpy()
        # print("flow pred is in range %f to %f" % (np.min(flow_pred), np.max(flow_pred)))
        # print("flow gt is in range %f to %f" % (np.min(flow_gt), np.max(flow_gt)))
        # input("flow")

        if flow_available:
            loss_flow = 0.
            num_blocks = 3 # 3 4 5
            gamma_coeff = 0.8 # 0.7 0.8 0.9
            for i in range(num_blocks):
                gamma = gamma_coeff ** i
                loss_flow += (flow[num_blocks - i - 1][:, 3:6] - gt_flow).abs().mean() * gamma
            if student_teacher:
                loss_flow += (flow_teacher[:, 3:6] - gt_flow).abs().mean() * 1.
        else:
            loss_flow = torch.tensor(0.)

        # loss_l1 = torch.nn.functional.l1_loss(merged[2], gt)
        loss_l1 = torch.nn.functional.l1_loss(merged[-1], gt) # l1
        # loss_l1 = (self.lap(merged[-1], gt)).mean() # laplacian
        # loss_tea = (self.lap(merged_teacher, gt)).mean()
        # loss_tea = torch.nn.functional.l1_loss(merged_teacher, gt)
        if student_teacher:
            loss_tea = torch.nn.functional.l1_loss(merged_teacher, gt) # l1 
            # loss_tea = (self.lap(merged_teacher, gt)).mean() # laplacian
        else:
            loss_tea = torch.tensor(0.) # no student-teacher
        # print(merged_teacher.shape, gt.shape)
        # input("loss")

        # num_params =  sum(p.numel() for p in self.flownet.parameters() if p.requires_grad)
        # print("Parameters in LRESTNet:", num_params) # 9641368
        # input("p")

        """ L1 regularization """
        if flow_available:
            l1_reg = torch.tensor(0.)
        else:
            block_params_norm = 0.
            for param_tensor in self.flownet.state_dict():
                # print(param_tensor, "\t", self.flownet.state_dict()[param_tensor].size())
                # if "block2" in param_tensor or "block_tea" in param_tensor:
                if "block3" in param_tensor or "block_tea" in param_tensor: # last block block3
                    block_params = self.flownet.state_dict()[param_tensor]
                    block_params_norm += torch.norm(block_params, 1)
            # print(block2_params_norm)
            # input("x")
            l1_reg = block_params_norm

        def charbonnier(x, alpha=0.25, epsilon=1.e-9):
            return torch.pow(torch.pow(x, 2) + epsilon**2, alpha)

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
        # TODO: check this if using 3D photometric loss!
        def generate_grid(B, H, W, Z, device):
            # xx = torch.arange(0, W, Z).view(1, W, Z).repeat(H, 1, 1)
            # yy = torch.arange(H, 0, Z).view(H, 1, Z).repeat(1, W, 1)
            # zz = torch.arange(H, W, 0).view(H, W, 1).repeat(1, 1, Z)
            xx = torch.arange(0, W * Z).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H * Z).view(1, -1).repeat(W, 1)
            zz = torch.arange(0, H * W).view(1, -1).repeat(Z, 1)
            # print("xx:", xx.size)

            xx = xx.view(1, 1, H, W, Z).repeat(B, 1, 1, 1, 1)
            yy = yy.view(1, 1, H, W, Z).repeat(B, 1, 1, 1, 1)
            zz = zz.view(1, 1, H, W, Z).repeat(B, 1, 1, 1, 1)
            # print("xx:", xx.size)

            grid = torch.cat((xx, yy, zz), 1).float()
            grid = torch.transpose(grid, 1, 2)
            grid = torch.transpose(grid, 2, 3)
            grid = torch.transpose(grid, 3, 4)
            grid = grid.to(device)
            return grid
        
        # def generate_grid(B, D, H, W, device):
        #     xx = torch.arange(0, W).view(1, 1, -1).repeat(B, H, 1)
        #     yy = torch.arange(0, H).view(1, -1, 1).repeat(B, 1, W)
        #     zz = torch.arange(0, D).view(-1, 1, 1).repeat(1, H, W)

        #     grid = torch.cat((xx, yy, zz), 0).float()
        #     grid = torch.transpose(grid, 1, 2)
        #     grid = torch.transpose(grid, 0, 2)
        #     grid = grid.to(device)
        #     return grid

        def backwrd_warp(flow, frame):
            # frame is img2
            # b, _, h, w = flow.shape
            b, c, h, w, z = flow.size()
            frame = F.interpolate(frame, size=(h, w, z), mode='trilinear', align_corners=True)
            flow = torch.transpose(flow, 1, 2)
            flow = torch.transpose(flow, 2, 3)
            flow = torch.transpose(flow, 3, 4)

            # print(flow.size(), generate_grid(b, h, w, z, flow.device).size())
            # input("x")

            grid = flow + generate_grid(b, h, w, z, flow.device)
            # print("grid:", grid.size())

            factor = torch.FloatTensor([[[[2 / w, 2 / h, 2 / z]]]]).to(flow.device)
            grid = grid * factor - 1
            warped_frame = F.grid_sample(frame, grid)

            return warped_frame

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
        # loss_photo = torch.tensor(0.)

        if flow_available:
            loss_photo = torch.tensor(0.)
        else:
            def photometric_loss(wraped, frame1):
                h, w, z = wraped.shape[2:]
                frame1 = F.interpolate(frame1, (h, w, z), mode='trilinear', align_corners=False)
                # frame1 = F.interpolate(frame1, (h, w), mode='nearest') 
                p_loss = charbonnier(wraped - frame1)
                p_loss = torch.sum(p_loss, dim=1) # / 3 this is for channels
                return torch.sum(p_loss) / frame1.size(0)
            
            frame1 = img0
            warped_frame2 = backwrd_warp(flow[-1][:, 3:6, ...], merged[-1])
            loss_photo = photometric_loss(warped_frame2, frame1)
            frame3 = img1
            warped_frame2 = backwrd_warp(flow[-1][:, :3, ...], merged[-1])
            loss_photo += photometric_loss(warped_frame2, frame3)
            loss_photo /= 2.

        lambda_l1 = 1. # 1.
        lambda_tea = 1. # 1.
        if flow_available:
            lambda_flow = 2e-1 # 0.01 0.2 1; 0.5 best on rectangle
            lambda_distill = 0
            lambda_reg = 0
            lambda_photo = 0
        else:
            lambda_distill = 1e-4 # 0.01 # 5e-3 # 0.001 # 0.01
            lambda_reg = 1e-8 # droplet # 1e-6 best on rectangle
            lambda_photo = 1e-6 # 2 3 4 5 # 1e-5 best 1e-6
            lambda_flow = 0
        lambda_smooth = 0 # 1e-8 not important

        # check if distill los is nan or overflow
        if student_teacher:
            if math.isnan(loss_distill) or loss_distill > 10.:
                loss_distill = torch.tensor(0.)
        else:
            loss_distill = torch.tensor(0.) 

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
            # epoch:0/1000 2/8 time:0.00+1.34 loss_G:1.4203e+00
            # loss_l1: tensor(0.2562, device='cuda:0', grad_fn=<MulBackward0>)
            # loss_tea: tensor(0.2556, device='cuda:0', grad_fn=<MulBackward0>)
            # loss_flow: tensor(0.9121, device='cuda:0', dtype=torch.float16, grad_fn=<MulBackward0>)
            # epoch:0/1000 3/8 time:0.00+1.34 loss_G:1.4238e+00
            # loss_l1: tensor(37.7109, device='cuda:0', grad_fn=<MulBackward0>)
            # loss_tea: tensor(37.8157, device='cuda:0', grad_fn=<MulBackward0>)
            # loss_flow: tensor(102.9375, device='cuda:0', dtype=torch.float16, grad_fn=<MulBackward0>)
            # input("x")
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[-1] # [2]
            merged_teacher = merged[-1] # [2]

        if student_teacher == False:
            merged_teacher = torch.tensor(0.)
            flow_teacher = torch.tensor(0.)

        # return merged[2], {
        return merged[-1], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[-1][:, :3], # sup_flow[2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1 * lambda_l1,
            'loss_tea': loss_tea * lambda_tea,
            'loss_distill': loss_distill * lambda_distill,
            'l1_reg': l1_reg * lambda_reg,
            'loss_photo': loss_photo * lambda_photo,
            # 'loss_smooth': loss_smooth * lambda_smooth,
            'loss_flow': loss_flow * lambda_flow,
            'loss_G': loss_G
            }
