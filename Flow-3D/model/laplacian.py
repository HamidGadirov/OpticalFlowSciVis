import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

def gauss_kernel(size=5, channels=1):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3], x.shape[4])
    # print(cc.shape)
    cc = cc.permute(0,1,3,2,4)
    # print(cc.shape)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, x.shape[4]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2, x.shape[4])
    # print(cc.shape)
    cc = cc.permute(0,1,3,2,4)
    # print("cc, xx:", cc.shape, x.shape)

    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2, x.shape[4]).to(device)], dim=4)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2, x.shape[4]*2)
    # print("last")
    # input(cc.shape)
    x_up = cc
    x_up = x_up * 8 #
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    # print(img.shape)
    # input("x")
    # p3d = (2, 2, 2, 2, 2, 2)
    # img = F.pad(img, p3d, "constant", 0)
    # img = torch.nn.functional.pad(img, (2, 2, 2, 2, 2, 2), mode='reflect')
    # print(img.shape)
    # input("after pad")
    # print(kernel.shape)
    # out = torch.nn.functional.conv3d(img, kernel, groups=img.shape[1])
    out = gaussian_filter(img.cpu().detach().numpy(), sigma=1)
    out = torch.from_numpy(out).to(device)
    # out = out.to(device)
    # print(out.shape)
    # input("after gauss filter")
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    # print(current.shape)
    # input("x")
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        # print("filtered", filtered.shape)
        # input("x")
        down = downsample(filtered)
        # print("down", down.shape)
        # input("x")
        up = upsample(down)
        # print("up", up.shape)
        # input("x")
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=1):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
