import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

def gauss_kernel(size=5, channels=1): # channels=3
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
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    # print(cc.shape)
    # input("cc")
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    # print(cc.shape)
    # input("cc")
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    # print(img.shape)
    # input("x")
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    # print(img.shape)
    # input("after pad")
    # print(kernel.shape)
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    # print(out.shape)
    # input("after conv2d")
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
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
        # print("pyramid", current.shape, up.shape)
        max_shape_2 = min(current.shape[2], up.shape[2])
        max_shape_3 = min(current.shape[3], up.shape[3])
        # print(max_shape_2, max_shape_3)
        current = current[:,:,:max_shape_2,:max_shape_3]
        up = up[:,:,:max_shape_2,:max_shape_3]
        diff = current - up
        # print("diff", diff.shape)
        # input("diff")
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=1): # channels=3
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        # print("pyr_input", pyr_input.shape)
        # input("x")
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        # print("pyr_target", pyr_target.shape)
        # input("x")
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
