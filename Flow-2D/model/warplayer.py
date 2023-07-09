import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

def warp(tenInput, tenFlow):
    # print(tenInput.shape, tenFlow.shape)
    # input("x")
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    # performs normalization on the flow tensor tenFlow to bring its values within the range of -1.0 to 1.0.

    # print(tenHorizontal.shape, backwarp_tenGrid[k].shape, tenFlow.shape)
    # input("warp")

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

    # print(g.shape, tenFlow.shape)
    # input("warp")

    if tenInput.dtype==torch.float16:
        # bilinear
        grid_s = torch.nn.functional.grid_sample(input=tenInput, grid=g.half(), mode='bilinear', padding_mode='border', align_corners=True)
    else:
        grid_s = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
        # grid_s = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='nearest', padding_mode='border')

    # grid_s = grid_s.to(torch.float16)

    # return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
    return grid_s
