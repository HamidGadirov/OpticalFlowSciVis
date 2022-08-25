import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

# TODO: change for volumetric (5-D) input +

def warp(tenInput, tenFlow): # img0, flow[:, :3]
    # print(tenInput.shape, tenFlow.shape)
    # input("x")
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        # In the case of 5D inputs, grid[n, d, h, w] specifies the x, y, z pixel locations for interpolating output[n, :, d, h, w]. 
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3], 1).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1, tenFlow.shape[4])
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1, 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3], tenFlow.shape[4])
        tenDepth = torch.linspace(-1.0, 1.0, tenFlow.shape[4], device=device).view(
            1, 1, 1, 1, tenFlow.shape[4]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], tenFlow.shape[3], -1)
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical, tenDepth], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
                         tenFlow[:, 2:3, :, :, :] / ((tenInput.shape[4] - 1.0) / 2.0)], 1)

    # print(backwarp_tenGrid[k].shape, tenFlow.shape)
    # input("warp")

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 4, 1) # 5D
    # When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear.

    # half precision
    # g.half()
    # tenInput = tenInput.to(torch.float32)
    grid_s = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
    # grid_s = grid_s.to(torch.float16)

    # return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
    return grid_s


