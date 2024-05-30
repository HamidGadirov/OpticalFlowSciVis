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

# def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
#     return nn.Sequential(
#         torch.nn.ConvTranspose3d(in_channels=in_planes, out_channels=out_planes, 
#                                 kernel_size=kernel_size, stride=stride, padding=padding),
#         nn.PReLU(out_planes),
#     )

def functional_conv_transpose3d(x, weight, bias, out_planes, stride=2, padding=1):
    x = F.conv_transpose3d(x, weight, bias, stride=stride, padding=padding)
    # x = nn.PReLU(num_parameters=out_planes)(x)
    prelu = nn.PReLU(num_parameters=out_planes).to(x.device)  # Move PReLU to the correct device
    x = prelu(x)
    return x

# def conv(in_planes, weights, stride=1, padding=1, dilation=1):
#     return nn.Sequential(
#         F.conv3d(in_planes, weights, stride=stride,
#                   padding=padding, dilation=dilation, bias=True),
#         nn.PReLU(out_planes),
#     )

def functional_conv3d(x, weight, bias, out_planes, stride=1, padding=1, dilation=1):
    x = F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
    prelu = nn.PReLU(num_parameters=out_planes).to(x.device)  # Move PReLU to the correct device
    x = prelu(x)
    return x

# """
# autoencoder: instead of pixel we have weights
# k channels times 3x3 spatial dims
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, weight_shapes):
        super(HyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.weight_shapes = weight_shapes

        total_params = sum(shape.numel() for shape in self.weight_shapes.values())

        # simple net: relu
        # optimizer
        # what they used in HyperINR

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128)
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, total_params)
        # )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.PReLU(),
            nn.Linear(64, 128),
            nn.PReLU(),
            nn.Linear(128, 256),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, total_params)
        )

        # pre-train, train with hypernet, fine-tuning

        # Qi, 23.05.24
        # limited capability of the LREST
        # implicit neural net (MLP) after hypernet 
        # CNN for hypernet
        # OR diffusion net
        # accept same data with params (conditional diffusion model)

        # implicit net for spatial SR
        # hyper net for temporal SR
        # coordinates for implicit net

        # pretrain LREST - target

        # NVIDIA Zurich - internship next year

        # David, 28.05
        # try hypernet only for some layers of LREST

    def forward(self, params):
        encoded = self.encoder(params)
        weights_flat = self.decoder(encoded)
        return self._reshape_weights(weights_flat)

    def _reshape_weights(self, weights_flat):
        weights = {}
        offset = 0
        for name, shape in self.weight_shapes.items():
            size = shape.numel()
            weights[name] = weights_flat[:, offset:offset + size].view(-1, *shape)
            offset += size
        return weights
# """

# class HyperNetwork(nn.Module):
#     def __init__(self, input_dim, weight_shapes):
#         super(HyperNetwork, self).__init__()
#         self.input_dim = input_dim
#         self.weight_shapes = weight_shapes

#         # Define the layers for the hypernetwork
#         self.fc = nn.Sequential(
#             nn.Linear(input_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU()
#         )

#         self.weight_generators = nn.ModuleDict()
#         for name, shape in weight_shapes.items():
#             self.weight_generators[name] = nn.Linear(128, torch.prod(torch.tensor(shape)).item())

#     def forward(self, params):
#         batch_size = params.size(0)
#         hidden = self.fc(params)
#         weights = {}
#         for name, generator in self.weight_generators.items():
#             weight_shape = self.weight_shapes[name]
#             weight = generator(hidden).view(batch_size, *weight_shape)
#             weights[name] = weight
#         return weights

refine = False # True
use_deconv = True # True scale = [1, 1, 1, 1] if use_deconv else [4, 2, 2, 1]

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, input_dim=3):
        super(IFBlock, self).__init__()
        self.in_planes = in_planes
        self.c = c

        # Define the shapes of the weights for the hypernetwork
        self.weight_shapes = {
            'conv0_0_weight': torch.Size([c//2, in_planes, 4, 4, 4]),
            'conv0_0_bias': torch.Size([c//2]),
            'conv0_1_weight': torch.Size([c, c//2, 4, 4, 4]),
            'conv0_1_bias': torch.Size([c]),
            'deconvblock_0_weight': torch.Size([c, c, 4, 4, 4]),
            'deconvblock_0_bias': torch.Size([c]),
            'deconvblock_1_weight': torch.Size([c, c//2, 4, 4, 4]),
            'deconvblock_1_bias': torch.Size([c//2]),
            'deconvblock_2_weight': torch.Size([c//2, 7, 4, 4, 4]),
            'deconvblock_2_bias': torch.Size([7]),
        }

        # self.weight_shapes = {
        #     'conv0_0_weight': torch.Size([c//2, in_planes, 4, 4, 4]), # out_channels, in_channels
        #     'conv0_0_bias': torch.Size([c//2]),
        #     'conv0_1_weight': torch.Size([c, c//2, 4, 4, 4]),
        #     'conv0_1_bias': torch.Size([c]),
        #     'convblock_0_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_0_bias': torch.Size([c]),
        #     'convblock_1_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_1_bias': torch.Size([c]),
        #     'convblock_2_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_2_bias': torch.Size([c]),
        #     'convblock_3_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_3_bias': torch.Size([c]),
        #     'convblock_4_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_4_bias': torch.Size([c]),
        #     'convblock_5_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_5_bias': torch.Size([c]),
        #     'convblock_6_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_6_bias': torch.Size([c]),
        #     'deconvblock_0_weight': torch.Size([c, c, 4, 4, 4]),
        #     'deconvblock_0_bias': torch.Size([c]),
        #     'convblock_7_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_7_bias': torch.Size([c]),
        #     'convblock_8_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_8_bias': torch.Size([c]),
        #     'convblock_9_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_9_bias': torch.Size([c]),
        #     'deconvblock_1_weight': torch.Size([c, c//2, 4, 4, 4]), # in_channels, out_channels
        #     'deconvblock_1_bias': torch.Size([c//2]),
        #     'deconvblock_2_weight': torch.Size([c//2, 7, 4, 4, 4]),
        #     'deconvblock_2_bias': torch.Size([7]),
        # }

        # Initialize the hypernetwork
        self.hypernet = HyperNetwork(input_dim, self.weight_shapes)

        # Define the layers to be trained normally (LREST)
        self.convblock_0 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_1 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_2 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_3 = nn.Conv3d(c, c, 3, stride=2, padding=1)
        self.convblock_4 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_5 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_6 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_7 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_8 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_9 = nn.Conv3d(c, c, 3, stride=1, padding=1)

        # Define PReLU activations
        self.prelu = nn.PReLU()

        # Freeze LREST layers
        # self.freeze_lrest_layers()

    # def freeze_lrest_layers(self):
    #     lrest_layers = [
    #         self.convblock_0,
    #         self.convblock_1,
    #         self.convblock_2,
    #         self.convblock_3,
    #         self.convblock_4,
    #         self.convblock_5,
    #         self.convblock_6,
    #         self.convblock_7,
    #         self.convblock_8,
    #         self.convblock_9
    #     ]
    #     for layer in lrest_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = False

    def forward(self, x, flow, scale, params):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="trilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="trilinear", align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)

        batch_size = x.size(0)
        # Initialize a list to store the output of each element in the batch
        outputs = []
        # Iterate over each element in the batch
        for i in range(batch_size):
            # print("i", i)
            # Generate weights for the current batch element
            current_params = params[i].unsqueeze(0)
            weight_dict = self.hypernet(current_params)
            weight_dict = {name: weight_dict[name].squeeze(0) for name in weight_dict}  # Remove the batch dim from weights

            x_i = x[i].unsqueeze(0)

            # Wrap in torch.no_grad() to prevent backpropagation through these operations
            with torch.no_grad():
                x_i = functional_conv3d(x_i, weight_dict['conv0_0_weight'], weight_dict['conv0_0_bias'], self.c//2, stride=2, padding=1)
                x_i = functional_conv3d(x_i, weight_dict['conv0_1_weight'], weight_dict['conv0_1_bias'], self.c, stride=2, padding=1)

            x_i.requires_grad_()

            x_i = self.prelu(self.convblock_0(x_i))
            x_i = self.prelu(self.convblock_1(x_i))
            x_i = self.prelu(self.convblock_2(x_i))
            x_i = self.prelu(self.convblock_3(x_i))
            x_i = self.prelu(self.convblock_4(x_i))
            x_i = self.prelu(self.convblock_5(x_i))
            x_i = self.prelu(self.convblock_6(x_i))

            with torch.no_grad():
                x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_0_weight'], weight_dict['deconvblock_0_bias'], self.c, stride=2)

            x_i.requires_grad_()

            x_i = self.prelu(self.convblock_7(x_i))
            x_i = self.prelu(self.convblock_8(x_i))
            x_i = self.prelu(self.convblock_9(x_i))

            convblock_out = x_i
            convblock_out = convblock_out[:, :, :x_i.shape[2], :x_i.shape[3], :x_i.shape[4]]
            x_i = convblock_out + x_i

            with torch.no_grad():
                x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_1_weight'], weight_dict['deconvblock_1_bias'], self.c//2)
                x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_2_weight'], weight_dict['deconvblock_2_bias'], 7)

            flow_i = x_i[:, :6]
            mask_i = x_i[:, 6:7]

            outputs.append((flow_i, mask_i))

            """
            # Process the current element using the generated weights
            x_i = x[i].unsqueeze(0)  # Add batch dimension back for conv operation

            # print(f"x_i is on device: {x_i.device}")
            # for name, tensor in weight_dict.items():
            #     print(f"{name} is on device: {tensor.device}")

            # Perform the convolutions using the extracted weights
            x_i = functional_conv3d(x_i, weight_dict['conv0_0_weight'], weight_dict['conv0_0_bias'], self.c//2, stride=2, padding=1)
            x_i = functional_conv3d(x_i, weight_dict['conv0_1_weight'], weight_dict['conv0_1_bias'], self.c, stride=2, padding=1)
            # print(f"conv0_1_weight")
            x_i = functional_conv3d(x_i, weight_dict['convblock_0_weight'], weight_dict['convblock_0_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_1_weight'], weight_dict['convblock_1_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_2_weight'], weight_dict['convblock_2_bias'], self.c)
            # print(f"convblock_2_weight")
            x_i = functional_conv3d(x_i, weight_dict['convblock_3_weight'], weight_dict['convblock_3_bias'], self.c, stride=2)
            x_i = functional_conv3d(x_i, weight_dict['convblock_4_weight'], weight_dict['convblock_4_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_5_weight'], weight_dict['convblock_5_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_6_weight'], weight_dict['convblock_6_bias'], self.c)
            # print(f"convblock_6_weight")
            x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_0_weight'], weight_dict['deconvblock_0_bias'], self.c, stride=2)
            x_i = functional_conv3d(x_i, weight_dict['convblock_7_weight'], weight_dict['convblock_7_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_8_weight'], weight_dict['convblock_8_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_9_weight'], weight_dict['convblock_9_bias'], self.c)
            print(f"convblock_9_weight")
            convblock_out = x_i
            convblock_out = convblock_out[:, :, :x_i.shape[2], :x_i.shape[3], :x_i.shape[4]]
            x_i = convblock_out + x_i

            x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_1_weight'], weight_dict['deconvblock_1_bias'], self.c//2)
            x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_2_weight'], weight_dict['deconvblock_2_bias'], 7)
            # print(f"deconvblock_2_weight")

            flow_i = x_i[:, :6]  # Best, since I don't use resize
            mask_i = x_i[:, 6:7]

            outputs.append((flow_i, mask_i))
            """

        # Concatenate the outputs to form the final output tensor
        flows = torch.cat([out[0] for out in outputs], dim=0)
        masks = torch.cat([out[1] for out in outputs], dim=0)

        return flows, masks
    """
    def forward(self, x, flow, scale, params):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="trilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="trilinear", align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)

        # Generate weights dynamically using the hypernetwork
        # iterate over each weights
        weights = self.hypernet(params)

        print("conv0", x.shape) # conv0 torch.Size([2, 3, 128, 128, 128])  
        for name, weight in weights.items():
            print("Shape of", name, ":", weight.shape)
            # Shape of conv0_0_weight : torch.Size([2, 128, 3, 4, 4, 4])  

        # David, 22.05.24
        # recursive LREST.set_weight()
        # hyperLREST base class that defines set_weight()
        # nn.Conv3d has weight - tensor that represent conv weight

        # LREST no_grad
        # unet or autoencoder

        # conv0 layers
        x = functional_conv3d(x, weights['conv0_0_weight'], weights['conv0_0_bias'], self.c//2, stride=2, padding=1)
        x = functional_conv3d(x, weights['conv0_1_weight'], weights['conv0_1_bias'], self.c, stride=2, padding=1)

        # convblock layers
        x = functional_conv3d(x, weights['convblock_0_weight'], weights['convblock_0_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_1_weight'], weights['convblock_1_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_2_weight'], weights['convblock_2_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_3_weight'], weights['convblock_3_bias'], self.c, stride=2)
        x = functional_conv3d(x, weights['convblock_4_weight'], weights['convblock_4_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_5_weight'], weights['convblock_5_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_6_weight'], weights['convblock_6_bias'], self.c)
        x = functional_conv_transpose3d(x, weights['deconvblock_0_weight'], weights['deconvblock_0_bias'], self.c, stride=2)
        x = functional_conv3d(x, weights['convblock_7_weight'], weights['convblock_7_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_8_weight'], weights['convblock_8_bias'], self.c)
        x = functional_conv3d(x, weights['convblock_9_weight'], weights['convblock_9_bias'], self.c)

        convblock_out = x
        convblock_out = convblock_out[:, :, :x.shape[2], :x.shape[3], :x.shape[4]]
        x = convblock_out + x

        # deconvblock layers
        x = functional_conv_transpose3d(x, weights['deconvblock_1_weight'], weights['deconvblock_1_bias'], self.c//2)
        x = functional_conv_transpose3d(x, weights['deconvblock_2_weight'], weights['deconvblock_2_bias'], 7)

        flow = x[:, :6]  # Best, since I don't use resize
        mask = x[:, 6:7]

        return flow, mask
    """

class HyperLRESTNet(nn.Module):
    def __init__(self, blocks_num=2, student_teacher=False):
        super(HyperLRESTNet, self).__init__()
        self.blocks_num = blocks_num
        self.student_teacher = student_teacher
        self.refine = refine
        self.use_deconv = use_deconv

        if blocks_num == 2:
            self.block0 = IFBlock(3, c=128, input_dim=3) # 6 c=240 128 256
            self.block1 = IFBlock(6+6, c=64, input_dim=3) # 13 c=90 96 64 128
        if blocks_num == 3:
            self.block0 = IFBlock(3, c=256, input_dim=3) # 6 c=240 128 256
            self.block1 = IFBlock(6+6, c=192, input_dim=3) # 13 c=150 96 192
            self.block2 = IFBlock(6+6, c=128, input_dim=3) # 13 c=90 96 64 128
        elif blocks_num == 4:
            self.block0 = IFBlock(3, c=256, input_dim=3) # 128 256 320 400
            self.block1 = IFBlock(6+6, c=192, input_dim=3) # 96 192 256 320
            self.block2 = IFBlock(6+6, c=192, input_dim=3) # 96 192 256 320
            self.block3 = IFBlock(6+6, c=128, input_dim=3) # 64 128 192 256
        elif blocks_num == 5:
            self.block0 = IFBlock(3, c=256, input_dim=3) # 128 256 320 400
            self.block1 = IFBlock(6+6, c=192, input_dim=3) # 96 192 256 320
            self.block2 = IFBlock(6+6, c=192, input_dim=3) # 96 192 256 320
            self.block3 = IFBlock(6+6, c=128, input_dim=3) # 64 128 192 256
            self.block4 = IFBlock(6+6, c=128, input_dim=3) # 64 128 192 256

        if student_teacher:
            self.block_tea = IFBlock(7+6, c=128, input_dim=3) # 90 128 192 256
        if refine:
            self.contextnet = Contextnet()
            self.unet = Unet()

    # @torch.cuda.amp.autocast(True)
    # def forward(self, x, scale=[1, 1, 1], timestep=0.5): # scale=[4,2,1] scale=[1, 1, 1]
    # def forward(self, x): # scale=[4, 2, 1] scale=[1, 1, 1] t=0.5
    def forward(self, x, params): 
        # img0 = x[:, :1] 
        # img1 = x[:, 1:2]
        # gt = x[:, 2:3] # In inference time, gt is None
        img0 = x[:, :1]
        img1 = x[:, 1:2]
        gt = x[:, 2:3]
        t_mask = x[:, 3:4]

        flow_list = [] # flow_list represents flow predictions by each IFBlock, 6 channels in 3D, Ft->0, Ft->1
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        if self.blocks_num == 2:
            stu = [self.block0, self.block1]
            scale = [1, 1] if use_deconv else [2, 1]
        elif self.blocks_num == 3:
            stu = [self.block0, self.block1, self.block2]
            scale = [1, 1, 1] if use_deconv else [4, 2, 1]
        elif self.blocks_num == 4:
            stu = [self.block0, self.block1, self.block2, self.block3]
            # scale = [1, 1, 1, 1] if use_deconv else [8, 4, 2, 1]
            scale = [1, 1, 1, 1] if use_deconv else [4, 2, 2, 1]
        elif self.blocks_num == 5:
            stu = [self.block0, self.block1, self.block2, self.block3, self.block4]
            scale = [1, 1, 1, 1, 1] if use_deconv else [4, 2, 2, 1, 1]
        # stu = [self.block0, self.block1, self.block2]
        # stu = [self.block0, self.block2]
        for i in range(self.blocks_num):
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
                # flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow_d, mask_d = stu[i](torch.cat((img0, img1, t_mask, warped_img0, warped_img1, mask), 1), flow, scale=scale[i], params=params)
                flow_d = flow_d[:,:,:img0.shape[2],:img0.shape[3],:img0.shape[4]]
                mask_d = mask_d[:,:,:img0.shape[2],:img0.shape[3],:img0.shape[4]]
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                # print(i)
                # print("cat", torch.cat((img0, img1), 1).shape)
                # input("flow == None")
                # flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i]) # stu[0]
                flow, mask = stu[i](torch.cat((img0, img1, t_mask), 1), None, scale=scale[i], params=params)
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
            # print("img0:", img0.shape)
            # print("flow:", flow.shape)
            # input("x")
            warped_img0 = warp(img0, flow[:, :3]) # Ft->0
            warped_img1 = warp(img1, flow[:, 3:6]) # Ft->1
            # print("img0:", img0.shape)
            # print("warped_img0:", warped_img0.shape)
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

            # print("gt:", gt.shape[1])
            # input("gt")
            
        if self.student_teacher:
            if gt.shape[1] == 1: # 3
                # input("gt")
                # print("gt.shape[1] = 1")
                # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
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
                # print("flow:", flow.shape)

                # """
                # print("before block tea", torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1).shape)
                # flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
                flow_d, mask_d = self.block_tea(torch.cat((img0, img1, t_mask, warped_img0, warped_img1, mask, gt), 1), flow, scale=1, params=params)
                flow_d = flow_d[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                mask_d = mask_d[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                
                flow_teacher = flow + flow_d
                # print("img0 before warp:", img0.shape)
                warped_img0_teacher = warp(img0, flow_teacher[:, :3])
                # print("warped_img0_teacher:", warped_img0_teacher.shape)
                warped_img1_teacher = warp(img1, flow_teacher[:, 3:6])
                mask_teacher = torch.sigmoid(mask + mask_d)
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
                # """
                # flow_teacher = 0
                # merged_teacher = 0
            else:
                # print("flow_teacher = None")
                flow_teacher = None
                merged_teacher = None

        for i in range(self.blocks_num):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            merged[i] = merged[i][:,:,:gt.shape[2],:gt.shape[3],:gt.shape[4]]
            
            # """
            if self.student_teacher:
                if gt.shape[1] == 1: # 3
                    flow_list[i] = flow_list[i][:,:,:flow_teacher.shape[2],:flow_teacher.shape[3],:flow_teacher.shape[4]]

                    loss_mask = ( (merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01 ).float().detach()
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
            # """

        # print("in LRESTNet, forward")
        if refine:
            # c0 = self.contextnet(img0, flow[:, :2])
            # c1 = self.contextnet(img1, flow[:, 2:4])
            # tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            # res = tmp[:, :3] * 2 - 1
            # merged[2] = torch.clamp(merged[2] + res, 0, 1)
            print("refined")
        # return flow_list, mask_list, merged, flow_teacher, merged_teacher, loss_distill # mask_list[2]
        if self.student_teacher:
            return flow_list, mask_list, merged, flow_teacher, merged_teacher, loss_distill
            # return flow_list, mask_list, merged, warped_img0, flow_teacher, merged_teacher, loss_distill
        else:
            return flow_list, mask_list, merged
