import torch.nn as nn
from .network_submodule import *
import numpy as np


class ConvAutoencoder(nn.Module):
    def __init__(self, type='train', submodule_type=0, leakiness = 0.01,
                 input_channel=2, input_res=256, first_output_channel=8, depth=6, feature_vector_size=512):
        super(ConvAutoencoder, self).__init__()


        self.depth = depth + 1
        # output_channel_list has depth size
        if self.depth > 1:
            self.output_channel_list = np.empty(self.depth, dtype=int)  
        else:
            raise Exception('depth of the network must be larger than 2')
        self.output_channel_list[0] = input_channel
        self.output_channel_list[1] = first_output_channel
        for i in range(2, self.depth):
            self.output_channel_list[i] = self.output_channel_list[i-1] * 2

        self.inverse_output_channel_list = self.output_channel_list[::-1]
        
        bottom_res = input_res // 2**(self.depth-1)

        if type == 'train':
            dropout_probability = 0.2
        elif type == 'test':
            dropout_probability = 0.0
        
        if submodule_type == 0:
            self.encoder_class = Conv2dBLD_E
            self.decoder_class = Conv2dBLD_D
        elif submodule_type == 1:
            self.encoder_class = Conv2dMBLD_E
            self.decoder_class = Conv2dMBLD_D
        elif submodule_type == 2:
            self.encoder_class = DoubleConv2dMBLD_E
            self.decoder_class = DoubleConv2dMBLD_D

        # Encoder
        self.encoder = nn.ModuleList(
            [self.encoder_class(self.output_channel_list[i], self.output_channel_list[i+1], 
                                kernel_size=3, stride=2, padding=1, leakiness=leakiness, 
                                dropout_probability=dropout_probability) 
                                for i in range(self.depth-1)]
        )

        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.output_channel_list[-1] * bottom_res * bottom_res, feature_vector_size), 
            nn.BatchNorm1d(feature_vector_size),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
            nn.Linear(feature_vector_size, self.output_channel_list[-1] * bottom_res * bottom_res),
            nn.BatchNorm1d(self.output_channel_list[-1] * bottom_res * bottom_res),
            nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
            nn.Unflatten(1, (int(self.output_channel_list[-1]), bottom_res, bottom_res)),
        )

        # Decoder
        self.decoder = nn.ModuleList(
            [self.decoder_class(self.inverse_output_channel_list[i], self.inverse_output_channel_list[i+1], 
                                kernel_size=3, stride=2, padding=1, output_padding=1, leakiness=leakiness, 
                                dropout_probability=dropout_probability) 
                                for i in range(self.depth-1)]
        )

    def forward(self, input, strategy='whole'):
        if strategy == 'whole':
            x = input
            for i in range(self.depth-1):
                x = self.encoder[i](x)
            x = self.bottleneck(x)
            for i in range(self.depth-1):
                x = self.decoder[i](x)
            
        if strategy == 'skip_bottleneck':
            x = input
            for i in range(self.depth-1):
                x = self.encoder[i](x)
            for i in range(self.depth-1):
                x = self.decoder[i](x)
        
        return x   


# leakiness = 0.01

# input_res = 256
# feature_vector_size = 512

# L0 = 2
# L1 = 8
# L2 = L1 * 2
# L3 = L2 * 2
# L4 = L3 * 2
# L5 = L4 * 2
# L6 = L5 * 2

# LM=L6

# feature_size = input_res // 2**6

# class ConvAutoencoder(nn.Module):
    
#     def __init__(self, type='train', depth = 6):
#         super(ConvAutoencoder, self).__init__()

#         if type == 'train':
#             dropout_probability = 0.2
#         elif type == 'test':
#             dropout_probability = 0.0

#         self.feature_maps = {}

#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(L0, L1, kernel_size=3, stride=2, padding=1), 
#             nn.BatchNorm2d(L1),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.Conv2d(L1, L2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(L2),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.Conv2d(L2, L3, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(L3),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.Conv2d(L3, L4, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(L4),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.Conv2d(L4, L5, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(L5),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.Conv2d(L5, L6, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(L6),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
#         )

#         self.bottleneck = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(LM * feature_size * feature_size, feature_vector_size), 
#             nn.BatchNorm1d(feature_vector_size),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
#             nn.Linear(feature_vector_size, LM * feature_size * feature_size),
#             nn.BatchNorm1d(LM * feature_size * feature_size),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),
#             nn.Unflatten(1, (LM, feature_size, feature_size)),
#         )

#         # Decoder
#         self.decoder = nn.Sequential(
            
#             nn.ConvTranspose2d(L6, L5, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(L5),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.ConvTranspose2d(L5, L4, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(L4),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.ConvTranspose2d(L4, L3, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(L3),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.ConvTranspose2d(L3, L2, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(L2),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.ConvTranspose2d(L2, L1, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(L1),
#             nn.LeakyReLU(leakiness), nn.Dropout(dropout_probability),

#             nn.ConvTranspose2d(L1, L0, kernel_size=3, stride=2, padding=1, output_padding=1),
#         )

#     def forward(self, input, strategy='whole'):
#         if strategy == 'whole':
#             x = self.encoder(input)
#             x = self.bottleneck(x)
#             x = self.decoder(x)
            
#         if strategy == 'skip_bottleneck':
#             x = self.encoder(input)
#             x = self.decoder(x)
        
#         return x
        
#     def get_activation(self, name):
#         def hook(model, input, output):
#             self.feature_maps[name] = output.detach()
#         return hook