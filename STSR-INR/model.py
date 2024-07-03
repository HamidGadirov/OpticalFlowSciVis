import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init
import time
import math
from skimage.transform import resize
#------------------test module-----------------------#
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Sine(nn.Module):
    def __init(self):
        super(Sine,self).__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30,use_bn=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.use_bn = use_bn
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        if self.use_bn:
            return torch.sin(self.bn(self.omega_0 * self.linear(input)))
        else:
            return torch.sin(self.omega_0 * self.linear(input))

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,nonlinearity='sine',use_bn=False,omega_0=30):
        super(ResBlock,self).__init__()
        nls_and_inits = {'sine':Sine(),
                         'relu':nn.ReLU(inplace=True),
                         'sigmoid':nn.Sigmoid(),
                         'tanh':nn.Tanh(),
                         'selu':nn.SELU(inplace=True),
                         'softplus':nn.Softplus(),
                         'elu':nn.ELU(inplace=True)}

        self.nl = nls_and_inits[nonlinearity]

        self.net = []

        self.net.append(SineLayer(in_features,out_features,omega_0=omega_0))

        self.net.append(SineLayer(out_features,out_features,use_bn=use_bn,omega_0=omega_0))

        self.flag = (in_features!=out_features)

        if self.flag:
            self.transform = SineLayer(in_features,out_features,use_bn=use_bn,omega_0=omega_0)

        self.net = nn.Sequential(*self.net)
    
    def forward(self,features):
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5*(outputs+features)
    
    
class Body(nn.Module):
    #A fully connected neural network that also allows swapping out the weights when used with a hypernetwork. Can be used just as a normal neural network though, as well.
    def __init__(self, in_features, init_features=64,num_res=5,omega_0=30):
        super(Body,self).__init__()
        self.num_res = num_res

        self.net = []
        self.net.append(SineLayer(in_features,init_features,omega_0=omega_0))
    
        self.net.append(SineLayer(init_features,2*init_features,omega_0=omega_0))

        self.net.append(SineLayer(2*init_features,4*init_features,omega_0=omega_0))


        for i in range(self.num_res):
            self.net.append(ResBlock(4*init_features,4*init_features))


        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


    
class STSR_INR(nn.Module):
    def __init__(self, in_coords_dims=4, out_features=1, init_features=64,num_res=5,outermost_linear=True,embedding_dims=256,omega_0=30):
        super(STSR_INR,self).__init__()
        self.in_coords_dims = in_coords_dims
        self.layer_num = num_res+3 # Middle ResBlock + Head ResBlocks + tail Blocks
        self.out_features = out_features
        self.Modulated_Net = Body(in_features=embedding_dims,init_features=init_features,num_res=num_res,omega_0=omega_0)
        self.Synthesis_Net = Body(in_features=in_coords_dims,init_features=init_features,num_res=num_res,omega_0=omega_0)
        self.final_layers = nn.ModuleList([Head(4*init_features,outermost_linear=outermost_linear) for _ in range(out_features)])

    def forward(self,coords,latent):
        latent_code = self.Modulated_Net.net[0](latent)
        coords_feat = self.Synthesis_Net.net[0](coords)
        for i in range(1,self.layer_num):
            coords_feat = self.Synthesis_Net.net[i](coords_feat*latent_code)
            latent_code = self.Modulated_Net.net[i](latent_code)
        if (self.out_features==1):
            output = self.final_layers[0](coords_feat,latent_code)
        else:
            output = torch.cat([self.final_layers[j](coords_feat,latent_code).reshape(-1,1) for j in range(self.out_features)],dim=-1)
        return output

    
class Head(nn.Module):
    def __init__(self,feature_dims,outermost_linear,output_features=1,omega_0=30):
        super(Head,self).__init__()
        self.num_layers = 3
        self.synthesis_net = nn.ModuleList([ResBlock(feature_dims,feature_dims,omega_0=omega_0),
                                            SineLayer(feature_dims,feature_dims//2,omega_0=omega_0),
                                            SineLayer(feature_dims//2,feature_dims//4,omega_0=omega_0)
                                            ])
        self.modulator_net = nn.ModuleList([ResBlock(feature_dims,feature_dims,omega_0=omega_0),
                                            SineLayer(feature_dims,feature_dims//2,omega_0=omega_0),
                                            SineLayer(feature_dims//2,feature_dims//4,omega_0=omega_0)
                                            ])
        if outermost_linear:
            self.final_layer = nn.Sequential(nn.Linear(feature_dims//4,output_features),
                                             nn.Tanh())
        else:
            self.final_layer = SineLayer(feature_dims//4,output_features,omega_0=omega_0)
            
    def forward(self,feature,latent):
        for i in range(self.num_layers):
            feature = self.synthesis_net[i](feature*latent)
            latent = self.modulator_net[i](latent)
        out = self.final_layer(feature*latent)
        return out

class VarVADEmbedding(nn.Module):
    def __init__(self,embedding_dims=256,embedding_nums=90):
        super(VarVADEmbedding,self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(embedding_nums, embedding_dims))
        self.weight_logvar = nn.Parameter(torch.ones_like(self.weight_mu)*0.001,requires_grad=False)
        self.dim = embedding_dims
        self.embedding_nums = embedding_nums
        self.reset_parameters()
    
    def reset_parameters(self):

        mu_init_std = 1.0 / np.sqrt(self.dim)

        torch.nn.init.normal_(
            self.weight_mu.data,
            0.0,
            mu_init_std,
        )
        
    def kl_loss(self):
        kl_loss = 0.5 * torch.sum(torch.exp(self.weight_logvar) + self.weight_mu**2 - 1. - self.weight_logvar)/self.embedding_nums
        return kl_loss
    
    def forward(self,query_index,train=True):
        noise = torch.randn_like(self.weight_logvar[query_index]) * torch.exp(0.5 * self.weight_logvar[query_index])
        x = self.weight_mu[query_index]+noise if train else self.weight_mu[query_index]
        return x
    