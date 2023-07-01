#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaption of Nic's generator based on ESRGAN+ model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr

##adjusted so each block has noise added
class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, resolution = 16, res_scale=0.8, noise_sd = 5):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.resolution = resolution
        self.noise_sd = noise_sd

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters + 1)
        self.b2 = block(in_features=2 * filters + 2)
        self.b3 = block(in_features=3 * filters + 3)
        self.b4 = block(in_features=4 * filters + 4)
        self.b5 = block(in_features=5 * filters + 5, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.noise_strength = torch.nn.Parameter(torch.mul(torch.ones([]),10))

    def forward(self, x):
        noise = torch.normal(0,self.noise_sd,size = [x.shape[0], 1, self.resolution, self.resolution], device=x.device)
        #print("Noise Size",noise.size())
        inputs = torch.cat([x,noise],1)
        #print(inputs.size())
        for block in self.blocks:
            out = block(inputs)
            noise = torch.normal(0,self.noise_sd,size = [x.shape[0], 1, self.resolution, self.resolution], device=x.device)
            inputs = torch.cat([inputs, out, noise], 1)
            # inputs = torch.cat([inputs, out], 1)
            # inputs.add_(noise)
            #print(inputs.size())
        
        noise = torch.normal(0,self.noise_sd,size = [x.shape[0], 1, self.resolution, self.resolution], device=x.device)
        noiseScale = noise * self.noise_strength
        out = out.mul(self.res_scale) + x
        out.add_(noiseScale)
        return out

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, resolution = 16, res_scale=0.2, noise_sd = 5):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters, resolution=resolution, noise_sd=noise_sd), DenseResidualBlock(filters,resolution=resolution,noise_sd=noise_sd), DenseResidualBlock(filters,resolution=resolution,noise_sd=noise_sd)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    
class Generator(nn.Module):
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(self, filters, fine_dims, channels_coarse, channels_invariant, n_predictands=2, num_res_blocks=16, num_res_blocks_fine = 4, num_upsample=3):
        super(Generator, self).__init__()
        self.fine_res = fine_dims
        # First layer
        self.conv1 = nn.Conv2d(channels_coarse, filters, kernel_size=3, stride=1, padding=1)
        self.conv1f = nn.Conv2d(channels_invariant, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters,resolution=filters,noise_sd = 5) for _ in range(num_res_blocks)])
        self.res_blocksf = nn.Sequential(*[ResidualInResidualDenseBlock(filters,resolution=fine_dims,noise_sd=3) for _ in range(num_res_blocks_fine)])
        # self.res_blocksf = DenseResidualBlock(filters,resolution=fine_dims)
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        self.LR_pre = nn.Sequential(self.conv1,ShortcutBlock(nn.Sequential(self.res_blocks,self.conv2)))
        self.HR_pre = nn.Sequential(self.conv1f,ShortcutBlock(nn.Sequential(self.res_blocksf,self.conv2)))
        #self.conv2f = nn.Conv2d(fine_dims, fine_dims, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            #nn.Conv2d(fine_dims + filters, fine_dims + filters, kernel_size=1, stride=1, padding=1), ##pointwise convolution
            nn.Conv2d(filters*2, filters + 1, kernel_size=3, stride=1, padding=1),
            ResidualInResidualDenseBlock(filters + 1,resolution=fine_dims, noise_sd=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters + 1, n_predictands, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x_coarse, x_fine):
        out = self.LR_pre(x_coarse)
        outc = self.upsampling(out)
        #print(outc.size())

        outf = self.HR_pre(x_fine)
        out = torch.cat((outc,outf),1)
        # noise = torch.normal(0,2,size = [outf.shape[0], 1, self.fine_res, self.fine_res], device=outf.device)
        # out = torch.cat((outc,outf,noise),1)
        out = self.conv3(out)
        return out