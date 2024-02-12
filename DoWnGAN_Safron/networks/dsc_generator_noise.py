# -*- coding: utf-8 -*-
"""
Adaption of Nic's generator based on ESRGAN+ model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianNoise(nn.Module):
    def __init__(self, sigma=1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

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

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.noise = GaussianNoise()

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return self.noise(out.mul(self.res_scale) + x)


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x
    
class Generator(nn.Module):
    # coarse_dim_n, fine_dim_n, n_covariates, n_predictands
    def __init__(self, filters, fine_dims, channels_coarse, channels_invariant, n_predictands=1, num_res_blocks=16, num_res_blocks_fine = 4, num_upsample=3):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(channels_coarse, filters, kernel_size=3, stride=1, padding=1)
        self.conv1f = nn.Conv2d(channels_invariant, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.res_blocksf = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks_fine)])
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
            nn.Conv2d(filters*2, filters, kernel_size=3, stride=1, padding=1),
            DenseResidualBlock(filters),
            nn.LeakyReLU(),
            nn.Conv2d(filters, n_predictands, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x_coarse, x_fine):
        out = self.LR_pre(x_coarse)
        outc = self.upsampling(out)
        #print(outc.size())

        outf = self.HR_pre(x_fine)

        out = torch.cat((outc,outf),1)
        out = self.conv3(out)
        return out
