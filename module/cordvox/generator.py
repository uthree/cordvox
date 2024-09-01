import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)
    

class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, 1), 1)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x
    

class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))

    def forward(self, x):
        for c1 in self.convs1:
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            x = x + xt
        return x
    

class FiLM(nn.Module):
    def __init__(self, channels, condition_channels):
        super().__init__()
        self.to_beta = weight_norm(nn.Conv1d(condition_channels, channels, 1))
        self.to_gamma = weight_norm(nn.Conv1d(condition_channels, channels, 1))

    def forward(self, x, c):
        x = x * self.to_gamma(c) + self.to_beta(c)
        return x


class Generator(nn.Module):
    def __init__(
            self,
            n_mels=80,
            sample_rate=24000,
            num_harmonics=32,
            upsample_initial_channels=512,
            resblock_type="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[24, 20, 4, 4],
            upsample_rates=[12, 10, 2, 2],
            output_channels=1
        ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.sample_rate = sample_rate
        self.num_harmonics = num_harmonics
        self.frame_size = 1
        
        for u in upsample_rates:
            self.frame_size = self.frame_size * u

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.conv_pre = weight_norm(nn.Conv1d(n_mels, upsample_initial_channels, 7, 1, 3))
        self.films = nn.ModuleList()
        self.ups = nn.ModuleList()
        downs = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels//(2**i)
            c2 = upsample_initial_channels//(2**(i+1))
            p = (k-u)//2
            downs.append(weight_norm(nn.Conv1d(c2, c1, k, u, p)))
            self.films.append(FiLM(c1, c1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
        downs.append(weight_norm(nn.Conv1d(num_harmonics, c2, 7, 1, 3)))
        self.films.append(FiLM(c2, c2))
        self.downs = nn.ModuleList(list(reversed(downs)))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, output_channels, 7, 1, padding=3))

        self.apply(init_weights)


    def forward(self, x, f0):
        # oscillator
        uv = (f0 > 0.0).to(f0.dtype)
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode='linear')
        uv = (F.interpolate(uv, scale_factor=self.frame_size, mode='linear') > 0.99).to(f0.dtype)
        mul = (torch.arange(self.num_harmonics, device=f0.device) + 1).unsqueeze(0).unsqueeze(2)
        integrated = torch.cumsum(f0 / self.sample_rate, dim=2) * mul
        rad = 2 * torch.pi * (integrated % 1)
        s = torch.sin(rad) * uv

        # downsamples
        source_signals = []
        for i in range(len(self.downs)):
            s = self.downs[i](s)
            s = F.leaky_relu(s, 0.1)
            source_signals.append(s)
        source_signals = list(reversed(source_signals))

        # upsamples
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.films[i](source_signals[i], x)
            x = self.ups[i](x)
            x = F.leaky_relu(x, 0.1)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = self.films[-1](x, source_signals[-1])
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x