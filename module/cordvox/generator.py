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
    

class HarmonicOscillator(nn.Module):
    def __init__(
            self,
            sample_rate,
            frame_size,
            num_harmonics=8,
            noise_scale=0.03
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.num_harmonics = num_harmonics
        self.noise_scale = noise_scale

        self.weights = nn.Parameter(torch.ones(1, num_harmonics, 1))

    def forward(self, f0, uv):
        '''
        Args:
            f0: fundamental frequency shape=[N, 1, L]
            uv: unvoiced=0 / voiced=1 flag, shape=[N, 1, L]
        Output
            shape=[N, 1, L * frame_size]
        '''
        with torch.no_grad():
            f0 = F.interpolate(f0, scale_factor=self.frame_size, mode='linear')
            voiced_mask = F.interpolate(uv, scale_factor=self.frame_size)
            mul = (torch.arange(self.num_harmonics, device=f0.device) + 1).unsqueeze(0).unsqueeze(2)
            fs = f0 * mul
            integrated = torch.cumsum(fs / self.sample_rate, dim=2)
            phi = torch.rand(1, self.num_harmonics, 1, device=f0.device)
            rad = 2 * math.pi * ((integrated + phi) % 1)
            noise = torch.randn_like(rad)
            harmonics = torch.sin(rad) * voiced_mask + noise * self.noise_scale
            voiced_part = harmonics + noise * self.noise_scale
            unvoiced_part = noise * 0.333
            source = voiced_part * voiced_mask + unvoiced_part * (1 - voiced_mask)
            source = (source * F.normalize(self.weights, p=2.0, dim=1)).sum(dim=1, keepdim=True)
        return source


class PeriodicHiFiGANFilter(nn.Module):
    def __init__(
            self,
            n_mels=80,
            sample_rate=24000,
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
        self.ups = nn.ModuleList()
        downs = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels//(2**i)
            c2 = upsample_initial_channels//(2**(i+1))
            p = (k-u)//2
            downs.append(weight_norm(nn.Conv1d(c2, c1, k, u, p)))
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
        downs.append(weight_norm(nn.Conv1d(1, c2, 7, 1, 3)))
        self.downs = nn.ModuleList(list(reversed(downs)))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, output_channels, 7, 1, padding=3))

        self.apply(init_weights)


    def forward(self, x, source):
        # downsamples
        s = source
        source_signals = []
        for i in range(len(self.downs)):
            s = self.downs[i](s)
            s = F.leaky_relu(s, 0.1)
            source_signals.append(s)
        source_signals = list(reversed(source_signals))

        # upsamples
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = x + source_signals[i]
            x = self.ups[i](x)
            x = F.leaky_relu(x, 0.1)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = x + source_signals[-1]
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x
    

class Generator(nn.Module):
    def __init__(
            self,
            source_module,
            filter_module,
            source_type="harmonic",
            filter_type="period_hifigan",
            min_frequency=20.0
    ):
        super().__init__()
        self.min_frequency=min_frequency

        if source_type == "harmonic":
            self.source_module = HarmonicOscillator(**source_module)
        else:
            raise "invailed source module type"
        
        if filter_type == "period_hifigan":
            self.filter_module = PeriodicHiFiGANFilter(**filter_module)

    def forward(self, x, f0):
        uv = (f0 > self.min_frequency).to(f0.dtype)
        source = self.source_module(f0, uv)
        output = self.filter_module(x, source)
        return output