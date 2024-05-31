import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class SineOscillator(nn.Module):
    def __init__(
            self,
            sample_rate=24000,
            frame_size=480,
            min_frequency=20.0,
            noise_scale=0.05
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.min_frequency = min_frequency
        self.noise_scale = noise_scale

    def forward(self, f0):
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode='linear')
        uv = (f0 >= self.min_frequency).to(torch.float)
        integrated = torch.cumsum(f0 / self.sample_rate, dim=2)
        theta = 2 * math.pi * (integrated % 1)
        sinusoid = torch.sin(theta) * uv
        sinusoid = sinusoid + torch.randn_like(sinusoid) * self.noise_scale
        return sinusoid


