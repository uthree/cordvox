import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)


def multiscale_stft_loss(x: torch.Tensor, y: torch.Tensor, scales=[16, 32, 64, 128, 256, 512], alpha=1.0, beta=1.0):
    '''
    shapes:
        x: [N, Waveform Length]
        y: [N, Waveform Length]

        Output: []
    '''
    x = x.to(torch.float)
    y = y.to(torch.float)

    loss = 0
    num_scales = len(scales)
    for s in scales:
        hop_length = s
        n_fft = s * 4
        window = torch.hann_window(n_fft, device=x.device)
        x_spec = torch.stft(x, n_fft, hop_length, return_complex=True, window=window).abs()
        y_spec = torch.stft(y, n_fft, hop_length, return_complex=True, window=window).abs()

        x_spec[x_spec.isnan()] = 0
        x_spec[x_spec.isinf()] = 0
        y_spec[y_spec.isnan()] = 0
        y_spec[y_spec.isinf()] = 0

        loss += F.l1_loss(safe_log(x_spec), safe_log(y_spec)) * alpha + F.mse_loss(x_spec, y_spec) * beta 
    return loss / num_scales


def oscillate_harmonics(f0: torch.Tensor, frame_size: int, sample_rate: float, num_harmonics: int, min_frequency):
    H = (torch.arange(num_harmonics + 1, device=f0.device) + 1)[None, :, None]
    f0 = F.interpolate(f0, scale_factor=frame_size, mode='linear') / sample_rate
    rad = torch.cumsum(f0, dim=2)
    harmonics = torch.sin(2.0 * math.pi * ((H * rad) % 1.0))
    return harmonics


def harmonic_masked_stft_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        f0: torch.Tensor,
        frame_size: int = 480,
        sample_rate: float = 24000.0,
        n_fft: int = 1920,
        num_harmonics: int = 7,
        min_frequency: float = 20.0,
        eps: float = 1e-6
    ):
    '''
    shapes:
        x: [N, L * frame_size]
        y: [N, L * frame_size]
        f0: [N, 1, frame_size]

        Output: []
    '''
    window = torch.hann_window(n_fft, device=x.device)
    x_spec = torch.stft(x, n_fft, frame_size, return_complex=True, window=window).abs()
    y_spec = torch.stft(y, n_fft, frame_size, return_complex=True, window=window).abs()
    harmonics = oscillate_harmonics(f0, frame_size, sample_rate, num_harmonics, min_frequency).sum(dim=1)
    h_spec = torch.stft(harmonics, n_fft, frame_size, return_complex=True, window=window).abs()
    loss = (torch.log((y_spec * h_spec + eps) / (x_spec * h_spec + eps)) ** 2).mean()
    return loss


# 1 = fake, 0 = real
def discriminator_adversarial_loss(real_outputs: torch.Tensor, fake_outputs: torch.Tensor):
    loss = 0
    n = min(len(real_outputs), len(fake_outputs))
    for dr, df in zip(real_outputs, fake_outputs):
        dr = dr.float()
        df = df.float()
        real_loss = (dr ** 2).mean()
        fake_loss = ((df - 1) ** 2).mean()
        loss += real_loss + fake_loss
    return loss / n


def generator_adversarial_loss(fake_outputs: torch.Tensor):
    loss = 0
    n = len(fake_outputs)
    for dg in fake_outputs:
        dg = dg.float()
        loss += (dg ** 2).mean()
    return loss / n

    
def feature_matching_loss(fmap_real: torch.Tensor, fmap_fake: torch.Tensor):
    loss = 0
    n = min(len(fmap_real), len(fmap_fake))
    for r, f in zip(fmap_real, fmap_fake):
        f = f.float()
        r = r.float()
        loss += F.huber_loss(f, r)
    return loss / n