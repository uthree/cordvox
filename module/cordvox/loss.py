import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)


def multiscale_stft_loss(x: torch.Tensor, y: torch.Tensor, scales=[8, 16, 32, 64, 128, 256, 512]):
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

        loss += F.l1_loss(safe_log(x_spec), safe_log(y_spec)) + F.mse_loss(x_spec, y_spec)
    return loss / num_scales



def discriminator_adversarial_loss(real_logits, fake_logits, real_dirs, fake_dirs):
    loss = 0.0
    for lr, lf, dr, df,  in zip(real_logits, fake_logits, real_dirs, fake_dirs):
        real_loss = ((lr - 1.0) ** 2).mean() - dr.mean()
        fake_loss = ((lf + 1.0) ** 2).mean() + df.mean()
        loss += real_loss + fake_loss
    return loss


def generator_adversarial_loss(fake_logits):
    loss = 0.0
    for dg in fake_logits:
        fake_loss = ((dg - 1.0) ** 2).mean()
        loss += fake_loss
    return loss

    
def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    for r, f in zip(fmap_real, fmap_fake):
        f = f.float()
        r = r.float()
        loss += F.l1_loss(f, r)
    return loss