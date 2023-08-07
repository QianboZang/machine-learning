import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, input_ch, inc_input, max_freq, N_freqs):
        super().__init__()
        self.freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        self.out_dim = 2 * N_freqs * input_ch
        if inc_input:
            self.out_dim += input_ch

    def forward(self, x):
        x = x * np.pi
        sin = [torch.sin(x * freq_band) for freq_band in self.freq_bands]
        cos = [torch.cos(x * freq_band) for freq_band in self.freq_bands]
        oup = torch.cat(sin + cos, -1)
        return oup


class GaussinEmbedder(nn.Module):
    def __init__(self, input_ch, output_ch=128):
        super().__init__()
        self.B = nn.Linear(input_ch, output_ch, bias=False)
        with torch.no_grad():
            self.B.weight.normal_()
        self.B.requires_grad_(False)
        self.out_dim = output_ch * 2

    def forward(self, x):
        x = 2 * x * torch.pi
        x = self.B(x)
        oup = torch.cat([torch.sin(x), torch.cos(x)], -1)
        return oup