import torch
import torch.nn as nn
import numpy as np


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = np.ones((1, 3))

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L/27

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=3, stride=1, padding=1, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL


class WavePool(nn.Module):
    def __init__(self, in_channels=3):
        super(WavePool, self).__init__()
        self.LL = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels=3):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.LL = get_wav(self.in_channels, pool=False)

    '''def forward(self, LL, LH, HL, HH, original=None):
        return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)'''
    def forward(self, LL):

        return self.LL(LL)


