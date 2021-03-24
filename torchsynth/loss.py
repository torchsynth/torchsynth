import torch

from torch import nn
from torch.nn import functional as F

"""
Multi-scale STFT loss in torch from https://github.com/vwcgaming/TTS
"""


class TorchSTFT:
    def __init__(self, n_fft, hop_length, win_length, window="hann_window"):
        """ Torch based STFT operation """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)

    def __call__(self, x):
        # B x D x T x 2
        o = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=True,
            pad_mode="reflect",  # compatible with audio.py
            normalized=False,
            onesided=True,
        )
        M = o[:, :, :, 0]
        P = o[:, :, :, 1]
        return torch.sqrt(torch.clamp(M ** 2 + P ** 2, min=1e-8))


#################################
# GENERATOR LOSSES
#################################


class STFTLoss(nn.Module):
    """ Single scale  STFT Loss """

    def __init__(self, n_fft, hop_length, win_length):
        super(STFTLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(n_fft, hop_length, win_length)

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(torch.nn.Module):
    """ Multi scale STFT loss """

    def __init__(
        self,
        n_ffts=(1024, 2048, 512),
        hop_lengths=(120, 240, 50),
        win_lengths=(600, 1200, 240),
    ):
        super(MultiScaleSTFTLoss, self).__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc
