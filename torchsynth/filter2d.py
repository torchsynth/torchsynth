"""
Time-varying filter.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.tensor as T

import torchsynth.util as util
from torchsynth.default import EPS
from torchsynth.globals import SynthGlobals
from torchsynth.parameter import ModuleParameter, ModuleParameterRange
from torchsynth.ts_signal import Signal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

# Fast convolve

# Overlap add

# Filter design (frequency sampling).

# GLOBALS (TODO inherit from OOP hell)

import torchsynth.util as util


class Dummy:
    SR = 44100
    frame_size = 4096


class VCF1d(Dummy):
    def __call__(self, signal: T, cutoff: T):

        # Pad or cut to match length.
        num_samples = cutoff * self.frame_size
        signal = util.fix_length(num_samples)

        framed_signal = self.make_framed(signal)
        impulse_matrix = self.get_impulse_matrix_1d(cutoff)
        return framed_signal

    def make_framed(self, signal):
        frame_size = self.frame_size
        num_frames = signal.size // frame_size  # TODO
        return torch.reshape(signal, [frame_size, num_frames])

    def get_impulse_matrix_1d(self, cutoff: T) -> T:  # 1 x num_frames
        """
        Make matrix of impulse responses for time-varying LP filter.
        """

        cutoff = torch.arange(440, 880)
        window = torch.hann_window(self.frame_size)

        hN = (self.frame_size / self.SR) / 2
        t = torch.arange(self.frame_size) / self.SR
        t = t - hN

        impulse = torch.sinc(2 * cutoff * t[:, None])
        impulse = impulse * window[:, None]
        return torch.fft.fftshift(impulse)  # frame_size x num_frames


# Single impulse

# Impulse matrix

# Time varying impulse matrix

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchsynth.module import SquareSawVCO
    import torchsynth.globals

    synthglobals = torchsynth.globals.SynthGlobals(
        sample_rate=T(44100), buffer_size=T(44100), batch_size=T(1)
    )

    vco = SquareSawVCO(midi_f0=T([30]), shape=T([1]), synthglobals=synthglobals)
    out = vco(torch.ones(44100))
