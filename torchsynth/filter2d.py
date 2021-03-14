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
    frame_size = 512


class VCF1d(Dummy):
    def __call__(self, signal: T, cutoff: T):

        # Pad or cut to match length.
        num_samples = len(cutoff) * self.frame_size
        signal = util.fix_length2D(signal, num_samples)

        framed_signal = self.make_framed(signal)
        impulse_matrix = self.get_impulse_matrix_1d(cutoff)
        convolved_frames = self.fast_convolve(framed_signal, impulse_matrix)

        y = self.overlap_add(convolved_frames)

        return y

    def make_framed(self, signal):
        frame_size = self.frame_size
        num_frames = signal.shape[1] // frame_size  # TODO
        return signal.view(num_frames, frame_size).T

    @staticmethod
    def fast_convolve(input1, input2):
        """
        Multiply spectra of two signals. Returns array of overlapping frames.
        """
        # For convenience, does not have to be strictly true for this technique.
        assert input1.shape == input2.shape

        frame_length, num_frames = input1.shape
        fft_size = util.next_power_of_two(frame_length) * 2

        # out_ = torch.zeros([fft_size, num_frames])

        fft1 = torch.fft.rfft(input1, n=fft_size, axis=0)
        fft2 = torch.fft.rfft(input2, n=fft_size, axis=0)
        tmp = fft1 * fft2

        return torch.fft.irfft(tmp, axis=0)

    @staticmethod
    def overlap_add(signal: T) -> T:
        """
        Take signal array of 2x overlapping frames and sum to 1D.
        """

        frame_size, num_frames = signal.shape

        # Assumes overlap by two. Maybe this is a bad idea for the future.
        hop_size = frame_size // 2

        samples_out = num_frames * hop_size + (frame_size - hop_size)
        out_ = torch.zeros(samples_out)

        signal = torch.roll(signal, -hop_size // 2 - 1, 0)

        a = torch.flatten(signal[hop_size:, :].T)
        b = torch.flatten(signal[:hop_size, :].T)

        out_[:-hop_size] += a
        out_[hop_size:] += b

        return out_

    def get_impulse_matrix_1d(self, cutoff: T) -> T:  # 1 x num_frames
        """
        Make matrix of impulse responses for time-varying LP filter.
        """

        window = torch.hamming_window(self.frame_size)

        hN = (self.frame_size / self.SR) / 2
        t = torch.arange(self.frame_size) / self.SR
        t = t - hN

        impulse = (2 * cutoff) * torch.sinc(2 * cutoff * t[:, None])
        impulse = impulse * window[:, None]

        return impulse


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchsynth.module import SquareSawVCO, MonophonicKeyboard
    import torchsynth.globals

    def my(signal):
        return signal.detach().numpy()

    synthglobals = torchsynth.globals.SynthGlobals(
        sample_rate=T(44100), buffer_size=T(88200), batch_size=T(1)
    )

    keyboard = MonophonicKeyboard(synthglobals, midi_f0=T([30.0]))

    square_saw = SquareSawVCO(
        tuning=T([0.0]),
        mod_depth=T([1.0]),
        shape=T([1.0]),
        synthglobals=synthglobals,
    )
    env2 = torch.zeros([1, square_saw.buffer_size])

    out_ = square_saw(keyboard.p("midi_f0"), env2)

    vcf = VCF1d()

    num_frames = 88200 // 512
    cutoff = torch.linspace(1.0, 0.0, num_frames) ** 4 * 5000 + 100

    my_out = vcf(out_, cutoff)

    x = my_out.detach().numpy()

    import IPython.display as ipd

    display(ipd.Audio(x, rate=44100))

    plt.plot(x[:20000])

num_frames
