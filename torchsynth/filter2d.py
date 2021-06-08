"""
Time-varying filter.
"""

import math
from typing import Any, Dict, List, Optional

import torch
import torch.tensor as T

import torchsynth.util as util
from torchsynth.config import SynthConfig
from torchsynth.module import SynthModule
from torchsynth.parameter import ModuleParameter, ModuleParameterRange
from torchsynth.signal import Signal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

# Fast convolve

# Overlap add

# Filter design (frequency sampling).

# GLOBALS (TODO inherit from OOP hell)


class VCF(SynthModule):
    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        filter_len: Optional[int] = 64,
        frame_size: Optional[int] = 64,
        **kwargs: Dict[str, T]
    ):
        super().__init__(synthconfig, device, **kwargs)
        self.frame_size = frame_size

        # Windowing function as buffer for module
        window = torch.hamming_window(self.frame_size, device=self.device)
        self.register_buffer("window", window)

        # Timestamp buffer for calculating sinc impulses
        half_frame_time = (self.frame_size / self.sample_rate) / 2
        t = torch.arange(self.frame_size, device=self.device) / self.sample_rate
        t = t - half_frame_time
        self.register_buffer("window_time", t)

    def output(self, signal: Signal, cutoff: Signal) -> Signal:

        assert signal.shape == cutoff.shape

        # Cut up the input signal into frames
        framed_signal = self.make_framed(signal)

        # Downsample the cutoff and create impulse response
        cutoff = cutoff[:, :: self.frame_size]
        impulse_matrix = self.get_impulse_matrix_1d(cutoff)
        assert framed_signal.shape == impulse_matrix.shape

        convolved_frames = self.fast_convolve(framed_signal, impulse_matrix)
        y = self.overlap_add(convolved_frames)

        return y

    def make_framed(self, signal: Signal) -> Signal:
        """
        Chop up a signal into frames with self.frame_size length.
        Zero-pads before slicing if an even number of frames can't be computed
        """

        pad = self.frame_size - signal.shape[1] % self.frame_size
        if pad != self.frame_size:
            signal = torch.nn.functional.pad(signal, (0, pad))

        print(signal.shape[1])
        num_frames = signal.shape[1] // self.frame_size
        return signal.view(self.batch_size, num_frames, self.frame_size)

    @staticmethod
    def fast_convolve(input1, input2):
        """
        Multiply spectra of two signals. Returns array of overlapping frames.
        """
        # Resulting signal length for acyclic convolution
        output_size = input1.shape[-1] + input2.shape[-1] - 1
        fft_size = util.next_power_of_two(output_size)

        fft1 = torch.fft.rfft(input1, n=fft_size)
        fft2 = torch.fft.rfft(input2, n=fft_size)
        tmp = fft1 * fft2

        return torch.fft.irfft(tmp)

    def overlap_add(self, frames: T) -> T:
        """
        Take signal array of 2x overlapping frames and sum to 1D.
        """

        batch_size, num_frames, frame_size = frames.shape

        print("Overlap start", frames.shape)
        # Need to swap the frame / signal dimensions for strides to be calculated
        # in the correct way with torch.nn.functional.fold
        frames = frames.swapaxes(1, 2)

        output_len = self.frame_size * num_frames + frame_size - self.frame_size
        frames = torch.nn.functional.fold(
            frames, (1, output_len), (1, frame_size), stride=(1, self.frame_size)
        )

        # Flatten extra single dimensions after folding
        frames = frames.squeeze(1).squeeze(1)
        print("After overlap", frames.shape)
        return frames

    def get_impulse_matrix_1d(self, cutoff: T) -> T:
        """
        Make matrix of impulse responses for time-varying LP filter.
        """
        cutoff = cutoff.unsqueeze(2)
        return 2 * cutoff * torch.sinc(2 * cutoff * self.window_time) * self.window
