"""
Time-varying filter.
"""

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

        print(impulse_matrix.shape, framed_signal.shape)
        assert framed_signal.shape == impulse_matrix.shape

        convolved_frames = self.fast_convolve(framed_signal, impulse_matrix)
        print(convolved_frames.shape)
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

        num_frames = signal.shape[1] // self.frame_size
        return signal.view(self.batch_size, num_frames, self.frame_size)

    @staticmethod
    def fast_convolve(input1, input2):
        """
        Multiply spectra of two signals. Returns array of overlapping frames.
        """
        # For convenience, does not have to be strictly true for this technique.
        assert input1.shape == input2.shape

        batch_size, num_frames, frame_length = input1.shape
        fft_size = util.next_power_of_two(frame_length) * 2

        fft1 = torch.fft.rfft(input1, n=fft_size)
        fft2 = torch.fft.rfft(input2, n=fft_size)
        tmp = fft1 * fft2

        return torch.fft.irfft(tmp)

    @staticmethod
    def overlap_add(signal: T) -> T:
        """
        Take signal array of 2x overlapping frames and sum to 1D.
        """

        # TODO need to figure out this step!
        batch_size, frame_size, num_frames = signal.shape
        print(signal.shape)

        # Assumes overlap by two. Maybe this is a bad idea for the future.
        hop_size = frame_size // 2

        samples_out = num_frames * hop_size + (frame_size - hop_size)
        print(samples_out)
        out_ = torch.zeros(samples_out)

        signal = torch.roll(signal, -hop_size // 2 - 1, 0)

        print(signal.shape)

        a = torch.flatten(signal[hop_size:, :].T)
        b = torch.flatten(signal[:hop_size, :].T)

        out_[:-hop_size] += a
        out_[hop_size:] += b

        return out_

    def get_impulse_matrix_1d(self, cutoff: T) -> T:
        """
        Make matrix of impulse responses for time-varying LP filter.
        """
        cutoff = cutoff.unsqueeze(2)
        return 2 * cutoff * torch.sinc(2 * cutoff * self.window_time) * self.window
