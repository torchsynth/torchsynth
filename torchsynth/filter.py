"""
Time-varying filters.
"""

import math
from typing import Dict, Optional

import torch
import torch.tensor as T

import torchsynth.util as util
from torchsynth.config import SynthConfig
from torchsynth.module import SynthModule
from torchsynth.signal import Signal


class TimeVaryingFIRBase(SynthModule):
    """
    A base class for time-varying FIR filters that are computed using fast convolution.
    Deriving classes must implement `get_impulse_matrix` which creates a matrix of
    impulses based on modulation signals.

    Args:
        synthconfig: An object containing synthesis settings that are shared
            across all modules, typically specified by
            :class:`~torchsynth.synth.Voice`, or some other, possibly custom
            :class:`~torchsynth.synth.AbstractSynth` subclass.
        device: An object representing the device on which the `torch` tensors
            are to be allocated (as per PyTorch, broadly).
        frame_size: input signals are split into non-overlapping frames of this size.
        filter_len: the length of filter impulse responses. Generally, longer filters
            will provide more accurate frequency response, especially with low frequency
            content. However, shorter filters will be faster.
        kwargs: keyword args to pass to base :class:`~torchsynth.module.SynthModule`
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        frame_size: Optional[int] = 64,
        filter_len: Optional[int] = 256,
        **kwargs: Dict[str, T]
    ):
        super().__init__(synthconfig, device, **kwargs)
        self.frame_size = frame_size
        self.filter_len = filter_len

        # Resulting signal length for acyclic convolution
        # https://ccrma.stanford.edu/~jos/ReviewFourier/FFT_Convolution.html
        output_size = self.frame_size + self.filter_len - 1
        self.fft_size = util.next_power_of_two(output_size)

    def output(self, signal: Signal, *args: Signal) -> Signal:
        """
        Applies filtering to input signal.

        Args:
            signal: a batch of audio signals to be filtered
            *args: modulation signals that will modify how impulses are generated,
                these are downsampled so there is one value per frame_size number of
                input samples, then passed through to `get_impulse_matrix`.

        Returns:
            Filter audio
        """
        n_samples = signal.shape[1]
        for mod_signal in args:
            assert signal.shape == mod_signal.shape

        # Cut up the input signal into frames
        framed_signal = self.frame_signal(signal)

        # Downsample the modulation signals and create impulse response matrix
        mod_signals = [mod[:, :: self.frame_size] for mod in args]
        impulse_matrix = self.get_impulse_matrix(*mod_signals)

        # Convolve the frames of input signals with the time-varying impulses
        convolved_frames = self.fast_convolve(framed_signal, impulse_matrix)
        y = self.overlap_add(convolved_frames)

        # Remove samples at beginning delay caused by convolution and samples at the
        # end to fix to length of input signal.
        delay = self.fft_size // 2
        y = y[:, delay : n_samples + delay]

        return y

    def frame_signal(self, signal: Signal) -> T:
        """
        Chop up a signal into non-overlapping frames with self.frame_size length.
        Zero-pads before slicing if an even number of frames can't be computed

        Args:
            signal: input signal to slice into frames

        Returns:
            A tensor of shape (batch_size, num_frames, frame_size)
        """
        pad = self.frame_size - (signal.shape[1] % self.frame_size)
        if pad != self.frame_size:
            signal = torch.nn.functional.pad(signal, (0, pad))

        num_frames = signal.shape[1] // self.frame_size
        return signal.view(self.batch_size, num_frames, self.frame_size)

    def fast_convolve(self, input1: T, input2: T) -> T:
        """
        Multiply spectra of two signals. Returns array of overlapping frames.

        Args:
            input1: tensor of audio frames.
            input2: tensor of audio frames.

        Returns:
            A tensor with shape (batch_size, num_frames, fft_size).
        """
        # Must have the same batch_size and number of frames
        assert input1.shape[0] == input2.shape[0]
        assert input1.shape[1] == input2.shape[1]

        # The size of the fft must be larger than the output size
        # resulting from acyclic convolution of the two frames sizes
        assert self.fft_size >= input1.shape[2] + input1.shape[2] - 1

        # Perform FFT convolution
        fft1 = torch.fft.rfft(input1, n=self.fft_size)
        fft2 = torch.fft.rfft(input2, n=self.fft_size)
        return torch.fft.irfft(fft1 * fft2)

    def overlap_add(self, frames: T) -> T:
        """
        Takes a set of overlapping and adds them back together using the original
        frame_size that was used to chop them up prior to convolution.

        Args:
            frames: Tensor of frames that have a shape
                (batch_size, num_frames, fft_size). This will be overlapped and added
                together using the input frame_size.

        Returns:
            A tensor of signals that have been reconstructed with overlap add
        """
        num_frames, fft_size = frames.shape[1:]
        assert self.fft_size == fft_size

        # Need to swap the frame / signal dimensions for strides to be calculated
        # in the correct direction with torch.nn.functional.fold
        frames = frames.swapaxes(1, 2)

        # The resulting output signals will be the length of frame_size x num_frames,
        # which is the length of the input signal prior to convolution, plus the
        # additional signal that is added by the acyclic convolution process.
        output_len = self.frame_size * num_frames + self.fft_size - self.frame_size

        # We can use fold to achieve overlap add. This expects a batch of images, so we
        # fake that for audio by setting one of the image dimensions to 1. The fft_size
        # is the kernel size and the frame_size used to initially split the input signal
        # is the hop_length, or stride.
        frames = torch.nn.functional.fold(
            frames, (1, output_len), (1, self.fft_size), stride=(1, self.frame_size)
        )

        # Fold returns a 4D tensor with two dimensions with a size of 1, flatten those.
        frames = frames.squeeze(1).squeeze(1)

        return frames

    def get_impulse_matrix(self, *args: Signal) -> T:
        """
        Make a matrix of impulse responses for time-varying filter based
        on the input modulation signals. Must be implemented by children.
        The type of impulses return will define what kind of filter this is.

        Args:
            *args: modulation signals to use to calculate the impulses

        Returns:
            A matrix of impulse responses of shape `(num_frames x filter_len)`
        """
        raise NotImplementedError


class LowPassSinc(TimeVaryingFIRBase):
    """
    Implements a lowpass filter using the the windowed sinc method.

    Args:
    synthconfig: An object containing synthesis settings that are shared
        across all modules, typically specified by
        :class:`~torchsynth.synth.Voice`, or some other, possibly custom
        :class:`~torchsynth.synth.AbstractSynth` subclass.
    device: An object representing the device on which the `torch` tensors
        are to be allocated (as per PyTorch, broadly).
    frame_size: input signals are split into non-overlapping frames of this size.
    filter_len: the length of filter impulse responses. Generally, longer filters
        will provide more accurate frequency response, especially with low frequency
        content. However, shorter filters will be faster.
    kwargs: keyword args to pass to base :class:`~torchsynth.module.SynthModule`
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        frame_size: Optional[int] = 64,
        filter_len: Optional[int] = 256,
        **kwargs: Dict[str, T]
    ):
        super().__init__(synthconfig, device, frame_size, filter_len, **kwargs)

        # Normalized windowing function to be applied to impulses
        window = torch.hamming_window(self.filter_len, device=self.device)
        window = window / math.sqrt(self.filter_len)
        self.register_buffer("window", window)

        # Timestamp buffer for calculating sinc impulses
        half_frame_time = (self.filter_len / self.sample_rate) / 2
        t = torch.arange(self.filter_len, device=self.device) / self.sample_rate
        t = t - half_frame_time
        self.register_buffer("window_time", t)

    def get_impulse_matrix(self, cutoff: T) -> T:
        """
        Make matrix of impulse responses for time-varying LP filter.

        Args:
            cutoff: time-varying cutoff values in Hz

        Returns:
            A matrix of impulse responses of shape `(num_frames x filter_len)`
        """
        cutoff = cutoff.unsqueeze(2)
        return torch.sinc(2 * cutoff * self.window_time) * self.window
