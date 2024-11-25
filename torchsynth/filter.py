"""
Time-varying filters.
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor as T

import torchsynth.util as util
from torchsynth.config import SynthConfig
from torchsynth.module import SynthModule
from torchsynth.signal import Signal


class TimeVaryingFIRBase(SynthModule):
    """
    An abstract base class for time-varying FIR filters that are computed using fast
    convolution. Deriving classes must implement
    :meth:`~torchsynth.filter.TimeVaryingFIRBase.get_impulse_matrix`, which creates a
    matrix of impulses based on input modulation signals.

    Args:
        synthconfig: An object containing synthesis settings that are shared
            across all modules, typically specified by
            :class:`~torchsynth.synth.Voice`, or some other, possibly custom
            :class:`~torchsynth.synth.AbstractSynth` subclass.
        frame_size: input signals are split into non-overlapping frames of this size.
        filter_len: the length of filter impulse responses. Generally, longer filters
            will provide more accurate frequency response, especially with low frequency
            content. However, shorter filters will be faster.
        device: An object representing the device on which the `torch` tensors
            are to be allocated (as per PyTorch, broadly).
        kwargs: keyword args to pass to base :class:`~torchsynth.module.SynthModule`
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        frame_size: int,
        filter_len: int,
        device: Optional[torch.device] = None,
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
                these are downsampled to one value per frame_size number of
                input samples.

        Returns:
            Filtered audio.
        """
        n_samples = signal.shape[1]
        for mod_signal in args:
            assert signal.shape == mod_signal.shape

        # Cut up the input signal into frames
        framed_signal = self.frame_signal(signal)

        # Downsample the modulation signals and create impulse response matrix
        mod_signals = [mod[:, :: self.frame_size] for mod in args]
        impulse_matrix, delay = self.get_impulse_matrix(*mod_signals)

        # Convolve the frames of input signals with the time-varying impulses
        convolved_frames = self.fast_convolve(framed_signal, impulse_matrix)
        y = self.overlap_add(convolved_frames)

        # Remove samples added through the convolution process and compensate for
        # the delay added through convolution with the particular impulse response.
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
        Multiply spectra of two signals. Returns an array of overlapping frames.

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
        return torch.fft.irfft(
            torch.fft.rfft(input1, n=self.fft_size)
            * torch.fft.rfft(input2, n=self.fft_size)
        )

    def overlap_add(self, frames: T) -> T:
        """
        Reconstructs signal from array of overlapping frames. The hop-size used
        is equivalent to the frame-size used to initially separate
        the input audio into non-overlapping frames prior to convolution.

        Args:
            frames: Tensor of frames that have a shape
                (batch_size, num_frames, fft_size).

        Returns:
            A tensor of signals that have been reconstructed with overlap add.
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

    def get_impulse_matrix(self, *args: Signal) -> Tuple[T, int]:
        """
        Make a matrix of impulse responses for a time-varying filter based
        on the input modulation signals. Must be implemented by children.

        Args:
            *args: modulation signals that will modify how impulses are calculated.

        Returns:
            A list which contains 1, a matrix of impulse responses of
            shape `(batch_size, num_frames, filter_len)`, and 2, the delay that will be
            added to a signal through convolution with the impulse response.
        """
        raise NotImplementedError


class SincFilterBase(TimeVaryingFIRBase):
    """
    Abstract base class for filters using the the windowed sinc method to generate
    impulses.

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
        frame_size: Optional[int] = 256,
        filter_len: Optional[int] = 256,
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, T]
    ):
        # TODO: Would be nice to have a cutoff and mod_depth parameter -- and then have
        #   the cutoff control signal be optional (like a filter module that is
        #   optionally modulate-able). Also, maybe have the modulation CV be 0-1 range
        #   (which is what most of our control signals operate at) -- and then scale
        #   that to log frequency in a range from 0 - nyquist.

        super().__init__(synthconfig, frame_size, filter_len, device, **kwargs)

        # Windowing function to be applied to impulses
        window = torch.hamming_window(self.filter_len, device=self.device)
        self.register_buffer("window", window)

        # Timestamp buffer for calculating sinc impulses
        t = torch.arange(self.filter_len, dtype=torch.float, device=self.device)
        t = (t - self.filter_len / 2) / self.sample_rate
        self.register_buffer("window_time", t)

    def get_impulse_matrix(self, *args: Signal) -> Tuple[T, int]:
        """
        Make a matrix of impulse responses for a time-varying filter based
        on the input modulation signals. Must be implemented by children.

        Args:
            *args: modulation signals that will modify how impulses are calculated.

        Returns:
            A list which contains 1, a matrix of impulse responses of
            shape `(batch_size, num_frames, filter_len)`, and 2, the delay that will be
            added to a signal through convolution with the impulse response.
        """
        raise NotImplementedError


class LowPassSinc(SincFilterBase):
    """
    Implements a lowpass filter using the the windowed sinc method.
    """

    def get_impulse_matrix(self, cutoff: T) -> Tuple[T, int]:
        """
        Make matrix of impulse responses for time-varying LP filter using a windowed
        sinc function.

        Args:
            cutoff: time-varying cutoff values in Hz

        Returns:
            A matrix of impulse responses of shape
            `(batch_size, num_frames, filter_len)` and the delay added by convolution
            with this impulse response, which is half the length of the frame.
        """
        cutoff = cutoff.unsqueeze(2)
        cutoff = cutoff * 2
        ir = (cutoff / self.sample_rate) * torch.sinc(cutoff * self.window_time)

        return ir * self.window, self.fft_size // 2


# TODO:
#   - Add other types of sinc filters
#   - sample IRs from resonant filters and make a look-up table
#       This may be helpful: https://ieeexplore.ieee.org/document/1164348
