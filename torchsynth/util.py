"""
Utility functions for torch DSP related things
"""

import torch
from torch import Tensor as T

from torchsynth.signal import Signal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


def midi_to_hz(midi: T) -> T:
    """
    Convert from midi (linear pitch) to frequency in Hz.

    Args:
        midi: Linear pitch on the MIDI scale.

    Return:
        Frequency in Hz.
    """
    return 440.0 * (torch.exp2((midi - 69.0) / 12.0))


def fix_length(signal: Signal, length: T) -> Signal:
    """
    Pad or truncate :class:`~torchsynth.Signal` to specified length.
    """
    assert length.ndim == 0
    assert signal.ndim == 2
    if signal.num_samples < length:
        signal = torch.nn.functional.pad(signal, (0, length - signal.num_samples))
    elif signal.num_samples > length:
        signal = signal[:, :length]
    assert signal.shape == (signal.batch_size, length)
    return signal


def normalize_if_clipping(signal: Signal) -> Signal:
    """
    Only normalize invidiaul signals in batch that have samples
    less than -1.0 or greater than 1.0
    """
    max_sample = torch.max(torch.abs(signal), dim=1, keepdim=True)[0]
    return torch.where(max_sample > 1.0, signal / max_sample, signal)


def normalize(signal: Signal) -> Signal:
    """
    Normalize every individual signal in batch.
    """
    max_sample = torch.max(torch.abs(signal), dim=1, keepdim=True)[0]
    return signal / max_sample


# Here are some old functions that we are not currently using but
# might want in the future.
# # What is amin here? And maybe we should convert it to a value in defaults?
# # What is the range of amplitude?
# def amplitude_to_db(amplitude: T, amin: T = tensor(1e-10)) -> T:
#     """
#     Convert an amplitude value to decibels
#     """
#     return 20 * torch.log10(torch.max(amplitude, amin))
#
#
# def db_to_amplitude(db: T) -> T:
#     """
#     Convert decibel value to an amplitude between 0 and 1
#     """
#     return torch.pow(10, db / 20)
#
#
# def peak_gain_for_Q(Q: T) -> T:
#     """
#     Calculate the peak gain for a given filter quality factor.
#     """
#     # No gain added for quality factor less then 1/sqrt(2)
#     if Q <= 0.707:
#         return tensor(1.0)
#
#     return Q * Q / torch.pow((Q * Q - 0.25), 0.5)
#
#
# def hz_to_midi(hz: T) -> T:
#     """
#     Convert from frequency in Hz to midi (linear pitch).
#     """
#     return 12 * torch.log2((hz + EPS) / 440) + 69
#
#
# def crossfade(in_1: Signal, in_2: Signal, ratio: T) -> Signal:
#     """
#     Equal power cross-fade.
#
#     TODO: Replace crossfade with this once everything is 2D
#     """
#     assert in_1.ndim == 2 and in_2.ndim == 2 and ratio.ndim == 1
#     assert torch.all(0.0 <= ratio) and torch.all(ratio <= 1.0)
#     ratio = ratio.unsqueeze(1)
#     return EQ_POW * (torch.sqrt(1 - ratio) * in_1 + torch.sqrt(ratio) * in_2)
#
#
# def reverse_signal(signal: T) -> T:
#     assert signal.ndim == 1
#     return torch.flip(signal, (0,))
#
#
# def linspace(start: T, stop: T, num: T, endpoint: T = False) -> T:
#     """
#     Wrapper for torch.linspace that allows to count to `stop` non-inclusive.
#     """
#
#     # Need to use `==` rather than `is` for correct behaviour w/ tensors.
#     if endpoint == False and num != 0:  # noqa: E712
#         temp = stop - start
#         stop = stop - (temp / num)
#
#     return torch.linspace(start, stop, num)
#
#
# def blackman(length: T) -> T:
#     num_samples = torch.ceil(length)
#     diff = num_samples - length
#     n = torch.arange(num_samples.detach() - (diff.detach() / 2), device=length.device)
#     cos_a = torch.cos(2 * torch.pi * n / (length - 1)).to(length.device)
#     cos_b = torch.cos(4 * torch.pi * n / (length - 1)).to(length.device)
#     window = 0.42 - 0.5 * cos_a + 0.08 * cos_b
#
#     # Linearly interpolate the ends of the window to achieve fractional length
#     window = torch.cat(
#         (
#             tensor([0.0 * diff + window[0] * (1.0 - diff)], device=length.device),
#             window[1:-1],
#             tensor([0.0 * diff + window[-1] * (1.0 - diff)], device=length.device),
#         )
#     ).to(length.device)
#
#     return window
#
#
# def sinc(x: T) -> T:
#     return torch.where(
#         x == tensor(0, device=x.device), tensor(1.0, device=x.device),
#               torch.sin(x) / x
#     )
