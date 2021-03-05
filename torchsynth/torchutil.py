"""
Utility functions for torch DSP related things

TODO: After everything is torch'ified, remove numpyutil.py version
and rename this to util.py.

TODO: These should operate on vectors, many of these assume scalar Tensors.
"""

import math

import torch
import torch.tensor as T

from torchsynth.defaults import EPSILON, EQ_POW


# What is amin here? And maybe we should convert it to a value in defaults?
# What is the range of amplitude?
def amplitude_to_db(amplitude: T, amin: T = T(1e-10)) -> T:
    """
    Convert an amplitude value to decibels
    """
    return 20 * torch.log10(torch.max(amplitude, amin))


def db_to_amplitude(db: T) -> T:
    """
    Convert decibel value to an amplitude between 0 and 1
    """
    return torch.pow(10, db / 20)


def peak_gain_for_Q(Q: T) -> T:
    """
    Calculate the peak gain for a given filter quality factor.
    """
    # No gain added for quality factor less then 1/sqrt(2)
    if Q <= 0.707:
        return T(1.0)

    return Q * Q / torch.pow((Q * Q - 0.25), 0.5)


def hz_to_midi(hz: T) -> T:
    """
    Convert from frequency in Hz to midi (linear pitch).
    """
    return 12 * torch.log2((hz + EPSILON) / 440) + 69


def midi_to_hz(midi: T) -> T:
    """
    Convert from midi (linear pitch) to frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def fix_length(signal: T, length: T) -> T:
    """
    Pad or truncate array to specified length.
    """

    assert signal.ndim == 1
    if len(signal) < length:
        signal = torch.nn.functional.pad(signal, [0, length - len(signal)])
    elif len(signal) > length:
        signal = signal[:length]
    assert signal.shape == (length,)
    return signal


def crossfade(in_1: T, in_2: T, ratio: T) -> T:
    """
    Equal power cross-fade.
    """
    assert 0.0 <= ratio <= 1.0
    return EQ_POW * (torch.sqrt(1 - ratio) * in_1 + torch.sqrt(ratio) * in_2)


def linspace(start: T, stop: T, num: T, endpoint: T = False) -> T:
    """
    Wrapper for torch.linspace that allows to count to `stop` non-inclusive.
    """

    # Need to use `==` rather than `is` for correct behaviour w/ tensors.
    if endpoint == False and num != 0:  # noqa: E712
        temp = stop - start
        stop = stop - (temp / num)

    return torch.linspace(start, stop, num)


def reverse_signal(signal: T) -> T:
    assert signal.ndim == 1
    return torch.flip(signal, (0,))


def normalize(signal: T) -> T:
    max_ = torch.max(torch.abs(signal))
    assert max_.item() != 0
    return signal / max_


def sinc(x: T) -> T:
    return torch.where(x == 0, T(1., device=x.device), torch.sin(x) / x)


def blackman(length: T) -> T:
    num_samples = torch.ceil(length)
    diff = num_samples - length
    n = torch.arange(num_samples.detach() - (diff.detach() / 2), device=length.device)
    cos_a = torch.cos(2 * math.pi * n / (length - 1))
    cos_b = torch.cos(4 * math.pi * n / (length - 1))
    window = 0.42 - 0.5 * cos_a + 0.08 * cos_b

    # Linearly interpolate the ends of the window to achieve fractional length
    window = torch.cat((
        T([0.0 * diff + window[0] * (1.0 - diff)], device=length.device),
        window[1:-1],
        T([0.0 * diff + window[-1] * (1.0 - diff)], device=length.device)
    ))

    return window
