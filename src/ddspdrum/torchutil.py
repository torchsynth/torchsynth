"""
Utility functions for torch DSP related things

TODO: After everything is torch'ified, remove numpyutil.py version
and rename this to util.py.

TODO: These should operate on vectors, many of these assume scalar Tensors.
"""

import torch
import torch.tensor as T

from ddspdrum.defaults import EPSILON, EQ_POW


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


def linspace(
        start: float, stop: float, num_steps: int, endpoint: bool = False
) -> T:
    """
    Wrapper for torch.linspace that allows to count to `stop` non-inclusive.
    """
    if endpoint is False:
        temp = stop - start
        stop = stop - (temp / num_steps)

    return torch.linspace(start, stop, num_steps)
