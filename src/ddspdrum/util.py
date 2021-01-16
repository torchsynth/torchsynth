"""
Utility functions for DSP related things
"""

import numpy as np

from ddspdrum.defaults import EPSILON, EQ_POW


# What is amin here? And maybe we should convert it to a value in defaults?
# What is the range of amplitude?
def amplitude_to_db(amplitude: float, amin: float = 1e-10):
    """
    Convert an amplitude value to decibels
    """
    return 20 * np.log10(np.maximum(amplitude, amin))


def db_to_amplitude(db: float):
    """
    Convert decibel value to an amplitude between 0 and 1
    """
    return np.power(10, db / 20)


def peak_gain_for_Q(Q: float):
    """
    Calculate the peak gain for a given filter quality factor.
    """
    # No gain added for quality factor less then 1/sqrt(2)
    if Q <= 0.707:
        return 1.0

    return Q * Q / pow((Q * Q - 0.25), 0.5)


def hz_to_midi(hz: float):
    """
    Convert from frequency in Hz to midi (linear pitch).
    """
    return 12 * np.log2((hz + EPSILON) / 440) + 69


def midi_to_hz(midi: float):
    """
    Convert from midi (linear pitch) to frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def fix_length(signal: np.array, length: int) -> np.array:
    """
    Pad or truncate array to specified length.
    """

    assert signal.ndim == 1
    if len(signal) < length:
        signal = np.pad(signal, [0, length - len(signal)])
    elif len(signal) > length:
        signal = signal[:length]
    assert signal.shape == (length,)
    return signal


def crossfade(in_1, in_2, ratio):
    """
    Equal power cross-fade.
    """
    return EQ_POW * (np.sqrt(1 - ratio) * in_1 + np.sqrt(ratio) * in_2)
