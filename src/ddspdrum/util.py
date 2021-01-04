"""
Utility functions for DSP related things
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from ddspdrum.defaults import EPSILON, CONTROL_RATE, SAMPLE_RATE


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

def time_plot(signal, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, len(signal)/sample_rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

def stft_plot(signal, sample_rate=SAMPLE_RATE):
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(Xdb, sr=SAMPLE_RATE, x_axis="time", y_axis="log")
    plt.show()
