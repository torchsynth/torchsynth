"""
Utility functions for DSP related things
"""

import numpy as np


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

