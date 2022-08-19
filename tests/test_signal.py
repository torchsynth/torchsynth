"""
Tests for torch signals
"""

from copy import deepcopy

import torch

from torchsynth.signal import Signal


class TestSignal:
    """
    Tests for Signal
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_deepcopy(self):
        signal = torch.zeros(1, 1).as_subclass(Signal)
        print(deepcopy(signal))
