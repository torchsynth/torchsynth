"""
Tests for torch DSP utils.
"""

import numpy as np
import torch
import torch.tensor as T

import torchsynth.torchutil as torchutil


class TestTorchUtil:
    """
    Tests for torchutil methods
    """

    def test_reverse_signal(self):
        signal = np.arange(10)
        tensor_signal = T(signal)
        tensor_reversed = torchutil.reverse_signal(tensor_signal)
        assert torch.all(tensor_reversed.eq(T([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])))

    def test_sinc(self):
        x = np.linspace(-4, 4, 41)
        numpy_sinc = np.sinc(x / np.pi)
        torch_sinc = torchutil.sinc(T(x).float())
        assert np.allclose(numpy_sinc, torch_sinc.numpy())

    def test_blackman(self):
        length = 128
        torch_blackman = torch.blackman_window(length, False)
        blackman_2 = torchutil.blackman(T(length).float())
        assert np.allclose(blackman_2.numpy(), torch_blackman.numpy(), atol=1e-07)
