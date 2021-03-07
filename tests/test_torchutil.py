"""
Tests for torch DSP utils.
"""

import numpy as np
import torch
import torch.tensor as T

import torchsynth.util as util


class TestTorchUtil:
    """
    Tests for util methods
    """

    def test_reverse_signal(self):
        signal = np.arange(10)
        tensor_signal = T(signal)
        tensor_reversed = util.reverse_signal(tensor_signal)
        assert torch.all(tensor_reversed.eq(T([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])))

    def test_sinc(self):
        x = np.linspace(-4, 4, 41)
        numpy_sinc = np.sinc(x / np.pi)
        torch_sinc = util.sinc(T(x).float())
        assert np.allclose(numpy_sinc, torch_sinc.numpy())

    def test_blackman(self):
        length = 128
        torch_blackman = torch.blackman_window(length, False)
        blackman_2 = util.blackman(T(length).float())
        assert np.allclose(blackman_2.numpy(), torch_blackman.numpy(), atol=1e-07)

    def test_fix_length2D(self):
        signal1 = torch.rand([2, 88100])
        assert util.fix_length2D(signal1, length=44100).shape == (2, 44100)
        assert util.fix_length2D(signal1, length=90000).shape == (2, 90000)
        signal2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
        assert torch.all(util.fix_length2D(signal2, length=4) == torch.tensor(
            [[1, 2, 3, 0], [4, 5, 6, 0]]
        ))
        assert torch.all(util.fix_length2D(signal2, length=2) == torch.tensor(
            [[1, 2], [4, 5]]
        ))