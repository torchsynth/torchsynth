"""
Tests for torch DSP utils.
"""

import numpy as np
import torch
import torch.tensor as T

import torchsynth.util as util
from torchsynth.signal import Signal


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

    """
    def test_blackman(self):
        length = 128
        torch_blackman = torch.blackman_window(length, False)
        blackman_2 = util.blackman(T(length).float())
        assert np.allclose(blackman_2.numpy(), torch_blackman.numpy(), atol=1e-07)
    """

    def test_fix_length(self):
        signal1 = torch.rand([2, 88100]).as_subclass(Signal)
        assert util.fix_length(signal1, length=T(44100)).shape == (2, 44100)
        assert util.fix_length(signal1, length=T(90000)).shape == (2, 90000)
        signal2 = T([[1, 2, 3], [4, 5, 6]]).float().as_subclass(Signal)
        assert torch.all(
            util.fix_length(signal2, length=T(4)) == T([[1, 2, 3, 0], [4, 5, 6, 0]])
        )
        assert torch.all(util.fix_length(signal2, length=T(2)) == T([[1, 2], [4, 5]]))

    def test_normalize_if_clipping(self):
        # Create a non-clipping signal, normalization shouldn't apply
        signal1 = torch.rand([2, 44100]).as_subclass(Signal)
        signal1_norm = util.normalize_if_clipping(signal1)
        assert signal1.eq(signal1_norm).all()

        # Now make a signal that have values greater than 1.0
        signal1[0, :] = torch.rand(44100).as_subclass(Signal) * 200.0 - 100.0
        max_val = torch.max(torch.abs(signal1))
        assert max_val > 1.0

        # Now normalize and make sure that the correct batch was normalized
        signal1_norm = util.normalize_if_clipping(signal1)
        assert torch.max(torch.abs(signal1_norm)) == 1.0
        assert signal1[1, :].eq(signal1_norm[1, :]).all()

    def test_normalize(self):
        signal = torch.rand([2, 44100]).as_subclass(Signal) * T([[100], [0.01]])
        signal_norm = util.normalize(signal)
        max_vals = torch.max(torch.abs(signal_norm), dim=1)
        assert max_vals[0][0].eq(1.0)
        assert max_vals[0][1].eq(1.0)
