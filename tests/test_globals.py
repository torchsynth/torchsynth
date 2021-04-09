"""
Tests for torch synths
"""


import pytest
import torch.tensor as tensor
from torch import Tensor as T

from torchsynth.default import (
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CONTROL_RATE,
    DEFAULT_SAMPLE_RATE,
)
from torchsynth.config import SynthConfig


def test_synth_globals():
    synthconfig = SynthConfig(tensor(64))
    assert synthconfig.sample_rate == DEFAULT_SAMPLE_RATE
    assert synthconfig.batch_size == 64
    assert synthconfig.buffer_size == DEFAULT_BUFFER_SIZE
    assert synthconfig.control_rate == DEFAULT_CONTROL_RATE
    assert (
        synthconfig.control_buffer_size
        == DEFAULT_BUFFER_SIZE / DEFAULT_SAMPLE_RATE * synthconfig.control_rate
    )

    # Test passing in specific values
    synthconfig = SynthConfig(
        tensor(65),
        sample_rate=tensor(16000),
        buffer_size=tensor(8000),
        control_rate=tensor(1000),
    )
    assert synthconfig.control_rate == 1000
    assert synthconfig.sample_rate == 16000
    assert synthconfig.buffer_size == 8000
    assert synthconfig.control_buffer_size == 500

    # Control rate must be passed in as a tensor
    with pytest.raises(AttributeError):
        SynthConfig(tensor(16), control_rate=1000)
