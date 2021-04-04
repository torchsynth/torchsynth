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
from torchsynth.globals import SynthGlobals


def test_synth_globals():
    synthglobals = SynthGlobals(tensor(64))
    assert synthglobals.sample_rate == DEFAULT_SAMPLE_RATE
    assert synthglobals.batch_size == 64
    assert synthglobals.buffer_size == DEFAULT_BUFFER_SIZE
    assert synthglobals.control_rate == DEFAULT_CONTROL_RATE
    assert (
        synthglobals.control_buffer_size
        == DEFAULT_BUFFER_SIZE / DEFAULT_SAMPLE_RATE * synthglobals.control_rate
    )

    # Test passing in specific values
    synthglobals = SynthGlobals(
        tensor(65),
        sample_rate=tensor(16000),
        buffer_size=tensor(8000),
        control_rate=tensor(1000),
    )
    assert synthglobals.control_rate == 1000
    assert synthglobals.sample_rate == 16000
    assert synthglobals.buffer_size == 8000
    assert synthglobals.control_buffer_size == 500

    # Control rate must be passed in as a tensor
    with pytest.raises(AttributeError):
        SynthGlobals(tensor(16), control_rate=1000)
