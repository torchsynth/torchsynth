"""
Tests for torch synths
"""


import pytest
import torch.tensor as T

from torchsynth.globals import SynthGlobals
from torchsynth.default import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_CONTROL_RATE,
)


def test_synth_globals():
    synthglobals = SynthGlobals(T(64))
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
        T(65), sample_rate=T(16000), buffer_size=T(8000), control_rate=T(1000)
    )
    assert synthglobals.control_rate == 1000
    assert synthglobals.sample_rate == 16000
    assert synthglobals.buffer_size == 8000
    assert synthglobals.control_buffer_size == 500

    # Control rate must be passed in as a tensor
    with pytest.raises(AttributeError):
        SynthGlobals(T(16), control_rate=1000)
