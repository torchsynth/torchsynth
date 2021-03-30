"""
Tests for torch synths
"""


import pytest
import torch.nn
import torch.tensor as T

from torchsynth.globals import SynthGlobals
from torchsynth.default import DEFAULT_SAMPLE_RATE, DEFAULT_BUFFER_SIZE


def test_synth_globals():
    synthglobals = SynthGlobals(T(64))
    assert synthglobals.sample_rate == DEFAULT_SAMPLE_RATE
    assert synthglobals.batch_size == 64
    assert synthglobals.buffer_size == DEFAULT_BUFFER_SIZE
    assert synthglobals.control_rate == DEFAULT_SAMPLE_RATE // 100
    assert (
        synthglobals.control_buffer_size
        == DEFAULT_BUFFER_SIZE / DEFAULT_SAMPLE_RATE * synthglobals.control_rate
    )

    synthglobals = SynthGlobals(T(65), control_rate=T(1000))
    assert synthglobals.control_rate == 1000

    with pytest.raises(AttributeError):
        SynthGlobals(T(16), control_rate=1000)
