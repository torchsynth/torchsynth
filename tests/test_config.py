"""
Tests for torch synths
"""

from torchsynth.config import DEFAULT_BATCH_SIZE, SynthConfig


def test_synth_config_debug():
    synthconfig = SynthConfig()
    assert synthconfig.debug


def test_synth_config():
    synthconfig = SynthConfig(DEFAULT_BATCH_SIZE)
    assert synthconfig.batch_size == DEFAULT_BATCH_SIZE

    # Test passing in specific values
    synthconfig = SynthConfig(
        batch_size=65,
        sample_rate=16000,
        buffer_size_seconds=0.5,
        control_rate=1000,
        reproducible=False,
    )
    assert synthconfig.control_rate == 1000
    assert synthconfig.sample_rate == 16000
    assert synthconfig.buffer_size_seconds == 0.5
    assert synthconfig.buffer_size == 8000
    assert synthconfig.control_buffer_size == 500
