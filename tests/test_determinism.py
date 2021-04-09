"""
Runs tests to make sure results in Synths are deterministic
"""

import pytest
import torch
from torchsynth.synth import Voice
from torchsynth.config import SynthConfig


class TestDeterminism:
    def test_voice_determinism(self):
        # TODO make this work with different batch sizes
        synthconfig = SynthConfig(64)
        voice_1 = Voice(synthconfig)
        voice_2 = Voice(synthconfig)

        # Randomly initialized voices will have different results
        with pytest.raises(AssertionError):
            self.compare_voices(voice_1, voice_2)

        # Seeding only one voice will also have different results
        voice_1.randomize(1)
        with pytest.raises(AssertionError):
            self.compare_voices(voice_1, voice_2)

        # Now seeding the second voice the same should be the same
        voice_2.randomize(1)
        self.compare_voices(voice_1, voice_2)

        # Running voice twice in a row with the same parameters should
        # lead to the same results
        voice_1.randomize(234)
        self.compare_voices(voice_1, voice_1)

    def compare_voices(self, voice_1, voice_2):
        # Test keyboard determinism
        (midi_f0_1, duration_1) = voice_1.keyboard()
        (midi_f0_2, duration_2) = voice_2.keyboard()

        assert torch.all(midi_f0_1 == midi_f0_2)
        assert torch.all(duration_1 == duration_2)

        # Test LFO ADSR determinism
        lfo_1_rate_1 = voice_1.lfo_1_rate_adsr(duration_1)
        lfo_2_rate_1 = voice_1.lfo_2_rate_adsr(duration_1)
        lfo_1_amp_1 = voice_1.lfo_1_amp_adsr(duration_1)
        lfo_2_amp_1 = voice_1.lfo_2_amp_adsr(duration_1)

        lfo_1_rate_2 = voice_2.lfo_1_rate_adsr(duration_2)
        lfo_2_rate_2 = voice_2.lfo_2_rate_adsr(duration_2)
        lfo_1_amp_2 = voice_2.lfo_1_amp_adsr(duration_2)
        lfo_2_amp_2 = voice_2.lfo_2_amp_adsr(duration_2)

        assert torch.all(lfo_1_rate_1 == lfo_1_rate_2)
        assert torch.all(lfo_2_rate_1 == lfo_2_rate_2)
        assert torch.all(lfo_1_amp_1 == lfo_1_amp_2)
        assert torch.all(lfo_2_amp_1 == lfo_2_amp_2)

        # Compute LFOs with envelopes
        lfo_1_1 = voice_1.lfo_1(lfo_1_rate_1)
        lfo_1_2 = voice_2.lfo_1(lfo_1_rate_2)

        lfo_2_1 = voice_1.lfo_2(lfo_2_rate_1)
        lfo_2_2 = voice_2.lfo_2(lfo_2_rate_2)
        assert torch.all(lfo_1_1 == lfo_1_2)
        assert torch.all(lfo_2_1 == lfo_2_2)

        lfo_1_1 = voice_1.control_vca(lfo_1_1, lfo_1_amp_1)
        lfo_1_2 = voice_2.control_vca(lfo_1_2, lfo_1_amp_2)

        lfo_2_1 = voice_1.control_vca(lfo_2_1, lfo_2_amp_1)
        lfo_2_2 = voice_2.control_vca(lfo_2_2, lfo_2_amp_2)

        assert torch.all(lfo_1_1 == lfo_1_2)
        assert torch.all(lfo_2_1 == lfo_2_2)

        # ADSRs for Oscillators and noise
        adsr_1_1 = voice_1.adsr_1(duration_1)
        adsr_1_2 = voice_2.adsr_1(duration_2)

        adsr_2_1 = voice_1.adsr_2(duration_1)
        adsr_2_2 = voice_2.adsr_2(duration_2)

        # Mix all modulation signals
        (
            vco_1_pitch_1,
            vco_1_amp_1,
            vco_2_pitch_1,
            vco_2_amp_1,
            noise_amp_1,
        ) = voice_1.mod_matrix(adsr_1_1, adsr_2_1, lfo_1_1, lfo_2_1)

        (
            vco_1_pitch_2,
            vco_1_amp_2,
            vco_2_pitch_2,
            vco_2_amp_2,
            noise_amp_2,
        ) = voice_2.mod_matrix(adsr_1_2, adsr_2_2, lfo_1_2, lfo_2_2)
        assert torch.all(vco_1_pitch_1 == vco_1_pitch_2)
        assert torch.all(vco_1_amp_1 == vco_1_amp_2)
        assert torch.all(vco_2_pitch_1 == vco_2_pitch_2)
        assert torch.all(vco_2_amp_1 == vco_2_amp_2)
        assert torch.all(noise_amp_1 == noise_amp_2)

        # Upsample operations
        vco_1_pitch_1 = voice_1.control_upsample(vco_1_pitch_1)
        vco_1_amp_1 = voice_1.control_upsample(vco_1_amp_1)
        vco_2_pitch_1 = voice_1.control_upsample(vco_2_pitch_1)
        vco_2_amp_1 = voice_1.control_upsample(vco_2_amp_1)
        noise_amp_1 = voice_1.control_upsample(noise_amp_1)

        vco_1_pitch_2 = voice_2.control_upsample(vco_1_pitch_2)
        vco_1_amp_2 = voice_2.control_upsample(vco_1_amp_2)
        vco_2_pitch_2 = voice_2.control_upsample(vco_2_pitch_2)
        vco_2_amp_2 = voice_2.control_upsample(vco_2_amp_2)
        noise_amp_2 = voice_2.control_upsample(noise_amp_2)

        assert torch.all(vco_1_pitch_1 == vco_1_pitch_2)
        assert torch.all(vco_1_amp_1 == vco_1_amp_2)
        assert torch.all(vco_2_pitch_1 == vco_2_pitch_2)
        assert torch.all(vco_2_amp_1 == vco_2_amp_2)
        assert torch.all(noise_amp_1 == noise_amp_2)

        # Check VCOs and noise
        vco_1_1 = voice_1.vco_1(midi_f0_1, vco_1_pitch_1)
        vco_1_2 = voice_2.vco_1(midi_f0_2, vco_1_pitch_2)
        assert torch.all(vco_1_1 == vco_1_2)

        vco_2_1 = voice_1.vco_2(midi_f0_1, vco_2_pitch_1)
        vco_2_2 = voice_2.vco_2(midi_f0_2, vco_2_pitch_2)
        assert torch.all(vco_2_1 == vco_2_2)

        noise_1 = voice_1.noise()
        noise_2 = voice_2.noise()
        assert torch.all(noise_1 == noise_2)

        vco_1_1 = voice_1.vca(vco_1_1, vco_1_amp_1)
        vco_1_2 = voice_2.vca(vco_1_2, vco_1_amp_2)
        assert torch.all(vco_1_1 == vco_1_2)

        vco_2_1 = voice_1.vca(vco_2_1, vco_2_amp_1)
        vco_2_2 = voice_2.vca(vco_2_2, vco_2_amp_2)
        assert torch.all(vco_2_1 == vco_2_2)

        noise_1 = voice_1.vca(noise_1, noise_amp_1)
        noise_2 = voice_2.vca(noise_2, noise_amp_2)
        assert torch.all(noise_1 == noise_2)

        output_1 = voice_1.mixer(vco_1_1, vco_2_1, noise_1)
        output_2 = voice_2.mixer(vco_1_2, vco_2_2, noise_2)

        assert torch.all(output_1 == output_2)
