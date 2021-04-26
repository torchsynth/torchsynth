"""
Runs tests to make sure results in Synths are reproducible
"""

import pytest
import torch

from torchsynth.config import BASE_REPRODUCIBLE_BATCH_SIZE, SynthConfig
from torchsynth.synth import Voice


class TestReproducibility:
    def test_voice_reproducibility(self):
        synthconfig = SynthConfig()
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

    def test_voice_nonreproducibility(self):
        with pytest.raises(ValueError):
            SynthConfig(batch_size=BASE_REPRODUCIBLE_BATCH_SIZE + 1)

    def test_voice_batch_size_reproducibility(self):
        # Test to make sure different batch sizes produce the same results
        self.run_batch_size_test("cpu")
        if torch.cuda.is_available():
            self.run_batch_size_test("cuda")

    def run_batch_size_test(self, device):
        # Runs test for reproducibility across batch sizes on a device
        voice256 = Voice(SynthConfig(batch_size=256)).to(device)
        out256 = voice256(0)

        voice128 = Voice(SynthConfig(batch_size=128)).to(device)
        out128 = torch.vstack([voice128(0), voice128(1)])

        voice64 = Voice(SynthConfig(batch_size=64)).to(device)
        out64 = torch.vstack([voice64(0), voice64(1), voice64(2), voice64(3)])

        voice32 = Voice(SynthConfig(batch_size=32)).to(device)
        out32 = torch.vstack(
            [
                voice32(0),
                voice32(1),
                voice32(2),
                voice32(3),
                voice32(4),
                voice32(5),
                voice32(6),
                voice32(7),
            ]
        )

        assert torch.all(torch.isclose(out256, out128))
        assert torch.all(torch.isclose(out256, out64))
        # 1e-12 seems close enough to me
        assert torch.mean(torch.abs(out256 - out32)) < 1e-12

    def compare_voices(self, voice_1, voice_2):
        # Test keyboard reproducibility
        (midi_f0_1, duration_1) = voice_1.keyboard()
        (midi_f0_2, duration_2) = voice_2.keyboard()

        assert torch.all(midi_f0_1 == midi_f0_2)
        assert torch.all(duration_1 == duration_2)

        # Test LFO ADSR reproducibility
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

    def test1024_is_train(self):
        voice1024 = Voice(SynthConfig(batch_size=1024, reproducible=False)).to(device)
        _, is_train = voice1024(0)
        assert torch.all(is_train == True)
        _, is_train = voice1024(8)
        assert torch.all(is_train == True)
        _, is_train = voice1024(9)
        assert torch.all(is_train == False)
        _, is_train = voice1024(10)
        assert torch.all(is_train == True)

    def test256_is_train(self):
        voice256 = Voice(SynthConfig(batch_size=256, reproducible=False)).to(device)
        _, is_train = voice256(0 * 4)
        assert torch.all(is_train == True)
        _, is_train = voice256(8 * 4)
        assert torch.all(is_train == True)
        _, is_train = voice256(9 * 4)
        assert torch.all(is_train == False)
        _, is_train = voice256(9 * 4 + 1)
        assert torch.all(is_train == False)
        _, is_train = voice256(9 * 4 + 2)
        assert torch.all(is_train == False)
        _, is_train = voice256(9 * 4 + 3)
        assert torch.all(is_train == False)
        _, is_train = voice256(10 * 4)
        assert torch.all(is_train == True)
