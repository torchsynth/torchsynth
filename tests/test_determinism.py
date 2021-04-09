import torch
import torchsynth.module as module
from torchsynth.synth import Voice
from torchsynth.globals import SynthGlobals


class TestDeterminism:
    def test_voice_determinism(self):

        synthglobals = SynthGlobals(torch.tensor(8))
        voice_1 = Voice(synthglobals=synthglobals)
        voice_2 = Voice(synthglobals=synthglobals)

        voice_1.randomize(50390)
        voice_2.randomize(50390)

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

        # lfo_1 = self.control_vca(self.lfo_1(lfo_1_rate), lfo_1_amp)
        # lfo_2 = self.control_vca(self.lfo_2(lfo_2_rate), lfo_2_amp)
