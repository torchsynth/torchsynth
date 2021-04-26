from typing import Optional
import torch
from torchsynth.module import ADSR, LFO, MonophonicKeyboard, SquareSawVCO, VCA
from torchsynth.synth import AbstractSynth
from torchsynth.config import SynthConfig


class SimpleSynth(AbstractSynth):
    def __init__(self, synthconfig: Optional[SynthConfig] = None):
        super().__init__(synthconfig=synthconfig)

        # Define the modules that will be used
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr", ADSR),
                ("lfo", LFO),
                ("vco", SquareSawVCO),
                ("vca", VCA),
            ]
        )

    def output(self) -> torch.Tensor:
        no_mod = torch.zeros(())
        midi_f0, note_on_duration = self.keyboard()
        envelope = self.adsr(note_on_duration)
        pitch_mod = self.lfo()
        output = self.vco(
            midi_f0,
        )
        return envelope


synth = SimpleSynth()

synth(0)
