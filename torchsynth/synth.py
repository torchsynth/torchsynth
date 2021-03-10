from typing import Dict

import torch
from torch import nn as nn
from torch import tensor as T

from torchsynth import util as util
from torchsynth.globals import SynthGlobals
from torchsynth.module import (
    ADSR,
    VCA,
    CrossfadeKnob,
    NoteOnButton,
    Noise,
    SineVCO,
    SquareSawVCO,
    SynthModule,
)


class AbstractSynth(nn.Module):
    """
    Base class for synthesizers that combine one or more SynthModules
    to create a full synth architecture.

    Parameters
    ----------
    sample_rate (int): sample rate to run this synth at
    buffer_size (int): number of samples expected at output of child modules
    """

    def __init__(
        self,
        synthglobals: SynthGlobals,
    ):
        super().__init__()
        self.synthglobals = synthglobals

    @property
    def batch_size(self) -> T:
        assert self.synthglobals.batch_size.ndim == 0
        return self.synthglobals.batch_size

    @property
    def sample_rate(self) -> T:
        assert self.synthglobals.sample_rate.ndim == 0
        return self.synthglobals.sample_rate

    @property
    def buffer_size(self) -> T:
        assert self.synthglobals.buffer_size.ndim == 0
        return self.synthglobals.buffer_size

    def add_synth_modules(self, modules: Dict[str, SynthModule]):
        """
        Add a set of named children TorchSynthModules to this synth. Registers them
        with the torch nn.Module so that all parameters are recognized.

        Parameters
        ----------
        modules (Dict): A dictionary of SynthModule0Ddeprecated
        """

        for name in modules:
            if not isinstance(modules[name], SynthModule):
                raise TypeError(f"{modules[name]} is not a SynthModule0Ddeprecated")

            if modules[name].batch_size != self.batch_size:
                raise ValueError(f"{modules[name]} batch_size does not match")

            if modules[name].sample_rate != self.sample_rate:
                raise ValueError(f"{modules[name]} sample rate does not match")

            if modules[name].buffer_size != self.buffer_size:
                raise ValueError(f"{modules[name]} buffer size does not match")

            self.add_module(name, modules[name])

    def randomize(self):
        """
        Randomize all parameters
        """
        for parameter in self.parameters():
            parameter.data = torch.rand_like(parameter)

    def forward(self) -> T:
        ...


class Voice(AbstractSynth):
    """
    In a synthesizer, one combination of VCO, VCA, VCF's is typically called a voice.
    """

    def __init__(
        self,
        synthglobals=SynthGlobals,
    ):
        super().__init__(synthglobals=synthglobals)

        # Register all modules as children
        self.add_synth_modules(
            {
                "note_on": NoteOnButton(synthglobals),
                "pitch_adsr": ADSR(synthglobals),
                "amp_adsr": ADSR(synthglobals),
                "vco_1": SineVCO(synthglobals),
                "vco_2": SquareSawVCO(synthglobals),
                "noise": Noise(synthglobals),
                "vca": VCA(synthglobals),
                "vca_ratio": CrossfadeKnob(synthglobals),
            }
        )

    def forward(self) -> T:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        note_on_duration = self.note_on.p("duration")
        pitch_envelope = self.pitch_adsr.forward(note_on_duration)
        amp_envelope = self.amp_adsr.forward(note_on_duration)

        vco_1_out = self.vco_1.forward(pitch_envelope)
        vco_2_out = self.vco_2.forward(pitch_envelope)

        audio_out = util.crossfade2D(vco_1_out, vco_2_out, self.vca_ratio.p("ratio"))
        audio_out = self.noise.forward(audio_out)

        return self.vca.forward(amp_envelope, audio_out)
