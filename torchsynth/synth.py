from typing import Any, List, Optional, Tuple

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import tensor as T

import torchcsprng as csprng

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
from torchsynth.signal import Signal

# https://github.com/turian/torchsynth/issues/131
# Lightning already handles this for us
# torch.use_deterministic_algorithms(True)


class AbstractSynth(LightningModule):
    """
    Base class for synthesizers that combine one or more SynthModules
    to create a full synth architecture.

    Parameters
    ----------
    sample_rate (int): sample rate to run this synth at
    buffer_size (int): number of samples expected at output of child modules
    """

    def __init__(self, synthglobals: SynthGlobals, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def add_synth_modules(self, modules: List[Tuple[str, SynthModule]]):
        """
        Add a set of named children TorchSynthModules to this synth. Registers them
        with the torch nn.Module so that all parameters are recognized.

        Parameters
        ----------
        modules List[Tuple[str, SynthModule]]: A list of SynthModules and
                                            their names.
        """

        for name, module in modules:
            if not isinstance(module, SynthModule):
                raise TypeError(f"{module} is not a SynthModule0Ddeprecated")

            if module.batch_size != self.batch_size:
                raise ValueError(f"{module} batch_size does not match")

            if module.sample_rate != self.sample_rate:
                raise ValueError(f"{module} sample rate does not match")

            if module.buffer_size != self.buffer_size:
                raise ValueError(f"{module} buffer size does not match")

            self.add_module(name, module)

    def _forward(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Each AbstractSynth should override this.
        """
        raise NotImplementedError("Derived classes must override this method")

    def forward(
        self, batch_idx: Optional[int] = None, *args: Any, **kwargs: Any
    ) -> Signal:  # pragma: no cover
        """
        Each AbstractSynth should override this.

        Parameter:
        batch_idx (Optional[int])   - If provided, we set the parameters of this
                                    synth for reproducibility, in a deterministic
                                    random way. If None (default), we just use
                                    the current module parameter settings.
        """
        if batch_idx:
            self.randomize(seed=batch_idx)
        return self._forward(*args, **kwargs)

    # For lightning
    def test_step(self, batch, batch_idx):
        assert batch.ndim == 1
        # TODO: Test with multiple lightning (not synth) batches
        _ = torch.stack([self(i) for i in batch])
        # You probably want to do something with the results above
        # We just return 0, which lightning accumulates as the test error
        return T(0.0, device=self.device)

    def randomize(self, seed: Optional[int]):
        """
        Randomize all parameters
        """
        if seed:
            # Profile to make sure this isn't too slow?
            mt19937_gen = csprng.create_mt19937_generator(seed)
            for parameter in self.parameters():
                parameter.data.uniform_(0, 1, generator=mt19937_gen)
        else:
            for parameter in self.parameters():
                parameter.data = torch.rand_like(parameter, device=self.device)


class Voice(AbstractSynth):
    """
    In a synthesizer, one combination of VCO, VCA, VCF's is typically called a voice.
    """

    def __init__(self, synthglobals: SynthGlobals, *args, **kwargs):
        AbstractSynth.__init__(self, synthglobals=synthglobals, *args, **kwargs)

        # Register all modules as children
        self.add_synth_modules(
            [
                ("note_on", NoteOnButton(synthglobals)),
                ("pitch_adsr", ADSR(synthglobals)),
                ("amp_adsr", ADSR(synthglobals)),
                ("vco_1", SineVCO(synthglobals)),
                ("vco_2", SquareSawVCO(synthglobals)),
                ("noise", Noise(synthglobals)),
                ("vca", VCA(synthglobals)),
                ("vca_ratio", CrossfadeKnob(synthglobals)),
            ]
        )

    def _forward(self) -> T:
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
