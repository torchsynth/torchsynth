from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import tensor as T

import torchcsprng as csprng

from torchsynth import util as util
from torchsynth.globals import SynthGlobals
from torchsynth.parameter import ModuleParameterRange
from torchsynth.module import (
    ADSR,
    VCA,
    CrossfadeKnob,
    MonophonicKeyboard,
    NormalizedFaderBank,
    FmControl,
    FmVCO,
    Noise,
    SineVCO,
    SoftModeSelector,
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

    def set_parameters(self, params: Dict[Tuple, T]):
        """
        Set parameters for synth by passing in a dictionary of modules and parameters
        """
        for module_params, value in params.items():
            module = getattr(self, module_params[0])

            # Supports nesting of AbstractSynths -- pass through the parameter to
            # the set_parameters method in the child AbstractSynth
            if isinstance(module, AbstractSynth):
                module.set_parameters({module_params[1:]: value})
            else:
                module.set_parameter(module_params[1], value.to(self.device))

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

    def test_step(self, batch, batch_idx):
        """
        This is boilerplate for lightning -- this is required by lightning Trainer
        when calling test, which we use to forward Synths on multi-gpu platforms
        """
        return T(0.0, device=self.device)

    def randomize(self, seed: Optional[int] = None):
        """
        Randomize all parameters
        """
        if seed is not None:
            # Profile to make sure this isn't too slow?
            mt19937_gen = csprng.create_mt19937_generator(seed)
            for parameter in self.parameters():
                parameter.data.uniform_(0, 1, generator=mt19937_gen)
            for module in self._modules:
                self._modules[module].seed = seed
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
                ("keyboard", MonophonicKeyboard(synthglobals)),
                ("pitch_adsr", ADSR(synthglobals)),
                ("amp_adsr", ADSR(synthglobals)),
                ("vco_1", SineVCO(synthglobals)),
                ("vco_2", SquareSawVCO(synthglobals)),
                ("noise", Noise(synthglobals)),
                ("vca", VCA(synthglobals)),
                ("vco_ratio", CrossfadeKnob(synthglobals)),
            ]
        )

    def _forward(self) -> T:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        midi_f0, note_on_duration = self.keyboard()
        pitch_envelope = self.pitch_adsr.forward(note_on_duration)
        amp_envelope = self.amp_adsr.forward(note_on_duration)

        vco_1_out = self.vco_1.forward(midi_f0, pitch_envelope)
        vco_2_out = self.vco_2.forward(midi_f0, pitch_envelope)
        audio_out = util.crossfade2D(vco_1_out, vco_2_out, self.vco_ratio.p("ratio"))

        audio_out = self.noise.forward(audio_out)
        return self.vca.forward(amp_envelope, audio_out)


class FmOperator(AbstractSynth):
    """
    An FM operator - a single unit containing a sine wave oscillator that can receive
    an audio rate frequency modulation signal and contains an amplitude envelope
    """

    def __init__(self, synthglobals: SynthGlobals, *args, **kwargs):
        AbstractSynth.__init__(self, synthglobals=synthglobals, *args, **kwargs)

        # Register all modules as children
        self.add_synth_modules(
            [
                ("osc", FmVCO(synthglobals)),
                ("env", ADSR(synthglobals)),
                ("amp", VCA(synthglobals)),
            ]
        )

    def _forward(self, midi_f0: T, duration: T, modulation: Signal) -> Signal:
        output = self.osc(midi_f0, modulation)
        env = self.env(duration)
        return self.amp(env, output)


class FmVoice(AbstractSynth):
    """
    A four operator FM synth.

    TODO: Add an LFO and maybe a pitch envelope? Try to implement DX7 style envs?
    """

    def __init__(self, synthglobals: SynthGlobals, *args, **kwargs):
        AbstractSynth.__init__(self, synthglobals=synthglobals, *args, **kwargs)

        # Register children SynthModules
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard(synthglobals)),
                ("modulation", FmControl(synthglobals)),
                ("op2_inputs", NormalizedFaderBank(synthglobals, [0.5])),
                ("op3_inputs", NormalizedFaderBank(synthglobals, [0.25, 0.5])),
                ("op4_inputs", NormalizedFaderBank(synthglobals, [0.25, 0.33, 0.5])),
                ("mixer", SoftModeSelector(synthglobals, n_modes=4)),
            ]
        )

        # Add the operators
        self.op1 = FmOperator(synthglobals)
        self.op2 = FmOperator(synthglobals)
        self.op3 = FmOperator(synthglobals)
        self.op4 = FmOperator(synthglobals)

    def _forward(self) -> T:

        # Trigger keyboard
        midi_f0, note_on_duration = self.keyboard()

        # No modulation for the first operator
        modulation = torch.zeros(
            (self.batch_size, self.buffer_size), device=self.device
        )

        # First operator
        op1_out = self.op1(
            midi_f0=midi_f0, duration=note_on_duration, modulation=modulation
        )

        # Second operator
        op2_out = self.op2(
            midi_f0=midi_f0,
            duration=note_on_duration,
            modulation=op1_out * self.op2_inputs()[:, 0].unsqueeze(1),
        )

        # Third operator
        op3_mix = self.op3_inputs().unsqueeze(2)
        op3_out = self.op3(
            midi_f0=midi_f0,
            duration=note_on_duration,
            modulation=op1_out * op3_mix[:, 0] + op2_out * op3_mix[:, 1],
        )

        # Fourth operator
        op4_mix = self.op4_inputs().unsqueeze(2)
        op4_out = self.op4(
            midi_f0=midi_f0,
            duration=note_on_duration,
            modulation=op1_out * op4_mix[:, 0]
            + op2_out * op4_mix[:, 1]
            + op3_out * op4_mix[:, 2],
        )

        # Mix all the operators to the output
        mix_weights = self.mixer().unsqueeze(2)
        return (
            op1_out * mix_weights[0]
            + op2_out * mix_weights[1]
            + op3_out * mix_weights[2]
            + op4_out * mix_weights[3]
        ).as_subclass(Signal)
