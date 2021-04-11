import sys
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

if sys.version_info.major == 3 and sys.version_info.minor >= 8:
    from typing import OrderedDict as OrderedDictType
else:
    # pypi version typing_extensions doesn't yet supports OrderedDict (only master)
    # from typing_extensions import OrderedDict as OrderedDictType
    from typing import Dict as OrderedDictType

import torch
import torch.tensor as tensor
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor as T

from torchsynth.config import BATCH_SIZE_FOR_REPRODUCIBILITY, SynthConfig
from torchsynth.module import (
    ADSR,
    LFO,
    VCA,
    AudioMixer,
    ControlRateUpsample,
    ControlRateVCA,
    ModulationMixer,
    MonophonicKeyboard,
    Noise,
    SineVCO,
    SquareSawVCO,
    SynthModule,
)
from torchsynth.parameter import ModuleParameter
from torchsynth.signal import Signal


class AbstractSynth(LightningModule):
    """
    Base class for synthesizers that combine one or more SynthModules
    to create a full synth architecture.

    Args:
        sample_rate (int): sample rate to run this synth at
        buffer_size (int): number of samples expected at output of child modules
    """

    def __init__(self, synthconfig: Optional[SynthConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if synthconfig is not None:
            self.synthconfig = synthconfig
        else:
            # Use the default
            self.synthconfig = SynthConfig()

    @property
    def batch_size(self) -> T:
        assert self.synthconfig.batch_size.ndim == 0
        return self.synthconfig.batch_size

    @property
    def sample_rate(self) -> T:
        assert self.synthconfig.sample_rate.ndim == 0
        return self.synthconfig.sample_rate

    @property
    def buffer_size(self) -> T:
        assert self.synthconfig.buffer_size.ndim == 0
        return self.synthconfig.buffer_size

    @property
    def buffer_size_seconds(self) -> T:
        assert self.synthconfig.buffer_size_seconds.ndim == 0
        return self.synthconfig.buffer_size_seconds

    def add_synth_modules(
        self, modules: List[Tuple[str, SynthModule, Optional[Dict[str, Any]]]]
    ):
        """
        Add a set of named children TorchSynthModules to this synth. Registers them
        with the torch nn.Module so that all parameters are recognized.

        Args:
            modules List[Tuple[str, SynthModule, Optional[Dict[str, Any]]]]: A list of
                SynthModule classes with their names and any parameters to pass to
                their constructor.
        """

        for module_tuple in modules:
            if len(module_tuple) == 3:
                name, module, params = module_tuple
            else:
                name, module = module_tuple
                params = {}

            if not issubclass(module, SynthModule):
                raise TypeError(f"{module} is not a SynthModule")

            self.add_module(
                name, module(self.synthconfig, device=self.device, **params)
            )

    def get_parameters(
        self, include_frozen: bool = False
    ) -> OrderedDictType[Tuple[str, str], ModuleParameter]:
        """
        Returns a dictionary of ModuleParameters for this synth keyed
        on a tuple of the SynthModule name and the parameter name
        """
        parameters = []

        # Each parameter in this synth will have a unique combination of module name
        # and parameter name -- create a dictionary keyed on that.
        for module_name, module in sorted(self.named_modules()):
            # Make sure this is a SynthModule, b/c we are using ParameterDict
            # and ParameterDict is a module, we get those returned as well
            # TODO: see https://github.com/torchsynth/torchsynth/issues/213
            if isinstance(module, SynthModule):
                for parameter in module.parameters():
                    if include_frozen or not ModuleParameter.is_parameter_frozen(
                        parameter
                    ):
                        parameters.append(
                            ((module_name, parameter.parameter_name), parameter)
                        )

        return OrderedDict(parameters)

    def set_parameters(self, params: Dict[Tuple, T], freeze: Optional[bool] = False):
        """
        Set parameters for synth by passing in a dictionary of modules and parameters.
        Can optionally freeze a parameter at this value to prevent further updates.
        """
        for (module_name, param_name), value in params.items():
            module = getattr(self, module_name)
            module.set_parameter(param_name, value.to(self.device))
            # Freeze this parameter at this value now if freeze is True
            if freeze:
                module.get_parameter(param_name).frozen = True

    def set_frozen_parameters(self, params: Dict[Tuple, float]):
        """
        Sets specific parameters within this Synth. All params within the batch
        will be set to the same value and frozen to prevent further updates.
        """
        params = {
            key: tensor([value] * self.batch_size, device=self.device)
            for key, value in params.items()
        }
        self.set_parameters(params, freeze=True)

    def freeze_parameters(self, params: List[Tuple]):
        """
        Freeze a set of parameters by passing in a tuple of the module and param name
        """
        for module_name, param_name in params:
            module = getattr(self, module_name)
            module.get_parameter(param_name).frozen = True

    def unfreeze_all_parameters(self):
        """
        Unfreeze all parameters in this synth
        """
        for param in self.parameters():
            if isinstance(param, ModuleParameter):
                param.frozen = False

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

        Args:
            batch_idx (Optional[int])   - If provided, we set the parameters of this
                                    synth for reproducibility, in a deterministic
                                    random way. If None (default), we just use
                                    the current module parameter settings.
        """
        if self.synthconfig.reproducible and batch_idx is None:
            raise ValueError(
                "Reproducible mode is on, you must "
                "pass a batch index when calling this synth"
            )
        if self.synthconfig.no_grad:
            with torch.no_grad():
                if batch_idx is not None:
                    self.randomize(seed=batch_idx)
                return self._forward(*args, **kwargs)
        else:
            if batch_idx is not None:
                self.randomize(seed=batch_idx)
            return self._forward(*args, **kwargs)

    def test_step(self, batch, batch_idx):
        """
        This is boilerplate for lightning -- this is required by lightning Trainer
        when calling test, which we use to forward Synths on multi-gpu platforms
        """
        return 0.0

    @property
    def hyperparameters(self) -> OrderedDictType[Tuple[str, str, str], Any]:
        """
        Returns a dictionary of curve and symmetry hyperparameter values keyed
        on a tuple of the module name, parameter name, and hyperparameter name
        """
        hparams = []
        for (module_name, parameter_name), parameter in self.get_parameters().items():
            hparams.append(
                (
                    (module_name, parameter_name, "curve"),
                    parameter.parameter_range.curve,
                )
            )
            hparams.append(
                (
                    (module_name, parameter_name, "symmetric"),
                    parameter.parameter_range.symmetric,
                )
            )

        return OrderedDict(hparams)

    def set_hyperparameter(self, hyperparameter: Tuple[str, str, str], value: Any):
        """
        Set a hyperparameter. Pass in the module name, parameter name, and
        hyperparameter to set, and the value to set it to.
        """
        module = getattr(self, hyperparameter[0])
        parameter = module.get_parameter(hyperparameter[1])
        assert not ModuleParameter.is_parameter_frozen(parameter)
        setattr(parameter.parameter_range, hyperparameter[2], value)

    def randomize(self, seed: Optional[int] = None):
        """
        Randomize all parameters
        """
        parameters = [param for _, param in sorted(self.named_parameters())]

        # https://github.com/torchsynth/torchsynth/issues/253
        if (
            self.batch_size != BATCH_SIZE_FOR_REPRODUCIBILITY
            and self.synthconfig.reproducible
        ):
            raise ValueError(
                "Reproducibility currently only supported "
                f"with batch_size = {BATCH_SIZE_FOR_REPRODUCIBILITY}. "
                "If you want a different batch_size, "
                "initialize SynthConfig with reproducible=False"
            )

        if seed is not None:
            # Generate batch_size x parameter number of random values
            # Reseed the random number generator for every item in the batch
            cpu_rng = torch.Generator(device="cpu")
            new_values = []
            for i in range(self.batch_size):
                cpu_rng.manual_seed(seed * self.batch_size.numpy().item() + i)
                new_values.append(
                    torch.rand((len(parameters),), device="cpu", generator=cpu_rng)
                )

            # Move to device if necessary
            new_values = torch.stack(new_values, dim=1)
            if self.device.type != "cpu":
                new_values = new_values.pin_memory().to(self.device, non_blocking=True)

            # Set parameter data
            for i, parameter in enumerate(parameters):
                if not ModuleParameter.is_parameter_frozen(parameter):
                    parameter.data = new_values[i]
        else:
            assert not self.synthconfig.reproducible
            for parameter in parameters:
                if not ModuleParameter.is_parameter_frozen(parameter):
                    parameter.data.uniform_(0, 1)

        # Add seed to all modules
        for module in self._modules:
            self._modules[module].seed = seed

    def on_post_move_to_device(self) -> None:
        """
        LightningModule trigger after this Synth has been moved to a different device.
        Use this to update children SynthModules device settings
        """
        self.synthconfig.to(self.device)
        for module in self.modules():
            if isinstance(module, SynthModule):
                # TODO look into performance of calling to instead
                module.update_device(self.device)


class Voice(AbstractSynth):
    """
    In a synthesizer, one combination of VCO, VCA, VCF's is typically called a voice.
    """

    def __init__(self, synthconfig: Optional[SynthConfig] = None, *args, **kwargs):
        AbstractSynth.__init__(self, synthconfig=synthconfig, *args, **kwargs)

        # Register all modules as children
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr_1", ADSR),
                ("adsr_2", ADSR),
                ("lfo_1", LFO),
                ("lfo_2", LFO),
                ("lfo_1_amp_adsr", ADSR),
                ("lfo_2_amp_adsr", ADSR),
                ("lfo_1_rate_adsr", ADSR),
                ("lfo_2_rate_adsr", ADSR),
                ("control_vca", ControlRateVCA),
                ("control_upsample", ControlRateUpsample),
                ("mod_matrix", ModulationMixer, {"n_input": 4, "n_output": 5}),
                ("vco_1", SineVCO),
                ("vco_2", SquareSawVCO),
                ("noise", Noise, {"seed": 13}),
                ("vca", VCA),
                ("mixer", AudioMixer, {"n_input": 3, "curves": [1.0, 1.0, 0.1]}),
            ]
        )

    def _forward(self) -> T:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        midi_f0, note_on_duration = self.keyboard()

        # ADSRs for modulating LFOs
        lfo_1_rate = self.lfo_1_rate_adsr(note_on_duration)
        lfo_2_rate = self.lfo_2_rate_adsr(note_on_duration)
        lfo_1_amp = self.lfo_1_amp_adsr(note_on_duration)
        lfo_2_amp = self.lfo_2_amp_adsr(note_on_duration)

        # Compute LFOs with envelopes
        lfo_1 = self.control_vca(self.lfo_1(lfo_1_rate), lfo_1_amp)
        lfo_2 = self.control_vca(self.lfo_2(lfo_2_rate), lfo_2_amp)

        # ADSRs for Oscillators and noise
        adsr_1 = self.adsr_1(note_on_duration)
        adsr_2 = self.adsr_2(note_on_duration)

        # Mix all modulation signals
        (vco_1_pitch, vco_1_amp, vco_2_pitch, vco_2_amp, noise_amp) = self.mod_matrix(
            adsr_1, adsr_2, lfo_1, lfo_2
        )

        # Create signal and with modulations and mix together
        vco_1_out = self.vca(
            self.vco_1(midi_f0, self.control_upsample(vco_1_pitch)),
            self.control_upsample(vco_1_amp),
        )
        vco_2_out = self.vca(
            self.vco_2(midi_f0, self.control_upsample(vco_2_pitch)),
            self.control_upsample(vco_2_amp),
        )
        noise_out = self.vca(self.noise(), self.control_upsample(noise_amp))

        return self.mixer(vco_1_out, vco_2_out, noise_out)
