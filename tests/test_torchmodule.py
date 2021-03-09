"""
Tests for torch synth modules.
"""

import pytest
import torch.nn
import torch.tensor as T

import torchsynth.defaults as defaults
import torchsynth.globals
import torchsynth.module as synthmodule
import torchsynth.parameter
from torchsynth.parameter import ModuleParameter, ModuleParameterRange


class TestTorchSynthModule:
    """
    Tests for TorchSynthModules
    """

    def test_get_parameter(self):
        module = synthmodule.TorchSynthModule0Ddeprecated()
        param_1 = ModuleParameter(data=T(1.0), parameter_name="param_1")
        module.add_parameters([param_1])
        assert module.get_parameter("param_1") == param_1

    def test_set_parameter(self):
        module = synthmodule.TorchSynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25

        module.set_parameter("param_1", 10000.0)
        assert module.torchparameters["param_1"] == 0.5
        assert module.torchparameters["param_1"].from_0to1() == 10000.0

        with pytest.raises(AssertionError):
            module.set_parameter_0to1("param_1", -100.0)

    def test_set_parameter_0to1(self):
        module = synthmodule.TorchSynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25

        module.set_parameter_0to1("param_1", 0.5)
        assert module.torchparameters["param_1"] == 0.5
        assert module.torchparameters["param_1"].from_0to1() == 10000.0

        # Passing a value outside of range should fail
        with pytest.raises(AssertionError):
            module.set_parameter_0to1("param_1", 5.0)

    def test_p(self):
        module = synthmodule.TorchSynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25
        assert module.p("param_1") == 5000.0


class TestTorchSynth:
    """
    Tests for TorchSynth
    """

    def test_construction(self):
        # Test empty construction
        synthglobals = torchsynth.globals.TorchSynthGlobals(
            sample_rate=T(16000), buffer_size=T(512), batch_size=T(2)
        )
        # Test construction with args
        synth = synthmodule.TorchSynth(synthglobals)
        assert synth.sample_rate == T(16000)
        assert synth.buffer_size == T(512)

    def test_add_synth_module(self):
        synthglobals = torchsynth.globals.TorchSynthGlobals(batch_size=T(2))
        synth = synthmodule.TorchSynth(synthglobals)
        vco = synthmodule.TorchSineVCO(
            midi_f0=T([12.0, 30.0]),
            mod_depth=T([50.0, 50.0]),
            synthglobals=synthglobals,
        )
        noise = synthmodule.Noise(ratio=T([0.25, 0.75]), synthglobals=synthglobals)

        synth.add_synth_modules({"vco": vco, "noise": noise})
        assert hasattr(synth, "vco")
        assert hasattr(synth, "noise")

        # Make sure all the ModuleParameters were registered correctly
        synth_params = [p for p in synth.parameters() if isinstance(p, ModuleParameter)]
        module_params = [p for p in vco.parameters() if isinstance(p, ModuleParameter)]
        module_params.extend(
            [p for p in noise.parameters() if isinstance(p, ModuleParameter)]
        )
        for p in module_params:
            fails = True
            for p2 in synth_params:
                if p.parameter_name == p2.parameter_name and torch.all(
                    p.data == p2.data
                ):
                    fails = False
            assert not fails

        # Expect a TypeError if a non TorchSynthModule0Ddeprecated is passed in
        with pytest.raises(TypeError):
            synth.add_synth_modules({"module": torch.nn.Module()})

        # Expect a ValueError if the incorrect sample rate or buffer size is passed in
        with pytest.raises(ValueError):
            synthglobals_weird_sr = torchsynth.globals.TorchSynthGlobals(
                batch_size=T(2), sample_rate=T(16000)
            )
            vco_2 = synthmodule.TorchSineVCO(
                midi_f0=T([12.0, 30.0]),
                mod_depth=T([50.0, 50.0]),
                synthglobals=synthglobals_weird_sr,
            )
            synth.add_synth_modules({"vco_2": vco_2})

        # This should raise an assertion because it has a different batch size than
        # the other modules
        with pytest.raises(ValueError):
            synthglobals_new_batchsize = torchsynth.globals.TorchSynthGlobals(
                batch_size=T(1)
            )
            adsr = synthmodule.ADSR(
                attack=T([0.5]),
                decay=T([0.25]),
                sustain=T([0.5]),
                release=T([1.0]),
                alpha=T([1.0]),
                synthglobals=synthglobals_new_batchsize,
            )
            synth.add_synth_modules({"adsr": adsr})

        # Same here
        with pytest.raises(ValueError):
            synthglobals_new_batchsize = torchsynth.globals.TorchSynthGlobals(
                batch_size=T(1)
            )
            adsr = synthmodule.ADSR(
                synthglobals=synthglobals_new_batchsize,
            )
            synth.add_synth_modules({"adsr": adsr})
