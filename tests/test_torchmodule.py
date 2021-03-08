"""
Tests for torch synth modules.
"""

import pytest
import torch.nn
import torch.tensor as T

import torchsynth.defaults as defaults
import torchsynth.module as synthmodule
from torchsynth.parameter import ModuleParameter, ModuleParameterRange


class TestTorchSynthModule:
    """
    Tests for TorchSynthModules
    """

    def test_get_parameter(self):
        module = synthmodule.TorchSynthModule()
        param_1 = ModuleParameter(data=T(1.0), parameter_name="param_1")
        module.add_parameters([param_1])
        assert module.get_parameter("param_1") == param_1

    def test_set_parameter(self):
        module = synthmodule.TorchSynthModule()
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
        module = synthmodule.TorchSynthModule()
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
        module = synthmodule.TorchSynthModule()
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
        synth = synthmodule.TorchSynth()
        assert synth.sample_rate == defaults.SAMPLE_RATE
        assert synth.buffer_size == defaults.BUFFER_SIZE

        # Test construction with args
        synth = synthmodule.TorchSynth(sample_rate=16000, buffer_size=512)
        assert synth.sample_rate == 16000
        assert synth.buffer_size == 512

    def test_add_synth_module(self):

        synth = synthmodule.TorchSynth()
        vco = synthmodule.TorchSineVCO(
            midi_f0=T([12.0, 30.0]), mod_depth=T([50.0, 50.0])
        )
        noise = synthmodule.TorchNoise(ratio=T([0.25, 0.75]))

        synth.add_synth_modules({"vco": vco, "noise": noise})
        assert hasattr(synth, "vco")
        assert hasattr(synth, "noise")

        # Make sure all the parameters were registered correctly
        synth_params = [p for p in synth.parameters()]
        module_params = [p for p in vco.parameters()]
        module_params.extend([p for p in noise.parameters()])
        # TODO: Jordi I think you meant to test something else here?
        for p in module_params:
            fails = True
            for p2 in synth_params:
                if p.parameter_name == p2.parameter_name and torch.all(
                    p.data == p2.data
                ):
                    fails = False
            assert not fails

        # Expect a TypeError if a non TorchSynthModule is passed in
        with pytest.raises(TypeError):
            synth.add_synth_modules({"module": torch.nn.Module()})

        # Expect a ValueError if the incorrect sample rate or buffer size is passed in
        with pytest.raises(ValueError):
            vco_2 = synthmodule.TorchSineVCO(
                midi_f0=T([12.0, 30.0]), mod_depth=T([50.0, 50.0]), sample_rate=T(16000)
            )
            synth.add_synth_modules({"vco_2": vco_2})

        with pytest.raises(ValueError):
            adsr = synthmodule.TorchADSR(
                a=T([0.5]),
                d=T([0.25]),
                s=T([0.5]),
                r=T([1.0]),
                alpha=T([1.0]),
                buffer_size=T(512),
            )
            synth.add_synth_modules({"adsr": adsr})
