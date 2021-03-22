"""
Tests for torch synth modules.
"""

import pytest
import torch
import torch.tensor as T

import torchsynth.deprecated
import torchsynth.module as synthmodule
from torchsynth.parameter import ModuleParameter, ModuleParameterRange


class TestSynthModule:
    """
    Tests for SynthModules
    """

    def test_get_parameter(self):
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
        param_1 = ModuleParameter(data=T(1.0), parameter_name="param_1")
        module.add_parameters([param_1])
        assert module.get_parameter("param_1") == param_1

    def test_set_parameter(self):
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
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
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
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
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25
        assert module.p("param_1") == 5000.0

    def test_modeselector(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        mode_selector = synthmodule.SoftModeSelector(synthglobals, n_modes=3)
        mode_selector.set_parameter("mode0weight", T([0.8, 1.0]))
        mode_selector.set_parameter("mode1weight", T([0.8, 0.0]))
        mode_selector.set_parameter("mode2weight", T([0.8, 0.0]))
        assert (
            torch.mean(
                mode_selector() - T([[1 / 3, 1.0000], [1 / 3, 0.0000], [1 / 3, 0.0000]])
            )
            < 1e-6
        )

    def test_audiomixer(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))

        # Make sure parameters get setup correctly
        mixer = synthmodule.AudioMixer(synthglobals, n_input=3)
        params = [p for p in mixer.parameters()]
        assert len(params) == 3
        for param in params:
            assert param.parameter_range.curve == 1.0

        mixer = synthmodule.AudioMixer(synthglobals, n_input=2, curves=[0.75, 1.5])
        params = [p for p in mixer.parameters()]
        assert len(params) == 2
        assert params[0].parameter_range.curve == 0.75
        assert params[1].parameter_range.curve == 1.5

        # if curves are passed in then the number of curves must equal n_input
        with pytest.raises(AssertionError):
            mixer = synthmodule.AudioMixer(synthglobals, n_input=3, curves=[0.75, 1.5])

    def test_modulationmixer(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))

        mixer = synthmodule.ModulationMixer(synthglobals, n_input=2, n_output=2)
        params = [p for p in mixer.parameters()]
        assert len(params) == 4
        for param in params:
            assert param.parameter_range.curve == 0.5

        mixer = synthmodule.ModulationMixer(
            synthglobals, n_input=1, n_output=2, curves=[1.0]
        )
        params = [p for p in mixer.parameters()]
        assert len(params) == 2
        for param in params:
            assert param.parameter_range.curve == 1.0

        # if curves are passed in then the number of curves must equal n_input
        with pytest.raises(AssertionError):
            mixer = synthmodule.AudioMixer(
                synthglobals, n_input=5, n_output=5, curves=[0.75, 1.5]
            )
