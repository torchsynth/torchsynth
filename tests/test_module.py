"""
Tests for torch synth modules.
"""

import pytest
import torch.tensor as T

import torchsynth.module as synthmodule
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
