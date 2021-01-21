"""
Tests for DDSP Drum Modules
"""

import pytest
from ddspdrum.module import SynthModule
from ddspdrum.parameter import Parameter
from ddspdrum.defaults import SAMPLE_RATE


class TestSynthModule:
    """
    Tests for the SynthModule base class
    """

    def test_default_constructor(self):
        module = SynthModule()
        assert module.sample_rate == SAMPLE_RATE
        assert module.parameters == {}

    def test_constructor(self):
        sr = 16000
        module = SynthModule(sample_rate=sr)
        assert module.sample_rate == sr
        assert module.parameters == {}

    def test_repr(self):
        module = SynthModule()
        param_1 = Parameter("param_1", 5.0, 0.0, 10.0)
        param_2 = Parameter("param_2", 0.5, 0.2, 1.0)
        module.add_parameters([param_1, param_2])

        expected_str = f"{module.__class__}(sample_rate={SAMPLE_RATE}, "\
                       f"parameters={repr(module.parameters)})"

        assert repr(module) == expected_str

    def test_seconds_to_samples(self):
        module = SynthModule()
        sr = module.sample_rate
        seconds = 2.5
        expected_samples = seconds * sr
        assert module.seconds_to_samples(seconds) == expected_samples

        # With rounding
        seconds = 3.3333333
        expected_samples = round(seconds * sr)
        assert module.seconds_to_samples(seconds) == expected_samples

    def test_add_parameter(self):
        module = SynthModule()
        parameter_1 = Parameter("param_1", 0.5, 0.0, 1.0)
        module.add_parameters([parameter_1])
        assert module.parameters["param_1"] == parameter_1

        # Add a couple more parameters
        parameter_2 = Parameter("param_2", 0.75, 0.0, 100.0)
        parameter_3 = Parameter("param_3", 0.11111, 0.1, 0.2)
        module.add_parameters([parameter_2, parameter_3])
        assert module.parameters["param_2"] == parameter_2
        assert module.parameters["param_3"] == parameter_3

        # Try add a repeat parameter -- should raise an assertion error
        with pytest.raises(AssertionError):
            module.add_parameters([parameter_1])

    def test_connect_parameter(self):
        sub_module = SynthModule()
        sub_param_1 = Parameter("param_1", 0.5, 0.0, 1.0, "log")
        sub_param_2 = Parameter("param_2", 0.5, 0.0, 1.0)
        sub_module.add_parameters([sub_param_1, sub_param_2])

        module = SynthModule()
        module.connect_parameter("connected_param_1", sub_module, "param_1")
        assert module.parameters["connected_param_1"] == sub_param_1

        # Try adding parameter with the same name
        with pytest.raises(ValueError):
            module.connect_parameter("connected_param_1", sub_module, "param_1")

        # Try adding a parameter that doesn't exist
        with pytest.raises(KeyError):
            module.connect_parameter("connected_param_2", sub_module, "no_param")

        # Now add the second param
        module.connect_parameter("connected_param_2", sub_module, "param_2")
        assert module.parameters['connected_param_2'] == sub_param_2

    def test_get_parameter(self):
        module = SynthModule()
        param_1 = Parameter("param_1", 0.5, 0.0, 1.0)
        module.add_parameters([param_1])
        assert module.get_parameter("param_1") == param_1

    def test_get_parameter_0to1(self):
        module = SynthModule()
        param_1 = Parameter("param_1", 5.0, 0.0, 10.0)
        module.add_parameters([param_1])
        assert module.get_parameter_0to1("param_1") == 0.5

    def test_set_parameter(self):
        module = SynthModule()
        param_1 = Parameter("param_1", 5.0, 0.0, 10.0)
        module.add_parameters([param_1])
        assert module.parameters["param_1"].value == 5.0

        # Update value
        module.set_parameter("param_1", 7.5)
        assert module.parameters["param_1"].value == 7.5

    def test_set_parameter0to1(self):
        module = SynthModule()
        param_1 = Parameter("param_1", 5.0, 0.0, 10.0)
        module.add_parameters([param_1])
        module.set_parameter_0to1("param_1", 0.25)
        assert module.parameters["param_1"].value == 2.5

    def test_p(self):
        module = SynthModule()
        param_1 = Parameter("param_1", 5.0, 0.0, 10.0)
        param_2 = Parameter("param_2", 0.5, 0.2, 1.0)
        module.add_parameters([param_1, param_2])
        assert module.p("param_1") == 5.0
        assert module.p("param_2") == 0.5
