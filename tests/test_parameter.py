"""
Tests for DDSP Parameters
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.tensor as T

from torchsynth.parameter import ModuleParameterRange, ModuleParameter


class TestParameterRange:

    def test_construction(self):
        param_range = ModuleParameterRange(0.0, 10.0)
        assert param_range.minimum == 0.0
        assert param_range.maximum == 10.0
        assert param_range.curve_type == "linear"
        assert param_range.curve == 1.0

        param_range = ModuleParameterRange(0.0, 10.0, curve="log")
        assert param_range.minimum == 0.0
        assert param_range.maximum == 10.0
        assert param_range.curve_type == "log"
        assert param_range.curve == 0.5

        param_range = ModuleParameterRange(0.0, 10.0, curve="exp")
        assert param_range.minimum == 0.0
        assert param_range.maximum == 10.0
        assert param_range.curve_type == "exp"
        assert param_range.curve == 2.0

        with pytest.raises(ValueError):
            ModuleParameterRange(0.0, 1.0, curve="not_a_curve")

    def test_repr(self):
        param_range = ModuleParameterRange(0.0, 1.0)
        assert repr(param_range) == "ModuleParameterRange(min=0.0, max=1.0, curve=linear)"

    def test_to_0to1(self):
        # Test linear scaling
        param_range = ModuleParameterRange(0.0, 10.0)
        assert param_range.to_0to1(T(5.0)) == T(0.5)

        # Test with a log scaling
        param_range = ModuleParameterRange(0.0, 10.0, curve="log")
        params = torch.linspace(0.0, 9.0, 10)
        norm_params = param_range.to_0to1(params)
        expected = torch.pow(params / 10.0, 0.5)
        assert torch.all(norm_params.eq(expected))

        # Test with an exponential scaling
        param_range = ModuleParameterRange(0.0, 10.0, curve="exp")
        params = torch.linspace(0.0, 9.0, 10)
        norm_params = param_range.to_0to1(params)
        expected = torch.pow(params / 10.0, 2.0)
        assert torch.all(norm_params.eq(expected))

    def test_from_0to1(self):
        # Test with linear range
        param_range = ModuleParameterRange(0.0, 10.0)
        assert param_range.from_0to1(T(0.5)) == 5.0

        norm_params = torch.linspace(0.0, 1.0, 10)
        params = param_range.from_0to1(norm_params)
        expected = norm_params * 10.0
        assert torch.all(params.eq(expected))

        # Test with log scaling
        param_range = ModuleParameterRange(0.0, 10.0, curve="log")
        norm_params = torch.linspace(0.0, 1.0, 10)
        params = param_range.from_0to1(norm_params)
        expected = torch.exp2(torch.log2(norm_params) / 0.5) * 10.0
        assert torch.all(params.eq(expected))

        # Test with exponential scaling
        param_range = ModuleParameterRange(0.0, 10.0, curve="exp")
        norm_params = torch.linspace(0.0, 1.0, 10)
        params = param_range.from_0to1(norm_params)
        expected = torch.exp2(torch.log2(norm_params) / 2.0) * 10.0
        assert torch.all(params.eq(expected))


class TestModuleParameter:

    def test_empty_construction(self):
        param = ModuleParameter()
        assert issubclass(ModuleParameter, nn.Parameter)
        assert isinstance(param, nn.Parameter)
        assert isinstance(param, ModuleParameter)

    def test_construction(self):
        # Test passing in the data directly
        data = torch.rand(10)
        param = ModuleParameter(data=data)
        assert torch.all(param.eq(data))

        # Test construction by passing in a value in a specific range
        data = torch.linspace(0.0, 9.0, 10)
        param = ModuleParameter(value=data, parameter_range=ModuleParameterRange(0.0, 10.))
        assert torch.all(param.eq(data / 10.))
        assert torch.all(param.from_0to1().eq(data))

        # Test naming a parameter
        param = ModuleParameter(parameter_name="param_1")
        assert param.parameter_name == "param_1"

        # Test to make sure we can require grad
        param = ModuleParameter(requires_grad=False)
        assert not param.requires_grad

        param = ModuleParameter()
        assert param.requires_grad

        # Test error thrown if a value is passed in with an incorrect range
        with pytest.raises(ValueError):
            ModuleParameter(value=T([0.0, 2555.0]))

    def test_repr(self):
        param_range = ModuleParameterRange(0.0, 10.0)
        param = ModuleParameter(
            value=T([5.0, 1.0]),
            parameter_range=param_range,
            parameter_name="param_1"
        )
        assert repr(param) == 'ModuleParameter(name=param_1, value=tensor([0.5000, 0.1000]))'

    def test_from_0to1(self):
        # Test with no range first -- should get original value back
        data = torch.linspace(0.0, 0.99, 100)
        param = ModuleParameter(data=data)
        assert torch.all(param.from_0to1().eq(data))

        # Make sure the requires_grad attribute is sustained
        assert param.requires_grad

        # Test with range now
        param_range = ModuleParameterRange(0.0, 100.0)
        expected = torch.linspace(0.0, 99.0, 100)
        param = ModuleParameter(data=data, parameter_range=param_range)
        assert torch.all(torch.isclose(param.from_0to1(), expected))
        assert param.requires_grad

    def test_to_0to1(self):
        param_range = ModuleParameterRange(0.0, 10.0)
        data = torch.linspace(0.0, 9.0, 10)
        param = ModuleParameter(value=T([0.0, 1.0]), parameter_range=param_range)
        param.to_0to1(data)
        assert torch.all(torch.isclose(data / 10., param))

        param.to_0to1(T(5.0))
        assert param == 0.5

        # Now test setting a ModuleParameter without a range set
        param = ModuleParameter(data=T(0.0))
        with pytest.raises(RuntimeError):
            param.to_0to1(T(0.2))
