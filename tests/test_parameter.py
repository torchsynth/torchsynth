"""
Tests for DDSP Parameters
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.tensor as T

from ddspdrum.parameter import ParameterRange, TorchParameter


class TestParameterRange:

    def test_empty_construction(self):
        param_range = ParameterRange()
        assert param_range.minimum == 0.0
        assert param_range.maximum == 1.0
        assert param_range.curve_type == "linear"

    def test_construction(self):
        param_range = ParameterRange(0.0, 10.0)
        assert param_range.minimum == 0.0
        assert param_range.maximum == 10.0
        assert param_range.curve_type == "linear"
        assert param_range.curve == 1.0

        param_range = ParameterRange(0.0, 10.0, curve="log")
        assert param_range.minimum == 0.0
        assert param_range.maximum == 10.0
        assert param_range.curve_type == "log"
        assert param_range.curve == 0.5

        param_range = ParameterRange(0.0, 10.0, curve="exp")
        assert param_range.minimum == 0.0
        assert param_range.maximum == 10.0
        assert param_range.curve_type == "exp"
        assert param_range.curve == 2.0

        with pytest.raises(ValueError):
            ParameterRange(curve="not_a_curve")

    def test_repr(self):
        param_range = ParameterRange(0.0, 1.0)
        assert repr(param_range) == "ParameterRange(min=0.0, max=1.0, curve=linear)"

    def test_to_0to1(self):
        # Test linear scaling
        param_range = ParameterRange(0.0, 10.0)
        assert param_range.to_0to1(T(5.0)) == T(0.5)

        # Test with a log scaling
        param_range = ParameterRange(0.0, 10.0, curve="log")
        params = torch.linspace(0.0, 9.0, 10)
        norm_params = param_range.to_0to1(params)
        expected = torch.pow(params / 10.0, 0.5)
        assert torch.all(norm_params.eq(expected))

        # Test with an exponential scaling
        param_range = ParameterRange(0.0, 10.0, curve="exp")
        params = torch.linspace(0.0, 9.0, 10)
        norm_params = param_range.to_0to1(params)
        expected = torch.pow(params / 10.0, 2.0)
        assert torch.all(norm_params.eq(expected))

    def test_from_0to1(self):
        # Test with linear range
        param_range = ParameterRange(0.0, 10.0)
        assert param_range.from_0to1(T(0.5)) == 5.0

        norm_params = torch.linspace(0.0, 1.0, 10)
        params = param_range.from_0to1(norm_params)
        expected = norm_params * 10.0
        assert torch.all(params.eq(expected))

        # Test with log scaling
        param_range = ParameterRange(0.0, 10.0, curve="log")
        norm_params = torch.linspace(0.0, 1.0, 10)
        params = param_range.from_0to1(norm_params)
        expected = torch.exp2(torch.log2(norm_params) / 0.5) * 10.0
        assert torch.all(params.eq(expected))

        # Test with exponential scaling
        param_range = ParameterRange(0.0, 10.0, curve="exp")
        norm_params = torch.linspace(0.0, 1.0, 10)
        params = param_range.from_0to1(norm_params)
        expected = torch.exp2(torch.log2(norm_params) / 2.0) * 10.0
        assert torch.all(params.eq(expected))


class TestTorchParameter:

    def test_empty_construction(self):
        param = TorchParameter()
        assert issubclass(TorchParameter, nn.Parameter)
        assert isinstance(param, nn.Parameter)
        assert isinstance(param, TorchParameter)

    def test_construction(self):
        # Test passing in the data directly
        param = TorchParameter(data=T(1.0))
        assert param == 1.0

        # Test construction by passing in a value in a specific range
        param = TorchParameter(value=5.0, parameter_range=ParameterRange(0.0, 10.))
        assert param == 0.5
        assert param.from_0to1() == 5.0

        # Test naming a parameter
        param = TorchParameter(parameter_name="param_1")
        assert param.parameter_name == "param_1"

        # Test to make sure we can require grad
        param = TorchParameter(requires_grad=False)
        assert not param.requires_grad

        param = TorchParameter()
        assert param.requires_grad

        # Test error thrown if a value is passed in with a range
        with pytest.raises(ValueError):
            TorchParameter(value=2555.0)

    def test_repr(self):
        param_range = ParameterRange(0.0, 10.0)
        param = TorchParameter(
            value=5.0,
            parameter_range=param_range,
            parameter_name="param_1"
        )
        assert repr(param) == 'TorchParameter(name=param_1, value=0.5)'

    def test_from_0to1(self):
        # Test with no range first -- should get original value back
        param = TorchParameter(data=T(0.45))
        np.testing.assert_almost_equal(param.from_0to1().item(), 0.45)

        # Make sure the requires_grad attribute is sustained
        assert param.requires_grad

        # Test with range now
        param_range = ParameterRange(0.0, 10.0)
        param = TorchParameter(data=T(0.5), parameter_range=param_range)
        np.testing.assert_almost_equal(param.from_0to1().item(), 5.0)
        assert param.requires_grad

    def test_to_0to1(self):
        param_range = ParameterRange(0.0, 10.0)
        param = TorchParameter(value=7.5, parameter_range=param_range)
        assert param == 0.75

        param.to_0to1(T(5.0))
        assert param == 0.5

        # Now test setting a TorchParameter without a range set
        param = TorchParameter(data=T(0.0))
        with pytest.raises(RuntimeError):
            param.to_0to1(T(0.2))
