"""
Parameters for DDSP Modules
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor as T


class ModuleParameterRange:
    """
    `ModuleParameterRange` class is a structure for keeping track of
    the specific range that a parameter might take on. Also handles
    functionality for converting between machine-readable range [0, 1] and a
    human-readable range [minimum, maximum].

    Args:
        minimum:   minimum value in human-readable range
        maximum:   maximum value in human-readable range
        curve:   strictly positive shape of the curve
            values less than 1 place more emphasis on smaller
            values and values greater than 1 place more emphasis on
            larger values. 1 is linear.
        symmetric:  whether or not the parameter range is symmetric,
            allows for curves around a center point. When this is True,
            a curve value of one is linear, greater than one emphasizes
            the minimum and maximum, and less than one emphasizes values
            closer to :math:`(maximum - minimum)/2`.
        name: name of this parameter
        description: optional description of this parameter
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        curve: float = 1,
        symmetric: bool = False,
        # TODO: Make this not optional
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.minimum = minimum
        self.maximum = maximum
        self.curve = curve
        self.symmetric = symmetric

    def __repr__(self):
        return (
            f"ModuleParameterRange(name={self.name}, minimum={self.minimum}, "
            + f"maximum={self.maximum}, curve={self.curve}, "
            + f"symmetric={self.symmetric}, "
            + f"description={self.description})"
        )

    def from_0to1(self, normalized: T) -> T:
        """
        Set value of this parameter using a normalized value in the range [0,1]

        Args:
            normalized: value within machine-readable range [0, 1] to convert to
                human-readable range [minimum, maximum].
        """
        # TODO: These asserts are very slow
        # assert torch.all(0.0 <= normalized)
        # assert torch.all(normalized <= 1.0)

        if not self.symmetric:
            if self.curve != 1.0:
                normalized = torch.exp2(torch.log2(normalized) / self.curve)

            return self.minimum + (self.maximum - self.minimum) * normalized

        # Compute the curve for a symmetric curve
        dist = 2.0 * normalized - 1.0
        if self.curve != 1.0:
            normalized = torch.where(
                dist == 0.0,
                dist,
                torch.exp2(torch.log2(torch.abs(dist)) / self.curve) * torch.sign(dist),
            )

        return self.minimum + (self.maximum - self.minimum) / 2.0 * (normalized + 1.0)

    def to_0to1(self, value: T) -> T:
        """
        Convert from human-readable range [minimum, maximum] to machine-range [0, 1].

        Args:
          value: value within the range defined by minimum and maximum
        """
        assert torch.all(self.minimum <= value)
        assert torch.all(value <= self.maximum)

        normalized = (value - self.minimum) / (self.maximum - self.minimum)

        if not self.symmetric:
            if self.curve != 1:
                normalized = torch.pow(normalized, self.curve)
            return normalized

        dist = 2.0 * normalized - 1.0
        return (1.0 + torch.pow(torch.abs(dist), self.curve) * torch.sign(dist)) / 2.0


class ModuleParameter(nn.Parameter):
    """
    `ModuleParameter` class that inherits from pytorch :class:`~torch.nn.Parameter`

    TODO: Rethink value vs data here
    see https://github.com/torchsynth/torchsynth/issues/101

    TODO: parameter_range shouldn't be optional
    see https://github.com/torchsynth/torchsynth/issues/340

    Args:
        value: initial value of this parameter in the human-readable range.
        parameter_name: A name for this parameter
        parameter_range: A :class:`~torchsynth.parameter.ModuleParameterRange`
            object that supports conversion between human-readable range and
            machine-readable [0,1] range.
        data: directly add data to this parameter in machine-readable range.
        requires_grad: whether or not a gradient is required for this parameter
        frozen: freeze parameter value and prevent updating
    """

    def __new__(
        cls,
        value: Optional[T] = None,
        parameter_name: str = "",
        parameter_range: Optional[ModuleParameterRange] = None,
        data: Optional[T] = None,
        requires_grad: bool = True,
        frozen: Optional[bool] = False,
    ):
        # TODO: Assert value is 1D after we have 1D'ified everything
        if value is not None:
            if parameter_range is not None:
                data = parameter_range.to_0to1(value)
            else:
                raise ValueError(
                    "A parameter range must be specified when passing in a value"
                )

        self = super().__new__(cls, data, requires_grad)

        # Additional members -- check to make sure they don't exist first
        # (This is sanity check in case something changes in pytorch in the future)
        assert "parameter_range" not in self.__dict__
        self.parameter_range = parameter_range

        assert "parameter_name" not in self.__dict__
        self.parameter_name = parameter_name

        self.frozen = frozen
        return self

    def __repr__(self):
        return "ModuleParameter(name={}, value={})".format(
            self.parameter_name, self.data
        )

    def from_0to1(self) -> T:
        """
        Get the value of this parameter in the human-readable range.

        TODO ModuleParameterRange should not be optional
        see https://github.com/torchsynth/torchsynth/issues/340
        If no parameter range was specified, then the original parameter is returned.
        """
        if self.parameter_range is not None:
            return self.parameter_range.from_0to1(self)

        return self

    def to_0to1(self, new_value: T):
        """
        Set the value of this parameter using an input that is
        in the human-readable range. Raises a runtime error if
        this parameter has been frozen.

        Args:
            new_value: new value to update this parameter with
        """
        if self.frozen:
            raise RuntimeError("Parameter is frozen")

        if self.parameter_range is not None:
            self.data = self.parameter_range.to_0to1(new_value)
        else:
            raise RuntimeError("A range was never set for this parameter")

    @staticmethod
    def is_parameter_frozen(parameter: "ModuleParameter"):
        """
        Check whether a `ModuleParameter` is frozen. Asserts
        that parameter is an instance of :class:`~torchsynth.parameter.ModuleParameter`,
        and returns a bool indicating whether it is frozen.

        Args:
            parameter: parameter to check
        """
        if isinstance(parameter, ModuleParameter):
            return parameter.frozen
        else:
            raise ValueError(f"Param {parameter} is not a ModuleParameter")
