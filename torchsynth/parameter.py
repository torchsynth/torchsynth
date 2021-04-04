"""
Parameters for DDSP Modules
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor as T


class ModuleParameterRange:
    """
    ModuleParameterRange class is a structure for keeping track of
    the specific range that a parameter might take on. Also handles
    functionality for converting to and from a range between 0 and
    1. This class does not store the value of a parameter, just the
    range.

    Args:
        minimum (float) :   minimum value in range
        maximum (float) :   maximum value in range
        device  (torch.device) : Device for storing tensors
        curve   (float) :   shape of the curve, values less than 1
        place more emphasis on smaller values and values greater than 1
        place more emphasis no larger values. Defaults to 1 which is linear.
        symmetric (bool) :  whether or not the parameter range is symmetric,
         allows for curves around a center point. Defaults to False.
        name    (str) : name of this parameter
        description (str) : optional description of this parameter
    """

    def __init__(
        self,
        minimum: float,
        maximum: float,
        device: Optional[torch.device] = None,
        curve: float = 1,
        symmetric: bool = False,
        # TODO: Make this not optional
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.device = None
        self.minimum = minimum
        self.maximum = maximum
        self.curve = curve
        self.symmetric = symmetric

    def to(self, device: torch.device):
        """
        Update the device attribute. Can do more here in the future if need be,
        but from initial profiling it is faster to not have the attributes as tensors
        """
        self.device = device
        return self

    def __repr__(self):
        return (
            f"ModuleParameterRange(name={self.name}, min={self.minimum}, "
            + f"max={self.maximum}, curve={self.curve}, "
            + f"symmetric={self.symmetric}, "
            + f"description={self.description})"
        )

    def from_0to1(self, normalized: T) -> T:
        """
        Set value of this parameter using a normalized value in the range [0,1]

        Args:
          normalized (T): value within [0,1] range to convert to range defined by
          minimum and maximum
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
        Convert a ranged parameter to a normalized range from 0 to 1

        Args:
          value (T): value within the range defined by minimum and maximum
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
    ModuleParameter class that inherits from pytorch nn.Parameter
    so it can be used for training. Can use a ModuleParameterRange
    object to help convert from a 0 to 1 range which is expected
    internally and an external user specified range.

    Args:
        value (T) : initial value of this parameter in the user-specific
        range. Must pass in a ModuleParameterRange object when using
        this to provide conversion to and from 0-to-1 range

        parameter_name (str) : A name for this parameter
        parameter_range (ModuleParameterRange) : A ModuleParameterRange
        object that supports conversion to and from 0-to-1 range
        and a user-specified range.

        data (Tensor) : directly add data to this parameter without a user-range
        requires_grad (bool) : whether or not a gradient is required for this parameter
        frozen (optional bool) : freeze parameter value and prevent updating
    """

    def __new__(
        cls,
        # TODO: REMOVEME
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

    # TODO: Pull this out?
    # Not sure if this works yet
    # def __eq__(self, other):
    #    # Should we be testing other attributes??
    #    return self.parameter_name == other.parameter_name and torch.all(
    #        self.data == other.data
    #    )

    # TODO: Move to ModuleRange
    def from_0to1(self) -> T:
        """
        Get the value of this parameter in the user-specified range. If no user range
        was specified, then the original parameter is returned.
        """
        if self.parameter_range is not None:
            return self.parameter_range.from_0to1(self)

        return self

    # TODO: Move to ModuleRange
    def to_0to1(self, new_value: T):
        """
        Set the value of this parameter using an input that is
        within the user-specified range. It will be converted to a
        0-to-1 range and stored internally.

        Args:
            new_value (Tensor) : new value to update this parameter with
        """
        if self.frozen:
            raise RuntimeError("Parameter is frozen")

        if self.parameter_range is not None:
            self.data = self.parameter_range.to_0to1(new_value)
        else:
            raise RuntimeError("A range was never set for this parameter")

    @staticmethod
    def is_parameter_frozen(parameter):
        if isinstance(parameter, ModuleParameter):
            return parameter.frozen
        else:
            raise ValueError(f"Param {parameter} is not a ModuleParameter")
