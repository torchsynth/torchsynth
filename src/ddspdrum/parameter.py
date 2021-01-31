"""
Parameters for DDSP Modules
"""

import torch
import torch.nn as nn
import torch.tensor as T


class ParameterRange:
    """
    ParameterRange class is a structure for keeping track of the specific range that a
    parameter might take on. Also handles functionality for converting to and from a
    range between 0 and 1. This class does not store the value of a parameter, just the
    range.

    Parameters
    ----------
    minimum (float) :   minimum value in range
    maximum (float) :   maximum value in range
    curve   (str)   :   relationship between parameter values and the normalized values
                        in the range [0,1]. Must be one of "linear", "log", or "exp".
                        Defaults to "linear"
    """

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 1.0,
        curve: str = "linear",
    ):
        self.minimum = minimum
        self.maximum = maximum

        self.curve_type = curve
        if curve == "linear":
            self.curve = 1
        elif curve == "log":
            self.curve = 0.5
        elif curve == "exp":
            self.curve = 2.0
        else:
            curve_types = ["linear", "log", "exp"]
            raise ValueError("Curve must be one of {}".format(", ".join(curve_types)))

    def __repr__(self):
        return "ParameterRange(min={}, max={}, curve={})".format(
            self.minimum, self.maximum, self.curve_type
        )

    def from_0to1(self, normalized: T) -> T:
        """
        Set value of this parameter using a normalized value in the range [0,1]

        Parameters
        ----------
        normalized (float)   : value within [0,1] range to convert to range defined by
            minimum and maximum
        """
        assert torch.all(0.0 <= normalized)
        assert torch.all(normalized <= 1.0)

        if self.curve != 1:
            normalized = torch.exp2(torch.log2(normalized) / self.curve)

        return self.minimum + (self.maximum - self.minimum) * normalized

    def to_0to1(self, value: T) -> T:
        """
        Convert a ranged parameter to a normalized range from 0 to 1

        Parameters
        ----------
        value (float)   : value within the range defined by minimum and maximum
        """
        assert torch.all(self.minimum <= value)
        assert torch.all(value <= self.maximum)

        normalized = (value - self.minimum) / (self.maximum - self.minimum)
        if self.curve != 1:
            normalized = torch.pow(normalized, self.curve)

        return normalized


class TorchParameter(nn.Parameter):
    """
    Parameter class that inherits from pytorch nn.Parameter so it can be used for
    training. Can use a ParameterRange object to help convert from a 0 to 1 range
    which is expected internally and an external user specified range.

    Parameters
    ----------
    value (float) : initial value of this parameter in the user-specific range.
        Must pass in a ParameterRange object when using this to provide conversion to
        and from 0-to-1 range
    parameter_name (str) : A name for this parameter
    parameter_range (ParameterRange) : A ParameterRange object that supports conversion
        to and from 0-to-1 range and a user-specified range.
    data (Tensor) : directly add data to this parameter without a user-range
    requires_grad (bool) : whether or not a gradient is required for this parameter
    """

    def __new__(
            cls,
            value: float = None,
            parameter_name: str = "",
            parameter_range: ParameterRange = None,
            data: torch.Tensor = None,
            requires_grad: bool = True,
    ):
        if value is not None:
            if parameter_range is not None:
                data = parameter_range.to_0to1(T(value))
            else:
                raise ValueError(
                    "A parameter range must be specified when passing in a value"
                )

        self = super().__new__(cls, data, requires_grad)

        # Additional members -- check to make sure they don't exist first
        # (This is sanity check in case something changes in pytorch in the future)
        assert 'parameter_range' not in self.__dict__
        self.parameter_range = parameter_range

        assert 'parameter_name' not in self.__dict__
        self.parameter_name = parameter_name

        return self

    def __repr__(self):
        return "TorchParameter(name={}, value={})".format(
            self.parameter_name, self.item()
        )

    def from_0to1(self) -> T:
        """
        Get the value of this parameter in the user-specified range. If no user range
        was specified, then the original parameter is returned.
        """
        if self.parameter_range is not None:
            return self.parameter_range.from_0to1(self)

        return self

    def to_0to1(self, new_value: T):
        """
        Set the value of this parameter using an input that is within the user-specified
        range. It will be converted to a 0-to-1 range and stored internally.

        Parameters
        ----------
        new_value (Tensor) : new value to update this parameter with
        """
        if self.parameter_range is not None:
            self.data = self.parameter_range.to_0to1(new_value)
        else:
            raise RuntimeError("A range was never set for this parameter")
