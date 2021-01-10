"""
Parameter Class
"""

import numpy as np


class Parameter:
    """
    Parameter class is a structure for keeping track of parameters
    that have a specific range. Also handles functionality for converting
    to and from a range between 0 and 1.

    Parameters
    ----------
    name    (str)   :   Unique name to give to this parameter.
    value   (float) :   initial value of this parameter
    minimum (float) :   minimum value that this parameter can take on
    maximum (float) :   maximum value that this parameter can take on
    curve   (str)   :   relationship between parameter values and the normalized values
                        in the range [0,1]. Must be one of "linear", "log", or "exp".
                        Defaults to "linear"
    """

    def __init__(
        self,
        name: str,
        value: float,
        minimum: float,
        maximum: float,
        curve: str = "linear",
    ):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        assert minimum <= value <= maximum
        self.value = value

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

    def __str__(self):
        name = "{}, ".format(self.name) if self.name else ""
        return "Parameter: {}Value: {}, Min: {}, Max: {}, Curve: {}".format(
            name,
            self.value,
            self.minimum,
            self.maximum,
            self.curve_type
        )

    def set_value(self, new_value):
        """
        Set value of this parameter clipped to the minimum and maximum range

        Parameters
        ---------
        new_value (float)   :   value to update parameter with
        """
        assert self.minimum <= new_value <= self.maximum
        self.value = new_value

    def set_value_0to1(self, new_value):
        """
        Set value of this parameter using a normalized value in the range [0,1]

        Parameters
        ----------
        new_value (float)   :   value to update parameter with, in the range [0,1]
        """
        assert 0 <= new_value <= 1
        if new_value != 0 and self.curve != 1:
            new_value = np.exp2(np.log2(new_value) / self.curve)

        self.value = self.minimum + (self.maximum - self.minimum) * new_value

    def get_value_0to1(self) -> float:
        """
        Get value of this parameter using a normalized value in range [0,1]
        """
        value = (self.value - self.minimum) / (self.maximum - self.minimum)
        if self.curve != 1:
            value = np.power(value, self.curve)

        return value
