"""
Synth modules in Torch.
"""

from typing import List

import torch.nn as nn
import torch.tensor as T

from ddspdrum.defaults import SAMPLE_RATE
from ddspdrum.module import SynthModule
from ddspdrum.parameter import Parameter

class TorchSynthModule(nn.Module, SynthModule):
    """
    Base class for synthesis modules, in torch.

    WARNING: For now, TorchSynthModules should be atomic and not contain other SynthModules.
    TODO: Later, we should deprecate SynthModule and fold everything into here.
    """

    def __init__(
        self, sample_rate: int = SAMPLE_RATE
    ):
        """
        NOTE:
        __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.
        """
        nn.Module.__init__(self)
        SynthModule.__init__(self, sample_rate = sample_rate)
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()

    def add_parameters(self, parameters: List[Parameter]):
        """
        Add parameters to this SynthModule's parameters dictionary.
        (Since there is inheritance, this might happen several times.)
        """
        SynthModule.add_parameters(self, parameters)
        for parameter in parameters:
            assert parameter.name not in self.torchparameters
            # TODO: I'm not 100% sure it's kosher to add nn.Parameters
            # outside of __init__, but here we go.
            # TODO: Internally we want to store all torch parameter
            # values using their 0/1 range, not their human-interpretable
            # range.
            # We might also rethink the syntactic sugar, e.g. sometimes
            # we want to expose the raw 0/1 and sometimes we want to expose the
            # clipped one.
            print(parameter.value)
            self.torchparameters[parameter.name] = nn.Parameter(T(parameter.value))
 

#    def get_parameter(self, parameter_id: str) -> Parameter:
#        """
#        Get a single parameter for this module
#
#        Parameters
#        ----------
#        parameter_id (str)  :   Id of the parameter to return
#        """
#        return self.parameters[parameter_id]
#
#    def get_parameter_0to1(self, parameter_id: str) -> float:
#        """
#        Get the value of a single parameter in the range of [0,1]
#
#        Parameters
#        ----------
#        parameter_id (str)  :   Id of the parameter to return the value for
#        """
#        return self.parameters[parameter_id].get_value_0to1()
#
#    def set_parameter(self, parameter_id: str, value: float):
#        """
#        Update a specific parameter value, ensuring that it is within a specified range
#
#        Parameters
#        ----------
#        parameter_id (str)  : Id of the parameter to update
#        value (float)       : Value to update parameter with
#        """
#        self.parameters[parameter_id].set_value(value)
#
#    def set_parameter_0to1(self, parameter_id: str, value: float):
#        """
#        Update a specific parameter with a value in the range [0,1]
#
#        Parameters
#        ----------
#        parameter_id (str)  : Id of the parameter to update
#        value (float)       : Value to update parameter with
#        """
#        self.parameters[parameter_id].set_value_0to1(value)
#
#    def p(self, parameter_id: str):
#        """
#        Convenience method for getting the parameter value.
#        """
#        return self.parameters[parameter_id].value
#

class TorchVCO(TorchSynthModule):
    """
    Voltage controlled oscillator.

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a modulation signal between 0 - 1. An array of 0's returns a
    stationary audio signal at its base pitch.


    Parameters
    ----------

    midi_f0 (flt)       :       pitch value in 'midi' (69 = 440Hz).
    mod_depth (flt)     :       depth of the pitch modulation in semitones.

    Examples
    --------

    >>> vco = VCO(midi_f0=69.0, mod_depth=24.0)
    >>> two_8ve_chirp = vco(np.linspace(0, 1, 1000, endpoint=False))
    """

    def __init__(
        self,
        midi_f0: float = 10,
        mod_depth: float = 50,
        phase: float = 0,
        sample_rate: int = SAMPLE_RATE
    ):
        super().__init__(sample_rate=sample_rate)
        self.add_parameters(
            [
                Parameter("pitch", midi_f0, 0, 127),
                Parameter("mod_depth", mod_depth, 0, 127),
            ]
        )
        # TODO: Make this a parameter too?
        self.phase = phase
