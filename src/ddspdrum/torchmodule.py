"""
Synth modules in Torch.
"""

from typing import Any, List

import torch.nn as nn
import torch.tensor as T

from ddspdrum.defaults import SAMPLE_RATE
from ddspdrum.module import SynthModule, VCO
from ddspdrum.modparameter import ModParameter


class TorchSynthModule(nn.Module, SynthModule):
    """
    Base class for synthesis modules, in torch.

    WARNING: For now, TorchSynthModules should be atomic and not contain other SynthModules.
    TODO: Later, we should deprecate SynthModule and fold everything into here.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """
        NOTE:
        __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.
        """
        nn.Module.__init__(self)
        SynthModule.__init__(self, sample_rate=sample_rate)
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()

    def add_modparameters(self, modparameters: List[ModParameter]):
        """
        Add parameters to this SynthModule's parameters dictionary.
        (Since there is inheritance, this might happen several times.)
        """
        print(modparameters)
        SynthModule.add_modparameters(self, modparameters)
        for modparameter in modparameters:
            assert modparameter.name not in self.torchparameters
            # TODO: I'm not 100% sure it's kosher to add nn.Parameters
            # outside of __init__, but here we go.
            # TODO: Internally we want to store all torch modparameter
            # values using their 0/1 range, not their human-interpretable
            # range.
            # We might also rethink the syntactic sugar, e.g. sometimes
            # we want to expose the raw 0/1 and sometimes we want to expose the
            # clipped one.
            print(modparameter.value)
            self.torchparameters[modparameter.name] = nn.Parameter(
                T(modparameter.value)
            )

    def forward(self, *input: Any) -> T:
        return self.npyforward(*input)


#    def get_modparameter(self, modparameter_id: str) -> Parameter:
#        """
#        Get a single modparameter for this module
#
#        Parameters
#        ----------
#        modparameter_id (str)  :   Id of the modparameter to return
#        """
#        return self.modparameters[modparameter_id]
#
#    def get_modparameter_0to1(self, modparameter_id: str) -> float:
#        """
#        Get the value of a single modparameter in the range of [0,1]
#
#        Parameters
#        ----------
#        modparameter_id (str)  :   Id of the modparameter to return the value for
#        """
#        return self.modparameters[modparameter_id].get_value_0to1()
#
#    def set_modparameter(self, modparameter_id: str, value: float):
#        """
#        Update a specific modparameter value, ensuring that it is within a specified range
#
#        Parameters
#        ----------
#        modparameter_id (str)  : Id of the modparameter to update
#        value (float)       : Value to update modparameter with
#        """
#        self.modparameters[modparameter_id].set_value(value)
#
#    def set_modparameter_0to1(self, modparameter_id: str, value: float):
#        """
#        Update a specific modparameter with a value in the range [0,1]
#
#        Parameters
#        ----------
#        modparameter_id (str)  : Id of the modparameter to update
#        value (float)       : Value to update modparameter with
#        """
#        self.modparameters[modparameter_id].set_value_0to1(value)
#
#    def p(self, modparameter_id: str):
#        """
#        Convenience method for getting the modparameter value.
#        """
#        return self.modparameters[modparameter_id].value
#


class TorchVCO(VCO, TorchSynthModule):
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
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            midi_f0=midi_f0, mod_depth=mod_depth, phase=phase, sample_rate=sample_rate
        )
        self.add_modparameters(
            [
                ModParameter("pitch", midi_f0, 0, 127),
                ModParameter("mod_depth", mod_depth, 0, 127),
            ]
        )
        # TODO: Make this a modparameter too?
        self.phase = phase
