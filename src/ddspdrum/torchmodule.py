"""
Synth modules in Torch.
"""

from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as T

from ddspdrum.defaults import SAMPLE_RATE
from ddspdrum.parameter import ParameterRange, TorchParameter
from ddspdrum.torchutil import linspace, midi_to_hz, reverse_signal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class TorchSynthModule(nn.Module):
    """
    Base class for synthesis modules, in torch.

    WARNING: For now, TorchSynthModules should be atomic and not contain other
    SynthModules.
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
        self.sample_rate = T(sample_rate)
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()

    def seconds_to_samples(self, seconds: T) -> T:
        return torch.round(seconds * self.sample_rate).int()

    def forward(self, *inputs: Any) -> T:  # pragma: no cover
        """
        Each TorchSynthModule should override this.
        """
        pass

    def npyforward(self, *inputs: Any) -> np.ndarray:  # pragma: no cover
        """
        This is the numpy version of the torch.nn.Module.forward command.
        All torch.tensor inputs and outputs are cast to ndarrays.
        """
        npyinput = []
        for i in inputs:
            if isinstance(i, T):
                npyinput.append(i.numpy())
            else:
                npyinput.append(i)
        return self.forward(*npyinput).numpy()

    def add_parameters(self, parameters: List[TorchParameter]):
        """
        Add parameters to this SynthModule's torch parameter dictionary.
        """
        for parameter in parameters:
            assert parameter.parameter_name not in self.torchparameters
            self.torchparameters[parameter.parameter_name] = parameter

    def get_parameter(self, parameter_id: str) -> TorchParameter:
        """
        Get a single TorchParameter for this module

        Parameters
        ----------
        parameter_id (str)  :   Id of the parameter to return
        """
        return self.torchparameters[parameter_id]

    def get_parameter_0to1(self, parameter_id: str) -> float:
        """
        Get the value of a single parameter in the range of [0,1]

        Parameters
        ----------
        modparameter_id (str)  :   Id of the modparameter to return the value for
        """
        return float(self.torchparameters[parameter_id].item())

    def set_parameter(self, parameter_id: str, value: float):
        """
        Update a specific parameter value, ensuring that it is within a specified
        range

        Parameters
        ----------
        parameter_id (str)  : Id of the parameter to update
        value (float)       : Value to update parameter with
        """
        self.torchparameters[parameter_id].set_with_range(value)

    def set_parameter_0to1(self, parameter_id: str, value: float):
        """
        Update a specific parameter with a value in the range [0,1]

        Parameters
        ----------
        parameter_id (str)  : Id of the parameter to update
        value (float)       : Value to update parameter with
        """
        assert 0 <= value <= 1
        self.torchparameters[parameter_id].data = T(value)

    def p(self, parameter_id: str) -> T:
        """
        Convenience method for getting the parameter value.
        """
        return self.torchparameters[parameter_id].get_in_range()


class TorchADSR(TorchSynthModule):
    """
    Envelope class for building a control rate ADSR signal
    """

    def __init__(
        self,
        a: float = 0.25,
        d: float = 0.25,
        s: float = 0.5,
        r: float = 0.5,
        alpha: float = 3.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        """
        Parameters
        ----------
        a                   :   attack time (sec), >= 0
        d                   :   decay time (sec), >= 0
        s                   :   sustain amplitude between 0-1. The only part of
                                ADSR that (confusingly, by convention) is not
                                a time value.
        r                   :   release time (sec), >= 0
        alpha               :   envelope curve, >= 0. 1 is linear, >1 is
                                exponential.
        """
        super().__init__(sample_rate=sample_rate)
        self.add_parameters(
            [
                TorchParameter(
                    initial_value=a,
                    parameter_name="attack",
                    parameter_range=ParameterRange(0.0, 2.0, curve="log")
                ),
                TorchParameter(
                    initial_value=d,
                    parameter_name="decay",
                    parameter_range=ParameterRange(0.0, 2.0, curve="log")
                ),
                TorchParameter(
                    initial_value=s,
                    parameter_name="sustain",
                    parameter_range=ParameterRange(0.0, 1.0)
                ),
                TorchParameter(
                    initial_value=r,
                    parameter_name="release",
                    parameter_range=ParameterRange(0.0, 5.0, curve="log")
                ),
                TorchParameter(
                    initial_value=alpha,
                    parameter_name="alpha",
                    parameter_range=ParameterRange(0.1, 6.0)
                )
            ]
        )

    def forward(self, note_on_duration: T = T(0)) -> np.ndarray:
        """Generate an ADSR envelope.

        By default, this envelope reacts as if it was triggered with midi, for
        example playing a keyboard. Each midi event has a beginning and end:
        note-on, when you press the key down; and note-off, when you release the
        key. `note_on_duration` is the amount of time that the key is depressed.

        During the note-on, the envelope moves through the attack and decay
        sections of the envelope. This leads to musically-intuitive, but
        programatically-counterintuitive behaviour:

        E.g., assume attack is .5 seconds, and decay is .5 seconds. If a note is
        held for .75 seconds, the envelope won't pass through the entire
        attack-and-decay (specifically, it will execute the entire attack, and
        only .25 seconds of the decay).

        Alternately, you can specify a `note_on_duration` of "0" which will
        switch the envelope to one-shot mode. In this case, the envelope moves
        through the entire attack, decay, and release, with no held "sustain"
        value.

        If this is confusing, don't worry about it. ADSR's do a lot of work
        behind the scenes to make the playing experience feel natural.

        """

        assert note_on_duration >= 0

        # If sustain is "0" go to one-shot mode (moves through ADR sections).
        if note_on_duration == T(0):
            note_on_duration = self.p("attack") + self.p("decay")

        num_samples = self.seconds_to_samples(note_on_duration)

        # Release decays from the last value of the attack-and-decay sections.
        ADS = self.note_on(num_samples)
        R = self.note_off(ADS[-1])

        return torch.cat((ADS, R))

    def _ramp(self, duration: T):
        """Makes a ramp of a given duration in seconds.

        This function is used for the piece-wise construction of the envelope
        signal. Its output monotonically increases from 0 to 1. As a result,
        each component of the envelope is a scaled and possibly reversed
        version of this ramp:

        attack      -->     returns an `a`-length ramp, as is.
        decay       -->     `d`-length reverse ramp, descends from 1 to `s`.
        release     -->     `r`-length reverse ramp, descends to 0.

        Its curve is determined by alpha:

        alpha = 1 --> linear,
        alpha > 1 --> exponential,
        alpha < 1 --> logarithmic.

        """

        assert duration.ndim == 0
        t = linspace(0, duration.item(), self.seconds_to_samples(duration))
        return (t / duration) ** self.p("alpha")

    @property
    def attack(self):
        return self._ramp(self.p("attack"))

    @property
    def decay(self):
        # `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        decay = self.p("decay")
        sustain = self.p("sustain")
        return reverse_signal(self._ramp(decay)) * (1 - sustain) + sustain

    @property
    def release(self):
        # `r`-length reverse ramp, reversed to descend to 0.
        release = self.p("release")
        return reverse_signal(self._ramp(release))

    def note_on(self, num_samples):
        assert self.attack.ndim == 1
        assert self.decay.ndim == 1
        out_ = torch.cat((self.attack, self.decay), 0)

        # Truncate or extend based on sustain duration.
        if num_samples < len(out_):
            out_ = out_[:num_samples]
        elif num_samples > len(out_):
            hold_samples = num_samples - len(out_)
            assert hold_samples.ndim == 0
            out_ = torch.nn.functional.pad(
                out_, [0, hold_samples], value=out_[-1].item()
            )
        return out_

    def note_off(self, last_val):
        return self.release * last_val

    def __str__(self):
        return (
            f"""TorchADSR(a={self.torchparameters['attack']}, """
            f"""d={self.torchparameters['decay']}, """
            f"""s={self.torchparameters['sustain']}, """
            f"""r={self.torchparameters['release']}, """
            f"""alpha={self.torchparameters['alpha']}"""
        )


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
    >>> two_8ve_chirp = vco(linspace(0, 1, 1000, endpoint=False))
    """

    def __init__(
        self,
        midi_f0: float = 10,
        mod_depth: float = 50,
        phase: float = 0,
        sample_rate: int = SAMPLE_RATE,
    ):
        TorchSynthModule.__init__(self, sample_rate=sample_rate)
        self.add_parameters(
            [
                TorchParameter(
                    T(midi_f0),
                    parameter_name="pitch",
                    parameter_range=ParameterRange(0.0, 127.0)
                ),
                TorchParameter(
                    T(mod_depth),
                    parameter_name="mod_depth",
                    parameter_range=ParameterRange(0.0, 127.0)
                )
            ]
        )
        # TODO: Make this a parameter too?
        self.phase = T(phase)

    def forward(self, mod_signal: T, phase: T = T(0.0)) -> T:
        """
        Generates audio signal from modulation signal.

        There are three representations of the 'pitch' at play here: (1) midi,
        (2) instantaneous frequency, and (3) phase, a.k.a. 'argument'.

        (1) midi    This is an abuse of the standard midi convention, where
                    semitone pitches are mapped from 0 - 127. Here it's a
                    convenient way to represent pitch linearly. An A above
                    middle C is midi 69.

        (2) freq    Pitch scales logarithmically in frequency. A is 440Hz.

        (3) phase   This is the argument of the cosine function that generates
                    sound. Frequency is the first derivative of phase; phase is
                    integrated frequency (~ish).

        First we generate the 'pitch contour' of the signal in midi values (mod
        contour + base pitch). Then we convert to a phase argument (via
        frequency), then output sound.

        """

        assert (mod_signal >= 0).all() and (mod_signal <= 1).all()

        modulation = self.p("mod_depth") * mod_signal
        control_as_midi = self.p("pitch") + modulation
        control_as_frequency = midi_to_hz(control_as_midi)
        cosine_argument = self.make_argument(control_as_frequency) + phase

        self.phase = cosine_argument[-1]
        return self.oscillator(cosine_argument)

    def make_argument(self, control_as_frequency: T) -> T:
        """
        Generates the phase argument to feed a cosine function to make audio.
        """
        assert control_as_frequency.ndim == 1
        return torch.cumsum(2 * torch.pi * control_as_frequency / SAMPLE_RATE, dim=0)

    @abstractmethod
    # TODO: Type me!
    def oscillator(self, argument: T) -> T:
        """
        Dummy method. Overridden by child class VCO's.
        """
        pass


class TorchSineVCO(TorchVCO):
    """
    Simple VCO that generates a pitched sinudoid.

    Built off the VCO base class, it simply implements a cosine function as oscillator.
    """

    def __init__(
        self, midi_f0: float = 10.0, mod_depth: float = 50.0, phase: float = 0.0
    ):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)

    def oscillator(self, argument):
        return torch.cos(argument)
