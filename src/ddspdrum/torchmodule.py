"""
Synth modules in Torch.
"""

from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as T

from ddspdrum.defaults import BUFFER_SIZE, SAMPLE_RATE
from ddspdrum.modparameter import ModParameter
from ddspdrum.torchutil import fix_length, linspace, midi_to_hz, reverse_signal

torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732


class TorchSynthModule(nn.Module):
    """
    Base class for synthesis modules, in torch.

    WARNING: For now, TorchSynthModules should be atomic and not contain other
    SynthModules.
    TODO: Later, we should deprecate SynthModule and fold everything into here.
    """

    def __init__(
            self,
            sample_rate: int = SAMPLE_RATE,
            buffer_size: int = BUFFER_SIZE
    ):
        """
        NOTE:
        __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.
        """
        nn.Module.__init__(self)
        self.sample_rate = T(sample_rate)
        self.buffer_size = T(buffer_size)
        self.modparameters: Dict[ModParameter] = {}
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()

    def to_buffer_size(self, signal: T) -> T:
        return fix_length(signal, self.buffer_size)

    def seconds_to_samples(self, seconds: T) -> T:
        return torch.round(seconds * self.sample_rate).int()

    def add_modparameters(self, modparameters: List[ModParameter]):
        """
        Add parameters to this SynthModule's parameters dictionary.
        (Since there is inheritance, this might happen several times.)
        """
        for modparameter in modparameters:
            assert modparameter.name not in self.modparameters
            self.modparameters[modparameter.name] = modparameter

            assert modparameter.name not in self.torchparameters
            # TODO: I'm not 100% sure it's kosher to add nn.Parameters
            # outside of __init__, but here we go.
            # TODO: Internally we want to store all torch modparameter
            # values using their 0/1 range, not their human-interpretable
            # range.
            # We might also rethink the syntactic sugar, e.g. sometimes
            # we want to expose the raw 0/1 and sometimes we want to expose the
            # clipped one.
            self.torchparameters[modparameter.name] = nn.Parameter(
                T(modparameter.value)
            )

    def _forward(self, *args: Any, **kwargs: Any) -> T:  # pragma: no cover
        """
        Each TorchSynthModule should override this.
        """
        pass

    def forward(self, *args: Any, **kwargs: Any) -> T:  # pragma: no cover
        """
        Wrapper for _forward that ensures a buffer_size length output.
        """
        return self.to_buffer_size(self._forward(*args, **kwargs))

    def npyforward(
            self,
            *args: Any,
            **kwargs: Any
    ) -> np.ndarray:  # pragma: no cover
        """
        This is the numpy version of the torch.nn.Module.forward command.
        All torch.tensor inputs and outputs are cast to ndarrays.
        """
        npyargs = []
        for i in args:
            if isinstance(i, T):
                npyargs.append(i.numpy())
            else:
                npyargs.append(i)

        npykwargs = {}
        for key in kwargs.keys():
            if isinstance(kwargs[key], T):
                npykwargs[key] = kwargs[key].numpy()
            else:
                npykwargs[key] = kwargs[key]

        return self.forward(*npyargs, **npykwargs).numpy()

    # The following is cheesy AF but needed because the
    # torchparameter is the master variable.
    # TODO: Remove .value from modparameter
    def _update_modparameters(self) -> None:
        for modparameter_id in self.modparameters:
            self.modparameters[modparameter_id].set_value(
                float(self.torchparameters[modparameter_id].numpy())
            )

    # In general, we should consider removing all the following
    # since we don't want to cast to int and then back to Tensor
    # Alternately, we should make a SynthParameter which is an nn.Parameter
    # but decorated with ModParameter stuff.

    def get_modparameter(self, modparameter_id: str) -> ModParameter:
        """
        Get a single modparameter for this module

        Parameters
        ----------
        modparameter_id (str)  :   Id of the modparameter to return
        """
        self._update_modparameters()
        return self.modparameters[modparameter_id]

    def get_modparameter_0to1(self, modparameter_id: str) -> float:
        """
        Get the value of a single modparameter in the range of [0,1]

        Parameters
        ----------
        modparameter_id (str)  :   Id of the modparameter to return the value for
        """
        self._update_modparameters()
        return self.modparameters[modparameter_id].get_value_0to1()

    def set_modparameter(self, modparameter_id: str, value: float):
        """
        Update a specific modparameter value, ensuring that it is within a specified
        range

        Parameters
        ----------
        modparameter_id (str)  : Id of the modparameter to update
        value (float)       : Value to update modparameter with
        """
        self.modparameters[modparameter_id].set_value(value)
        self.torchparameters[modparameter_id].data = T(
            self.modparameters[modparameter_id].value
        )

    def set_modparameter_0to1(self, modparameter_id: str, value: float):
        """
        Update a specific modparameter with a value in the range [0,1]

        Parameters
        ----------
        modparameter_id (str)  : Id of the modparameter to update
        value (float)       : Value to update modparameter with
        """
        self.modparameters[modparameter_id].set_value_0to1(value)
        self.torchparameters[modparameter_id].data = T(
            self.modparameters[modparameter_id].value
        )

    def p(self, modparameter_id: str) -> T:
        """
        Convenience method for getting the modparameter value.
        """
        return self.torchparameters[modparameter_id]


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
        buffer_size: int = BUFFER_SIZE
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
        super().__init__(sample_rate=sample_rate, buffer_size=buffer_size)
        self.add_modparameters(
            [
                ModParameter("attack", a, 0.0, 20.0, curve="log"),
                ModParameter("decay", d, 0.0, 20.0, curve="log"),
                ModParameter("sustain", s, 0.0, 1.0),
                ModParameter("release", r, 0.0, 20.0, curve="log"),
                ModParameter("alpha", alpha, 0.0, 10.0),
            ]
        )

    def _forward(self, note_on_duration: T = T(0)) -> np.ndarray:
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
        buffer_size: int = BUFFER_SIZE
    ):
        TorchSynthModule.__init__(
            self,
            sample_rate=sample_rate,
            buffer_size=buffer_size
        )
        self.add_modparameters(
            [
                ModParameter("pitch", midi_f0, 0.0, 127.0),
                ModParameter("mod_depth", mod_depth, 0.0, 127.0),
            ]
        )
        # TODO: Make this a parameter too?
        self.phase = T(phase)

    def _forward(self, mod_signal: T, phase: T = T(0.0)) -> T:
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
