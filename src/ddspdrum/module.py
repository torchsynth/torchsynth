"""
Synth modules.

    TODO    :   - Convert operations to tensors, obvs.

"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from ddspdrum.defaults import BUFFER_SIZE, SAMPLE_RATE
from ddspdrum.modparameter import ModParameter
from ddspdrum.numpyutil import crossfade, fix_length, midi_to_hz, normalize


class SynthModule:
    """
    Base class for synthesis modules.

    WARNING: For now, SynthModules should be atomic and not contain other SynthModules.
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
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.modparameters: Dict[ModParameter] = {}

    def _npyforward(
            self,
            *args: Any,
            **kwargs: Any
    ) -> np.ndarray:  # pragma: no cover
        """
        Each SynthModule should override this.
        This is the numpy version of the torch.nn.Module.forward command.
        """
        pass

    def npyforward(
            self,
            *args: Any,
            **kwargs: Any
    ) -> np.ndarray:  # pragma: no cover
        """
        Wrapper for _npyforward that ensures a buffer_size length output.
        """
        return self.to_buffer_size(self._npyforward(*args, **kwargs))

    def __repr__(self):
        """
        Return a string representation of this synth module and all its parameters
        """
        return "{}(sample_rate={}, parameters={})".format(
            self.__class__, repr(self.sample_rate), repr(self.modparameters)
        )

    def to_buffer_size(self, signal: np.ndarray) -> np.ndarray:
        return fix_length(signal, self.buffer_size)

    def seconds_to_samples(self, seconds: float) -> int:
        return int(round(seconds * self.sample_rate))

    def add_modparameters(self, modparameters: List[ModParameter]):
        """
        Add parameters to this SynthModule's parameters dictionary.
        (Since there is inheritance, this might happen several times.)
        """
        for modparameter in modparameters:
            assert modparameter.name not in self.modparameters
            self.modparameters[modparameter.name] = modparameter

    def connect_modparameter(
        self, modparameter_id: str, module: SynthModule, module_parameter_id: str
    ):
        """
        Create a named parameter for this synthesizer that is connected to a parameter
        in an underlying synth module

        Parameters
        ----------
        modparameter_id (str)       : name of the new parameter
        module (SynthModule)        : the SynthModule to connect to this parameter
        module_parameter_id (str)   : parameter_id in SynthModule to target
        """
        if modparameter_id in self.modparameters:
            raise ValueError("parameter_id: {} already used".format(modparameter_id))

        if module_parameter_id not in module.modparameters:
            raise KeyError(
                "parameter_id: {} not a parameter in {}".format(
                    module_parameter_id, module
                )
            )

        self.modparameters[modparameter_id] = module.get_parameter(module_parameter_id)

    def get_parameter(self, modparameter_id: str) -> ModParameter:
        """
        Get a single parameter for this module

        Parameters
        ----------
        modparameter_id (str)  :   Id of the parameter to return
        """
        return self.modparameters[modparameter_id]

    def get_modparameter_0to1(self, modparameter_id: str) -> float:
        """
        Get the value of a single parameter in the range of [0,1]

        Parameters
        ----------
        modparameter_id (str)  :   Id of the parameter to return the value for
        """
        return self.modparameters[modparameter_id].get_value_0to1()

    def set_modparameter(self, modparameter_id: str, value: float):
        """
        Update a specific parameter value, ensuring that it is within a specified range

        Parameters
        ----------
        parameter_id (str)  : Id of the parameter to update
        value (float)       : Value to update parameter with
        """
        self.modparameters[modparameter_id].set_value(value)

    def set_modparameter_0to1(self, modparameter_id: str, value: float):
        """
        Update a specific parameter with a value in the range [0,1]

        Parameters
        ----------
        modparameter_id (str)  : Id of the parameter to update
        value (float)       : Value to update parameter with
        """
        self.modparameters[modparameter_id].set_value_0to1(value)

    def p(self, modparameter_id: str):
        """
        Convenience method for getting the parameter value.
        """
        return self.modparameters[modparameter_id].value

    def randomize(self) -> None:
        """
        Randomize all modparameters.
        """
        for modparameter_id in self.modparameters:
            self.modparameters[modparameter_id].randomize()


class ADSR(SynthModule):
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
                ModParameter("attack", a, 0, 2, curve="log"),
                ModParameter("decay", d, 0, 2, curve="log"),
                ModParameter("sustain", s, 0, 1),
                ModParameter("release", r, 0, 5, curve="log"),
                ModParameter("alpha", alpha, 0.1, 6),
            ]
        )

    def _npyforward(self, note_on_duration: float = 0) -> np.ndarray:
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
        if note_on_duration == 0:
            note_on_duration = self.p("attack") + self.p("decay")

        num_samples = self.seconds_to_samples(note_on_duration)

        # Release decays from the last value of the attack-and-decay sections.
        ADS = self.note_on(num_samples)
        R = self.note_off(ADS[-1])

        return np.concatenate((ADS, R))

    def _ramp(self, duration: float):
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

        t = np.linspace(0, duration, self.seconds_to_samples(duration), endpoint=False)
        return (t / duration) ** self.p("alpha")

    @property
    def attack(self):
        return self._ramp(self.p("attack"))

    @property
    def decay(self):
        # `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        decay = self.p("decay")
        sustain = self.p("sustain")
        return self._ramp(decay)[::-1] * (1 - sustain) + sustain

    @property
    def release(self):
        # `r`-length reverse ramp, reversed to descend to 0.
        release = self.p("release")
        return self._ramp(release)[::-1]

    def note_on(self, num_samples):
        out_ = np.append(self.attack, self.decay)

        # Truncate or extend based on sustain duration.
        if num_samples < len(out_):
            out_ = out_[:num_samples]
        elif num_samples > len(out_):
            hold_samples = num_samples - len(out_)
            out_ = np.pad(out_, [0, hold_samples], mode="edge")
        return out_

    def note_off(self, last_val):
        return self.release * last_val

    def __str__(self):
        return f"""ADSR(a={self.modparameters['attack']}, d={self.modparameters['decay']},
                s={self.modparameters['sustain']}, r={self.modparameters['release']},
                alpha={self.get_parameter('alpha')})"""


class VCO(SynthModule):
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

    >>> myVCO = VCO(midi_f0=69, mod_depth=24)
    >>> two_8ve_chirp = myVCO(np.linspace(0, 1, 1000, endpoint=False))
    """

    def __init__(
        self,
        midi_f0: float = 10.0,
        mod_depth: float = 50.0,
        phase: float = 0.0,
        sample_rate: int = SAMPLE_RATE,
        buffer_size: int = BUFFER_SIZE
    ):
        super().__init__(sample_rate=sample_rate, buffer_size=buffer_size)
        self.add_modparameters(
            [
                ModParameter("pitch", midi_f0, 0.0, 127.0),
                ModParameter("mod_depth", mod_depth, 0.0, 127.0),
            ]
        )
        # TODO: Make this a parameter too?
        self.phase = phase

    def _npyforward(self, mod_signal: np.array, phase: float = 0.0) -> np.ndarray:
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

        assert (mod_signal >= -1).all() and (mod_signal <= 1).all()

        control_as_frequency = self.make_control_as_frequency(mod_signal)

        cosine_argument = self.make_argument(control_as_frequency) + phase

        # Store final phase.
        self.phase = cosine_argument[-1]
        return self.oscillator(cosine_argument)

    def make_control_as_frequency(self, mod_signal: np.ndarray):
        modulation = self.p("mod_depth") * mod_signal
        control_as_midi = self.p("pitch") + modulation
        return midi_to_hz(control_as_midi)

    def make_argument(self, control_as_frequency: np.array):
        """
        Generates the phase argument to feed a cosine function to make audio.
        """

        return np.cumsum(2 * np.pi * control_as_frequency / SAMPLE_RATE)

    @abstractmethod
    def oscillator(self, argument):
        """
        Dummy method. Overridden by child class VCO's.
        """
        pass


class SineVCO(VCO):
    """
    Simple VCO that generates a pitched sinusoid.

    Built off the VCO base class, it simply implements a cosine function as oscillator.
    """

    def __init__(
        self,
        midi_f0: float = 10.0,
        mod_depth: float = 50.0,
        phase: float = 0.0,
    ):

        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)

    def oscillator(self, argument):
        return np.cos(argument)


class FmVCO(VCO):
    """
    Frequency modulation VCO. Takes `mod_signal` as instantaneous frequency.

    Typical modulation is calculated in pitch-space (midi). For FM to work,
    we have to change the order of calculations. Here `mod_depth` is interpreted
    as the "modulation index" which is tied to the fundamental of the oscillator
    being modulated:

        modulation_index = frequency_deviation / modulation_frequency

    """

    def __init__(
        self,
        midi_f0: float = 10.0,
        mod_depth: float = 50.0,
        phase: float = 0.0
    ):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)

    def make_control_as_frequency(self, mod_signal: np.array):
        # Compute modulation in Hz space (rather than midi-space).
        f0_hz = midi_to_hz(self.p("pitch"))
        fm_depth = self.p("mod_depth") * f0_hz
        modulation_hz = fm_depth * mod_signal
        return f0_hz + modulation_hz

    def oscillator(self, argument):
        # Classically, FM operators are sine waves.
        return np.cos(argument)


class SquareSawVCO(VCO):
    """
    VCO that can be either a square or a sawtooth waveshape.
    Tweak with the shape parameter. (0 is square.)

    With apologies to:

    Lazzarini, Victor, and Joseph Timoney. "New perspectives on distortion synthesis for
        virtual analog oscillators." Computer Music Journal 34, no. 1 (2010): 28-40.
    """

    def __init__(
        self,
        shape: float = 0.0,
        midi_f0: float = 10.0,
        mod_depth: float = 50.0,
        phase: float = 0.0,
    ):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)
        self.add_modparameters(
            [
                ModParameter("shape", shape, 0.0, 1.0),
            ]
        )

    def oscillator(self, argument):
        square = np.tanh(np.pi * self.partials_constant * np.sin(argument) / 2)
        shape = self.p("shape")
        return (1 - shape / 2) * square * (1 + shape * np.cos(argument))

    @property
    def partials_constant(self):
        """
        Constant value that determines the number of partials in the resulting
        square / saw wave in order to keep aliasing at an acceptable level.
        Higher frequencies require fewer partials whereas lower frequency sounds
        can safely have more partials without causing audible aliasing.
        """
        max_pitch = self.p("pitch") + self.p("mod_depth")
        max_f0 = midi_to_hz(max_pitch)
        return 12000 / (max_f0 * np.log10(max_f0))


class VCA(SynthModule):
    """
    Voltage controlled amplifier.
    """

    def __init__(
            self,
            sample_rate: int = SAMPLE_RATE,
            buffer_size: int = BUFFER_SIZE
    ):
        super().__init__(sample_rate=sample_rate, buffer_size=buffer_size)

    def _npyforward(self, control_in: np.array, audio_in: np.array) -> np.ndarray:
        assert (control_in >= 0).all() and (control_in <= 1).all()

        if (audio_in <= -1).any() or (audio_in >= 1).any():
            audio_in = normalize(audio_in)

        audio_in = fix_length(audio_in, len(control_in))
        return control_in * audio_in


class NoiseModule(SynthModule):
    """
    Adds noise.
    """

    def __init__(
            self,
            ratio: float = 0.25,
            sample_rate: int = SAMPLE_RATE,
            buffer_size: int = BUFFER_SIZE
    ):
        super().__init__(sample_rate=sample_rate, buffer_size=buffer_size)
        self.add_modparameters(
            [
                ModParameter("ratio", ratio, 0.0, 1.0),
            ]
        )

    def _npyforward(self, audio_in: np.ndarray) -> np.ndarray:
        noise = self.noise_of_length(audio_in)
        return crossfade(audio_in, noise, self.p("ratio"))

    @staticmethod
    def noise_of_length(audio_in: np.ndarray):
        return np.random.rand(len(audio_in))


class DummyModule(SynthModule):
    """
    A dummy module just encapsulates some parameters and doesn't create sound.
    This is a temporary workaround that allows a Synth to contain parameters
    without nesting SynthModules and without forcing Synth to be a SynthModule.
    """

    def __init__(self, parameters: List[ModParameter], sample_rate: int = SAMPLE_RATE):
        """
        Parameters
        ----------
        dict                :   List of parameter dictionaries, for
                                `Parameter`.,
                                TODO: Take this as a list of Parameters instead?
        """
        super().__init__(sample_rate=sample_rate)
        self.add_modparameters(parameters)

    def _npyforward(self) -> np.ndarray:
        assert False


# TODO: Remove SynthModule
class Synth(SynthModule):
    """
    A base class for a modular synth, ensuring that all modules
    have the same sample rate.
    """

    def __init__(self, modules: Tuple[SynthModule]):
        """
        NOTE: __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.
        Instead, consider @property getters that use the instance's parameters.
        """
        sample_rate = modules[0].sample_rate
        super().__init__(sample_rate=sample_rate)

        # Parameter list
        # TODO: We can remove this later
        self.parameters = {}

        # Check that we are not mixing different sample rates.
        for m in modules[:1]:
            assert m.sample_rate == sample_rate


class Drum(Synth):
    """
    A package of modules that makes one drum hit.
    """

    def __init__(
        self,
        note_on_duration: float,
        drum_params: DummyModule = DummyModule(
            parameters=[
                ModParameter(
                    name="vco_ratio",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                )
            ]
        ),
        pitch_adsr: ADSR = ADSR(),
        amp_adsr: ADSR = ADSR(),
        vco_1: VCO = SineVCO(),
        vco_2: VCO = SquareSawVCO(),
        noise_module: NoiseModule = NoiseModule(),
        vca: VCA = VCA(),
    ):
        super().__init__(
            modules=[pitch_adsr, amp_adsr, vco_1, vco_2, noise_module, vca]
        )
        assert note_on_duration >= 0

        # We assume that sustain duration is a hyper-parameter,
        # with the mindset that if you are trying to learn to
        # synthesize a sound, you won't be adjusting the note_on_duration.
        # However, this is easily changed if desired.
        self.note_on_duration = note_on_duration

        self.drum_params = drum_params
        self.pitch_adsr = pitch_adsr
        self.amp_adsr = amp_adsr
        self.vco_1 = vco_1
        self.vco_2 = vco_2
        self.noise_module = noise_module
        self.vca = vca

        # Pitch Envelope
        self.connect_modparameter("pitch_attack", self.pitch_adsr, "attack")
        self.connect_modparameter("pitch_decay", self.pitch_adsr, "decay")
        self.connect_modparameter("pitch_sustain", self.pitch_adsr, "sustain")
        self.connect_modparameter("pitch_release", self.pitch_adsr, "release")
        self.connect_modparameter("pitch_alpha", self.pitch_adsr, "alpha")

        # Amplitude Envelope
        self.connect_modparameter("amp_attack", self.amp_adsr, "attack")
        self.connect_modparameter("amp_decay", self.amp_adsr, "decay")
        self.connect_modparameter("amp_sustain", self.amp_adsr, "sustain")
        self.connect_modparameter("amp_release", self.amp_adsr, "release")
        self.connect_modparameter("amp_alpha", self.amp_adsr, "alpha")

        # VCO 1
        self.connect_modparameter("vco_1_pitch", self.vco_1, "pitch")
        self.connect_modparameter("vco_1_mod_depth", self.vco_1, "mod_depth")

        # VCO 2
        self.connect_modparameter("vco_2_pitch", self.vco_2, "pitch")
        self.connect_modparameter("vco_2_mod_depth", self.vco_2, "mod_depth")
        self.connect_modparameter("vco_2_shape", self.vco_2, "shape")

        # Mix between the two VCOs
        self.connect_modparameter("vco_ratio", self.drum_params, "vco_ratio")

        # Noise
        self.connect_modparameter("noise_ratio", self.noise_module, "ratio")

    def _npyforward(self) -> np.ndarray:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        note_on_duration = self.note_on_duration
        pitch_envelope = self.pitch_adsr.npyforward(note_on_duration)
        amp_envelope = self.amp_adsr.npyforward(note_on_duration)
        pitch_envelope = fix_length(pitch_envelope, len(amp_envelope))

        vco_1_out = self.vco_1.npyforward(pitch_envelope)
        vco_2_out = self.vco_2.npyforward(pitch_envelope)

        audio_out = crossfade(vco_1_out, vco_2_out, self.p("vco_ratio"))

        audio_out = self.noise_module.npyforward(audio_out)

        return self.vca.npyforward(amp_envelope, audio_out)


class SVF(SynthModule):
    """
    A State Variable Filter that can do low-pass, high-pass, band-pass, and
    band-reject filtering. Allows modulation of the cutoff frequency and an
    adjustable resonance parameter. Can self-oscillate to make a sinusoid
    oscillator.

    Parameters
    ----------

    mode (str)              :   filter type, one of LPF, HPF, BPF, or BSF
    cutoff (float)          :   cutoff frequency in Hz must be between 5 and
                                half the sample rate. Defaults to 1000Hz.
    resonance (float)       :   filter resonance, or "Quality Factor". Higher
                                values cause the filter to resonate more. Must
                                be greater than 0.5. Defaults to 0.707.
    self_oscillate (bool)   :   Set the filter into self-oscillation mode, which
                                turns this into a sine wave oscillator with the
                                filter cutoff as frequency. Defaults to False.
    sample_rate (float)     :   Processing sample rate.
    """

    def __init__(
        self,
        mode: str,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(sample_rate=sample_rate)
        self.mode = mode
        self.self_oscillate = self_oscillate
        self.add_modparameters(
            [
                ModParameter(
                    "cutoff", cutoff, 5.0, self.sample_rate / 2.0, curve="log"
                ),
                ModParameter("resonance", resonance, 0.5, 1000.0, curve="log"),
            ]
        )

    def npyforward(
        self,
        audio: np.ndarray,
        cutoff_mod: np.ndarray = None,
        cutoff_mod_amount: float = 0.0,
    ) -> np.ndarray:
        """
        Process audio samples and return filtered results.

        Parameters
        ----------

        audio (np.ndarray)          :   Audio samples to filter
        cutoff_mod (np.ndarray)     :   Control signal used to modulate the filter
                                        cutoff. Values must be in range [0,1]
        cutoff_mod_amount (float)   :   How much to apply the control signal to the
                                        filter cutoff in Hz. Can be positive or
                                        negative. Defaults to 0.
        """

        h = np.zeros(2)
        y = np.zeros_like(audio)

        # Calculate initial coefficients
        cutoff = self.p("cutoff")
        coeff0, coeff1, rho = self.calculate_coefficients(cutoff)

        # Check if there is a filter cutoff envelope to apply
        if cutoff_mod_amount != 0.0:
            # Cutoff modulation must be same length as audio input
            assert len(cutoff_mod) == len(audio)

        # Processing loop
        for i in range(len(audio)):

            # If there is a cutoff modulation envelope, update coefficients
            if cutoff_mod_amount != 0.0:
                cutoff_val = cutoff + cutoff_mod[i] * cutoff_mod_amount
                coeff0, coeff1, rho = self.calculate_coefficients(cutoff_val)

            # Calculate each of the filter components
            hpf = coeff0 * (audio[i] - rho * h[0] - h[1])
            bpf = coeff1 * hpf + h[0]
            lpf = coeff1 * bpf + h[1]

            # Feedback samples
            h[0] = coeff1 * hpf + bpf
            h[1] = coeff1 * bpf + lpf

            if self.mode == "LPF":
                y[i] = lpf
            elif self.mode == "BPF":
                y[i] = bpf
            elif self.mode == "BSF":
                y[i] = hpf + lpf
            else:
                y[i] = hpf

        return y

    def calculate_coefficients(self, cutoff: float) -> (float, float, float):
        """
        Calculates the filter coefficients for SVF.

        Parameters
        ----------
        cutoff (float)  :   Filter cutoff frequency in Hz.
        """

        g = np.tan(np.pi * cutoff / self.sample_rate)
        resonance = self.p("resonance")
        R = 0.0 if self.self_oscillate else 1.0 / (2.0 * resonance)
        coeff0 = 1.0 / (1.0 + 2.0 * R * g + g * g)
        coeff1 = g
        rho = 2.0 * R + g

        return coeff0, coeff1, rho


class LowPassSVF(SVF):
    """
    Low-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="LPF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class HighPassSVF(SVF):
    """
    High-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="HPF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class BandPassSVF(SVF):
    """
    Band-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="BPF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class BandRejectSVF(SVF):
    """
    Band-reject / band-stop filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            mode="BSF",
            cutoff=cutoff,
            resonance=resonance,
            self_oscillate=self_oscillate,
            sample_rate=sample_rate,
        )


class FIR(SynthModule):
    """
    A finite impulse response low-pass filter. Uses convolution with a symmetric
    windowed sinc function.

    Parameters
    ----------

    cutoff (float)      :   cutoff frequency of low-pass in Hz, must be between 5 and
                            half the sampling rate. Defaults to 1000Hz.
    filter_length (int) :   The length of the filter in samples. A longer filter will
                            result in a steeper filter cutoff. Should be greater than 4.
                            Defaults to 512 samples.
    sample_rate (int)   :   Sampling rate to run processing at.
    """

    def __init__(
        self,
        cutoff: float = 1000.0,
        filter_length: int = 512,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(sample_rate=sample_rate)
        self.add_modparameters(
            [
                ModParameter("cutoff", cutoff, 5.0, sample_rate / 2.0, curve="log"),
                ModParameter("length", filter_length, 4.0, 4096.0),
            ]
        )

    def _npyforward(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples
        TODO: Cutoff frequency modulation, if there is an efficient way to do it

        Parameters
        ----------

        audio (np.ndarray)  :   audio samples to filter
        """

        impulse = self.windowed_sinc(self.p("cutoff"), self.p("length"))
        y = np.convolve(audio, impulse)
        return y

    def windowed_sinc(self, cutoff: float, filter_length: int) -> np.ndarray:
        """
        Calculates the impulse response for FIR low-pass filter using the
        windowed sinc function method

        Parameters
        ----------

        cutoff (float)      :   Low-pass cutoff frequency in Hz. Must be between 0 and
                                half the sampling rate.
        filter_length (int) :   Length of the filter impulse response to create. Creates
                                a symmetric filter so if this is even then the filter
                                returned will have a length of filter_length + 1.
        """

        # Normalized frequency
        omega = 2 * np.pi * cutoff / self.sample_rate

        # Create a symmetric sinc function
        half_length = int(filter_length / 2)
        t = np.arange(0 - half_length, half_length + 1)
        ir = np.sin(t * omega)
        ir[half_length] = omega
        ir = np.divide(ir, t, out=ir, where=t != 0)

        # Window using blackman-harris
        n = np.arange(len(ir))
        cos_a = np.cos(2 * np.pi * n / len(n))
        cos_b = np.cos(4 * np.pi * n / len(n))
        window = 0.42 - 0.5 * cos_a + 0.08 * cos_b
        ir *= window

        return ir


class MovingAverage(SynthModule):
    """
    A finite impulse response moving average filter.

    Parameters
    ----------

    filter_length (int) :   Length of filter and number of samples to take average over.
                            Must be greater than 0. Defaults to 32.
    sample_rate (int)   :   Sampling rate to run processing at.
    """

    def __init__(self, filter_length: int = 32, sample_rate: int = SAMPLE_RATE):
        super().__init__(sample_rate=sample_rate)
        self.add_modparameters(
            [
                ModParameter("length", filter_length, 1.0, 4096.0),
            ]
        )

    def _npyforward(self, audio_in: np.ndarray) -> np.ndarray:
        """
        Filter audio samples

        Parameters
        ----------

        audio (np.ndarray)  :   audio samples to filter
        """
        length = int(self.p("length"))
        impulse = np.ones(length) / length
        y = np.convolve(audio_in, impulse)
        return y
