"""
Synth modules in Torch.
"""

from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.tensor as T

import torchsynth.torchutil as util
from torchsynth.defaults import BUFFER_SIZE, SAMPLE_RATE
from torchsynth.parameter import ParameterRange, TorchParameter

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
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()

    def to_buffer_size(self, signal: T) -> T:
        return util.fix_length(signal, self.buffer_size)

    def seconds_to_samples(self, seconds: T) -> T:
        return torch.round(seconds * self.sample_rate).int()

    def _forward(self, *args: Any, **kwargs: Any) -> T:  # pragma: no cover
        """
        Each TorchSynthModule should override this.
        """
        raise NotImplementedError("Derived classes must override this method")

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
        parameter_id (str)  :   Id of the parameter to return the value for
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
        self.torchparameters[parameter_id].to_0to1(T(value))

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
        return self.torchparameters[parameter_id].from_0to1()


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
        self.add_parameters(
            [
                TorchParameter(
                    value=a,
                    parameter_name="attack",
                    parameter_range=ParameterRange(0.0, 2.0, curve="log")
                ),
                TorchParameter(
                    value=d,
                    parameter_name="decay",
                    parameter_range=ParameterRange(0.0, 2.0, curve="log")
                ),
                TorchParameter(
                    value=s,
                    parameter_name="sustain",
                    parameter_range=ParameterRange(0.0, 1.0)
                ),
                TorchParameter(
                    value=r,
                    parameter_name="release",
                    parameter_range=ParameterRange(0.0, 5.0, curve="log")
                ),
                TorchParameter(
                    value=alpha,
                    parameter_name="alpha",
                    parameter_range=ParameterRange(0.1, 6.0)
                )
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

        If this is confusing, don't worry about it. ADSR's do a lot of work
        behind the scenes to make the playing experience feel natural.
        """

        assert note_on_duration > 0
        num_samples = self.seconds_to_samples(note_on_duration)

        # Release decays from the last value of the attack-and-decay sections.
        ADS = self.note_on(num_samples)
        R = self.note_off(ADS[-1])

        return torch.cat((ADS, R))

    def _ramp(self, duration: T, inverse: bool = False):
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
        t = torch.arange(
            self.seconds_to_samples(duration).item(),
            device=duration.device
        )
        ramp = t * (1 / duration) / self.sample_rate

        if inverse:
            ramp = 1.0 - ramp

        return torch.pow(ramp, self.p("alpha"))

    @property
    def attack(self):
        return self._ramp(self.p("attack"))

    @property
    def decay(self):
        # `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        decay = self._ramp(self.p("decay"), inverse=True)
        return decay * (1 - self.p("sustain")) + self.p("sustain")

    @property
    def release(self):
        # `r`-length reverse ramp, reversed to descend to 0.
        return self._ramp(self.p("release"), inverse=True)

    def note_on(self, num_samples):
        assert self.attack.ndim == 1
        assert self.decay.ndim == 1
        out_ = torch.cat((self.attack, self.decay), 0)

        # Truncate or extend based on sustain duration.
        if num_samples <= len(out_):
            out_ = out_[:num_samples]
        else:
            hold_samples = num_samples - len(out_)
            sustain = torch.ones(hold_samples, device=self.p("sustain").device)
            sustain = sustain * self.p("sustain")
            out_ = torch.cat((out_, sustain))

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
        self.add_parameters(
            [
                TorchParameter(
                    value=midi_f0,
                    parameter_name="pitch",
                    parameter_range=ParameterRange(0.0, 127.0)
                ),
                TorchParameter(
                    value=mod_depth,
                    parameter_name="mod_depth",
                    parameter_range=ParameterRange(0.0, 127.0)
                )
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

        assert (mod_signal >= -1).all() and (mod_signal <= 1).all()

        control_as_frequency = self.make_control_as_frequency(mod_signal)

        cosine_argument = self.make_argument(control_as_frequency) + phase

        self.phase = cosine_argument[-1]
        return self.oscillator(cosine_argument)

    def make_control_as_frequency(self, mod_signal: T):
        modulation = self.p("mod_depth") * mod_signal
        control_as_midi = self.p("pitch") + modulation
        return util.midi_to_hz(control_as_midi)

    def make_argument(self, control_as_frequency: T) -> T:
        """
        Generates the phase argument to feed a cosine function to make audio.
        """
        assert control_as_frequency.ndim == 1
        return torch.cumsum(2 * torch.pi * control_as_frequency / SAMPLE_RATE, dim=0)

    @abstractmethod
    def oscillator(self, argument: T) -> T:
        """
        Dummy method. Overridden by child class VCO's.
        """
        pass


class TorchSineVCO(TorchVCO):
    """
    Simple VCO that generates a pitched sinusoid.

    Built off the VCO base class, it simply implements a cosine function as oscillator.
    """

    def __init__(
        self,
        midi_f0: float = 10.0,
        mod_depth: float = 50.0,
        phase: float = 0.0,
        **kwargs
    ):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase, **kwargs)

    def oscillator(self, argument):
        return torch.cos(argument)


class TorchFmVCO(TorchVCO):
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
            phase: float = 0.0):
        super().__init__(midi_f0=midi_f0, mod_depth=mod_depth, phase=phase)

    def make_control_as_frequency(self, mod_signal: T):
        # Compute modulation in Hz space (rather than midi-space).
        f0_hz = util.midi_to_hz(self.p("pitch"))
        fm_depth = self.p("mod_depth") * f0_hz
        modulation_hz = fm_depth * mod_signal
        return f0_hz + modulation_hz

    def oscillator(self, argument):
        # Classically, FM operators are sine waves.
        return torch.cos(argument)


class TorchSquareSawVCO(TorchVCO):
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
        self.add_parameters(
            [
                TorchParameter(
                    value=shape,
                    parameter_name="shape",
                    parameter_range=ParameterRange(0.0, 1.0)
                )
            ]
        )

    def oscillator(self, argument):
        square = torch.tanh(torch.pi * self.partials_constant * torch.sin(argument) / 2)
        shape = self.p("shape")
        return (1 - shape / 2) * square * (1 + shape * torch.cos(argument))

    @property
    def partials_constant(self):
        """
        Constant value that determines the number of partials in the resulting
        square / saw wave in order to keep aliasing at an acceptable level.
        Higher frequencies require fewer partials whereas lower frequency sounds
        can safely have more partials without causing audible aliasing.
        """
        max_pitch = self.p("pitch") + self.p("mod_depth")
        max_f0 = util.midi_to_hz(max_pitch)
        return 12000 / (max_f0 * torch.log10(max_f0))


class TorchVCA(TorchSynthModule):
    """
    Voltage controlled amplifier.
    """

    def __init__(
            self,
            sample_rate: int = SAMPLE_RATE,
            buffer_size: int = BUFFER_SIZE
    ):
        super().__init__(sample_rate=sample_rate, buffer_size=buffer_size)

    def _forward(self, control_in: T, audio_in: T) -> T:
        assert (control_in >= 0).all() and (control_in <= 1).all()

        if (audio_in <= -1).any() or (audio_in >= 1).any():
            util.normalize(audio_in)

        audio_in = util.fix_length(audio_in, len(control_in))
        return control_in * audio_in


class TorchNoise(TorchSynthModule):
    """
    Adds noise to a signal

    Parameters
    ----------
    ratio (float): mix ratio between the incoming signal and the produced noise
    **kwargs: see TorchSynthModule
    """

    def __init__(
            self,
            ratio: float = 0.25,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.add_parameters(
            [
                TorchParameter(
                    value=ratio,
                    parameter_name="ratio",
                    parameter_range=ParameterRange(0.0, 1.0)
                )
            ]
        )

    def _forward(self, audio_in: T) -> T:
        noise = self.noise_of_length(audio_in)
        return util.crossfade(audio_in, noise, self.p("ratio"))

    @staticmethod
    def noise_of_length(audio_in: T) -> T:
        return torch.rand_like(audio_in) * 2 - 1


class TorchSynthParameters(TorchSynthModule):
    """
    A SynthModule that is strictly for managing parameters
    """

    def __init__(
            self,
            sample_rate: int = SAMPLE_RATE,
            buffer_size: int = BUFFER_SIZE
    ):
        super().__init__(sample_rate, buffer_size)

    def _forward(self, *args: Any, **kwargs: Any) -> T:
        raise RuntimeError("TorchSynthParameters cannot be called")


class TorchSynth(nn.Module):
    """
    Base class for synthesizers that combine one or more TorchSynthModules
    to create a full synth architecture.

    Parameters
    ----------
    sample_rate (int): sample rate to run this synth at
    buffer_size (int): number of samples expected at output of child modules
    """

    def __init__(
            self,
            sample_rate: int = SAMPLE_RATE,
            buffer_size: int = BUFFER_SIZE
    ):
        super().__init__()
        self.sample_rate = T(sample_rate)
        self.buffer_size = T(buffer_size)

        # Global Parameter Module
        self.global_params = TorchSynthParameters(sample_rate, buffer_size)

    def add_synth_modules(self, modules: Dict[str, TorchSynthModule]):
        """
        Add a set of named children TorchSynthModules to this synth. Registers them
        with the torch nn.Module so that all parameters are recognized.

        Parameters
        ----------
        modules (Dict): A dictionary of TorchSynthModule
        """

        for name in modules:
            if not isinstance(modules[name], TorchSynthModule):
                raise TypeError(f"{modules[name]} is not a TorchSynthModule")

            if modules[name].sample_rate != self.sample_rate:
                raise ValueError(f"{modules[name]} sample rate does not match")

            if modules[name].buffer_size != self.buffer_size:
                raise ValueError(f"{modules[name]} buffer size does not match")

            self.add_module(name, modules[name])

    def randomize(self):
        """
        Randomize all parameters
        """
        for parameter in self.parameters():
            parameter.data = torch.rand_like(parameter)


class TorchDrum(TorchSynth):
    """
    A package of modules that makes one drum hit.
    """

    def __init__(
        self,
        note_on_duration: float,
        vco_ratio: float = 0.5,
        pitch_adsr: TorchADSR = TorchADSR(),
        amp_adsr: TorchADSR = TorchADSR(),
        vco_1: TorchVCO = TorchSineVCO(),
        vco_2: TorchVCO = TorchSquareSawVCO(),
        noise: TorchNoise = TorchNoise(),
        vca: TorchVCA = TorchVCA(),
        **kwargs
    ):
        super().__init__(**kwargs)
        assert note_on_duration >= 0

        # We assume that sustain duration is a hyper-parameter,
        # with the mindset that if you are trying to learn to
        # synthesize a sound, you won't be adjusting the note_on_duration.
        # However, this is easily changed if desired.
        self.note_on_duration = T(note_on_duration)

        # Add required global parameters
        self.global_params.add_parameters([
            TorchParameter(vco_ratio, "vco_ratio", ParameterRange(0.0, 1.0))
        ])

        # Register all modules as children
        self.add_synth_modules({
            'pitch_adsr': pitch_adsr,
            'amp_adsr': amp_adsr,
            'vco_1': vco_1,
            'vco_2': vco_2,
            'noise': noise,
            'vca': vca
        })

    def forward(self) -> T:
        # The convention for triggering a note event is that it has
        # the same note_on_duration for both ADSRs.
        note_on_duration = self.note_on_duration
        pitch_envelope = self.pitch_adsr(note_on_duration)
        amp_envelope = self.amp_adsr(note_on_duration)

        vco_1_out = self.vco_1(pitch_envelope)
        vco_2_out = self.vco_2(pitch_envelope)

        audio_out = util.crossfade(
            vco_1_out,
            vco_2_out,
            self.global_params.p("vco_ratio")
        )

        audio_out = self.noise(audio_out)

        return self.vca(amp_envelope, audio_out)


class FIRLowPass(TorchSynthModule):
    """
    A finite impulse response low-pass filter. Uses convolution with a windowed
    sinc function.

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
        self.add_parameters(
            [
                TorchParameter(
                    value=cutoff,
                    parameter_name="cutoff",
                    parameter_range=ParameterRange(5.0, sample_rate / 2.0, "log")
                ),
                TorchParameter(
                    value=filter_length,
                    parameter_name="length",
                    parameter_range=ParameterRange(4.0, 4096.)
                )
            ]
        )

    def _forward(self, audio_in: T) -> T:
        """
        Filter audio samples
        TODO: Cutoff frequency modulation, if there is an efficient way to do it

        Parameters
        ----------

        audio (T)  :   audio samples to filter
        """

        impulse = self.windowed_sinc(self.p("cutoff"), self.p("length"))
        impulse = impulse.view(1, 1, impulse.size()[0])
        audio_resized = audio_in.view(1, 1, audio_in.size()[0])
        y = nn.functional.conv1d(
            audio_resized,
            impulse,
            padding=int(self.p("length") / 2)
        )
        return y[0][0]

    def windowed_sinc(self, cutoff: T, length: T) -> T:
        """
        Calculates the impulse response for FIR low-pass filter using the
        windowed sinc function method. Updated to allow for a fractional filter length.

        Parameters
        ----------

        cutoff (T)      :   Low-pass cutoff frequency in Hz. Must be between 0 and
                                half the sampling rate.
        length (T) :   Length of the filter impulse response to create.
        """

        # Normalized frequency
        omega = 2 * torch.pi * cutoff / self.sample_rate

        # Create a sinc function
        num_samples = torch.ceil(length)
        half_length = (length - 1.) / 2.
        t = torch.arange(num_samples.detach(), device=length.device)
        ir = util.sinc((t - half_length) * omega)

        return ir * util.blackman(length)


class TorchMovingAverage(TorchSynthModule):
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
        self.add_parameters(
            [
                TorchParameter(
                    value=filter_length,
                    parameter_name="length",
                    parameter_range=ParameterRange(1.0, 4096.0),
                )
            ]
        )

    def _forward(self, audio_in: T) -> T:
        """
        Filter audio samples

        Parameters
        ----------

        audio (T)  :   audio samples to filter
        """
        length = self.p("length")
        impulse = torch.ones((1, 1, int(length)), device=length.device) / length

        # For non-integer impulse lengths
        if torch.sum(impulse) < 1.0:
            additional = torch.ones(1, 1, 1, device=length.device)
            additional *= (1.0 - torch.sum(impulse))
            impulse = torch.cat((impulse, additional), dim=2)

        audio_resized = audio_in.view(1, 1, audio_in.size()[0])
        y = nn.functional.conv1d(
            audio_resized,
            impulse,
            padding=int(impulse.size()[0] / 2)
        )
        return y[0][0]


class TorchSVF(TorchSynthModule):
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
    mod_depth (float)       :   Amount of modulation to apply to the cutoff from
                                the control input during processing. Can be negative
                                or positive in Hertz. Defaults to zero.
    """

    def __init__(
            self,
            mode: str,
            cutoff: float = 1000.0,
            resonance: float = 0.707,
            mod_depth: float = 0.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        # Set the filter type
        self.mode = mode.lower()
        assert mode in ['lpf', 'hpf', 'bpf', 'bsf']

        nyquist = self.sample_rate / 2.0
        self.add_parameters(
            [
                TorchParameter(
                    value=cutoff,
                    parameter_range=ParameterRange(5.0, nyquist, "log"),
                    parameter_name="cutoff"
                ),
                TorchParameter(
                    value=resonance,
                    parameter_range=ParameterRange(0.01, 1000.0, "log"),
                    parameter_name="resonance"
                ),
                TorchParameter(
                    value=mod_depth,
                    parameter_range=ParameterRange(-nyquist, nyquist, "log"),
                    parameter_name="mod_depth"
                )
            ]
        )

    def _forward(
        self,
        audio_in: T,
        control_in: T = None,
    ) -> T:
        """
        Process audio samples and return filtered results.

        Parameters
        ----------

        audio (torch.tensor)          :   Audio samples to filter
        cutoff_mod (torch.tensor)     :   Control signal used to modulate the filter
                                        cutoff. Values must be in range [0,1]
        """

        h0 = 0.0
        h1 = 0.0
        y = torch.zeros_like(audio_in, device=audio_in.device)

        if control_in is None:
            control_in = torch.zeros_like(audio_in, device=audio_in.device)
        else:
            assert control_in.size() == audio_in.size()

        cutoff = self.p("cutoff")
        mod_depth = self.p("mod_depth")
        res_coefficient = 1.0 / self.p("resonance")

        # Processing loop
        for i in range(len(audio_in)):
            # If there is a cutoff modulation envelope, update coefficients
            cutoff_val = cutoff + control_in[i] * mod_depth
            coeff0, coeff1, rho = TorchSVF.svf_coefficients(
                cutoff_val,
                res_coefficient,
                self.sample_rate
            )

            # Calculate each of the filter components
            hpf = coeff0 * (audio_in[i] - rho * h0 - h1)
            bpf = coeff1 * hpf + h0
            lpf = coeff1 * bpf + h1

            # Feedback samples
            h0 = coeff1 * hpf + bpf
            h1 = coeff1 * bpf + lpf

            if self.mode == "lpf":
                y[i] = lpf
            elif self.mode == "bpf":
                y[i] = bpf
            elif self.mode == "bsf":
                y[i] = hpf + lpf
            else:
                y[i] = hpf

        return y

    @staticmethod
    def svf_coefficients(cutoff, res_coefficient, sample_rate):
        """
        Calculates the filter coefficients for SVF.

        Parameters
        ----------
        cutoff (T)  :   Filter cutoff frequency in Hz.
        resonance (T) : Filter resonance
        sample_rate (T) : Sample rate to process at
        """

        g = torch.tan(torch.pi * cutoff / sample_rate)
        coeff0 = 1.0 / (1.0 + res_coefficient * g + g * g)
        rho = res_coefficient + g

        return coeff0, g, rho


class TorchLowPassSVF(TorchSVF):
    """
    IIR Low-pass using SVF architecture

    Parameters
    ----------
    kwargs: see TorchSVF
    """

    def __init__(self, **kwargs):
        super().__init__('lpf', **kwargs)


class TorchHighPassSVF(TorchSVF):
    """
    IIR High-pass using SVF architecture

    Parameters
    ----------
    kwargs: see TorchSVF
    """

    def __init__(self, **kwargs):
        super().__init__('hpf', **kwargs)


class TorchBandPassSVF(TorchSVF):
    """
    IIR Band-pass using SVF architecture

    Parameters
    ----------
    kwargs: see TorchSVF
    """

    def __init__(self, **kwargs):
        super().__init__('bpf', **kwargs)


class TorchBandStopSVF(TorchSVF):
    """
    IIR Band-stop using SVF architecture

    Parameters
    ----------
    kwargs: see TorchSVF
    """

    def __init__(self, **kwargs):
        super().__init__('bsf', **kwargs)
