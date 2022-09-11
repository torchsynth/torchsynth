"""
Synth modules in Torch.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch import tensor

import torchsynth.util as util
from torchsynth.config import BASE_REPRODUCIBLE_BATCH_SIZE, SynthConfig
from torchsynth.parameter import ModuleParameter, ModuleParameterRange
from torchsynth.signal import Signal


class SynthModule(nn.Module):
    """
    A base class for synthesis modules. A :class:`~.SynthModule`
    optionally takes input from other :class:`~.SynthModule` instances.
    The :class:`~.SynthModule` uses its (optional) input and its
    set of :class:`~torchsynth.parameter.ModuleParameter` to generate
    output. All :class:`~torchsynth.parameter.ModuleParameter` of
    the :class:`~.SynthModule` are assumed to be
    :attr:`~torchsynth.config.SynthConfig.batch_size`-length 1-D
    tensors.

    All :class:`~.SynthModule` objects should be atomic, i.e., they
    should not contain other :class:`~.SynthModule` objects. This
    design choice is in the spirit of modular synthesis.

    Args:
        synthconfig: An object containing synthesis settings that are shared
            across all modules, typically specified by
            :class:`~torchsynth.synth.Voice`, or some other, possibly custom
            :class:`~torchsynth.synth.AbstractSynth` subclass.

        device: An object representing the device on which the `torch` tensors
            are to be allocated (as per PyTorch, broadly).
    """

    # This outlines all the parameters available in this module
    # TODO: Make this non-optional
    default_parameter_ranges: Optional[List[ModuleParameterRange]] = None

    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, T],
    ):
        nn.Module.__init__(self)
        self.synthconfig = synthconfig
        self.device = device
        self.synthconfig.to(device)
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()
        self.parameter_ranges = []
        # If this module needs a random seed, here it is
        self.seed: Optional[int] = None

        if self.default_parameter_ranges is not None:
            # We want to create copies of the parameter ranges otherwise each
            # instance of the same module type (ex. ADSR) will reference the
            # same param range.
            assert isinstance(self.default_parameter_ranges, list)
            self.parameter_ranges = copy.deepcopy(self.default_parameter_ranges)
            self._parameter_ranges_dict: Dict[str, ModuleParameterRange] = {
                p.name: p for p in self.parameter_ranges
            }
            assert len(self._parameter_ranges_dict) == len(self.parameter_ranges)
            self.add_parameters(
                [
                    ModuleParameter(
                        value=None,
                        parameter_name=parameter_range.name,
                        data=torch.rand((self.synthconfig.batch_size,), device=device),
                        parameter_range=parameter_range,
                    )
                    for parameter_range in self.parameter_ranges
                ]
            )
            if kwargs:
                # Parameter values can also be passed in as keyword args.
                for name, data in kwargs.items():
                    if data.device != self.device:
                        data = data.to(self.device)
                    self.set_parameter(name, data)

    @property
    def batch_size(self) -> T:
        """Size of the batch to be generated."""
        assert self.synthconfig.batch_size.ndim == 0
        return self.synthconfig.batch_size

    @property
    def sample_rate(self) -> T:
        """Sample rate frequency in Hz."""
        assert self.synthconfig.sample_rate.ndim == 0
        return self.synthconfig.sample_rate

    @property
    def nyquist(self):
        """Convenience property for the highest frequency that can be
        represented at :attr:`~.sample_rate` (as per Shannon-Nyquist)."""
        return self.sample_rate / 2.0

    @property
    def eps(self) -> float:
        """A very small value used to avoid computational errors."""
        return self.synthconfig.eps

    @property
    def buffer_size(self) -> T:
        """Size of the module output in samples."""
        assert self.synthconfig.buffer_size.ndim == 0
        return self.synthconfig.buffer_size

    def to_buffer_size(self, signal: Signal) -> Signal:
        """
        Fixes the length of a signal to the default buffer size of this module,
        as specified by :attr:`~.SynthModule.buffer_size`. Longer signals are
        truncated to length; shorter signals are zero-padded.

        Args:
            signal: A signal to pad or truncate.
        """
        return util.fix_length(signal, self.buffer_size)

    def seconds_to_samples(self, seconds: T) -> T:
        """
        Convenience function to calculate the number of samples corresponding to
        given a time value and :attr:`~.sample_rate`. Returns a possibly
        fractional value.

        Args:
            seconds: Time value in seconds.
        """
        return seconds * self.sample_rate

    def output(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Performs the main action of :class:`~.SynthModule`. Each child class
        should override this method.
        """
        raise NotImplementedError("Derived classes must override this method")

    def forward(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Wrapper for output that ensures a :attr:`~.SynthModule.buffer_size`
        length output.
        """
        signal = self.output(*args, **kwargs)
        buffered = self.to_buffer_size(signal)
        return buffered

    def add_parameters(self, parameters: List[ModuleParameter]):
        """
        Adds parameters to the :class:`~.SynthModule` parameter dictionary. Used
        by the class constructor.

        Args:
            parameters: List of parameters to register with this module.
        """
        for parameter in parameters:
            assert parameter.parameter_name not in self.torchparameters
            assert parameter.shape == (self.batch_size,)
            self.torchparameters[parameter.parameter_name] = parameter

    def get_parameter(self, parameter_id: str) -> ModuleParameter:
        """
        Retrieves a single :class:`~torchsynth.parameter.ModuleParameter`, as
        specified by its parameter Id.

        Args:
            parameter_id: Id of the parameter to retrieve.
        """
        value = self.torchparameters[parameter_id]
        assert value.shape == (self.batch_size,)
        return value

    def get_parameter_0to1(self, parameter_id: str) -> T:
        """
        Retrieves a specified parameter value in the normalized range [0,1].

        Args:
            parameter_id: Id of the parameter to retrieve.
        """
        value = self.torchparameters[parameter_id]
        assert value.shape == (self.batch_size,)
        return value

    def set_parameter(self, parameter_id: str, value: T):
        """
        Updates a parameter value in a parameter-specific non-normalized range.

        Args:
            parameter_id: Id of the parameter to update.
            value:  Value to assign to the parameter.
        """
        value = value.to(self.device)
        self.torchparameters[parameter_id].to_0to1(value)
        value = self.torchparameters[parameter_id].data
        assert torch.all(0.0 <= value) and torch.all(value <= 1.0)
        assert value.shape == (self.batch_size,)

    def set_parameter_0to1(self, parameter_id: str, value: T):
        """
        Update a parameter value in a normalized range [0,1].

        Args:
            parameter_id: Id of the parameter to update.
            value: Value to assign to the parameter.
        """
        value = value.to(self.device)
        assert torch.all(0.0 <= value) and torch.all(value <= 1.0)
        assert value.shape == (self.batch_size,)
        self.torchparameters[parameter_id].data = value

    def p(self, parameter_id: str) -> T:
        """
        Convenience method for retrieving a parameter value. Returns
        the value in parameter-specific, non-normalized range.

        Args:
            parameter_id: Id of the parameter to retrieve.
        """
        value = self.torchparameters[parameter_id].from_0to1()
        assert value.shape == (self.batch_size,)
        return value

    def to(self, device: Optional[torch.device] = None, **kwargs):
        """
        This function overrides the :func:`~torch.nn.Module.to` call in
        :class:`torch.nn.Module`. It ensures that the related values
        :class:`~torchsynth.parameter.ModuleParameterRange` and
        :class:`~torchsynth.parameter.ModuleParameter`, as well as
        :attr:`~.SynthModule.synthconfig` are also transferred to the correct
        device.

        Args:
            device: device to send this module to
        """
        self._update_device(device)
        return super().to(device=device, **kwargs)

    def _update_device(self, device: Optional[torch.device] = None):
        """
        This method handles the device transfer tasks that are not managed by
        PyTorch.

        Args:
            device: Device to assign to this module.
        """
        self.synthconfig.to(device)
        self.device = device


class ControlRateModule(SynthModule):
    """
    An abstract base class for non-audio modules that adapts the functions of
    :class:`.~SynthModule` to run at :attr:`~.ControlRateModule.control_rate`.
    """

    @property
    def sample_rate(self) -> T:
        raise NotImplementedError("This module operates at control rate")

    @property
    def buffer_size(self) -> T:
        raise NotImplementedError("This module uses control buffer size")

    @property
    def control_rate(self) -> T:
        """Control rate frequency in Hz."""
        assert self.synthconfig.control_rate.ndim == 0
        return self.synthconfig.control_rate

    @property
    def control_buffer_size(self) -> T:
        """Size of the module output in samples."""
        assert self.synthconfig.control_buffer_size.ndim == 0
        return self.synthconfig.control_buffer_size

    def to_buffer_size(self, signal: Signal) -> Signal:
        """
        Fixes the length of a signal to the control buffer size of this module,
        as specified by :attr:`~.ControlRateModule.control_buffer_size`. Longer
        signals are truncated to length; shorter signals are zero-padded.

        Args:
            signal: A signal to pad or truncate.
        """
        return util.fix_length(signal, self.control_buffer_size)

    def seconds_to_samples(self, seconds: T) -> T:
        """
        Convenience function to calculate the number of samples corresponding to
        given a time value and :attr:`~.control_rate`. Returns a possibly
        fractional value.

        Args:
            seconds: Time value in seconds.
        """
        return seconds * self.control_rate

    def output(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Performs the main action of :class:`~.ControlRateModule`. Each child
        class should override this method.
        """
        raise NotImplementedError("Derived classes must override this method")


class ADSR(ControlRateModule):
    """
    Envelope class for building a control-rate ADSR signal.

    Args:
        synthconfig: An object containing synthesis settings that are shared
            across all modules, typically specified by
            :class:`~torchsynth.synth.Voice`, or some other, possibly custom
            :class:`~torchsynth.synth.AbstractSynth` subclass.

        device: An object representing the device on which the `torch` tensors
            are allocated (as per PyTorch, broadly).
    """

    #: ADSR Parameters
    default_parameter_ranges: List[ModuleParameterRange] = [
        ModuleParameterRange(
            0.0, 2.0, curve=0.5, name="attack", description="attack time (sec)"
        ),
        ModuleParameterRange(
            0.0, 2.0, curve=0.5, name="decay", description="decay time (sec)"
        ),
        ModuleParameterRange(
            0.0,
            1.0,
            name="sustain",
            description="sustain amplitude 0-1. The only part of ADSR that "
            + "(confusingly, by convention) is not a time value.",
        ),
        ModuleParameterRange(
            0.0, 5.0, curve=0.5, name="release", description="release time (sec)"
        ),
        ModuleParameterRange(
            0.1,
            6.0,
            name="alpha",
            description="envelope curve. 1 is linear, >1 is exponential.",
        ),
    ]

    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, T],
    ):
        super().__init__(synthconfig, device=device, **kwargs)

        # Create some values that will be automatically loaded on device
        self.register_buffer("zero", tensor(0.0, device=self.device))
        self.register_buffer("one", tensor(1.0, device=self.device))
        self.register_buffer(
            "range", torch.arange(self.control_buffer_size, device=self.device)
        )

    def output(self, note_on_duration: T) -> Signal:
        """Generate an ADSR envelope.

        By default, this envelope reacts as if it was triggered with midi, for
        example playing a keyboard. Each midi event has a beginning and end:
        note-on, when you press the key down; and note-off, when you release the
        key. `note_on_duration` is the amount of time that the key is depressed.

        During the note-on, the envelope moves through the attack and decay
        sections of the envelope. This leads to musically-intuitive, but
        programatically-counterintuitive behaviour:

        Example:
            Assume attack is .5 seconds, and decay is .5 seconds. If a note is
            held for .75 seconds, the envelope won't pass through the entire
            attack-and-decay (specifically, it will execute the entire attack,
            and only .25 seconds of the decay).

        If this is confusing, don't worry about it. ADSR's do a lot of work
        behind the scenes to make the playing experience feel natural.

        Args:
            note_on_duration: Duration of note on event in seconds.
        """

        if self.synthconfig.debug:
            assert note_on_duration.ndim == 1
            assert torch.all(note_on_duration > 0.0)

        # Calculations to accommodate attack/decay phase cut by note duration.
        attack = self.p("attack")
        decay = self.p("decay")
        self.alpha = self.p("alpha").unsqueeze(1)

        new_attack = torch.minimum(attack, note_on_duration)
        new_decay = torch.maximum(note_on_duration - attack, self.zero)
        new_decay = torch.minimum(new_decay, decay)

        attack_signal = self.make_attack(new_attack)
        decay_signal = self.make_decay(new_attack, new_decay)
        release_signal = self.make_release(note_on_duration)

        return (attack_signal * decay_signal * release_signal).as_subclass(Signal)

    def ramp(
        self, duration: T, start: Optional[T] = None, inverse: Optional[bool] = False
    ) -> Signal:
        """
        Makes a ramp of a given duration in seconds.

        The construction of this matrix is rather cryptic. Essentially, this
        method works by tilting and clipping ramps between 0 and 1, then
        applying a scaling factor :attr:`~alpha`.

        Args:
            duration: Length of the ramp in seconds.
            start: Initial delay of ramp in seconds.
            inverse: Toggle to flip the ramp from ascending to descending.
        """

        assert duration.ndim == 1
        duration = self.seconds_to_samples(duration).unsqueeze(1)

        # Convert to number of samples.
        if start is not None:
            start = self.seconds_to_samples(start).unsqueeze(1)
        else:
            start = 0.0

        # Build ramps template.
        ramp = self.range.expand((self.batch_size, self.range.shape[0]))

        # Shape ramps.
        ramp = ramp - start
        ramp = torch.maximum(ramp, self.zero)
        ramp = (ramp + self.eps) / duration + self.eps
        ramp = torch.minimum(ramp, self.one)

        # The following is a workaround. In inverse mode, a ramp with 0 duration
        # (that is all 1's) becomes all 0's, which is a problem for the
        # ultimate calculation of the ADSR signal (a * d * r => 0's). So this
        # replaces only rows who sum to 0 (i.e., all components are zero).

        if inverse:
            ramp = torch.where(duration > 0.0, 1.0 - ramp, ramp)

        # Apply scaling factor.
        ramp = torch.pow(ramp, self.alpha)
        return ramp.as_subclass(Signal)

    def make_attack(self, attack_time) -> Signal:
        """
        Builds the attack portion of the envelope.

        Args:
            attack_time: Length of the attack in seconds.
        """
        return self.ramp(attack_time)

    def make_decay(self, attack_time, decay_time) -> Signal:
        """
        Creates the decay portion of the envelope.

        Args:
            attack_time: Length of the attack in seconds.
            decay_time: Length of the decay time in seconds.
        """
        sustain = self.p("sustain").unsqueeze(1)
        a = 1.0 - sustain
        b = self.ramp(decay_time, start=attack_time, inverse=True)
        return torch.squeeze(a * b + sustain)

    def make_release(self, note_on_duration) -> Signal:
        """
        Creates the release portion of the envelope.

        Args:
            note_on_duration: Duration of midi note in seconds (release starts
                when the midi note is released).
        """
        return self.ramp(self.p("release"), start=note_on_duration, inverse=True)

    def __str__(self):  # pragma: no cover
        return (
            f"""ADSR(a={self.torchparameters['attack']}, """
            f"""d={self.torchparameters['decay']}, """
            f"""s={self.torchparameters['sustain']}, """
            f"""r={self.torchparameters['release']}, """
            f"""alpha={self.torchparameters['alpha']}"""
        )


class VCO(SynthModule):
    """
    Base class for voltage controlled oscillators.

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a modulation signal between [-1, 1]. An array of 0's returns a
    stationary audio signal at its base pitch.

    Args:
        synthconfig: An object containing synthesis settings that are shared
            across all modules, typically specified by
            :class:`~torchsynth.synth.Voice`, or some other, possibly custom
            :class:`~torchsynth.synth.AbstractSynth` subclass.
        phase: Initial oscillator phase.
    """

    default_parameter_ranges: List[ModuleParameterRange] = [
        ModuleParameterRange(
            -24.0,
            24.0,
            name="tuning",
            description="tuning adjustment for VCO in midi",
        ),
        ModuleParameterRange(
            -96.0,
            96.0,
            curve=0.2,
            symmetric=True,
            name="mod_depth",
            description="depth of the pitch modulation in semitones",
        ),
        ModuleParameterRange(
            -torch.pi,
            torch.pi,
            name="initial_phase",
            description="Initial phase for this oscillator",
        ),
    ]

    def output(self, midi_f0: T, mod_signal: Optional[Signal] = None) -> Signal:
        """
        Generates audio signal from modulation signal.

        Args:
            midi_f0: Fundamental of note in midi note value (0-127).
            mod_signal: Modulation signal to apply to the pitch.
        """
        assert midi_f0.shape == (self.batch_size,)

        if mod_signal is not None and mod_signal.shape != (
            self.batch_size,
            self.buffer_size,
        ):
            raise ValueError(
                "mod_signal has incorrect shape. Expected "
                f"{torch.Size([self.batch_size, self.buffer_size])}, "
                f"and received {mod_signal.shape}. Make sure the mod_signal "
                "being passed in is at full audio sampling rate."
            )

        control_as_frequency = self.make_control_as_frequency(midi_f0, mod_signal)

        if self.synthconfig.debug:
            assert (control_as_frequency >= 0).all() and (
                control_as_frequency <= self.nyquist
            ).all()

        cosine_argument = self.make_argument(control_as_frequency)
        cosine_argument += self.p("initial_phase").unsqueeze(1)
        output = self.oscillator(cosine_argument, midi_f0)
        return output.as_subclass(Signal)

    def make_control_as_frequency(
        self, midi_f0: T, mod_signal: Optional[Signal] = None
    ) -> Signal:
        """
        Generates a time-varying control signal in frequency (Hz) from a midi
        fundamental pitch and pitch-modulation signal.

        Args:
            midi_f0: Fundamental pitch value in midi.
            mod_signal: Pitch modulation signal in midi.
        """
        midi_f0 = (midi_f0 + self.p("tuning")).unsqueeze(1)

        # If there is no modulation, then convert the midi_f0 values to
        # frequency and return an expanded view that contains buffer size
        # number of values
        if mod_signal is None:
            control_hz = util.midi_to_hz(midi_f0)
            return control_hz.expand(-1, self.buffer_size)

        # If there is modulation, then add that to the fundamental,
        # clamp to a range [0.0, 127.0], then return in frequency Hz.
        modulation = self.p("mod_depth").unsqueeze(1) * mod_signal
        control = torch.clamp(midi_f0 + modulation, 0.0, 127.0)
        return util.midi_to_hz(control)

    def make_argument(self, freq: Signal) -> Signal:
        """
        Generates the phase argument to feed an oscillating function to
        generate an audio signal.

        Args:
            freq: Time-varying instantaneous frequency in Hz.
        """
        return torch.cumsum(2 * torch.pi * freq / self.sample_rate, dim=1)

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        """
        This function accepts a phase argument and generates output audio. It is
        implemented by the child class.

        Args:
            argument: The phase of the oscillator at each time sample.
            midi_f0: Fundamental frequency in midi.
        """
        raise NotImplementedError("Derived classes must override this method")


class SineVCO(VCO):
    """
    Simple VCO that generates a pitched sinusoid.
    """

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        """
        A cosine oscillator. ...Good ol' cosine.

        Args:
            argument: The phase of the oscillator at each time sample.
            midi_f0: Fundamental frequency in midi (ignored in this VCO).
        """
        return torch.cos(argument)


class FmVCO(VCO):
    """
    Frequency modulation VCO. Takes a modulation signal as instantaneous
    frequency (in Hz) rather than as a midi value.

    Typical modulation is calculated in pitch-space (midi). For FM to work,
    we have to change the order of calculations. Here the modulation depth is
    re-interpreted as the "modulation index" which is tied to the fundamental of
    the oscillator being modulated:

        :math:`I = \\Delta f / f_m`

    where :math:`I` is the modulation index, :math:`\\Delta f` is the frequency
    deviation imparted by the modulation, and :math:`f_m` is the modulation
    frequency, both in Hz.
    """

    # We include this override to output to make mod_signal non-optional
    def output(self, midi_f0: T, mod_signal: Signal) -> Signal:
        """
        Args:
            midi_f0: note value in midi
            mod_signal: audio rate frequency modulation signal
        """
        return super().output(midi_f0, mod_signal)

    def make_control_as_frequency(self, midi_f0: T, mod_signal) -> Signal:
        """
        Creates a time-varying control signal in instantaneous frequency (Hz).

        Args:
            midi_f0: Fundamental frequency in midi.
            mod_signal: FM modulation signal (interpreted as modulation index).
        """
        # Compute modulation in Hz space (rather than midi-space).
        f0_hz = util.midi_to_hz(midi_f0 + self.p("tuning")).unsqueeze(1)
        fm_depth = self.p("mod_depth").unsqueeze(1) * f0_hz
        modulation_hz = fm_depth * mod_signal
        return torch.clamp(f0_hz + modulation_hz, 0.0, self.nyquist)

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        """
        A cosine oscillator. ...Good ol' cosine.

        Args:
            argument: The phase of the oscillator at each time sample.
            midi_f0: Fundamental frequency in midi (ignored in this VCO).
        """
        return torch.cos(argument)


class SquareSawVCO(VCO):
    """
    An oscillator that can take on either a square or a sawtooth waveshape, and
    can sweep continuously between them, as determined by the
    :attr:`~torchsynth.module.SquareSawVCO.shape` parameter. A shape value of 0
    makes a square wave; a shape of 1 makes a saw wave.

    With apologies to Lazzarini and Timoney (2010).
    `"New perspectives on distortion synthesis for virtual analog oscillators."
    <https://doi.org/10.1162/comj.2010.34.1.28>`_
    Computer Music Journal 34, no. 1: 28-40.
    """

    default_parameter_ranges: List[
        ModuleParameterRange
    ] = VCO.default_parameter_ranges + [
        ModuleParameterRange(
            0.0, 1.0, name="shape", description="Waveshape - square to saw [0,1]"
        )
    ]

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        """
        Generates output square/saw audio given a phase argument.

        Args:
            argument: The phase of the oscillator at each time sample.
            midi_f0: Fundamental frequency in midi.
        """
        partials = self.partials_constant(midi_f0).unsqueeze(1)
        square = torch.tanh(torch.pi * partials * torch.sin(argument) / 2)
        shape = self.p("shape").unsqueeze(1)
        return (1 - shape / 2) * square * (1 + shape * torch.cos(argument))

    def partials_constant(self, midi_f0):
        """
        Calculates a value to determine the number of overtones in the resulting
        square / saw wave, in order to keep aliasing at an acceptable level.
        Higher fundamental frequencies require fewer partials for a rich sound;
        lower-frequency sounds can safely have more partials without causing
        audible aliasing.

        Args:
            midi_f0: Fundamental frequency in midi.
        """
        max_pitch = (
            midi_f0 + self.p("tuning") + torch.maximum(self.p("mod_depth"), tensor(0))
        )
        max_f0 = util.midi_to_hz(max_pitch)
        return 12000 / (max_f0 * torch.log10(max_f0))


class VCA(SynthModule):
    """
    Voltage controlled amplifier.

    The VCA shapes the amplitude of an audio input signal over time, as
    determined by a control signal. To shape control-rate signals, use
    :class:`torchsynth.module.ControlRateVCA`.
    """

    def output(self, audio_in: Signal, control_in: Signal) -> Signal:
        """
        Args:
            audio: Audio input to shape with the VCA.
            amp_control: Time-varying amplitude modulation signal.
        """
        return audio_in * control_in


class ControlRateVCA(ControlRateModule):
    """
    Voltage controlled amplifier.

    The VCA shapes the amplitude of a control input signal over time, as
    determined by another control signal. To shape audio-rate signals, use
    :class:`torchsynth.module.VCA`.
    """

    def output(self, audio_in: Signal, control_in: Signal) -> Signal:
        """
        Args:
            control: Control signal input to shape with the VCA.
            amp_control: Time-varying amplitude modulation signal.
        """
        return audio_in * control_in


class Noise(SynthModule):
    """
    Generates white noise that is the same length as the buffer.

    For performance noise is pre-computed. In order to maintain
    reproducibility noise must be computed on the CPU and then transferred
    to the GPU, if a GPU is being used. We pre-compute
    :attr:`~torchsynth.config.BASE_REPRODUCIBLE_BATCH_SIZE`
    samples of noise and then repeat those for larger batch sizes.

    To keep things fast we only support multiples of
    :attr:`~torchsynth.config.BASE_REPRODUCIBLE_BATCH_SIZE`
    when reproducibility mode is enabled. For example, if you batch size
    is 4 times :attr:`~torchsynth.config.BASE_REPRODUCIBLE_BATCH_SIZE`, then
    you get the same noise signals repeated 4 times.

    `Note`: If you have multiple `Noise` modules in the same
    :class:`~torchsynth.synth.AbstractSynth`, make sure you instantiate
    each `Noise` with a unique seed.

    Args:
        synthconfig: See :class:`~torchsynth.module.SynthModule`
        seed: random number generator seed for white noise
    """

    __noise_batch_size: int = BASE_REPRODUCIBLE_BATCH_SIZE
    # Unfortunately, Final is not supported until Python 3.8
    # noise_batch_size: Final[int] = BATCH_SIZE_FOR_REPRODUCIBILITY

    def __init__(self, synthconfig: SynthConfig, seed: int, **kwargs):
        super().__init__(synthconfig, **kwargs)

        # Pre-compute default batch size number of noise samples
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # In reproducible mode, we support batch sizes that are multiples
        # of the BASE_REPRODUCIBLE_BATCH_SIZE
        if self.synthconfig.reproducible:
            if self.batch_size % self.__noise_batch_size != 0:
                raise ValueError(
                    f"Batch size must be a multiple of {self.__noise_batch_size} "
                    "when using reproducible mode. Either change your batch size,"
                    "or set reproducible=False in the SynthConfig for this module."
                )

            noise = torch.empty(
                (self.__noise_batch_size, self.buffer_size), device="cpu"
            )
            noise.data.uniform_(-1.0, 1.0, generator=generator)
            if self.batch_size > self.__noise_batch_size:
                noise = noise.repeat(self.batch_size // self.__noise_batch_size, 1)
        else:
            # Non-reproducible mode, just render noise of batch size
            noise = torch.empty((self.batch_size, self.buffer_size), device="cpu")
            noise.data.uniform_(-1.0, 1.0, generator=generator)

        self.register_buffer("noise", noise.to(self.device))

    def output(self) -> Signal:
        return self.noise.as_subclass(Signal)


class LFO(ControlRateModule):
    """
    Low Frequency Oscillator.

    The LFO shape can be any mixture of sine, triangle, saw, reverse saw, and
    square waves. Contributions of each base-shape are determined by the
    :attr:`~torchsynth.module.LFO.lfo_types` values, which are between 0 and 1.

    Args:
        synthconfig: See :class:`~torchsynth.module.SynthConfig`.
        exponent: A non-negative value that determines the discrimination of the
            soft-max selector for LFO shapes. Higher values will tend to favour
            one LFO shape over all others. Lower values will result in a more
            even blend of LFO shapes.
    """

    default_ranges: List[ModuleParameterRange] = [
        ModuleParameterRange(
            0.0,
            20.0,
            curve=0.25,
            name="frequency",
            description="Frequency in Hz of oscillation",
        ),
        ModuleParameterRange(
            -10.0,
            20.0,
            curve=0.5,
            symmetric=True,
            name="mod_depth",
            description="LFO rate modulation in Hz",
        ),
        ModuleParameterRange(
            -torch.pi,
            torch.pi,
            name="initial_phase",
            description="Initial phase of LFO",
        ),
    ]

    def __init__(
        self,
        synthconfig: SynthConfig,
        exponent: T = tensor(2.718281828),  # e
        **kwargs: Dict[str, T],
    ):
        self.lfo_types = ["sin", "tri", "saw", "rsaw", "sqr"]
        self.default_parameter_ranges = self.default_ranges.copy()
        for lfo in self.lfo_types:
            self.default_parameter_ranges.append(
                ModuleParameterRange(
                    0.0,
                    1.0,
                    name=f"{lfo}",
                    description=f"Selection parameter for {lfo} LFO",
                )
            )
        super().__init__(synthconfig, **kwargs)
        self.exponent = exponent

    def output(self, mod_signal: Optional[Signal] = None) -> Signal:
        """
        Generates low frequency oscillator control signal.

        Args:
            mod_signal:  LFO rate modulation signal in Hz. To modulate the
                depth of the LFO, use :class:`torchsynth.module.ControlRateVCA`.
        """
        # This module accepts signals at control rate!
        if mod_signal is not None:
            assert mod_signal.shape == (self.batch_size, self.control_buffer_size)

        # Create frequency signal
        frequency = self.make_control(mod_signal)
        argument = torch.cumsum(2 * torch.pi * frequency / self.control_rate, dim=1)
        argument = argument + self.p("initial_phase").unsqueeze(1)

        # Get LFO shapes
        shapes = torch.stack(self.make_lfo_shapes(argument), dim=1).as_subclass(Signal)

        # Apply mode selection to the LFO shapes
        mode = torch.stack([self.p(lfo) for lfo in self.lfo_types], dim=1)
        mode = torch.pow(mode, self.exponent)
        mode = mode / torch.sum(mode, dim=1, keepdim=True)

        return torch.matmul(mode.unsqueeze(1), shapes).squeeze(1).as_subclass(Signal)

    def make_control(self, mod_signal: Optional[Signal] = None) -> Signal:
        """
        Applies the LFO-rate modulation signal to the LFO base frequency.

        Args:
            mod_signal: Modulation signal in Hz. Positive values increase the
                LFO base rate; negative values decrease it.
        """
        frequency = self.p("frequency").unsqueeze(1)

        # If no modulation, then return a view of the frequency of this
        # LFO expanded to the control buffer size
        if mod_signal is None:
            return frequency.expand(-1, self.control_buffer_size)

        modulation = self.p("mod_depth").unsqueeze(1) * mod_signal
        return torch.maximum(frequency + modulation, tensor(0.0))

    def make_lfo_shapes(self, argument: Signal) -> Tuple[T, T, T, T, T]:
        """
        Generates five separate signals for each LFO shape and returns them as a
        tuple, to be mixed by :func:`torchsynth.module.LFO.output`.

        Args:
            argument: Time-varying phase to generate LFO signals.
        """
        cos = torch.cos(argument + torch.pi)
        square = torch.sign(cos)

        cos = (cos + 1.0) / 2.0
        square = (square + 1.0) / 2.0

        saw = torch.remainder(argument, 2 * torch.pi) / (2 * torch.pi)
        rev_saw = 1.0 - saw

        triangle = 2 * saw
        triangle = torch.where(triangle > 1.0, 2.0 - triangle, triangle)

        return cos, triangle, saw, rev_saw, square


class ModulationMixer(SynthModule):
    """
    A modulation matrix that combines :math:`N` input modulation signals to make
    :math:`M` output modulation signals. Each output is a linear combination of
    all in input signals, as determined by an :math:`N \times M` mixing matrix.

    Args:
        synthconfig: See :class:`~torchsynth.module.SynthConfig`.
        n_input: Number of input signals to module mix.
        n_output: Number of output signals to generate.
        curves: A positive value that determines the contribution of each
            input signal to the other signals. A low value discourages
            over-mixing.
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        n_input: int,
        n_output: int,
        curves: Optional[List[float]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        **kwargs: Dict[str, T],
    ):
        # Parameter curves can be used to modify the parameter mapping
        # for each input modulation source to the outputs
        if curves is not None:
            assert len(curves) == n_input
        else:
            curves = [0.5] * n_input

        custom_names = False
        if input_names is not None:
            assert len(input_names) == n_input
            assert output_names is not None
            assert len(output_names) == n_output
            custom_names = True

        # Need to create the parameter ranges before calling super().__init
        self.default_parameter_ranges = []
        for i in range(n_input):
            for j in range(n_output):
                # Apply custom param name if it was passed in
                if custom_names:
                    name = f"{input_names[i]}->{output_names[j]}"
                    description = f"Modulation {input_names[i]} to {output_names[j]}"
                else:
                    name = f"{i}->{j}"
                    description = f"Modulation {i} to {j}"

                self.default_parameter_ranges.append(
                    ModuleParameterRange(
                        0.0,
                        1.0,
                        curve=curves[i],
                        name=name,
                        description=description,
                    )
                )

        super().__init__(synthconfig, **kwargs)
        self.n_input = n_input
        self.n_output = n_output

    def forward(self, *signals: Signal) -> Tuple[Signal]:
        """
        Performs mixture of modulation signals.
        """

        # Get params into batch_size x n_output x n_input matrix
        params = torch.stack([self.p(p) for p in self.torchparameters], dim=1)
        params = params.view(self.batch_size, self.n_input, self.n_output)
        params = torch.swapaxes(params, 1, 2)

        # Make sure there is the same number of input signals as mix params
        assert len(signals) == params.shape[2]
        signals = torch.stack(signals, dim=1)

        modulation = torch.chunk(torch.matmul(params, signals), self.n_output, dim=1)
        return tuple(m.squeeze(1).as_subclass(Signal) for m in modulation)


class AudioMixer(SynthModule):
    """
    Sums together N audio signals and applies range-normalization if the
    resulting signal is outside of [-1, 1].
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        n_input: int,
        curves: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
        **kwargs: Dict[str, T],
    ):
        # Parameter curves can be used to modify the parameter mapping
        # for each input modulation source to the outputs
        if curves is not None:
            assert len(curves) == n_input
        else:
            curves = [1.0] * n_input

        # If param names were passed in, make sure we got the right number
        if names is not None:
            assert len(names) == n_input

        # Need to create the parameter ranges before calling super().__init
        self.default_parameter_ranges = []
        for i in range(n_input):
            name = f"level{i}" if names is None else names[i]
            self.default_parameter_ranges.append(
                ModuleParameterRange(
                    0.0,
                    1.0,
                    curve=curves[i],
                    name=name,
                    description=f"{name} mix level",
                )
            )

        super().__init__(synthconfig, **kwargs)
        self.n_input = n_input

    def output(self, *signals: Signal) -> Signal:
        """
        Returns a mixed signal from an array of input signals.
        """

        # Turn params into matrix
        params = torch.stack([self.p(p) for p in self.torchparameters], dim=1)

        # Make sure we received the correct number of input signals
        signals = torch.stack(signals, dim=1)
        assert signals.shape[1] == params.shape[1]

        # Mix signals and normalize output if required
        output = torch.matmul(params.unsqueeze(1), signals).squeeze(1)
        return util.normalize_if_clipping(output)


class ControlRateUpsample(SynthModule):
    """
    Upsample control signals to the global sampling rate

    Uses linear interpolation to resample an input control signal to the
    audio buffer size set in synthconfig.
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, T],
    ):
        super().__init__(synthconfig, device, **kwargs)
        self.upsample = torch.nn.Upsample(
            self.synthconfig.buffer_size, mode="linear", align_corners=True
        )

    def output(self, signal: Signal) -> Signal:
        return self.upsample(signal.unsqueeze(1)).squeeze(1)


class CrossfadeKnob(SynthModule):
    """
    Crossfade knob parameter with no signal generation
    """

    default_parameter_ranges: List[ModuleParameterRange] = [
        ModuleParameterRange(
            0.0,
            1.0,
            name="ratio",
            description="crossfade knob",
        ),
    ]


class MonophonicKeyboard(SynthModule):
    """
    A keyboard controller module. Mimics a mono-synth keyboard and contains
    parameters that output a midi_f0 and note duration.
    """

    default_parameter_ranges: List[ModuleParameterRange] = [
        ModuleParameterRange(
            0.0,
            127.0,
            curve=1.0,
            name="midi_f0",
            description="pitch value in 'midi' (69 = 440Hz)",
        ),
        ModuleParameterRange(
            0.01,
            4.0,
            curve=0.5,
            name="duration",
            description="note-on button, in seconds",
        ),
    ]

    def forward(self) -> Tuple[T, T]:
        return self.p("midi_f0"), self.p("duration")


class SoftModeSelector(SynthModule):
    """
    A soft mode selector.
    If there are n different modes, return a probability distribution over them.

    TODO: Would be nice to sample in a way that maximizes
    KL-divergence from uniform: https://github.com/torchsynth/torchsynth/issues/165
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        n_modes: int,
        exponent: T = tensor(2.718281828),  # e
        **kwargs: Dict[str, T],
    ):
        """
        exponent determines how strongly to scale each [0,1] value prior
        to normalization. We should probably tune this:
        https://github.com/torchsynth/torchsynth/issues/165
        """
        # Need to create the parameter ranges before calling super().__init
        self.default_parameter_ranges = [
            ModuleParameterRange(
                0.0,
                1.0,
                name=f"mode{i}weight",
                description=f"mode{i} weight, before normalization",
            )
            for i in range(n_modes)
        ]
        super().__init__(synthconfig=synthconfig, **kwargs)
        self.exponent = exponent

    def forward(self) -> Tuple[T, T]:
        """
        Normalize all mode weights so they sum to 1.0
        """
        # Is this tensor creation slow?
        # But usually parameter stuff is not the bottleneck
        params = torch.stack([p.data for p in self.torchparameters.values()])
        params = torch.pow(params, exponent=self.exponent)
        return params / torch.sum(params, dim=0)


class HardModeSelector(SynthModule):
    """
    A hard mode selector.
    NOTE: This is non-differentiable.
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        n_modes: int,
        **kwargs: Dict[str, T],
    ):
        # Need to create the parameter ranges before calling super().__init
        self.default_parameter_ranges = [
            ModuleParameterRange(
                0.0,
                1.0,
                name=f"mode{i}weight",
                description=f"mode{i} weight, before argmax",
            )
            for i in range(n_modes)
        ]
        super().__init__(synthconfig=synthconfig, **kwargs)

    def forward(self) -> Tuple[T, T]:
        # Is this tensor creation slow?
        # But usually parameter stuff is not the bottleneck
        origparams = torch.stack([p.data for p in self.torchparameters.values()])
        idx = torch.argmax(origparams, dim=0)
        return F.one_hot(idx, num_classes=origparams.shape[0]).T
