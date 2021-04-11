"""
Synth modules in Torch.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as tensor
from torch import Tensor as T

import torchsynth.util as util
from torchsynth.config import BATCH_SIZE_FOR_REPRODUCIBILITY, SynthConfig
from torchsynth.parameter import ModuleParameter, ModuleParameterRange
from torchsynth.signal import Signal


class SynthModule(nn.Module):
    """
    Base class for synthesis modules, in torch.
    All parameters are assumed to be 1D tensors,
    the dimension size being the batch size.

    WARNING: TorchSynthModules should be atomic and not
    contain other SynthModules. This is similar to a modular synth,
    where modules don't contain submodules.
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
        """
        Args:
            synthconfig (:obj:`SynthGlobals`): These are global settings
            shared across all modules in the same synth.

            device (:obj:`torch.device`):
            The device for creating all tensors.

        NOTE:
        __init__ should only set parameters.
        We shouldn't be doing computations in __init__ because
        the computations will change when the parameters change.

        batch_size is the number of settings we are rendering at once.
        """
        nn.Module.__init__(self)
        self.synthconfig = synthconfig
        self.device = device
        self.synthconfig.to(device)
        self.torchparameters: nn.ParameterDict = nn.ParameterDict()
        self.parameter_ranges = []
        # If this module needs a random seed, here it is
        self.seed: Optional[int] = None

        if self.default_parameter_ranges is not None:
            # We want to create copies of the parameter ranges otherwise each instance
            # of the same module type (ex. ADSR) will reference the same param range
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
        assert self.synthconfig.batch_size.ndim == 0
        return self.synthconfig.batch_size

    @property
    def sample_rate(self) -> T:
        assert self.synthconfig.sample_rate.ndim == 0
        return self.synthconfig.sample_rate

    @property
    def nyquist(self):
        return self.sample_rate / 2.0

    @property
    def eps(self) -> float:
        return self.synthconfig.eps

    @property
    def buffer_size(self) -> T:
        assert self.synthconfig.buffer_size.ndim == 0
        return self.synthconfig.buffer_size

    def to_buffer_size(self, signal: Signal) -> Signal:
        return util.fix_length(signal, self.buffer_size)

    def seconds_to_samples(self, seconds: T) -> T:
        """
        Converts a tensor of seconds to number of samples at the current sample rate.
        Returns the number of samples as a float and can be fractional.
        """
        return seconds * self.sample_rate

    def _forward(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Each SynthModule should override this.
        """
        raise NotImplementedError("Derived classes must override this method")

    def forward(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Wrapper for _forward that ensures a buffer_size length output.
        TODO: Make this forward0d() after everything is 1D
        """
        signal = self._forward(*args, **kwargs)
        buffered = self.to_buffer_size(signal)
        return buffered

    def add_parameters(self, parameters: List[ModuleParameter]):
        """
        Add parameters to this SynthModule's torch parameter dictionary.
        """
        for parameter in parameters:
            assert parameter.parameter_name not in self.torchparameters
            assert parameter.shape == (self.batch_size,)
            self.torchparameters[parameter.parameter_name] = parameter

    def get_parameter(self, parameter_id: str) -> ModuleParameter:
        """
        Get a single ModuleParameter for this module

        Args:
            parameter_id: str: Id of the parameter to return
        """
        value = self.torchparameters[parameter_id]
        assert value.shape == (self.batch_size,)
        return value

    def get_parameter_0to1(self, parameter_id: str) -> T:
        """
        Get the value of a parameter in the range of [0,1]

        Args:
            parameter_id (str)  :   Id of the parameter to return the value for
        """
        value = self.torchparameters[parameter_id]
        assert value.shape == (self.batch_size,)
        return value

    def set_parameter(self, parameter_id: str, value: T):
        """
        Update a specific parameter value, ensuring that it is within a specified
        range

        Args:
            parameter_id (str)  : Id of the parameter to update
            value (T)           : Value to update parameter with
        """
        value = value.to(self.device)
        self.torchparameters[parameter_id].to_0to1(value)
        value = self.torchparameters[parameter_id].data
        assert torch.all(0.0 <= value) and torch.all(value <= 1.0)
        assert value.shape == (self.batch_size,)

    def set_parameter_0to1(self, parameter_id: str, value: T):
        """
        Update a specific parameter with a value in the range [0,1]

        Args:
            parameter_id (str)  : Id of the parameter to update
            value (T)           : Value to update parameter with
        """
        value = value.to(self.device)
        assert torch.all(0.0 <= value) and torch.all(value <= 1.0)
        assert value.shape == (self.batch_size,)
        self.torchparameters[parameter_id].data = value

    def p(self, parameter_id: str) -> T:
        """
        Convenience method for getting the parameter value.
        """
        value = self.torchparameters[parameter_id].from_0to1()
        assert value.shape == (self.batch_size,)
        return value

    def to(self, device=None, **kwargs):
        """
        Overriding the to call for nn.Module to transfer this module to device. This
        makes sure that the ParameterRanges for each ModuleParamter and the globals
        are also transferred to the correct device.
        """
        self.update_device(device)
        return super().to(device=device, **kwargs)

    def update_device(self, device=None):
        """
        This handles the device transfer tasks that are not managed by PyTorch.
        """
        self.synthconfig.to(device)
        self.device = device
        # Currently the parameters ranges themselves are not move to a different
        # device, but each ParameterRange object is device aware if it needs to be
        # TODO: Do ParameterRanges need to be device aware? See issue #240
        for name, parameter in self.torchparameters.items():
            parameter.parameter_range.to(device)


class ControlRateModule(SynthModule):
    """
    Control rate module is an abstract base class for SynthModules that run
    at a control rate as opposed to full audio rate.
    """

    @property
    def sample_rate(self) -> T:
        raise NotImplementedError("This module operates at control rate")

    @property
    def buffer_size(self) -> T:
        raise NotImplementedError("This module uses control buffer size")

    @property
    def control_rate(self) -> T:
        assert self.synthconfig.control_rate.ndim == 0
        return self.synthconfig.control_rate

    @property
    def control_buffer_size(self) -> T:
        assert self.synthconfig.control_buffer_size.ndim == 0
        return self.synthconfig.control_buffer_size

    def to_buffer_size(self, signal: Signal) -> Signal:
        return util.fix_length(signal, self.control_buffer_size)

    def seconds_to_samples(self, seconds: T) -> T:
        """
        Converts a tensor of seconds to number of samples at the current control rate.
        Returns the number of samples as a float and can be fractional.
        """
        return seconds * self.control_rate

    def _forward(self, *args: Any, **kwargs: Any) -> Signal:  # pragma: no cover
        """
        Each SynthModule should override this.
        """
        raise NotImplementedError("Derived classes must override this method")


class ADSR(ControlRateModule):
    """
    Envelope class for building a control rate ADSR signal.
    """

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

    def _forward(self, note_on_duration: T) -> Signal:
        """Generate an ADSR envelope.

        By default, this envelope reacts as if it was triggered with midi, for
        example playing a keyboard. Each midi event has a beginning and end:
        note-on, when you press the key down; and note-off, when you release the
        key. `note_on_duration` is the amount of time that the key is depressed.

        During the note-on, the envelope moves through the attack and decay
        sections of the envelope. This leads to musically-intuitive, but
        programatically-counterintuitive behaviour:

        Example:
            E.g., assume attack is .5 seconds, and decay is .5 seconds. If a note is
            held for .75 seconds, the envelope won't pass through the entire
            attack-and-decay (specifically, it will execute the entire attack, and
            only .25 seconds of the decay).

        If this is confusing, don't worry about it. ADSR's do a lot of work
        behind the scenes to make the playing experience feel natural.
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

    def _ramp(
        self, duration: T, start: Optional[T] = None, inverse: Optional[bool] = False
    ) -> Signal:
        """
        Makes a ramp of a given duration in seconds.

        The construction of this matrix is rather cryptic. Essentially, this
        method works by tilting and clipping ramps between 0 and 1, then
        applying some scaling factor (`alpha`).

        Parameters
        ----------
        duration T: Length of the ramp in seconds
        start Optional[T]: Initial delay of ramp in seconds. Defaults to None (0 start)
        inverse Optional[bool]: Flip ramp. Becomes a ramp down. Defaults to False
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

        """
        The following is a workaround. In inverse mode, a ramp with 0 duration
        (that is all 1's) becomes all 0's, which is a problem for the
        ultimate calculation of the ADSR signal (a * d * r => 0's). So this
        replaces only rows who sum to 0 (i.e., all components are zero).
        """
        if inverse:
            ramp = torch.where(duration > 0.0, 1.0 - ramp, ramp)

        # Apply scaling factor.
        ramp = torch.pow(ramp, self.alpha)
        return ramp.as_subclass(Signal)

    def make_attack(self, attack_time) -> Signal:
        return self._ramp(attack_time)

    def make_decay(self, attack_time, decay_time) -> Signal:
        sustain = self.p("sustain").unsqueeze(1)
        a = 1.0 - sustain
        b = self._ramp(decay_time, start=attack_time, inverse=True)
        return torch.squeeze(a * b + sustain)

    def make_release(self, note_on_duration) -> Signal:
        return self._ramp(self.p("release"), start=note_on_duration, inverse=True)

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
    Base class for voltage controlled oscillators (VCO).

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a modulation signal between [-1, 1]. An array of 0's returns a
    stationary audio signal at its base pitch.

    Args:
        synthconfig (SynthConfig) : global args, see SynthModule
        phase (:obj:'T',optional) :   initial phase values
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

    def _forward(self, midi_f0: T, mod_signal: Signal) -> Signal:
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
        assert midi_f0.shape == (self.batch_size,)

        control_as_frequency = self.make_control_as_frequency(midi_f0, mod_signal)

        if self.synthconfig.debug:
            assert (control_as_frequency >= 0).all() and (
                control_as_frequency <= self.nyquist
            ).all()

        cosine_argument = self.make_argument(control_as_frequency)
        cosine_argument += self.p("initial_phase").unsqueeze(1)
        output = self.oscillator(cosine_argument, midi_f0)
        return output.as_subclass(Signal)

    def make_control_as_frequency(self, midi_f0: T, mod_signal: Signal) -> Signal:
        modulation = self.p("mod_depth").unsqueeze(1) * mod_signal
        control_as_midi = torch.clamp(
            (midi_f0 + self.p("tuning")).unsqueeze(1) + modulation, 0.0, 127.0
        )
        return util.midi_to_hz(control_as_midi)

    def make_argument(self, freq: Signal) -> Signal:
        """
        Generates the phase argument to feed a cosine function to make audio.
        """
        return torch.cumsum(2 * torch.pi * freq / self.sample_rate, dim=1)

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        """
        Dummy method. Overridden by child class VCO's.
        """
        raise NotImplementedError("Derived classes must override this method")


class SineVCO(VCO):
    """
    Simple VCO that generates a pitched sinusoid.

    Derives from VCO, it simply implements a cosine function as oscillator.
    """

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        return torch.cos(argument)


class TorchFmVCO(VCO):
    """
    # TODO Turn this into its own voice so we can be a bit smarter about aliasing
    # See https://github.com/torchsynth/torchsynth/issues/145
    Frequency modulation VCO. Takes `mod_signal` as instantaneous frequency.

    Typical modulation is calculated in pitch-space (midi). For FM to work,
    we have to change the order of calculations. Here `mod_depth` is interpreted
    as the "modulation index" which is tied to the fundamental of the oscillator
    being modulated:

        modulation_index = frequency_deviation / modulation_frequency
    """

    def make_control_as_frequency(self, midi_f0: T, mod_signal: Signal) -> Signal:
        # Compute modulation in Hz space (rather than midi-space).
        f0_hz = util.midi_to_hz(midi_f0 + self.p("tuning")).unsqueeze(1)
        fm_depth = self.p("mod_depth").unsqueeze(1) * f0_hz
        modulation_hz = fm_depth * mod_signal
        return torch.clamp(f0_hz + modulation_hz, 0.0, self.nyquist)

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        # Classically, FM operators are sine waves.
        return torch.cos(argument)


class SquareSawVCO(VCO):
    """
    VCO that can be either a square or a sawtooth waveshape.
    Tweak with the shape parameter. (0 is square.)

    With apologies to:

    Lazzarini, Victor, and Joseph Timoney. "New perspectives on distortion synthesis for
        virtual analog oscillators." Computer Music Journal 34, no. 1 (2010): 28-40.
    """

    default_parameter_ranges: List[
        ModuleParameterRange
    ] = VCO.default_parameter_ranges + [
        ModuleParameterRange(
            0.0, 1.0, name="shape", description="Waveshape - square to saw [0,1]"
        )
    ]

    def oscillator(self, argument: Signal, midi_f0: T) -> Signal:
        partials = self.partials_constant(midi_f0).unsqueeze(1)
        square = torch.tanh(torch.pi * partials * torch.sin(argument) / 2)
        shape = self.p("shape").unsqueeze(1)
        return (1 - shape / 2) * square * (1 + shape * torch.cos(argument))

    def partials_constant(self, midi_f0):
        """
        Constant value that determines the number of partials in the resulting
        square / saw wave in order to keep aliasing at an acceptable level.
        Higher frequencies require fewer partials whereas lower frequency sounds
        can safely have more partials without causing audible aliasing.
        """
        max_pitch = (
            midi_f0 + self.p("tuning") + torch.maximum(self.p("mod_depth"), tensor(0))
        )
        max_f0 = util.midi_to_hz(max_pitch)
        return 12000 / (max_f0 * torch.log10(max_f0))


class VCA(SynthModule):
    """
    Voltage controlled amplifier.
    """

    def _forward(self, audio_in: Signal, control_in: Signal) -> Signal:
        return audio_in * control_in


class ControlRateVCA(ControlRateModule):
    """
    A VCA that operates at control rate
    """

    def _forward(self, audio_in: Signal, control_in: Signal) -> Signal:
        return audio_in * control_in


class Noise(SynthModule):
    """
    Generates white noise that is the same length as the buffer.

    A batch size number of noise samples are pre-computed for performance.
    A seed for the noise generator is required. If you use multiple of
    these in a synth then choose different seeds for each instance.
    """

    # Do we really want deterministic noise within each batch?
    # https://github.com/torchsynth/torchsynth/issues/250
    noise_batch_size: int = BATCH_SIZE_FOR_REPRODUCIBILITY
    # Unfortunately, Final is not supported until Python 3.8
    # noise_batch_size: Final[int] = BATCH_SIZE_FOR_REPRODUCIBILITY

    def __init__(self, synthconfig: SynthConfig, seed: int, **kwargs):
        super().__init__(synthconfig, **kwargs)

        # https://github.com/torchsynth/torchsynth/issues/255
        if (
            self.batch_size != BATCH_SIZE_FOR_REPRODUCIBILITY
            and self.synthconfig.reproducible
        ):
            raise ValueError(
                "Reproducibility currently only supported "
                f"with batch_size = {BATCH_SIZE_FOR_REPRODUCIBILITY}. "
                "If you want a different batch_size, "
                "initialize SynthConfig with reproducible=False"
            )

        # Pre-compute default batch size number of noise samples
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.empty((self.noise_batch_size, self.buffer_size), device="cpu")
        noise.data.uniform_(-1.0, 1.0, generator=generator)

        # Repeat the noise batches if the current batch size is larger than the default
        if self.batch_size > self.noise_batch_size:
            if self.batch_size % self.noise_batch_size != 0:
                raise ValueError(
                    "Batch size is not divisible by the default "
                    f"batch size of {self.noise_batch_size}"
                )

            noise = noise.repeat(self.batch_size // self.noise_batch_size, 1)

        self.register_buffer("noise", noise.to(self.device))
        self.offset = 0

    def _forward(self) -> Signal:
        # Return noise quickly if we are returning the whole batch
        # and don't need to offset at all.
        # TODO: Is this actually correct?
        if self.batch_size >= self.noise_batch_size and self.offset == 0:
            return self.noise.as_subclass(Signal)

        # If the batch size is smaller than the default batch size then
        # we need to select a portion of the noise samples. Offset
        # is used to cycle through all the noise in the default batch size
        # or select the correct noise sample from the default batch
        # TODO https://github.com/torchsynth/torchsynth/issues/255
        if self.offset == 0:
            noise = self.noise[: self.batch_size].as_subclass(Signal)
        else:
            noise = torch.roll(self.noise, (-self.offset, 0), dims=(0, 1))

        self.offset = (self.offset + self.batch_size) % self.noise_batch_size
        return noise[: self.batch_size].as_subclass(Signal)


class LFO(ControlRateModule):
    """
    An abstract base class for low frequency oscillators
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

    def _forward(self, mod_signal: Signal) -> Signal:
        """
        Must be implemented in deriving classes
        """
        # This module accepts signals at control rate!
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

    def make_control(self, mod_signal: Signal) -> Signal:
        modulation = self.p("mod_depth").unsqueeze(1) * mod_signal
        return torch.maximum(self.p("frequency").unsqueeze(1) + modulation, tensor(0.0))

    def make_lfo_shapes(self, argument: Signal) -> Tuple[T, T, T, T, T]:
        """
        Creates a set of LFO shapes at the correct frequency:
        Sinusoid, Triangle, Saw, Reverse Saw, and Square
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
    A modulation matrix that takes in N input modulation signals
    and combines them together to create M output modulation signals.
    Each output is a linear combination of all in the input signals
    using the mixing levels defined by NxM mixing parameters.

    Parameters
    ----------
    synthconfig (SynthGlobals) :   Synth config settings
    n_input     (int)           :   Number of signals this module will mix
    n_output    (int)           :   Number of output signals
    curves  (optional, list)    :   A list of curve values to use for underlying
                                    mix level parameters for each mod source.
                                    Defaults to 0.5. Must be same number of input.
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        n_input: int,
        n_output: int,
        curves: Optional[List[float]] = None,
        **kwargs: Dict[str, T],
    ):
        # Parameter curves can be used to modify the parameter mapping
        # for each input modulation source to the outputs
        if curves is not None:
            assert len(curves) == n_input
        else:
            curves = [0.5] * n_input

        # Need to create the parameter ranges before calling super().__init
        self.default_parameter_ranges = []
        for i in range(n_input):
            for j in range(n_output):
                self.default_parameter_ranges.append(
                    ModuleParameterRange(
                        0.0,
                        1.0,
                        curve=curves[i],
                        name=f"level{i}_{j}",
                        description=f"Mix level for input {i} to output {j}",
                    )
                )

        super().__init__(synthconfig, **kwargs)
        self.n_input = n_input
        self.n_output = n_output

    def forward(self, *signals: Signal) -> Tuple[Signal]:
        """
        Mix together a set of modulation signals.
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
    Sums together a set of N signals together and applies normalization if required
    """

    def __init__(
        self,
        synthconfig: SynthConfig,
        n_input: int,
        curves: Optional[List[float]] = None,
        **kwargs: Dict[str, T],
    ):
        # Parameter curves can be used to modify the parameter mapping
        # for each input modulation source to the outputs
        if curves is not None:
            assert len(curves) == n_input
        else:
            curves = [1.0] * n_input

        # Need to create the parameter ranges before calling super().__init
        self.default_parameter_ranges = []
        for i in range(n_input):
            self.default_parameter_ranges.append(
                ModuleParameterRange(
                    0.0,
                    1.0,
                    curve=curves[i],
                    name=f"level{i}",
                    description=f"Mix level for input {i}",
                )
            )

        super().__init__(synthconfig, **kwargs)
        self.n_input = n_input

    def _forward(self, *signals: Signal) -> Signal:

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

    def _forward(self, signal: Signal) -> Signal:
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
