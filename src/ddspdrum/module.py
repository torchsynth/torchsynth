"""
Synth modules.

    TODO    :   - ADSR needs fixing. sustain duration should actually decay.
                - Convert operations to tensors, obvs.
"""

import numpy as np
from scipy.signal import resample

from ddspdrum.defaults import CONTROL_RATE, SAMPLE_RATE
from ddspdrum.util import fix_length, midi_to_hz


class SynthModule:
    """
    Base class for synthesis modules.
    """

    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.control_rate = CONTROL_RATE

    def control_to_sample_rate(self, control: np.array) -> np.array:
        """
        Resample a control signal to the sample rate
        TODO: One thing I worry about with all these casts to
        ints back and forth between sample and control rate is
        off-by-one errors.
        I'm beginning to believe we should standardize as much
        as possible on nsamples in most places instead of
        float duration in seconds, and just convert to ncontrol
        when needed..
        """
        # Right now it appears that all signals are 1d, but later
        # we'll probably convert things to 2d: instance x signal

        if self.control_rate == self.sample_rate:
            return control
        else:
            assert control.ndim == 1
            num_samples = int(
                round(len(control) * self.sample_rate / self.control_rate)
            )
            return resample(control, num_samples)


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

        super().__init__()
        assert alpha >= 0
        self.alpha = alpha

        assert s >= 0 and s <= 1
        self.s = s

        assert a >= 0
        self.a = a

        assert d >= 0
        self.d = d

        assert r >= 0
        self.r = r

    def __call__(self, sustain_duration: float = 0):
        """Generate an envelope that sustains for a given duration in seconds.

        Generates a control-rate envelope signal with given attack, decay and
        release times, sustained for `sustain_duration` in seconds. E.g., an
        envelope with no attack or decay, a sustain duration of 1 and a 0.5
        release will last for 1.5 seconds.

        """

        assert sustain_duration >= 0
        return np.concatenate(
            (
                self.note_on,
                self.sustain(sustain_duration),
                self.note_off,
            )
        )

    def _ramp(self, duration: float):
        """Makes a ramp of a given duration in seconds at control rate.

        This function is used for the piece-wise construction of the envelope
        signal. Its output monotonically increases from 0 to 1. As a result,
        each component of the envelope is a scaled and possibly reversed
        version of this ramp:

        attack      -->     returns an `a`-length ramp, as is.
        decay       -->     `d`-length reverse ramp, descends from 1 to `s`.
        release     -->     `r`-length reverse ramp, descends from `s` to 0.

        Its curve is determined by alpha:

        alpha = 1 --> linear,
        alpha > 1 --> exponential,
        alpha < 1 --> logarithmic.

        """

        t = np.linspace(
            0, duration, int(round(duration * self.control_rate)), endpoint=False
        )
        return (t / duration) ** self.alpha

    @property
    def attack(self):
        return self._ramp(self.a)

    @property
    def decay(self):
        # `d`-length reverse ramp, scaled and shifted to descend from 1 to `s`.
        return self._ramp(self.d)[::-1] * (1 - self.s) + self.s

    def sustain(self, duration):
        return np.full(round(int(duration * CONTROL_RATE)), fill_value=self.s)

    @property
    def release(self):
        # `r`-length reverse ramp, scaled and shifted to descend from `s` to 0.
        return self._ramp(self.r)[::-1] * self.s

    @property
    def note_on(self):
        return np.append(self.attack, self.decay)

    @property
    def note_off(self):
        return self.release

    def __str__(self):
        return (
            f"ADSR(a={self.a}, d={self.d}, s={self.s}, r={self.r}, alpha={self.alpha})"
        )


class VCO(SynthModule):
    """Voltage controlled oscillator.

    Think of this as a VCO on a modular synthesizer. It has a base pitch
    (specified here as a midi value), and a pitch modulation depth. Its call
    accepts a control-rate modulation signal between 0 - 1. An array of 0's
    returns a stationary audio signal at its base pitch.


    Parameters
    ----------

    midi_f0 (flt)       :       pitch value in 'midi' (69 = 440Hz).
    mod_depth (flt)     :       depth of the pitch modulation; 0 means none.

    TODO:   - more than just cosine.

    Examples
    --------

    >>> myVCO = VCO(midi_f0=69, mod_depth=24)
    >>> two_8ve_chirp = myVCO(np.linspace(0, 1, 1000, endpoint=False))
    """

    def __init__(self, midi_f0: float = 10, mod_depth: float = 50, phase: float = 0):
        super().__init__()

        assert 0 <= midi_f0 <= 127
        self.midi_f0 = midi_f0

        assert mod_depth >= 0
        self.mod_depth = mod_depth

        self.phase = phase

    def __call__(self, mod_signal: np.array, phase: float = 0) -> np.array:
        """Generates audio signal from control-rate mod.

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

        control_as_midi = self.mod_depth * mod_signal + self.midi_f0
        control_as_frequency = midi_to_hz(control_as_midi)
        cosine_argument = self.make_argument(control_as_frequency) + phase

        self.phase = cosine_argument[-1]
        return self.oscillator(cosine_argument)

    def make_argument(self, control_as_frequency: np.array):
        """
        Generates the phase argument to feed a cosine function to make audio.
        """

        up_sampled = self.control_to_sample_rate(control_as_frequency)
        return np.cumsum(2 * np.pi * up_sampled / SAMPLE_RATE)

    def oscillator(self, argument):
        """
        Dummy method. Overridden by child class VCO's.
        """
        pass


class SineVCO(VCO):
    """Simple VCO that generates a pitched sinudoid.

    Built off the VCO base class, it simply implements a cosine function as oscillator.
    """

    def __init__(self, midi_f0: float = 10, mod_depth: float = 50, phase: float = 0):
        super().__init__(midi_f0, mod_depth, phase)

    def oscillator(self, argument):
        return np.cos(argument)


class SquareSawVCO(VCO):
    """VCO that can be either a square or a sawtooth waveshape. Tweak with the shape parameter.

    With apologies to:

    Lazzarini, Victor, and Joseph Timoney. "New perspectives on distortion synthesis for
        virtual analog oscillators." Computer Music Journal 34, no. 1 (2010): 28-40.
    """

    def __init__(
        self,
        shape: float = 0,
        midi_f0: float = 10,
        mod_depth: float = 50,
        phase: float = 0,
    ):
        super().__init__(midi_f0, mod_depth, phase)
        assert 0 <= shape <= 1
        self.shape = shape

    def oscillator(self, argument):
        k = self.get_k()
        square = np.tanh(np.pi * k * np.sin(argument) / 2)
        return (1 - self.shape / 2) * square * (1 + self.shape * np.cos(argument))

    def get_k(self):
        f0 = midi_to_hz(self.midi_f0 + self.mod_depth)
        return 12000 / (f0 * np.log10(f0))


class VCA(SynthModule):
    """
    Voltage controlled amplifier.
    """

    def __init__(self):
        super().__init__()
        self.__envelope = np.array([])
        self.__audio = np.array([])

    def __call__(self, envelopecontrol: np.array, audiosample: np.array):
        envelopecontrol = np.clip(envelopecontrol, 0, 1)
        audiosample = np.clip(audiosample, -1, 1)
        amp = self.control_to_sample_rate(envelopecontrol)
        signal = fix_length(audiosample, len(amp))
        return amp * signal


class Drum:
    """
    A package of modules that makes one drum hit.
    """

    def __init__(
        self,
        pitch_adsr: ADSR = ADSR(),
        amp_adsr: ADSR = ADSR(),
        vco: VCO = VCO(),
        vca: VCA = VCA(),
        sustain_duration: float = 0,
    ):

        self.pitch_envelope = pitch_adsr(sustain_duration)
        self.amp_envelope = amp_adsr(sustain_duration)
        self.vco = vco
        self.vca = vca

    def __call__(self):
        self.pitch_envelope = fix_length(self.pitch_envelope, len(self.amp_envelope))
        vco_out = self.vco(self.pitch_envelope)
        return self.vca(self.amp_envelope, vco_out)


class SVF(SynthModule):
    """
    A State Variable Filter that can do low-pass, high-pass, band-pass, and
    band-reject filtering. Allows modulation of the cutoff frequency and an
    adjustable resonance parameter. Can self-oscillate to make a sinusoid oscillator.

    Parameters
    ----------

    mode (str)              :   filter type, one of LPF, HPF, BPF, or BSF
    cutoff (float)          :   cutoff frequency in Hz must be between 0 and half the
                                sample rate. Defaults to 1000Hz
    resonance (float)       :   filter resonance, or "Quality Factor". Higher values cause the filter to resonate more.
                                Must be greater than 0.5. Defaults to 0.707.
    self_oscillate (bool)   :   Set the filter into self-oscillation mode, which turns this into a sine wave oscillator
                                with the filter cutoff as the frequency. Defaults to False.
    sample_rate (float)     :   Processing sample rate.
    """

    def __init__(
        self,
        mode: str,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mode = mode

        self.cutoff = cutoff
        assert 0 <= self.cutoff < self.sample_rate / 2.0
        self.resonance = resonance
        assert 0.5 <= self.resonance
        self.self_oscillate = self_oscillate

    def __call__(
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
        cutoff_mod (np.ndarray)     :   Control signal used to modulate the filter cutoff. Values must be in range [0,1]
        cutoff_mod_amount (float)   :   How much to apply the control signal to the filter cutoff in Hz. Can be positive
                                        or negative. Defaults to 0.
        """

        h = np.zeros(2)
        y = np.zeros_like(audio)

        # Calculate initial coefficients
        coeff0, coeff1, rho = self.calculate_coefficients(self.cutoff)

        # Check if there is a filter cutoff envelope to apply
        if cutoff_mod_amount != 0.0:
            # Cutoff modulation must be same length as audio input
            assert len(cutoff_mod) == len(audio)

        # Processing loop
        for i in range(len(audio)):

            # If there is a cutoff modulation envelope, update coefficients
            if cutoff_mod_amount != 0.0:
                cutoff = self.cutoff + cutoff_mod[i] * cutoff_mod_amount
                coeff0, coeff1, rho = self.calculate_coefficients(
                    self.cutoff + cutoff_mod[i] * cutoff_mod_amount
                )

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
        R = 0.0 if self.self_oscillate else 1.0 / (2.0 * self.resonance)
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
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(mode="LPF", cutoff=cutoff, resonance=resonance,
                         self_oscillate=self_oscillate, sample_rate=sample_rate)


class HighPassSVF(SVF):
    """
    High-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(mode="HPF", cutoff=cutoff, resonance=resonance,
                         self_oscillate=self_oscillate, sample_rate=sample_rate)


class BandPassSVF(SVF):
    """
    Band-pass filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(mode="BPF", cutoff=cutoff, resonance=resonance,
                         self_oscillate=self_oscillate, sample_rate=sample_rate)


class BandRejectSVF(SVF):
    """
    Band-reject / band-stop filter using SVF architecture
    """

    def __init__(
        self,
        cutoff: float = 1000,
        resonance: float = 0.707,
        self_oscillate: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(mode="BSF", cutoff=cutoff, resonance=resonance,
                         self_oscillate=self_oscillate, sample_rate=sample_rate)


class FIR(SynthModule):
    """
    A Finite Impulse Response filter
    """

    def __init__(
        self,
        cutoff: float = 1000,
        filter_length: int = 512,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.cutoff = cutoff

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples
        TODO: Cutoff frequency modulation, if there is an efficient way to do it
        """

        impulse = self.windowed_sinc(self.cutoff, self.filter_length)
        y = np.convolve(audio, impulse)
        return y

    def windowed_sinc(self, cutoff: float, filter_length: float) -> np.ndarray:
        """
        Calculates the impulse response for FIR low-pass filter using the
        windowed sinc function method
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
        cos_a = np.cos(2 * np.pi * n / filter_length)
        cos_b = np.cos(4 * np.pi * n / filter_length)
        window = 0.42 - 0.5 * cos_a + 0.08 * cos_b
        ir *= window

        return ir


class MovingAverage(SynthModule):
    """
    A finite impulse response moving average filter.
    """

    def __init__(self, filter_length: int = 32, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.filter_length = filter_length

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Filter audio samples
        """

        impulse = np.ones(self.filter_length) / self.filter_length
        y = np.convolve(audio, impulse)
        return y
