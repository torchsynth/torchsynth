"""
    TODO    :   - VCA is slow. Blocks? Not sure what's best for tensors.
                - Convert operations to tensors, obvs.
"""

import numpy as np
from scipy.signal import resample as resample
from ddrum.parameters import SAMPLE_RATE, CONTROL_RATE


class SynthModule:
    """
    Base class for synthesis modules. Mostly helper functions for the moment.

    """
    def __init__(self):
        pass

    @staticmethod
    def get_rate():
        return {'control': CONTROL_RATE, 'sample': SAMPLE_RATE}

    @staticmethod
    def to_sample_rate(x):
        out_length = len(x) * SAMPLE_RATE / CONTROL_RATE
        return resample(x, int(out_length))

    @staticmethod
    def fix_length(x, length):
        if len(x) < length:
            x = np.pad(x, [0, length - len(x)])
        elif len(x) > length:
            x = x[:length]
        return x


class ADSR(SynthModule):
    """
    Envelope class for building a control rate ADSR signal. Use play() to output envelope.

    Parameters
    ----------

    a (flt)             :   attack time (sec)
    d (flt)             :   decay time (sec)
    s (flt)             :   sustain value between 0-1
    r (flt)             :   release time (sec)
    alpha (flt)         :   envelope curve. 1 is linear envelope, >1 is exponential
    sustain_for (flt)   :   sustain length (sec)

    Examples
    --------

    >>> myADSR = ADSR(0.5, 0.5, 0.5, 0.5, 3)
    >>> print(myADSR(1))

    """

    def __init__(self, a=0.25, d=0.25, s=0.5, r=0.5, alpha=3, sustain_for=0.5):
        SynthModule.__init__(self)
        self.__a = 0
        self.__d = 0
        self.__s = 0
        self.__r = 0
        self.__alpha = 0
        self.__sustain_for = 0

        self._attack = np.array([])
        self._decay = np.array([])
        self._sustain = 0
        self._release = np.array([])

        self.set_alpha(alpha)
        self.set_s(s)
        self.set_a(a)
        self.set_d(d)
        self.set_r(r)
        self.set_sustain_for(sustain_for)

    def __call__(self, dur=0):
        self.set_sustain_for(dur)
        return self.play()

    def get_a(self):
        return self.__a

    def set_a(self, a):
        if a < 0:
            self.__a = 0
        else:
            self.__a = a

        self._attack = self.__ramp(self.__a)

    def get_d(self):
        return self.__d

    def set_d(self, d):
        if d < 0:
            self.__d = 0
        elif d > 1:
            self.__d = 1
        else:
            self.__d = d

        self._decay = self.__ramp(self.__d)[::-1] * (1 - self.get_s()) + self.get_s()

    def get_s(self):
        return self.__s

    def set_s(self, s):
        if s < 0:
            self.__s = 0
        elif s > 1:
            self.__s = 1
        else:
            self.__s = s

    def get_r(self):
        return self.__r

    def set_r(self, r):
        if r < 0:
            self.__r = 0
        else:
            self.__r = r

        self._release = self.__ramp(self.__r)[::-1] * self.get_s()

    def get_alpha(self):
        return self.__alpha

    def set_alpha(self, alpha):
        if alpha < 0:
            self.__alpha = 0
        else:
            self.__alpha = alpha

    def get_sustain_for(self):
        return self.__sustain_for

    def set_sustain_for(self, val):
        if val < 0:
            self.__sustain_for = 0
        else:
            self.__sustain_for = val

    def __ramp(self, value):
        t = np.linspace(0, value, int(value*CONTROL_RATE), endpoint=False)
        return (t/value)**self.__alpha

    def note_on(self):
        return np.append(self._attack, self._decay)

    def note_off(self):
        return self._release

    def play(self):
        return np.concatenate((self.note_on(),
                               np.full(int(self.get_sustain_for()*CONTROL_RATE), self.get_s()),
                               self.note_off()))


class VCO(SynthModule):
    """
    Voltage controlled oscillator. Accepts control rate instantaneous frequency, outputs audio.

    TODO: more than just cosine.

    Parameters
    ---------

    None.

    Examples
    --------

    >>> myVCO = VCO()
    >>> two_8ve_chirp = myVCO(np.linspace(440, 1760, 1000))

    """

    def __init__(self):
        SynthModule.__init__(self)
        self.__f0 = []
        self.__phase = 0

    def __call__(self, f0):
        self.set_f0(f0)
        return self.play()

    def get_f0(self):
        return self.__f0

    def set_f0(self, val):
        nyq = SAMPLE_RATE // 2
        val[np.where(val < 0)] = 0
        val[np.where(val > nyq)] = nyq
        self.__f0 = val

    def get_phase(self):
        return self.__phase

    def set_phase(self, val):
        self.__phase = val % (2 * np.pi)

    def play(self):
        arg = self.to_arg(self.get_f0()) + self.get_phase()
        self.set_phase(arg[-1])
        return np.cos(arg)

    def to_arg(self, f0):
        up_sampled = self.to_sample_rate(f0)
        return np.cumsum(2 * np.pi * up_sampled / SAMPLE_RATE)


class VCA(SynthModule):
    """
    Voltage controlled amplifier. Shapes amplitude of audio rate signal with control rate level.

    """

    def __init__(self):
        SynthModule.__init__(self)
        self.__envelope = np.array([])
        self.__audio = np.array([])

    def __call__(self, envelope, audio):
        self.set_envelope(envelope)
        self.set_audio(audio)
        return self.play()

    def get_envelope(self):
        return self.__envelope

    def set_envelope(self, envelope):
        envelope = np.clip(envelope, 0, 1)
        self.__envelope = envelope

    def get_audio(self):
        return self.__audio

    def set_audio(self, audio):
        audio = np.clip(audio, -1, 1)
        self.__audio = audio

    def play(self):
        amp = self.to_sample_rate(self.get_envelope())
        signal = self.fix_length(self.get_audio(), len(amp))
        return amp * signal


class SVF(SynthModule):
    """
    A State Variable Filter

    Recursive filter structure for low-pass
    """

    def __init__(self, mode: str = 'LPF', cutoff: float = 1000, resonance: float = 0.707,
                 self_oscillate: bool = False, sample_rate: int = SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.mode = mode
        self.cutoff = cutoff
        self.resonance = resonance
        self.self_oscillate = self_oscillate

    def __call__(self, audio: np.ndarray, cutoff_mod: np.ndarray = None,
                 cutoff_mod_amount: float = 0.0) -> np.ndarray:
        """
        Process audio samples
        """

        h = np.zeros(2)
        y = np.zeros_like(audio)

        # Calculate initial coefficients
        coeff0, coeff1, rho = self.calculate_coefficients(self.cutoff)

        # Check if there is a filter cutoff envelope to apply
        apply_modulation = False
        if cutoff_mod is not None and cutoff_mod_amount != 0.0:
            # Cutoff modulation must be same length as audio input
            assert len(cutoff_mod) == len(audio)
            apply_modulation = True

        # Processing loop
        for i in range(len(audio)):

            # If there is a cutoff modulation envelope, update coefficients
            if apply_modulation:
                cutoff = self.cutoff + cutoff_mod[i] * cutoff_mod_amount
                coeff0, coeff1, rho = self.calculate_coefficients(self.cutoff + cutoff_mod[i] * cutoff_mod_amount)

            # Calculate each of the filter components
            hpf = coeff0 * (audio[i] - rho * h[0] - h[1])
            bpf = coeff1 * hpf + h[0]
            lpf = coeff1 * bpf + h[1]

            # Feedback samples
            h[0] = coeff1 * hpf + bpf
            h[1] = coeff1 * bpf + lpf

            if self.mode == 'LPF':
                y[i] = lpf
            elif self.mode == 'BPF':
                y[i] = bpf
            elif self.mode == 'BSF':
                y[i] = hpf + lpf
            else:
                y[i] = hpf

        return y

    def calculate_coefficients(self, cutoff: float) -> (float, float, float):
        """
        Calculates the filter coefficients for SVF given a cutoff frequency
        """

        g = np.tan(np.pi * cutoff / self.sample_rate)
        R = 0.0 if self.self_oscillate else 1.0 / (2.0 * self.resonance)
        coeff0 = 1.0 / (1.0 + 2.0 * R * g + g * g)
        coeff1 = g
        rho = 2.0 * R + g

        return coeff0, coeff1, rho


class FIR(SynthModule):
    """
    A Finite Impulse Response filter
    """

    def __init__(self, cutoff: float = 1000, filter_length: int = 512, sample_rate: int = SAMPLE_RATE):
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
        Calculates the impulse response for FIR lowpass filter using the
        windowed sinc function method
        """

        ir = np.zeros(filter_length + 1)
        omega = 2 * np.pi * cutoff / self.sample_rate

        for i in range(filter_length + 1):
            n = (i - filter_length / 2)
            if n != 0:
                ir[i] = np.sin(n * omega) / n
            else:
                ir[i] = omega

            window = 0.42 - 0.5 * np.cos(2 * np.pi * i / filter_length) + 0.08 * np.cos(2 * np.pi * i / filter_length)
            ir[i] *= window

        ir /= omega

        return ir


class MovingAverage(SynthModule):
    """
    A finite impulse response moving average filter
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
