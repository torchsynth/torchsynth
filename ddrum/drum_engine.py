import numpy as np
from scipy.signal import resample as resample
from parameters import SAMPLE_RATE, CONTROL_RATE


class ADSR:
    """
    Envelope class for building a control rate ADSR signal. Use play() to output envelope.

    Parameters
    ----------

    a (flt)     :   attack time (sec)
    d (flt)     :   decay time (sec)
    s (flt)     :   sustain value between 0-1
    r (flt)     :   release time (sec)
    alpha (flt) :   envelope curve. 1 is linear envelope, >1 is exponential.

    Examples
    --------

    >>> myADSR = ADSR(0.5, 0.5, 0.5, 0.5, 3)
    >>> print(myADSR(1))

    """

    def __init__(self, a=0.25, d=0.25, s=0.5, r=0.5, alpha=3):
        self.__a = 0
        self.__d = 0
        self.__s = 0
        self.__r = 0
        self.__alpha = 0

        self._attack = np.array([])
        self._decay = np.array([])
        self._sustain = 0
        self._release = np.array([])

        self.set_alpha(alpha)
        self.set_s(s)
        self.set_a(a)
        self.set_d(d)
        self.set_r(r)

    def __call__(self, dur):
        return self.play(dur)

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

    def __ramp(self, value):
        t = np.linspace(0, value, int(value*CONTROL_RATE), endpoint=False)
        return (t/value)**self.__alpha

    def note_on(self):
        return np.append(self._attack, self._decay)

    def note_off(self):
        return self._release

    def play(self, dur):
        return np.concatenate((self.note_on(), np.full(int(dur*CONTROL_RATE), self.get_s()), self.note_off()))


class VCO:
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

    @staticmethod
    def to_arg(f0):
        out_length = len(f0) * SAMPLE_RATE / CONTROL_RATE
        up_sampled = resample(f0, int(out_length))
        return np.cumsum(2 * np.pi * up_sampled / SAMPLE_RATE)