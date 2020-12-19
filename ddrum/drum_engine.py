import numpy as np
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
        t = np.linspace(0, value, value*CONTROL_RATE, endpoint=False)
        return (t/value)**self.__alpha

    def note_on(self):
        return np.append(self._attack, self._decay)

    def note_off(self):
        return self._release

    def play(self, dur):
        return np.concatenate((self.note_on(), np.full(dur*CONTROL_RATE, self.get_s()), self.note_off()))
