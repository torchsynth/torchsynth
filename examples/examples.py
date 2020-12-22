# # ddsp-drum examples
#
# We walk through basic functionality of `ddsp-drum` in this Jupyter notebook.
#
# Just note that all ipd.Audio play widgets normalize the audio.

# +
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import os
from ddrum.drum_engine import ADSR, VCO, VCA
from scipy.signal import stft as stft
from scipy.io import wavfile as wavfile
import librosa.display
import IPython.display as ipd
# -

# Synthesis parameters.
a = 0.1
d = 0.1
s = 0.5
r = 0.5
alpha = 3
sustain_for = 0.25

f0 = 440

# TODO: Why don't we add 's' here?
dur = a + d + r + sustain_for

# Envelope test
myADSR = ADSR(a, d, s, r, alpha)
# TODO: Can you explain the envelope here?
env = myADSR(sustain_for)

t_cr = np.linspace(0, dur, int(dur * myADSR.get_rate()['control']), endpoint=False)
t_sr = np.linspace(0, dur, int(dur * myADSR.get_rate()['sample']), endpoint=False)

plt.plot(t_cr, env)
plt.title('a={}, d={}, s={}, r={}, alpha={}'.format(myADSR.get_a(),
                                                    myADSR.get_d(),
                                                    myADSR.get_s(),
                                                    myADSR.get_r(),
                                                    myADSR.get_alpha()))
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.show()

# VCO test
test_f0 = f0*(env + 1)
myVCO = VCO()
vco_out = myVCO(test_f0)

# +
X = librosa.stft(vco_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=myADSR.get_rate()['sample'], x_axis='time', y_axis='hz')
plt.ylim(0, 2000)
plt.show()

ipd.Audio(vco_out, rate=myADSR.get_rate()['sample'])
# -

# VCA test
myVCA = VCA()
vca_out = myVCA(env, vco_out)

# +
plt.plot(t_sr, vca_out)
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.show()

X = librosa.stft(vca_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=myADSR.get_rate()['sample'], x_axis='time', y_axis='hz')
plt.ylim(0, 2000)
plt.show()

# This is weird, why do we grab the sample rate from myADSR and not myVCA?
ipd.Audio(vca_out, rate=myADSR.get_rate()['sample'])
# -


