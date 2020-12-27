# # ddsp-drum examples
#
# We walk through basic functionality of `ddsp-drum` in this Jupyter notebook.
#
# Just note that all ipd.Audio play widgets normalize the audio.

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import os

import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wavfile
from scipy.signal import stft as stft

from ddspdrum.module import ADSR, VCA, VCO

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
adsr = ADSR(a, d, s, r, alpha)
# TODO: Can you explain the envelope here?
env = adsr(sustain_for)

# Let's avoid cryptic variables names when possible
t_cr = np.linspace(0, dur, int(dur * adsr.control_rate), endpoint=False)
t_sr = np.linspace(0, dur, int(dur * adsr.sample_rate), endpoint=False)

plt.plot(t_cr, env)
plt.title(adsr)
plt.xlabel("time (sec)")
plt.ylabel("amplitude")
plt.show()

# VCO test
test_f0 = f0 * (env + 1)
vco = VCO()
vco_out = vco(test_f0)

# +
X = librosa.stft(vco_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=vco.sample_rate, x_axis="time", y_axis="hz")
plt.ylim(0, 2000)
plt.show()

ipd.Audio(vco_out, rate=vco.sample_rate)
# -

# VCA test
vca = VCA()
vca_out = vca(env, vco_out)

# +
plt.plot(t_sr, vca_out)
plt.xlabel("time (sec)")
plt.ylabel("amplitude")
plt.show()

X = librosa.stft(vca_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=vca.sample_rate, x_axis="time", y_axis="hz")
plt.ylim(0, 2000)
plt.show()

ipd.Audio(vca_out, rate=vca.sample_rate)
# -
