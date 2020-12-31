# # ddsp-drum examples
#
# We walk through basic functionality of `ddsp-drum` in this Jupyter notebook. Just note that all ipd.Audio play widgets normalize the audio.
#
# In this example, we'll build a very basic kick sound.

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from ddspdrum.module import ADSR, VCA, VCO, SquareSaw, Drum

# -

# Synthesis parameters.
a = 0.01
d = 0.05
s = 0.75
r = 0.5
alpha = 3
sustain_duration = 0

dur = a + d + r + sustain_duration

# ## The Envelope
# Our module is based on an ADSR envelope, standing for "attack, decay, sustain, release," which is specified by four
# values:
#
# - a: the attack time, in seconds; the time it takes for the signal to ramp from 0 to 1.
# - d: the decay time, in seconds; the time to 'decay' from a peak of 1 to a sustain level.
# - s: the sustain level; a value between 0 and 1 that the envelope holds during a sustained note (**not a time value**).
# - r: the release time, in seconds; the time it takes the signal to decay from the sustain value to 0.
#
# Envelopes are used to modulate a variety of signals; usually one of pitch, amplitude, or filter cutoff frequency. In
# this notebook we will use the same envelope to modulate several different audio parameters.

# Envelope test
adsr = ADSR(a, d, s, r, alpha)
envelope = adsr(sustain_duration)

# Timelines for plots, in seconds based on control rate and sample rate, respectively.
t_cr = np.linspace(0, dur, int(dur * adsr.control_rate), endpoint=False)
t_sr = np.linspace(0, dur, int(dur * adsr.sample_rate), endpoint=False)

plt.plot(t_cr, envelope)
plt.title(adsr)
plt.xlabel("time (sec)")
plt.ylabel("amplitude")
plt.show()

# VCO test
midi_f0 = 12
vco = VCO(midi_f0=midi_f0, mod_depth=50)
vco_out = vco(envelope, phase=0)

# +
X = librosa.stft(vco_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=vco.sample_rate, x_axis="time", y_axis="hz")
plt.ylim(0, 2000)
plt.show()

ipd.Audio(vco_out, rate=vco.sample_rate)
# -

# Check this out, it's a square / saw oscillator. Use the shape parameter to interpolate between a square wave (shape = 0) and a sawtooth wave (shape = 1).

# +
# SquareSaw test: shape 0 --> square, 1 --> saw.

shape = 0

midi_f0 = 55
sqs = SquareSaw(shape=shape, midi_f0=midi_f0, mod_depth=6)
sqs_out = sqs(envelope, phase=0)

# +
X = librosa.stft(sqs_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=vco.sample_rate, x_axis="time", y_axis="hz")
plt.ylim(0, 5000)
plt.show()

ipd.Audio(sqs_out, rate=vco.sample_rate)
# -

# Notice that this sound is rather clicky. We'll add an envelope to the amplitude to smooth it out.

# VCA test
vca = VCA()
vca_out = vca(envelope, vco_out)

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

# Alternately, you can just use the Drum class that composes all these modules together automatically.

# +
my_drum = Drum(
    pitch_adsr=ADSR(0.25, 0.25, 0.25, 0.25, alpha=3),
    amp_adsr=ADSR(0.25, 0.25, 0.25, 0.25),
    vco=VCO(midi_f0=69, mod_depth=12),
    sustain_duration=1
    )

drum_out = my_drum()

X = librosa.stft(drum_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=vca.sample_rate, x_axis="time", y_axis="hz")
plt.ylim(0, 2000)
plt.show()

ipd.Audio(drum_out, rate=vca.sample_rate)
# -
# You can also use the **SquareSaw oscillator** in the drum module.

# +
my_drum = Drum(
    pitch_adsr=ADSR(0.25, 0.25, 0.25, 0.25, alpha=3),
    amp_adsr=ADSR(0.25, 0.25, 0.25, 0.25),
    vco=SquareSaw(shape=0, midi_f0=24, mod_depth=12),
    sustain_duration=1
    )

drum_out = my_drum()

X = librosa.stft(drum_out)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(5, 5))
librosa.display.specshow(Xdb, sr=vca.sample_rate, x_axis="time", y_axis="hz")
plt.ylim(0, 5000)
plt.show()

ipd.Audio(drum_out, rate=vca.sample_rate)
# -



