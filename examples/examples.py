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
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from ddspdrum.defaults import SAMPLE_RATE
from ddspdrum.module import ADSR, VCA, Drum, NoiseModule, SineVCO, SquareSawVCO

# -


def time_plot(signal, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def stft_plot(signal, sample_rate=SAMPLE_RATE):
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
    plt.show()


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
time_plot(envelope, adsr.control_rate)

# SineVCO test
midi_f0 = 12
sine_vco = SineVCO(midi_f0=midi_f0, mod_depth=50)
sine_out = sine_vco(envelope, phase=0)

stft_plot(sine_out)
ipd.Audio(sine_out, rate=sine_vco.sample_rate)

# Check this out, it's a square / saw oscillator. Use the shape parameter to interpolate between a square wave (shape = 0) and a sawtooth wave (shape = 1).

# +
# SquareSawVCO test: shape 0 --> square, 1 --> saw.

shape = 0
midi_f0 = 55
sqs = SquareSawVCO(shape=shape, midi_f0=midi_f0, mod_depth=6)
sqs_out = sqs(envelope, phase=0)
# -

stft_plot(sqs_out)
ipd.Audio(sqs_out, rate=sqs.sample_rate)

# Add noise with the NoiseModule class.

# +
# NoiseModule test.

noiser = NoiseModule(ratio=0.5)
noisey_out = noiser(sqs_out)
# -

time_plot(noisey_out)
stft_plot(noisey_out)
ipd.Audio(noisey_out, rate=sqs.sample_rate)

# Notice that this sound is rather clicky. We'll add an envelope to the amplitude to smooth it out.

# VCA test
vca = VCA()
vca_out = vca(envelope, noisey_out)

time_plot(vca_out)
stft_plot(vca_out)
ipd.Audio(vca_out, rate=vca.sample_rate)

# Alternately, you can just use the Drum class that composes all these modules together automatically.

# +
my_drum = Drum(
    pitch_adsr=ADSR(0.25, 0.25, 0.25, 0.25, alpha=3),
    amp_adsr=ADSR(0.25, 0.25, 0.25, 0.25),
    vco_1=SineVCO(midi_f0=69, mod_depth=12),
    noise_module=NoiseModule(ratio=0.5),
    sustain_duration=1,
)

drum_out = my_drum()

stft_plot(drum_out)

ipd.Audio(drum_out, rate=vca.sample_rate)
# -
# Additionally, the Drum class can take two oscillators.

my_drum = Drum(
    pitch_adsr=ADSR(0.25, 0.25, 0.25, 0.25, alpha=3),
    amp_adsr=ADSR(0.25, 0.25, 0.25, 0.25),
    vco_1=SquareSawVCO(shape=0, midi_f0=23.95, mod_depth=12),
    vco_2=SquareSawVCO(shape=0, midi_f0=24.05, mod_depth=12),
    noise_module=NoiseModule(ratio=0.1),
    sustain_duration=1,
)

drum_out = my_drum()
stft_plot(drum_out)
ipd.Audio(drum_out, rate=vca.sample_rate)

# ### Parameters
# All synth modules and synth classes have named parameters which can be quered and updated. Let's look at the parameters for the Drum we just created. Each of these parameters shows the current value, minimum, maximum, and scale. The min and max refer to the smallest and largest values that parameter can take on. The scale value controls conversion between a range of 0 and 1. Let's look at that more below.

my_drum.parameters

# Can also look at parameters by printing the object
print(my_drum)

# Setting a parameter with a range of [0,1]

my_drum.set_parameter_0to1("pitch_attack", 0.25)
print(my_drum.parameters['pitch_attack'])

drum_out = my_drum()
stft_plot(drum_out)
ipd.Audio(drum_out, rate=vca.sample_rate)

# Setting a parameter with regular range

# +
my_drum.set_parameter("amp_attack", 1.25)
print(my_drum.parameters['amp_attack'])

# Get the value in the range 0 to 1
print("Value in 0 to 1 range: ", my_drum.get_parameter_0to1('amp_attack'))
# -

drum_out = my_drum()
stft_plot(drum_out)
ipd.Audio(drum_out, rate=vca.sample_rate)

# # Filter Examples
#
# Example usage of three types of filters. Two finite impulse
# response (FIR) lowpasses and an infinite impulse response (IIR)
# state variable filter.

from ddspdrum.module import (
    FIR,
    BandPassSVF,
    BandRejectSVF,
    HighPassSVF,
    LowPassSVF,
    MovingAverage,
)

# Create some white noise to perform filtering on -- **Watch out it is loud!**

# +
sample_rate = 44100
duration = 2.0
noise = np.random.rand(int(sample_rate * duration)) * 2 - 1
ipd.display(ipd.Audio(noise, rate=sample_rate))

# Spectrogram
plt.specgram(noise, Fs=sample_rate)
plt.show()
# -

# **FIR - Windowed Sinc**
#
# Finite Impulse Response (FIR) lowpass with a cutoff frequency of 5000Hz. Filter length controls the slope of the cutoff. A longer filter will have a sharper cutoff.

# +
lpf = FIR(cutoff=5000, filter_length=1024)
filtered_fir = lpf(noise)
ipd.display(ipd.Audio(filtered_fir, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_fir, Fs=sample_rate)
plt.show()
# -

# Now with a shorter filter of length 32

# +
lpf = FIR(cutoff=5000, filter_length=32)
filtered_fir = lpf(noise)
ipd.display(ipd.Audio(filtered_fir, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_fir, Fs=sample_rate)
plt.show()
# -

# **FIR - Moving Average**

# +
ma_lpf = MovingAverage()
filtered_ma = ma_lpf(noise)
ipd.display(ipd.Audio(filtered_ma, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_ma, Fs=sample_rate)
plt.show()
# -

# Moving average with a longer filter

# +
ma_lpf = MovingAverage(filter_length=64)
filtered_ma = ma_lpf(noise)
ipd.display(ipd.Audio(filtered_ma, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_ma, Fs=sample_rate)
plt.show()
# -

# **IIR -State Variable Filter**
#
# State variable filter with the same cutoff -- the slope is much more relaxed with this filter

# +
svf = LowPassSVF(cutoff=5000)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# With an SVF we can resonate at the cutoff frequency

# +
svf = LowPassSVF(cutoff=5000, resonance=20.0)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a high-pass filter

# +
svf = HighPassSVF(cutoff=5000, resonance=20.0)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a band-pass filter

# +
svf = BandPassSVF(cutoff=5000, resonance=20.0)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a band-reject / band-stop filter (no resonance now)

# +
svf = BandRejectSVF(cutoff=5000)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# **Kick drum with SVF**
#
# With high resonance and an envelope applied to the cutoff frequency we can get something like a kick drum. To get the filter resonating we can use a short burst of noise.

duration = 1.0
signal = np.zeros(int(sample_rate * duration))
click = np.random.rand(int(sample_rate * 0.001)) * 2 - 1
signal[: len(click)] = click
plt.plot(signal)
ipd.Audio(signal, rate=sample_rate)

# Envelope to apply to the cutoff frequency
cutoff_mod = np.linspace(1, 0, len(signal)) ** 12.0
plt.plot(cutoff_mod)

# Apply filter and listen to results
svf = LowPassSVF(cutoff=45, resonance=50)
kick = svf(signal, cutoff_mod=cutoff_mod, cutoff_mod_amount=150)
plt.plot(kick)
ipd.Audio(kick, rate=sample_rate)






