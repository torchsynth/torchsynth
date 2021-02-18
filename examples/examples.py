# # ddsp-drum examples
#
# We walk through basic functionality of `ddsp-drum` in this Jupyter notebook.
# Just note that all ipd.Audio play widgets normalize the audio.

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import IPython.display as ipd
from IPython.core.display import display
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from ddspdrum.defaults import SAMPLE_RATE
from ddspdrum import ADSR, Drum, FmVCO, NoiseModule, SineVCO, SquareSawVCO, VCA

# -


def time_plot(signal, sample_rate=SAMPLE_RATE, show=True):
    t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
    plt.plot(t, signal)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    if show:
        plt.show()


def stft_plot(signal, sample_rate=SAMPLE_RATE):
    X = librosa.stft(signal)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
    plt.show()


# ## The Envelope
# Our module is based on an ADSR envelope, standing for "attack, decay, sustain,
# release," which is specified by four values:
#
# - a: the attack time, in seconds; the time it takes for the signal to ramp
#      from 0 to 1.
# - d: the decay time, in seconds; the time to 'decay' from a peak of 1 to a
#      sustain level.
# - s: the sustain level; a value between 0 and 1 that the envelope holds during
# a sustained note (**not a time value**).
# - r: the release time, in seconds; the time it takes the signal to decay from
#      the sustain value to 0.
#
# Envelopes are used to modulate a variety of signals; usually one of pitch,
# amplitude, or filter cutoff frequency. In this notebook we will use the same
# envelope to modulate several different audio parameters.
#
# ### A note about note-on, note-off behaviour 
#
# By default, this envelope reacts as if it was triggered with midi, for example
# playing a keyboard. Each midi event has a beginning and end: note-on, when you
# press the key down; and note-off, when you release the key. `note_on_duration`
# is the amount of time that the key is depressed. During the note-on, the
# envelope moves through the attack and decay sections of the envelope. This
# leads to musically-intuitive, but programatically-counterintuitive behaviour.
#
# Assume attack is 0.5 seconds, and decay is 0.5 seconds. If a note is held for
# 0.75 seconds, the envelope won't traverse through the entire attack-and-decay
# phase (specifically, it will execute the entire attack, and 0.25 seconds of
# the decay).
#
# If this is confusing, don't worry about it. ADSR's do a lot of work behind the
# scenes to make the playing experience feel natural. Alternately, you may
# specify one-shot mode (see below), which is more typical of drum machines.

# +
# Synthesis parameters.
a = 0.1
d = 0.1
s = 0.75
r = 0.5
alpha = 3.0
note_on_duration = 0.5

# Envelope test
adsr = ADSR(a, d, s, r, alpha)
envelope = adsr.npyforward(note_on_duration)
time_plot(envelope, adsr.sample_rate)
# -

# ## Oscillators

# SineVCO test
midi_f0 = 12
sine_vco = SineVCO(midi_f0=midi_f0, mod_depth=50)
sine_out = sine_vco.npyforward(envelope, phase=0)
stft_plot(sine_out)
ipd.Audio(sine_out, rate=sine_vco.sample_rate)

# Check this out, it's a square / saw oscillator. Use the shape parameter to
# interpolate between a square wave (shape = 0) and a sawtooth wave (shape = 1).

# +
# SquareSawVCO test: shape 0 --> square, 1 --> saw.

shape = 0
midi_f0 = 24
sqs = SquareSawVCO(shape=shape, midi_f0=midi_f0, mod_depth=6)
sqs_out = sqs.npyforward(envelope, phase=0)
# -

stft_plot(sqs_out)
ipd.Audio(sqs_out, rate=sqs.sample_rate)

# Add noise with the NoiseModule class.

# +
# NoiseModule test.

noiser = NoiseModule(ratio=0.5)
noisey_out = noiser.npyforward(sqs_out)
# -

time_plot(noisey_out)
stft_plot(noisey_out)
ipd.Audio(noisey_out, rate=sqs.sample_rate)

# Notice that this sound is rather clicky. We'll add an envelope to the
# amplitude to smooth it out.

# VCA test
vca = VCA()
vca_out = vca.npyforward(envelope, noisey_out)

time_plot(vca_out)
stft_plot(vca_out)
ipd.Audio(vca_out, rate=vca.sample_rate)

# What about **FM synthesis**? You bet. Use the `FmVCO` class. It accepts any audio input.
#
# Just a note that, as in classic FM synthesis, you're dealing with a complex architecture of modulators. Each 'operator ' has its own pitch envelope, and amplitude envelope. The 'amplitude' envelope of an operator is really the *modulation depth* of the oscillator it operates on. So in the example below, we're using an ADSR to shape the depth of the *operator*, and this affects the modulation depth of the resultant signal.

# +
# FM Example.

# Create operator that doesn't modulate in pitch.
op_osc = SineVCO(midi_f0=50, mod_depth=0)
op_raw = op_osc.npyforward(envelope)

# Shape "amplitude" of modulator.
modulator = vca.npyforward(envelope, op_raw)
time_plot(modulator[:22050])

fm_vco = FmVCO(midi_f0=62, mod_depth=3)
fm_out = fm_vco.npyforward(modulator)
fm_out = vca.npyforward(envelope, fm_out)

stft_plot(fm_out[:22050])
time_plot(fm_out[:22050])
ipd.Audio(fm_out, rate=sine_vco.sample_rate)
# -

# Alternately, you can just use the Drum class that composes all these modules
# together automatically.

# +
my_drum = Drum(
    pitch_adsr=ADSR(0.25, 0.25, 0.25, 0.25, alpha=3),
    amp_adsr=ADSR(0.25, 0.25, 0.25, 0.25),
    vco_1=SineVCO(midi_f0=69, mod_depth=12),
    noise_module=NoiseModule(ratio=0.5),
    note_on_duration=1,
)

drum_out = my_drum.npyforward()

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
    note_on_duration=1,
)

drum_out = my_drum.npyforward()
stft_plot(drum_out)
ipd.Audio(drum_out, rate=vca.sample_rate)

# ### Parameters

# All synth modules and synth classes have named parameters which can be quered
# and updated. Let's look at the parameters for the Drum we just created. Each
# of these parameters shows the current value, minimum, maximum, and scale. The
# min and max refer to the smallest and largest values that parameter can take
# on. The scale value controls conversion between a range of 0 and 1. Let's look
# at that more below.

my_drum.modparameters

# Can also look at parameters by printing the object
# TODO: this looks a little messy. I know it gets tricky with nested repr. But... just saying.
print(my_drum)

# Setting a parameter with a range of [0,1]

my_drum.set_modparameter_0to1("pitch_attack", 0.25)
print(my_drum.modparameters['pitch_attack'])

drum_out = my_drum.npyforward()
stft_plot(drum_out)
ipd.Audio(drum_out, rate=vca.sample_rate)

# Setting a parameter with regular range

# +
my_drum.set_modparameter("amp_attack", 1.25)
print(my_drum.modparameters['amp_attack'])

# Get the value in the range 0 to 1
print("Value in 0 to 1 range: ", my_drum.get_modparameter_0to1('amp_attack'))
# -

drum_out = my_drum.npyforward()
stft_plot(drum_out)
ipd.Audio(drum_out, rate=vca.sample_rate)

# # Random synths
#
# Let's generate some random synths

drum = Drum(note_on_duration=1.0)
for i in range(10):
    drum.randomize()
    drum_out = drum.npyforward()
    stft_plot(drum_out)
    display(ipd.Audio(drum_out, rate=vca.sample_rate))

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
# Finite Impulse Response (FIR) lowpass with a cutoff frequency of 5000Hz.
# Filter length controls the slope of the cutoff. A longer filter will have a
# sharper cutoff.

# +
lpf = FIR(cutoff=5000, filter_length=1024)
filtered_fir = lpf.npyforward(noise)
ipd.display(ipd.Audio(filtered_fir, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_fir, Fs=sample_rate)
plt.show()
# -

# Now with a shorter filter of length 32

# +
lpf = FIR(cutoff=5000, filter_length=32)
filtered_fir = lpf.npyforward(noise)
ipd.display(ipd.Audio(filtered_fir, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_fir, Fs=sample_rate)
plt.show()
# -

# **FIR - Moving Average**

# +
ma_lpf = MovingAverage()
filtered_ma = ma_lpf.npyforward(noise)
ipd.display(ipd.Audio(filtered_ma, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_ma, Fs=sample_rate)
plt.show()
# -

# Moving average with a longer filter

# +
ma_lpf = MovingAverage(filter_length=64)
filtered_ma = ma_lpf.npyforward(noise)
ipd.display(ipd.Audio(filtered_ma, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_ma, Fs=sample_rate)
plt.show()
# -

# **IIR -State Variable Filter**
#
# State variable filter with the same cutoff -- the slope is much more relaxed
# with this filter

# +
svf = LowPassSVF(cutoff=5000)
filtered_svf = svf.npyforward(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# With an SVF we can resonate at the cutoff frequency

# +
svf = LowPassSVF(cutoff=5000, resonance=20.0)
filtered_svf = svf.npyforward(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a high-pass filter

# +
svf = HighPassSVF(cutoff=5000, resonance=20.0)
filtered_svf = svf.npyforward(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a band-pass filter

# +
svf = BandPassSVF(cutoff=5000, resonance=20.0)
filtered_svf = svf.npyforward(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a band-reject / band-stop filter (no resonance now)

# +
svf = BandRejectSVF(cutoff=5000)
filtered_svf = svf.npyforward(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# **Kick drum with SVF**
#
# With high resonance and an envelope applied to the cutoff frequency we can get
# something like a kick drum. To get the filter resonating we can use a short
# burst of noise.

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
kick = svf.npyforward(signal, cutoff_mod=cutoff_mod, cutoff_mod_amount=150)
plt.plot(kick)
ipd.Audio(kick, rate=sample_rate)

# ## Torch examples


import torch
from ddspdrum.torchmodule import TorchADSR

# Create a simple envelope

# +
# Synthesis parameters.
a = 0.1
d = 0.1
s = 0.75
r = 0.5
alpha = 3.0
note_on_duration = 0.5

# Envelope test
adsr = TorchADSR(a, d, s, r, alpha)
envelope = adsr(note_on_duration)
time_plot(envelope.detach(), adsr.sample_rate)
# -

# Create a second envelope, higher decay

# +
# Synthesis parameters.
a = 0.1
d = 0.5
s = 0.5
r = 0.5
alpha = 3.0
note_on_duration = 0.5

# Envelope test
adsr2 = TorchADSR(a, d, s, r, alpha)
envelope2 = adsr2(note_on_duration)
time_plot(envelope2.detach(), adsr.sample_rate)
# -

# Here's the l1 error

err = torch.mean(torch.abs(envelope - envelope2))
print("Error =", err)
plt.plot(torch.abs(envelope - envelope2).detach())

# And here are the gradients

err.backward(retain_graph=True)
for p in adsr.torchparameters:
    print(f"{p} grad1={adsr.torchparameters[p].grad.item()} grad2={adsr2.torchparameters[p].grad.item()}")

# +
#optimizer = torch.optim.SGD(list(adsr.parameters()) + list(adsr2.parameters()), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(list(adsr2.parameters()), lr=0.01)

for i in range(100):
    optimizer.zero_grad()
    #print(list(adsr.parameters()))
    #print(list(adsr2.parameters()))
    #print(note_on_duration)
    envelope = adsr(note_on_duration)
    envelope2 = adsr2(note_on_duration)
    
    if i % 10 == 0:
        time_plot(envelope.detach(), adsr.sample_rate, show=False)
        time_plot(envelope2.detach(), adsr.sample_rate, show=False)
        plt.show()
    
    #print(envelope.shape)
    #print(envelope2.shape)
    err = torch.mean(torch.abs(envelope - envelope2))
    print(err)
    err.backward()
    optimizer.step()

# -

# SineVCO vs SineVCO with higher midi_f0

# +
from ddspdrum.torchmodule import TorchSineVCO

# Reset envelope
adsr = TorchADSR(a, d, s, r, alpha)
envelope = adsr(note_on_duration)

# SineVCO test
sine_vco = TorchSineVCO(midi_f0=12.0, mod_depth=50.0)
sine_out = sine_vco(envelope, phase=0.0)
stft_plot(sine_out.detach().numpy())
ipd.Audio(sine_out.detach().numpy(), rate=sine_vco.sample_rate.item())
# -

# SineVCO test
midi_f0 = 12.0
sine_vco2 = TorchSineVCO(midi_f0=30.0, mod_depth=50.0)
sine_out2 = sine_vco2(envelope, phase=0.0)
stft_plot(sine_out2.detach().numpy())
ipd.Audio(sine_out2.detach().numpy(), rate=sine_vco2.sample_rate.item())

# We can use auraloss instead of raw waveform loss. This is just to show that gradient computations occur

err = torch.mean(torch.abs(sine_out - sine_out2))
print("Error =", err)
plt.plot(torch.abs(sine_out - sine_out2).detach())

err.backward(retain_graph=True)
for p in sine_vco.torchparameters:
    print(f"{p} grad1={sine_vco.torchparameters[p].grad.item()} grad2={sine_vco2.torchparameters[p].grad.item()}")
# Both SineVCOs use the sample envelope
for p in adsr.torchparameters:
    print(f"{p} grad={adsr.torchparameters[p].grad.item()}")

# Testing the `TorchVCA` class.

# +
from ddspdrum.torchmodule import TorchVCA

vca = TorchVCA()
test_output = vca(envelope, sine_out)

time_plot(test_output.detach().numpy())
# -

# Torch **FM synthesis.**

# +
from ddspdrum.torchmodule import TorchFmVCO

# FmVCO test
midi_f0 = 50.0

# Make steady-pitched sine (no pitch modulation).
sine_operator = TorchSineVCO(midi_f0=midi_f0, mod_depth=0.0)
operator_out = sine_operator(envelope)

# Shape the modulation depth.
operator_out = vca(envelope, operator_out)

# Feed into FM oscillator as modulator signal.
fm_vco = TorchFmVCO(midi_f0=midi_f0, mod_depth=5.0)
fm_out = fm_vco(operator_out)

stft_plot(fm_out.detach().numpy())
ipd.Audio(fm_out.detach().numpy(), rate=fm_vco.sample_rate.item())
# -

# ### Filters

# +
from ddspdrum.torchmodule import TorchMovingAverage, FIRLowPass

# Create some noise to filter
duration = 2
noise = torch.tensor(np.random.rand(int(44100 * duration)) * 2 - 1).float()
stft_plot(noise.detach().numpy())
# -

# **Moving Average Filter**
#
# A moving average filter is a simple finite impulse response (FIR) filter that calculates that value of a sample by taking the average of M input samples at a time. The filter_length defines how many samples M to include in the average.

# +
ma_filter = TorchMovingAverage(filter_length=32.)
filtered = ma_filter(noise)

stft_plot(filtered.detach().numpy())
ipd.Audio(filtered.detach().numpy(), rate=44100)

# +
# Second example with a longer filter -- notice that the filter length can be fractional
ma_filter2 = TorchMovingAverage(filter_length=64.25)
filtered2 = ma_filter2(noise)

stft_plot(filtered2.detach().numpy())
ipd.Audio(filtered2.detach().numpy(), rate=44100)
# -

# Compute the error between the two examples and get the gradient for the filter length

# +
import torch.fft
fft1 = torch.abs(torch.fft.fft(filtered))
fft2 = torch.abs(torch.fft.fft(filtered2))

err = torch.mean(torch.abs(fft1 - fft2))
print("Error =", err)

err.backward(retain_graph=True)
for p in ma_filter.torchparameters:
    print(f"{p} grad1={ma_filter.torchparameters[p].grad.item()} grad2={ma_filter2.torchparameters[p].grad.item()}")
# -

# Try to optimze the parameters of the second filter to match the first. **This isn't working**

# +
optimizer = torch.optim.Adam(list(ma_filter2.parameters()), lr=0.01)

print(list(ma_filter.parameters()))
print(list(ma_filter2.parameters()))

for i in range(100):
    optimizer.zero_grad()
    
    filtered1 = ma_filter(noise)
    filtered2 = ma_filter2(noise)
    
    fft1 = torch.abs(torch.fft.fft(filtered1))
    fft2 = torch.abs(torch.fft.fft(filtered2))
    
    err = torch.mean(torch.abs(fft1 - fft2))
    
    print(err)
    err.backward()
    optimizer.step()

print(list(ma_filter.parameters()))
print(list(ma_filter2.parameters()))
# -

# **FIR Lowpass**
#
# The TorchFIR filter implements a low-pass filter by approximating the impulse response of an ideal lowpass filter, which is a windowed sinc function in the time domain. We can set the exact cut-off frequency for this filter, all frequencies above this point are attenuated. The quality of the approximation is determined by the length of the filter, choosing a larger filter length will result in a filter with a steeper slope at the cutoff and more attenuation of high frequencies.

# +
fir1 = FIRLowPass(cutoff=1024, filter_length=128.0)
filtered1 = fir1(noise)

stft_plot(filtered1.detach().numpy())
ipd.Audio(filtered1.detach().numpy(), rate=44100)

# +
# Second filter with a lower cutoff and a longer filter
fir2 = FIRLowPass(cutoff=256., filter_length=1024)
filtered2 = fir2(noise)

stft_plot(filtered2.detach().numpy())
ipd.Audio(filtered2.detach().numpy(), rate=44100)
# -

# Compute the error between the two examples and check the gradient

# +
fft1 = torch.abs(torch.fft.fft(filtered1))
fft2 = torch.abs(torch.fft.fft(filtered2))
err = torch.mean(torch.abs(fft1 - fft2))
print("Error =", err)

err.backward(retain_graph=True)
for p in fir1.torchparameters:
    print(f"{p} grad1={fir1.torchparameters[p].grad.item()} grad2={fir2.torchparameters[p].grad.item()}")
# -
# Check that we can optimize towards the parameters of one of the filters using spectral loss

# +
optimizer = torch.optim.Adam(list(fir2.parameters()), lr=0.01)

print(list(fir1.parameters()))
print(list(fir2.parameters()))

for i in range(100):
    optimizer.zero_grad()

    filtered1 = fir1(noise)
    filtered2 = fir2(noise)
    
    fft1 = torch.abs(torch.fft.fft(filtered1))
    fft2 = torch.abs(torch.fft.fft(filtered2))
    
    err = torch.mean(torch.abs(fft1 - fft2))
        
    print(err)
    err.backward()
    optimizer.step()

print(list(fir1.parameters()))
print(list(fir2.parameters()))
# -



