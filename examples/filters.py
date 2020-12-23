# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Filter Examples
#
# Example usage of three types of filters. Two finite impulse response (FIR) lowpasses and an infinite impulse response (IIR) state variable filter.

import numpy as np
import matplotlib.pyplot as plt
from ddrum.drum_engine import SVF, FIR, MovingAverage
import ddrum.dsp_utils as utils
import IPython.display as ipd

# Create some white noise to perform filtering on

# +
sample_rate = 44100
duration = 2.0
noise = np.random.rand(int(sample_rate * duration)) * 2 -1
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
svf = SVF(mode='LPF', cutoff=5000)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# With an SVF we can resonate at the cutoff frequency

# +
svf = SVF(mode='LPF', cutoff=5000, resonance=20.0)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a high-pass filter

# +
svf = SVF(mode='HPF', cutoff=5000, resonance=20.0)
filtered_svf = svf(noise)
ipd.display(ipd.Audio(filtered_svf, rate=sample_rate))

# Plot Spectrogram
plt.specgram(filtered_svf, Fs=sample_rate)
plt.show()
# -

# SVF as a band-pass filter

# +
svf = SVF(mode='BPF', cutoff=5000, resonance=20.0)
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
click = np.random.rand(int(sample_rate * 0.001)) * 2 -1
signal[:len(click)] = click
plt.plot(signal)
ipd.Audio(signal, rate=sample_rate)

# Envelope to apply to the cutoff frequency
cutoff_mod = np.linspace(1, 0, len(signal)) ** 12.0
plt.plot(cutoff_mod)

# Apply filter and listen to results
svf = SVF(mode='LPF', cutoff=45, resonance=50)
kick = svf(signal, cutoff_mod=cutoff_mod, cutoff_mod_amount=150)
plt.plot(kick)
ipd.Audio(kick, rate=sample_rate)

# + pycharm={"name": "#%%\n"}

# -


