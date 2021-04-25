#!/usr/bin/env python3
# # torchsynth examples
#
# We walk through basic functionality of `torchsynth` in this Jupyter notebook.
#
# Just note that all ipd.Audio play widgets normalize the audio.
#
# If you're in Colab, remember to set the runtime to GPU.
# and get the latest torchsynth:
#
# ```
# # !pip install git+https://github.com/torchsynth/torchsynth.git
# ```

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
def iscolab():  # pragma: no cover
    return "google.colab" in str(get_ipython())


def isnotebook():  # pragma: no cover
    try:
        if iscolab():
            return True
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interprete


print(f"isnotebook = {isnotebook()}")

# +
if isnotebook():  # pragma: no cover
    import IPython.display as ipd
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    from IPython.core.display import display
else:

    class IPD:
        def Audio(*args, **kwargs):
            pass

        def display(*args, **kwargs):
            pass

    ipd = IPD()
import random

import numpy as np
import numpy.random
import torch.fft
import torch.tensor as tensor
from torch import Tensor as T

from torchsynth.config import SynthConfig
from torchsynth.module import (
    ADSR,
    VCA,
    ControlRateUpsample,
    MonophonicKeyboard,
    Noise,
    SineVCO,
    FmVCO,
)
from torchsynth.parameter import ModuleParameterRange

# Determenistic seeds for replicable testing
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)

# -


# Run examples on GPU if available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def time_plot(signal, sample_rate=44100, show=True):
    if isnotebook():  # pragma: no cover
        t = np.linspace(0, len(signal) / sample_rate, len(signal), endpoint=False)
        plt.plot(t, signal)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        if show:
            plt.show()


def stft_plot(signal, sample_rate=44100):
    if isnotebook():  # pragma: no cover
        X = librosa.stft(signal)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
        plt.show()


# ## Globals
# We'll generate 2 sounds at once, 4 seconds each
synthconfig = SynthConfig(
    batch_size=2, reproducible=False, sample_rate=44100, buffer_size_seconds=4.0
)

# For a few examples, we'll only generate one sound
synthconfig1 = SynthConfig(
    batch_size=1, reproducible=False, sample_rate=44100, buffer_size_seconds=4.0
)

# And a short one sound
synthconfig1short = SynthConfig(
    batch_size=1, reproducible=False, sample_rate=44100, buffer_size_seconds=0.1
)

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
a = tensor([0.1, 0.2])
d = tensor([0.1, 0.2])
s = tensor([0.75, 0.8])
r = tensor([0.5, 0.8])
alpha = tensor([3.0, 4.0])
note_on_duration = tensor([0.5, 1.5], device=device)

# Envelope test
adsr = ADSR(
    attack=a,
    decay=d,
    sustain=s,
    release=r,
    alpha=alpha,
    synthconfig=synthconfig,
    device=device,
)


envelope = adsr(note_on_duration)
time_plot(envelope.clone().detach().cpu().T, adsr.control_rate.item())
print(adsr)
# -

# Here's the l1 error between the two envelopes

err = torch.mean(torch.abs(envelope[0, :] - envelope[1, :]))
print("Error =", err)
time_plot(torch.abs(envelope[0, :] - envelope[1, :]).detach().cpu().T)

# ##### And here are the gradients

# +
# err.backward(retain_graph=True)
# for p in adsr.torchparameters:
#    print(adsr.torchparameters[p].data.grad)
#    print(f"{p} grad1={adsr.torchparameters[p].data.grad} grad2={adsr.torchparameters[p].data.grad}")
# -

# **Generating Random Envelopes**
#
# If we don't set parameters for an ADSR, then the parameters will be random when
# initialized.

# Note that module parameters are optional. If they are not provided,
# they will be randomly initialized (like a typical neural network module)
adsr = ADSR(synthconfig, device=device)
envelope = adsr(note_on_duration)
print(envelope.shape)
time_plot(envelope.clone().detach().cpu().T)

#
# We can also use an optimizer to match the parameters of the two ADSRs

# optimizer = torch.optim.Adam(list(adsr2.parameters()), lr=0.01)

# fig, ax = plt.subplots()
# time_plot(envelope.detach().cpu(), adsr.sample_rate, show=False)
# time_plot(envelope2.detach().cpu(), adsr.sample_rate, show=False)
# plt.show()

# for i in range(100):
#     optimizer.zero_grad()

#     envelope = adsr(note_on_duration)
#     envelope2 = adsr2(note_on_duration)
#     err = torch.mean(torch.abs(envelope - envelope2))

#     if i % 10 == 0:
#         ax.set_title(f"Optimization Step {i} - Error: {err.item()}")
#         ax.lines[0].set_ydata(envelope.detach().cpu())
#         ax.lines[1].set_ydata(envelope2.detach().cpu())
#         fig.canvas.draw()

#     err.backward()
#     optimizer.step()
# -


# ## Oscillators
#
# There are several types of oscillators and sound generators available. Oscillators
# that can be controlled by an external signal are called voltage-coltrolled
# oscillators (VCOs) in the analog world and we adpot a similar approach here;
# oscillators accept an input control signal and produce audio output. We have a simple
# sine oscilator:`SineVCO`, a square/saw oscillator: `SquareSawVCO`, and an FM
# oscillator: `FmVCO`. There is also a white noise generator: `Noise`.

# +
# %matplotlib inline

# Set up a Keyboard module
keyboard = MonophonicKeyboard(
    synthconfig, device, midi_f0=tensor([69.0, 50.0]), duration=note_on_duration
)

# Reset envelope
adsr = ADSR(
    attack=a,
    decay=d,
    sustain=s,
    release=r,
    alpha=alpha,
    synthconfig=synthconfig,
    device=device,
)

# Trigger the keyboard, which returns a midi_f0 and note duration
midi_f0, duration = keyboard()

# Create an envelope -- modulation signals are computed at a lower
# sampling rate and must be upsampled prior to feeding into audio
# rate modules
envelope = adsr(duration)
upsample = ControlRateUpsample(synthconfig)
envelope = upsample(envelope)

# SineVCO test -- call to(device) instead of passing in device to constructor also works
sine_vco = SineVCO(
    tuning=tensor([0.0, 0.0]),
    mod_depth=tensor([-12.0, 12.0]),
    synthconfig=synthconfig,
).to(device)


sine_out = sine_vco(midi_f0, envelope)

stft_plot(sine_out[0].detach().cpu().numpy())
ipd.display(
    ipd.Audio(sine_out[0].detach().cpu().numpy(), rate=sine_vco.sample_rate.item())
)
stft_plot(sine_out[1].detach().cpu().numpy())
ipd.display(
    ipd.Audio(sine_out[1].detach().cpu().numpy(), rate=sine_vco.sample_rate.item())
)

# We can use auraloss instead of raw waveform loss. This is just
# to show that gradient computations occur
err = torch.mean(torch.abs(sine_out[0] - sine_out[1]))
print("Error =", err)
time_plot(torch.abs(sine_out[0] - sine_out[1]).detach().cpu())

# +
# err.backward(retain_graph=True)
# for p in sine_vco.torchparameters:
#    print(f"{p} grad1={sine_vco.torchparameters[p].grad.item()} grad2={sine_vco2.torchparameters[p].grad.item()}")
## Both SineVCOs use the sample envelope
# for p in adsr.torchparameters:
#    print(f"{p} grad={adsr.torchparameters[p].grad.item()}")
# -

# ### SquareSaw Oscillator
#
# Check this out, it's a square / saw oscillator. Use the shape parameter to
# interpolate between a square wave (shape = 0) and a sawtooth wave (shape = 1).

# +
from torchsynth.module import SquareSawVCO

keyboard = MonophonicKeyboard(synthconfig, device, midi_f0=tensor([30.0, 30.0])).to(
    device
)

square_saw = SquareSawVCO(
    tuning=tensor([0.0, 0.0]),
    mod_depth=tensor([0.0, 0.0]),
    shape=tensor([0.0, 1.0]),
    synthconfig=synthconfig,
    device=device,
)
env2 = torch.zeros([2, square_saw.buffer_size], device=device)

square_saw_out = square_saw(keyboard.p("midi_f0"), env2)
stft_plot(square_saw_out[0].cpu().detach().numpy())
ipd.display(
    ipd.Audio(
        square_saw_out[0].cpu().detach().numpy(), rate=square_saw.sample_rate.item()
    )
)
stft_plot(square_saw_out[1].cpu().detach().numpy())
ipd.display(
    ipd.Audio(
        square_saw_out[1].cpu().detach().numpy(), rate=square_saw.sample_rate.item()
    )
)


err = torch.mean(torch.abs(square_saw_out[0] - square_saw_out[1]))
print(err)
# err.backward(retain_graph=True)
# for p in square_saw.torchparameters:
#    print(f"{p} grad1={square_saw.torchparameters[p][0].grad.item()} grad2={square_saw.torchparameters[p][1].grad.item()}")

# ### VCA
#
# Notice that this sound is rather clicky. We'll add an envelope to the
# amplitude to smooth it out.

# +
vca = VCA(synthconfig, device=device)
test_output = vca(envelope, sine_out)

time_plot(test_output[0].detach().cpu())
# -

# ### FM Synthesis
#
# What about FM? You bet. Use the `FmVCO` class. It accepts any audio input.
#
# Just a note that, as in classic FM synthesis, you're dealing with a complex
# architecture of modulators. Each 'operator ' has its own pitch envelope, and
# amplitude envelope. The 'amplitude' envelope of an operator is really the
# *modulation depth* of the oscillator it operates on. So in the example below,
# we're using an ADSR to shape the depth of the *operator*, and this affects
# the modulation depth of the resultant signal.

# +

# FmVCO test

keyboard = MonophonicKeyboard(
    synthconfig, device=device, midi_f0=tensor([50.0, 50.0])
).to(device)

# Make steady-pitched sine (no pitch modulation).
sine_operator = SineVCO(
    tuning=tensor([0.0, 0.0]),
    mod_depth=tensor([0.0, 5.0]),
    synthconfig=synthconfig,
    device=device,
)
operator_out = sine_operator(keyboard.p("midi_f0"), envelope)

# Shape the modulation depth.
operator_out = vca(envelope, operator_out)

# Feed into FM oscillator as modulator signal.
fm_vco = FmVCO(
    tuning=tensor([0.0, 0.0]),
    mod_depth=tensor([2.0, 5.0]),
    synthconfig=synthconfig,
    device=device,
)
fm_out = fm_vco(keyboard.p("midi_f0"), operator_out)

stft_plot(fm_out[0].cpu().detach().numpy())
ipd.display(ipd.Audio(fm_out[0].cpu().detach().numpy(), rate=fm_vco.sample_rate.item()))

stft_plot(fm_out[1].cpu().detach().numpy())
ipd.display(ipd.Audio(fm_out[1].cpu().detach().numpy(), rate=fm_vco.sample_rate.item()))
# -

# ### Noise
#
# The noise generator creates white noise the same length as the SynthModule buffer length

# +
noise = Noise(synthconfig, seed=42, device=device)
out = noise()

stft_plot(out[0].detach().cpu().numpy())
ipd.Audio(out[0].detach().cpu().numpy(), rate=noise.sample_rate.item())
# -

# ## Audio Mixer

# +
from torchsynth.module import AudioMixer

env = torch.zeros((synthconfig.batch_size, synthconfig.buffer_size), device=device)

keyboard = MonophonicKeyboard(synthconfig, device=device)
sine = SineVCO(synthconfig, device=device)
square_saw = SquareSawVCO(synthconfig, device=device)
noise = Noise(synthconfig, seed=123, device=device)

midi_f0, note_on_duration = keyboard()
sine_out = sine(midi_f0, env)
sqr_out = square_saw(midi_f0, env)
noise_out = noise()

mixer = AudioMixer(synthconfig, 3, curves=[1.0, 1.0, 0.25]).to(device)
output = mixer(sine_out, sqr_out, noise_out)

ipd.Audio(out[0].cpu().detach().numpy(), rate=mixer.sample_rate.item(), normalize=False)

# +
# Mixer params are set in dB
mixer.set_parameter("level0", tensor([0.25, 0.25], device=device))
mixer.set_parameter("level1", tensor([0.25, 0.25], device=device))
mixer.set_parameter("level2", tensor([0.125, 0.125], device=device))

out = mixer(sine_out, sqr_out, noise_out)
ipd.Audio(out[0].cpu().detach().numpy(), rate=mixer.sample_rate.item())
# -

# ## Modulation
#
# Besides envelopes, LFOs can be used to modulate parameters

# +
from torchsynth.module import LFO, ModulationMixer

adsr = ADSR(synthconfig=synthconfig, device=device)

# Trigger the keyboard, which returns a midi_f0 and note duration
midi_f0, duration = keyboard()

envelope = adsr(duration)

lfo = LFO(synthconfig, device=device)
lfo.set_parameter("mod_depth", tensor([10.0, 0.0]))
lfo.set_parameter("frequency", tensor([1.0, 1.0]))
out = lfo(envelope)

lfo2 = LFO(synthconfig, device=device)
out2 = lfo2(envelope)

print(out.shape)

time_plot(out[0].detach().cpu().numpy(), sample_rate=lfo.control_rate.item())
time_plot(out2[0].detach().cpu().numpy(), sample_rate=lfo.control_rate.item())

# A modulation mixer can be used to mix a modulation sources together
# and maintain a 0 to 1 amplitude range
mixer = ModulationMixer(synthconfig=synthconfig, device=device, n_input=2, n_output=1)
mods_mixed = mixer(out, out2)

print(f"Mixed: LFO 1:{mixer.p('0->0')[0]:.2}, LFO 2: {mixer.p('1->0')[0]:.2}")
time_plot(mods_mixed[0][0].detach().cpu().numpy(), sample_rate=lfo.control_rate.item())
# -

# ## Voice Module
#
# Alternately, you can just use the Voice class that composes all these modules
# together automatically.

from torchsynth.synth import Voice

voice1 = Voice(synthconfig=synthconfig1).to(device)
voice1.set_parameters(
    {
        ("keyboard", "midi_f0"): tensor([69.0]),
        ("keyboard", "duration"): tensor([1.0]),
        ("vco_1", "tuning"): tensor([0.0]),
        ("vco_1", "mod_depth"): tensor([12.0]),
    }
)

voice_out1 = voice1()
stft_plot(voice_out1.cpu().view(-1).detach().numpy())
ipd.Audio(voice_out1.cpu().detach().numpy(), rate=voice1.sample_rate.item())


# Additionally, the Voice class can take two oscillators.


# +
voice2 = Voice(synthconfig=synthconfig1).to(device)
voice2.set_parameters(
    {
        ("keyboard", "midi_f0"): tensor([40.0]),
        ("keyboard", "duration"): tensor([3.0]),
        ("vco_1", "tuning"): tensor([19.0]),
        ("vco_1", "mod_depth"): tensor([24.0]),
        ("vco_2", "tuning"): tensor([0.0]),
        ("vco_2", "mod_depth"): tensor([12.0]),
        ("vco_2", "shape"): tensor([1.0]),
    }
)

voice_out2 = voice2()
stft_plot(voice_out2.cpu().view(-1).detach().numpy())
ipd.Audio(voice_out2.cpu().detach().numpy(), rate=voice2.sample_rate.item())
# -


# Test gradients on entire voice

err = torch.mean(torch.abs(voice_out1 - voice_out2))
print(err)

# ## Random synths
#
# Let's generate some random synths in batch

synthconfig16 = SynthConfig(
    batch_size=16, reproducible=False, sample_rate=44100, buffer_size_seconds=4
)
voice = Voice(synthconfig=synthconfig16).to(device)
voice_out = voice()
for i in range(synthconfig16.batch_size):
    stft_plot(voice_out[i].cpu().view(-1).detach().numpy())
    ipd.display(
        ipd.Audio(voice_out[i].cpu().detach().numpy(), rate=voice.sample_rate.item())
    )

# Parameters can be set and frozen before randomization as well

# +
voice.unfreeze_all_parameters()
voice.set_frozen_parameters(
    {
        ("keyboard", "midi_f0"): 42.0,
        ("keyboard", "duration"): 3.0,
        ("vco_1", "tuning"): 0.0,
        ("vco_2", "tuning"): 0.0,
    },
)

voice_out = voice()
for i in range(synthconfig16.batch_size):
    stft_plot(voice_out[i].cpu().view(-1).detach().numpy())
    ipd.display(
        ipd.Audio(voice_out[i].cpu().detach().numpy(), rate=voice.sample_rate.item())
    )

# +
# ### Parameters

# All synth modules and synth classes have named parameters which can be quered
# and updated. Let's look at the parameters for the Voice we just created.
for n, p in voice1.named_parameters():
    print(f"{n:40}")

# Parameters are passed into SynthModules during creation with an initial value and a parameter range. The parameter range is a human readable range of values, for example MIDI note numbers from 1-127 for a VCO. These values are stored in a normalized range between 0 and 1. Parameters can be accessed and set using either ranges with specific methods.
#
# Parameters of individual modules can be accessed in several ways:

# Get the full ModuleParameter object by name from the module
print(voice1.vco_1.get_parameter("tuning"))

# Access the value as a Tensor in the full value human range
print(voice1.vco_1.p("tuning"))

# Access the value as a float in the range from 0 to 1
print(voice1.vco_1.get_parameter_0to1("tuning"))

# Parameters of individual modules can also be set using the human range or a normalized range between 0 and 1

# Set the vco pitch using the human range, which is MIDI note number
voice1.vco_1.set_parameter("tuning", tensor([12.0]))
print(voice1.vco_1.p("tuning"))

# Set the vco pitch using a normalized range between 0 and 1
voice1.vco_1.set_parameter_0to1("tuning", tensor([0.5]))
print(voice1.vco_1.p("tuning"))

# #### Parameter Ranges
#
# Conversion between [0,1] range and a human range is handled by `ModuleParameterRange`. The conversion from [0,1] can be shaped by specifying a curve. Curve values less than 1 put more emphasis on lower values in the human range and curve values greater than 1 put more emphasis on larger values in the human range. A curve of 1 is a linear relationship between the two ranges.

# +
# ModuleParameterRange with scaling of a range from 0-127
param_range_exp = ModuleParameterRange(0.0, 127.0, curve=0.5)
param_range_lin = ModuleParameterRange(0.0, 127.0, curve=1.0)
param_range_log = ModuleParameterRange(0.0, 127.0, curve=2.0)

# Linearly spaced values from 0.0 1.0
param_values = torch.linspace(0.0, 1.0, 100)

if isnotebook():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    axes[0].plot(param_values, param_range_exp.from_0to1(param_values))
    axes[0].set_title("Exponential Scaling")

    axes[1].plot(param_values, param_range_lin.from_0to1(param_values))
    axes[1].set_title("Linear Scaling")

    axes[2].plot(param_values, param_range_log.from_0to1(param_values))
    axes[2].set_title("Logarithmic Scaling")
# +
# ModuleParameterRange with symmetric scaling of a range from -127 to 127
param_range_exp = ModuleParameterRange(-127.0, 127.0, curve=0.5, symmetric=True)
param_range_log = ModuleParameterRange(-127.0, 127.0, curve=2.0, symmetric=True)

# Linearly spaced values from 0.0 1.0
param_values = torch.linspace(0.0, 1.0, 100)

if isnotebook():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].plot(param_values, param_range_exp.from_0to1(param_values))
    axes[0].set_title("Exponential Scaling")

    axes[1].plot(param_values, param_range_log.from_0to1(param_values))
    axes[1].set_title("Logarithmic Scaling")
# -

# ### Hyperparameters
#
# ParameterRanges are considered hyperparameters in torchsynth and can be viewed and modified through a Synth

# View all hyperparameters
voice1.hyperparameters

# Set a specific hyperparameter
voice1.set_hyperparameter(("keyboard", "midi_f0", "curve"), 0.1)
print(voice1.hyperparameters[("keyboard", "midi_f0", "curve")])

# ### Nebulae
#
# Different hyperparameter settings cause the parameters of a synth to be sampled in a
# different way when generating random synths. We call these different versions of the
# same synth different nebula. For example, here is the Voice loaded with the Drum
# nebula, which is more likely to produce sounds that are similar synth drum hits
# during random sampling.

synthconfig16 = SynthConfig(
    batch_size=16, reproducible=False, sample_rate=44100, buffer_size_seconds=4
)
voice = Voice(synthconfig=synthconfig16, nebula="drum").to(device)
voice_out = voice()
for i in range(synthconfig16.batch_size):
    stft_plot(voice_out[i].cpu().view(-1).detach().numpy())
    ipd.display(
        ipd.Audio(voice_out[i].cpu().detach().numpy(), rate=voice.sample_rate.item())
    )
