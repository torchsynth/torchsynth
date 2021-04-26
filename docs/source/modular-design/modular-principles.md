Modular Principles
==================

```{contents}
:depth: 2
```

## Synth Modules

The design of torchsynth is inspired by hardware modular synthesizers
which contain individual units. Each module has a specific function
and parameters, and they can be connected together in various
configurations to construct a synthesizer. There are three types
of modules in torchsynth: audio modules, control modules, and
parameter modules. Audio modules operate at audio sampling rate
(default 44.1kHz) and output audio {class}`~torchsynth.signal.Signal`.
Examples include voltage-controlled oscillators
({class}`~torchsynth.module.VCO`)
and voltage-controlled amplifiers
({class}`~torchsynth.module.VCA`s).  Control modules output control
signals that are used modulate the parameters of another module.
For speed, these modules operate at a reduced control rate (default
441Hz). Examples of control modules include
{class}`~torchsynth.module.ADSR` envelope generators and low frequency
oscillators ({class}`~torchsynth.module.LFO`s).  Finally, parameter
modules simply output parameters. Examples of these include a
{class}`~torchsynth.module.MonophonicKeyboard` module that has no
input, and outputs the note midi f0 value and duration.

## Synth Architectures

The default configuration in torchsynth is the
{class}`~torchsynth.synth.Voice`, which is the architecture used
in synth1B1. The {class}`~torchsynth.synth.Voice` architecture
comprises the following modules: a
{class}`~torchsynth.module.MonophonicKeyboard`, two
{class}`~torchsynth.module.LFO`, six {class}`~torchsynth.module.ADSR`
envelopes (each {class}`~torchsynth.module.LFO` module includes two
dedicated {class}`~torchsynth.module.ADSR`: one for rate modulation
and another for amplitude modulation), one
{class}`~torchsynth.module.SineVCO`, one
{class}`~torchsynth.module.SquareSawVCO`, one
{class}`~torchsynth.module.Noise` generator,
{class}`~torchsynth.module.VCA`, a
{class}`~torchsynth.module.ModulationMixer` and an
{class}`~torchsynth.module.AudioMixer`. Modulation signals generated
from control modules ({class}`~torchsynth.module.ADSR` and
{class}`~torchsynth.module.LFO`) are upsampled to the audio sample
rate before being passed to audio rate modules.

The figure below shows the configuration and
routing of the modules composing {class}`~torchsynth.synth.Voice`.

<img width="350px" src="../_static/images/Voice-diagram.svg">
