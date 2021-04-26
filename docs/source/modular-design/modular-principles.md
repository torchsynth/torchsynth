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
(default 44.1kHz) and output audio signals. Examples include
voltage-controlled oscillators (VCOs) and voltage-controlled
amplifiers (VCAs). Control modules output control signals that are
used modulate the parameters of another module. For speed, these
modules operate at a reduced control rate (default 44.1KHz). Examples
of control modules include ADSR envelope generators and low frequency
oscillators (LFOs). Finally, parameter modules simply output
parameters. Examples of these include a monophonic "keyboard"
module that has no input, and outputs the note midi f0 value and
duration.

## Synth Architectures

The default configuration in torchsynth is the Voice, which is the
architecture used in synth1B1. The Voice architecture comprises the
following modules: a Monophonic Keyboard, two LFOs, six ADSR envelopes
(each LFO module includes two dedicated ADSRs: one for rate modulation
and another for amplitude modulation), one Sine VCO, one SquareSaw
VCO, one Noise generator, VCAs, a Modulation Mixer and an Audio
Mixer. Modulation signals generated from control modules (ADSR and
LFO) are upsampled to the audio sample rate before being passed to
audio rate modules. The figure below shows the configuration and
routing of the modules comprised by Voice.


<img width="350px" src="../_static/images/Voice-diagram.svg">
