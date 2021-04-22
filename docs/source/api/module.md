torchsynth.module
=================

```{eval-rst}
SynthModule
-----------
.. autoclass:: torchsynth.module.SynthModule
    :members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: default_parameter_ranges

Audio Rate Modules
------------------
These modules operate at full audio sampling rate.

AudioMixer
==========
.. autoclass:: torchsynth.module.AudioMixer
    :members:
    :undoc-members:
    :show-inheritance:

ControlRateUpsample
===================
.. autoclass:: torchsynth.module.ControlRateUpsample
    :members:
    :undoc-members:
    :show-inheritance:

Noise
=====
.. autoclass:: torchsynth.module.Noise
    :members:
    :undoc-members:
    :show-inheritance:

SineVCO
=======
.. autoclass:: torchsynth.module.SineVCO
    :members:
    :undoc-members:
    :show-inheritance:

SquareSawVCO
============
.. autoclass:: torchsynth.module.SquareSawVCO
    :members:
    :undoc-members:
    :show-inheritance:

TorchFmVCO
==========
.. autoclass:: torchsynth.module.TorchFmVCO
    :members:
    :undoc-members:
    :show-inheritance:

VCA
===
.. autoclass:: torchsynth.module.VCA
    :members:
    :undoc-members:
    :show-inheritance:

VCO
===
.. autoclass:: torchsynth.module.VCO
    :members:
    :undoc-members:
    :show-inheritance:


Control Rate Modules
--------------------
Control rate modules produce signals that are used to modulate parameters of
other modules. For performance these modules run at a reduced sampling rate.

ControlRateModule
=================
.. autoclass:: torchsynth.module.ControlRateModule
    :members:
    :undoc-members:
    :show-inheritance:

ADSR
====
.. autoclass:: torchsynth.module.ADSR
    :members:
    :undoc-members:
    :show-inheritance:

ControlRateVCA
==============
.. autoclass:: torchsynth.module.ControlRateVCA
    :members:
    :undoc-members:
    :show-inheritance:

LFO
===
.. autoclass:: torchsynth.module.LFO
    :members:
    :undoc-members:
    :show-inheritance:

ModulationMixer
===============
.. autoclass:: torchsynth.module.ModulationMixer
    :members:
    :undoc-members:
    :show-inheritance:

Parameter Modules
-----------------
Parameter modules simply output values.

CrossfadeKnob
=============
.. autoclass:: torchsynth.module.CrossfadeKnob
    :members:
    :undoc-members:
    :show-inheritance:

HardModeSelector
================
.. autoclass:: torchsynth.module.HardModeSelector
    :members:
    :undoc-members:
    :show-inheritance:

MonophonicKeyboard
==================
.. autoclass:: torchsynth.module.MonophonicKeyboard
    :members:
    :undoc-members:
    :show-inheritance:

SoftModeSelector
================
.. autoclass:: torchsynth.module.SoftModeSelector
    :members:
    :undoc-members:
    :show-inheritance:
```
