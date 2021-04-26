synth1B1
========

```{contents}
:depth: 2
```

## Overview

synth1B1 is a corpus consisting of one million hours of audio: one
billion 4-second synthesized sounds. The corpus is multi-modal:
Each sound includes its corresponding synthesis parameters.

synth1B1 is generated *on the fly* by torchsynth 1.x, using the
Voice synth with its default settings.

## Experimental Control

Researchers can denote subsamples of this corpus as synth1M1,
synth10M1, *etc.*, which would refer to the first 1 million and 10
million samples of Synth1B1 respectively.

If you change any of the defaults, *e.g.* in SynthConfig, please
call that in your work, and use a variant of the name synth1B1.

One tenth of the examples are designated as the test set. See
[Reproducibility > Train vs. Test](../reproducibility/reproducibility)
for more information.

The nomenclature "synth1B1-312-6" denotes batch 312 (assuming
batch size of 128) and sound 6 within that batch.

## Semantic Versioning

We use a slightly different convention than traditional [Semantic
Versioning](https://semver.org/).

Given a version number MAJOR.MINOR.PATCH, we increment the:

* MAJOR version when the default output of Voice changes.
* MINOR version when we make incompatible API changes, but the
default output of Voice remains reproducible.
* PATCH version when we make backwards compatible bug fixes and
improvements.

For example, any torchsynth 1.x release can generate synth1B1.

When torchsynth 2.x is released, it will generate synth1B2.  Any
pre-release (*e.g.* 2.0.0-pre1) is *not* guaranteed to generate
synth1B2 until 2.0.0 is released.
