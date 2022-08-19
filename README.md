<div align="center">

# torchsynth

The fastest synth in the universe.

<img width="450px" src="https://raw.githubusercontent.com/torchsynth/torchsynth/main/assets/logo-with-caption.jpg">

</div>

## Introduction

torchsynth is based upon traditional modular synthesis written in
pytorch. It is GPU-optional and differentiable.

Most synthesizers are fast in terms of latency. torchsynth is fast
in terms of throughput. It synthesizes audio 16200x faster than
realtime (714MHz) on a single GPU. This is of particular interest
to audio ML researchers seeking large training corpora.

Additionally, all synthesized audio is returned with the underlying
latent parameters used for generating the corresponding audio. This
is useful for multi-modal training regimes.

<div align="center">

[Documentation](https://torchsynth.rtfd.io/en/latest/)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/torchsynth/torchsynth/blob/main/examples/examples.ipynb)

[![PyPI](https://img.shields.io/pypi/v/torchsynth)](https://pypi.org/project/torchsynth/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/torchsynth)
![PyPI - License](https://img.shields.io/pypi/l/torchsynth)
[![codecov.io](https://codecov.io/gh/torchsynth/torchsynth/branch/main/graphs/badge.svg?logoWidth=18)](https://codecov.io/github/torchsynth/torchsynth?branch=master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/torchsynth/torchsynth.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/torchsynth/torchsynth/alerts/)
[![Travis CI build status](https://app.travis-ci.com/torchsynth/torchsynth.svg?branch=main)](https://app.travis-ci.com/github/torchsynth/torchsynth)
[![Documentation Status](https://readthedocs.org/projects/torchsynth/badge/?version=latest)](https://torchsynth.readthedocs.io/en/latest/?badge=latest)

</div>

## Installation

```
pip3 install torchsynth
```

Note that torchsynth requires PyTorch version 1.8 or greater.

## Listen

If you'd like to hear torchsynth, check out
[synth1K1](https://github.com/torchsynth/synth1K1), a dataset of
1024 4-second sounds rendered from the
[Voice](https://torchsynth.readthedocs.io/en/latest/api/synth.html#torchsynth.synth.Voice)
synthesizer, or listen [on SoundCloud](https://soundcloud.com/user-357924775/synth1k1).

## Citation

If you use this work in your research, please cite:

```
@inproceedings{turian2021torchsynth,
	title        = {One Billion Audio Sounds from {GPU}-enabled Modular Synthesis},
	author       = {Joseph Turian and Jordie Shier and George Tzanetakis and Kirk McNally and Max Henry},
	year         = 2021,
	month        = Sep,
	booktitle    = {Proceedings of the 23rd International Conference on Digital Audio Effects (DAFx2020)},
	location     = {Vienna, Austria}
}
```
