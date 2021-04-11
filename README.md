<div align="center">

# torchsynth

The fastest synth in the universe.

<img width="450px" src="https://raw.githubusercontent.com/torchsynth/torchsynth/main/assets/logo-with-caption.jpg">

</div>

## Introduction

torchsynth is based upon traditional modular synthesis written in
pytorch. It is GPU-optional and differentiable.

Most synthesizers are fast in terms of latency. torchsynth is fast
in terms of throughput, achieving over 15000x realtime throughput
on a single GPU.

<div align="center">

[Documentation](https://torchsynth.rtfd.io/en/latest/)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/torchsynth/torchsynth/blob/main/examples/examples.ipynb)

[![PyPI](https://img.shields.io/pypi/v/torchsynth)](https://pypi.org/project/torchsynth/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/torchsynth)
![PyPI - License](https://img.shields.io/pypi/l/torchsynth)
[![codecov.io](https://codecov.io/gh/torchsynth/torchsynth/branch/main/graphs/badge.svg?logoWidth=18)](https://codecov.io/github/torchsynth/torchsynth?branch=master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/torchsynth/torchsynth.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/torchsynth/torchsynth/alerts/)
[![Travis CI build status](https://travis-ci.com/torchsynth/torchsynth.png)](https://travis-ci.com/torchsynth/torchsynth)
[![Documentation Status](https://readthedocs.org/projects/torchsynth/badge/?version=latest)](https://torchsynth.readthedocs.io/en/latest/?badge=latest)

</div>

## Profiling

<div align="center">

<img width="350px" src="https://media.githubusercontent.com/media/torchsynth/torchsynth/main/docs/source/_static/images/gpu-speed-profiles.svg">

</div>

## Installation

```
pip3 install torchsynth
```

Note that torchsynth requires PyTorch version 1.8 or greater.
