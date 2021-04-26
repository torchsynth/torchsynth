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
[![Travis CI build status](https://travis-ci.com/torchsynth/torchsynth.png)](https://travis-ci.com/torchsynth/torchsynth)
[![Documentation Status](https://readthedocs.org/projects/torchsynth/badge/?version=latest)](https://torchsynth.readthedocs.io/en/latest/?badge=latest)

</div>

## Installation

```
pip3 install torchsynth
```

Note that torchsynth requires PyTorch version 1.8 or greater.

If you'd like to hear torchsynth, check out synth1K1, a dataset of 
1024 4-second sounds rendered from the {class}`~torchsynth.synth.Voice`
synthesizer.

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" 
src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1035755485&color=%23792ee5&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/user-357924775" title="torchsynth" target="_blank" style="color: #cccccc; text-decoration: none;">torchsynth</a> · <a href="https://soundcloud.com/user-357924775/synth1k1" title="Synth1K1" target="_blank" style="color: #cccccc; text-decoration: none;">Synth1K1</a></div>

All the individual sound files and code to generate synth1K1 are available in a 
git repo: https://github.com/torchsynth/synth1K1 
