```{eval-rst}
.. torchsynth documentation master file, created by
   sphinx-quickstart on Tue Mar  9 01:30:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```


# torchsynth documentation

<div align="center">

The fastest synth in the universe.

<img width="450px" src="_static/images/logo-with-caption.jpg">

</div>

<hr>

torchsynth is based upon traditional modular synthesis written in
pytorch. It is GPU-optional and differentiable.

Most synthesizers are fast in terms of latency. torchsynth is fast
in terms of throughput. It synthesizes audio 16200x faster than
realtime (714MHz) on a single GPU. This is of particular interest
to audio ML researchers seeking large training corpora.

Additionally, all synthesized audio is returned with the underlying
latent parameters used for generating the corresponding audio. This
is useful for multi-modal training regimes.

If you'd like to hear torchsynth, check out
[synth1K1](https://github.com/torchsynth/synth1K1), a dataset of
1024 4-second sounds rendered from the {class}`~torchsynth.synth.Voice`
synthesizer, or listen to the following SoundCloud embed:

<iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" 
src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/1035755485&color=%23792ee5&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/user-357924775" title="torchsynth" target="_blank" style="color: #cccccc; text-decoration: none;">torchsynth</a> · <a href="https://soundcloud.com/user-357924775/synth1k1" title="Synth1K1" target="_blank" style="color: #cccccc; text-decoration: none;">Synth1K1</a></div>

```{toctree}
---
maxdepth: 1
name: getting_started
caption: Getting started
---
getting-started/installation
getting-started/quickstart
getting-started/detailed-walkthrough
```


```{toctree}
---
maxdepth: 1
name: performance
caption: Performance
---
performance/batch-processing
performance/multi-gpu
```

```{toctree}
---
maxdepth: 1
name: modular_design
caption: Modular Design
---
modular-design/modular-principles
modular-design/new-synths
```


```{toctree}
---
maxdepth: 1
name: reproducibility
caption: Reproducibility
---
reproducibility/reproducibility
reproducibility/synth1B1
```


```{toctree}
---
maxdepth: 1
name: contributing
caption: Contributing
---
contributing/contributing
```


```{toctree}
---
maxdepth: 3
name: modules
caption: API
---
api/config
api/module
api/parameter
api/signal
api/synth
api/util
```


```{toctree}
---
maxdepth: 1
name: docmap
caption: INDICES AND TABLES
---
genindex
search
```
