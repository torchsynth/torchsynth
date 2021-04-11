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

## Introduction

torchsynth is based upon traditional modular synthesis written in
pytorch. It is GPU-optional and differentiable.

Most synthesizers are fast in terms of latency. torchsynth is fast
in terms of throughput. It synthesizes audio 15000x faster than
realtime on a single GPU.

```{toctree}
---
maxdepth: 1
name: getting_started
caption: Getting started
---
getting-started/installation
getting-started/quickstart
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
