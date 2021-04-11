Reproduciblity
==============

```{contents}
:depth: 2
```

## Overview

We use deterministic random number generation to ensure replicability,
even of noise oscillators. Nonetheless, there are small numeric
differences between the CPU and GPU. The mean-average-error between
audio samples generated on CPU and GPU are < 1e-2.

## Defaults

Reproducibility is currently guaranteed when using the default
SynthConfig settings: `batch_size`=128, `sample_rate`=44100,
`control_rate`=441. In future releases, we will enable reproducibility
with larger batch sizes.
