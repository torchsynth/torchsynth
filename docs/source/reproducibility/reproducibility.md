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

Reproducibility is currently guaranteed when `batch_size` is multiple
of 32 and you use the default {class}`~torchsynth.config.SynthConfig`
settings: `sample_rate`=44100, `control_rate`=441.

## Train vs Test

If a train/test split is desired, 10% of the samples are marked as
test. Because researchers with larger GPUs seek higher-throughput
with batchsize 1024, $9 \cdot 1024$ samples are designated as train,
the next 1024 samples as test, etc.

All {class}`~torchsynth.synth.AbstractSynth`
{func}`~torchsynth.synth.AbstractSynth.forward` methods return three
batched tensors: audio, latent parameters, and an `is_train` boolean
vector.
