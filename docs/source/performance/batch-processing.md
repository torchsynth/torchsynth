Batch Processing and Performance
================================

```{contents}
:depth: 2
```

## Batch Processing

To take advantage of the parallel processing power of a GPU, all
modules render audio in batches. Larger batches enable higher
throughput on GPUs. The default batch size is 128, which requires
$\approx$2.3GB of GPU memory, and is 16200x faster than realtime on a V100.
(GPU memory consumption is approximately $\approxeq$ 1216 + 8.19
$\cdot$ batch_size MB, including the torchsynth model.)

<div align="center">

<img alt="gpu-speed-profiles" src="../_static/images/gpu-speed-profiles.svg">

<img alt="gpu-mem-profiles" src="../_static/images/gpu-mem-profiles.svg">

</div>

## ADSR Batches

An example of a batch of 4 of randomly generated ADSR envelopes is shown below:

<div align="center">

<img alt="ADSR" src="../_static/images/ADSR.svg">

</div>
