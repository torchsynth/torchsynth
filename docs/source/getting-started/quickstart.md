Quickstart
==========

```{contents}
:depth: 2
```

## Which way to synth1B1-312-6?

In this simple example, we use Voice to generate the 312th batch
of synth1B1. We then select sample 6 from this batch, and save it
to a WAV file.

You will need to `pip install torchaudio` in order to save the WAV
file. Alternately, you could modify the code slightly and use
[SoundFile](https://pypi.org/project/SoundFile/).

```
import torch
import torchaudio
from torchsynth.synth import Voice

voice = Voice()
# Run on the GPU if it's available
if torch.cuda.is_available():
    voice = voice.to("cuda")

# Generate batch 312
# All batches are [128, 176400], i.e. 128 4-second sounds at 44100Hz
# Each sound is a monophonic 1D tensor.
synth1B1_312 = voice(312)

# Select synth1B1-312-6
synth1B1_312_6 = synth1B1_312[6]

# We add one channel at the beginning, for torchaudio
torchaudio.save("synth1B1-312-6.wav", synth1B1_312_6.unsqueeze(0).cpu(), voice.sample_rate)
```
