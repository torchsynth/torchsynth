# Example of how to build new synths

In this example we'll create a new synthesizer using
{class}`~torchsynth.module.SynthModule`s. Synths in torchsynth are created using 
an approach inspired by modular synthesizers that involves interconnecting individual 
modules. We'll create a simple single oscillator synth with an 
attack-decay-sustain-release ({class}`~torchsynth.module.ADSR`) envelope controlling 
the amplitude. More complicated architectures can be created
using the same ideas!

You can also view this example in Colab. [![Open in
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/torchsynth/torchsynth/blob/main/examples/simplesynth.ipynb)

## Creating the SimpleSynth class

All synths in torchsynth derive from {class}`~torchsynth.synth.AbstractSynth`, which
provides helpful functionality for managing children modules and parameters.

There are two steps involved in creating a class that derives from
{class}`~torchsynth.synth.AbstractSynth`:
   1. Define the modules that will be used in the `__init__` method
   2. Define how modules are connected in the `output` method
   
### Defining the modules

Here we create our `SimpleSynth` class that derives from 
{class}`~torchsynth.synth.AbstractSynth`. Override the `__init__` method and include
an optional parameter for {class}`~torchsynth.config.SynthConfig`. 
{class}`~torchsynth.config.SynthConfig` holds configuration info including the batch
size, sample rate, buffer rate, ..., etc. that defines how synthesis is carried out.

To register modules for use within `SimpleSynth`, we pass them in as a list to the
class method {func}`~torchsynth.synth.AbstractSynth.add_synth_modules`. This list
contains tuples with the name that we want to have for the module in the synth
as well as the {class}`~torchsynth.synth.SynthModule`. Each module passed in this list
will be instantiated using the same {class}`~torchsynth.config.SynthConfig` object and
added as a class attribute with the name defined by the first item in the tuple.

```python
from typing import Optional
import torch
from torchsynth.synth import AbstractSynth
from torchsynth.config import SynthConfig
from torchsynth.module import (
    ADSR,
    ControlRateUpsample,
    MonophonicKeyboard,
    SquareSawVCO,
    VCA,
)

class SimpleSynth(AbstractSynth):

    def __init__(self, synthconfig: Optional[SynthConfig] = None):
    
        # Call the constructor in the parent AbstractSynth class
        super().__init__(synthconfig=synthconfig)
        
        # Add all the modules that we'll use for this synth
        self.add_synth_modules(
            [
                ("keyboard", MonophonicKeyboard),
                ("adsr", ADSR),
                ("upsample", ControlRateUpsample),
                ("vco", SquareSawVCO),
                ("vca", VCA),
            ]
        ) 
```
### Connecting Modules

Now that we have registered the modules that we are going to use. We define how they
all are connected together in the overridden 
{func}`~torchsynth.synth.AbstractSynth.output` method.

```python
    def output(self) -> torch.Tensor:
         # Keyboard is parameter module, it returns parameter
        # values for the midi_f0 note value and the duration
        # that note is held for.
        midi_f0, note_on_duration = self.keyboard()

        # The amplitude envelope is generated based on note duration
        envelope = self.adsr(note_on_duration)

        # The envelope that we get from ADSR is at the control rate,
        # which is by default 100x less than the sample rate. This
        # reduced control rate is used for performance reasons.
        # We need to upsample the envelope prior to use with the VCO output.
        envelope = self.upsample(envelope)

        # Generate SquareSaw output at frequency for the midi note
        out = self.vco(midi_f0)

        # Apply the amplitude envelope to the oscillator output
        out = self.vca(out, envelope)

        return out
```

## Playing our SimpleSynth

That's out simple synth! Let's test it out now.

If we instantiate `SimpleSynth` without passing in a 
{class}`~torchsynth.config.SynthConfig` object then it will create one with the default 
options. We don't need to render a full batch size for this example, so let's use the 
smallest batch size that will support reproducible output. All the parameters in a 
synth are randomly assigned values, with reproducible mode on, we pass a batch_id 
value into our synth when calling it. The same sounds will always be returned for the 
same batch_id.

```python
from torchsynth.config import BASE_REPRODUCIBLE_BATCH_SIZE

# Create SynthConfig with smallest reproducible batch size.
# Reproducible mode is on by default.
synthconfig = SynthConfig(batch_size=BASE_REPRODUCIBLE_BATCH_SIZE)
synth = SimpleSynth(synthconfig)

# If you have access to a GPU.
if torch.cuda.is_available():
    synth.to("cuda")
```

Now, let's make some sounds! We just call synth with a batch_id.

```python
audio = synth(0)
```

