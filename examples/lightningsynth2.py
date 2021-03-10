# -*- coding: utf-8 -*-
"""lightningsynth.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/turian/torchsynth/blob/lightning-synth/examples/lightningsynth.ipynb

# lightningsynth

Profiling for our synth on GPUs

Make sure you are on GPU runtime

If this hasn't been merged to master yet, run:
```
!pip uninstall -y torchsynth
!pip install git+https://github.com/turian/torchsynth.git@lightning-synth
```
"""

#!pip uninstall -y torchsynth
#!pip install git+https://github.com/turian/torchsynth.git@lightning-synth

#!pip install torchvision

import torch
import torch.tensor as T
from tqdm.auto import tqdm

import torch
import torchvision.models as models
import torch.autograd.profiler as profiler
import pytorch_lightning as pl

from torchsynth.globals import SynthGlobals
from torchsynth.synth import Voice
import torchsynth.module

gpus = torch.cuda.device_count()
print("Usings %d gpus" % gpus)

# Note this is the batch size for our synth!
# Not the batch size of the datasets
BATCH_SIZE = 1024

import multiprocessing

ncores = multiprocessing.cpu_count()
print(f"Using ncores {ncores} for generating batch numbers (low CPU usage)")


class batch_idx_dataset(torch.utils.data.Dataset):
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_batches


synth1M = batch_idx_dataset(1024 * 1024 // BATCH_SIZE)

# Probably don't need to pin memory for generating ints
# We use batch_size 1 here because the synth modules are already batched!
test_dataloader = torch.utils.data.DataLoader(synth1M, num_workers=ncores, batch_size=1)

synthglobals = SynthGlobals(batch_size=T(256))
voice = Voice(synthglobals)

# TODO: Change precision?
# specifies all available GPUs (if only one GPU is not occupied, uses one gpu)
# Use deterministic?
# trainer = pl.Trainer(precision=16, gpus=-1, auto_select_gpus=True, accelerator='ddp', deterministic=True)
trainer = pl.Trainer(precision=16, gpus=gpus, accelerator="ddp", deterministic=True)
trainer.test(voice, test_dataloaders=test_dataloader)

voice = Voice(synthglobals)
if torch.cuda.is_available():
    voice.cuda()
voice.eval()
with torch.no_grad():
    for i in tqdm(range(10)):
        voice()

voice = Voice(synthglobals)
if torch.cuda.is_available():
    voice.cuda()
voice.eval()
with torch.no_grad():
    for i in tqdm(range(10)):
        voice(i)

voice = Voice(synthglobals)
voice.vco_2 = torchsynth.module.Identity(synthglobals)
if torch.cuda.is_available():
    voice.cuda()
voice.eval()
with torch.no_grad():
    for i in tqdm(range(10)):
        voice(i)

voice = Voice(synthglobals)
voice.vco_1 = torchsynth.module.Identity(synthglobals)
if torch.cuda.is_available():
    voice.cuda()
voice.eval()
with torch.no_grad():
    for i in tqdm(range(10)):
        voice(i)

voice.eval()
with torch.no_grad():
    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        with profiler.record_function("forward"):
            for i in tqdm(range(10)):
                voice(i)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=10
    )
)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

prof.export_chrome_trace("trace.json")
