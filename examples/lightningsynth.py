#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # lightningsynth
#
# Profiling for our synth on GPUs
#
# If this hasn't been merged to master yet, run:
# ```
# # !pip uninstall -y torchsynth
# # !pip install git+https://github.com/turian/torchsynth.git@lightning-synth
# ```

import torch
import torch.tensor as T
from tqdm.auto import tqdm

from torchsynth.globals import SynthGlobals
from torchsynth.synth import Voice

synthglobals = SynthGlobals(batch_size=T(256))
voice = Voice(synthglobals)

if torch.cuda.is_available():
    voice.cuda()

voice.eval()
with torch.no_grad():
    for i in tqdm(range(1000)):
        voice(i)
