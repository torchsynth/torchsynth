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

from torchsynth.synth import Voice
from torchsynth.globals import SynthGlobals

from tqdm.auto import tqdm

import torch.tensor as T
import torch

synthglobals = SynthGlobals(batch_size=T(256))
voice = Voice(synthglobals).cuda()

voice.eval()
with torch.no_grad():
    for i in tqdm(range(1000)):
        voice()
