"""
FIR Filtering implement
"""

from typing import Dict, Optional

import torch
from torch import Tensor as T

from torchsynth.module import SynthModule
from torchsynth.signal import Signal
from torchsynth.config import SynthConfig
import torchsynth.util as util


class LowPassFilter(SynthModule):
    def __init__(
        self,
        synthconfig: SynthConfig,
        device: Optional[torch.device] = None,
        **kwargs: Dict[str, T],
    ):
        super().__init__(synthconfig, device, **kwargs)
        impulse = util.sinc()

    def output(self, audio_in: Signal) -> Signal:
        return audio_in
