"""
A convenience type for signals.
I'd like to call this filename signal.py but it conflicts with
a core python module.
"""

import torch


class Signal(torch.Tensor):
    @property
    def batch_size(self):
        assert self.ndim == 2
        return self.shape[0]

    @property
    def num_samples(self):
        assert self.ndim == 2
        return self.shape[1]
