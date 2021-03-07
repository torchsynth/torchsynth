"""
A convenience type for signals.
"""

import torch

class Signal(torch.Tensor):
    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x):
        super().__init__(x) # optional

    @property
    def batch_size(self):
        assert self.ndim == 2
        return self.shape[0]

    @property
    def nsamples(self):
        assert self.ndim == 2
        return self.shape[1]
