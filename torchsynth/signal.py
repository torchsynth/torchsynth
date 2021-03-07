"""
A convenience type for signals.
"""

import torch.tensor as T


class Signal(T):
    @property
    def batch_size(self):
        assert self.ndim == 2
        return self.shape[0]

    @property
    def nsamples(self):
        assert self.ndim == 2
        return self.shape[1]
