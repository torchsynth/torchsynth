import torch


class Signal(torch.Tensor):
    """
    A convenience type for batched signals, either audio signals
    or control signals. A signal is 2D :class:`~torch.Tensor`:
    `batch` x `num_samples`.

    Note: To cast a tensor as a signal:
    ``torch.zeros(batch_size, N).as_subclass(Signal)``
    """

    @property
    def batch_size(self):
        assert self.ndim == 2
        return self.shape[0]

    @property
    def num_samples(self):
        assert self.ndim == 2
        return self.shape[1]
