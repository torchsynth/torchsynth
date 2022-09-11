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

    def new_empty(self, *args, **kwargs):
        # noqa: E501
        """
            Implement
        [torch.Tensor.new_empty](https://pytorch.org/docs/stable/generated/torch.Tensor.new_empty.html)
        so that
        [`deepcopy`](https://docs.python.org/3/library/copy.html#copy.deepcopy)
        can be run on Signal objects.
        """

        return super().new_empty(*args, **kwargs).as_subclass(self.__class__)
