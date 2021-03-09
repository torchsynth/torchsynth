import torch.tensor as T

from torchsynth.default import DEFAULT_BUFFER_SIZE, DEFAULT_SAMPLE_RATE


class SynthGlobals:
    """
    Any synth module requires these "global" values.
    The should be the same for every module that is connected.
    """

    def __init__(
        self,
        batch_size: T,
        sample_rate: T = T(DEFAULT_SAMPLE_RATE),
        buffer_size: T = T(DEFAULT_BUFFER_SIZE),
    ):
        """
        Parameters
        ----------
        batch_size (T)  : Scalar that indicates how many parameter settings
                          there are, i.e. how many different sounds to generate.
        sample_rate (T) : Scalar sample rate for audio generation.
        buffer_size (T) : Duration of the output, 4 seconds by default.
        """
        assert batch_size.ndim == 0
        assert sample_rate.ndim == 0
        assert buffer_size.ndim == 0
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

    def __repr__(self):
        return (
            f"SynthGlobals(batch_size={self.batch_size}, "
            + "sample_rate={self.sample_rate}, buffer_size={self.buffer_size}"
        )
