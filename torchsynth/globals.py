from typing import Optional

import torch
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
        control_rate: Optional[T] = None,
    ):
        """
        Args:
            batch_size (T)  : Scalar that indicates how many parameter settings
            there are, i.e. how many different sounds to generate.
            sample_rate (T) : Scalar sample rate for audio generation.
            buffer_size (T) : Duration of the output, 4 seconds by default.
            control_rate (T) : Scalar sample rate for control signal generation.
        """
        assert batch_size.ndim == 0
        assert sample_rate.ndim == 0
        assert buffer_size.ndim == 0
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Default control rate to 100 times less than audio rate
        if control_rate is None:
            self.control_rate = (self.sample_rate // 100).clone().detach()
        else:
            assert self.control_rate.ndim == 0
            self.control_rate = control_rate

        # Buffer size for control signals
        self.control_buffer_size = (
            torch.ceil((buffer_size / sample_rate * self.control_rate))
            .clone()
            .detach()
            .int()
        )

    def to(self, device: torch.device):
        # Only helpful to have sample and control rates on device, and as a float
        self.sample_rate = self.sample_rate.to(device).float()
        self.control_rate = self.control_rate.to(device).float()

    def __repr__(self):
        return (
            f"SynthGlobals(batch_size={self.batch_size}, "
            + f"sample_rate={self.sample_rate}, buffer_size={self.buffer_size}, "
            + f"control_rate={self.control_rate}, "
            + f"control_buffer_size={self.control_buffer_size})"
        )
