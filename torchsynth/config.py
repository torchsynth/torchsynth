from typing import Optional

import torch


class SynthConfig:
    """
    Any SynthModule and AbstractSynth might use
    these global configuration values.
    Every SynthModule in the same AbstractSynth
    should have the save SynthConfig.
    """

    def __init__(
        self,
        batch_size: int,
        sample_rate: Optional[int] = 44100,
        buffer_size_seconds: Optional[float] = 4.0,
        control_rate: Optional[int] = 441,
        debug: bool = False,
    ):
        """
        Args:
            batch_size (int)  : Scalar that indicates how many parameter settings
            there are, i.e. how many different sounds to generate.
            sample_rate (int) : Scalar sample rate for audio generation.
            buffer_size (float) : Duration of the output in seconds [default: 4.0]
            control_rate (int) : Scalar sample rate for control signal generation.
            debug (bool) : Run slow assertion tests. (Default: False)
        """
        self.batch_size = torch.tensor(batch_size)
        self.sample_rate = torch.tensor(sample_rate)
        self.buffer_size_seconds = torch.tensor(buffer_size_seconds)
        self.buffer_size = torch.tensor(int(round(buffer_size_seconds * sample_rate)))
        self.control_rate = torch.tensor(control_rate)
        self.debug = debug

        # Buffer size for control signals -- this is calculated to have the
        # same duration in seconds as that buffer size for the audio rate
        # signals. Rounded to the nearest integer number of samples.
        self.control_buffer_size = torch.tensor(
            int(torch.round((self.buffer_size / sample_rate * control_rate)))
        )

    def to(self, device: torch.device):
        # Only helpful to have sample and control rates on device, and as a float
        self.sample_rate = self.sample_rate.to(device).float()
        self.control_rate = self.control_rate.to(device).float()

    def __repr__(self):  # pragma: no cover
        return (
            f"SynthGlobals(batch_size={self.batch_size}, "
            + f"sample_rate={self.sample_rate}, buffer_size={self.buffer_size}, "
            + f"control_rate={self.control_rate}, "
            + f"control_buffer_size={self.control_buffer_size})"
        )
