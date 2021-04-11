import os
from typing import Optional

import torch

# Currently, noise module (https://github.com/torchsynth/torchsynth/issues/255)
# and abstract synth parameter randomization
# (https://github.com/torchsynth/torchsynth/issues/253)
# are non-reproducible unless batch_size == BATCH_SIZE_FOR_REPRODUCIBILITY.
BATCH_SIZE_FOR_REPRODUCIBILITY = 128


class SynthConfig:
    """
    Any SynthModule and AbstractSynth might use
    these global configuration values.
    Every SynthModule in the same AbstractSynth
    should have the save SynthConfig.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE_FOR_REPRODUCIBILITY,
        sample_rate: Optional[int] = 44100,
        buffer_size_seconds: Optional[float] = 4.0,
        control_rate: Optional[int] = 441,
        reproducible: bool = True,
        no_grad: bool = True,
        debug: bool = "TORCHSYNTH_DEBUG" in os.environ,
        eps: float = 1e-6,
        # Unfortunately, Final is not supported until Python 3.8
        # eps: Final[float] = 1e-6,
    ):
        """
        Args:
            batch_size (int)  : Scalar that indicates how many parameter settings
            there are, i.e. how many different sounds to generate. [default: 64]
            sample_rate (int) : Scalar sample rate for audio generation.
            buffer_size (float) : Duration of the output in seconds [default: 4.0]
            control_rate (int) : Scalar sample rate for control signal generation.
            reproducible (bool) : Reproducible results, with a
                    small performance impact. (Default: True)
            no_grad (bool) : Disables gradient computations. (Default: True)
            debug (bool) : Run slow assertion tests. (Default: False, unless
                    environment variable TORCHSYNTH_DEBUG exists.)
            eps (float) : Epsilon to avoid log underrun and divide by
                          zero.
        """
        self.batch_size = torch.tensor(batch_size)
        self.sample_rate = torch.tensor(sample_rate)
        self.buffer_size_seconds = torch.tensor(buffer_size_seconds)
        self.buffer_size = torch.tensor(int(round(buffer_size_seconds * sample_rate)))
        self.control_rate = torch.tensor(control_rate)
        self.control_buffer_size = torch.tensor(
            int(round(buffer_size_seconds * control_rate))
        )
        self.no_grad = no_grad
        if not self.no_grad:
            raise ValueError(
                "Gradients have not been explicitly tested in 1.0."
                "Disable this exception at your own risk"
            )
        self.reproducible = reproducible
        if self.reproducible:
            # Currently, noise module
            # (https://github.com/torchsynth/torchsynth/issues/255)
            # and abstract synth parameter randomization
            # (https://github.com/torchsynth/torchsynth/issues/253)
            # are non-reproducible unless batch_size == BATCH_SIZE_FOR_REPRODUCIBILITY.
            if batch_size != BATCH_SIZE_FOR_REPRODUCIBILITY:
                raise ValueError(
                    "Reproducibility currently only supported "
                    f"with batch_size = {BATCH_SIZE_FOR_REPRODUCIBILITY}. "
                    "If you want a different batch_size, "
                    "initialize SynthConfig with reproducible=False"
                )
            check_for_reproducibility()

        self.debug = debug
        self.eps = eps

        # Buffer size for control signals -- this is calculated to have the
        # same duration in seconds as that buffer size for the audio rate
        # signals. Rounded to the nearest integer number of samples.
        self.control_buffer_size = (
            torch.round((self.buffer_size / sample_rate * control_rate))
            .clone()
            .detach()
            .int()
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


def check_for_reproducibility():
    """
    Reproducible results are important to torchsynth and synth1B1,
    so we are testing to make sure that the expected random results
    are produced by torch.rand when seeded. This raises an error
    indicating if reproducibility is not guaranteed.

    Running torch.rand on CPU and GPU give different results, so
    all seeded randomization where reproducibility is important
    occurs on the CPU and then is transferred over to the GPU, if
    one is being used.
    See https://discuss.pytorch.org/t/deterministic-prng-across-cpu-cuda/116275

    torchcsprng allowed for determinism between the CPU and GPU, however
    profiling indicated that torch.rand on CPU was more efficient.
    See https://github.com/pytorch/csprng/issues/126
    """
    expected = torch.tensor(
        [
            [
                4.962565898895263672e-01,
                7.682217955589294434e-01,
                8.847743272781372070e-02,
            ],
            [
                1.320304870605468750e-01,
                3.074228167533874512e-01,
                6.340786814689636230e-01,
            ],
            [
                4.900934100151062012e-01,
                8.964447379112243652e-01,
                4.556279778480529785e-01,
            ],
        ]
    )

    generator = torch.Generator(device="cpu").manual_seed(0)
    sample = torch.rand((3, 3), device="cpu", dtype=torch.float, generator=generator)
    if not torch.all(sample.eq(expected)):  # pragma: no cover
        # TODO Make this a warning before we release v1
        raise EnvironmentError(
            "Random number generator produced unexpected results. "
            "Reproducible dataset generation is not supported on your system."
            "Please comment on the discussion board: "
            "https://github.com/torchsynth/torchsynth/discussions/293 "
            "with details about your "
            f"GPU/CPU architecture and what random results you get:\n {sample}"
        )
