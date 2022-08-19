"""
Global configuration for :class:`~torchsynth.synth.AbstractSynth` and its
component :class:`~torchsynth.module.SynthModule`.
"""

import os
from typing import Optional

import torch

#: This batch size is a nice trade-off between speed and memory consumption. On
#: a typical GPU this consumes ~2.3GB of memory for the default Voice.
#: Learn more about `batch processing <../performance/batch-processing.html>`_.
DEFAULT_BATCH_SIZE = 128

#: Smallest batch size divisor that is supported for any reproducible output
#: This is because :class:`~torch.module.Noise`: creates deterministic
#: noise batches in advance, for speed.
BASE_REPRODUCIBLE_BATCH_SIZE = 32

#: If a train/test split is desired, 10% of the samples are marked
#: as test. Because researchers with larger GPUs seek higher-throughput
#: with batchsize 1024, $9 \cdot 1024$ samples are designated as train,
#: the next 1024 samples as test, etc.
N_BATCHSIZE_FOR_TRAIN_TEST_REPRODUCIBILITY = 1024


class SynthConfig:
    """
    Any :class:`~torchsynth.module.SynthModule` and
    :class:`~torchsynth.synth.AbstractSynth` might use these global
    configuration values. Every :class:`~torchsynth.module.SynthModule`
    in the same :class:`~torchsynth.synth.AbstractSynth` should
    have the save SynthConfig.

    Args:
        batch_size: Scalar that indicates how many parameter settings
            there are, i.e. how many different sounds to generate.
        sample_rate: Scalar sample rate for audio generation.
        buffer_size: Duration of the output in seconds.
        control_rate: Scalar sample rate for control signal generation.
            reproducible: Reproducible results, with a
            small performance impact.
        no_grad: Disables gradient computations.
        debug: Run slow assertion tests. (Default: False, unless
            environment variable TORCHSYNTH_DEBUG exists.)
        eps: Epsilon to avoid log underrun and divide by zero.
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
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
            # We currently only support reproducibility with batch sizes that
            # are multiples of BASE_REPRODUCIBLE_BATCH_SIZE
            if batch_size % BASE_REPRODUCIBLE_BATCH_SIZE != 0.0:
                raise ValueError(
                    "Reproducibility currently only supported "
                    f"with batch_size multiples of {BASE_REPRODUCIBLE_BATCH_SIZE}. "
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
        """
        For speed, we've noticed that it is only helpful to have
        sample and control rates on device, and as a float.
        """
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
    This method is called automatically if your
    :class:`~torchsynth.config.SynthConfig` specifies
    ``reproducibility=True``.

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
