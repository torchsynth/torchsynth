import torch


def check_for_determinism():
    """
    Reproducible results are important to torchsynth and Synth1B1, so we are testing
    to make sure that the expected random results are produced by torch.rand when
    seeded. This raises an error indicating if reproducibility is not guaranteed.

    Running torch.rand on CPU and GPU give different results, so all seeded
    randomization where determinism is important occurs on the CPU and then is
    transferred over to the GPU, if one is being used.
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
            "Please file an issue on github, see: "
            "https://github.com/turian/torchsynth/issues/248 with details about your "
            f"CPU architecture and what random results you get:\n {sample}"
        )
