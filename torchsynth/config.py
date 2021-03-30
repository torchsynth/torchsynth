import warnings
import torch


def check_for_determinism():
    """
    This checks to make sure that the expected random results
    are produced by torch.rand when seeded. This raises a warning
    indicating if reproducibility is not guaranteed.
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

    torch.random.manual_seed(0)
    sample = torch.rand((3, 3), dtype=torch.float)
    if not torch.all(sample.eq(expected)):
        # TODO Make this a warning before we release v1
        raise EnvironmentError(
            "Random number generator produced incorrect results. "
            "Reproducible dataset generation is not supported on this system."
        )
