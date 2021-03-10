"""
Tests for torch csprng
"""

import torch
import torchcsprng as csprng


class TestTorchUtil:
    def test_determinism(self):
        """
        This determines if your version of csprng will have
        different values than we expect, in which case you
        should install 0.2.0:
        https://github.com/pytorch/csprng

        This is a CPU only tes
        """
        mt19937_gen = csprng.create_mt19937_generator(42)
        generated = torch.empty(5, device="cpu").uniform_(0, 1, generator=mt19937_gen)
        expected = torch.tensor(
            [
                0.7278736233711243,
                0.0012899230932816863,
                0.2927267551422119,
                0.5945193767547607,
                0.9566778540611267,
            ]
        )
        assert torch.mean(torch.abs(generated - expected)) < 1e-6

    def test_determinism_gpu(self):
        """
        As above, if you have a GPU
        """
        if torch.cuda.is_available():
            mt19937_gen = csprng.create_mt19937_generator(42)
            generated = torch.empty(5, device="cuda").uniform_(
                0, 1, generator=mt19937_gen
            )
            expected = torch.tensor(
                [
                    0.7278736233711243,
                    0.0012899230932816863,
                    0.2927267551422119,
                    0.5945193767547607,
                    0.9566778540611267,
                ],
                device="cuda",
            )
            assert torch.mean(torch.abs(generated - expected)) < 1e-6
