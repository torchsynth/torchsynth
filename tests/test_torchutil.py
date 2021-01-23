"""
Tests for torch DSP utils.
"""

from typing import Dict

import numpy as np
import torch.tensor as T
import pytest

import ddspdrum.torchutil as torchutil
import ddspdrum.numpyutil as numpyutil

import random

random.seed(0)


class TestTorchUtil:
    """
    Tests for torchutil methods
    """

    def _compare_values(self, numpyf, torchf, param_name_to_type: Dict[str, str]):
        for i in range(1000):
            params = {}
            for name, ty in param_name_to_type.items():
                if ty == "float":
                    params[name] = random.uniform(-10, 10)
                elif ty == "pr":
                    params[name] = random.uniform(0, 1)
                elif ty == "signal":
                    params[name] = np.random((512,))
                else:
                    assert False
            ny = np.array(numpyf(**params))
            for name in params:
                params[name] = T(params[name])
            ty = torchf(**params).numpy()
            print(ny)
            print(ty)
            print()
            np.testing.assert_allclose(ny, ty, rtol=1e-5)

    def test_amplitude_to_db(self):
        self._compare_values(
            numpyutil.amplitude_to_db,
            torchutil.amplitude_to_db,
            {"amplitude": "float", "amin": "float"},
        )

    def test_db_to_amplitude(self):
        self._compare_values(
            numpyutil.db_to_amplitude,
            torchutil.db_to_amplitude,
            {"db": "float"},
        )
