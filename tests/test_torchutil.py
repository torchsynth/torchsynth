"""
Tests for torch DSP utils.
"""

import random
from typing import Dict

import numpy as np
import pytest
import torch.tensor as T

import ddspdrum.numpyutil as numpyutil
import ddspdrum.torchutil as torchutil

random.seed(0)


class TestTorchUtil:
    """
    Tests for torchutil methods
    """

    def _compare_values(self, numpyf, torchf, param_name_to_type: Dict[str, str]):
        """
        Fuzz tester, for seeing that numpy and torch methods give the same values.
        """
        for i in range(1000):
            params = {}
            for name, ty in param_name_to_type.items():
                if ty == "float":
                    params[name] = random.uniform(-10, 10)
                elif ty == "float1000":
                    params[name] = random.uniform(0, 1000)
                elif ty == "pr":
                    params[name] = random.uniform(0, 1)
                elif ty == "signal":
                    params[name] = np.random.rand(512)
                elif ty == "int":
                    params[name] = random.randint(0, 1000)
                elif ty == "bool":
                    params[name] = random.choice([True, False])
                else:
                    assert False

            threw = False
            try:
                ny = np.array(numpyf(**params))
            except Exception as e:
                ny = type(e)
                threw = True
            for name in params:
                params[name] = T(params[name])
            try:
                ty = torchf(**params).numpy()
            except Exception as e:
                ty = type(e)
                threw = True
            if threw:
                assert ny == ty
                return
            print(ny)
            print(ty)
            print()
            np.testing.assert_allclose(ny, ty, rtol=1e-4)

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

    def test_peak_gain_for_Q(self):
        # I'm not sure the range of Q, so just try both narrow and coarse
        self._compare_values(
            numpyutil.peak_gain_for_Q,
            torchutil.peak_gain_for_Q,
            {"Q": "float"},
        )
        self._compare_values(
            numpyutil.peak_gain_for_Q,
            torchutil.peak_gain_for_Q,
            {"Q": "pr"},
        )

    def test_hz_to_midi(self):
        self._compare_values(
            numpyutil.hz_to_midi,
            torchutil.hz_to_midi,
            {"hz": "float1000"},
        )

    def test_fix_length(self):
        self._compare_values(
            numpyutil.fix_length,
            torchutil.fix_length,
            {"signal": "signal", "length": "int"},
        )

    def test_crossfade(self):
        self._compare_values(
            numpyutil.crossfade,
            torchutil.crossfade,
            {"in_1": "signal", "in_2": "signal", "ratio": "pr"},
        )

    def test_linspace(self):
        self._compare_values(
            np.linspace,
            torchutil.linspace,
            {
                "start": "float",
                "stop": "float",
                "num": "int",
                "endpoint": "bool"
            },
        )
