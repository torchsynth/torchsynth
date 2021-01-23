"""
Tests for torch synth modules.
"""

import random
from typing import Dict

import numpy as np
import pytest
import torch.tensor as T

import ddspdrum.module as numpymodule
import ddspdrum.torchmodule as torchmodule

random.seed(0)


class TestTorchSynthModule:
    """
    Tests for TorchSynthModules
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
                else:
                    assert False
            try:
                ny = np.array(numpyf(**params))
            except RuntimeWarning:
                ny = "RuntimeWarning"
            for name in params:
                params[name] = T(params[name])
            try:
                ty = torchf(**params).numpy()
            except RuntimeWarning:
                ty = "RuntimeWarning"
            if ny == "RuntimeWarning" or ty == "RuntimeWarning":
                assert ny == ty
                return
            print(ny)
            print(ty)
            print()
            np.testing.assert_allclose(ny, ty, rtol=1e-4)

    def test_TorchSineVCO(self):
        nmod = numpymodule.SineVCO()

#        # SineVCO test
#        midi_f0 = T(12.0)
#        mod_depth = T(50.0)
#        sine_vco = TorchSineVCO(midi_f0=midi_f0, mod_depth=mod_depth)
#        sine_out = sine_vco(envelope, phase=T(0.0))

