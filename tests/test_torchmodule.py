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

    def _randomize_numpy_and_torch_mods(
        self,
        numpymod: numpymodule.SynthModule,
        torchmod: torchmodule.TorchSynthModule,
        param_name_to_type: Dict[str, str],
    ):
        """
        Randomize numpymod and set torchmod to the same.
        """
        numpymod.randomize()
        for modparameter_id in numpymod.modparameters:
            torchmod.set_modparameter(
                modparameter_id, T(numpymod.p(modparameter_id))
            )

    def _choose_random_params(
        self,
        param_name_to_type: Dict[str, str]
    ):
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
        return params

    def _compare_values(
            self,
            numpymod: numpymodule.SynthModule,
            torchmod: torchmodule.TorchSynthModule,
            param_name_to_type: Dict[str, str],
    ):
        """
        Fuzz tester, for seeing that numpy and torch methods give the same values.
        """
        for i in range(1000):
            self._randomize_numpy_and_torch_mods(numpymod, torchmod)
            params = self._choose_random_params(param_name_to_type)

            threw = False
            try:
                ny = np.array(numpymod.npyforward(**params))
            except Exception as e:
                ny = str(type(e))
                threw = True
            for name in params:
                params[name] = T(params[name])
            try:
                ty = torchmod(**params).numpy()
            except Exception as e:
                ty = str(type(e))
                threw = True
            if threw:
                assert ny == ty
                return
            print(ny)
            print(ty)
            print()
            np.testing.assert_allclose(ny, ty, rtol=1e-4)

    def test_TorchADSR(self):
        numpymod = numpymodule.ADSR()
        torchmod = torchmodule.TorchADSR()
        self._compare_values(
            numpymod, torchmod, param_name_to_type={"note_on_duration": "float"}
        )

    def test_TorchSineVCO(self):
        numpymod = numpymodule.SineVCO()
        torchmod = torchmodule.TorchSineVCO()
        numpyadsr = numpymodule.SineADSR()
        torchadsr = torchmodule.TorchSineADSR()
        self._compare_values(
            numpymod, torchmod, param_name_to_type={"note_on_duration": "float"}
        )
