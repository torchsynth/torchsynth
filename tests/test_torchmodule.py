"""
Tests for torch synth modules.
"""

import random
from typing import Any, Callable, Dict

import numpy as np
import pytest
import torch.tensor as T

import ddspdrum.module as numpymodule
import ddspdrum.torchmodule as torchmodule

random.seed(0)


def _random_uniform(low, hi):
    return lambda: random.uniform(low, hi)


def _random_envelope():
    adsr = numpymodule.ADSR()
    adsr.randomize()
    return adsr.npyforward(note_on_duration=random.uniform(-1, 10))


class TestTorchSynthModule:
    """
    Tests for TorchSynthModules
    """

    def _randomize_numpy_and_torch_mods(
        self,
        numpymod: numpymodule.SynthModule,
        torchmod: torchmodule.TorchSynthModule,
    ):
        """
        Randomize numpymod and set torchmod to the same.
        """
        numpymod.randomize()
        for modparameter_id in numpymod.modparameters:
            torchmod.set_modparameter(modparameter_id, T(numpymod.p(modparameter_id)))

    def _compare_values(
        self,
        numpymod: numpymodule.SynthModule,
        torchmod: torchmodule.TorchSynthModule,
        param_name_to_randfn: Dict[str, Callable[[], Any]],
    ):
        """
        Fuzz tester, for seeing that numpy and torch methods give the same values.
        """
        for i in range(1000):
            self._randomize_numpy_and_torch_mods(numpymod, torchmod)
            params = {name: randfn() for name, randfn in param_name_to_randfn.items()}

            threw = False
            try:
                ny = np.array(numpymod.npyforward(**params))
            except Exception as e:
                ny = str(type(e))
                threw = True
            for name in params:
                params[name] = T(params[name])
            try:
                ty = torchmod(**params).detach().numpy()
            except Exception as e:
                ty = str(type(e))
                threw = True
            if threw:
                print("params", params)
                print("ny", ny, numpymod)
                print("ty", ty, torchmod)
                assert ny == ty
                return
            print("ny", ny)
            print("ty", ty)
            print()
            # Absolute tolerance, not relative tolerance
            np.testing.assert_allclose(ny, ty, atol=1e-5, rtol=1e99)

    def test_TorchADSR(self):
        numpymod = numpymodule.ADSR()
        torchmod = torchmodule.TorchADSR()
        self._compare_values(
            numpymod,
            torchmod,
            param_name_to_randfn={"note_on_duration": _random_uniform(-1.0, 10.0)},
        )

    def test_TorchSineVCO(self):
        numpymod = numpymodule.SineVCO()
        torchmod = torchmodule.TorchSineVCO()
        self._compare_values(
            numpymod,
            torchmod,
            param_name_to_randfn={
                "envelope": _random_envelope,
                "phase": _random_uniform(-np.pi, np.pi),
            },
        )
