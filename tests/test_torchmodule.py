"""
Tests for torch synth modules.
"""

import random
from typing import Any, Callable, Dict

import numpy as np
import pytest
import torch.nn
import torch.tensor as T

import ddspdrum.module as numpymodule
import ddspdrum.torchmodule as torchmodule
from ddspdrum.parameter import ParameterRange, TorchParameter
import ddspdrum.defaults as defaults

random.seed(0)


def _random_uniform(low, hi):
    return lambda: random.uniform(low, hi)


def _random_envelope():
    adsr = numpymodule.ADSR()
    adsr.randomize()
    return adsr.npyforward(note_on_duration=random.uniform(-1, 10))


def _random_signal():
    return np.random.rand(44100)


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
        for parameter_id in numpymod.modparameters:
            torchmod.set_parameter(parameter_id, numpymod.p(parameter_id))

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
                params[name] = T(params[name]).float()
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

    def test_TorchFmVCO(self):
        # Is this correct?
        numpymod = numpymodule.FmVCO()
        torchmod = torchmodule.TorchFmVCO()
        self._compare_values(
            numpymod,
            torchmod,
            param_name_to_randfn={
                "mod_signal": _random_signal
            }
        )

    def test_TorchSquareSawVCO(self):
        numpymod = numpymodule.SquareSawVCO()
        torchmod = torchmodule.TorchSquareSawVCO()
        self._compare_values(
            numpymod,
            torchmod,
            param_name_to_randfn={
                "envelope": _random_envelope,
                "phase": _random_uniform(-np.pi, np.pi),
                "shape": _random_uniform(0, 1)
            }
        )

    def test_get_parameter(self):
        module = torchmodule.TorchSynthModule()
        param_1 = TorchParameter(data=T(1.0), parameter_name="param_1")
        module.add_parameters([param_1])
        assert module.get_parameter("param_1") == param_1

    def test_get_parameter_0to1(self):
        module = torchmodule.TorchSynthModule()
        param_1 = TorchParameter(data=T(0.5), parameter_name="param_1")
        module.add_parameters([param_1])
        assert module.get_parameter_0to1("param_1") == 0.5

        param_2 = TorchParameter(
            value=5000.0,
            parameter_range=ParameterRange(0.0, 20000.0),
            parameter_name="param_2"
        )
        module.add_parameters([param_2])
        assert module.get_parameter_0to1("param_2") == 0.25

    def test_set_parameter(self):
        module = torchmodule.TorchSynthModule()
        param_1 = TorchParameter(
            value=5000.0,
            parameter_range=ParameterRange(0.0, 20000.0),
            parameter_name="param_1"
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25

        module.set_parameter("param_1", 10000.0)
        assert module.torchparameters["param_1"] == 0.5
        assert module.torchparameters["param_1"].from_0to1() == 10000.0

        with pytest.raises(AssertionError):
            module.set_parameter_0to1("param_1", -100.0)

    def test_set_parameter_0to1(self):
        module = torchmodule.TorchSynthModule()
        param_1 = TorchParameter(
            value=5000.0,
            parameter_range=ParameterRange(0.0, 20000.0),
            parameter_name="param_1"
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25

        module.set_parameter_0to1("param_1", 0.5)
        assert module.torchparameters["param_1"] == 0.5
        assert module.torchparameters["param_1"].from_0to1() == 10000.0

        # Passing a value outside of range should fail
        with pytest.raises(AssertionError):
            module.set_parameter_0to1("param_1", 5.0)

    def test_p(self):
        module = torchmodule.TorchSynthModule()
        param_1 = TorchParameter(
            value=5000.0,
            parameter_range=ParameterRange(0.0, 20000.0),
            parameter_name="param_1"
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25
        assert module.p("param_1") == 5000.0


class TestTorchSynth():
    """
    Tests for TorchSynth
    """

    def test_construction(self):
        # Test empty construction
        synth = torchmodule.TorchSynth()
        assert synth.sample_rate == defaults.SAMPLE_RATE
        assert synth.buffer_size == defaults.BUFFER_SIZE

        # Test construction with args
        synth = torchmodule.TorchSynth(sample_rate=16000, buffer_size=512)
        assert synth.sample_rate == 16000
        assert synth.buffer_size == 512

    def test_add_synth_module(self):

        synth = torchmodule.TorchSynth()
        vco = torchmodule.TorchSineVCO()
        noise = torchmodule.TorchNoise()

        synth.add_synth_modules({
            "vco": vco,
            "noise": noise
        })
        assert hasattr(synth, "vco")
        assert hasattr(synth, "noise")

        # Make sure all the parameters were registered correctly
        synth_params = [p for p in synth.parameters()]
        module_params = [p for p in vco.parameters()]
        module_params.extend([p for p in noise.parameters()])
        for p in module_params:
            assert p in synth_params

        # Expect a TypeError if a non TorchSynthModule is passed in
        with pytest.raises(TypeError):
            synth.add_synth_modules({
                'module': torch.nn.Module()
            })

        # Expect a ValueError if the incorrect sample rate or buffer size is passed in
        with pytest.raises(ValueError):
            vco_2 = torchmodule.TorchSineVCO(sample_rate=16000)
            synth.add_synth_modules({
                'vco_2': vco_2
            })

        with pytest.raises(ValueError):
            adsr = torchmodule.TorchADSR(buffer_size=512)
            synth.add_synth_modules({
                'adsr': adsr
            })
