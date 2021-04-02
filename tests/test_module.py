"""
Tests for torch synth modules.
"""

import pytest
import torch
import torch.tensor as T

import torchsynth
from torchsynth.globals import SynthGlobals
import torchsynth.module as synthmodule
from torchsynth.parameter import ModuleParameter, ModuleParameterRange


class TestSynthModule:
    """
    Tests for SynthModules
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Disabling tests that relied upon deprecated synth
    # module, but we can add them back later for SynthModule.
    """
    def test_get_parameter(self):
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
        param_1 = ModuleParameter(data=T(1.0), parameter_name="param_1")
        module.add_parameters([param_1])
        assert module.get_parameter("param_1") == param_1

    def test_set_parameter(self):
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25

        module.set_parameter("param_1", 10000.0)
        assert module.torchparameters["param_1"] == 0.5
        assert module.torchparameters["param_1"].from_0to1() == 10000.0

        with pytest.raises(AssertionError):
            module.set_parameter_0to1("param_1", -100.0)

    def test_set_parameter_0to1(self):
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
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
        module = torchsynth.deprecated.SynthModule0Ddeprecated()
        param_1 = ModuleParameter(
            value=T(5000.0),
            parameter_range=ModuleParameterRange(0.0, 20000.0),
            parameter_name="param_1",
        )
        module.add_parameters([param_1])
        assert module.torchparameters["param_1"] == 0.25
        assert module.p("param_1") == 5000.0
    """

    def test_set_parameter(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        module = synthmodule.ADSR(synthglobals, attack=T([0.5, 1.0]))

        # Confirm value set correctly from constructor
        assert torch.all(module.p("attack") == T([0.5, 1.0]))

        # Confirm value set correctly from 0to1 range
        module.set_parameter_0to1("attack", T([0.33, 0.25]))
        assert torch.all(module.get_parameter_0to1("attack") == T([0.33, 0.25]))

        # Mode module to device (GPU if available) and make sure parameters have moved
        module.to(self.device)
        for parameter in module.torchparameters.values():
            assert parameter.device.type == self.device

    def test_softmodeselector(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        mode_selector = synthmodule.SoftModeSelector(
            synthglobals, device=self.device, n_modes=3
        )
        mode_selector.set_parameter("mode0weight", T([0.8, 1.0]))
        mode_selector.set_parameter("mode1weight", T([0.8, 0.0]))
        mode_selector.set_parameter("mode2weight", T([0.8, 0.0]))
        assert (
            torch.mean(
                mode_selector()
                - T(
                    [[1 / 3, 1.0000], [1 / 3, 0.0000], [1 / 3, 0.0000]],
                    device=self.device,
                )
            )
            < 1e-6
        )

    def test_hardmodeselector(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        mode_selector = synthmodule.HardModeSelector(
            synthglobals, device=self.device, n_modes=3
        )
        mode_selector.set_parameter("mode0weight", T([0.8, 0.0]))
        mode_selector.set_parameter("mode1weight", T([0.7, 0.5]))
        mode_selector.set_parameter("mode2weight", T([0.7, 0.0]))
        assert (
            torch.mean(
                mode_selector()
                - T([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], device=self.device)
            )
            < 1e-6
        )

    def test_audiomixer(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))

        # Make sure parameters get setup correctly
        mixer = synthmodule.AudioMixer(synthglobals, device=self.device, n_input=3)
        params = [p for p in mixer.parameters()]
        assert len(params) == 3
        for param in params:
            assert param.parameter_range.curve == 1.0

        mixer = synthmodule.AudioMixer(
            synthglobals, device=self.device, n_input=2, curves=[0.75, 1.5]
        )
        params = [p for p in mixer.parameters()]
        assert len(params) == 2
        assert params[0].parameter_range.curve == 0.75
        assert params[1].parameter_range.curve == 1.5

        # if curves are passed in then the number of curves must equal n_input
        with pytest.raises(AssertionError):
            mixer = synthmodule.AudioMixer(
                synthglobals, device=self.device, n_input=3, curves=[0.75, 1.5]
            )

    def test_modulationmixer(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))

        mixer = synthmodule.ModulationMixer(
            synthglobals, device=self.device, n_input=2, n_output=2
        )
        params = [p for p in mixer.parameters()]
        assert len(params) == 4
        for param in params:
            assert param.parameter_range.curve == 0.5

        mixer = synthmodule.ModulationMixer(
            synthglobals, device=self.device, n_input=1, n_output=2, curves=[1.0]
        )
        params = [p for p in mixer.parameters()]
        assert len(params) == 2
        for param in params:
            assert param.parameter_range.curve == 1.0

        # if curves are passed in then the number of curves must equal n_input
        with pytest.raises(AssertionError):
            mixer = synthmodule.AudioMixer(
                synthglobals,
                device=self.device,
                n_input=5,
                n_output=5,
                curves=[0.75, 1.5],
            )

    def test_noise(self):
        # Here we create there noise modules with different batch sizes.
        # We each noise module a number of times to obtain an equal number
        # of noise signals. All these noise samples should equal each other.
        # i.e., noise should be returned deterministically regardless of the
        # batch size.
        synthglobals32 = SynthGlobals(T(32))
        synthglobals64 = SynthGlobals(T(64))
        synthglobals128 = SynthGlobals(T(128))

        noise32 = synthmodule.Noise(synthglobals32, seed=0)
        noise64 = synthmodule.Noise(synthglobals64, seed=0)
        noise128 = synthmodule.Noise(synthglobals128, seed=0)
        # A different seed should give a different result
        noise128_diff = synthmodule.Noise(synthglobals128, seed=42)

        out1 = torch.vstack((noise32(), noise32(), noise32(), noise32()))
        out2 = torch.vstack((noise64(), noise64()))
        out3 = noise128()
        out4 = noise128_diff()

        assert torch.all(out1 == out2)
        assert torch.all(out2 == out3)
        assert torch.all(out3 != out4)

        with pytest.raises(ValueError):
            # If the batch size if larger than the default
            # of 64, then this should complain
            synthglobals65 = SynthGlobals(T(65))
            synthmodule.Noise(synthglobals65, seed=0)
