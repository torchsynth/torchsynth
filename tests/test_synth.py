"""
Tests for torch synths
"""

import os
import json
import pytest
import torch.nn
import torch.tensor as tensor

import torchsynth.config
import torchsynth.module as synthmodule
import torchsynth.parameter
import torchsynth.synth
from torchsynth.parameter import ModuleParameter


class TestAbstractSynth:
    """
    Tests for AbstractSynth
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def test_construction(self):
        # Test empty construction
        synthconfig = torchsynth.config.SynthConfig(
            sample_rate=16000,
            buffer_size_seconds=0.3,
            batch_size=2,
            reproducible=False,
        )
        # Test construction with args
        synth = torchsynth.synth.AbstractSynth(synthconfig)
        assert synth.sample_rate == 16000
        assert synth.buffer_size_seconds == 0.3

    def test_add_synth_module(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        synth = torchsynth.synth.AbstractSynth(synthconfig).to(self.device)
        synth.add_synth_modules(
            [
                (
                    "vco",
                    synthmodule.SineVCO,
                    {
                        "tuning": tensor([-12.0, 3.0]),
                        "mod_depth": tensor([50.0, -50.0]),
                    },
                ),
                ("noise", synthmodule.Noise, {"seed": 42}),
            ]
        )

        assert hasattr(synth, "vco")
        assert hasattr(synth, "noise")

        # Make sure all the ModuleParameters were registered correctly
        synth_params = [p for p in synth.parameters() if isinstance(p, ModuleParameter)]
        module_params = [
            p for p in synth.vco.parameters() if isinstance(p, ModuleParameter)
        ]
        module_params.extend(
            [p for p in synth.noise.parameters() if isinstance(p, ModuleParameter)]
        )
        for p in module_params:
            fails = True
            for p2 in synth_params:
                if p.parameter_name == p2.parameter_name and torch.all(
                    p.data == p2.data
                ):
                    fails = False
            assert not fails

        # Make sure that all the SynthModules and params are on the correct device now
        for module in synth.modules():
            if not isinstance(module, synthmodule.SynthModule):
                continue

            assert module.device.type == self.device
            for parameter in module.parameters():
                assert parameter.device.type == self.device

        # Expect a TypeError if a non SynthModule is passed in
        with pytest.raises(TypeError):
            synth.add_synth_modules([("module", torch.nn.Module)])

    def test_deterministic_noise(self):
        # This test confirms that randomizing a synth with the same
        # seed results in the same audio results.

        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        cpusynth = torchsynth.synth.Voice(synthconfig)
        x11 = torchsynth.synth.Voice(synthconfig)
        x11 = x11(1)
        x2 = torchsynth.synth.Voice(synthconfig)
        x2 = x2(2)
        x12 = torchsynth.synth.Voice(synthconfig)
        x12 = x12(1)

        assert torch.mean(torch.abs(x11 - x2)) > 1e-6
        assert torch.mean(torch.abs(x11 - x12)) < 1e-6

        # If a GPU is available then make sure the same
        # tests as above are also deterministic on the GPU.
        if self.device == "cuda":
            cudasynth = torchsynth.synth.Voice(synthconfig).to(self.device)

            # Confirm that randomizing the cpu synth and cuda synth results
            # in the same set of parameters
            cudasynth.randomize(42)
            cpusynth.randomize(42)
            cuda_params = cudasynth.get_parameters()
            cpu_params = cpusynth.get_parameters()
            for name, param in cuda_params.items():
                assert torch.all(param.data.detach().cpu() == cpu_params[name].data)

            # Confirm that we get deterministic results when
            # randomizing the cuda synth with the same seed
            cuda11 = torchsynth.synth.Voice(synthconfig).to(self.device)
            cuda11 = cuda11(1)
            cuda2 = torchsynth.synth.Voice(synthconfig).to(self.device)
            cuda2 = cuda2(2)
            cuda12 = torchsynth.synth.Voice(synthconfig).to(self.device)
            cuda12 = cuda12(1)

            assert torch.mean(torch.abs(cuda11 - cuda2)) > 1e-6
            assert torch.mean(torch.abs(cuda11 - cuda12)) < 1e-6

            # Finally, compare the output audio from the cuda synth to the
            # cpu synth. Small numerical differences between computations of
            # GPU and CPU add up, so we need to relax the constraints here.
            # TODO https://github.com/torchsynth/torchsynth/issues/256
            assert torch.mean(torch.abs(cuda11.detach().cpu() - x11)) < 2e-1
            assert torch.mean(torch.abs(cuda2.detach().cpu() - x2)) < 2e-1

    def test_parameter_randomization(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        cpusynth1 = torchsynth.synth.Voice(synthconfig)
        cpusynth2 = torchsynth.synth.Voice(synthconfig)

        cpusynth1.randomize(1)
        cpusynth2.randomize(1)

        params_1 = cpusynth1.get_parameters()
        params_2 = cpusynth2.get_parameters()
        for name, param in params_1.items():
            assert torch.all(param.data.detach().cpu() == params_2[name].data)

    def test_randomize_parameter_freezing(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        synth = torchsynth.synth.Voice(synthconfig)

        midi_val = tensor([69.0, 40.0])
        dur_val = tensor([0.25, 3.0])

        # Tests the freezing parameters with current value
        synth.set_parameters(
            {
                ("keyboard", "midi_f0"): midi_val,
                ("keyboard", "duration"): dur_val,
            }
        )
        synth.freeze_parameters(
            [
                ("keyboard", "midi_f0"),
                ("keyboard", "duration"),
            ]
        )
        synth.randomize()
        assert torch.all(synth.keyboard.p("midi_f0").isclose(midi_val))
        assert torch.all(synth.keyboard.p("duration").isclose(dur_val))

        synth.randomize(1)
        assert torch.all(synth.keyboard.p("midi_f0").isclose(midi_val))
        assert torch.all(synth.keyboard.p("duration").isclose(dur_val))

        # Test that trying to set a frozen parameter raises an error
        with pytest.raises(RuntimeError, match="Parameter is frozen"):
            synth.set_parameters(
                {
                    ("keyboard", "midi_f0"): midi_val,
                    ("keyboard", "duration"): dur_val,
                }
            )

        # Unfreezing parameters and randomizing now leads to different results
        synth.unfreeze_all_parameters()
        synth.randomize(1)
        assert torch.all(~synth.keyboard.p("midi_f0").isclose(midi_val))
        assert torch.all(~synth.keyboard.p("duration").isclose(dur_val))

        # Can set parameters directly with freeze arg that they should be frozen
        synth.set_parameters(
            {
                ("keyboard", "midi_f0"): midi_val,
                ("keyboard", "duration"): dur_val,
            },
            freeze=True,
        )
        assert torch.all(synth.keyboard.p("midi_f0").isclose(midi_val))
        assert torch.all(synth.keyboard.p("duration").isclose(dur_val))

        synth.randomize()
        assert torch.all(synth.keyboard.p("midi_f0").isclose(midi_val))
        assert torch.all(synth.keyboard.p("duration").isclose(dur_val))

        synth.randomize(1)
        assert torch.all(synth.keyboard.p("midi_f0").isclose(midi_val))
        assert torch.all(synth.keyboard.p("duration").isclose(dur_val))

        # Test randomization with synth with non-ModuleParameters raises error
        synth.register_parameter("param", torch.nn.Parameter(tensor(0.0)))
        with pytest.raises(ValueError, match="Param 0.0 is not a ModuleParameter"):
            synth.randomize()

        with pytest.raises(ValueError, match="Param 0.0 is not a ModuleParameter"):
            synth.randomize(1)

    def test_freeze_parameters(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        synth = torchsynth.synth.Voice(synthconfig)

        for param in synth.parameters():
            if isinstance(param, ModuleParameter):
                assert not param.frozen

        synth.freeze_parameters(
            [
                ("keyboard", "midi_f0"),
                ("keyboard", "duration"),
            ]
        )

        # Make sure the correct params are frozen now
        for name, param in synth.named_parameters():
            if isinstance(param, ModuleParameter):
                if param.parameter_name in ["midi_f0", "duration"]:
                    assert param.frozen
                else:
                    assert not param.frozen

        # Now unfreeze all of them
        synth.unfreeze_all_parameters()
        for param in synth.parameters():
            if isinstance(param, ModuleParameter):
                assert not param.frozen

    def test_set_frozen_parameters(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        synth = torchsynth.synth.Voice(synthconfig)

        synth.set_parameters(
            {
                ("vco_1", "tuning"): tensor([0.0] * synthconfig.batch_size),
                ("adsr_1", "attack"): tensor([1.5] * synthconfig.batch_size),
            },
            freeze=True,
        )

        # Parameters should have been set with correct batch size
        assert synth.vco_1.p("tuning").shape == (synth.batch_size,)
        assert torch.all(synth.vco_1.p("tuning").eq(tensor([0.0, 0.0])))

        assert synth.adsr_1.p("attack").shape == (synth.batch_size,)
        assert torch.all(synth.adsr_1.p("attack").eq(tensor([1.5, 1.5])))

        # Randomizing now shouldn't effect these parameters
        synth.randomize()
        assert torch.all(synth.vco_1.p("tuning").eq(tensor([0.0, 0.0])))
        assert torch.all(synth.adsr_1.p("attack").eq(tensor([1.5, 1.5])))

        synth.randomize(1)
        assert torch.all(synth.vco_1.p("tuning").eq(tensor([0.0, 0.0])))
        assert torch.all(synth.adsr_1.p("attack").eq(tensor([1.5, 1.5])))

    def test_get_parameters(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        synth = torchsynth.synth.Voice(synthconfig)
        params = synth.get_parameters()
        assert len(params) > 0
        for param in params:
            assert hasattr(synth, param[0])
            module = getattr(synth, param[0])
            parameter = module.get_parameter(param[1])
            assert isinstance(parameter, ModuleParameter)

        # Now make sure that frozen parameters don't get returned unless specified
        synth.freeze_parameters([("keyboard", "midi_f0")])
        assert ("keyboard", "midi_f0") not in synth.get_parameters()
        assert ("keyboard", "midi_f0") in synth.get_parameters(include_frozen=True)

        # Now unfreeze and make sure they are returned
        synth.unfreeze_all_parameters()
        assert ("keyboard", "midi_f0") in synth.get_parameters()

    def test_set_hyperparameters(self):
        synthconfig = torchsynth.config.SynthConfig(batch_size=2, reproducible=False)
        synth = torchsynth.synth.Voice(synthconfig)
        hparams = synth.hyperparameters
        for (module_name, param_name, subname), value in hparams.items():
            if subname == "curve":
                value = 1.0 - value
            if subname == "symmetric":
                value = not value
            synth.set_hyperparameter((module_name, param_name, subname), value)
        hparams2 = synth.hyperparameters
        for (module_name, param_name, subname), value in hparams2.items():
            if subname == "curve":
                assert value == 1.0 - hparams[(module_name, param_name, subname)]
            if subname == "symmetric":
                assert value == (not hparams[(module_name, param_name, subname)])

    def test_saving_hyperparameters(self, tmp_path):
        synth = torchsynth.synth.Voice()

        # Save current voice hyperparameters as json
        filename = os.path.join(tmp_path, "hyperparams.json")
        synth.save_hyperparameters(filename)

        # Load saved hyperparams from json and make sure they match
        hyperparameters = synth.hyperparameters
        with open(filename, "r") as fp:
            data = json.load(fp)
            for hp in data:
                assert hyperparameters[tuple(hp["name"])] == hp["value"]

        # Update a couple hyperparameters
        synth.set_hyperparameter(("adsr_1", "attack", "curve"), 100.0)
        synth.set_hyperparameter(("keyboard", "duration", "symmetric"), True)

        # Save updated hyperparams and reload the default
        filename2 = os.path.join(tmp_path, "hyperparams2.json")
        synth.save_hyperparameters(filename2)
        synth.load_hyperparameters("default")

        assert synth.hyperparameters[("adsr_1", "attack", "curve")] != 100.0
        assert not synth.hyperparameters[("keyboard", "duration", "symmetric")]

        # Now load the saved updated hyperparams and confirm the updated values
        synth.load_hyperparameters(filename2)
        assert synth.hyperparameters[("adsr_1", "attack", "curve")] == 100.0
        assert synth.hyperparameters[("keyboard", "duration", "symmetric")]
