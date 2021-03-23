"""
Tests for torch synths
"""


import pytest
import torch.nn
import torch.tensor as T

import torchsynth.globals
import torchsynth.module as synthmodule
import torchsynth.parameter
import torchsynth.synth
from torchsynth.parameter import ModuleParameter


class TestAbstractSynth:
    """
    Tests for AbstractSynth
    """

    def test_construction(self):
        # Test empty construction
        synthglobals = torchsynth.globals.SynthGlobals(
            sample_rate=T(16000), buffer_size=T(512), batch_size=T(2)
        )
        # Test construction with args
        synth = torchsynth.synth.AbstractSynth(synthglobals)
        assert synth.sample_rate == T(16000)
        assert synth.buffer_size == T(512)

    def test_add_synth_module(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        synth = torchsynth.synth.AbstractSynth(synthglobals)
        vco = synthmodule.SineVCO(
            tuning=T([-12.0, 3.0]),
            mod_depth=T([50.0, -50.0]),
            synthglobals=synthglobals,
        )
        noise = synthmodule.Noise(ratio=T([0.25, 0.75]), synthglobals=synthglobals)

        synth.add_synth_modules([("vco", vco), ("noise", noise)])
        assert hasattr(synth, "vco")
        assert hasattr(synth, "noise")

        # Make sure all the ModuleParameters were registered correctly
        synth_params = [p for p in synth.parameters() if isinstance(p, ModuleParameter)]
        module_params = [p for p in vco.parameters() if isinstance(p, ModuleParameter)]
        module_params.extend(
            [p for p in noise.parameters() if isinstance(p, ModuleParameter)]
        )
        for p in module_params:
            fails = True
            for p2 in synth_params:
                if p.parameter_name == p2.parameter_name and torch.all(
                    p.data == p2.data
                ):
                    fails = False
            assert not fails

        # Expect a TypeError if a non SynthModule0Ddeprecated is passed in
        with pytest.raises(TypeError):
            synth.add_synth_modules([("module", torch.nn.Module())])

        # Expect a ValueError if the incorrect sample rate or buffer size is passed in
        with pytest.raises(ValueError):
            synthglobals_weird_sr = torchsynth.globals.SynthGlobals(
                batch_size=T(2), sample_rate=T(16000)
            )
            vco_2 = synthmodule.SineVCO(
                tuning=T([12.0, -5.0]),
                mod_depth=T([50.0, 50.0]),
                synthglobals=synthglobals_weird_sr,
            )
            synth.add_synth_modules([("vco_2", vco_2)])

        # This should raise an assertion because it has a different batch size than
        # the other modules
        with pytest.raises(ValueError):
            synthglobals_new_batchsize = torchsynth.globals.SynthGlobals(
                batch_size=T(1)
            )
            adsr = synthmodule.ADSR(
                attack=T([0.5]),
                decay=T([0.25]),
                sustain=T([0.5]),
                release=T([1.0]),
                alpha=T([1.0]),
                synthglobals=synthglobals_new_batchsize,
            )
            synth.add_synth_modules([("adsr", adsr)])

        # Same here
        with pytest.raises(ValueError):
            synthglobals_new_batchsize = torchsynth.globals.SynthGlobals(
                batch_size=T(1)
            )
            adsr = synthmodule.ADSR(
                synthglobals=synthglobals_new_batchsize,
            )
            synth.add_synth_modules([("adsr", adsr)])

        # Same here
        with pytest.raises(ValueError):
            synthglobals_new_buffersize = torchsynth.globals.SynthGlobals(
                batch_size=T(2), buffer_size=T(2048)
            )
            adsr = synthmodule.ADSR(
                synthglobals=synthglobals_new_buffersize,
            )
            synth.add_synth_modules([("adsr", adsr)])

    def test_deterministic_noise(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        synth = torchsynth.synth.Voice(synthglobals)

        synth.randomize(1)
        x11 = synth()
        synth.randomize()
        x2 = synth()
        synth.randomize(1)
        x12 = synth()

        assert torch.mean(torch.abs(x11 - x2)) > 1e-6
        assert torch.mean(torch.abs(x11 - x12)) < 1e-6

    def test_randomize_parameter_freezing(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        synth = torchsynth.synth.Voice(synthglobals)

        midi_val = T([69.0, 40.0])
        dur_val = T([0.25, 3.0])

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
        assert torch.all(synth.keyboard.p("midi_f0").eq(midi_val))
        assert torch.all(synth.keyboard.p("duration").eq(dur_val))

        synth.randomize(1)
        assert torch.all(synth.keyboard.p("midi_f0").eq(midi_val))
        assert torch.all(synth.keyboard.p("duration").eq(dur_val))

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
        assert torch.all(~synth.keyboard.p("midi_f0").eq(midi_val))
        assert torch.all(~synth.keyboard.p("duration").eq(dur_val))

        # Can set parameters directly with freeze arg that they should be frozen
        synth.set_parameters(
            {
                ("keyboard", "midi_f0"): midi_val,
                ("keyboard", "duration"): dur_val,
            },
            freeze=True,
        )
        assert torch.all(synth.keyboard.p("midi_f0").eq(midi_val))
        assert torch.all(synth.keyboard.p("duration").eq(dur_val))

        synth.randomize()
        assert torch.all(synth.keyboard.p("midi_f0").eq(midi_val))
        assert torch.all(synth.keyboard.p("duration").eq(dur_val))

        synth.randomize(1)
        assert torch.all(synth.keyboard.p("midi_f0").eq(midi_val))
        assert torch.all(synth.keyboard.p("duration").eq(dur_val))

        # Test randomization with synth with non-ModuleParameters raises error
        synth.register_parameter("param", torch.nn.Parameter(T(0.0)))
        with pytest.raises(ValueError, match="Param 0.0 is not a ModuleParameter"):
            synth.randomize()

        with pytest.raises(ValueError, match="Param 0.0 is not a ModuleParameter"):
            synth.randomize(1)

    def test_freeze_parameters(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        synth = torchsynth.synth.Voice(synthglobals)

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
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        synth = torchsynth.synth.Voice(synthglobals)

        synth.set_frozen_parameters(
            {
                ("vco_1", "tuning"): 0.0,
                ("adsr_1", "attack"): 1.5,
            }
        )

        # Parameters should have been set with correct batch size
        assert synth.vco_1.p("tuning").shape == (synth.batch_size,)
        assert torch.all(synth.vco_1.p("tuning").eq(T([0.0, 0.0])))

        assert synth.adsr_1.p("attack").shape == (synth.batch_size,)
        assert torch.all(synth.adsr_1.p("attack").eq(T([1.5, 1.5])))

        # Randomizing now shouldn't effect these parameters
        synth.randomize()
        assert torch.all(synth.vco_1.p("tuning").eq(T([0.0, 0.0])))
        assert torch.all(synth.adsr_1.p("attack").eq(T([1.5, 1.5])))

        synth.randomize(1)
        assert torch.all(synth.vco_1.p("tuning").eq(T([0.0, 0.0])))
        assert torch.all(synth.adsr_1.p("attack").eq(T([1.5, 1.5])))

    def test_set_hyperparameters(self):
        synthglobals = torchsynth.globals.SynthGlobals(batch_size=T(2))
        synth = torchsynth.synth.Voice(synthglobals)
        hparams = synth.hyperparameters
        for (name, subname), value in hparams.items():
            if subname == "curve":
                value = 1.0 - value
            if subname == "symmetric":
                value = not value
            synth.set_hyperparameter(name, subname, value)
        hparams2 = synth.hyperparameters
        for (name, subname), value in hparams2.items():
            if subname == "curve":
                assert value == 1.0 - hparams[(name, subname)]
            if subname == "symmetric":
                assert value == (not hparams[(name, subname)])
