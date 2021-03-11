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
            midi_f0=T([12.0, 30.0]),
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
                midi_f0=T([12.0, 30.0]),
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
