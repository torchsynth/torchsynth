#!/usr/bin/env python3
# # explore-nebula
#
# Explore a nebula
# If you want to freeze the MIDI_F0, otherwise use None
# MIDI_F0 = 48
MIDI_F0 = 69
BATCH_SIZE = 64


import hashlib
import json
import os
import os.path
import pprint
import sys

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
import torch.tensor as tensor
from torch import Tensor as T
from torchsynth.config import SynthConfig
from torchsynth.synth import Voice
from tqdm.auto import tqdm

# Bump this each time you run this script, to get different sounds
SALT = 20

default_nebula = {
    ("adsr_1", "attack", "curve"): 0.5,
    ("adsr_1", "attack", "symmetric"): False,
    ("adsr_1", "decay", "curve"): 0.5,
    ("adsr_1", "decay", "symmetric"): False,
    ("adsr_1", "sustain", "curve"): 1,
    ("adsr_1", "sustain", "symmetric"): False,
    ("adsr_1", "release", "curve"): 0.5,
    ("adsr_1", "release", "symmetric"): False,
    ("adsr_1", "alpha", "curve"): 1,
    ("adsr_1", "alpha", "symmetric"): False,
    ("adsr_2", "attack", "curve"): 0.5,
    ("adsr_2", "attack", "symmetric"): False,
    ("adsr_2", "decay", "curve"): 0.5,
    ("adsr_2", "decay", "symmetric"): False,
    ("adsr_2", "sustain", "curve"): 1,
    ("adsr_2", "sustain", "symmetric"): False,
    ("adsr_2", "release", "curve"): 0.5,
    ("adsr_2", "release", "symmetric"): False,
    ("adsr_2", "alpha", "curve"): 1,
    ("adsr_2", "alpha", "symmetric"): False,
    ("keyboard", "midi_f0", "curve"): 1.0,
    ("keyboard", "midi_f0", "symmetric"): False,
    ("keyboard", "duration", "curve"): 0.5,
    ("keyboard", "duration", "symmetric"): False,
    ("lfo_1", "frequency", "curve"): 0.25,
    ("lfo_1", "frequency", "symmetric"): False,
    ("lfo_1", "mod_depth", "curve"): 0.5,
    ("lfo_1", "mod_depth", "symmetric"): True,
    ("lfo_1", "initial_phase", "curve"): 1,
    ("lfo_1", "initial_phase", "symmetric"): False,
    ("lfo_1", "sin", "curve"): 1,
    ("lfo_1", "sin", "symmetric"): False,
    ("lfo_1", "tri", "curve"): 1,
    ("lfo_1", "tri", "symmetric"): False,
    ("lfo_1", "saw", "curve"): 1,
    ("lfo_1", "saw", "symmetric"): False,
    ("lfo_1", "rsaw", "curve"): 1,
    ("lfo_1", "rsaw", "symmetric"): False,
    ("lfo_1", "sqr", "curve"): 1,
    ("lfo_1", "sqr", "symmetric"): False,
    ("lfo_1_amp_adsr", "attack", "curve"): 0.5,
    ("lfo_1_amp_adsr", "attack", "symmetric"): False,
    ("lfo_1_amp_adsr", "decay", "curve"): 0.5,
    ("lfo_1_amp_adsr", "decay", "symmetric"): False,
    ("lfo_1_amp_adsr", "sustain", "curve"): 1,
    ("lfo_1_amp_adsr", "sustain", "symmetric"): False,
    ("lfo_1_amp_adsr", "release", "curve"): 0.5,
    ("lfo_1_amp_adsr", "release", "symmetric"): False,
    ("lfo_1_amp_adsr", "alpha", "curve"): 1,
    ("lfo_1_amp_adsr", "alpha", "symmetric"): False,
    ("lfo_1_rate_adsr", "attack", "curve"): 0.5,
    ("lfo_1_rate_adsr", "attack", "symmetric"): False,
    ("lfo_1_rate_adsr", "decay", "curve"): 0.5,
    ("lfo_1_rate_adsr", "decay", "symmetric"): False,
    ("lfo_1_rate_adsr", "sustain", "curve"): 1,
    ("lfo_1_rate_adsr", "sustain", "symmetric"): False,
    ("lfo_1_rate_adsr", "release", "curve"): 0.5,
    ("lfo_1_rate_adsr", "release", "symmetric"): False,
    ("lfo_1_rate_adsr", "alpha", "curve"): 1,
    ("lfo_1_rate_adsr", "alpha", "symmetric"): False,
    ("lfo_2", "frequency", "curve"): 0.25,
    ("lfo_2", "frequency", "symmetric"): False,
    ("lfo_2", "mod_depth", "curve"): 0.5,
    ("lfo_2", "mod_depth", "symmetric"): True,
    ("lfo_2", "initial_phase", "curve"): 1,
    ("lfo_2", "initial_phase", "symmetric"): False,
    ("lfo_2", "sin", "curve"): 1,
    ("lfo_2", "sin", "symmetric"): False,
    ("lfo_2", "tri", "curve"): 1,
    ("lfo_2", "tri", "symmetric"): False,
    ("lfo_2", "saw", "curve"): 1,
    ("lfo_2", "saw", "symmetric"): False,
    ("lfo_2", "rsaw", "curve"): 1,
    ("lfo_2", "rsaw", "symmetric"): False,
    ("lfo_2", "sqr", "curve"): 1,
    ("lfo_2", "sqr", "symmetric"): False,
    ("lfo_2_amp_adsr", "attack", "curve"): 0.5,
    ("lfo_2_amp_adsr", "attack", "symmetric"): False,
    ("lfo_2_amp_adsr", "decay", "curve"): 0.5,
    ("lfo_2_amp_adsr", "decay", "symmetric"): False,
    ("lfo_2_amp_adsr", "sustain", "curve"): 1,
    ("lfo_2_amp_adsr", "sustain", "symmetric"): False,
    ("lfo_2_amp_adsr", "release", "curve"): 0.5,
    ("lfo_2_amp_adsr", "release", "symmetric"): False,
    ("lfo_2_amp_adsr", "alpha", "curve"): 1,
    ("lfo_2_amp_adsr", "alpha", "symmetric"): False,
    ("lfo_2_rate_adsr", "attack", "curve"): 0.5,
    ("lfo_2_rate_adsr", "attack", "symmetric"): False,
    ("lfo_2_rate_adsr", "decay", "curve"): 0.5,
    ("lfo_2_rate_adsr", "decay", "symmetric"): False,
    ("lfo_2_rate_adsr", "sustain", "curve"): 1,
    ("lfo_2_rate_adsr", "sustain", "symmetric"): False,
    ("lfo_2_rate_adsr", "release", "curve"): 0.5,
    ("lfo_2_rate_adsr", "release", "symmetric"): False,
    ("lfo_2_rate_adsr", "alpha", "curve"): 1,
    ("lfo_2_rate_adsr", "alpha", "symmetric"): False,
    ("mixer", "level0", "curve"): 1.0,
    ("mixer", "level0", "symmetric"): False,
    ("mixer", "level1", "curve"): 1.0,
    ("mixer", "level1", "symmetric"): False,
    ("mixer", "level2", "curve"): 0.1,
    ("mixer", "level2", "symmetric"): False,
    ("mod_matrix", "level0_0", "curve"): 0.5,
    ("mod_matrix", "level0_0", "symmetric"): False,
    ("mod_matrix", "level0_1", "curve"): 0.5,
    ("mod_matrix", "level0_1", "symmetric"): False,
    ("mod_matrix", "level0_2", "curve"): 0.5,
    ("mod_matrix", "level0_2", "symmetric"): False,
    ("mod_matrix", "level0_3", "curve"): 0.5,
    ("mod_matrix", "level0_3", "symmetric"): False,
    ("mod_matrix", "level0_4", "curve"): 0.5,
    ("mod_matrix", "level0_4", "symmetric"): False,
    ("mod_matrix", "level1_0", "curve"): 0.5,
    ("mod_matrix", "level1_0", "symmetric"): False,
    ("mod_matrix", "level1_1", "curve"): 0.5,
    ("mod_matrix", "level1_1", "symmetric"): False,
    ("mod_matrix", "level1_2", "curve"): 0.5,
    ("mod_matrix", "level1_2", "symmetric"): False,
    ("mod_matrix", "level1_3", "curve"): 0.5,
    ("mod_matrix", "level1_3", "symmetric"): False,
    ("mod_matrix", "level1_4", "curve"): 0.5,
    ("mod_matrix", "level1_4", "symmetric"): False,
    ("mod_matrix", "level2_0", "curve"): 0.5,
    ("mod_matrix", "level2_0", "symmetric"): False,
    ("mod_matrix", "level2_1", "curve"): 0.5,
    ("mod_matrix", "level2_1", "symmetric"): False,
    ("mod_matrix", "level2_2", "curve"): 0.5,
    ("mod_matrix", "level2_2", "symmetric"): False,
    ("mod_matrix", "level2_3", "curve"): 0.5,
    ("mod_matrix", "level2_3", "symmetric"): False,
    ("mod_matrix", "level2_4", "curve"): 0.5,
    ("mod_matrix", "level2_4", "symmetric"): False,
    ("mod_matrix", "level3_0", "curve"): 0.5,
    ("mod_matrix", "level3_0", "symmetric"): False,
    ("mod_matrix", "level3_1", "curve"): 0.5,
    ("mod_matrix", "level3_1", "symmetric"): False,
    ("mod_matrix", "level3_2", "curve"): 0.5,
    ("mod_matrix", "level3_2", "symmetric"): False,
    ("mod_matrix", "level3_3", "curve"): 0.5,
    ("mod_matrix", "level3_3", "symmetric"): False,
    ("mod_matrix", "level3_4", "curve"): 0.5,
    ("mod_matrix", "level3_4", "symmetric"): False,
    ("vco_1", "tuning", "curve"): 1,
    ("vco_1", "tuning", "symmetric"): False,
    ("vco_1", "mod_depth", "curve"): 0.2,
    ("vco_1", "mod_depth", "symmetric"): True,
    ("vco_1", "initial_phase", "curve"): 1,
    ("vco_1", "initial_phase", "symmetric"): False,
    ("vco_2", "tuning", "curve"): 1,
    ("vco_2", "tuning", "symmetric"): False,
    ("vco_2", "mod_depth", "curve"): 0.2,
    ("vco_2", "mod_depth", "symmetric"): True,
    ("vco_2", "initial_phase", "curve"): 1,
    ("vco_2", "initial_phase", "symmetric"): False,
    ("vco_2", "shape", "curve"): 1,
    ("vco_2", "shape", "symmetric"): False,
}

proposed_nebula = {
    ("adsr_1", "alpha", "curve"): 1.9961787803060782,
    ("adsr_1", "alpha", "symmetric"): 1,
    ("adsr_1", "attack", "curve"): 0.46250555022122,
    ("adsr_1", "attack", "symmetric"): 1,
    ("adsr_1", "decay", "curve"): 11.695660400451677,
    ("adsr_1", "decay", "symmetric"): 1,
    ("adsr_1", "release", "curve"): 0.07049607051161247,
    ("adsr_1", "release", "symmetric"): 0,
    ("adsr_1", "sustain", "curve"): 0.12907542804040373,
    ("adsr_1", "sustain", "symmetric"): 1,
    ("adsr_2", "alpha", "curve"): 0.30604871589870714,
    ("adsr_2", "alpha", "symmetric"): 0,
    ("adsr_2", "attack", "curve"): 0.8217191087210552,
    ("adsr_2", "attack", "symmetric"): 0,
    ("adsr_2", "decay", "curve"): 0.7777376001398456,
    ("adsr_2", "decay", "symmetric"): 1,
    ("adsr_2", "release", "curve"): 6.376201993305432,
    ("adsr_2", "release", "symmetric"): 1,
    ("adsr_2", "sustain", "curve"): 0.25011399134154866,
    ("adsr_2", "sustain", "symmetric"): 1,
    ("keyboard", "duration", "curve"): 0.015195215527109649,
    ("keyboard", "duration", "symmetric"): 0,
    ("lfo_1", "frequency", "curve"): 2.598422643041545,
    ("lfo_1", "frequency", "symmetric"): 1,
    ("lfo_1", "initial_phase", "curve"): 1.7274861975060194,
    ("lfo_1", "initial_phase", "symmetric"): 0,
    ("lfo_1", "mod_depth", "curve"): 3.3417273923831368,
    ("lfo_1", "mod_depth", "symmetric"): 0,
    ("lfo_1", "rsaw", "curve"): 0.7137511323379911,
    ("lfo_1", "rsaw", "symmetric"): 0,
    ("lfo_1", "saw", "curve"): 0.16949057158133907,
    ("lfo_1", "saw", "symmetric"): 1,
    ("lfo_1", "sin", "curve"): 0.3257344035389772,
    ("lfo_1", "sin", "symmetric"): 0,
    ("lfo_1", "sqr", "curve"): 5.741307914272704,
    ("lfo_1", "sqr", "symmetric"): 0,
    ("lfo_1", "tri", "curve"): 15.65237130797288,
    ("lfo_1", "tri", "symmetric"): 0,
    ("lfo_1_amp_adsr", "alpha", "curve"): 0.15526340053797494,
    ("lfo_1_amp_adsr", "alpha", "symmetric"): 0,
    ("lfo_1_amp_adsr", "attack", "curve"): 0.7940490377419372,
    ("lfo_1_amp_adsr", "attack", "symmetric"): 1,
    ("lfo_1_amp_adsr", "decay", "curve"): 0.990390146038856,
    ("lfo_1_amp_adsr", "decay", "symmetric"): 1,
    ("lfo_1_amp_adsr", "release", "curve"): 0.05534311197731066,
    ("lfo_1_amp_adsr", "release", "symmetric"): 0,
    ("lfo_1_amp_adsr", "sustain", "curve"): 0.5626797261734475,
    ("lfo_1_amp_adsr", "sustain", "symmetric"): 1,
    ("lfo_1_rate_adsr", "alpha", "curve"): 2.0815598739601326,
    ("lfo_1_rate_adsr", "alpha", "symmetric"): 1,
    ("lfo_1_rate_adsr", "attack", "curve"): 0.23904632340824827,
    ("lfo_1_rate_adsr", "attack", "symmetric"): 1,
    ("lfo_1_rate_adsr", "decay", "curve"): 0.993883748916903,
    ("lfo_1_rate_adsr", "decay", "symmetric"): 1,
    ("lfo_1_rate_adsr", "release", "curve"): 1.7946540502926707,
    ("lfo_1_rate_adsr", "release", "symmetric"): 1,
    ("lfo_1_rate_adsr", "sustain", "curve"): 0.8603582851084216,
    ("lfo_1_rate_adsr", "sustain", "symmetric"): 1,
    ("lfo_2", "frequency", "curve"): 5.168477764190512,
    ("lfo_2", "frequency", "symmetric"): 1,
    ("lfo_2", "initial_phase", "curve"): 0.21488640757659264,
    ("lfo_2", "initial_phase", "symmetric"): 1,
    ("lfo_2", "mod_depth", "curve"): 1.1677725911035184,
    ("lfo_2", "mod_depth", "symmetric"): 0,
    ("lfo_2", "rsaw", "curve"): 1.9292838381414954,
    ("lfo_2", "rsaw", "symmetric"): 0,
    ("lfo_2", "saw", "curve"): 0.3244204843242318,
    ("lfo_2", "saw", "symmetric"): 0,
    ("lfo_2", "sin", "curve"): 10.831797830965916,
    ("lfo_2", "sin", "symmetric"): 0,
    ("lfo_2", "sqr", "curve"): 0.5808153563684666,
    ("lfo_2", "sqr", "symmetric"): 1,
    ("lfo_2", "tri", "curve"): 3.6858488567032883,
    ("lfo_2", "tri", "symmetric"): 1,
    ("lfo_2_amp_adsr", "alpha", "curve"): 3.0403164141556704,
    ("lfo_2_amp_adsr", "alpha", "symmetric"): 1,
    ("lfo_2_amp_adsr", "attack", "curve"): 9.660756714451104,
    ("lfo_2_amp_adsr", "attack", "symmetric"): 0,
    ("lfo_2_amp_adsr", "decay", "curve"): 1.045223710414908,
    ("lfo_2_amp_adsr", "decay", "symmetric"): 0,
    ("lfo_2_amp_adsr", "release", "curve"): 6.64797195309235,
    ("lfo_2_amp_adsr", "release", "symmetric"): 1,
    ("lfo_2_amp_adsr", "sustain", "curve"): 0.4715272066344784,
    ("lfo_2_amp_adsr", "sustain", "symmetric"): 0,
    ("lfo_2_rate_adsr", "alpha", "curve"): 1.8102754226123818,
    ("lfo_2_rate_adsr", "alpha", "symmetric"): 0,
    ("lfo_2_rate_adsr", "attack", "curve"): 0.9815150184921098,
    ("lfo_2_rate_adsr", "attack", "symmetric"): 0,
    ("lfo_2_rate_adsr", "decay", "curve"): 1.3572815487507401,
    ("lfo_2_rate_adsr", "decay", "symmetric"): 1,
    ("lfo_2_rate_adsr", "release", "curve"): 1.2524125754552466,
    ("lfo_2_rate_adsr", "release", "symmetric"): 1,
    ("lfo_2_rate_adsr", "sustain", "curve"): 3.8062094209251502,
    ("lfo_2_rate_adsr", "sustain", "symmetric"): 0,
    ("mixer", "level0", "curve"): 1.7963958866163088,
    ("mixer", "level0", "symmetric"): 1,
    ("mixer", "level1", "curve"): 2.3121333085972386,
    ("mixer", "level1", "symmetric"): 0,
    ("mixer", "level2", "curve"): 50.00216647333782,
    ("mixer", "level2", "symmetric"): 1,
    ("mod_matrix", "level0_0", "curve"): 0.2913616304902357,
    ("mod_matrix", "level0_0", "symmetric"): 0,
    ("mod_matrix", "level0_1", "curve"): 2.8884783926237048,
    ("mod_matrix", "level0_1", "symmetric"): 0,
    ("mod_matrix", "level0_2", "curve"): 0.8035904462664656,
    ("mod_matrix", "level0_2", "symmetric"): 0,
    ("mod_matrix", "level0_3", "curve"): 0.4724908804616324,
    ("mod_matrix", "level0_3", "symmetric"): 1,
    ("mod_matrix", "level0_4", "curve"): 0.30714923752422296,
    ("mod_matrix", "level0_4", "symmetric"): 1,
    ("mod_matrix", "level1_0", "curve"): 0.11156064151369302,
    ("mod_matrix", "level1_0", "symmetric"): 1,
    ("mod_matrix", "level1_1", "curve"): 0.8421739003285549,
    ("mod_matrix", "level1_1", "symmetric"): 0,
    ("mod_matrix", "level1_2", "curve"): 1.4840111503052225,
    ("mod_matrix", "level1_2", "symmetric"): 1,
    ("mod_matrix", "level1_3", "curve"): 1.5888131527712834,
    ("mod_matrix", "level1_3", "symmetric"): 1,
    ("mod_matrix", "level1_4", "curve"): 0.17051034230203557,
    ("mod_matrix", "level1_4", "symmetric"): 0,
    ("mod_matrix", "level2_0", "curve"): 4.04559359924554,
    ("mod_matrix", "level2_0", "symmetric"): 0,
    ("mod_matrix", "level2_1", "curve"): 1.723003717098155,
    ("mod_matrix", "level2_1", "symmetric"): 0,
    ("mod_matrix", "level2_2", "curve"): 0.05481505897155334,
    ("mod_matrix", "level2_2", "symmetric"): 0,
    ("mod_matrix", "level2_3", "curve"): 0.5606624950431192,
    ("mod_matrix", "level2_3", "symmetric"): 1,
    ("mod_matrix", "level2_4", "curve"): 0.256882596223966,
    ("mod_matrix", "level2_4", "symmetric"): 1,
    ("mod_matrix", "level3_0", "curve"): 0.13022234577348926,
    ("mod_matrix", "level3_0", "symmetric"): 1,
    ("mod_matrix", "level3_1", "curve"): 0.5861813285382993,
    ("mod_matrix", "level3_1", "symmetric"): 0,
    ("mod_matrix", "level3_2", "curve"): 11.44212203514677,
    ("mod_matrix", "level3_2", "symmetric"): 0,
    ("mod_matrix", "level3_3", "curve"): 1.0997524755960648,
    ("mod_matrix", "level3_3", "symmetric"): 0,
    ("mod_matrix", "level3_4", "curve"): 3.2151732800925226,
    ("mod_matrix", "level3_4", "symmetric"): 1,
    ("vco_1", "initial_phase", "curve"): 1.1751338445188213,
    ("vco_1", "initial_phase", "symmetric"): 1,
    ("vco_2", "initial_phase", "curve"): 2.613855645351914,
    ("vco_2", "initial_phase", "symmetric"): 0,
    ("vco_2", "shape", "curve"): 0.7769495203149225,
    ("vco_2", "shape", "symmetric"): 0,
}

for name, value in proposed_nebula.items():
    if name[2] == "symmetric":
        continue

    proposed_update = {
        (name[0], name[1], "curve"): proposed_nebula[(name[0], name[1], "curve")],
        (name[0], name[1], "symmetric"): proposed_nebula[
            (name[0], name[1], "symmetric")
        ],
    }

    print(proposed_update)

    synthconfig = SynthConfig(batch_size=BATCH_SIZE, reproducible=False)
    voice = Voice(synthconfig)

    voice.eval()

    for name, value in default_nebula.items():
        voice.set_hyperparameter(name, value)

    for name, value in proposed_update.items():
        voice.set_hyperparameter(name, value)

    voice.unfreeze_all_parameters()
    if MIDI_F0:
        voice.set_frozen_parameters({("keyboard", "midi_f0"): MIDI_F0})

    # pprint.pprint(voice.hyperparameters)
    # for name, value in voice.hyperparameters.items():
    #    print(name, value)

    xs = voice(SALT)
    silence = np.zeros(int(synthconfig.sample_rate.numpy().item() // 2))
    bigx = []
    for i, x in enumerate(xs):
        x = x.detach().numpy()
        bigx.append(x)
        bigx.append(silence)
    bigx = np.hstack(bigx)
    display(ipd.Audio(bigx, rate=int(synthconfig.sample_rate.numpy())))
