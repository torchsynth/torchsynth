"""
Optuna example that searches for synth parameter settings to match a target audio

    $ python optunematch.py [--ntrials] target
"""
import argparse
import os

import torch
import torch.tensor as T
import librosa
import soundfile as sf

import optuna

from torchsynth.synth import Voice
from torchsynth.module import SynthModule
from torchsynth.globals import SynthGlobals
import torchsynth.util as util
from torchsynth.loss import MultiScaleSTFTLoss


BATCH_SIZE = T(1)
synthglobals = SynthGlobals(BATCH_SIZE)

# Run on GPU if available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Audio distance
distance = MultiScaleSTFTLoss().to(device)

import openl3


def add_parameters(synth, trial):
    for name, module in synth.named_modules():
        if isinstance(module, SynthModule):
            for p in module.torchparameters:
                param_suggestion = trial.suggest_float((name, p), 0.0, 1.0)
                module.set_parameter_0to1(p, T([param_suggestion]))


def objective(trial: optuna.trial.Trial) -> float:

    # Optimize the parameters in Voice
    voice = FmVoice(synthglobals)
    add_parameters(voice, trial)

    with torch.no_grad():
        output = util.fix_length2D(voice(), target.shape[1])
        dist, _ = distance(output, target)

    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "target",
        type=str,
        nargs=1,
        help="Input file / directory for optimization target",
    )
    parser.add_argument(
        "--ntrials",
        "-n",
        type=int,
        default=[100],
        nargs=1,
        help="Number of trials to run",
    )
    args = parser.parse_args()

    # Look for files in target directory or use single input file
    filename = args.target[0]
    if os.path.isdir(filename):
        _, _, file_targets = next(os.walk(filename))
        file_targets = [os.path.join(filename, f) for f in file_targets]
    else:
        file_targets = [filename]

    global target
    for target_file in file_targets:

        # Load the target sound
        target, sr = librosa.load(target_file, sr=synthglobals.sample_rate)
        target = T([target])

        # CmaESsSampler recommended for n_trials greater than 1000
        sampler = optuna.samplers.CmaEsSampler()
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=args.ntrials[0], timeout=600)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("Saving output as wav")
        voice = FmVoice(synthglobals)

        for name, module in voice.named_modules():
            if isinstance(module, SynthModule):
                for p in module.torchparameters:
                    module.set_parameter_0to1(p, T([trial.params[(name, p)]]))

        with torch.no_grad():
            output = voice()

        target_name = os.path.splitext(os.path.basename(target_file))[0]
        sf.write(
            f"torchsynth-err={trial.value:.2f}-{target_name}.wav",
            output[0].numpy(),
            voice.sample_rate,
        )
