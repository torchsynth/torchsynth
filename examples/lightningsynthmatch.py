# -*- coding: utf-8 -*-
# +
import multiprocessing
from typing import Any

import auraloss
import pytorch_lightning as pl
import soundfile as sf
import torch

from torchsynth.config import SynthConfig
from torchsynth.synth import Voice

# -

gpus = torch.cuda.device_count()
print("Usings %d gpus" % gpus)

ncores = multiprocessing.cpu_count()
print(f"Using ncores {ncores} for generating batch numbers (low CPU usage)")


class batch_idx_dataset(torch.utils.data.Dataset):
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_batches


synthconfig = SynthConfig()
voice = Voice(synthconfig)
synth1B = batch_idx_dataset(1000000000 // synthconfig.batch_size)
test_dataloader = torch.utils.data.DataLoader(synth1B, num_workers=0, batch_size=1)

target_sound, sr = sf.read("909-snare-4sec.ogg")
# This won't work with multi-GPU :(
device = "cuda" if torch.cuda.is_available() else "cpu"
target_sound = torch.tensor(target_sound, device=device)
# Normalize, but not entirely sure this is necessary
target_sound_max = torch.max(torch.abs(target_sound))
print("Normalizing target sound by", target_sound_max)
target_sound /= target_sound_max
assert sr == synthconfig.sample_rate
assert target_sound.ndim == 1
assert target_sound.shape[0] == synthconfig.buffer_size


# TODO Add this to torchsynth API
# see https://github.com/turian/torchsynth/issues/154
class TorchSynthCallback(pl.Callback):
    def __init__(self, target_sound, sr, **kwargs):
        super().__init__(**kwargs)
        self.target_sound = target_sound
        self.sr = sr
        self.best_distance = 1e10
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    def on_test_batch_end(
        self,
        trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        assert batch.ndim == 1
        batchsounds = pl_module(batch_idx)
        for i, sound in enumerate(batchsounds):
            assert sound.shape == self.target_sound.shape
            loss = self.mrstft(self.target_sound, sound)
            if loss < self.best_distance:
                print(f"New best loss {loss} for {batch_idx}-{i}")
                sf.write(
                    f"predicted_sound-{loss}-{batch_idx}-{i}.wav",
                    sound.cpu().detach().numpy(),
                    self.sr,
                )
                self.best_distance = loss


accelerator = None
if gpus == 0:
    use_gpus = None
    precision = 32
else:
    # specifies all available GPUs (if only one GPU is not occupied,
    # auto_select_gpus=True uses one gpu)
    use_gpus = -1
    # TODO: Change precision?
    precision = 16
    if gpus > 1:
        accelerator = "ddp"

# Use deterministic?
trainer = pl.Trainer(
    precision=precision,
    gpus=use_gpus,
    auto_select_gpus=True,
    accelerator=accelerator,
    deterministic=True,
    max_epochs=0,
    callbacks=[TorchSynthCallback(target_sound, synthconfig.sample_rate)],
)

trainer.test(voice, test_dataloaders=test_dataloader)
