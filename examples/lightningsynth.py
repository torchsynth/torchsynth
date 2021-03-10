import multiprocessing

# import torchvision.models as models
# import torch.autograd.profiler as profiler
import pytorch_lightning as pl
import torch
import torch.tensor as T
from tqdm.auto import tqdm

from torchsynth.globals import SynthGlobals
from torchsynth.synth import Voice

ngpus = torch.cuda.device_count()
print("Usings %d gpus" % ngpus)

# Note this is the batch size for our synth!
# Not the batch size of the datasets
BATCH_SIZE = 256


ncores = multiprocessing.cpu_count()
print(f"Using ncores {ncores} for generating batch numbers (low CPU usage)")

torch.use_deterministic_algorithms(True)


class batch_idx_dataset(torch.utils.data.Dataset):
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_batches


synth1B = batch_idx_dataset(1024 * 1024 * 1024 // BATCH_SIZE)

# Probably don't need to pin memory for generating ints
# We use batch_size 1 here because the synth modules are already batched!
train_dataloader = torch.utils.data.DataLoader(
    batch_idx_dataset(10), num_workers=0, batch_size=1
)
test_dataloader = torch.utils.data.DataLoader(synth1B, num_workers=0, batch_size=1)


synthglobals = SynthGlobals(batch_size=T(BATCH_SIZE))
model = Voice(synthglobals)
# model = LM()

accelerator = None
if ngpus == 0:
    use_gpus = None
    precision = 32
else:
    # specifies all available GPUs (if only one GPU is not occupied,
    # auto_select_gpus=True uses one gpu)
    use_gpus = -1
    # TODO: Change precision?
    precision = 16
    if ngpus > 1:
        accelerator = "ddp"

# Use deterministic?
trainer = pl.Trainer(
    precision=precision,
    gpus=use_gpus,
    auto_select_gpus=True,
    accelerator=accelerator,
    deterministic=True,
    max_epochs=0,
)

trainer.fit(model, train_dataloader=train_dataloader)
trainer.test(model, test_dataloaders=test_dataloader)
