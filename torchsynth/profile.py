#!/usr/bin/env python3
"""
Script for running profiles on specific synths in torchsynth

usage: torchsynth.profile [-h] [--batch-size BATCH_SIZE] [--num_batches NUM_BATCHES]
                          [--profile] [--save SAVE] [--device DEVICE] module


Args:
    module (required): Name of the synth class to profile (e.g. Voice)
    batch-size: size of batch-size to run module at, defaults to 64
    num-batches: number of batches to run, defaults to 64
    profile: whether to run cProfile, defaults to False
    save: file to save results as csv, defaults to None, which doesn't save file
    device: Set to run on cuda or cpu. Defaults to None, selects cuda if available
"""

import argparse
import cProfile
import io
import multiprocessing
import pstats
import sys
from typing import Any

import pytorch_lightning as pl
import torch

import torchsynth.synth  # noqa: E402
from torchsynth.config import SynthConfig  # noqa: E402
from torchsynth.synth import AbstractSynth

# TODO: Disable DEBUG


# Check for available GPUs and processing cores
GPUS = torch.cuda.device_count()
NCORES = multiprocessing.cpu_count()


class BatchIDXDataset(torch.utils.data.Dataset):
    def __init__(self, num_batches):
        self.num_batches = num_batches

    def __getitem__(self, idx):
        return idx

    def __len__(self):
        return self.num_batches


class TorchSynthCallback(pl.Callback):
    def on_test_batch_end(
        self,
        trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        _ = pl_module(batch_idx)


def instantiate_module(name: str, synthconfig: SynthConfig, **kwargs) -> AbstractSynth:
    """
    Try to instantiate the module corresponding to the name providing
    """
    module = getattr(torchsynth.synth, name)
    return module(synthconfig, **kwargs)


def run_lightning_module(
    module: AbstractSynth,
    batch_size: int,
    n_batches: int,
    output: str,
    profile: bool,
    device: None,
):
    mock_dataset = BatchIDXDataset(batch_size * n_batches)
    dataloader = torch.utils.data.DataLoader(
        mock_dataset, num_workers=NCORES, batch_size=batch_size
    )

    if GPUS == 0 and device == "cuda":
        raise SystemExit("cuda specified but no gpus are avaiable")

    accelerator = None
    if GPUS == 0 or device == "cpu":
        use_gpus = None
    else:  # pragma: no cover
        # specifies all available GPUs (if only one GPU is not occupied,
        # auto_select_gpus=True uses one gpu)
        use_gpus = -1
        if GPUS > 1:
            accelerator = "ddp"

    # Use deterministic?
    trainer = pl.Trainer(
        gpus=use_gpus,
        auto_select_gpus=True,
        accelerator=accelerator,
        deterministic=True,
        max_epochs=0,
        callbacks=[TorchSynthCallback()],
    )

    if profile:
        # Run module with profiling
        pr = cProfile.Profile()
        pr.enable()
        trainer.test(module, dataloaders=dataloader)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")

        # Save profiling results to a csv file
        if output is not None:
            ps.print_stats()
            result = s.getvalue()
            result = "ncalls" + result.split("ncalls")[-1]
            result = "\n".join(
                [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
            )

            with open(output, "w+") as fp:
                fp.write(result)
        else:
            ps.print_stats()
            print(s.getvalue())

    else:
        trainer.test(module, dataloaders=dataloader)


def main():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("module", help="module to profile", type=str)
    parser.add_argument(
        "--batch-size",
        "-b",
        help="Batch size to run profiling at",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_batches", "-n", help="Number of batches to run", type=int, default=64
    )
    parser.add_argument(
        "--profile",
        "-p",
        help="Whether to run cProfile",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save",
        "-s",
        help="File to save profiler results. If this is left out then profiling "
        "results are printed. ",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to run. Default is None which will select cuda if available, "
        "otherwise will run on cpu",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.save is not None and not args.profile:
        raise SystemExit(
            "Profile (-p) flag must be set in order for profile results to be saved"
        )

    # Try to create the synth module that is being profiled
    synthconfig = SynthConfig(args.batch_size, reproducible=False)
    module = instantiate_module(args.module, synthconfig)

    run_lightning_module(
        module, args.batch_size, args.num_batches, args.save, args.profile, args.device
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
