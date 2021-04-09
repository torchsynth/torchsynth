"""
Tests for the profile script - hard to capture the output, but this also serves
as an integration test to make sure the PyTorch Lightning training stuff is working.
"""

import csv
import os
import sys
import unittest.mock

import pytest

from torchsynth import profile


class TestProfile:
    @staticmethod
    def run_profile(
        module="Voice", batch_size=2, n_iters=1, device=None, cprofile=False, save=None
    ):
        args = ["prog", module, "-b", str(batch_size), "-n", str(n_iters)]

        if device is not None:
            args.extend(["-d", device])
        if cprofile:
            args.append("-p")
        if save is not None:
            args.extend(["-s", save])

        # This mocks arguments as if this was called from command line
        with unittest.mock.patch.object(sys, "argv", args):
            profile.main()

    def test_main(self):
        # Test run on Voice with batch size of 2 and 2 iterations
        self.run_profile(n_iters=2)

    def test_main_cpu(self):
        # Confirm running on CPU works
        self.run_profile(device="cpu")

    def test_main_profiling(self):
        # Test run with profiling
        self.run_profile(cprofile=True)

    def test_main_profiling_csv(self, tmp_path):
        # Test run with profiling and save to tmp dir
        temp_csv_out = os.path.join(tmp_path, "profile.csv")

        # Trying to save results with out profiling should raise an error
        with pytest.raises(SystemExit):
            self.run_profile(save=temp_csv_out)

        self.run_profile(cprofile=True, save=temp_csv_out)

        # Check to make sure the output looks correct
        with open(temp_csv_out) as fp:
            reader = csv.reader(fp)
            header = next(reader)
            assert header[0] == "ncalls"
            first_line = next(reader)
            assert first_line[0] == "1"

    def test_main_no_gpu(self):
        # Make running with no GPUs available (in case there are), and then
        # confirm that passing in the cuda device arg causes an error
        with unittest.mock.patch.object(profile, "GPUS", 0):
            with pytest.raises(SystemExit):
                self.run_profile(device="cuda")

    def test_main_no_module(self):
        with pytest.raises(AttributeError):
            self.run_profile(module="NotAModule")
