#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.0.2"
__author__ = "Joseph Turian, Jordie Shier, Max Henry"
__contact__ = ""
__url__ = ""
__license__ = "Apache-2.0"


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="torchsynth",
    version=__version__,
    author=__author__,
    author_email=__contact__,
    description="A modular synthesizer in pytorch, GPU-optional and differentiable",
    long_description=readme,
    long_description_content_type="text/markdown",
    url=__url__,
    licence=__license__,
    packages=find_packages(exclude=("tests", "examples")),
    # package_dir={"": "src"},
    # package_data={
    #    "": [],
    # },
    scripts=[],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=1.7",
        "pytorch-lightning",
        "torchcsprng==0.2.0",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
        ],
        "dev": [
            "pre-commit",
            "nbstripout==0.3.9",  # Used in precommit hooks
            "black==20.8b1",  # Used in precommit hooks
            "jupytext==v1.10.3",  # Used in precommit hooks
            "pytest",
            "pytest-cov",
            "ipython",
            "librosa",
            "matplotlib",
            "numba>=0.49.0",  # not directly required, pinned by Snyk to avoid a vulnerability
        ],
    },
)
