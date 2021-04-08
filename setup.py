#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.9.0"
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
    license=__license__,
    packages=find_packages(exclude=("tests", "examples")),
    # package_dir={"": "src"},
    # package_data={
    #    "": [],
    # },
    entry_points={"console_scripts": ["torchsynth.profile=torchsynth.profile:main"]},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "torch>=1.7",
        "pytorch-lightning",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
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
            "pygments>=2.7.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            "sphinx>=3.0.4",  # not directly required, pinned by Snyk to avoid a vulnerability
            # Temporarily disabled so we can push to pypi
            # "pt-lightning-sphinx-theme @ https://github.com/PyTorchLightning/lightning_sphinx_theme/tarball/master#egg=pt-lightning-sphinx-theme",
            "sphinxcontrib-napoleon",
            "mock",
            "sphinx_rtd_theme",
            "myst_parser",
            "linkify-it-py",
        ],
    },
)
