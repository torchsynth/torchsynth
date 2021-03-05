#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.0.1"
__author__ = ""
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
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "": [],
    },
    scripts=[],
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "torch>=1.7"],
    extras_require={
        "dev": ["pytest", "pytest-cov", "ipython", "librosa", "matplotlib"],
    },
)
