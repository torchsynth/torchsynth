# torchsynth

torchsynth is based upon traditional modular synthesis written in
pytorch. It is GPU-optional and differentiable.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/turian/torchsynth/blob/main/examples/examples.ipynb)

[![PyPI](https://img.shields.io/pypi/v/torchsynth)](https://pypi.org/project/torchsynth/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/torchsynth)
![PyPI - License](https://img.shields.io/pypi/l/torchsynth)
[![codecov.io](https://codecov.io/gh/turian/torchsynth/branch/main/graphs/badge.svg?logoWidth=18)](https://codecov.io/github/turian/torchsynth?branch=master)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/turian/torchsynth.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/turian/torchsynth/alerts/)
[![Travis CI build status](https://travis-ci.com/turian/torchsynth.png)](https://travis-ci.com/turian/torchsynth)
![Snyk Vulnerabilities for GitHub Repo](https://img.shields.io/snyk/vulnerabilities/github/turian/torchsynth)


## Development Installation

```
git clone https://github.com/turian/torchsynth
cd torchsynth
pip install -e ".[dev]"
```

Make sure you have pre-commit hooks installed:
```
pre-commit install
```
This helps us avoid checking dirty jupyter notebook cells into the
repo.

Note that torchsynth requires PyTorch version 1.7 or greater.

### Examples

We recommend that you run examples through Jupyter notebooks, and
that you have
[jupytext](https://towardsdatascience.com/introducing-jupytext-9234fdff6c57)
installed. It's a little fiddly to install, and those instructions
are the best. jupytext makes it easy to put demo notebooks into the
repo as Python files. (Larger assets like ipynb files we should
avoid.)

To run examples, you should also do:
```
pip install -e ".[dev]"
```

Unfortunately, Python 3.9 (e.g. OSX Big Sur) won't work, because
librosa repends upon numba which isn't packaged for 3.9 yet. In
which case you'll have to create a Python 3.7 conda environment.
(You might also need to downgrade LLVM to 10 or 9.):
```
conda install -c conda-forge ipython librosa matplotlib numpy matplotlib scipy jupytext
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=envname
```
and change the kernel to `envname`.

### Tests
Unit testing is performed using `pytest`.

`pytest` and other project development dependencies can be installed as follows: 
```
pip install -e ".[dev]"
```

To run tests, run `pytest` from the project root:
```
pytest
```

To run tests with a coverage report:
```
pytest --cov=./torchsynth
```

