# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import torch

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))
sys.path.insert(0, os.path.abspath('../..'))

sys.path.insert(0, os.path.abspath("torchsynth"))

# sys.path.insert(0, os.path.abspath("torchsynth"))

import torchsynth  # noqa: E402
import pt_lightning_sphinx_theme
import sphinx_rtd_theme
import mock

# -- Project information -----------------------------------------------------

project = "torchsynth"
copyright = "2021, Joseph Turian, Jordie Schier, Max Henry"
author = "Joseph Turian, Jordie Schier, Max Henry"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------


# MOCK_MODULES = ['torch','torch.nn','']
autodoc_mock_imports = ["torch"]
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = []

# extensions = []
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.napoleon",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "recommonmark",
]

# This configures Sphinx to parse all files with the extensions .md and .txt as Markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
    "pytorch_lightning": ("https://pytorch-lightning.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------


# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# http://www.sphinx-doc.org/en/master/usage/theming.html#builtin-themes
# html_theme = 'bizstyle'
# https://sphinx-themes.org
html_theme = "pt_lightning_sphinx_theme"
html_theme_path = [pt_lightning_sphinx_theme.get_html_theme_path()]


html_logo = '_static/images/torchsynth-logotype.svg'

# html_favicon = '_static/images/favicon.svg'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Include all CSS style in the default theme settings
html_css_files = [
    'css/custom.css',
]
