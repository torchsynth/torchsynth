import sphinx_rtd_theme

extensions = [
    #    ...
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphiex",
    "sphinx_rtd_theme",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

html_theme = "sphinx_rtd_theme"
