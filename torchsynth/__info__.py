import time
import os.path

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..")

_this_year = time.strftime("%Y")
__version__ = "0.9.1"
__author__ = "Jordie Shier, Joseph Turian, Max Henry"
__author_email__ = "firstnamelastname@gmail.com"
# __contact__ = ""
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-{_this_year}, {__author__}."
__homepage__ = "https://github.com/torchsynth/torchsynth"
__docs_url__ = "https://torchsynth.readthedocs.io/en/stable/"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = "A modular synthesizer in pytorch, GPU-optional and differentiable"
# ptl gets fancy and replaces the shields 'master' with '__version__'
# but I don't think we need to do that
__long_docs__ = open(os.path.join(PATH_ROOT, "README.md"), "r", encoding="utf-8").read()
