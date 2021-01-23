# # ddsp-drum torch examples
#
# We walk through basic functionality of `ddsp-drum` in this Jupyter notebook.
#
# Just note that all ipd.Audio play widgets normalize the audio.

# +
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from ddspdrum.defaults import SAMPLE_RATE

# -


import ddspdrum.torchmodule
vco = ddspdrum.torchmodule.TorchVCO(midi_f0=69.0, mod_depth=24.0)

two_8ve_chirp = vco(np.linspace(0, 1, 1000, endpoint=False))

print(vco.parameters)


