import numpy as np
import matplotlib.pyplot as plt
import os
from ddrum.drum_engine import ADSR, VCO, VCA
from scipy.signal import stft as stft
from scipy.io import wavfile as wavfile

WAVOUT_DIR = 'audio_out'

# Plots.
make_plot = {'ADSR': True,
             'VCO':  True,
             'VCA':  True
             }

# Audio out.
make_wavfile = {'VCO': True,
                'VCA': True
                }

# Synthesis parameters.
a = 0.1
d = 0.1
s = 0.5
r = 0.5
alpha = 3
sustain_for = 0.25

f0 = 440

dur = a + d + r + sustain_for

# Envelope test
myADSR = ADSR(a, d, s, r, alpha)
env = myADSR(sustain_for)

t_cr = np.linspace(0, dur, int(dur * myADSR.get_rate()['control']), endpoint=False)
t_sr = np.linspace(0, dur, int(dur * myADSR.get_rate()['sample']), endpoint=False)

if make_plot['ADSR']:
    plt.plot(t_cr, env)
    plt.title('a={}, d={}, s={}, r={}, alpha={}'.format(myADSR.get_a(),
                                                        myADSR.get_d(),
                                                        myADSR.get_s(),
                                                        myADSR.get_r(),
                                                        myADSR.get_alpha()))
    plt.xlabel('time (sec)')
    plt.ylabel('amplitude')
    plt.show()

# VCO test
test_f0 = f0*(env + 1)
myVCO = VCO()

vco_out = myVCO(test_f0)

if make_plot['VCO']:
    mX = np.abs(stft(vco_out)[2])
    plt.imshow(mX, origin='lower', aspect='auto')
    plt.title('VCO output')
    plt.show()

if make_wavfile['VCO']:
    filename = 'vco_out.wav'
    path = os.path.join(WAVOUT_DIR, filename)
    wavfile.write(path, myADSR.get_rate()['sample'], vco_out)
    print('Saving VCO demo as {}'.format(filename))


# VCA test
myVCA = VCA()
vca_out = myVCA(env, vco_out)

if make_plot['VCA']:
    plt.plot(t_sr, vca_out)
    plt.xlabel('time (sec)')
    plt.ylabel('amplitude')
    plt.show()

if make_wavfile['VCA']:
    filename = 'vca_out.wav'
    path = os.path.join(WAVOUT_DIR, filename)
    wavfile.write(path, myADSR.get_rate()['sample'], vca_out)
    print('Saving VCA demo as {}'.format(filename))
