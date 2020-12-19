import numpy as np
import matplotlib.pyplot as plt
from ddrum.drum_engine import ADSR, VCO, VCA
from scipy.signal import stft as stft

# Plots
make_plot = {'ADSR': True,
             'VCO':  True,
             'VCA':  True
             }

a = 0.5
d = 0.5
s = 0.5
r = 0.5
alpha = 3
sustain_for = 1

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
test_f0 = env*20000
myVCO = VCO()

x = myVCO(test_f0)

if make_plot['VCO']:
    mX = np.abs(stft(x)[2])
    plt.imshow(mX, origin='lower', aspect='auto')
    plt.title('VCO output')
    plt.show()

# VCA test
myVCA = VCA()
vca_out = myVCA(env, x)

if make_plot['VCA']:
    plt.plot(t_sr, vca_out)
    plt.xlabel('time (sec)')
    plt.ylabel('amplitude')
    plt.show()