import numpy as np
import matplotlib.pyplot as plt
from ddrum.drum_engine import ADSR, VCO
from scipy.signal import stft as stft

# Envelope test
myADSR = ADSR()
plt.plot(myADSR(1))
plt.title('a={}, d={}, s={}, r={}, alpha={}'.format(myADSR.get_a(),
                                                    myADSR.get_d(),
                                                    myADSR.get_s(),
                                                    myADSR.get_r(),
                                                    myADSR.get_alpha()))
plt.show()

# VCO test
test_f0 = myADSR(1)*20000
myVCO = VCO()

x = myVCO(test_f0)
mX = np.abs(stft(x)[2])

plt.imshow(mX, origin='lower', aspect='auto')
plt.title('VCO output')
plt.show()