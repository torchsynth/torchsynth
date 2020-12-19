import matplotlib.pyplot as plt
from ddrum.drum_engine import ADSR

# myADSR = ADSR(0.5, 0.5, 0.5, 0.5, 3)
myADSR = ADSR()
plt.plot(myADSR(1))
plt.show()