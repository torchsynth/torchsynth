import torch
import torch.tensor as T
from torchsynth.module import TorchADSR

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Debugging.

    a = T([0.1, 0.1])
    d = T([0.5, 0.5])
    s = T([0.6, 0.6])
    r = T([0.5, 0.5])
    alpha = T([2.0, 2.0])

    note_on_duration = T([1.0, 1.5])

    adsr = TorchADSR(a, d, s, r, alpha)

    test = adsr.forward(note_on_duration)
    plt.plot(test.detach().numpy().T)
    plt.show()
