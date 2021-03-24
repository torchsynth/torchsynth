import torch
import torch.nn as nn
import torch.tensor as T

import torchsynth.util as util
from torchsynth.default import DEFAULT_SAMPLE_RATE
from torchsynth.deprecated import SynthModule0Ddeprecated
from torchsynth.parameter import ModuleParameter, ModuleParameterRange


class FIRLowPass(SynthModule0Ddeprecated):
    """
    A finite impulse response low-pass filter. Uses convolution with a windowed
    sinc function.

    Args:
        cutoff (float) : cutoff frequency of low-pass in Hz, must be between 5 and
        half the sampling rate. Defaults to 1000Hz.
        filter_length (int) :   The length of the filter in samples.
         A longer filter will
        result in a steeper filter cutoff. Should be greater than 4.
        Defaults to 512 samples.
        sample_rate (int)   :   Sampling rate to run processing at.
    """

    def __init__(
        self,
        cutoff: float = 1000.0,
        filter_length: int = 512,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        super().__init__(sample_rate=sample_rate)
        self.add_parameters(
            [
                ModuleParameter(
                    value=cutoff,
                    parameter_name="cutoff",
                    parameter_range=ModuleParameterRange(5.0, sample_rate / 2.0, 0.5),
                ),
                ModuleParameter(
                    value=filter_length,
                    parameter_name="length",
                    parameter_range=ModuleParameterRange(4.0, 4096.0),
                ),
            ]
        )

    def _forward(self, audio_in: T) -> T:
        """
        Filter audio samples
        TODO: Cutoff frequency modulation, if there is an efficient way to do it

        Args:
            audio (T)  :   audio samples to filter
        """

        impulse = self.windowed_sinc(self.p("cutoff"), self.p("length"))
        impulse = impulse.view(1, 1, impulse.size()[0])
        audio_resized = audio_in.view(1, 1, audio_in.size()[0])
        y = nn.functional.conv1d(
            audio_resized, impulse, padding=int(self.p("length") / 2)
        )
        return y[0][0]

    def windowed_sinc(self, cutoff: T, length: T) -> T:
        """
        Calculates the impulse response for FIR low-pass filter using the
        windowed sinc function method. Updated to allow for a fractional filter length.

        Args:
            cutoff (T) : Low-pass cutoff frequency in Hz. Must be between 0 and
            half the sampling rate.
            length (T) : Length of the filter impulse response to create.
        """

        # Normalized frequency
        omega = 2 * torch.pi * cutoff / self.sample_rate

        # Create a sinc function
        num_samples = torch.ceil(length)
        half_length = (length - 1.0) / 2.0
        t = torch.arange(num_samples.detach(), device=length.device)
        ir = util.sinc((t - half_length) * omega)

        return ir * util.blackman(length)


class MovingAverage(SynthModule0Ddeprecated):
    """
    A finite impulse response moving average filter.

    Args:
        filter_length (int) : Length of filter and number of samples
         to take average over.
        Must be greater than 0. Defaults to 32.
        sample_rate (int) : Sampling rate to run processing at.
    """

    def __init__(self, filter_length: int = 32, sample_rate: int = DEFAULT_SAMPLE_RATE):
        super().__init__(sample_rate=sample_rate)
        self.add_parameters(
            [
                ModuleParameter(
                    value=filter_length,
                    parameter_name="length",
                    parameter_range=ModuleParameterRange(1.0, 4096.0),
                )
            ]
        )

    def _forward(self, audio_in: T) -> T:
        """
        Filter audio samples

        Args:
            audio (T) : audio samples to filter
        """
        length = self.p("length")
        impulse = torch.ones((1, 1, int(length)), device=length.device) / length

        # For non-integer impulse lengths
        if torch.sum(impulse) < 1.0:
            additional = torch.ones(1, 1, 1, device=length.device)
            additional *= 1.0 - torch.sum(impulse)
            impulse = torch.cat((impulse, additional), dim=2)

        audio_resized = audio_in.view(1, 1, audio_in.size()[0])
        y = nn.functional.conv1d(
            audio_resized, impulse, padding=int(impulse.size()[0] / 2)
        )
        return y[0][0]


class SVF(SynthModule0Ddeprecated):
    """
    A State Variable Filter that can do low-pass, high-pass, band-pass, and
    band-reject filtering. Allows modulation of the cutoff frequency and an
    adjustable resonance parameter. Can self-oscillate to make a sinusoid
    oscillator.

    Args:
        mode (str) : filter type, one of LPF, HPF, BPF, or BSF
        cutoff (float) : cutoff frequency in Hz must be between 5 and
        half the sample rate. Defaults to 1000Hz.
        resonance (float) : filter resonance, or "Quality Factor". Higher
        values cause the filter to resonate more. Must
        be greater than 0.5. Defaults to 0.707.
    mod_depth (float) : Amount of modulation to apply to the cutoff from
    the control input during processing. Can be negative
    or positive in Hertz. Defaults to zero.
    """

    def __init__(
        self,
        mode: str,
        cutoff: float = 1000.0,
        resonance: float = 0.707,
        mod_depth: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Set the filter type
        self.mode = mode.lower()
        assert mode in ["lpf", "hpf", "bpf", "bsf"]

        nyquist = self.sample_rate / 2.0
        self.add_parameters(
            [
                ModuleParameter(
                    value=cutoff,
                    parameter_range=ModuleParameterRange(5.0, nyquist, 0.5),
                    parameter_name="cutoff",
                ),
                ModuleParameter(
                    value=resonance,
                    parameter_range=ModuleParameterRange(0.01, 1000.0, 0.5),
                    parameter_name="resonance",
                ),
                ModuleParameter(
                    value=mod_depth,
                    parameter_range=ModuleParameterRange(-nyquist, nyquist, 0.5),
                    parameter_name="mod_depth",
                ),
            ]
        )

    def _forward(
        self,
        audio_in: T,
        control_in: T = None,
    ) -> T:
        """
        Process audio samples and return filtered results.

        Args
        audio (torch.tensor) : Audio samples to filter
        cutoff_mod (torch.tensor) : Control signal used to modulate the filter
        cutoff. Values must be in range [0,1]
        """

        h0 = 0.0
        h1 = 0.0
        y = torch.zeros_like(audio_in, device=audio_in.device)

        if control_in is None:
            control_in = torch.zeros_like(audio_in, device=audio_in.device)
        else:
            assert control_in.size() == audio_in.size()

        cutoff = self.p("cutoff")
        mod_depth = self.p("mod_depth")
        res_coefficient = 1.0 / self.p("resonance")

        # Processing loop
        for i in range(len(audio_in)):
            # If there is a cutoff modulation envelope, update coefficients
            cutoff_val = cutoff + control_in[i] * mod_depth
            coeff0, coeff1, rho = SVF.svf_coefficients(
                cutoff_val, res_coefficient, self.sample_rate
            )

            # Calculate each of the filter components
            hpf = coeff0 * (audio_in[i] - rho * h0 - h1)
            bpf = coeff1 * hpf + h0
            lpf = coeff1 * bpf + h1

            # Feedback samples
            h0 = coeff1 * hpf + bpf
            h1 = coeff1 * bpf + lpf

            if self.mode == "lpf":
                y[i] = lpf
            elif self.mode == "bpf":
                y[i] = bpf
            elif self.mode == "bsf":
                y[i] = hpf + lpf
            else:
                y[i] = hpf

        return y

    @staticmethod
    def svf_coefficients(cutoff, res_coefficient, sample_rate):
        """
        Calculates the filter coefficients for SVF.

        Args:
            cutoff (T)  :   Filter cutoff frequency in Hz.
            resonance (T) : Filter resonance
            sample_rate (T) : Sample rate to process at
        """

        g = torch.tan(torch.pi * cutoff / sample_rate)
        coeff0 = 1.0 / (1.0 + res_coefficient * g + g * g)
        rho = res_coefficient + g

        return coeff0, g, rho


class TorchLowPassSVF(SVF):
    """
    IIR Low-pass using SVF architecture

    Args:
        kwargs: see SVF
    """

    def __init__(self, **kwargs):
        super().__init__("lpf", **kwargs)


class TorchHighPassSVF(SVF):
    """
    IIR High-pass using SVF architecture

    Parameters
    ----------
    kwargs: see SVF
    """

    def __init__(self, **kwargs):
        super().__init__("hpf", **kwargs)


class TorchBandPassSVF(SVF):
    """
    IIR Band-pass using SVF architecture

    Args:
        kwargs: see SVF
    """

    def __init__(self, **kwargs):
        super().__init__("bpf", **kwargs)


class TorchBandStopSVF(SVF):
    """
    IIR Band-stop using SVF architecture

    Args:
        kwargs: see SVF
    """

    def __init__(self, **kwargs):
        super().__init__("bsf", **kwargs)
