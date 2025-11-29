"""Digital Signal Processing modules for the Vibravox package."""

from vibravox.torch_modules.dsp.data_augmentation import WaveformDataAugmentation
from vibravox.torch_modules.dsp.pqmf import PseudoQMFBanks
from vibravox.torch_modules.dsp.time_masking_waveform import TimeMaskingBlockWaveform

__all__ = [
    "WaveformDataAugmentation",
    "PseudoQMFBanks",
    "TimeMaskingBlockWaveform",
]
