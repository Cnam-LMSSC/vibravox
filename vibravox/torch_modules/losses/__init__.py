"""Loss functions for the Vibravox package."""

from vibravox.torch_modules.losses.feature_loss import FeatureLossForDiscriminatorMelganMultiScales
from vibravox.torch_modules.losses.hinge_loss import HingeLossForDiscriminatorMelganMultiScales

__all__ = [
    "FeatureLossForDiscriminatorMelganMultiScales",
    "HingeLossForDiscriminatorMelganMultiScales",
]
