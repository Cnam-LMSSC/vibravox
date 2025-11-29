"""Deep Neural Network modules for the Vibravox package."""

from vibravox.torch_modules.dnn.eben_discriminator import (
    DiscriminatorEBENMultiScales,
    DiscriminatorEBEN,
)
from vibravox.torch_modules.dnn.eben_generator import EBENGenerator
from vibravox.torch_modules.dnn.melgan_discriminator import (
    MelganMultiScalesDiscriminator,
    DiscriminatorMelGAN,
)

__all__ = [
    "DiscriminatorEBENMultiScales",
    "DiscriminatorEBEN",
    "EBENGenerator",
    "MelganMultiScalesDiscriminator",
    "DiscriminatorMelGAN",
]
