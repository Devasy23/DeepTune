"""
Loss functions for DeepTuner
"""

from .triplet_loss import triplet_loss
from .arcface_loss import arcface_loss

__all__ = ['triplet_loss', 'arcface_loss']
