"""Synthesizers module."""

from ctgan.synthesizers.ctgan import CTGAN
from ctgan.synthesizers.tvae import TVAE
from ctgan.synthesizers.transformer import Encoder

__all__ = (
    'CTGAN',
    'TVAE',
    'Encoder',
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
