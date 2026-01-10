"""Source-specific download implementations."""

from spectra_download.sources.elodie import ElodieSource
from spectra_download.sources.eso import EsoHarpsSource, EsoNirpsSource, EsoUvesSource
from spectra_download.sources.narval import NarvalSource

__all__ = [
    "ElodieSource",
    "EsoHarpsSource",
    "EsoNirpsSource",
    "EsoUvesSource",
    "NarvalSource",
]
