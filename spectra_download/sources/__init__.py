"""Source-specific download implementations."""

from spectra_download.sources.elodie import ElodieSource
from spectra_download.sources.eso import EsoFerosSource, EsoHarpsSource, EsoNirpsSource, EsoUvesSource
from spectra_download.sources.cafe import CafeSource
from spectra_download.sources.espadons import EspadonsSource
from spectra_download.sources.narval import NarvalSource
from spectra_download.sources.keys import SpectrumKeys, DataKeys, ObservedFrame

__all__ = [
    "CafeSource",
    "ElodieSource",
    "EsoFerosSource",
    "EsoHarpsSource",
    "EsoNirpsSource",
    "EsoUvesSource",
    "EspadonsSource",
    "NarvalSource",
]
