"""Spectra download utilities."""

from spectra_download.bulk import bulk_download
from spectra_download.models import SpectraRequest, SpectraResult, SpectrumRecord
from spectra_download.sources.elodie import ElodieSource
from spectra_download.sources.eso import EsoHarpsSource, EsoNirpsSource, EsoUvesSource
from spectra_download.sources.narval import NarvalSource

__all__ = [
    "ElodieSource",
    "EsoHarpsSource",
    "EsoNirpsSource",
    "EsoUvesSource",
    "NarvalSource",
    "SpectraRequest",
    "SpectraResult",
    "SpectrumRecord",
    "bulk_download",
]
