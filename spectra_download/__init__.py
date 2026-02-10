"""Spectra download utilities."""

from spectra_download.bulk import bulk_download
from spectra_download.models import SpectraRequest, SpectraResult, SpectrumRecord
from spectra_download.sources.cafe import CafeSource
from spectra_download.sources.elodie import ElodieSource
from spectra_download.sources.eso import EsoFerosSource, EsoHarpsSource, EsoNirpsSource, EsoUvesSource
from spectra_download.sources.espadons import EspadonsSource
from spectra_download.sources.narval import NarvalSource

__all__ = [
    "CafeSource",
    "ElodieSource",
    "EsoFerosSource",
    "EsoHarpsSource",
    "EsoNirpsSource",
    "EsoUvesSource",
    "EspadonsSource",
    "NarvalSource",
    "SpectraRequest",
    "SpectraResult",
    "SpectrumRecord",
    "bulk_download",
]
