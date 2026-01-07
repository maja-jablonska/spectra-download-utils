"""Source-specific download implementations."""

from spectra_download.sources.elodie import ElodieSource
from spectra_download.sources.eso import EsoHarpsSource, EsoNirpsSource, EsoUvesSource
from spectra_download.sources.gnps import GnpSource
from spectra_download.sources.massbank import MassBankSource
from spectra_download.sources.narval import NarvalSource
from spectra_download.sources.nist import NistSource

__all__ = [
    "ElodieSource",
    "EsoHarpsSource",
    "EsoNirpsSource",
    "EsoUvesSource",
    "GnpSource",
    "MassBankSource",
    "NarvalSource",
    "NistSource",
]
