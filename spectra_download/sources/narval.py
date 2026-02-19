"""Downloader for NARVAL spectra (PolarBase) via TAP."""

from __future__ import annotations

from spectra_download.sources.polarbase import PolarBaseSource


class NarvalSource(PolarBaseSource):
    """NARVAL spectra downloads via CDS TAPVizieR (PolarBase)."""

    name = "narval"
    _inst_value = "narval"
