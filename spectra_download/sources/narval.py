"""Downloader for the NARVAL archive via TAP."""

from __future__ import annotations

from spectra_download.sources.tap import TapSpectraSource


class NarvalSource(TapSpectraSource):
    """NARVAL spectra downloads via the TAP service."""

    name = "narval"
    tap_url = "https://polarbase.irap.omp.eu/tap/sync"
    table = "ivoa.obscore"
