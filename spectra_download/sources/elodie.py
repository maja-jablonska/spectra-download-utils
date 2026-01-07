"""Downloader for the ELODIE archive via TAP."""

from __future__ import annotations

from spectra_download.sources.tap import TapSpectraSource


class ElodieSource(TapSpectraSource):
    """ELODIE spectra downloads via the TAP service."""

    name = "elodie"
    tap_url = "https://vo.obs-hp.fr/tap/sync"
    table = "ivoa.obscore"
