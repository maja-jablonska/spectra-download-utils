"""Downloader for ESO archive spectra via TAP."""

from __future__ import annotations

from spectra_download.sources.tap import TapSpectraSource


class EsoTapSource(TapSpectraSource):
    """ESO TAP spectra source for a specific instrument."""

    tap_url = "https://archive.eso.org/tap_obs/sync"
    table = "ivoa.obscore"


class EsoHarpsSource(EsoTapSource):
    """ESO HARPS spectra downloads."""

    name = "eso_harps"
    extra_conditions = ("instrument_name='HARPS'",)


class EsoNirpsSource(EsoTapSource):
    """ESO NIRPS spectra downloads."""

    name = "eso_nirps"
    extra_conditions = ("instrument_name='NIRPS'",)


class EsoUvesSource(EsoTapSource):
    """ESO UVES spectra downloads."""

    name = "eso_uves"
    extra_conditions = ("instrument_name='UVES'",)
