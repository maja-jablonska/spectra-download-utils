"""Downloader for ESO archive spectra via TAP."""

from __future__ import annotations
from typing import Any, Dict

from astropy.coordinates import Angle
from astropy.io import fits as astro_fits
from io import BytesIO
import numpy as np

from spectra_download.models import SpectrumRecord
from spectra_download.sources.keys import DataKeys, ObservedFrame, SpectrumKeys
from spectra_download.sources.tap import TapSpectraSource


class EsoTapSource(TapSpectraSource):
    """ESO TAP spectra source for a specific instrument."""

    tap_url = "https://archive.eso.org/tap_obs/sync"
    table = "ivoa.obscore"


class EsoHarpsSource(EsoTapSource):
    """ESO HARPS spectra downloads."""

    name = "eso_harps"
    extra_conditions = ("instrument_name='HARPS'",)
    
    def extract_spectrum_arrays_from_fits_payload(
        self, *, fits_bytes: bytes, spectrum: SpectrumRecord
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract Zarr-ready arrays for ESO HARPS spectra from FITS bytes."""
        fits = astro_fits.open(BytesIO(fits_bytes))
        if len(fits) != 2:
            raise ValueError(f"ESO HARPS FITS files should have 2 extensions. This fits file has {len(fits)} extensions.")
        hdr = fits[0].header
        arrays: Dict[str, Any] = {
            SpectrumKeys.wavelengths.value: np.atleast_1d(np.array(fits[1].data['WAVE'], dtype=np.float64)),
            SpectrumKeys.intensity.value: np.atleast_1d(np.array(fits[1].data['FLUX'], dtype=np.float64)),
            SpectrumKeys.error.value: np.atleast_1d(np.array(fits[1].data['ERR'], dtype=np.float64))
        }
        info: Dict[str, Any] = {
            DataKeys.exptime.value: hdr['TEXPTIME'],
            DataKeys.ra.value: hdr['RA'],
            DataKeys.dec.value: hdr['DEC'],
            DataKeys.reference_frame.value: hdr['RADECSYS'],
            DataKeys.reference_frame_epoch.value: hdr['EQUINOX'],
            DataKeys.date.value: hdr['DATE-OBS'],
            DataKeys.mjd.value: hdr['MJD-OBS'],
            DataKeys.object.value: hdr['OBJECT'],
            DataKeys.frame.value: ObservedFrame.barycentric.value,
            DataKeys.normalized.value: False,
            DataKeys.snr.value: hdr['SNR']
        }
        # NOTE: HARPS S1D products store their main arrays in extension 1 as columns
        # (WAVE/FLUX/ERR). The primary HDU typically has no data payload.
        return arrays, info


class EsoNirpsSource(EsoTapSource):
    """ESO NIRPS spectra downloads."""

    name = "eso_nirps"
    extra_conditions = ("instrument_name='NIRPS'",)


class EsoUvesSource(EsoTapSource):
    """ESO UVES spectra downloads."""

    name = "eso_uves"
    extra_conditions = ("instrument_name='UVES'",)
