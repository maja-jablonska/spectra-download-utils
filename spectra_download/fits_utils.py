"""Helpers for extracting arrays from FITS bytes.

This module is intentionally "optional-dependency-friendly": it only imports
`astropy` inside functions. If `astropy` is not installed, extraction functions
will raise a RuntimeError with an install hint.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple


def extract_1d_wavelength_intensity_from_fits_bytes(
    fits_bytes: bytes,
) -> Tuple[Optional["Any"], Optional["Any"], Dict[str, Any]]:
    """Best-effort extraction of (wavelength, intensity) from FITS bytes.

    This is meant as a generic fallback. Individual sources should override the
    Zarr conversion hook for exact semantics.

    Strategy:
    - Prefer binary tables with columns matching common names for wavelength/flux.
    - Otherwise, if a 1D image is found, treat it as intensity and build a linear
      wavelength grid from standard WCS keywords when present.
    """

    try:
        import numpy as np  # type: ignore
        from astropy.io import fits  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "FITS extraction requires `astropy`. Install it (e.g. `pip install astropy`) "
            "or override spectrum_to_zarr_components in your source."
        ) from exc

    wl: Optional[Any] = None
    inten: Optional[Any] = None
    info: Dict[str, Any] = {}

    with fits.open(io.BytesIO(fits_bytes)) as hdul:
        info["hdus"] = len(hdul)

        # 1) Try binary tables for wavelength/flux columns.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None or not hasattr(data, "columns"):
                continue

            names = [n.lower() for n in getattr(data, "names", []) or []]
            if not names:
                continue

            def _find_name(candidates: tuple[str, ...]) -> Optional[str]:
                for cand in candidates:
                    if cand in names:
                        return cand
                # fuzzy match
                for n in names:
                    for cand in candidates:
                        if cand in n:
                            return n
                return None

            wl_name = _find_name(("wavelength", "lambda", "wave", "lam", "wl"))
            fl_name = _find_name(("intensity", "flux", "flx", "spec", "signal"))
            if wl_name and fl_name:
                wl = np.asarray(data[wl_name])
                inten = np.asarray(data[fl_name])
                info["extraction"] = "bintable"
                info["wavelength_column"] = wl_name
                info["intensity_column"] = fl_name
                return wl, inten, info

        # 2) Try 1D image + linear WCS keywords.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim != 1 or arr.size == 0:
                continue

            inten = arr.astype(float, copy=False)
            hdr = getattr(hdu, "header", {}) or {}
            crval1 = hdr.get("CRVAL1")
            cdelt1 = hdr.get("CDELT1")
            crpix1 = hdr.get("CRPIX1", 1.0)
            if crval1 is not None and cdelt1 is not None:
                # FITS WCS is 1-indexed.
                x = np.arange(inten.size, dtype=float)
                wl = (float(crval1) + (x - (float(crpix1) - 1.0)) * float(cdelt1)).astype(float)
                info["extraction"] = "image_wcs_linear"
                info["crval1"] = float(crval1)
                info["cdelt1"] = float(cdelt1)
                info["crpix1"] = float(crpix1)
            else:
                info["extraction"] = "image_no_wcs"
            return wl, inten, info

    return wl, inten, info


def extract_ccf_from_fits_bytes(
    fits_bytes: bytes,
) -> Tuple[Optional["Any"], Optional["Any"], Dict[str, Any]]:
    """Best-effort extraction of (velocity, ccf) from FITS bytes.

    Strategy:
    - Prefer binary tables with columns matching common names for velocity/RV and CCF.
    - Otherwise, if a 1D image is found, treat it as CCF values and build a linear
      velocity grid from standard WCS keywords when present.
    """

    try:
        import numpy as np  # type: ignore
        from astropy.io import fits  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CCF extraction requires `astropy`. Install it (e.g. `pip install astropy`) "
            "or override ccf_to_zarr_components in your source."
        ) from exc

    vel: Optional[Any] = None
    ccf: Optional[Any] = None
    info: Dict[str, Any] = {}

    with fits.open(io.BytesIO(fits_bytes)) as hdul:
        info["hdus"] = len(hdul)

        # 1) Try binary tables for velocity/ccf columns.
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None or not hasattr(data, "columns"):
                continue

            names = [n.lower() for n in getattr(data, "names", []) or []]
            if not names:
                continue

            def _find_name(candidates: tuple[str, ...]) -> Optional[str]:
                for cand in candidates:
                    if cand in names:
                        return cand
                for n in names:
                    for cand in candidates:
                        if cand in n:
                            return n
                return None

            vel_name = _find_name(("rv", "radvel", "v", "vel", "velocity", "vrad", "v_rad"))
            ccf_name = _find_name(("ccf", "correlation", "corr", "xcorr"))
            if vel_name and ccf_name:
                vel = np.asarray(data[vel_name])
                ccf = np.asarray(data[ccf_name])
                info["extraction"] = "bintable"
                info["velocity_column"] = vel_name
                info["ccf_column"] = ccf_name
                return vel, ccf, info

        # 2) Try 1D image + linear WCS keywords (treat axis as velocity).
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim != 1 or arr.size == 0:
                continue

            ccf = arr.astype(float, copy=False)
            hdr = getattr(hdu, "header", {}) or {}
            crval1 = hdr.get("CRVAL1")
            cdelt1 = hdr.get("CDELT1")
            crpix1 = hdr.get("CRPIX1", 1.0)
            if crval1 is not None and cdelt1 is not None:
                x = np.arange(ccf.size, dtype=float)
                vel = (float(crval1) + (x - (float(crpix1) - 1.0)) * float(cdelt1)).astype(float)
                info["extraction"] = "image_wcs_linear"
                info["crval1"] = float(crval1)
                info["cdelt1"] = float(cdelt1)
                info["crpix1"] = float(crpix1)
            else:
                info["extraction"] = "image_no_wcs"
            return vel, ccf, info

    return vel, ccf, info

