"""Helpers for extracting arrays from FITS bytes.

This module is intentionally "optional-dependency-friendly": it only imports
`astropy` inside functions. If `astropy` is not installed, extraction functions
will raise a RuntimeError with an install hint.
"""

from __future__ import annotations

import io
from typing import Any, Dict, Optional, Tuple


def _first_header_value(headers: list["Any"], keys: tuple[str, ...]) -> Any:
    for hdr in headers:
        for key in keys:
            if key in hdr:
                value = hdr.get(key)
                if value is not None and value != "":
                    return value
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _parse_ra_deg(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    try:
        from astropy.coordinates import Angle  # type: ignore
        import astropy.units as u  # type: ignore
    except Exception:
        return None
    for unit in (u.hourangle, u.deg):
        try:
            return Angle(value, unit=unit).degree
        except Exception:
            continue
    return None


def _parse_dec_deg(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    try:
        from astropy.coordinates import Angle  # type: ignore
        import astropy.units as u  # type: ignore
    except Exception:
        return None
    for unit in (u.deg, u.hourangle):
        try:
            return Angle(value, unit=unit).degree
        except Exception:
            continue
    return None


def extract_data_keys_from_fits_bytes(fits_bytes: bytes) -> Dict[str, Any]:
    """Best-effort extraction of DataKeys from FITS headers."""

    try:
        from astropy.io import fits  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "FITS header parsing requires `astropy`. Install it (e.g. `pip install astropy`)."
        ) from exc

    with fits.open(io.BytesIO(fits_bytes)) as hdul:
        headers = [getattr(hdu, "header", {}) or {} for hdu in hdul]

    data: Dict[str, Any] = {}

    exptime = _coerce_float(_first_header_value(headers, ("EXPTIME", "TEXPTIME")))
    if exptime is not None:
        data["exptime"] = exptime

    ra_val = _first_header_value(headers, ("RA", "RA_DEG", "ALPHA"))
    ra_deg = _parse_ra_deg(ra_val)
    if ra_deg is not None:
        data["ra"] = ra_deg

    dec_val = _first_header_value(headers, ("DEC", "DEC_DEG", "DELTA"))
    dec_deg = _parse_dec_deg(dec_val)
    if dec_deg is not None:
        data["dec"] = dec_deg

    date_val = _first_header_value(headers, ("DATE-OBS", "DATE"))
    if date_val is not None:
        data["date"] = str(date_val)

    ref_frame = _first_header_value(headers, ("RADECSYS", "RADESYS", "REFSYS"))
    if ref_frame is not None:
        data["reference_frame"] = str(ref_frame)

    ref_epoch = _first_header_value(headers, ("EQUINOX", "EPOCH"))
    if ref_epoch is not None:
        data["reference_frame_epoch"] = ref_epoch

    mjd_val = _coerce_float(_first_header_value(headers, ("MJD-OBS", "MJD", "MJDOBS")))
    if mjd_val is not None:
        data["mjd"] = mjd_val

    airmass = _coerce_float(_first_header_value(headers, ("AIRMASS",)))
    if airmass is not None:
        data["airmass"] = airmass

    obj = _first_header_value(headers, ("OBJECT", "OBJNAME", "TARGET"))
    if obj is not None:
        data["object"] = str(obj)

    berv = _coerce_float(_first_header_value(headers, ("BERV", "BARYCORR")))
    if berv is not None:
        data["berv"] = berv

    snr = _coerce_float(_first_header_value(headers, ("SNR", "SNRMED", "SNR_MEAN", "SNR_MEAN")))
    if snr is not None:
        data["snr"] = snr

    normalized = _coerce_bool(_first_header_value(headers, ("NORMALIZED", "NORMED", "NORM", "NORMALIZ")))
    if normalized is not None:
        data["normalized"] = normalized

    frame = _first_header_value(headers, ("FRAME", "OBSFRAME", "REF_FRAME"))
    if frame is not None:
        data["frame"] = str(frame)

    return data


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

