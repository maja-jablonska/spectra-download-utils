#!/usr/bin/env python3
"""Convert ELODIE spectra (and matched CCFs) into a contract-compliant Zarr store."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import zarr
from astropy.coordinates import Angle
from astropy.io import fits
from numcodecs import Blosc

from _eso_tabular_to_zarr import (
    _compute_chunks,
    _get_git_commit,
    _has_usable_series,
    _normalize_header_keyword,
    _safe_float,
    _safe_header_value,
    _safe_str,
    _string_array,
    _validate_before_commit as _validate_core_before_commit,
    _write_index_parquet,
)

_UNIFORM_HEADER_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "airmass": ("AIRMASS", "ESO TEL AIRM START"),
    "airmass_start": ("ESO TEL AIRM START", "AIRMASS"),
    "airmass_end": ("ESO TEL AIRM END",),
    "continuum_normalized": ("CONTNORM",),
    "header_date_utc": ("DATE",),
    "obs_date_utc": ("DATE-OBS",),
    "target_ra_deg_header": ("RA", "ALPHA"),
    "target_dec_deg_header": ("DEC", "DELTA"),
    "coord_equinox": ("EQUINOX",),
    "coord_frame": ("RADECSYS",),
    "instrument_header": ("INSTRUME",),
    "telescope_header": ("TELESCOP",),
    "observer_header": ("OBSERVER",),
    "origin_center": ("ORIGIN",),
    "object_header": ("OBJECT", "ESO OBS TARG NAME"),
    "target_name_header": ("ESO OBS TARG NAME", "OBJECT"),
    "program_id_header": ("PROG_ID", "ESO OBS PROG ID", "NOPROG"),
    "observation_id_header": ("OBID1", "ESO OBS ID"),
    "observation_name_header": ("ESO OBS NAME",),
    "observation_start_utc": ("ESO OBS START", "DATE-OBS"),
    "observation_technique": ("OBSTECH", "ESO DPR TECH", "IMATYP"),
    "product_category": ("PRODCATG",),
    "flux_calibration": ("FLUXCAL",),
    "spectral_reference_frame": ("SPECSYS", "CTYPE1"),
    "wavelength_min_nm_header": ("WAVELMIN",),
    "wavelength_max_nm_header": ("WAVELMAX",),
    "spectral_bin_nm_header": ("SPEC_BIN",),
    "spectral_resolution_header": ("SPEC_RES", "H_WRESOL"),
    "ncombine_header": ("NCOMBINE",),
    "exposure_time_s_header": ("EXPTIME", "TEXPTIME"),
    "total_exposure_time_s_header": ("TEXPTIME", "EXPTIME"),
    "mjd_start_header": ("MJD-OBS", "MJDATE", "MJD"),
    "mjd_end_header": ("MJD-END",),
    "utc_start_s_header": ("UTC",),
    "lst_start_s_header": ("LST",),
    "berv_kms_header": ("ESO DRS BERV", "BERV"),
    "berv_max_kms_header": ("ESO DRS BERVMX", "BERVMX"),
    "bjd_header": ("ESO DRS BJD", "BJD"),
    "telescope_alt_deg": ("ESO TEL ALT",),
    "telescope_az_deg": ("ESO TEL AZ",),
    "parallactic_angle_start_deg": ("ESO TEL PARANG START",),
    "parallactic_angle_end_deg": ("ESO TEL PARANG END",),
    "seeing_fwhm_start_arcsec": ("ESO TEL AMBI FWHM START",),
    "seeing_fwhm_end_arcsec": ("ESO TEL AMBI FWHM END",),
    "ambient_pressure_start_hpa": ("ESO TEL AMBI PRES START",),
    "ambient_pressure_end_hpa": ("ESO TEL AMBI PRES END",),
    "ambient_rhum_pct": ("ESO TEL AMBI RHUM",),
    "ambient_temp_c": ("ESO TEL AMBI TEMP",),
    "ambient_wind_dir_deg": ("ESO TEL AMBI WINDDIR",),
    "ambient_wind_speed_mps": ("ESO TEL AMBI WINDSP",),
    "pi_name_header": ("PI-COI", "ESO OBS PI-COI NAME"),
    "obs_category_header": ("ESO DPR CATG",),
    "obs_type_header": ("ESO DPR TYPE",),
    "template_id_header": ("ESO TPL ID",),
    "template_name_header": ("ESO TPL NAME",),
    "template_version_header": ("ESO TPL VERSION",),
    "template_start_utc_header": ("ESO TPL START",),
    "drs_version_header": ("ESO DRS VERSION",),
    "pipeline_software_header": ("PROCSOFT",),
    "archive_file_name_header": ("ARCFILE", "FILENAME", "H_AFR000"),
    "original_file_name_header": ("ORIGFILE", "PROV1", "FILENAME"),
    "ancillary_file_name_header": ("ASSON1",),
    "ancillary_file_md5_header": ("ASSOM1",),
    "wavelength_calibration_file_header": ("ESO DRS CAL TH FILE",),
    "flat_calibration_file_header": ("ESO DRS CAL FLAT FILE",),
    "blaze_calibration_file_header": ("ESO DRS BLAZE FILE",),
    "order_location_file_header": ("ESO DRS CAL LOC FILE", "IMALOC"),
    "wavelength_lamp_header": ("ESO DRS CAL TH LAMP USED",),
    "telescope_operator_header": ("ESO TEL OPER",),
    "dome_status_header": ("ESO TEL DOME STATUS",),
    "tracking_status_header": ("ESO TEL TRAK STATUS",),
    "moon_ra_deg_header": ("ESO TEL MOON RA",),
    "moon_dec_deg_header": ("ESO TEL MOON DEC",),
    "target_radial_velocity_kms_header": ("ESO TEL TARG RADVEL",),
    "target_pm_ra_header": ("ESO TEL TARG PMA", "PROPERM1"),
    "target_pm_dec_header": ("ESO TEL TARG PMD", "PROPERM2"),
    "target_parallax_header": ("ESO TEL TARG PARALLAX",),
    "target_epoch_header": ("ESO TEL TARG EPOCH",),
    "target_equinox_header": ("ESO TEL TARG EQUINOX", "EQUINOX"),
    "site_lat_deg_header": ("ESO TEL GEOLAT",),
    "site_lon_deg_header": ("ESO TEL GEOLON",),
    "site_elevation_m_header": ("ESO TEL GEOELEV",),
    "flux_capture_complete_header": ("TOT_FLUX",),
    "extended_object_header": ("EXT_OBJ",),
    "single_exposure_header": ("SINGLEXP",),
    "multi_epoch_header": ("M_EPOCH",),
    "product_level_header": ("PRODLVL",),
    "num_detector_chips_header": ("ESO DET CHIPS",),
    "detector_bits_header": ("ESO DET BITS",),
    "detector_gain_e_per_adu_header": ("GAIN", "ESO DRS CCD CONAD", "CCDGAIN"),
    "detector_read_noise_e_header": ("DETRON", "ESO DRS CCD SIGDET"),
    "detector_read_mode_header": ("ESO DET READ MODE",),
    "detector_read_speed_header": ("ESO DET READ SPEED",),
    "instrument_mode_header": ("ESO INS MODE",),
    "optical_path_header": ("ESO INS PATH",),
}

_RE_SPEC_NAME = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.fits$", re.IGNORECASE)
_RE_CCF_NAME = re.compile(r"^(?P<prefix>.+)_ccf_(?P<idx>\d+)\.fits$", re.IGNORECASE)
_RE_ELODIE_SPEC_TAG = re.compile(r"elodie:(?P<date>\d{8})/(?P<num>\d+)", re.IGNORECASE)
_RE_ELODIE_OBJ_TAG = re.compile(r"n(?P<date>\d{8}).*?obj(?P<num>\d+)\.fits", re.IGNORECASE)
_RE_ELODIE_OBJ_NUM = re.compile(r"obj(?P<num>\d+)\.fits", re.IGNORECASE)


@dataclass
class CCFRow:
    file_name: str
    pair_keys: tuple[str, ...]
    velocity: np.ndarray
    profile: np.ndarray
    velocity_units: str


@dataclass
class SpectrumRow:
    file_name: str
    spec_id: str
    object_name: str
    wavelength_raw: np.ndarray
    flux_raw: np.ndarray
    uncertainty_raw: np.ndarray
    continuum_model: np.ndarray | None
    continuum_uncertainty: np.ndarray | None
    continuum_method: str
    normalized_flux: np.ndarray | None
    normalized_uncertainty: np.ndarray | None
    exptime: float
    mjdate: float
    ra_deg: float
    dec_deg: float
    snr: float
    date_obs: str
    gaia_id: str
    gaia_dr: str
    flux_units: str
    header_metadata: dict[str, str]
    pair_keys: tuple[str, ...]
    ccf_velocity: np.ndarray | None
    ccf_profile: np.ndarray | None
    ccf_file_name: str

def _extract_hdu_header_metadata(hdus: fits.HDUList) -> dict[str, str]:
    metadata: dict[str, str] = {}
    skip = {"", "COMMENT", "HISTORY", "END"}

    header = getattr(hdus[0], "header", None)
    if header is None:
        return metadata

    raw_values: dict[str, str] = {}
    for card in header.cards:
        raw_key = (card.keyword or "").strip()
        if raw_key in skip:
            continue
        key = _normalize_header_keyword(raw_key)
        value = _safe_header_value(card.value)
        if key not in raw_values or (not raw_values[key] and value):
            raw_values[key] = value

    for field, candidates in _UNIFORM_HEADER_FIELD_MAP.items():
        selected = ""
        for candidate in candidates:
            value = raw_values.get(candidate)
            if value:
                selected = value
                break
        metadata[field] = selected

    # ELODIE legacy products expose some contract-equivalent values only via
    # CRVAL/CD/CUNIT and JDB1/JDB2; derive those fields when direct keywords
    # are unavailable.
    def _set_if_empty(field: str, value: str) -> None:
        if value and not metadata.get(field, ""):
            metadata[field] = value

    def _safe_float_text(value: str) -> float:
        try:
            if value == "":
                return float("nan")
            return float(value)
        except Exception:
            return float("nan")

    def _format_float(value: float) -> str:
        if np.isfinite(value):
            return f"{value:.12g}"
        return ""

    crval1 = _safe_float_text(raw_values.get("CRVAL1", ""))
    cd1 = _safe_float_text(raw_values.get("CD1_1", ""))
    if not np.isfinite(cd1):
        cd1 = _safe_float_text(raw_values.get("CDELT1", ""))
    naxis1 = _safe_float_text(raw_values.get("NAXIS1", ""))
    cunit1 = raw_values.get("CUNIT1", "").lower().replace(" ", "")

    unit_scale = float("nan")
    if "0.1nm" in cunit1:
        unit_scale = 0.1
    elif "nm" in cunit1:
        unit_scale = 1.0

    if np.isfinite(crval1) and np.isfinite(cd1) and np.isfinite(naxis1) and np.isfinite(unit_scale):
        n_pix = int(max(0, round(naxis1)))
        if n_pix > 0:
            wmin_nm = crval1 * unit_scale
            wbin_nm = cd1 * unit_scale
            wmax_nm = (crval1 + (n_pix - 1) * cd1) * unit_scale
            _set_if_empty("wavelength_min_nm_header", _format_float(wmin_nm))
            _set_if_empty("spectral_bin_nm_header", _format_float(wbin_nm))
            _set_if_empty("wavelength_max_nm_header", _format_float(wmax_nm))

    jdb1 = _safe_float_text(raw_values.get("JDB1", ""))
    jdb2 = _safe_float_text(raw_values.get("JDB2", ""))
    if np.isfinite(jdb1) and np.isfinite(jdb2):
        _set_if_empty("bjd_header", _format_float(jdb1 + jdb2))

    return metadata


def _safe_angle_deg(value: Any, *, unit: u.UnitBase) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except Exception:
        pass
    try:
        return float(Angle(value, unit=unit).degree)
    except Exception:
        return float("nan")


def _safe_ra_from_pos1(pos1: Any) -> float:
    value = _safe_float(pos1)
    if not np.isfinite(value):
        return float("nan")
    # ELODIE POS1 is typically in hours; convert to degrees when it looks like hour angle.
    if -24.0 <= value <= 24.0:
        return value * 15.0
    return value


def _normalize_units(value: Any, *, default: str) -> str:
    text = _safe_str(value).strip()
    if not text:
        return default
    if text.lower() == "instrumental":
        return "instrumental"
    return text.replace(" ", "_")


def _build_linear_axis(length: int, header: fits.Header) -> np.ndarray:
    crval1 = header.get("CRVAL1")
    cdelt1 = header.get("CDELT1")
    crpix1 = header.get("CRPIX1", 1.0)
    if crval1 is None or cdelt1 is None:
        return np.arange(length, dtype=np.float32)
    x = np.arange(length, dtype=np.float64)
    axis = float(crval1) + (x - (float(crpix1) - 1.0)) * float(cdelt1)
    return axis.astype(np.float32)


def _extract_pair_keys(path_name: str, header_filename: str, *, is_ccf: bool) -> tuple[str, ...]:
    keys: list[str] = []
    name = path_name.strip()
    hdr_name = header_filename.strip()

    if is_ccf:
        name_match = _RE_CCF_NAME.match(name)
    else:
        name_match = _RE_SPEC_NAME.match(name)
        if name_match and name_match.group("prefix").lower().endswith("_ccf"):
            name_match = None
    if name_match:
        prefix = name_match.group("prefix").strip().lower()
        idx = int(name_match.group("idx"))
        keys.append(f"name:{prefix}:{idx}")
        keys.append(f"idx:{idx}")

    spec_match = _RE_ELODIE_SPEC_TAG.search(hdr_name)
    if spec_match:
        date = spec_match.group("date")
        num = int(spec_match.group("num"))
        keys.append(f"elodie:{date}:{num}")
        keys.append(f"idx:{num}")

    obj_match = _RE_ELODIE_OBJ_TAG.search(hdr_name)
    if obj_match:
        date = obj_match.group("date")
        num = int(obj_match.group("num"))
        keys.append(f"elodie:{date}:{num}")
        keys.append(f"idx:{num}")

    num_match = _RE_ELODIE_OBJ_NUM.search(hdr_name)
    if num_match:
        num = int(num_match.group("num"))
        keys.append(f"idx:{num}")

    ordered: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return tuple(ordered)


def _looks_like_ccf(hdus: fits.HDUList, file_name: str) -> bool:
    if "_ccf_" in file_name.lower():
        return True
    if len(hdus) == 0:
        return False
    header = hdus[0].header
    if _safe_str(header.get("IMATYP")).strip().upper() == "COR":
        return True
    if len(hdus) == 1 and getattr(hdus[0], "data", None) is not None:
        arr = np.asarray(hdus[0].data)
        if arr.ndim == 1:
            cunit = _safe_str(header.get("CUNIT")).lower()
            if "km/s" in cunit or "cor" in cunit:
                return True
            filename = _safe_str(header.get("FILENAME")).lower()
            if "/cor/" in filename or "obj" in filename:
                return True
    return False


def _parse_ccf_hdus(fits_path: Path, hdus: fits.HDUList) -> CCFRow:
    if len(hdus) < 1:
        raise ValueError("expected at least 1 HDU for CCF")
    header = hdus[0].header
    data = getattr(hdus[0], "data", None)
    if data is None:
        raise ValueError("CCF primary HDU has no data")
    profile = np.asarray(data, dtype=np.float32).ravel()
    if profile.size == 0:
        raise ValueError("CCF array is empty")
    velocity = _build_linear_axis(profile.size, header)

    cunit = _safe_str(header.get("CUNIT")).lower()
    velocity_units = "km/s" if "km/s" in cunit else "km/s"

    pair_keys = _extract_pair_keys(
        fits_path.name,
        _safe_str(header.get("FILENAME", "")),
        is_ccf=True,
    )
    return CCFRow(
        file_name=fits_path.name,
        pair_keys=pair_keys,
        velocity=velocity,
        profile=profile,
        velocity_units=velocity_units,
    )


def _parse_spectrum_hdus(fits_path: Path, hdus: fits.HDUList) -> SpectrumRow:
    if len(hdus) < 1:
        raise ValueError("expected at least 1 HDU")
    header = hdus[0].header
    flux_data = getattr(hdus[0], "data", None)
    if flux_data is None:
        raise ValueError("primary HDU has no flux array")
    flux_raw = np.asarray(flux_data, dtype=np.float32).ravel()
    if flux_raw.size == 0:
        raise ValueError("empty flux array")

    if len(hdus) >= 3 and getattr(hdus[2], "data", None) is not None:
        uncertainty_raw = np.asarray(hdus[2].data, dtype=np.float32).ravel()
    else:
        uncertainty_raw = np.full_like(flux_raw, np.nan, dtype=np.float32)

    n_pix = min(flux_raw.size, uncertainty_raw.size)
    if n_pix == 0:
        raise ValueError("empty aligned arrays")

    wavelength_raw = _build_linear_axis(n_pix, header)
    flux_raw = flux_raw[:n_pix]
    uncertainty_raw = uncertainty_raw[:n_pix]

    ra_deg = _safe_float(header.get("RA"))
    if not np.isfinite(ra_deg):
        ra_deg = _safe_angle_deg(header.get("ALPHA"), unit=u.hourangle)
    if not np.isfinite(ra_deg):
        ra_deg = _safe_ra_from_pos1(header.get("POS1"))
    dec_deg = _safe_float(header.get("DEC"))
    if not np.isfinite(dec_deg):
        dec_deg = _safe_angle_deg(header.get("DELTA"), unit=u.deg)
    if not np.isfinite(dec_deg):
        dec_deg = _safe_float(header.get("POS2"))

    spec_id = _safe_str(header.get("FILENAME", "")).strip() or fits_path.stem
    object_name = _safe_str(header.get("OBJECT", "")).strip() or fits_path.stem
    date_obs = _safe_str(header.get("DATE-OBS", header.get("DATE", "")))
    flux_units = _normalize_units(header.get("BUNIT"), default="instrumental")

    pair_keys = _extract_pair_keys(
        fits_path.name,
        _safe_str(header.get("FILENAME", "")),
        is_ccf=False,
    )

    return SpectrumRow(
        file_name=fits_path.name,
        spec_id=spec_id,
        object_name=object_name,
        wavelength_raw=wavelength_raw,
        flux_raw=flux_raw,
        uncertainty_raw=uncertainty_raw,
        continuum_model=None,
        continuum_uncertainty=None,
        continuum_method="none",
        normalized_flux=None,
        normalized_uncertainty=None,
        exptime=_safe_float(header.get("EXPTIME")),
        mjdate=_safe_float(header.get("MJD-OBS", header.get("MJD"))),
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        snr=_safe_float(header.get("SNR", header.get("SNR_R", header.get("SN")))),
        date_obs=date_obs,
        gaia_id=_safe_str(header.get("GAIAID", "")).strip(),
        gaia_dr=_safe_str(header.get("GAIADR", "")).strip(),
        flux_units=flux_units,
        header_metadata=_extract_hdu_header_metadata(hdus),
        pair_keys=pair_keys,
        ccf_velocity=None,
        ccf_profile=None,
        ccf_file_name="",
    )

def _load_records(fits_dir: Path) -> tuple[list[SpectrumRow], list[str], int]:
    spectra: list[SpectrumRow] = []
    ccfs: list[CCFRow] = []
    skipped: list[str] = []

    for fits_path in sorted(fits_dir.glob("*.fits")):
        try:
            with fits.open(fits_path, memmap=False) as hdus:
                if _looks_like_ccf(hdus, fits_path.name):
                    ccfs.append(_parse_ccf_hdus(fits_path, hdus))
                else:
                    spectra.append(_parse_spectrum_hdus(fits_path, hdus))
        except Exception as exc:
            skipped.append(f"{fits_path.name}: {exc}")

    ccf_by_key: dict[str, CCFRow] = {}
    for ccf in ccfs:
        for key in ccf.pair_keys:
            if key and key not in ccf_by_key:
                ccf_by_key[key] = ccf

    used_ccf_files: set[str] = set()
    matched = 0
    for row in spectra:
        hit: CCFRow | None = None
        for key in row.pair_keys:
            candidate = ccf_by_key.get(key)
            if candidate is None:
                continue
            if candidate.file_name in used_ccf_files:
                continue
            hit = candidate
            break
        if hit is None:
            continue
        used_ccf_files.add(hit.file_name)
        row.ccf_velocity = hit.velocity
        row.ccf_profile = hit.profile
        row.ccf_file_name = hit.file_name
        matched += 1

    return spectra, skipped, matched


def _validate_before_commit(root: zarr.Group, expected_n_spec: int, index_path: Path) -> None:
    _validate_core_before_commit(root, expected_n_spec=expected_n_spec, index_path=index_path)

    reps = root["representations"]

    has_ccf_vel = "ccf_velocity" in reps
    has_ccf_prof = "ccf_profile" in reps
    if has_ccf_vel != has_ccf_prof:
        raise ValueError("representations/ccf_velocity and representations/ccf_profile must be written together")
    if has_ccf_vel:
        ccf_velocity = reps["ccf_velocity"]
        ccf_profile = reps["ccf_profile"]
        if ccf_velocity.ndim != 2 or ccf_profile.ndim != 2:
            raise ValueError("CCF arrays must be 2D")
        if ccf_velocity.shape != ccf_profile.shape:
            raise ValueError("CCF velocity/profile shapes must match")
        if ccf_velocity.shape[0] != expected_n_spec:
            raise ValueError("CCF arrays must align on N_spec")


def build_elodie_zarr(
    fits_dir: Path,
    output_zarr: Path,
    *,
    overwrite: bool = False,
    ingest_pipeline: str = "elodie_raw_ingest_v2",
) -> None:
    records, skipped, matched_ccf = _load_records(fits_dir)
    if not records:
        raise RuntimeError(f"No readable ELODIE spectrum FITS files found in {fits_dir}")

    n_spec = len(records)
    n_pix = max(r.flux_raw.size for r in records)
    n_ccf_pix = max((r.ccf_profile.size for r in records if r.ccf_profile is not None), default=0)
    chunks = _compute_chunks(n_spec, n_pix)
    ccf_chunks = _compute_chunks(n_spec, max(1, n_ccf_pix)) if n_ccf_pix > 0 else None
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    tmp_zarr = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp_zarr.exists():
        shutil.rmtree(tmp_zarr)
    final_exists = output_zarr.exists()
    if final_exists and not overwrite:
        raise FileExistsError(f"Output already exists: {output_zarr}. Use --overwrite to replace it.")

    try:
        # Use Zarr v2 for stable string dtype encoding across readers.
        root = zarr.open_group(str(tmp_zarr), mode="w", zarr_format=2)
        root.attrs["schema_version"] = "2.0"
        root.attrs["ingest_pipeline"] = ingest_pipeline
        root.attrs["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        root.attrs["git_commit"] = _get_git_commit()

        signal = root.create_group("signal")
        continuum = root.create_group("continuum")
        reps = root.create_group("representations")
        params = root.create_group("params")
        metadata = root.create_group("metadata")

        wave_arr = signal.create_array(
            "wavelength", shape=(n_spec, n_pix), chunks=chunks, dtype="f4", compressors=codec, fill_value=np.nan
        )
        flux_arr = signal.create_array(
            "flux_raw", shape=(n_spec, n_pix), chunks=chunks, dtype="f4", compressors=codec, fill_value=np.nan
        )
        unc_arr = signal.create_array(
            "uncertainty_raw", shape=(n_spec, n_pix), chunks=chunks, dtype="f4", compressors=codec, fill_value=np.nan
        )

        cont_model_arr = continuum.create_array(
            "continuum_model", shape=(n_spec, n_pix), chunks=chunks, dtype="f4", compressors=codec, fill_value=np.nan
        )
        cont_unc_arr = continuum.create_array(
            "continuum_uncertainty", shape=(n_spec, n_pix), chunks=chunks, dtype="f4", compressors=codec, fill_value=np.nan
        )
        continuum.create_array("method", data=_string_array([r.continuum_method for r in records]))

        has_norm_flux = any(_has_usable_series(r.normalized_flux) for r in records)
        norm_flux_arr = None
        norm_unc_arr = None
        if has_norm_flux:
            norm_flux_arr = reps.create_array(
                "normalized_flux", shape=(n_spec, n_pix), chunks=chunks, dtype="f4", compressors=codec, fill_value=np.nan
            )
            norm_unc_arr = reps.create_array(
                "normalized_uncertainty",
                shape=(n_spec, n_pix),
                chunks=chunks,
                dtype="f4",
                compressors=codec,
                fill_value=np.nan,
            )

        ccf_velocity_arr = None
        ccf_profile_arr = None
        if n_ccf_pix > 0 and ccf_chunks is not None:
            ccf_velocity_arr = reps.create_array(
                "ccf_velocity",
                shape=(n_spec, n_ccf_pix),
                chunks=ccf_chunks,
                dtype="f4",
                compressors=codec,
                fill_value=np.nan,
            )
            ccf_profile_arr = reps.create_array(
                "ccf_profile",
                shape=(n_spec, n_ccf_pix),
                chunks=ccf_chunks,
                dtype="f4",
                compressors=codec,
                fill_value=np.nan,
            )

        flux_units = next((r.flux_units for r in records if r.flux_units), "instrumental")
        ccf_velocity_units = "km/s"

        wave_arr.attrs["units"] = "Angstrom"
        flux_arr.attrs["units"] = flux_units
        unc_arr.attrs["units"] = flux_units
        cont_model_arr.attrs["units"] = flux_units
        cont_unc_arr.attrs["units"] = flux_units
        if norm_flux_arr is not None and norm_unc_arr is not None:
            norm_flux_arr.attrs["units"] = "dimensionless"
            norm_unc_arr.attrs["units"] = "dimensionless"
        if ccf_velocity_arr is not None and ccf_profile_arr is not None:
            ccf_velocity_arr.attrs["units"] = ccf_velocity_units
            ccf_profile_arr.attrs["units"] = "dimensionless"

        spec_id = [r.spec_id for r in records]
        object_name = [r.object_name for r in records]
        file_name = [r.file_name for r in records]
        source_type = ["observed"] * n_spec
        flux_type = ["flux"] * n_spec
        instrument = ["elodie"] * n_spec
        simulator = [""] * n_spec
        snr = np.asarray([r.snr for r in records], dtype=np.float32)
        exptime = np.asarray([r.exptime for r in records], dtype=np.float32)
        mjdate = np.asarray([r.mjdate for r in records], dtype=np.float64)
        ra_deg = np.asarray([r.ra_deg for r in records], dtype=np.float64)
        dec_deg = np.asarray([r.dec_deg for r in records], dtype=np.float64)
        date_obs = [r.date_obs for r in records]
        gaia_id = [r.gaia_id for r in records]
        gaia_dr = [r.gaia_dr for r in records]
        ccf_file_name = [r.ccf_file_name for r in records]
        has_ccf = np.asarray([bool(r.ccf_profile is not None) for r in records], dtype=np.bool_)

        metadata.create_array("spec_id", data=_string_array(spec_id))
        metadata.create_array("object_name", data=_string_array(object_name))
        metadata.create_array("file_name", data=_string_array(file_name))
        metadata.create_array("source_type", data=_string_array(source_type))
        metadata.create_array("flux_type", data=_string_array(flux_type))
        metadata.create_array("instrument", data=_string_array(instrument))
        metadata.create_array("simulator", data=_string_array(simulator))
        metadata.create_array("date_obs", data=_string_array(date_obs))
        metadata.create_array("gaia_id", data=_string_array(gaia_id))
        metadata.create_array("gaia_dr", data=_string_array(gaia_dr))
        metadata.create_array("ccf_file_name", data=_string_array(ccf_file_name))
        metadata.create_array("has_ccf", data=has_ccf, chunks=(max(1, min(4096, n_spec)),), compressors=codec)

        snr_arr = metadata.create_array("snr", data=snr, chunks=(max(1, min(4096, n_spec)),), compressors=codec)
        exptime_arr = metadata.create_array(
            "exptime", data=exptime, chunks=(max(1, min(4096, n_spec)),), compressors=codec
        )
        mjdate_arr = metadata.create_array(
            "mjdate", data=mjdate, chunks=(max(1, min(4096, n_spec)),), compressors=codec
        )
        ra_deg_arr = metadata.create_array(
            "ra_deg", data=ra_deg, chunks=(max(1, min(4096, n_spec)),), compressors=codec
        )
        dec_deg_arr = metadata.create_array(
            "dec_deg", data=dec_deg, chunks=(max(1, min(4096, n_spec)),), compressors=codec
        )

        snr_arr.attrs["units"] = "dimensionless"
        exptime_arr.attrs["units"] = "s"
        mjdate_arr.attrs["units"] = "d"
        ra_deg_arr.attrs["units"] = "deg"
        dec_deg_arr.attrs["units"] = "deg"

        for key in _UNIFORM_HEADER_FIELD_MAP:
            values = [row.header_metadata.get(key, "") for row in records]
            metadata.create_array(key, data=_string_array(values))

        for i, row in enumerate(records):
            n = row.flux_raw.size
            wave_arr[i, :n] = row.wavelength_raw[:n]
            flux_arr[i, :n] = row.flux_raw[:n]
            unc_arr[i, :n] = row.uncertainty_raw[:n]

            if row.continuum_model is not None:
                m = min(n, row.continuum_model.size)
                cont_model_arr[i, :m] = row.continuum_model[:m]
            if row.continuum_uncertainty is not None:
                m = min(n, row.continuum_uncertainty.size)
                cont_unc_arr[i, :m] = row.continuum_uncertainty[:m]

            normalized_flux = row.normalized_flux
            if norm_flux_arr is not None and _has_usable_series(normalized_flux):
                m = min(n, normalized_flux.size)
                norm_flux_arr[i, :m] = normalized_flux[:m]
                normalized_uncertainty = row.normalized_uncertainty
                if norm_unc_arr is not None and normalized_uncertainty is not None and normalized_uncertainty.size > 0:
                    mu = min(m, normalized_uncertainty.size)
                    norm_unc_arr[i, :mu] = normalized_uncertainty[:mu]

            if ccf_profile_arr is not None and ccf_velocity_arr is not None:
                if row.ccf_profile is not None and row.ccf_velocity is not None:
                    m = min(n_ccf_pix, row.ccf_profile.size, row.ccf_velocity.size)
                    ccf_profile_arr[i, :m] = row.ccf_profile[:m]
                    ccf_velocity_arr[i, :m] = row.ccf_velocity[:m]

        index_columns = {
            "spec_id": spec_id,
            "source_type": source_type,
            "teff": np.full((n_spec,), np.nan, dtype=np.float32),
            "logg": np.full((n_spec,), np.nan, dtype=np.float32),
            "feh": np.full((n_spec,), np.nan, dtype=np.float32),
            "instrument": instrument,
            "simulator": simulator,
            "snr": snr,
        }
        index_path = tmp_zarr / "index.parquet"
        _write_index_parquet(index_path, index_columns)

        _validate_before_commit(root, expected_n_spec=n_spec, index_path=index_path)

        if final_exists:
            shutil.rmtree(output_zarr)
        os.replace(tmp_zarr, output_zarr)
    except Exception:
        if tmp_zarr.exists():
            shutil.rmtree(tmp_zarr)
        raise
    finally:
        if skipped:
            print(f"Skipped {len(skipped)} file(s):")
            for item in skipped:
                print(f"  - {item}")
        print(f"Matched CCFs: {matched_ccf}/{n_spec}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fits_dir", type=Path, help="Directory containing ELODIE FITS files (spectra and CCF).")
    p.add_argument("output_zarr", type=Path, help="Output dataset path ending in .zarr.")
    p.add_argument("--overwrite", action="store_true", help="Replace output dataset if it already exists.")
    p.add_argument(
        "--ingest-pipeline",
        default="elodie_raw_ingest_v2",
        help="Value written to root attrs as ingest_pipeline.",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    fits_dir = args.fits_dir
    output_zarr = args.output_zarr

    if not fits_dir.exists() or not fits_dir.is_dir():
        print(f"Input directory not found: {fits_dir}", file=sys.stderr)
        return 2
    if output_zarr.suffix != ".zarr":
        print("Output path must end with .zarr", file=sys.stderr)
        return 2

    try:
        build_elodie_zarr(
            fits_dir=fits_dir,
            output_zarr=output_zarr,
            overwrite=args.overwrite,
            ingest_pipeline=args.ingest_pipeline,
        )
    except Exception as exc:
        print(f"Failed to build dataset: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote ELODIE dataset to {output_zarr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
