"""Shared converter for ESO-style table FITS spectra to contract Zarr stores."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import zarr
from astropy.io import fits
from numcodecs import Blosc

_UNIFORM_HEADER_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "airmass": ("AIRMASS", "ESO TEL AIRM START"),
    "airmass_start": ("ESO TEL AIRM START", "AIRMASS"),
    "airmass_end": ("ESO TEL AIRM END",),
    "continuum_normalized": ("CONTNORM",),
    "header_date_utc": ("DATE",),
    "obs_date_utc": ("DATE-OBS",),
    "target_ra_deg_header": ("RA",),
    "target_dec_deg_header": ("DEC",),
    "coord_equinox": ("EQUINOX",),
    "coord_frame": ("RADECSYS", "RADESYS"),
    "instrument_header": ("INSTRUME",),
    "telescope_header": ("TELESCOP",),
    "observer_header": ("OBSERVER",),
    "origin_center": ("ORIGIN",),
    "object_header": ("OBJECT", "ESO OBS TARG NAME"),
    "target_name_header": ("ESO OBS TARG NAME", "OBJECT"),
    "program_id_header": ("PROG_ID", "ESO OBS PROG ID"),
    "observation_id_header": ("OBID1", "ESO OBS ID"),
    "observation_name_header": ("ESO OBS NAME",),
    "observation_start_utc": ("ESO OBS START", "DATE-OBS"),
    "observation_technique": ("OBSTECH", "ESO DPR TECH", "EXPTYPE"),
    "product_category": ("PRODCATG",),
    "flux_calibration": ("FLUXCAL",),
    "spectral_reference_frame": ("SPECSYS",),
    "wavelength_min_nm_header": ("WAVELMIN",),
    "wavelength_max_nm_header": ("WAVELMAX",),
    "spectral_bin_nm_header": ("SPEC_BIN",),
    "spectral_resolution_header": ("SPEC_RES",),
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
    "obs_type_header": ("ESO DPR TYPE", "EXPTYPE"),
    "template_id_header": ("ESO TPL ID", "TPL ID"),
    "template_name_header": ("ESO TPL NAME", "TPL NAME"),
    "template_version_header": ("ESO TPL VERSION", "TPL VERSION"),
    "template_start_utc_header": ("ESO TPL START", "TPL START"),
    "drs_version_header": ("ESO DRS VERSION", "ESO PRO REC1 PIPE ID", "ESO PRO REC1 DRS ID"),
    "pipeline_software_header": ("PROCSOFT", "PIPEFILE"),
    "archive_file_name_header": ("ARCFILE",),
    "original_file_name_header": ("ORIGFILE", "PROV1"),
    "ancillary_file_name_header": ("ASSON1",),
    "ancillary_file_md5_header": ("ASSOM1",),
    "wavelength_calibration_file_header": ("ESO DRS CAL TH FILE",),
    "flat_calibration_file_header": ("ESO DRS CAL FLAT FILE",),
    "blaze_calibration_file_header": ("ESO DRS BLAZE FILE",),
    "order_location_file_header": ("ESO DRS CAL LOC FILE",),
    "wavelength_lamp_header": ("ESO DRS CAL TH LAMP USED",),
    "telescope_operator_header": ("ESO TEL OPER",),
    "dome_status_header": ("ESO TEL DOME STATUS",),
    "tracking_status_header": ("ESO TEL TRAK STATUS",),
    "moon_ra_deg_header": ("ESO TEL MOON RA",),
    "moon_dec_deg_header": ("ESO TEL MOON DEC",),
    "target_radial_velocity_kms_header": ("ESO TEL TARG RADVEL",),
    "target_pm_ra_header": ("ESO TEL TARG PMA",),
    "target_pm_dec_header": ("ESO TEL TARG PMD",),
    "target_parallax_header": ("ESO TEL TARG PARALLAX",),
    "target_epoch_header": ("ESO TEL TARG EPOCH",),
    "target_equinox_header": ("ESO TEL TARG EQUINOX",),
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
    "detector_gain_e_per_adu_header": ("GAIN", "ESO DRS CCD CONAD"),
    "detector_read_noise_e_header": ("DETRON", "ESO DRS CCD SIGDET"),
    "detector_read_mode_header": ("ESO DET READ MODE",),
    "detector_read_speed_header": ("ESO DET READ SPEED", "ESO DET READ CLOCK"),
    "instrument_mode_header": ("ESO INS MODE",),
    "optical_path_header": ("ESO INS PATH",),
}


@dataclass(frozen=True)
class PipelineConfig:
    instrument: str
    ingest_pipeline_default: str
    wave_candidates: tuple[str, ...]
    flux_candidates: tuple[str, ...]
    uncertainty_candidates: tuple[str, ...]
    flux_units_default: str = "instrumental"
    wavelength_units: str = "Angstrom"
    normalized_units: str = "dimensionless"
    load_records_fn: Callable[[Path, "PipelineConfig"], tuple[list["SpectrumRow"], list[str]]] | None = None


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


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_units(value: Any, *, default: str) -> str:
    text = _safe_str(value).strip()
    if not text:
        return default
    return text.replace(" ", "_")


def _has_usable_series(values: np.ndarray | None) -> bool:
    if values is None or values.size == 0:
        return False
    return bool(np.isfinite(values).any())


def _safe_header_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, (bool, np.bool_)):
        return "T" if bool(value) else "F"
    return str(value).strip()


def _normalize_header_keyword(raw_key: str) -> str:
    key = raw_key.strip().upper()
    if key.startswith("HIERARCH "):
        key = key[len("HIERARCH ") :]
    return key


def _extract_hdu_header_metadata(hdus: fits.HDUList) -> dict[str, str]:
    metadata: dict[str, str] = {}
    skip = {"", "COMMENT", "HISTORY", "END"}

    raw_values: dict[str, str] = {}
    for hdu in hdus:
        header = getattr(hdu, "header", None)
        if header is None:
            continue
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

    return metadata


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _string_array(values: list[str]) -> np.ndarray:
    encoded = [v.encode("utf-8", errors="replace") for v in values]
    max_len = max(1, max((len(v) for v in encoded), default=1))
    return np.asarray(encoded, dtype=f"S{max_len}")


def _compute_chunks(n_spec: int, n_pix: int) -> tuple[int, int]:
    if n_pix <= 10_000:
        return max(1, min(1024, n_spec)), n_pix
    return max(1, min(256, n_spec)), max(1, min(4096, n_pix))


def _write_index_parquet(index_path: Path, columns: dict[str, Any]) -> None:
    try:
        import polars as pl  # type: ignore

        pl.DataFrame(columns).write_parquet(index_path)
        return
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(columns).to_parquet(index_path, index=False)
        return
    except Exception:
        pass

    np.savez(index_path.with_suffix(".npz"), **columns)


def _read_index_len(index_path: Path) -> int:
    if index_path.exists():
        try:
            import polars as pl  # type: ignore

            return pl.read_parquet(index_path).height
        except Exception:
            pass

        try:
            import pandas as pd  # type: ignore

            return len(pd.read_parquet(index_path))
        except Exception as exc:
            raise RuntimeError(
                "Failed to read index.parquet for parity check. Install `polars` or `pandas` with parquet support."
            ) from exc

    if index_path.with_suffix(".npz").exists():
        arr = np.load(index_path.with_suffix(".npz"), allow_pickle=False)
        if "spec_id" not in arr:
            raise RuntimeError("index fallback NPZ missing spec_id")
        return int(len(arr["spec_id"]))

    raise RuntimeError("index.parquet and fallback index.npz are both missing")


def _build_column_lookup(hdu_data: Any) -> dict[str, str]:
    names = getattr(hdu_data, "names", None)
    if not names:
        dtype = getattr(hdu_data, "dtype", None)
        names = getattr(dtype, "names", []) if dtype is not None else []
    return {str(name).strip().upper(): str(name) for name in names}


def _extract_optional_with_key(hdu_data: Any, candidates: tuple[str, ...]) -> tuple[np.ndarray, str] | None:
    if hdu_data is None:
        return None
    lookup = _build_column_lookup(hdu_data)
    for key in candidates:
        hit = lookup.get(key.strip().upper())
        if hit is not None:
            return np.asarray(hdu_data[hit]).ravel(), key
    return None


def _load_records(fits_dir: Path, config: PipelineConfig) -> tuple[list[SpectrumRow], list[str]]:
    records: list[SpectrumRow] = []
    skipped: list[str] = []

    for fits_path in sorted(fits_dir.glob("*.fits")):
        try:
            with fits.open(fits_path, memmap=False) as hdus:
                if len(hdus) < 2:
                    skipped.append(f"{fits_path.name}: expected at least 2 HDUs")
                    continue

                header = hdus[0].header
                table = hdus[1].data
                header_metadata = _extract_hdu_header_metadata(hdus)

                wavelength_hit = _extract_optional_with_key(table, config.wave_candidates)
                if wavelength_hit is None:
                    raise ValueError("missing wavelength column in HDU[1]")
                flux_hit = _extract_optional_with_key(table, config.flux_candidates)
                if flux_hit is None:
                    raise ValueError("missing flux column in HDU[1]")

                wavelength_raw = wavelength_hit[0].astype(np.float32, copy=False)
                flux_raw = flux_hit[0].astype(np.float32, copy=False)

                err_hit = _extract_optional_with_key(table, config.uncertainty_candidates)
                if err_hit is not None:
                    uncertainty_raw = err_hit[0].astype(np.float32, copy=False)
                else:
                    uncertainty_raw = np.full_like(flux_raw, np.nan, dtype=np.float32)

                continuum_hit = _extract_optional_with_key(table, ("CONTINUUM", "CONT", "BLAZE"))
                continuum_model = continuum_hit[0].astype(np.float32, copy=False) if continuum_hit is not None else None
                continuum_method = f"column:{continuum_hit[1]}" if continuum_hit is not None else "none"

                continuum_unc_hit = _extract_optional_with_key(
                    table, ("CONT_ERR", "CONTINUUM_ERR", "ERR_CONT", "BLAZE_ERR")
                )
                continuum_uncertainty = (
                    continuum_unc_hit[0].astype(np.float32, copy=False) if continuum_unc_hit is not None else None
                )

                norm_flux_hit = _extract_optional_with_key(table, ("NORMALIZED_FLUX", "NORM_FLUX", "FLUX_NORM"))
                normalized_flux = None
                if norm_flux_hit is not None:
                    norm_flux_candidate = norm_flux_hit[0].astype(np.float32, copy=False)
                    if _has_usable_series(norm_flux_candidate):
                        normalized_flux = norm_flux_candidate

                norm_unc_hit = _extract_optional_with_key(table, ("NORMALIZED_ERR", "NORM_ERR", "ERR_NORM"))
                normalized_uncertainty = None
                if normalized_flux is not None and norm_unc_hit is not None:
                    norm_unc_candidate = norm_unc_hit[0].astype(np.float32, copy=False)
                    if norm_unc_candidate.size > 0:
                        normalized_uncertainty = norm_unc_candidate

                n_pix = min(wavelength_raw.size, flux_raw.size, uncertainty_raw.size)
                if n_pix == 0:
                    skipped.append(f"{fits_path.name}: empty arrays")
                    continue

                object_name = _safe_str(header.get("OBJECT", "")).strip() or fits_path.stem
                spec_id = _safe_str(header.get("ARCFILE", "")).strip() or fits_path.stem
                flux_units = _normalize_units(header.get("BUNIT"), default=config.flux_units_default)

                records.append(
                    SpectrumRow(
                        file_name=fits_path.name,
                        spec_id=spec_id,
                        object_name=object_name,
                        wavelength_raw=wavelength_raw[:n_pix],
                        flux_raw=flux_raw[:n_pix],
                        uncertainty_raw=uncertainty_raw[:n_pix],
                        continuum_model=continuum_model,
                        continuum_uncertainty=continuum_uncertainty,
                        continuum_method=continuum_method,
                        normalized_flux=normalized_flux,
                        normalized_uncertainty=normalized_uncertainty,
                        exptime=_safe_float(header.get("EXPTIME", header.get("TEXPTIME"))),
                        mjdate=_safe_float(header.get("MJD-OBS")),
                        ra_deg=_safe_float(header.get("RA")),
                        dec_deg=_safe_float(header.get("DEC")),
                        snr=_safe_float(header.get("SNR")),
                        date_obs=_safe_str(header.get("DATE-OBS", "")),
                        gaia_id=_safe_str(header.get("GAIAID", "")).strip(),
                        gaia_dr=_safe_str(header.get("GAIADR", "")).strip(),
                        flux_units=flux_units,
                        header_metadata=header_metadata,
                    )
                )
        except Exception as exc:
            skipped.append(f"{fits_path.name}: {exc}")

    return records, skipped


def _validate_before_commit(root: zarr.Group, expected_n_spec: int, index_path: Path) -> None:
    for g in ("signal", "continuum", "representations", "params", "metadata"):
        if g not in root:
            raise ValueError(f"Missing required group: {g}")

    signal = root["signal"]
    continuum = root["continuum"]
    reps = root["representations"]

    for key in ("wavelength", "flux_raw", "uncertainty_raw"):
        if key not in signal:
            raise ValueError(f"Missing required array: signal/{key}")

    for key in ("continuum_model", "continuum_uncertainty", "method"):
        if key not in continuum:
            raise ValueError(f"Missing required array: continuum/{key}")

    flux = signal["flux_raw"]
    wave = signal["wavelength"]
    raw_unc = signal["uncertainty_raw"]

    if flux.ndim != 2:
        raise ValueError("signal/flux_raw must be 2D")
    nspec, npix = flux.shape
    if nspec != expected_n_spec:
        raise ValueError(f"N_spec mismatch: expected={expected_n_spec} got={nspec}")

    if wave.ndim == 1:
        if wave.shape[0] != npix:
            raise ValueError("signal/wavelength length != N_pix")
    elif wave.ndim == 2:
        if wave.shape != flux.shape:
            raise ValueError("signal/wavelength must match signal/flux_raw shape")
    else:
        raise ValueError("signal/wavelength must be 1D or 2D")

    if raw_unc.shape != flux.shape:
        raise ValueError("signal/uncertainty_raw must match signal/flux_raw shape")

    if continuum["continuum_model"].shape != flux.shape:
        raise ValueError("continuum/continuum_model must match signal/flux_raw shape")
    if continuum["continuum_uncertainty"].shape != flux.shape:
        raise ValueError("continuum/continuum_uncertainty must match signal/flux_raw shape")
    if continuum["method"].shape != (nspec,):
        raise ValueError("continuum/method must have shape (N_spec,)")

    has_norm_flux = "normalized_flux" in reps
    has_norm_unc = "normalized_uncertainty" in reps
    if has_norm_flux != has_norm_unc:
        raise ValueError(
            "representations/normalized_flux and representations/normalized_uncertainty "
            "must be written together"
        )
    if has_norm_flux:
        if reps["normalized_flux"].shape != flux.shape:
            raise ValueError("representations/normalized_flux must match signal/flux_raw shape")
        if reps["normalized_uncertainty"].shape != flux.shape:
            raise ValueError("representations/normalized_uncertainty must match signal/flux_raw shape")

    for name, arr in root["params"].arrays():
        if arr.shape != (nspec,):
            raise ValueError(f"params/{name} misaligned: {arr.shape}")

    for name, arr in root["metadata"].arrays():
        if arr.shape != (nspec,):
            raise ValueError(f"metadata/{name} misaligned: {arr.shape}")

    for i in range(nspec):
        row = flux[i]
        if not np.isfinite(row).any():
            raise ValueError(f"signal/flux_raw[{i}] is all-NaN/non-finite")
        finite = np.where(np.isfinite(row), row, 0.0)
        if not np.any(finite != 0.0):
            raise ValueError(f"signal/flux_raw[{i}] is all-zero")

    if not index_path.exists() and not index_path.with_suffix(".npz").exists():
        raise ValueError("index.parquet (or fallback index.npz) is missing")
    index_len = _read_index_len(index_path)
    if index_len != nspec:
        raise ValueError(f"index.parquet mismatch: len(index)={index_len} N_spec={nspec}")


def build_tabular_zarr(
    *,
    config: PipelineConfig,
    fits_dir: Path,
    output_zarr: Path,
    overwrite: bool = False,
    ingest_pipeline: str | None = None,
) -> None:
    pipeline_name = ingest_pipeline or config.ingest_pipeline_default
    loader = config.load_records_fn or _load_records
    records, skipped = loader(fits_dir, config)
    if not records:
        raise RuntimeError(f"No readable {config.instrument.upper()} FITS files found in {fits_dir}")

    n_spec = len(records)
    n_pix = max(r.flux_raw.size for r in records)
    chunks = _compute_chunks(n_spec, n_pix)
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    tmp_zarr = output_zarr.with_name(f"{output_zarr.name}.tmp")
    if tmp_zarr.exists():
        shutil.rmtree(tmp_zarr)
    final_exists = output_zarr.exists()
    if final_exists and not overwrite:
        raise FileExistsError(f"Output already exists: {output_zarr}. Use --overwrite to replace it.")

    try:
        root = zarr.open_group(str(tmp_zarr), mode="w", zarr_format=2)
        root.attrs["schema_version"] = "2.0"
        root.attrs["ingest_pipeline"] = pipeline_name
        root.attrs["created_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        root.attrs["git_commit"] = _get_git_commit()

        signal = root.create_group("signal")
        continuum = root.create_group("continuum")
        reps = root.create_group("representations")
        root.create_group("params")
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

        wave_arr.attrs["units"] = config.wavelength_units
        flux_units = next((r.flux_units for r in records if r.flux_units), config.flux_units_default)
        flux_arr.attrs["units"] = flux_units
        unc_arr.attrs["units"] = flux_units
        cont_model_arr.attrs["units"] = flux_units
        cont_unc_arr.attrs["units"] = flux_units
        if norm_flux_arr is not None and norm_unc_arr is not None:
            norm_flux_arr.attrs["units"] = config.normalized_units
            norm_unc_arr.attrs["units"] = config.normalized_units

        spec_id = [r.spec_id for r in records]
        object_name = [r.object_name for r in records]
        file_name = [r.file_name for r in records]
        source_type = ["observed"] * n_spec
        flux_type = ["flux"] * n_spec
        instrument = [config.instrument] * n_spec
        simulator = [""] * n_spec
        snr = np.asarray([r.snr for r in records], dtype=np.float32)
        exptime = np.asarray([r.exptime for r in records], dtype=np.float32)
        mjdate = np.asarray([r.mjdate for r in records], dtype=np.float64)
        ra_deg = np.asarray([r.ra_deg for r in records], dtype=np.float64)
        dec_deg = np.asarray([r.dec_deg for r in records], dtype=np.float64)
        date_obs = [r.date_obs for r in records]
        gaia_id = [r.gaia_id for r in records]
        gaia_dr = [r.gaia_dr for r in records]

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
            wave_arr[i, :n] = row.wavelength_raw
            flux_arr[i, :n] = row.flux_raw
            unc_arr[i, :n] = row.uncertainty_raw

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


def run_cli(*, config: PipelineConfig, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("fits_dir", type=Path, help=f"Directory containing raw {config.instrument.upper()} FITS files.")
    parser.add_argument("output_zarr", type=Path, help="Output dataset path ending in .zarr.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output dataset if it already exists.")
    parser.add_argument(
        "--ingest-pipeline",
        default=config.ingest_pipeline_default,
        help="Value written to root attrs as ingest_pipeline.",
    )
    args = parser.parse_args()

    fits_dir = args.fits_dir
    output_zarr = args.output_zarr

    if not fits_dir.exists() or not fits_dir.is_dir():
        print(f"Input directory not found: {fits_dir}", file=sys.stderr)
        return 2
    if output_zarr.suffix != ".zarr":
        print("Output path must end with .zarr", file=sys.stderr)
        return 2

    try:
        build_tabular_zarr(
            config=config,
            fits_dir=fits_dir,
            output_zarr=output_zarr,
            overwrite=args.overwrite,
            ingest_pipeline=args.ingest_pipeline,
        )
    except Exception as exc:
        print(f"Failed to build dataset: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {config.instrument.upper()} dataset to {output_zarr}")
    return 0
