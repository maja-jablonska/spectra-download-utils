#!/usr/bin/env python3
"""Convert raw ESPaDOnS FITS spectra into a contract-compliant Zarr store."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from astropy.io import fits

from _eso_tabular_to_zarr import (
    PipelineConfig,
    SpectrumRow,
    _extract_hdu_header_metadata,
    _has_usable_series,
    _safe_float,
    _safe_str,
    build_tabular_zarr,
    run_cli,
)


def _load_espadons_records(fits_dir: Path, config: PipelineConfig) -> tuple[list[SpectrumRow], list[str]]:
    records: list[SpectrumRow] = []
    skipped: list[str] = []

    for fits_path in sorted(fits_dir.glob("*.fits")):
        try:
            with fits.open(fits_path, memmap=False) as hdus:
                data = hdus[0].data
                header = hdus[0].header
                header_metadata = _extract_hdu_header_metadata(hdus)

                if data is None or getattr(data, "ndim", 0) != 2 or data.shape[0] < 3:
                    skipped.append(f"{fits_path.name}: unsupported shape")
                    continue

                has_norm = data.shape[0] >= 3
                has_unnorm = data.shape[0] >= 6

                # ESPaDOnS convention:
                # rows 0-2 contain normalized spectra, rows 3-5 contain unnormalized spectra.
                if has_unnorm:
                    wavelength_raw = np.asarray(data[3], dtype=np.float32).ravel()
                    flux_raw = np.asarray(data[4], dtype=np.float32).ravel()
                    uncertainty_raw = np.asarray(data[5], dtype=np.float32).ravel()
                else:
                    wavelength_raw = np.asarray(data[0], dtype=np.float32).ravel()
                    flux_raw = np.asarray(data[1], dtype=np.float32).ravel()
                    uncertainty_raw = np.asarray(data[2], dtype=np.float32).ravel()

                normalized_flux = None
                normalized_uncertainty = None
                if has_norm:
                    norm_flux_candidate = np.asarray(data[1], dtype=np.float32).ravel()
                    if _has_usable_series(norm_flux_candidate):
                        normalized_flux = norm_flux_candidate
                        norm_unc_candidate = np.asarray(data[2], dtype=np.float32).ravel()
                        if norm_unc_candidate.size > 0:
                            normalized_uncertainty = norm_unc_candidate

                n_pix = min(wavelength_raw.size, flux_raw.size, uncertainty_raw.size)
                if n_pix == 0:
                    skipped.append(f"{fits_path.name}: empty arrays")
                    continue

                object_name = _safe_str(header.get("OBJNAME", header.get("OBJECT", ""))).strip() or fits_path.stem
                spec_id = _safe_str(header.get("FILENAME", header.get("ARCFILE", ""))).strip() or fits_path.stem

                records.append(
                    SpectrumRow(
                        file_name=fits_path.name,
                        spec_id=spec_id,
                        object_name=object_name,
                        wavelength_raw=wavelength_raw[:n_pix],
                        flux_raw=flux_raw[:n_pix],
                        uncertainty_raw=uncertainty_raw[:n_pix],
                        continuum_model=None,
                        continuum_uncertainty=None,
                        continuum_method="none",
                        normalized_flux=normalized_flux,
                        normalized_uncertainty=normalized_uncertainty,
                        exptime=_safe_float(header.get("EXPTIME", header.get("TEXPTIME"))),
                        mjdate=_safe_float(header.get("MJDATE", header.get("MJD-OBS"))),
                        ra_deg=_safe_float(header.get("RA_DEG", header.get("RA"))),
                        dec_deg=_safe_float(header.get("DEC_DEG", header.get("DEC"))),
                        snr=_safe_float(header.get("SNR", header.get("SNR_R"))),
                        date_obs=_safe_str(header.get("DATE-OBS", header.get("DATE", ""))),
                        gaia_id=_safe_str(header.get("GAIAID", "")).strip(),
                        gaia_dr=_safe_str(header.get("GAIADR", "")).strip(),
                        flux_units=config.flux_units_default,
                        header_metadata=header_metadata,
                    )
                )
        except Exception as exc:
            skipped.append(f"{fits_path.name}: {exc}")

    return records, skipped


CONFIG = PipelineConfig(
    instrument="espadons",
    ingest_pipeline_default="espadons_raw_ingest_v2",
    wave_candidates=("WAVE", "WAVELENGTH", "LAMBDA", "LAMBDA_AIR", "AWAV"),
    flux_candidates=("FLUX", "INTENSITY", "SCI_FLUX", "SPEC_FLUX"),
    uncertainty_candidates=("ERR", "ERROR", "FLUX_ERR", "SIGMA", "UNCERTAINTY"),
    flux_units_default="relative_flux",
    wavelength_units="nm",
    normalized_units="relative_flux",
    load_records_fn=_load_espadons_records,
)


def build_espadons_zarr(
    fits_dir: Path,
    output_zarr: Path,
    *,
    overwrite: bool = False,
    ingest_pipeline: str = CONFIG.ingest_pipeline_default,
) -> None:
    build_tabular_zarr(
        config=CONFIG,
        fits_dir=fits_dir,
        output_zarr=output_zarr,
        overwrite=overwrite,
        ingest_pipeline=ingest_pipeline,
    )


def main() -> int:
    return run_cli(config=CONFIG, description=__doc__ or "")


if __name__ == "__main__":
    raise SystemExit(main())
