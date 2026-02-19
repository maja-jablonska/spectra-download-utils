#!/usr/bin/env python3
"""Convert raw HARPS FITS spectra into a contract-compliant Zarr store."""

from __future__ import annotations

from pathlib import Path

from _eso_tabular_to_zarr import PipelineConfig, build_tabular_zarr, run_cli

CONFIG = PipelineConfig(
    instrument="harps",
    ingest_pipeline_default="harps_raw_ingest_v2",
    wave_candidates=("WAVE", "WAVELENGTH", "LAMBDA", "LAMBDA_AIR", "AWAV"),
    flux_candidates=("FLUX", "INTENSITY", "SCI_FLUX", "SPEC_FLUX"),
    uncertainty_candidates=("ERR", "ERROR", "FLUX_ERR", "SIGMA", "UNCERTAINTY"),
    flux_units_default="adu",
)


def build_harps_zarr(
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
