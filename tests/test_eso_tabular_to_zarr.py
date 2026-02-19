import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    from astropy.io import fits  # type: ignore
except Exception:  # pragma: no cover
    fits = None  # type: ignore

try:
    import zarr  # type: ignore
except Exception:  # pragma: no cover
    zarr = None  # type: ignore

import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _eso_tabular_to_zarr as tabular  # type: ignore

from _eso_tabular_to_zarr import (  # type: ignore
    PipelineConfig,
    _extract_hdu_header_metadata,
    _extract_optional_with_key,
    _load_records,
    build_tabular_zarr,
)


DEFAULT_CONFIG = PipelineConfig(
    instrument="testinst",
    ingest_pipeline_default="test_pipeline",
    wave_candidates=("WAVE", "WAVELENGTH", "LAMBDA", "LAMBDA_AIR", "AWAV"),
    flux_candidates=("FLUX", "INTENSITY", "SCI_FLUX", "SPEC_FLUX"),
    uncertainty_candidates=("ERR", "ERROR", "FLUX_ERR", "SIGMA", "UNCERTAINTY"),
)


def _write_table_fits(path: Path, columns: dict[str, np.ndarray], header: dict[str, object], ext_header: dict[str, object] | None = None) -> None:
    if fits is None:
        raise RuntimeError("astropy not available")

    n = int(next(iter(columns.values())).size)
    cols = [
        fits.Column(name=name, format=f"{n}D", array=[np.asarray(values, dtype=np.float64)])
        for name, values in columns.items()
    ]

    primary = fits.PrimaryHDU()
    for k, v in header.items():
        primary.header[k] = v

    table = fits.BinTableHDU.from_columns(cols)
    if ext_header:
        for k, v in ext_header.items():
            table.header[k] = v

    fits.HDUList([primary, table]).writeto(path, overwrite=True)


class TestEsoTabularCore(unittest.TestCase):
    def setUp(self) -> None:
        if fits is None or zarr is None:
            self.skipTest("astropy/zarr not installed")

    def test_extract_optional_with_key_is_case_insensitive(self) -> None:
        data = np.array([(1.0, 2.0)], dtype=[("Wave", "f8"), ("Flux", "f8")])
        hit = _extract_optional_with_key(data, ("WAVE",))
        self.assertIsNotNone(hit)
        assert hit is not None
        arr, key = hit
        self.assertEqual(key, "WAVE")
        self.assertEqual(arr.shape, (1,))
        self.assertAlmostEqual(float(arr[0]), 1.0)

    def test_load_records_fallbacks_err_to_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fits_path = Path(tmp) / "sample.fits"
            _write_table_fits(
                fits_path,
                columns={
                    "wave": np.array([5000.0, 5001.0, 5002.0]),
                    "flux": np.array([1.0, 2.0, 3.0]),
                },
                header={
                    "OBJECT": "OBJ",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJD-OBS": 61041.0,
                    "EXPTIME": 100.0,
                    "RA": 120.0,
                    "DEC": -30.0,
                },
            )

            records, skipped = _load_records(Path(tmp), DEFAULT_CONFIG)
            self.assertEqual(skipped, [])
            self.assertEqual(len(records), 1)
            self.assertTrue(np.isnan(records[0].uncertainty_raw).all())

    def test_header_metadata_reads_hierarch_cards_from_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fits_path = Path(tmp) / "sample.fits"
            _write_table_fits(
                fits_path,
                columns={
                    "WAVE": np.array([1.0, 2.0]),
                    "FLUX": np.array([3.0, 4.0]),
                },
                header={"OBJECT": "OBJ"},
                ext_header={"HIERARCH ESO OBS PROG ID": "60.A-9999(A)"},
            )

            with fits.open(fits_path, memmap=False) as hdus:  # type: ignore[arg-type]
                metadata = _extract_hdu_header_metadata(hdus)

            self.assertEqual(metadata["program_id_header"], "60.A-9999(A)")

    def test_build_tabular_zarr_omits_norm_unc_when_norm_flux_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "out.zarr"
            _write_table_fits(
                in_dir / "sample.fits",
                columns={
                    "WAVE": np.array([5000.0, 5001.0, 5002.0]),
                    "FLUX": np.array([1.0, 2.0, 3.0]),
                    "ERR": np.array([0.1, 0.1, 0.1]),
                    "NORMALIZED_ERR": np.array([0.01, 0.01, 0.01]),
                },
                header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00"},
            )

            build_tabular_zarr(config=DEFAULT_CONFIG, fits_dir=in_dir, output_zarr=out, overwrite=True)

            root = zarr.open_group(str(out), mode="r")
            reps = root["representations"]
            self.assertNotIn("normalized_flux", reps)
            self.assertNotIn("normalized_uncertainty", reps)

    def test_build_tabular_zarr_accepts_tiny_nonzero_flux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "out.zarr"
            _write_table_fits(
                in_dir / "sample.fits",
                columns={
                    "WAVE": np.array([5000.0, 5001.0, 5002.0]),
                    "FLUX": np.array([1e-12, -2e-12, 3e-12]),
                    "ERR": np.array([1e-13, 1e-13, 1e-13]),
                },
                header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00"},
            )

            build_tabular_zarr(config=DEFAULT_CONFIG, fits_dir=in_dir, output_zarr=out, overwrite=True)

            root = zarr.open_group(str(out), mode="r")
            flux = np.asarray(root["signal"]["flux_raw"][:])
            self.assertGreater(int(np.isfinite(flux).sum()), 0)

    def test_build_tabular_zarr_rejects_all_zero_flux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "out.zarr"
            _write_table_fits(
                in_dir / "sample.fits",
                columns={
                    "WAVE": np.array([5000.0, 5001.0, 5002.0]),
                    "FLUX": np.array([0.0, 0.0, 0.0]),
                    "ERR": np.array([0.1, 0.1, 0.1]),
                },
                header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00"},
            )

            with self.assertRaisesRegex(ValueError, "all-zero"):
                build_tabular_zarr(config=DEFAULT_CONFIG, fits_dir=in_dir, output_zarr=out, overwrite=True)

    def test_build_tabular_zarr_accepts_npz_index_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "out.zarr"
            _write_table_fits(
                in_dir / "sample.fits",
                columns={
                    "WAVE": np.array([5000.0, 5001.0, 5002.0]),
                    "FLUX": np.array([1.0, 2.0, 3.0]),
                    "ERR": np.array([0.1, 0.1, 0.1]),
                },
                header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00"},
            )

            def _write_npz(index_path: Path, columns: dict[str, object]) -> None:
                coerced = {k: np.asarray(v) for k, v in columns.items()}
                np.savez(index_path.with_suffix(".npz"), **coerced)

            with mock.patch.object(tabular, "_write_index_parquet", side_effect=_write_npz):
                build_tabular_zarr(config=DEFAULT_CONFIG, fits_dir=in_dir, output_zarr=out, overwrite=True)

            self.assertTrue((out / "index.npz").exists())
            root = zarr.open_group(str(out), mode="r")
            self.assertEqual(tuple(root["signal"]["flux_raw"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
