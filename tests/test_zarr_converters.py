import tempfile
import unittest
from pathlib import Path

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

import feros_to_zarr  # type: ignore
import harps_to_zarr  # type: ignore
import nirps_to_zarr  # type: ignore
import espadons_to_zarr  # type: ignore
import elodie_to_zarr  # type: ignore
import uves_to_zarr  # type: ignore


def _write_table_fits(path: Path, columns: dict[str, np.ndarray], header: dict[str, object]) -> None:
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
    fits.HDUList([primary, table]).writeto(path, overwrite=True)


def _write_espadons_matrix_fits(path: Path, rows: np.ndarray, header: dict[str, object]) -> None:
    if fits is None:
        raise RuntimeError("astropy not available")

    primary = fits.PrimaryHDU(data=np.asarray(rows, dtype=np.float32))
    for k, v in header.items():
        primary.header[k] = v
    fits.HDUList([primary]).writeto(path, overwrite=True)


def _write_elodie_spectrum_fits(
    path: Path,
    flux: np.ndarray,
    header: dict[str, object],
    *,
    uncertainty: np.ndarray | None = None,
) -> None:
    if fits is None:
        raise RuntimeError("astropy not available")

    primary = fits.PrimaryHDU(data=np.asarray(flux, dtype=np.float32))
    for k, v in header.items():
        primary.header[k] = v

    hdus = [primary]
    if uncertainty is not None:
        hdus.append(fits.ImageHDU())
        hdus.append(fits.ImageHDU(data=np.asarray(uncertainty, dtype=np.float32)))

    fits.HDUList(hdus).writeto(path, overwrite=True)


def _assert_basic_output(test: unittest.TestCase, out_path: Path, expected_instrument: str, expected_n: int) -> None:
    root = zarr.open_group(str(out_path), mode="r")
    test.assertIn("signal", root)
    test.assertIn("metadata", root)
    signal = root["signal"]
    test.assertIn("wavelength", signal)
    test.assertIn("flux_raw", signal)
    test.assertIn("uncertainty_raw", signal)

    wave = np.asarray(signal["wavelength"][:])
    flux = np.asarray(signal["flux_raw"][:])
    unc = np.asarray(signal["uncertainty_raw"][:])

    test.assertEqual(flux.shape[0], 1)
    test.assertEqual(flux.shape[1], expected_n)
    test.assertEqual(wave.shape, flux.shape)
    test.assertEqual(unc.shape, flux.shape)
    test.assertGreater(int(np.isfinite(wave).sum()), 0)
    test.assertGreater(int(np.isfinite(flux).sum()), 0)

    inst = root["metadata"]["instrument"][:].astype("U")[0]
    test.assertEqual(inst, expected_instrument)


class TestZarrConverters(unittest.TestCase):
    def setUp(self) -> None:
        if fits is None or zarr is None:
            self.skipTest("astropy/zarr not installed")

    def test_harps_converter_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "harps.zarr"
            _write_table_fits(
                in_dir / "h.fits",
                columns={
                    "WAVE": np.array([5000.0, 5001.0, 5002.0]),
                    "FLUX": np.array([10.0, 11.0, 12.0]),
                    "ERR": np.array([0.1, 0.1, 0.1]),
                },
                header={
                    "OBJECT": "OBJ",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJD-OBS": 61041.0,
                    "TEXPTIME": 100.0,
                    "RA": 120.0,
                    "DEC": -30.0,
                    "SNR": 90.0,
                    "ARCFILE": "A.fits",
                },
            )

            harps_to_zarr.build_harps_zarr(in_dir, out, overwrite=True)
            _assert_basic_output(self, out, "harps", 3)

    def test_feros_converter_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "feros.zarr"
            _write_table_fits(
                in_dir / "f.fits",
                columns={
                    "wave": np.array([5000.0, 5001.0, 5002.0, 5003.0]),
                    "flux": np.array([1.0, 2.0, 3.0, 4.0]),
                    "err": np.array([0.2, 0.2, 0.2, 0.2]),
                },
                header={
                    "OBJECT": "OBJ",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJD-OBS": 61041.0,
                    "EXPTIME": 100.0,
                    "RA": 120.0,
                    "DEC": -30.0,
                    "SNR": 90.0,
                    "ARCFILE": "A.fits",
                },
            )

            feros_to_zarr.build_feros_zarr(in_dir, out, overwrite=True)
            _assert_basic_output(self, out, "feros", 4)

    def test_nirps_converter_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "nirps.zarr"
            _write_table_fits(
                in_dir / "n.fits",
                columns={
                    "WAVE": np.array([9700.0, 9700.5, 9701.0]),
                    "FLUX": np.array([1e-12, 2e-12, -1e-12]),
                    "ERR": np.array([1e-13, 1e-13, 1e-13]),
                },
                header={
                    "OBJECT": "OBJ",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJD-OBS": 61041.0,
                    "EXPTIME": 100.0,
                    "RA": 120.0,
                    "DEC": -30.0,
                    "SNR": 90.0,
                    "ARCFILE": "A.fits",
                },
            )

            nirps_to_zarr.build_nirps_zarr(in_dir, out, overwrite=True)
            _assert_basic_output(self, out, "nirps", 3)

    def test_uves_converter_smoke_prefers_reduced_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "uves.zarr"
            _write_table_fits(
                in_dir / "u.fits",
                columns={
                    "WAVE": np.array([4000.0, 4000.1, 4000.2]),
                    "FLUX_REDUCED": np.array([10.0, 20.0, 30.0]),
                    "ERR_REDUCED": np.array([0.1, 0.2, 0.3]),
                    "FLUX": np.array([1000.0, 1000.0, 1000.0]),
                    "ERR": np.array([9.0, 9.0, 9.0]),
                },
                header={
                    "OBJECT": "OBJ",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJD-OBS": 61041.0,
                    "EXPTIME": 100.0,
                    "RA": 120.0,
                    "DEC": -30.0,
                    "SNR": 90.0,
                    "ARCFILE": "A.fits",
                },
            )

            uves_to_zarr.build_uves_zarr(in_dir, out, overwrite=True)
            _assert_basic_output(self, out, "uves", 3)

            root = zarr.open_group(str(out), mode="r")
            flux = np.asarray(root["signal"]["flux_raw"][0, :3])
            self.assertTrue(np.allclose(flux, [10.0, 20.0, 30.0]))

    def test_elodie_converter_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "elodie.zarr"
            _write_elodie_spectrum_fits(
                in_dir / "e.fits",
                flux=np.array([10.0, 11.0, 12.0], dtype=np.float32),
                uncertainty=np.array([0.1, 0.1, 0.1], dtype=np.float32),
                header={
                    "OBJECT": "OBJ",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJD-OBS": 61041.0,
                    "EXPTIME": 100.0,
                    "RA": 120.0,
                    "DEC": -30.0,
                    "SNR": 90.0,
                    "FILENAME": "elodie:20260101/0001",
                    "PROG_ID": "60.A-9700(A)",
                },
            )

            elodie_to_zarr.build_elodie_zarr(in_dir, out, overwrite=True)
            _assert_basic_output(self, out, "elodie", 3)

    def test_missing_required_columns_raise_for_all_converters(self) -> None:
        converters = [
            ("harps", harps_to_zarr.build_harps_zarr),
            ("feros", feros_to_zarr.build_feros_zarr),
            ("nirps", nirps_to_zarr.build_nirps_zarr),
            ("uves", uves_to_zarr.build_uves_zarr),
        ]

        for instrument, builder in converters:
            with self.subTest(instrument=instrument):
                with tempfile.TemporaryDirectory() as tmp:
                    in_dir = Path(tmp) / "in"
                    in_dir.mkdir()
                    out = Path(tmp) / f"{instrument}.zarr"
                    _write_table_fits(
                        in_dir / "bad.fits",
                        columns={
                            "FLUX": np.array([1.0, 2.0, 3.0]),
                            "ERR": np.array([0.1, 0.1, 0.1]),
                        },
                        header={"OBJECT": "OBJ"},
                    )
                    with self.assertRaisesRegex(RuntimeError, "No readable"):
                        builder(in_dir, out, overwrite=True)

    def test_no_hdu1_raise_for_all_converters(self) -> None:
        converters = [
            ("harps", harps_to_zarr.build_harps_zarr),
            ("feros", feros_to_zarr.build_feros_zarr),
            ("nirps", nirps_to_zarr.build_nirps_zarr),
            ("uves", uves_to_zarr.build_uves_zarr),
        ]

        for instrument, builder in converters:
            with self.subTest(instrument=instrument):
                with tempfile.TemporaryDirectory() as tmp:
                    in_dir = Path(tmp) / "in"
                    in_dir.mkdir()
                    out = Path(tmp) / f"{instrument}.zarr"
                    fits.PrimaryHDU().writeto(in_dir / "bad.fits", overwrite=True)
                    with self.assertRaisesRegex(RuntimeError, "No readable"):
                        builder(in_dir, out, overwrite=True)

    def test_all_zero_flux_rejected_for_all_converters(self) -> None:
        converters = [
            ("harps", harps_to_zarr.build_harps_zarr, "FLUX", "ERR"),
            ("feros", feros_to_zarr.build_feros_zarr, "FLUX", "ERR"),
            ("nirps", nirps_to_zarr.build_nirps_zarr, "FLUX", "ERR"),
            ("uves", uves_to_zarr.build_uves_zarr, "FLUX_REDUCED", "ERR_REDUCED"),
        ]

        for instrument, builder, flux_col, err_col in converters:
            with self.subTest(instrument=instrument):
                with tempfile.TemporaryDirectory() as tmp:
                    in_dir = Path(tmp) / "in"
                    in_dir.mkdir()
                    out = Path(tmp) / f"{instrument}.zarr"
                    _write_table_fits(
                        in_dir / "z.fits",
                        columns={
                            "WAVE": np.array([5000.0, 5000.1, 5000.2]),
                            flux_col: np.array([0.0, 0.0, 0.0]),
                            err_col: np.array([0.1, 0.1, 0.1]),
                        },
                        header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00"},
                    )
                    with self.assertRaisesRegex(ValueError, "all-zero"):
                        builder(in_dir, out, overwrite=True)

    def test_all_nan_flux_rejected_for_all_converters(self) -> None:
        converters = [
            ("harps", harps_to_zarr.build_harps_zarr, "FLUX", "ERR"),
            ("feros", feros_to_zarr.build_feros_zarr, "FLUX", "ERR"),
            ("nirps", nirps_to_zarr.build_nirps_zarr, "FLUX", "ERR"),
            ("uves", uves_to_zarr.build_uves_zarr, "FLUX_REDUCED", "ERR_REDUCED"),
        ]

        for instrument, builder, flux_col, err_col in converters:
            with self.subTest(instrument=instrument):
                with tempfile.TemporaryDirectory() as tmp:
                    in_dir = Path(tmp) / "in"
                    in_dir.mkdir()
                    out = Path(tmp) / f"{instrument}.zarr"
                    _write_table_fits(
                        in_dir / "nan.fits",
                        columns={
                            "WAVE": np.array([5000.0, 5000.1, 5000.2]),
                            flux_col: np.array([np.nan, np.nan, np.nan]),
                            err_col: np.array([0.1, 0.1, 0.1]),
                        },
                        header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00"},
                    )
                    with self.assertRaisesRegex(ValueError, "all-NaN/non-finite"):
                        builder(in_dir, out, overwrite=True)

    def test_mismatched_lengths_across_spectra_are_padded(self) -> None:
        converters = [
            ("harps", harps_to_zarr.build_harps_zarr, "FLUX", "ERR"),
            ("feros", feros_to_zarr.build_feros_zarr, "FLUX", "ERR"),
            ("nirps", nirps_to_zarr.build_nirps_zarr, "FLUX", "ERR"),
            ("uves", uves_to_zarr.build_uves_zarr, "FLUX_REDUCED", "ERR_REDUCED"),
        ]

        for instrument, builder, flux_col, err_col in converters:
            with self.subTest(instrument=instrument):
                with tempfile.TemporaryDirectory() as tmp:
                    in_dir = Path(tmp) / "in"
                    in_dir.mkdir()
                    out = Path(tmp) / f"{instrument}.zarr"
                    _write_table_fits(
                        in_dir / "short.fits",
                        columns={
                            "WAVE": np.array([6000.0, 6000.1, 6000.2]),
                            flux_col: np.array([1.0, 2.0, 3.0]),
                            err_col: np.array([0.1, 0.1, 0.1]),
                        },
                        header={
                            "OBJECT": "OBJ1",
                            "DATE-OBS": "2026-01-01T00:00:00",
                            "MJD-OBS": 61041.0,
                            "EXPTIME": 100.0,
                            "TEXPTIME": 100.0,
                            "RA": 120.0,
                            "DEC": -30.0,
                            "SNR": 90.0,
                            "ARCFILE": "A1.fits",
                        },
                    )
                    _write_table_fits(
                        in_dir / "long.fits",
                        columns={
                            "WAVE": np.array([6100.0, 6100.1, 6100.2, 6100.3, 6100.4]),
                            flux_col: np.array([4.0, 5.0, 6.0, 7.0, 8.0]),
                            err_col: np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                        },
                        header={
                            "OBJECT": "OBJ2",
                            "DATE-OBS": "2026-01-02T00:00:00",
                            "MJD-OBS": 61042.0,
                            "EXPTIME": 110.0,
                            "TEXPTIME": 110.0,
                            "RA": 121.0,
                            "DEC": -31.0,
                            "SNR": 91.0,
                            "ARCFILE": "A2.fits",
                        },
                    )

                    builder(in_dir, out, overwrite=True)
                    root = zarr.open_group(str(out), mode="r")
                    flux = np.asarray(root["signal"]["flux_raw"][:])
                    wave = np.asarray(root["signal"]["wavelength"][:])
                    unc = np.asarray(root["signal"]["uncertainty_raw"][:])
                    self.assertEqual(flux.shape, (2, 5))
                    self.assertEqual(wave.shape, (2, 5))
                    self.assertEqual(unc.shape, (2, 5))
                    finite_per_row = [int(np.isfinite(row).sum()) for row in flux]
                    self.assertEqual(sorted(finite_per_row), [3, 5])

    def test_metadata_mapping_assertions_for_all_converters(self) -> None:
        converters = [
            ("harps", harps_to_zarr.build_harps_zarr, "FLUX", "ERR"),
            ("feros", feros_to_zarr.build_feros_zarr, "FLUX", "ERR"),
            ("nirps", nirps_to_zarr.build_nirps_zarr, "FLUX", "ERR"),
            ("uves", uves_to_zarr.build_uves_zarr, "FLUX_REDUCED", "ERR_REDUCED"),
        ]

        for instrument, builder, flux_col, err_col in converters:
            with self.subTest(instrument=instrument):
                with tempfile.TemporaryDirectory() as tmp:
                    in_dir = Path(tmp) / "in"
                    in_dir.mkdir()
                    out = Path(tmp) / f"{instrument}.zarr"
                    _write_table_fits(
                        in_dir / "m.fits",
                        columns={
                            "WAVE": np.array([5000.0, 5000.1, 5000.2]),
                            flux_col: np.array([2.0, 3.0, 4.0]),
                            err_col: np.array([0.2, 0.2, 0.2]),
                        },
                        header={
                            "OBJECT": "HD95456",
                            "DATE-OBS": "2026-02-01T07:20:25.740",
                            "MJD-OBS": 61072.30585347,
                            "EXPTIME": 450.0456,
                            "TEXPTIME": 450.0456,
                            "RA": 165.138776,
                            "DEC": -31.82874,
                            "SNR": 168.5486883812362,
                            "ARCFILE": "ADP.2026-02-11T11:30:18.046.fits",
                            "PROG_ID": "60.A-9700(A)",
                        },
                    )

                    builder(in_dir, out, overwrite=True)

                    root = zarr.open_group(str(out), mode="r")
                    md = root["metadata"]
                    instrument_value = md["instrument"][:].astype("U")[0]
                    object_name = md["object_name"][:].astype("U")[0]
                    program_id = md["program_id_header"][:].astype("U")[0]
                    ra_deg = float(np.asarray(md["ra_deg"][:])[0])
                    dec_deg = float(np.asarray(md["dec_deg"][:])[0])
                    snr = float(np.asarray(md["snr"][:])[0])

                    self.assertEqual(instrument_value, instrument)
                    self.assertEqual(object_name, "HD95456")
                    self.assertEqual(program_id, "60.A-9700(A)")
                    self.assertAlmostEqual(ra_deg, 165.138776, places=6)
                    self.assertAlmostEqual(dec_deg, -31.82874, places=6)
                    self.assertAlmostEqual(snr, 168.5486883812362, places=5)

    def test_espadons_omits_normalized_arrays_when_norm_flux_is_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "espadons.zarr"
            _write_espadons_matrix_fits(
                in_dir / "e.fits",
                rows=np.asarray(
                    [
                        [500.0, 500.1, 500.2],  # normalized wavelength
                        [np.nan, np.nan, np.nan],  # normalized flux (invalid)
                        [0.01, 0.01, 0.01],  # normalized uncertainty
                        [500.0, 500.1, 500.2],  # raw wavelength
                        [10.0, 11.0, 12.0],  # raw flux
                        [0.1, 0.1, 0.1],  # raw uncertainty
                    ],
                    dtype=np.float32,
                ),
                header={
                    "OBJNAME": "OBJ",
                    "FILENAME": "ESP001.fits",
                    "DATE-OBS": "2026-01-01T00:00:00",
                    "MJDATE": 61041.0,
                    "EXPTIME": 100.0,
                    "RA_DEG": 120.0,
                    "DEC_DEG": -30.0,
                    "SNR": 90.0,
                },
            )

            espadons_to_zarr.build_espadons_zarr(in_dir, out, overwrite=True)

            root = zarr.open_group(str(out), mode="r")
            self.assertEqual(root["metadata"]["instrument"][:].astype("U")[0], "espadons")
            self.assertEqual(root["signal"]["wavelength"].attrs.get("units"), "nm")
            self.assertEqual(root["signal"]["flux_raw"].attrs.get("units"), "relative_flux")
            reps = root["representations"]
            self.assertNotIn("normalized_flux", reps)
            self.assertNotIn("normalized_uncertainty", reps)

    def test_elodie_missing_primary_flux_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "elodie.zarr"
            fits.PrimaryHDU().writeto(in_dir / "bad.fits", overwrite=True)
            with self.assertRaisesRegex(RuntimeError, "No readable"):
                elodie_to_zarr.build_elodie_zarr(in_dir, out, overwrite=True)

    def test_elodie_all_zero_flux_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "elodie.zarr"
            _write_elodie_spectrum_fits(
                in_dir / "zero.fits",
                flux=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                uncertainty=np.array([0.1, 0.1, 0.1], dtype=np.float32),
                header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00", "FILENAME": "elodie:20260101/0002"},
            )
            with self.assertRaisesRegex(ValueError, "all-zero"):
                elodie_to_zarr.build_elodie_zarr(in_dir, out, overwrite=True)

    def test_elodie_all_nan_flux_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "elodie.zarr"
            _write_elodie_spectrum_fits(
                in_dir / "nan.fits",
                flux=np.array([np.nan, np.nan, np.nan], dtype=np.float32),
                uncertainty=np.array([0.1, 0.1, 0.1], dtype=np.float32),
                header={"OBJECT": "OBJ", "DATE-OBS": "2026-01-01T00:00:00", "FILENAME": "elodie:20260101/0003"},
            )
            with self.assertRaisesRegex(ValueError, "all-NaN/non-finite"):
                elodie_to_zarr.build_elodie_zarr(in_dir, out, overwrite=True)

    def test_elodie_metadata_mapping_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = Path(tmp) / "in"
            in_dir.mkdir()
            out = Path(tmp) / "elodie.zarr"
            _write_elodie_spectrum_fits(
                in_dir / "m.fits",
                flux=np.array([2.0, 3.0, 4.0], dtype=np.float32),
                uncertainty=np.array([0.2, 0.2, 0.2], dtype=np.float32),
                header={
                    "OBJECT": "HD95456",
                    "DATE-OBS": "2026-02-01T07:20:25.740",
                    "MJD-OBS": 61072.30585347,
                    "EXPTIME": 450.0456,
                    "RA": 165.138776,
                    "DEC": -31.82874,
                    "SNR": 168.5486883812362,
                    "FILENAME": "elodie:20260201/0004",
                    "PROG_ID": "60.A-9700(A)",
                },
            )

            elodie_to_zarr.build_elodie_zarr(in_dir, out, overwrite=True)

            root = zarr.open_group(str(out), mode="r")
            md = root["metadata"]
            instrument_value = md["instrument"][:].astype("U")[0]
            object_name = md["object_name"][:].astype("U")[0]
            program_id = md["program_id_header"][:].astype("U")[0]
            ra_deg = float(np.asarray(md["ra_deg"][:])[0])
            dec_deg = float(np.asarray(md["dec_deg"][:])[0])
            snr = float(np.asarray(md["snr"][:])[0])

            self.assertEqual(instrument_value, "elodie")
            self.assertEqual(object_name, "HD95456")
            self.assertEqual(program_id, "60.A-9700(A)")
            self.assertAlmostEqual(ra_deg, 165.138776, places=6)
            self.assertAlmostEqual(dec_deg, -31.82874, places=6)
            self.assertAlmostEqual(snr, 168.5486883812362, places=5)


if __name__ == "__main__":
    unittest.main()
