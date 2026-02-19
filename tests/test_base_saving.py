import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spectra_download.models import SpectrumRecord
from spectra_download.sources.base import SpectraSource


class DummySource(SpectraSource):
    name = "dummy"

    def build_request_url(self, identifier: str, extra_params: dict) -> str:  # type: ignore[override]
        return f"https://example.test/query?id={identifier}"

    def parse_response(self, payload: dict, identifier: str) -> list[SpectrumRecord]:  # type: ignore[override]
        return [
            SpectrumRecord(
                spectrum_id="spec1",
                source=self.name,
                metadata={"access_url": "https://example.test/spec1.fits", "identifier": identifier},
            )
        ]


class TestBaseSaving(unittest.TestCase):
    def test_normalize_download_extra_params_injects_kwargs(self) -> None:
        source = DummySource(timeout=1, max_retries=1)
        params = source._normalize_download_extra_params(
            {},
            raw_save_path="/tmp/raw",
            zarr_paths="/tmp/out.zarr",
            not_found_path="/tmp/not_found.jsonl",
            error_path="/tmp/errors.jsonl",
        )
        self.assertEqual(params["raw_save_path"], "/tmp/raw")
        self.assertEqual(params["zarr_paths"], "/tmp/out.zarr")
        self.assertEqual(params["not_found_path"], "/tmp/not_found.jsonl")
        self.assertEqual(params["error_path"], "/tmp/errors.jsonl")

    def test_normalize_download_extra_params_preserves_existing_values(self) -> None:
        source = DummySource(timeout=1, max_retries=1)
        params = source._normalize_download_extra_params(
            {
                "raw_save_path": "/already/raw",
                "zarr_paths": "/already/out.zarr",
                "not_found_path": "/already/not_found.jsonl",
                "error_path": "/already/errors.jsonl",
            },
            raw_save_path="/new/raw",
            zarr_paths="/new/out.zarr",
            not_found_path="/new/not_found.jsonl",
            error_path="/new/errors.jsonl",
        )
        self.assertEqual(params["raw_save_path"], "/already/raw")
        self.assertEqual(params["zarr_paths"], "/already/out.zarr")
        self.assertEqual(params["not_found_path"], "/already/not_found.jsonl")
        self.assertEqual(params["error_path"], "/already/errors.jsonl")

    def test_normalize_download_extra_params_respects_save_dir_and_save_path(self) -> None:
        source = DummySource(timeout=1, max_retries=1)
        params = source._normalize_download_extra_params(
            {"save_dir": "/legacy/raw", "save_path": "/legacy/out.zarr"},
            raw_save_path="/new/raw",
            zarr_paths="/new/out.zarr",
        )
        self.assertNotIn("raw_save_path", params)
        self.assertNotIn("zarr_paths", params)
        self.assertEqual(params["save_dir"], "/legacy/raw")
        self.assertEqual(params["save_path"], "/legacy/out.zarr")

    def test_raw_save_path_writes_fits_bytes_and_sets_raw_path(self) -> None:
        source = DummySource(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("spectra_download.sources.base.download_json", return_value={}), patch(
                "spectra_download.sources.base.download_bytes",
                return_value=b"SIMPLE  =                    TEND",
            ):
                spectra = source.download("HD1", {"raw_save_path": tmp})

            self.assertEqual(len(spectra), 1)
            s = spectra[0]
            self.assertIn("raw_path", s.metadata)
            raw_path = Path(s.metadata["raw_path"])
            self.assertTrue(raw_path.exists())
            self.assertEqual(raw_path.read_bytes(), b"SIMPLE  =                    TEND")

    def test_save_path_writes_zarr_store_if_available(self) -> None:
        try:
            import zarr  # type: ignore
        except Exception:
            self.skipTest("zarr not installed")

        source = DummySource(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, "out.zarr")
            with patch("spectra_download.sources.base.download_json", return_value={}), patch(
                "spectra_download.sources.base.download_bytes",
                return_value=b"FITSBYTES",
            ), patch(
                "spectra_download.sources.base.extract_data_keys_from_fits_bytes",
                return_value={},
            ), patch(
                "spectra_download.sources.base.extract_1d_wavelength_intensity_from_fits_bytes",
                return_value=([1.0, 2.0], [10.0, 20.0], {"extraction": "stub"}),
            ):
                spectra = source.download("HD1", {"save_path": store_path})

            self.assertEqual(len(spectra), 1)
            s = spectra[0]
            self.assertEqual(s.metadata["zarr_store"], store_path)
            self.assertEqual(s.metadata["zarr_key"], "spectra/HD1")

            root = zarr.open_group(store_path, mode="r")
            self.assertIn("spectra/HD1/intensity", root)
            self.assertIn("spectra/HD1/wavelength", root)
            self.assertNotIn("spectra/HD1/fits_bytes", root)
            self.assertNotIn("spectra/HD1/n_points", root)

            intensity = root["spectra/HD1/intensity"]
            wavelength = root["spectra/HD1/wavelength"]
            self.assertEqual(intensity.shape[0], 1)
            self.assertEqual(wavelength.shape[0], 1)

    def test_zarr_paths_alias_works_like_save_path(self) -> None:
        try:
            import zarr  # type: ignore
        except Exception:
            self.skipTest("zarr not installed")

        source = DummySource(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, "out.zarr")
            with patch("spectra_download.sources.base.download_json", return_value={}), patch(
                "spectra_download.sources.base.download_bytes",
                return_value=b"FITSBYTES",
            ), patch(
                "spectra_download.sources.base.extract_data_keys_from_fits_bytes",
                return_value={},
            ), patch(
                "spectra_download.sources.base.extract_1d_wavelength_intensity_from_fits_bytes",
                return_value=([1.0, 2.0], [10.0, 20.0], {"extraction": "stub"}),
            ):
                spectra = source.download("HD1", {"zarr_paths": store_path})

            self.assertEqual(len(spectra), 1)
            s = spectra[0]
            self.assertEqual(s.metadata["zarr_store"], store_path)
            self.assertEqual(s.metadata["zarr_key"], "spectra/HD1")

            root = zarr.open_group(store_path, mode="r")
            self.assertIn("spectra/HD1/intensity", root)
            self.assertNotIn("spectra/HD1/fits_bytes", root)
            self.assertNotIn("spectra/HD1/n_points", root)

            intensity = root["spectra/HD1/intensity"]
            self.assertEqual(intensity.shape[0], 1)

    def test_multiple_spectra_write_multiple_intensity_arrays_under_same_group(self) -> None:
        try:
            import zarr  # type: ignore
        except Exception:
            self.skipTest("zarr not installed")

        class TwoSpecSource(SpectraSource):
            name = "two"

            def build_request_url(self, identifier: str, extra_params: dict) -> str:  # type: ignore[override]
                return f"https://example.test/query?id={identifier}"

            def parse_response(self, payload: dict, identifier: str) -> list[SpectrumRecord]:  # type: ignore[override]
                return [
                    SpectrumRecord(
                        spectrum_id="s1",
                        source=self.name,
                        metadata={"access_url": "https://example.test/s1.fits", "identifier": identifier, "product": "spectrum"},
                    ),
                    SpectrumRecord(
                        spectrum_id="s2",
                        source=self.name,
                        metadata={"access_url": "https://example.test/s2.fits", "identifier": identifier, "product": "spectrum"},
                    ),
                ]

        source = TwoSpecSource(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, "out.zarr")
            # Return different intensity arrays per call.
            extracted = [
                ([1.0, 2.0], [10.0, 20.0], {"extraction": "stub"}),
                ([1.0, 2.0, 3.0], [30.0, 40.0, 50.0], {"extraction": "stub"}),
            ]
            with patch("spectra_download.sources.base.download_json", return_value={}), patch(
                "spectra_download.sources.base.download_bytes",
                return_value=b"FITSBYTES",
            ), patch(
                "spectra_download.sources.base.extract_data_keys_from_fits_bytes",
                return_value={},
            ), patch(
                "spectra_download.sources.base.extract_1d_wavelength_intensity_from_fits_bytes",
                side_effect=extracted,
            ):
                _ = source.download("HD1", {"save_path": store_path})

            root = zarr.open_group(store_path, mode="r")
            self.assertIn("spectra/HD1/intensity", root)
            self.assertIn("spectra/HD1/wavelength", root)

            intensity = root["spectra/HD1/intensity"]
            wavelength = root["spectra/HD1/wavelength"]
            self.assertEqual(intensity.shape, (2, 3))
            self.assertEqual(wavelength.shape, (2, 3))
            self.assertNotIn("spectra/HD1/n_points", root)

            import numpy as np  # type: ignore

            # First row was shorter and should be padded with NaN.
            self.assertTrue(np.isnan(float(intensity[0, 2])))
            self.assertTrue(np.isnan(float(wavelength[0, 2])))

    def test_ccf_products_use_ccf_to_zarr_components_when_enabled(self) -> None:
        try:
            import zarr  # type: ignore
        except Exception:
            self.skipTest("zarr not installed")

        class DummyCcfSource(SpectraSource):
            name = "dummy_ccf"

            def build_request_url(self, identifier: str, extra_params: dict) -> str:  # type: ignore[override]
                return f"https://example.test/query?id={identifier}"

            def parse_response(self, payload: dict, identifier: str) -> list[SpectrumRecord]:  # type: ignore[override]
                return [
                    SpectrumRecord(
                        spectrum_id="ccf1",
                        source=self.name,
                        metadata={
                            "access_url": "https://example.test/ccf1.fits",
                            "identifier": identifier,
                            "product": "ccf",
                        },
                    )
                ]

        source = DummyCcfSource(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            store_path = os.path.join(tmp, "out.zarr")
            with patch("spectra_download.sources.base.download_json", return_value={}), patch(
                "spectra_download.sources.base.download_bytes",
                return_value=b"FITSBYTES",
            ), patch(
                "spectra_download.sources.base.extract_data_keys_from_fits_bytes",
                return_value={},
            ), patch(
                "spectra_download.sources.base.extract_ccf_from_fits_bytes",
                return_value=([0.0, 1.0], [10.0, 9.0], {"extraction": "stub"}),
            ):
                spectra = source.download("HD1", {"save_path": store_path, "process_ccf": True})

            self.assertEqual(len(spectra), 1)
            root = zarr.open_group(store_path, mode="r")
            # For CCF, Zarr group is the object identifier (no _ccf suffix) and only the CCF is stored.
            self.assertIn("spectra/HD1/ccf", root)
            ccf = root["spectra/HD1/ccf"]
            self.assertEqual(ccf.shape[0], 1)
            self.assertNotIn("spectra/HD1/n_ccf_points", root)

    def test_not_found_path_appends_identifier_when_no_records(self) -> None:
        class EmptySource(SpectraSource):
            name = "empty"

            def build_request_url(self, identifier: str, extra_params: dict) -> str:  # type: ignore[override]
                return f"https://example.test/query?id={identifier}"

            def parse_response(self, payload: dict, identifier: str) -> list[SpectrumRecord]:  # type: ignore[override]
                return []

        source = EmptySource(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            not_found = os.path.join(tmp, "not_found.jsonl")
            with patch("spectra_download.sources.base.download_json", return_value={}):
                spectra = source.download("HD_NOPE", {"not_found_path": not_found})
            self.assertEqual(spectra, [])

            content = Path(not_found).read_text(encoding="utf-8")
            self.assertIn('"identifier": "HD_NOPE"', content)
