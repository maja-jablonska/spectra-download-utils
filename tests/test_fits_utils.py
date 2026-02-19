import io
import unittest

import numpy as np

try:
    from astropy.io import fits  # type: ignore
except Exception:  # pragma: no cover
    fits = None  # type: ignore

from spectra_download import fits_utils


def _fits_bytes(hdus: list["fits.hdu.base.ExtensionHDU"]) -> bytes:  # type: ignore[name-defined]
    mem = io.BytesIO()
    fits.HDUList(hdus).writeto(mem)  # type: ignore[union-attr]
    return mem.getvalue()


class TestFitsUtils(unittest.TestCase):
    def setUp(self) -> None:
        if fits is None:
            self.skipTest("astropy not installed")

    def test_extract_data_keys_from_headers_with_sexagesimal_coordinates(self) -> None:
        primary = fits.PrimaryHDU()  # type: ignore[union-attr]
        primary.header["EXPTIME"] = ""
        primary.header["TEXPTIME"] = 1200.5
        primary.header["ALPHA"] = "20:00:17"
        primary.header["DELTA"] = "+03:11:00"
        primary.header["DATE"] = "2026-01-01T00:00:00"
        primary.header["RADESYS"] = "FK5"
        primary.header["EQUINOX"] = 2000.0
        primary.header["MJD"] = 61000.5
        primary.header["AIRMASS"] = 1.234
        primary.header["TARGET"] = "HD190007"
        primary.header["BARYCORR"] = 18.6
        primary.header["SNRMED"] = 95.0
        primary.header["NORMED"] = "yes"
        primary.header["OBSFRAME"] = "BARYCENT"

        bintable = fits.BinTableHDU.from_columns([])  # type: ignore[union-attr]
        keys = fits_utils.extract_data_keys_from_fits_bytes(_fits_bytes([primary, bintable]))

        self.assertAlmostEqual(keys["exptime"], 1200.5)
        self.assertAlmostEqual(keys["ra"], 300.0708333333, places=6)
        self.assertAlmostEqual(keys["dec"], 3.1833333333, places=6)
        self.assertEqual(keys["date"], "2026-01-01T00:00:00")
        self.assertEqual(keys["reference_frame"], "FK5")
        self.assertEqual(keys["reference_frame_epoch"], 2000.0)
        self.assertAlmostEqual(keys["mjd"], 61000.5)
        self.assertAlmostEqual(keys["airmass"], 1.234)
        self.assertEqual(keys["object"], "HD190007")
        self.assertAlmostEqual(keys["berv"], 18.6)
        self.assertAlmostEqual(keys["snr"], 95.0)
        self.assertTrue(keys["normalized"])
        self.assertEqual(keys["frame"], "BARYCENT")

    def test_extract_data_keys_uses_primary_header_before_extensions(self) -> None:
        primary = fits.PrimaryHDU()  # type: ignore[union-attr]
        primary.header["RA"] = 120.0
        ext = fits.ImageHDU(data=np.array([1.0, 2.0], dtype=np.float64))  # type: ignore[union-attr]
        ext.header["RA"] = 130.0

        keys = fits_utils.extract_data_keys_from_fits_bytes(_fits_bytes([primary, ext]))
        self.assertAlmostEqual(keys["ra"], 120.0)

    def test_extract_data_keys_boolean_false_values(self) -> None:
        primary = fits.PrimaryHDU()  # type: ignore[union-attr]
        primary.header["NORMED"] = 0

        keys = fits_utils.extract_data_keys_from_fits_bytes(_fits_bytes([primary]))
        self.assertIn("normalized", keys)
        self.assertFalse(keys["normalized"])

    def test_extract_1d_wavelength_intensity_from_bintable_exact_columns(self) -> None:
        wave = np.array([5000.0, 5000.1, 5000.2], dtype=np.float64)
        flux = np.array([10.0, 11.0, 12.0], dtype=np.float64)
        cols = [
            fits.Column(name="WAVE", format="3D", array=[wave]),  # type: ignore[union-attr]
            fits.Column(name="FLUX", format="3D", array=[flux]),  # type: ignore[union-attr]
        ]
        hdus = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]  # type: ignore[union-attr]
        wl, inten, info = fits_utils.extract_1d_wavelength_intensity_from_fits_bytes(_fits_bytes(hdus))

        assert wl is not None and inten is not None
        self.assertEqual(info["extraction"], "bintable")
        self.assertEqual(info["wavelength_column"], "wave")
        self.assertEqual(info["intensity_column"], "flux")
        np.testing.assert_allclose(np.asarray(wl).ravel(), wave)
        np.testing.assert_allclose(np.asarray(inten).ravel(), flux)

    def test_extract_1d_wavelength_intensity_from_bintable_fuzzy_columns(self) -> None:
        wave = np.array([7000.0, 7000.2], dtype=np.float64)
        flux = np.array([2.0, 3.0], dtype=np.float64)
        cols = [
            fits.Column(name="obs_wavelength_nm", format="2D", array=[wave]),  # type: ignore[union-attr]
            fits.Column(name="flux_reduced", format="2D", array=[flux]),  # type: ignore[union-attr]
        ]
        hdus = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]  # type: ignore[union-attr]
        wl, inten, info = fits_utils.extract_1d_wavelength_intensity_from_fits_bytes(_fits_bytes(hdus))

        assert wl is not None and inten is not None
        self.assertEqual(info["extraction"], "bintable")
        self.assertEqual(info["wavelength_column"], "obs_wavelength_nm")
        self.assertEqual(info["intensity_column"], "flux_reduced")
        np.testing.assert_allclose(np.asarray(wl).ravel(), wave)
        np.testing.assert_allclose(np.asarray(inten).ravel(), flux)

    def test_extract_1d_wavelength_intensity_from_image_with_wcs(self) -> None:
        hdu = fits.PrimaryHDU(data=np.array([10.0, 11.0, 12.0], dtype=np.float64))  # type: ignore[union-attr]
        hdu.header["CRVAL1"] = 4000.0
        hdu.header["CDELT1"] = 0.05
        hdu.header["CRPIX1"] = 1.0

        wl, inten, info = fits_utils.extract_1d_wavelength_intensity_from_fits_bytes(_fits_bytes([hdu]))
        assert wl is not None and inten is not None
        self.assertEqual(info["extraction"], "image_wcs_linear")
        np.testing.assert_allclose(wl, np.array([4000.0, 4000.05, 4000.10]))
        np.testing.assert_allclose(inten, np.array([10.0, 11.0, 12.0]))

    def test_extract_1d_wavelength_intensity_from_image_without_wcs(self) -> None:
        hdu = fits.PrimaryHDU(data=np.array([3.0, 2.0], dtype=np.float64))  # type: ignore[union-attr]
        wl, inten, info = fits_utils.extract_1d_wavelength_intensity_from_fits_bytes(_fits_bytes([hdu]))

        self.assertIsNone(wl)
        assert inten is not None
        self.assertEqual(info["extraction"], "image_no_wcs")
        np.testing.assert_allclose(inten, np.array([3.0, 2.0]))

    def test_extract_ccf_from_bintable_exact_columns(self) -> None:
        vel = np.array([-5.0, 0.0, 5.0], dtype=np.float64)
        ccf = np.array([0.9, 1.0, 0.8], dtype=np.float64)
        cols = [
            fits.Column(name="RV", format="3D", array=[vel]),  # type: ignore[union-attr]
            fits.Column(name="CCF", format="3D", array=[ccf]),  # type: ignore[union-attr]
        ]
        hdus = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]  # type: ignore[union-attr]
        out_vel, out_ccf, info = fits_utils.extract_ccf_from_fits_bytes(_fits_bytes(hdus))

        assert out_vel is not None and out_ccf is not None
        self.assertEqual(info["extraction"], "bintable")
        self.assertEqual(info["velocity_column"], "rv")
        self.assertEqual(info["ccf_column"], "ccf")
        np.testing.assert_allclose(np.asarray(out_vel).ravel(), vel)
        np.testing.assert_allclose(np.asarray(out_ccf).ravel(), ccf)

    def test_extract_ccf_from_bintable_fuzzy_columns(self) -> None:
        vel = np.array([-1.0, 1.0], dtype=np.float64)
        ccf = np.array([0.2, 0.3], dtype=np.float64)
        cols = [
            fits.Column(name="velocity_kms", format="2D", array=[vel]),  # type: ignore[union-attr]
            fits.Column(name="xcorr_value", format="2D", array=[ccf]),  # type: ignore[union-attr]
        ]
        hdus = [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]  # type: ignore[union-attr]
        out_vel, out_ccf, info = fits_utils.extract_ccf_from_fits_bytes(_fits_bytes(hdus))

        assert out_vel is not None and out_ccf is not None
        self.assertEqual(info["extraction"], "bintable")
        self.assertEqual(info["velocity_column"], "velocity_kms")
        self.assertEqual(info["ccf_column"], "xcorr_value")
        np.testing.assert_allclose(np.asarray(out_vel).ravel(), vel)
        np.testing.assert_allclose(np.asarray(out_ccf).ravel(), ccf)

    def test_extract_ccf_from_image_with_wcs(self) -> None:
        hdu = fits.PrimaryHDU(data=np.array([0.8, 1.0, 0.7], dtype=np.float64))  # type: ignore[union-attr]
        hdu.header["CRVAL1"] = -20.0
        hdu.header["CDELT1"] = 0.5
        hdu.header["CRPIX1"] = 1.0

        vel, ccf, info = fits_utils.extract_ccf_from_fits_bytes(_fits_bytes([hdu]))
        assert vel is not None and ccf is not None
        self.assertEqual(info["extraction"], "image_wcs_linear")
        np.testing.assert_allclose(vel, np.array([-20.0, -19.5, -19.0]))
        np.testing.assert_allclose(ccf, np.array([0.8, 1.0, 0.7]))

    def test_extract_ccf_from_image_without_wcs(self) -> None:
        hdu = fits.PrimaryHDU(data=np.array([0.1, 0.2], dtype=np.float64))  # type: ignore[union-attr]
        vel, ccf, info = fits_utils.extract_ccf_from_fits_bytes(_fits_bytes([hdu]))

        self.assertIsNone(vel)
        assert ccf is not None
        self.assertEqual(info["extraction"], "image_no_wcs")
        np.testing.assert_allclose(ccf, np.array([0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
