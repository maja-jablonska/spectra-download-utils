import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

from spectra_download.models import SpectrumRecord
from spectra_download.sources.eso import EsoFerosSource, EsoHarpsSource, EsoUvesSource


class TestEsoTapSourcesBuildRequestUrl(unittest.TestCase):
    def test_harps_instrument_condition(self) -> None:
        src = EsoHarpsSource()
        url = src.build_request_url("ANY", {})
        params = parse_qs(urlparse(url).query)
        query = unquote(params["QUERY"][0])
        self.assertIn("instrument_name='HARPS'", query)

    def test_uves_instrument_condition(self) -> None:
        src = EsoUvesSource()
        url = src.build_request_url("ANY", {})
        params = parse_qs(urlparse(url).query)
        query = unquote(params["QUERY"][0])
        self.assertIn("instrument_name='UVES'", query)

    def test_feros_instrument_condition(self) -> None:
        src = EsoFerosSource()
        url = src.build_request_url("ANY", {})
        params = parse_qs(urlparse(url).query)
        query = unquote(params["QUERY"][0])
        self.assertIn("instrument_name='FEROS'", query)


class TestEsoHarpsExtraction(unittest.TestCase):
    def test_extract_spectrum_arrays_from_fits_payload_reads_wave_flux_err(self) -> None:
        try:
            import numpy as np  # type: ignore
            from astropy.io import fits  # type: ignore
        except Exception:
            self.skipTest("astropy/numpy not installed")

        # Build a minimal HARPS-like FITS: primary HDU with header keys and a bintable
        # extension with WAVE/FLUX/ERR columns.
        wave = np.asarray([1.0, 2.0, 3.0], dtype="f8")
        flux = np.asarray([10.0, 20.0, 30.0], dtype="f4")
        err = np.asarray([0.1, 0.2, 0.3], dtype="f4")

        primary = fits.PrimaryHDU()
        primary.header["TEXPTIME"] = 1.0
        primary.header["RA"] = 10.0
        primary.header["DEC"] = -5.0
        primary.header["RADECSYS"] = "FK5"
        primary.header["EQUINOX"] = 2000.0
        primary.header["DATE-OBS"] = "2000-01-01T00:00:00"
        primary.header["MJD-OBS"] = 51544.0
        primary.header["OBJECT"] = "TEST"
        primary.header["SNR"] = 100.0

        cols = [
            fits.Column(name="WAVE", format="3D", array=[wave]),
            fits.Column(name="FLUX", format="3E", array=[flux]),
            fits.Column(name="ERR", format="3E", array=[err]),
        ]
        table = fits.BinTableHDU.from_columns(cols)

        hdul = fits.HDUList([primary, table])
        buf = BytesIO()
        hdul.writeto(buf)
        payload = buf.getvalue()

        src = EsoHarpsSource()
        arrays, info = src.extract_spectrum_arrays_from_fits_payload(
            fits_bytes=payload,
            spectrum=SpectrumRecord(spectrum_id="1", source=src.name, metadata={"identifier": "TEST"}),
        )

        self.assertIn("wavelengths", arrays)
        self.assertIn("intensity", arrays)
        self.assertIn("error", arrays)
        self.assertEqual(arrays["wavelengths"].shape[-1], 3)
        self.assertEqual(arrays["intensity"].shape[-1], 3)
        self.assertEqual(arrays["error"].shape[-1], 3)
        self.assertEqual(info["object"], "TEST")


class TestEsoFerosExtraction(unittest.TestCase):
    def test_extract_spectrum_arrays_from_fits_payload_reads_wave_flux_err(self) -> None:
        try:
            import numpy as np  # type: ignore
            from astropy.io import fits  # type: ignore
        except Exception:
            self.skipTest("astropy/numpy not installed")

        wave = np.asarray([1.0, 2.0, 3.0], dtype="f8")
        flux = np.asarray([10.0, 20.0, 30.0], dtype="f4")
        err = np.asarray([0.1, 0.2, 0.3], dtype="f4")

        primary = fits.PrimaryHDU()
        primary.header["OBJECT"] = "TEST"
        cols = [
            fits.Column(name="WAVE", format="3D", array=[wave]),
            fits.Column(name="FLUX", format="3E", array=[flux]),
            fits.Column(name="ERR", format="3E", array=[err]),
        ]
        table = fits.BinTableHDU.from_columns(cols)
        hdul = fits.HDUList([primary, table])
        buf = BytesIO()
        hdul.writeto(buf)
        payload = buf.getvalue()

        src = EsoFerosSource()
        arrays, info = src.extract_spectrum_arrays_from_fits_payload(
            fits_bytes=payload,
            spectrum=SpectrumRecord(spectrum_id="1", source=src.name, metadata={"identifier": "TEST"}),
        )
        self.assertIn("wavelengths", arrays)
        self.assertIn("intensity", arrays)
        self.assertIn("error", arrays)
        self.assertEqual(info["object"], "TEST")


class TestEsoDataLinkAndAuthFetch(unittest.TestCase):
    def test_fetch_fits_payload_resolves_datalink_this_url(self) -> None:
        # Minimal DataLink VOTable with fields access_url/semantics/content_type and a #this row.
        votable = b"""<?xml version='1.0'?>
<VOTABLE xmlns='http://www.ivoa.net/xml/VOTable/v1.3' version='1.3'>
  <RESOURCE type='results'>
    <TABLE>
      <FIELD name='ignored' datatype='char' arraysize='*'/>
      <FIELD name='access_url' datatype='char' arraysize='*'/>
      <FIELD name='ignored2' datatype='char' arraysize='*'/>
      <FIELD name='ignored3' datatype='char' arraysize='*'/>
      <FIELD name='semantics' datatype='char' arraysize='*'/>
      <FIELD name='ignored4' datatype='char' arraysize='*'/>
      <FIELD name='content_type' datatype='char' arraysize='*'/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>x</TD>
            <TD>https://example.test/file.fits</TD>
            <TD>x</TD>
            <TD>x</TD>
            <TD>#this</TD>
            <TD>x</TD>
            <TD>application/fits</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
"""
        fits_magic = b"SIMPLE  =                    TEND"

        src = EsoUvesSource(timeout=1, max_retries=1)
        spectrum = SpectrumRecord(spectrum_id="1", source=src.name, metadata={"identifier": "X"})

        with patch("spectra_download.sources.eso.download_bytes", side_effect=[votable, fits_magic]):
            out_bytes, updates = src.fetch_fits_payload(access_url="http://example.test/dl", spectrum=spectrum)

        self.assertEqual(out_bytes, fits_magic)
        self.assertEqual(updates["access_url"], "https://example.test/file.fits")
        self.assertEqual(updates["datalink_url"], "http://example.test/dl")

    def test_download_eso_bytes_retries_with_bearer_token_on_401(self) -> None:
        from spectra_download.http_client import DownloadError

        src = EsoUvesSource(timeout=1, max_retries=1)
        src._get_eso_id_token = lambda: "TOKEN"  # type: ignore[method-assign]

        calls = []

        def fake_download_bytes(url: str, *, timeout: int, max_retries: int, headers=None):  # type: ignore[no-untyped-def]
            calls.append({"url": url, "headers": headers})
            if headers is None:
                raise DownloadError("401", url=url, status_code=401)
            return b"OK"

        with patch("spectra_download.sources.eso.download_bytes", side_effect=fake_download_bytes):
            out = src._download_eso_bytes("https://example.test/private.fits")  # type: ignore[attr-defined]

        self.assertEqual(out, b"OK")
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[0]["headers"])
        self.assertEqual(calls[1]["headers"], {"Authorization": "Bearer TOKEN"})


if __name__ == "__main__":
    unittest.main()

