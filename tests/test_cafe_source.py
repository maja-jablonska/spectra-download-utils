import io
import zipfile
import unittest
from unittest.mock import patch

from spectra_download.models import SpectrumRecord
from spectra_download.sources.cafe import CafeSource


class TestCafeSourceParseResponse(unittest.TestCase):
    def test_parse_response_extracts_fetchsci_links(self) -> None:
        html = """
        <html><body>
        <a href="/calto/servlet/FetchSci?id=148663&tipe=red&t=web" target="_blank">FITS</a>
        <a href="/calto/servlet/FetchSci?id=148664&tipe=raw&t=web" target="_blank">FITS</a>
        </body></html>
        """
        src = CafeSource()
        spectra = src.parse_response({"html": html, "tipe": "red", "base_url": src.search_url}, "OBJ")
        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0].spectrum_id, "148663")
        self.assertIn("/calto/servlet/FetchSci?id=148663", spectra[0].metadata["access_url"])


class TestCafeFetchFitsPayload(unittest.TestCase):
    def test_fetch_fits_payload_extracts_fits_member_from_zip(self) -> None:
        # Create an in-memory ZIP with a .fits member.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("test.fits", b"SIMPLE  =                    TEND")
        zip_bytes = buf.getvalue()

        src = CafeSource(timeout=1, max_retries=1)
        spectrum = SpectrumRecord(spectrum_id="1", source=src.name, metadata={"identifier": "X"})

        with patch("spectra_download.sources.cafe.download_bytes", return_value=zip_bytes):
            out, md = src.fetch_fits_payload(access_url="https://example.test/FetchSci?id=1&tipe=red&t=web", spectrum=spectrum)

        self.assertTrue(out.startswith(b"SIMPLE"))
        self.assertEqual(md["caha_zip_member"], "test.fits")


if __name__ == "__main__":
    unittest.main()

