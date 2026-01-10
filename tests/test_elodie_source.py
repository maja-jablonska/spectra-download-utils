import unittest
import tempfile
from pathlib import Path

from spectra_download.sources.elodie import (
    ElodieSource,
    _extract_ccf_fits_urls,
    _extract_search_ccf_page_urls,
    _extract_spectrum_fits_urls,
    _sanitize_url,
    _spectrum_id_from_url,
)


class TestElodieParsing(unittest.TestCase):
    def test_extract_spectrum_fits_urls(self) -> None:
        html = """
        <html><body>
          <a href="fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:19951220/0009">get_spec</a>
          <a href="fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:19951220/0009">get_spec</a>
          <a href="fE.cgi?a=mime:application/fits&o=ccf:19951220/0009">get_ccf</a>
          <a href="other.cgi">something else</a>
        </body></html>
        """
        urls = _extract_spectrum_fits_urls(html, base_url="http://atlas.obs-hp.fr/elodie/")
        self.assertEqual(
            urls,
            ["http://atlas.obs-hp.fr/elodie/fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:19951220/0009"],
        )

    def test_extract_search_ccf_pages(self) -> None:
        html = """
        <html><body>
          <a href="fE.cgi?x=1">Search_CCF</a>
          <a href="fE.cgi?x=1">search_ccf</a>
          <a href="fE.cgi?x=2">nope</a>
        </body></html>
        """
        pages = _extract_search_ccf_page_urls(html, base_url="http://atlas.obs-hp.fr/elodie/")
        self.assertEqual(pages, ["http://atlas.obs-hp.fr/elodie/fE.cgi?x=1"])

    def test_extract_ccf_fits_urls(self) -> None:
        html = """
        <html><body>
          <a href="fE.cgi?a=mime:application/fits&o=ccf:19951220/0009">get_ccf</a>
          <a href="get_ccf.html">get_ccf</a>
        </body></html>
        """
        urls = _extract_ccf_fits_urls(html, base_url="http://atlas.obs-hp.fr/elodie/")
        self.assertEqual(
            urls,
            ["http://atlas.obs-hp.fr/elodie/fE.cgi?a=mime:application/fits&o=ccf:19951220/0009"],
        )

    def test_spectrum_id_from_url_prefers_o_param(self) -> None:
        url = "http://atlas.obs-hp.fr/elodie/fE.cgi?a=mime:application/fits&o=elodie:19951220/0009"
        self.assertEqual(_spectrum_id_from_url(url, fallback="fallback"), "elodie:19951220/0009")


class TestElodieSourceOfflineDownload(unittest.TestCase):
    def test_download_stubs_fetch_and_returns_spectrum_and_ccf(self) -> None:
        object_url = "http://atlas.obs-hp.fr/elodie/fE.cgi?ob=objname,dataset,imanum&c=o&o=HD224221"
        search_ccf_url = "http://atlas.obs-hp.fr/elodie/fE.cgi?x=search_ccf"
        spectrum_url = "http://atlas.obs-hp.fr/elodie/fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:19951220/0009"
        ccf_url = "http://atlas.obs-hp.fr/elodie/fE.cgi?a=mime:application/fits&o=ccf:19951220/0009"

        html_object = f"""
        <html><body>
          <a href="{spectrum_url}">get_spec</a>
          <a href="{search_ccf_url}">search_ccf</a>
        </body></html>
        """
        html_search_ccf = f"""
        <html><body>
          <a href="{ccf_url}">get_ccf</a>
        </body></html>
        """

        class StubElodieSource(ElodieSource):
            def __init__(self) -> None:
                super().__init__(timeout=1, max_retries=1)
                self._responses = {
                    object_url: html_object,
                    search_ccf_url: html_search_ccf,
                }

            def _fetch_html(self, url: str) -> str:  # type: ignore[override]
                return self._responses[url]

        source = StubElodieSource()
        spectra = source.download("HD224221")
        self.assertEqual(len(spectra), 2)
        self.assertEqual({s.metadata["product"] for s in spectra}, {"spectrum", "ccf"})
        self.assertEqual({s.metadata["access_url"] for s in spectra}, {spectrum_url, ccf_url})

    def test_download_respects_overridden_base_url_for_spectrum_links(self) -> None:
        base_url = "http://example.test/elodie/"
        object_url = f"{base_url}fE.cgi?ob=objname,dataset,imanum&c=o&o=HD224221"

        # Use relative hrefs so url-joining behavior is exercised.
        html_object = """
        <html><body>
          <a href="fE.cgi?n=e500&c=i&z=s1d&a=mime:application/fits&o=elodie:19951220/0009">get_spec</a>
        </body></html>
        """

        class StubElodieSource(ElodieSource):
            def __init__(self) -> None:
                super().__init__(timeout=1, max_retries=1)
                self._responses = {object_url: html_object}

            def _fetch_html(self, url: str) -> str:  # type: ignore[override]
                return self._responses[url]

        source = StubElodieSource()
        spectra = source.download("HD224221", {"base_url": base_url, "include_ccf": False})
        self.assertEqual(len(spectra), 1)
        self.assertTrue(spectra[0].metadata["access_url"].startswith(base_url))

    def test_sanitize_url_encodes_spaces(self) -> None:
        url = "/elodie/fE.cgi?fql=[datenuit ='19970828'],[imanum ='0018']"
        sanitized = _sanitize_url(url)
        self.assertNotIn(" ", sanitized)

    def test_error_path_records_any_exception(self) -> None:
        class BoomElodie(ElodieSource):
            def _fetch_html(self, url: str) -> str:  # type: ignore[override]
                raise RuntimeError("boom")

        src = BoomElodie(timeout=1, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp:
            error_path = Path(tmp) / "errors.jsonl"
            with self.assertRaises(RuntimeError):
                src.download("HDNOPE", error_path=error_path)

            text = error_path.read_text(encoding="utf-8")
            self.assertIn('"identifier": "HDNOPE"', text)
            self.assertIn('"error_type": "RuntimeError"', text)


if __name__ == "__main__":
    unittest.main()

