import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

from spectra_download.sources.narval import NarvalSource


class TestNarvalSourceBuildRequestUrl(unittest.TestCase):
    def test_build_request_url_includes_inst_and_object(self) -> None:
        src = NarvalSource()
        url = src.build_request_url("HD  18474", {"limit": 3})
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["REQUEST"], ["doQuery"])
        self.assertEqual(params["LANG"], ["ADQL"])
        self.assertEqual(params["FORMAT"], ["json"])
        query = unquote(params["QUERY"][0])
        self.assertIn('FROM "B/polarbase/polarbase"', query)
        self.assertIn("Inst='narval'", query)
        self.assertIn("Object='HD  18474'", query)
        self.assertIn("SELECT TOP 3", query)


class TestNarvalSourceParseResponse(unittest.TestCase):
    def test_parse_response_sets_access_url(self) -> None:
        src = NarvalSource()
        payload = {
            "metadata": [{"name": "ID"}, {"name": "Object"}, {"name": "Inst"}, {"name": "url"}],
            "data": [[123, "HD  18474", "narval", "http://example.test/file.fts"]],
        }
        spectra = src.parse_response(payload, "HD  18474")
        self.assertEqual(len(spectra), 1)
        s = spectra[0]
        self.assertEqual(s.spectrum_id, "123")
        self.assertEqual(s.metadata["Inst"], "narval")
        self.assertEqual(s.metadata["access_url"], "http://example.test/file.fts")


class TestNarvalSourceFallback(unittest.TestCase):
    def test_download_falls_back_to_like_match(self) -> None:
        src = NarvalSource(timeout=1, max_retries=1)

        empty = {"metadata": [{"name": "ID"}], "data": []}
        hit = {
            "metadata": [{"name": "ID"}, {"name": "Object"}, {"name": "Inst"}, {"name": "url"}],
            "data": [[1, "SOME OBJECT", "narval", "http://example.test/a.fts"]],
        }

        def fake_download_json(url: str, *, timeout: int = 30, max_retries: int = 3, headers=None):  # type: ignore[no-untyped-def]
            query = unquote(parse_qs(urlparse(url).query)["QUERY"][0])
            if "Object LIKE '%Siri%'" in query:
                return hit
            return empty

        with patch("spectra_download.sources.base.download_json", side_effect=fake_download_json):
            spectra = src.download("Siri", {"limit": 1})

        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0].metadata["identifier"], "Siri")
        self.assertTrue(str(spectra[0].metadata.get("query_identifier", "")).startswith("LIKE:"))

    def test_download_falls_back_to_coordinate_cone_search(self) -> None:
        src = NarvalSource(timeout=1, max_retries=1)

        empty = {"metadata": [{"name": "ID"}], "data": []}
        hit = {
            "metadata": [{"name": "ID"}, {"name": "Object"}, {"name": "Inst"}, {"name": "url"}],
            "data": [[1, "ANY", "narval", "http://example.test/a.fts"]],
        }

        def fake_download_json(url: str, *, timeout: int = 30, max_retries: int = 3, headers=None):  # type: ignore[no-untyped-def]
            query = unquote(parse_qs(urlparse(url).query)["QUERY"][0])
            if "CONTAINS(" in query and "POINT('ICRS', RA_ICRS, DE_ICRS)" in query:
                return hit
            return empty

        with patch("spectra_download.sources.base.download_json", side_effect=fake_download_json), patch(
            "spectra_download.sources.tap._resolve_name_to_icrs_degrees", return_value=(10.0, -5.0)
        ):
            spectra = src.download("Betelgeuse", {"limit": 1, "use_like": False, "search_radius_arcsec": 7.0})

        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0].metadata["identifier"], "Betelgeuse")
        self.assertTrue(str(spectra[0].metadata.get("query_identifier", "")).startswith("COORDS:"))


if __name__ == "__main__":
    unittest.main()

