import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

from spectra_download.sources.espadons import EspadonsSource


class TestEspadonsSourceBuildRequestUrl(unittest.TestCase):
    def test_build_request_url_includes_inst_and_object(self) -> None:
        src = EspadonsSource()
        url = src.build_request_url("HD  18474", {"limit": 3})
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["REQUEST"], ["doQuery"])
        self.assertEqual(params["LANG"], ["ADQL"])
        self.assertEqual(params["FORMAT"], ["json"])
        query = unquote(params["QUERY"][0])
        self.assertIn('FROM "B/polarbase/polarbase"', query)
        self.assertIn("Inst='espadons'", query)
        self.assertIn("Object='HD  18474'", query)
        self.assertIn("SELECT TOP 3", query)


class TestEspadonsSourceParseResponse(unittest.TestCase):
    def test_parse_response_sets_access_url(self) -> None:
        src = EspadonsSource()
        payload = {
            "metadata": [{"name": "ID"}, {"name": "Object"}, {"name": "Inst"}, {"name": "url"}],
            "data": [[123, "HD  18474", "espadons", "http://example.test/file.fts"]],
        }
        spectra = src.parse_response(payload, "HD  18474")
        self.assertEqual(len(spectra), 1)
        s = spectra[0]
        self.assertEqual(s.spectrum_id, "123")
        self.assertEqual(s.metadata["Inst"], "espadons")
        self.assertEqual(s.metadata["access_url"], "http://example.test/file.fts")


class TestEspadonsSourceFallback(unittest.TestCase):
    def test_download_falls_back_to_hd_spacing_variant(self) -> None:
        src = EspadonsSource(timeout=1, max_retries=1)

        # First query (HD18474) returns no rows, second query (HD  18474) returns one row.
        empty = {"metadata": [{"name": "ID"}], "data": []}
        hit = {
            "metadata": [{"name": "ID"}, {"name": "Object"}, {"name": "Inst"}, {"name": "url"}],
            "data": [[1, "HD  18474", "espadons", "http://example.test/a.fts"]],
        }
        calls = []

        def fake_download_json(url: str, *, timeout: int = 30, max_retries: int = 3, headers=None):  # type: ignore[no-untyped-def]
            calls.append(url)
            query = unquote(parse_qs(urlparse(url).query)["QUERY"][0])
            if "Object='HD  18474'" in query:
                return hit
            return empty

        with patch("spectra_download.sources.base.download_json", side_effect=fake_download_json):
            spectra = src.download("HD18474", {"limit": 1, "use_like": False})

        self.assertEqual(len(spectra), 1)
        self.assertEqual(spectra[0].metadata["identifier"], "HD18474")
        self.assertEqual(spectra[0].metadata["query_identifier"], "HD  18474")

    def test_download_falls_back_to_coordinate_cone_search(self) -> None:
        src = EspadonsSource(timeout=1, max_retries=1)

        empty = {"metadata": [{"name": "ID"}], "data": []}
        hit = {
            "metadata": [{"name": "ID"}, {"name": "Object"}, {"name": "Inst"}, {"name": "url"}],
            "data": [[1, "ANY", "espadons", "http://example.test/a.fts"]],
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

