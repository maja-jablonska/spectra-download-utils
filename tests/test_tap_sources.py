import unittest
from urllib.parse import parse_qs, unquote, urlparse

from spectra_download.sources.eso import EsoHarpsSource
from spectra_download.sources.tap import TapSpectraSource, _tap_records, _cone_where


class DummyTapSource(TapSpectraSource):
    name = "dummy"
    tap_url = "https://example.test/tap/sync"
    table = "ivoa.obscore"


class TestTapRecords(unittest.TestCase):
    def test_tap_records_from_list(self) -> None:
        payload = [{"obs_id": "OBS1"}, {"obs_id": "OBS2"}]
        self.assertEqual(_tap_records(payload), payload)

    def test_tap_records_from_metadata(self) -> None:
        payload = {
            "metadata": [{"name": "obs_id"}, {"name": "access_url"}],
            "data": [["OBS1", "https://example.test/obs1"], ["OBS2", None]],
        }
        records = _tap_records(payload)
        self.assertEqual(
            records,
            [
                {"obs_id": "OBS1", "access_url": "https://example.test/obs1"},
                {"obs_id": "OBS2", "access_url": None},
            ],
        )


class TestTapSourceBuildRequestUrl(unittest.TestCase):
    def test_build_request_url_defaults(self) -> None:
        source = DummyTapSource()
        url = source.build_request_url("OBS123", {})
        params = parse_qs(urlparse(url).query)
        self.assertEqual(params["REQUEST"], ["doQuery"])
        self.assertEqual(params["LANG"], ["ADQL"])
        self.assertEqual(params["FORMAT"], ["json"])
        query = unquote(params["QUERY"][0])
        self.assertIn("SELECT * FROM ivoa.obscore WHERE obs_id='OBS123'", query)

    def test_build_request_url_with_extras(self) -> None:
        source = DummyTapSource()
        url = source.build_request_url(
            "OBS123",
            {
                "select_fields": ["obs_id", "access_url"],
                "limit": 5,
                "where": "calib_level=2",
                "conditions": ["data_product_type='spectrum'"],
            },
        )
        params = parse_qs(urlparse(url).query)
        query = unquote(params["QUERY"][0])
        self.assertIn("SELECT TOP 5 obs_id, access_url", query)
        self.assertIn("WHERE obs_id='OBS123' AND calib_level=2 AND data_product_type='spectrum'", query)

    def test_build_request_url_without_identifier_field_uses_where_only(self) -> None:
        source = DummyTapSource()
        url = source.build_request_url(
            "IGNORED",
            {
                "identifier_field": None,
                "where": "1=1",
            },
        )
        params = parse_qs(urlparse(url).query)
        query = unquote(params["QUERY"][0])
        self.assertIn("WHERE 1=1", query)
        self.assertNotIn("obs_id='IGNORED'", query)

    def test_eso_harps_instrument_condition(self) -> None:
        source = EsoHarpsSource()
        url = source.build_request_url("OBS123", {})
        params = parse_qs(urlparse(url).query)
        query = unquote(params["QUERY"][0])
        self.assertIn("instrument_name='HARPS'", query)

    def test_cone_where_format(self) -> None:
        frag = _cone_where(ra_deg=10.0, dec_deg=-5.0, radius_arcsec=5.0)
        self.assertIn("CONTAINS(", frag)
        self.assertIn("CIRCLE('ICRS', 10.0, -5.0", frag)


class TestTapSourceParseResponse(unittest.TestCase):
    def test_parse_response_maps_rows(self) -> None:
        source = DummyTapSource()
        payload = {
            "metadata": [{"name": "obs_id"}, {"name": "peaks"}, {"name": "obs_publisher_did"}],
            "data": [["OBS1", [{"mz": 1, "intensity": 2}], "PUB1"]],
        }
        spectra = source.parse_response(payload, "OBS1")
        self.assertEqual(len(spectra), 1)
        spectrum = spectra[0]
        self.assertEqual(spectrum.spectrum_id, "OBS1")
        self.assertEqual(spectrum.metadata["peaks"], [{"mz": 1, "intensity": 2}])
        self.assertEqual(spectrum.metadata["obs_publisher_did"], "PUB1")
        self.assertEqual(spectrum.metadata["obs_id"], "OBS1")


if __name__ == "__main__":
    unittest.main()
