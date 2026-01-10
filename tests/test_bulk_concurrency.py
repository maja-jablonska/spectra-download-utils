import threading
import time
import unittest

from spectra_download.bulk import bulk_download
from spectra_download.models import SpectraRequest, Spectrum
from spectra_download.sources.base import SpectraSource


class BlockingSource(SpectraSource):
    name = "blocking"

    def __init__(self, barrier: threading.Barrier) -> None:
        super().__init__(timeout=1, max_retries=1)
        self._barrier = barrier

    def build_request_url(self, identifier: str, extra_params: dict) -> str:  # type: ignore[override]
        return "https://example.test"

    def parse_response(self, payload: dict, identifier: str) -> list[Spectrum]:  # type: ignore[override]
        return []

    def download(self, identifier: str, extra_params: dict | None = None) -> list[Spectrum]:  # type: ignore[override]
        # Both requests must enter here concurrently to pass quickly.
        self._barrier.wait(timeout=2.0)
        time.sleep(0.05)
        return [
            Spectrum(
                spectrum_id=identifier,
                source=self.name,
                intensity=[],
                wavelength=[],
                normalized=False,
                metadata={"identifier": identifier},
            )
        ]


class TestBulkDownloadConcurrency(unittest.TestCase):
    def test_bulk_download_runs_requests_concurrently(self) -> None:
        barrier = threading.Barrier(2)
        source = BlockingSource(barrier)
        reqs = [
            SpectraRequest(source="blocking", identifier="A"),
            SpectraRequest(source="blocking", identifier="B"),
        ]

        t0 = time.time()
        results = bulk_download(reqs, {"blocking": source}, max_workers=2)
        dt = time.time() - t0

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        # If not concurrent, barrier would timeout and raise, failing the test.
        # Also ensure it didn't take unreasonably long.
        self.assertLess(dt, 1.0)

    def test_bulk_download_calls_on_result(self) -> None:
        barrier = threading.Barrier(2)
        source = BlockingSource(barrier)
        reqs = [
            SpectraRequest(source="blocking", identifier="A"),
            SpectraRequest(source="blocking", identifier="B"),
        ]

        calls: list[tuple[str, int, int]] = []

        def cb(res, done, total):  # type: ignore[no-untyped-def]
            calls.append((res.request.identifier, done, total))

        _ = bulk_download(reqs, {"blocking": source}, max_workers=2, on_result=cb)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][2], 2)
